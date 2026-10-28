import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..', '..')))
sys.path.insert(0, osp.abspath(osp.dirname(__file__)))

from end2reg.engine.iter_based_trainer import IterBasedTrainer
from end2reg.utils.torch import build_warmup_cosine_lr_scheduler
import torch.multiprocessing as mp
from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator


class Trainer(IterBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_iteration=cfg.optim.max_iteration, snapshot_steps=cfg.optim.snapshot_steps)

        self.cfg = cfg

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        cfg.device = next(model.parameters()).device
        model = self.register_model(model)

        # dataloader
        start_time = time.time()
        train_loader, val_loader = train_valid_data_loader(cfg)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)



        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

        if cfg.loss.seg_loss == "tloss":
            optimizer = optim.Adam(
                [
                    {
                        "params": model.parameters(),
                        "lr": cfg.optim.lr,
                        "weight_decay": cfg.optim.weight_decay,
                        "name": "model",
                    },
                    {
                        "params": self.loss_func.seg_loss.parameters(),
                        "lr": cfg.student.lr,
                        "name": "tloss",
                    },
                ]
            )
        if cfg.loss.seg_loss == "celoss":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = cfg.optim.lr, weight_decay = cfg.optim.weight_decay)


        self.register_optimizer(optimizer)
        scheduler = build_warmup_cosine_lr_scheduler(
            optimizer,
            total_steps=cfg.optim.max_iteration,
            warmup_steps=cfg.optim.warmup_steps,
            eta_init=cfg.optim.eta_init,
            eta_min=cfg.optim.eta_min,
            grad_acc_steps=cfg.optim.grad_acc_steps,
        )
        self.register_scheduler(scheduler)

    def train_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        if iteration <= self.cfg.optim.warmup_steps_seg:
            self.seg_loss_weight = 1.0
        if iteration > self.cfg.optim.warmup_steps_seg:
            self.seg_loss_weight = 0.5
        loss_dict = self.loss_func(output_dict, data_dict, self.seg_loss_weight)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict, self.seg_loss_weight)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    # import multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
