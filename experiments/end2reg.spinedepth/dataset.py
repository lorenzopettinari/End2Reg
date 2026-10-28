from typing import Callable, List, Dict
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from end2reg.datasets.spinedepth import SpineDepthTrainingDataset, SpineDepthTestingDataset
from end2reg.utils.v3d_dataloader import build_dataloader


class SpineDepthTestingCollateFn(Callable):
    def __init__(self):
        pass

    def __call__(self, data_dicts: List[Dict]) -> Dict:
        assert len(data_dicts) == 1, "Batch size must be 1 during testing."
        data_dict = data_dicts[0]

        return data_dict

def train_valid_data_loader(cfg):

    train_dataset = SpineDepthTrainingDataset(
        cfg.dataset_dir,
        cfg,
        training=True,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=SpineDepthTestingCollateFn(),
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = SpineDepthTestingDataset(
        cfg.dataset_dir,
        cfg
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=SpineDepthTestingCollateFn(),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def test_data_loader(cfg):
    test_dataset = SpineDepthTestingDataset(
        cfg.dataset_dir,
        cfg
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=SpineDepthTestingCollateFn(),
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
