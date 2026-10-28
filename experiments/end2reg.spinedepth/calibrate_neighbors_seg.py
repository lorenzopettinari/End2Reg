import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from end2reg.ops.calibrate_neighbors import calibrate_neighbors_pack_mode
from end2reg.utils.collate import SimpleSingleCollateFnPackMode
from end2reg.datasets.spinedepth import SpineDepthSegmTrainingDataset

from config import make_cfg


def main():
    cfg = make_cfg()

    train_dataset = SpineDepthSegmTrainingDataset(
        cfg.dataset_dir,
        cfg,
        training=True,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        SimpleSingleCollateFnPackMode(),
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_voxel_size * cfg.backbone.base_radius,
    )

    print(f"Neighbor limits: {neighbor_limits}. Please fill neighbor_limits in config.py.")


if __name__ == "__main__":
    main()
