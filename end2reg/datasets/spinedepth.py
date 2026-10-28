import os.path as osp
from typing import Dict
import pandas as pd
import open3d as o3d
from end2reg.utils.open3d import estimate_normals
from end2reg.utils.pointcloud import random_sample_transform, apply_transform, inverse_transform, regularize_normals
import numpy as np
import torch.utils.data
import sys
import os
import colorsys

from end2reg.utils.point_cloud_utils import normalize_points_on_xy_plane
from end2reg.utils.dataset_utils import divide_point_cloud_into_blocks, random_sample_block

NUM_CLASSES = 2
CLASS_NAMES = (
    "background",
    "bone"
)

def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)

    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist

    return np.expand_dims(t, 1)


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()

    return T

def jitter_pcd(points, sigma=0.01, clip=0.05):

    jitter = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    
    return points + jitter

def saturation_augmentation(rgb_colors: np.ndarray, saturation_scale: float) -> np.ndarray:

    adjusted_colors = []

    for rgb in rgb_colors:
        hsv = colorsys.rgb_to_hsv(*rgb)
        s = np.clip(hsv[1] * saturation_scale, 0, 1)
        r, g, b = colorsys.hsv_to_rgb(hsv[0], s, hsv[2])
        adjusted_colors.append([r, g, b])

    return np.array(adjusted_colors)


def create_data_list(load_path, partition):

    XLSX_DIR = load_path + f"/Dataset_info_{partition}.xlsx"

    if partition == "train":
        partition = "Train/"
    if partition == "test":
        partition = "Test/"

    df = pd.read_excel(XLSX_DIR)

    all_src = []
    all_ref = []
    all_transformations = []
    all_metadata = []

    for index, row in df.iterrows():
        spec_num = row['spec_num']
        rec_num = row['rec_num']
        vid_num = row['vid_num']
        frame_num = row['frame_num']
        GT_transformation_matrix = row['GT_transformation_matrix']

        GT_transformation_matrix = GT_transformation_matrix.replace('[', '').replace(']', '').replace('\n', ' ')
        matrix_list = list(map(float, GT_transformation_matrix.split()))
        GT_transformation_matrix = np.array(matrix_list).reshape(4, 4)

        vertebra_path = os.path.join(load_path, partition, f'Spec_{spec_num}', f'recording{rec_num}', f'Vertebrae_{vid_num}', f'vertebrae_frame_{frame_num}.ply')

        pointcloud_path = os.path.join(load_path, partition, f'Spec_{spec_num}', f'recording{rec_num}', f'Pointclouds_{vid_num}', f'point_cloud_frame_{frame_num}.ply')

        point_cloud_ref = o3d.t.io.read_point_cloud(pointcloud_path)

        point_cloud_src = o3d.io.read_point_cloud(vertebra_path)


        metadata = {
            "spec_num": spec_num,
            "rec_num": rec_num,
            "vid_num": vid_num,
            "frame_num": frame_num,
            "GT_transformation_matrix": GT_transformation_matrix,
        }

        raw_labels = np.asarray(point_cloud_ref.point["labels"].numpy().flatten())
        labels_ref = np.where(raw_labels == 0, 0, 1)

        src = {
            'pos_src': np.asarray(point_cloud_src.points),
        }

        ref = {
            'pos_ref': point_cloud_ref.point["positions"].numpy(),
            'color_ref': point_cloud_ref.point["colors"].numpy(),
            'labels_ref': labels_ref
        }

        all_src.append(src)
        all_ref.append(ref)
        all_transformations.append(GT_transformation_matrix)
        all_metadata.append(metadata)


    return all_src, all_ref, all_transformations, all_metadata


def load_sample(point_cloud_src: str, point_cloud_ref: str, transform, augment, cfg) -> Dict:

    transform = transform.copy()

    pos_ref = point_cloud_ref["pos_ref"].copy()
    color_ref = point_cloud_ref["color_ref"].copy()
    labels_ref = point_cloud_ref["labels_ref"].copy()

    pos_src = point_cloud_src['pos_src'].copy()


    if augment:
        random_augmentation_probability = 0.5

        if cfg.train.jitter:
            # Position jittering
            pos_ref = jitter_pcd(pos_ref, sigma=cfg.train.noise_sigma, clip=cfg.train.noise_clip)

        if cfg.train.random_scaling:
            if np.random.rand() < random_augmentation_probability:
                # Random scaling
                scale = np.random.uniform(cfg.train.min_scale, cfg.train.max_scale)
                pos_ref *= scale  # Apply scaling to all coordinates
                pos_src *= scale

        if cfg.train.random_pose_augmentation:
            if np.random.rand() < random_augmentation_probability:
                random_transform = random_pose(max_angle = cfg.train.max_angle, max_trans = (cfg.train.max_translation / 2))
                pos_ref = pos_ref @ random_transform[:3, :3].T + random_transform[:3, 3]

                transform = random_transform @ transform

        if cfg.train.brightness_augm:
            if np.random.rand() < random_augmentation_probability:
                brightness_factor = np.random.uniform(0.6, 1.2)
                color_ref = np.clip(color_ref * brightness_factor, 0, 1)

        if cfg.train.contrast_augm:
            if np.random.rand() < random_augmentation_probability:
                mean = color_ref.mean(axis=0, keepdims=True)
                contrast_factor = np.random.uniform(0.6, 1.2)
                color_ref = np.clip((color_ref - mean) * contrast_factor + mean, 0, 1)

        if cfg.train.saturation_augm:
            if np.random.rand() < random_augmentation_probability:
                saturation_factor = np.random.uniform(0.6, 1.2)
                color_ref = saturation_augmentation(color_ref, saturation_factor)
                color_ref = color_ref.astype(np.float32)

        if cfg.train.random_flip:
            rot_180_options = {
                0: np.array([[1,  0,  0],
                                [0, -1,  0],
                                [0,  0, -1]]),

                1: np.array([[-1, 0,  0],
                                [ 0, 1,  0],
                                [ 0, 0, -1]]),

                2: np.array([[-1, 0, 0],
                                [ 0, -1, 0],
                                [ 0,  0, 1]])
            }

            applied_rotation = np.eye(3)
            for axis in [0, 1, 2]:
                if np.random.rand() < random_augmentation_probability:
                    applied_rotation = rot_180_options[axis] @ applied_rotation
                    pos_ref = pos_ref @ rot_180_options[axis].T

            if not np.allclose(applied_rotation, np.eye(3)):
                R_hom = np.eye(4)
                R_hom[:3, :3] = applied_rotation
                transform = R_hom @ transform


        # Augmentation for src pointcloud (just pose)

        random_transform = random_pose(45 / 180 * np.pi, 0.1)

        pos_src = pos_src @ random_transform[:3, :3].T + random_transform[:3, 3]

        transform = transform @ np.linalg.inv(random_transform)


    # Rescale GT translation between segmentation and registration coordinate spaces
    scale_factor = getattr(
        cfg.model if hasattr(cfg, "model") else cfg,
        "seg_to_reg_scale_factor",
        getattr(cfg, "reg_scale_factor", 4.6),
    )
    transform[:3, 3] *= scale_factor


    data_dict = {
        "points_ref": torch.tensor(pos_ref, dtype=torch.float32),
        "feats_ref": torch.tensor(color_ref, dtype=torch.float32),
        "labels_ref": torch.tensor(labels_ref, dtype=torch.float32),
        "points_src": torch.tensor(pos_src, dtype=torch.float32),
        "transform": torch.tensor(transform, dtype=torch.float32),
    }


    return data_dict


class SpineDepthTrainingDataset(torch.utils.data.Dataset):
    """
    'SpineDepthTrainingDataset'
    """

    def __init__(
        self,
        dataset_dir,
        cfg,
        training=True
    ):
        super().__init__()

        self.src_paths, self.ref_paths, self.transforms, self.metadata = create_data_list(dataset_dir, "train")

        self.config = cfg

        self.training = training
        self.num_samples = len(self.src_paths)



    def __getitem__(self, index):

        data_dict = load_sample(self.src_paths[index], self.ref_paths[index], self.transforms[index], True, self.config)
        data_dict["metadata"] = self.metadata[index]

        return data_dict

    def __len__(self):
        return self.num_samples


class SpineDepthTestingDataset(torch.utils.data.Dataset):
    """
    'SpineDepthTestingDataset'
    """

    def __init__(
            self,
            dataset_dir,
            cfg,
            training=False
    ):
        super().__init__()

        self.src_paths, self.ref_paths, self.transforms, self.metadata = create_data_list(dataset_dir, "test")

        self.config = cfg
        self.training = training
        self.num_samples = len(self.src_paths)


    def __getitem__(self, index):

        data_dict = load_sample(self.src_paths[index], self.ref_paths[index], self.transforms[index], False, self.config)
        data_dict["metadata"] = self.metadata[index]

        return data_dict

    def __len__(self):
        return self.num_samples


class SpineDepthSegmTrainingDataset(torch.utils.data.Dataset):
    """
    'SpineDepthSegmTrainingDataset'
    """

    def __init__(
            self,
            dataset_dir,
            cfg,
            training=True
    ):
        super().__init__()

        self.src_paths, self.ref_paths, self.transforms, self.metadata = create_data_list(dataset_dir, "train")

        self.config = cfg

        self.training = training
        self.num_samples = len(self.src_paths)



    def __getitem__(self, index):

        data_dict = load_sample(self.src_paths[index], self.ref_paths[index], self.transforms[index], False, self.config)
        data_dict["metadata"] = self.metadata[index]

        points_ref = data_dict["points_ref"]
        labels_ref = data_dict["labels_ref"]
        filtered_points = points_ref[labels_ref == 1]

        data_dict["points"]= filtered_points
        data_dict["feats"]= torch.ones_like(filtered_points[:, :1]).to(torch.float32)


        return data_dict

    def __len__(self):
        return self.num_samples