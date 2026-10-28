import os
import os.path as osp
import argparse
import sys

from easydict import EasyDict as edict

from end2reg.utils.common import ensure_dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import end2reg.datasets.spinedepth as spinedepth


_C = edict()

# common
# _C.seed = 7351
_C.seed = 42

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.dataset_type = "Deformable_frame0_spec2"
_C.dataset_dir = osp.join(_C.root_dir, "data", _C.dataset_type)
_C.output_dir = osp.join(_C.root_dir, "output", _C.exp_name, "Scores_feat_noinit_split2")
_C.snapshot_dir = osp.join(_C.output_dir, "snapshots")
_C.log_dir = osp.join(_C.output_dir, "logs")
_C.event_dir = osp.join(_C.output_dir, "events")

# data
_C.data = edict()
_C.data.dataset_root = osp.join(_C.root_dir, "data", "SpineDepth")
_C.data.num_classes = spinedepth.NUM_CLASSES
_C.data.class_names = spinedepth.CLASS_NAMES
_C.data.num_points = 20000
_C.data.num_points_src = 2500
_C.data.num_points_ref = 2500

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 0
_C.train.num_points = None

# Training augmentation
_C.train.augment = True

_C.train.random_pose_augmentation = True
_C.train.max_angle = 0
_C.train.max_translation = 0.05

_C.train.jitter = False
_C.train.noise_sigma = 0.001
_C.train.noise_clip = 0.005

_C.train.random_scaling = True
_C.train.min_scale = 0.9
_C.train.max_scale = 1.1

_C.train.random_flip = False
_C.train.brightness_augm = True
_C.train.contrast_augm = True
_C.train.saturation_augm = True

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 0

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0
_C.eval.acceptance_radius = 0.1
_C.eval.inlier_ratio_threshold = 0.05
_C.eval.rre_threshold = 1.0
_C.eval.rte_threshold = 0.1

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05
_C.ransac.num_points = 3
_C.ransac.num_iterations = 1000

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.weight_decay = 1e-6
_C.optim.warmup_steps_seg = 0
_C.optim.warmup_steps_reg = 10000
_C.optim.warmup_steps = _C.optim.warmup_steps_seg + _C.optim.warmup_steps_reg
_C.optim.eta_init = 0.1
_C.optim.eta_min = 0.1
_C.optim.max_iteration = 4000000
_C.optim.snapshot_steps = 10000
_C.optim.grad_acc_steps = 1

# model - segmentation
_C.model_seg = edict()
_C.model_seg.num_stages = 5
_C.model_seg.basic_voxel_size = 0.04
_C.model_seg.kernel_size = 15
_C.model_seg.kpconv_radius = 2.5
_C.model_seg.kpconv_sigma = 2.0
_C.model_seg.input_dim = 4
_C.model_seg.init_dim = 64
_C.model_seg.neighbor_limits =[104, 31, 20, 16,  8]


# model - backbone registration
_C.backbone = edict()
_C.backbone.num_stages = 3
_C.backbone.init_voxel_size = 0.025
_C.backbone.kernel_size = 20
_C.backbone.base_radius = 7
_C.backbone.base_sigma = 3.5
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64
_C.backbone.output_dim = 256
_C.backbone.neighbor_limits = [208, 350, 384]


# model - Global
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05
_C.model.num_points_in_patch = 128
_C.model.num_sinkhorn_iterations = 100
# Scaling factor between segmentation and registration space
_C.model.seg_to_reg_scale_factor = 4.6
_C.reg_scale_factor = _C.model.seg_to_reg_scale_factor

# model - Coarse Matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 128
_C.coarse_matching.overlap_threshold = 0.1
_C.coarse_matching.num_correspondences = 128
_C.coarse_matching.dual_normalization = True

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 512
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ["self", "cross", "self", "cross", "self", "cross"]
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = "max"

# model - Fine Matching
_C.fine_matching = edict()
_C.fine_matching.topk = 3
_C.fine_matching.acceptance_radius = 0.1
_C.fine_matching.mutual = True
_C.fine_matching.confidence_threshold = 0.05
_C.fine_matching.use_dustbin = False
_C.fine_matching.use_global_score = False
_C.fine_matching.correspondence_threshold = 3
_C.fine_matching.correspondence_limit = None
_C.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.1
_C.coarse_loss.negative_margin = 1.4
_C.coarse_loss.positive_optimal = 0.1
_C.coarse_loss.negative_optimal = 1.4
_C.coarse_loss.log_scale = 24
_C.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.05

# loss registration - Overall
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0
_C.loss.weight_fine_loss = 1.0

# loss - Segmentation loss
_C.loss.seg_loss = "celoss"


# Student
_C.student = edict()
_C.student.nu = 1.0
_C.student.epsilon = 1e-8
_C.student.lr = 1e-3


def make_cfg():
    # Detect dataset_type from command line arguments if present
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="Deformable_frame0_spec2")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--max_iteration", type=int, default=None)
    args, _ = parser.parse_known_args()
    
    if args.max_iteration is not None:
        _C.optim.max_iteration = args.max_iteration
        
    _C.dataset_type = args.dataset_type
    if args.dataset_dir:
        _C.dataset_dir = args.dataset_dir
    elif os.environ.get("DATASET_DIR"):
        _C.dataset_dir = os.environ.get("DATASET_DIR")
    else:
        _C.dataset_dir = osp.join(_C.root_dir, "data", _C.dataset_type)
            
    # Update other directories depending on dataset_type
    _C.output_dir = osp.join(_C.root_dir, "output", _C.exp_name, f"Scores_feat_noinit_split2_{_C.dataset_type}")
    _C.snapshot_dir = osp.join(_C.output_dir, "snapshots")
    _C.log_dir = osp.join(_C.output_dir, "logs")
    _C.event_dir = osp.join(_C.output_dir, "events")

    ensure_dir(_C.output_dir)
    ensure_dir(_C.snapshot_dir)
    ensure_dir(_C.log_dir)
    ensure_dir(_C.event_dir)
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link_output", dest="link_output", action="store_true", help="link output dir")
    parser.add_argument("--dataset_type", type=str, default="Deformable_frame0_spec2", help="dataset type")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = make_cfg()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
