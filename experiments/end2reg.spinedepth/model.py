import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from end2reg.ops.pointcloud_partition import point_to_node_partition
from end2reg.ops.index_select import index_select
from end2reg.modules.registration.matching import get_node_correspondences
from end2reg.modules.sinkhorn import LearnableLogOptimalTransport
from end2reg.modules.geotransformer import (
    GeometricTransformer,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
    SuperPointMatching,
)
from end2reg.modules.point_selector import PointSelector

from end2reg.ops.graph_pyramid import build_graph_pyramid as build_grid_and_radius_graph_pyramid

from end2reg.backbones.kpconv_fpn import KPConvFPN
from end2reg.backbones.kpfcnn import KPFCNN

import open3d as o3d

def nanmean(x, dim=None):
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.tensor(0.0, device=x.device))
    count = mask.sum(dim=dim)
    sum_ = masked_x.sum(dim=dim)
    return sum_ / count

def compute_coarse_seg_scores(scores, graph_pyramid):
    len_c1 = graph_pyramid["lengths"][1][0]
    tot_len_c1 = graph_pyramid["lengths"][1][1] + len_c1
    len_c2 = graph_pyramid["lengths"][2][0]

    sub_c1 = graph_pyramid["subsampling"][0][:len_c1]
    sub_c1[sub_c1 == 22500] = 20000
    sub_c2 = graph_pyramid["subsampling"][1][:len_c2]
    sub_c2[sub_c2 == tot_len_c1] = len_c1

    nan_val = torch.tensor([float('nan')], device=scores.device)

    scores_ext = torch.cat([scores, nan_val], dim=0)
    sub_scores = scores_ext[sub_c1]
    scores_c1 = nanmean(sub_scores, dim=1)

    scores_ext = torch.cat([scores_c1, nan_val], dim=0)
    sub_scores = scores_ext[sub_c2]
    scores_c2 = nanmean(sub_scores, dim=1)

    return scores_c2




class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius
        self.seg_to_reg_scale_factor = getattr(
            cfg.model, "seg_to_reg_scale_factor", getattr(cfg, "reg_scale_factor", 4.6)
        )

        self.backbone_seg = KPFCNN(cfg)

        self.num_stages = cfg.backbone.num_stages
        self.voxel_size = cfg.backbone.init_voxel_size
        self.first_radius = cfg.backbone.init_radius
        self.neighbor_limits = cfg.backbone.neighbor_limits

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        # Module used to match attended features from geotransformer
        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        self.selector = PointSelector(tau=1.0, hard=True)


    def forward(self, data_dict):
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output_dict = {}

        # Segmentation
        output_seg_dict = self.backbone_seg(data_dict)
        scores = output_seg_dict["scores"]
        output_dict["scores_seg"] = scores

        ref_points = data_dict["points_ref"]
        ref_length = torch.tensor([ref_points.shape[0]], device=ref_points.device)
        ref_feats = torch.ones_like(ref_points[:, :1]).to(torch.float32)

        ref_feats = self.selector(scores, ref_feats)

        src_points = data_dict["points_src"]
        src_length = torch.tensor([src_points.shape[0]], device=src_points.device)
        src_feats = torch.ones_like(src_points[:, :1]).to(torch.float32)

        # Normalization and recentering
        scale_factor = self.seg_to_reg_scale_factor
        ref_points = ref_points * scale_factor
        src_points = src_points * scale_factor

        with torch.no_grad():
            non_zero_mask = ~(ref_points == 0).all(dim=1)
            filtered_points = ref_points[non_zero_mask]
            centroid_ref = filtered_points.mean(dim=0)
            translation_ref = torch.zeros_like(ref_points)  # (N, 3)
            translation_ref[non_zero_mask] = centroid_ref

            centroid_src = src_points.mean(dim=0)

            T_src = torch.eye(4, device=ref_points.device)
            T_src[:3, 3] = centroid_src

            T_ref = torch.eye(4, device=ref_points.device)
            T_ref[:3, 3] = -centroid_ref

        ref_points = ref_points - translation_ref
        src_points = src_points - centroid_src

        transform = data_dict['transform'].detach().clone()
        transform = T_ref @ transform @ T_src
        data_dict['transform'] = transform.clone()

        feats = torch.cat([ref_feats, src_feats], dim=0)
        points = torch.cat([ref_points, src_points], dim=0)
        lengths = torch.tensor([ref_length, src_length], device=ref_points.device)

        graph_pyramid = build_grid_and_radius_graph_pyramid(
            points, lengths, self.num_stages, self.voxel_size, self.first_radius, self.neighbor_limits
        )

        ref_length_c = graph_pyramid['lengths'][-1][0].item()
        ref_length_f = graph_pyramid['lengths'][0][0].item()
        ref_length = graph_pyramid['lengths'][0][0].item()
        points_c = graph_pyramid['points'][-1].detach()
        points_f = graph_pyramid['points'][0].detach()
        points = graph_pyramid['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences: computes 128 fine point correspondences for each superpoint
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, graph_pyramid)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )

        # Feature normalization L2
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            # Function that matches superpoints and the matching scores
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)

        # For each correspondence between two patches a correlation matrices is computed for the points in each patch
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        if ref_node_corr_knn_feats is None:
            return None

        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 8. Optimal transport
        if ref_node_corr_indices.shape[0] == 0 and src_node_corr_indices.shape[0] == 0:
            raise ValueError("No corresponding features found. Skipping iteration.")

        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)

        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

            output_dict['metadata'] = data_dict['metadata']
            output_dict['metadata']['new_GT'] = transform
            output_dict['metadata']['T_ref'] = T_ref
            output_dict['metadata']['T_src'] = T_src
            output_dict['metadata']['scale_factor'] = scale_factor

        output_dict['start_event'] = start_event
        
        import torch.compiler as torch_compiler
        if torch_compiler.is_compiling():
            output_dict['corr_event'] = None
            output_dict['corr_extracted_event'] = None
        else:
            output_dict['corr_event'] = getattr(self.fine_matching, 'corr_event', None)
            output_dict['corr_extracted_event'] = getattr(self.fine_matching, 'corr_extracted_event', None)
        return output_dict

    def compile_submodules(self):
        print("[INFO] Compiling model forward pass...")
        self.forward = torch.compile(self.forward, dynamic=True)


def create_model(cfg):
    model = GeoTransformer(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)


if __name__ == '__main__':
    main()
