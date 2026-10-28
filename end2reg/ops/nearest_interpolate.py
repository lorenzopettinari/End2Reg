import torch

from end2reg.ops.index_select import index_select


def nearest_interpolate_pack_mode(inputs, neighbor_indices):
    """Pools features from the closest neighbors.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        inputs: [n1, d] features matrix
        neighbor_indices: [n2, max_num] Only the first column is used for pooling

    Returns:
        outputs: [n2, d] pooled features matrix
    """
    padded_inputs = torch.cat([inputs, torch.zeros_like(inputs[:1])], dim=0)
    outputs = index_select(padded_inputs, neighbor_indices[:, 0], dim=0)
    return outputs


def knn_interpolate(s_feats, q_points, s_points, neighbor_indices, k, eps=1e-8):
    r"""K-NN interpolate.

    WARNING: this function assumes the neighbors are ordered.

    Args:
        s_feats (Tensor): (M, C)
        q_points (Tensor): (N, 3)
        s_points (Tensor): (M, 3)
        neighbor_indices (LongTensor): (N, X)
        k (int)
        eps (float)

    Returns:
        q_feats (Tensor): (N, C)
    """
    s_points = torch.cat((s_points, torch.zeros_like(s_points[:1, :])), 0)  # (M + 1, 3)
    s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (M + 1, C)
    knn_indices = neighbor_indices[:, :k].contiguous()
    knn_points = index_select(s_points, knn_indices, dim=0)  # (N, k, 3)
    knn_feats = index_select(s_feats, knn_indices, dim=0)  # (N, k, C)
    knn_sq_distances = (q_points.unsqueeze(1) - knn_points).pow(2).sum(dim=-1)  # (N, k)
    knn_masks = torch.ne(knn_indices, s_points.shape[0] - 1).float()  # (N, k)
    knn_weights = knn_masks / (knn_sq_distances + eps)  # (N, k)
    knn_weights = knn_weights / (knn_weights.sum(dim=1, keepdim=True) + eps)  # (N, k)
    q_feats = (knn_feats * knn_weights.unsqueeze(-1)).sum(dim=1)  # (N, C)
    return q_feats


# Alias for backward compatibility with geotransformer code
nearest_upsample = nearest_interpolate_pack_mode
