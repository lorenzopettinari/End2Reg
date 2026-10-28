from typing import Tuple

import torch
from torch import Tensor


def _pairwise_sq_dists_block(q: Tensor, s: Tensor) -> Tensor:
    """Compute squared Euclidean distances between two 2D tensors.

    Args:
        q: (nq, c)
        s: (ns, c)

    Returns:
        d2: (nq, ns) with non-negative values (up to numerical error).
    """
    # ||q - s||^2 = ||q||^2 + ||s||^2 - 2 q s^T
    # Use float32 accumulation for stability even under autocast.
    qf = q.float()
    sf = s.float()
    q2 = (qf * qf).sum(dim=1, keepdim=True)  # (nq, 1)
    s2 = (sf * sf).sum(dim=1, keepdim=True).transpose(0, 1)  # (1, ns)
    d2 = q2 + s2 - 2.0 * (qf @ sf.transpose(0, 1))
    return d2.clamp_min_(0.0)


def _blockwise_knn_single(q_points: Tensor, s_points: Tensor, k: int, q_block_size: int, s_block_size: int) -> Tuple[Tensor, Tensor]:
    """Blockwise kNN for a single batch (no leading batch dims)."""
    device = q_points.device
    k = min(k, s_points.shape[0])
    if k <= 0:
        empty_d = torch.zeros((q_points.shape[0], 0), device=device, dtype=q_points.dtype)
        empty_i = torch.zeros((q_points.shape[0], 0), device=device, dtype=torch.long)
        return empty_d, empty_i

    n_q = q_points.shape[0]
    n_s = s_points.shape[0]

    # Fast path to bypass loops for tensors within block size
    if n_q <= q_block_size and n_s <= s_block_size:
        d2 = _pairwise_sq_dists_block(q_points, s_points)
        best_d2, sel = torch.topk(d2, k, dim=1, largest=False)
        best_d = torch.sqrt(best_d2).to(dtype=q_points.dtype)
        return best_d, sel

    all_best_d2 = []
    all_best_i = []

    # Process query points in blocks to cap temporary memory.
    for q0 in range(0, n_q, q_block_size):
        q1 = min(q0 + q_block_size, n_q)
        q_blk = q_points[q0:q1]  # (qb, c)

        best_d2 = torch.full((q1 - q0, k), float("inf"), device=device, dtype=torch.float32)
        best_i = torch.full((q1 - q0, k), -1, device=device, dtype=torch.long)

        # Sweep over support blocks and keep a running top-k.
        for s0 in range(0, n_s, s_block_size):
            s1 = min(s0 + s_block_size, n_s)
            s_blk = s_points[s0:s1]  # (sb, c)

            d2 = _pairwise_sq_dists_block(q_blk, s_blk)  # (qb, sb) in fp32
            # Take local top-k within this support block (or fewer if block smaller).
            kk = min(k, d2.shape[1])
            d2k, ik = torch.topk(d2, kk, dim=1, largest=False)
            ik = ik + s0

            # Merge current best with candidates and take top-k again.
            merged_d2 = torch.cat([best_d2, d2k], dim=1)
            merged_i = torch.cat([best_i, ik], dim=1)
            best_d2, sel = torch.topk(merged_d2, k, dim=1, largest=False)
            best_i = torch.gather(merged_i, 1, sel)

        all_best_d2.append(best_d2)
        all_best_i.append(best_i)

    best_d2 = torch.cat(all_best_d2, dim=0)  # (nq, k) fp32
    best_i = torch.cat(all_best_i, dim=0)  # (nq, k)
    best_d = torch.sqrt(best_d2).to(dtype=q_points.dtype)
    return best_d, best_i


def pytorch_knn(
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    q_block_size: int = 65536,
    s_block_size: int = 65536,
) -> Tuple[Tensor, Tensor]:
    """kNN with pure PyTorch (blockwise to avoid NxM VRAM blowups).

    This avoids materializing the full pairwise distance matrix, which can easily
    dominate VRAM for large point sets.

    Args:
        q_points (Tensor): query points, (*, N, C).
        s_points (Tensor): support points, (*, M, C).
        k (int): number of nearest neighbors.
        q_block_size (int): query block size for memory management.
        s_block_size (int): support block size for memory management.

    Returns:
        knn_distances (Tensor): distances to k nearest neighbors, (*, N, k).
        knn_indices (LongTensor): indices of k nearest neighbors, (*, N, k).
    """
    k = min(k, s_points.shape[-2])
    if k <= 0:
        return (
            torch.zeros((*q_points.shape[:-1], 0), device=q_points.device, dtype=q_points.dtype),
            torch.zeros((*q_points.shape[:-1], 0), device=q_points.device, dtype=torch.long),
        )

    # Handle leading batch dims by flattening them.
    if q_points.dim() == 2:
        return _blockwise_knn_single(q_points, s_points, k, q_block_size=q_block_size, s_block_size=s_block_size)

    batch_shape = q_points.shape[:-2]
    n_q, c = q_points.shape[-2], q_points.shape[-1]
    n_s = s_points.shape[-2]
    q_flat = q_points.reshape(-1, n_q, c)
    s_flat = s_points.reshape(-1, n_s, c)

    d_list = []
    i_list = []
    for b in range(q_flat.shape[0]):
        d_b, i_b = _blockwise_knn_single(q_flat[b], s_flat[b], k, q_block_size=q_block_size, s_block_size=s_block_size)
        d_list.append(d_b)
        i_list.append(i_b)

    knn_distances = torch.stack(d_list, dim=0).reshape(*batch_shape, n_q, k)
    knn_indices = torch.stack(i_list, dim=0).reshape(*batch_shape, n_q, k)
    return knn_distances, knn_indices


def knn(
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = False,
    remove_nearest: bool = False,
    transposed: bool = False,
    padding_mode: str = "nearest",
    inf: float = 1e10,
):
    """Compute the kNNs of the points in `q_points` from the points in `s_points`.

    Uses pure PyTorch to accelerate computation and avoid JIT/KeOps issues.

    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, neighbors are replaced according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): padding mode for neighbors further than distance limit ('nearest', 'empty').
        inf (float=1e10): infinity value for padding.

    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)
        s_points = s_points.transpose(-1, -2)

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    knn_distances, knn_indices = pytorch_knn(q_points, s_points, final_k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances[knn_masks] = knn_distances[..., 0]
            knn_indices[knn_masks] = knn_indices[..., 0]
        else:
            knn_distances[knn_masks] = inf
            knn_indices[knn_masks] = num_s_points

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices
