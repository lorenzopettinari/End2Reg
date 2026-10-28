import torch

from end2reg.ops.knn import pytorch_knn


def radius_search_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    """Radius search in pack mode (pure PyTorch, per-sample loop).

    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): search radius.
        neighbor_limit (int): maximum number of neighbors.

    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """
    batch_size = q_lengths.shape[0]
    
    if batch_size == 1:
        # Optimized fast path for batch size 1 (bypasses loop and dynamic slicing graph breaks)
        knn_distances, knn_indices = pytorch_knn(q_points, s_points, neighbor_limit)
        knn_masks = torch.gt(knn_distances, radius)
        knn_indices = knn_indices.clone()
        knn_indices.masked_fill_(knn_masks, s_points.shape[0])
        return knn_indices

    q_start_index = 0
    s_start_index = 0
    knn_indices_list = []
    for i in range(batch_size):
        cur_q_length = q_lengths[i]
        cur_s_length = s_lengths[i]
        q_end_index = q_start_index + cur_q_length
        s_end_index = s_start_index + cur_s_length
        cur_q_points = q_points[q_start_index:q_end_index]
        cur_s_points = s_points[s_start_index:s_end_index]
        knn_distances, knn_indices = pytorch_knn(cur_q_points, cur_s_points, neighbor_limit)
        knn_indices = knn_indices + s_start_index
        knn_masks = torch.gt(knn_distances, radius)
        knn_indices.masked_fill_(knn_masks, s_points.shape[0])
        knn_indices_list.append(knn_indices)
        q_start_index = q_end_index
        s_start_index = s_end_index
    knn_indices = torch.cat(knn_indices_list, dim=0)
    return knn_indices


def radius_count_pack_mode(q_points, s_points, q_lengths, s_lengths, radius):
    """Count neighbors within radius in pack mode.

    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): search radius.

    Returns:
        radius_counts (LongTensor): the numbers of neighbors within radius for the query points.
    """
    device = q_points.device
    batch_size = q_lengths.shape[0]
    q_start_index = 0
    s_start_index = 0
    radius_counts_list = []

    q_block_size = 2048
    s_block_size = 2048
    radius2 = radius ** 2

    for i in range(batch_size):
        cur_q_length = q_lengths[i].item()
        cur_s_length = s_lengths[i].item()
        q_end_index = q_start_index + cur_q_length
        s_end_index = s_start_index + cur_s_length
        cur_q_points = q_points[q_start_index:q_end_index]
        cur_s_points = s_points[s_start_index:s_end_index]

        counts = torch.zeros(cur_q_length, dtype=torch.long, device=device)

        for q0 in range(0, cur_q_length, q_block_size):
            q1 = min(q0 + q_block_size, cur_q_length)
            q_blk = cur_q_points[q0:q1]

            sub_counts = torch.zeros(q1 - q0, dtype=torch.long, device=device)
            for s0 in range(0, cur_s_length, s_block_size):
                s1 = min(s0 + s_block_size, cur_s_length)
                s_blk = cur_s_points[s0:s1]

                # Pairwise squared distances block (from end2reg.ops.knn)
                from end2reg.ops.knn import _pairwise_sq_dists_block
                d2 = _pairwise_sq_dists_block(q_blk, s_blk)
                sub_counts += (d2 <= radius2).sum(dim=1)

            counts[q0:q1] = sub_counts

        radius_counts_list.append(counts)
        q_start_index = q_end_index
        s_start_index = s_end_index

    return torch.cat(radius_counts_list, dim=0)
