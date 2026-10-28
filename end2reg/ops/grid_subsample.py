import torch


def ravel_hash_func(voxels, dimensions):
    dimension = voxels.shape[1]
    hash_values = voxels[:, 0].clone()
    for i in range(1, dimension):
        hash_values *= dimensions[i]
        hash_values += voxels[:, i]
    return hash_values


def grid_subsample_pack_mode(points, lengths, voxel_size):
    """Grid subsample in pack mode (pure PyTorch, device-agnostic).

    Args:
        points (Tensor): the original points (N, 3).
        lengths (LongTensor): the numbers of points in the batch (B,).
        voxel_size (float): the voxel size.

    Returns:
        sampled_points (Tensor): the subsampled points (M, 3).
        sampled_lengths (Tensor): the numbers of subsampled points in the batch (B,).
    """
    device = points.device
    batch_size = lengths.shape[0]

    # voxelize
    inv_voxel_size = 1.0 / voxel_size
    voxels = torch.floor(points * inv_voxel_size).long()

    # normalize, pad with batch indices (loop-free vectorization)
    batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), lengths
    )
    
    # Compute amin per batch element using scatter_reduce
    min_voxels = torch.empty((batch_size, 3), dtype=voxels.dtype, device=device)
    min_voxels = min_voxels.scatter_reduce(
        0, 
        batch_indices.unsqueeze(1).expand(-1, 3), 
        voxels, 
        reduce='amin', 
        include_self=False
    )
    voxels_normalized = voxels - min_voxels[batch_indices]
    voxels = torch.cat([batch_indices.unsqueeze(1), voxels_normalized], dim=1)

    # scatter
    dimensions = voxels.amax(0) + 1  # (4)
    hash_values = ravel_hash_func(voxels, dimensions)  # (N)
    unique_values, inv_indices, unique_counts = torch.unique(
        hash_values, return_inverse=True, return_counts=True
    )  # (M) (N) (M)
    inv_indices = inv_indices.unsqueeze(1).expand(-1, 3)  # (N, 3)
    s_points = torch.zeros(size=(unique_counts.shape[0], 3), device=device, dtype=points.dtype)  # (M, 3)
    s_points.scatter_add_(0, inv_indices, points)  # (M, 3)
    s_points = s_points / unique_counts.unsqueeze(1).float()  # (M, 3)

    # compute lengths
    total_dimension = torch.cumprod(dimensions, dim=0)[-1] / dimensions[0]
    s_batch_indices = torch.clamp(
        torch.div(unique_values, total_dimension, rounding_mode="floor").long(),
        max=batch_size - 1
    )
    s_lengths = torch.bincount(s_batch_indices, minlength=batch_size)
    assert (
        s_lengths.shape[0] == batch_size
    ), f"Invalid length of s_lengths ({batch_size} expected, {s_lengths.shape} got)."

    return s_points, s_lengths
