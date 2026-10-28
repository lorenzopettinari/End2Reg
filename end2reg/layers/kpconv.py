# Modified from [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch).
import math

import torch
import torch.nn as nn
from torch import Tensor

from end2reg.ops.index_select import index_select
from end2reg.layers.kpconv_utils import load_kernels


class KPConv(nn.Module):
    """Rigid KPConv.

    Paper: https://arxiv.org/abs/1904.08889.

    Unified implementation supporting both geotransformer and easy_kpconv calling conventions.

    Args:
         in_channels (int): The number of the input channels.
         out_channels (int): The number of the output channels.
         kernel_size (int): The number of kernel points.
         radius (float): The radius used for kernel point init.
         sigma (float): The influence radius of each kernel point.
         bias (bool, optional): If True, use bias. Default: False.
         dimension (int, optional): The dimension of the point space. Default: 3.
         inf (float, optional): The value of infinity to generate the padding point. Default: 1e6.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        radius: float,
        sigma: float,
        bias: bool = False,
        dimension: int = 3,
        inf: float = 1e6,
    ):
        """Initialize a rigid KPConv."""
        super().__init__()

        # Save parameters
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.inf = inf

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(self.kernel_size, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias", None)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()  # (N, 3)
        self.register_buffer("kernel_points", kernel_points)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def initialize_kernel_points(self) -> Tensor:
        """Initialize the kernel point positions in a sphere."""
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed="center")
        return torch.from_numpy(kernel_points).float()

    def forward(self, q_points: Tensor, s_points: Tensor, s_feats: Tensor, neighbor_indices: Tensor) -> Tensor:
        """KPConv forward.

        Args:
            q_points (Tensor): query points (M, 3)
            s_points (Tensor): support points (N, 3)
            s_feats (Tensor): support features (N, C_in)
            neighbor_indices (LongTensor): neighbor indices (M, H)

        Returns:
            q_feats (Tensor): query features (M, C_out)
        """
        padded_s_points = torch.cat([s_points, torch.zeros_like(s_points[:1, :]) + self.inf], 0)  # (N+1, 3)
        neighbors = index_select(padded_s_points, neighbor_indices, dim=0)  # (M, H, 3)
        neighbors = neighbors - q_points.unsqueeze(1)  # (M, H, 3)

        # Get Kernel point influences
        x2 = torch.sum(neighbors**2, dim=-1, keepdim=True)  # (M, H, 1)
        y2 = torch.sum(self.kernel_points**2, dim=-1).view(1, 1, self.kernel_size)  # (1, 1, K)
        xy = torch.matmul(neighbors, self.kernel_points.transpose(0, 1))  # (M, H, K)
        sq_distances = x2 + y2 - 2.0 * xy
        neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)
        neighbor_weights = torch.transpose(neighbor_weights, 1, 2)  # (M, K, H)

        # apply neighbor weights
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N+1, C)
        neighbor_feats = index_select(padded_s_feats, neighbor_indices, dim=0)  # (M, H, C)
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, C)

        # apply convolutional weights
        output_feats = torch.einsum("mkc,kcd->md", weighted_feats, self.weights)

        # density normalization
        neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        output_feats = output_feats / neighbor_num.unsqueeze(1)

        # add bias
        if self.bias is not None:
            output_feats = output_feats + self.bias

        return output_feats

    def extra_repr(self) -> str:
        param_strings = [
            f"kernel_size={self.kernel_size}",
            f"in_channels={self.in_channels}",
            f"out_channels={self.out_channels}",
            f"radius={self.radius:g}",
            f"sigma={self.sigma:g}",
            f"bias={self.bias is not None}",
        ]
        if self.dimension != 3:
            param_strings.append(f"dimension={self.dimension}")
        return ", ".join(param_strings)
