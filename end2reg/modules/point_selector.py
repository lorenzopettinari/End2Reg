import torch
import torch.nn as nn
import torch.nn.functional as F


class PointSelector(nn.Module):
    def __init__(self, tau: float = 1.0, hard: bool = True):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, scores: torch.Tensor, points: torch.Tensor):
        """
        Args:
            scores: (N, 2) logits per point [class_0_logit, class_1_logit]
            points: (N, F) points to select

        Returns:
            selected_points: (N, F) points weighted by the differentiable mask
            mask: (N,) binary mask (STE) or soft mask (relaxed)
        """

        # Sample with Gumbel-Softmax
        y = F.gumbel_softmax(scores, tau=self.tau, hard=self.hard)  # (N, 2)

        # Probability (or STE hard selection) for class 1 (bone)
        mask = y[:, 1]  # (N,)

        # Apply mask differentiably
        selected_points = mask[:, None] * points  # (N, F)

        return selected_points
