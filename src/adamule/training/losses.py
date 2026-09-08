"""Loss functions for fraud detection and legitimacy preservation."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss for severe fraud class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[float] = None
) -> torch.Tensor:
    """Compute binary cross-entropy with auto-balanced positive class weight."""
    if pos_weight is None:
        num_pos = (targets == 1).sum().float()
        num_neg = (targets == 0).sum().float()
        w = (num_neg / (num_pos + 1e-5)).clamp(min=1.0, max=20.0)
        pos_weight_tensor = w.unsqueeze(0)
    else:
        pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)

    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight_tensor)
