"""Loss functions for GNN decoder training."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.config import TrainConfig


class FocalBCEWithLogitsLoss(nn.Module):
    """Sigmoid focal loss for imbalanced binary classification.

    Down-weights well-classified examples, focusing training on hard
    positives/negatives.  Standard choice for rare-positive detection
    tasks (logical flip rate is low at small ``p`` and large ``d``).

    Parameters
    ----------
    alpha : float
        Balancing factor for the positive class (default: 0.25).
    gamma : float
        Focusing exponent- higher values suppress easy examples more
        aggressively (default: 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.float()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()


def build_criterion(cfg: TrainConfig) -> nn.Module:
    """Build the loss function from training config.

    Parameters
    ----------
    cfg : TrainConfig
        Training hyperparameters (uses ``focal_alpha``, ``focal_gamma``).

    Returns
    -------
    nn.Module
    """
    return FocalBCEWithLogitsLoss(
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
    )
