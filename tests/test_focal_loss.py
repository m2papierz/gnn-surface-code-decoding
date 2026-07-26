"""Tests for focal loss and weighted sampler for the direct training case."""

import pytest
import torch
import torch.nn as nn

from training import FocalBCEWithLogitsLoss


class TestFocalBCEWithLogitsLoss:
    """Tests for the sigmoid focal loss."""

    @pytest.fixture
    def focal(self) -> FocalBCEWithLogitsLoss:
        return FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0)

    def test_gradient_flows(self, focal: FocalBCEWithLogitsLoss) -> None:
        logits = torch.randn(16, requires_grad=True)
        target = torch.randint(0, 2, (16,)).float()
        focal(logits, target).backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_perfect_positive_near_zero(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """Confident correct positive prediction → near-zero loss."""
        loss = focal(torch.full((10,), 10.0), torch.ones(10))
        assert loss.item() < 1e-4

    def test_perfect_negative_near_zero(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """Confident correct negative prediction → near-zero loss."""
        loss = focal(torch.full((10,), -10.0), torch.zeros(10))
        assert loss.item() < 1e-4

    def test_suppresses_easy_negatives(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """Focal loss on easy examples is strictly less than BCE."""
        bce = nn.BCEWithLogitsLoss()
        logits = torch.full((100,), -3.0)
        target = torch.zeros(100)
        assert focal(logits, target).item() < bce(logits, target).item()

    def test_hard_example_ratio(
        self,
        focal: FocalBCEWithLogitsLoss,
    ) -> None:
        """At p=0.5: focal = alpha * (1-0.5)^gamma * BCE = 0.0625 * BCE."""
        bce = nn.BCEWithLogitsLoss()
        logits = torch.zeros(100)
        target = torch.ones(100)
        ratio = focal(logits, target).item() / bce(logits, target).item()
        assert ratio == pytest.approx(0.0625, abs=0.01)

    def test_gamma_zero_is_weighted_bce(self) -> None:
        """With gamma=0, focal reduces to alpha-weighted BCE."""
        focal_g0 = FocalBCEWithLogitsLoss(alpha=0.5, gamma=0.0)
        bce = nn.BCEWithLogitsLoss()
        logits = torch.randn(50)
        target = torch.randint(0, 2, (50,)).float()
        assert focal_g0(logits, target).item() == pytest.approx(
            0.5 * bce(logits, target).item(), abs=0.01
        )
