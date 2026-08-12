"""Tests for reliability diagram and Expected Calibration Error."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.calibration import (
    expected_calibration_error,
    reliability_diagram,
)


def _logit(p: float) -> float:
    """Inverse sigmoid: logit(p) = log(p / (1-p))."""
    return float(np.log(p / (1.0 - p)))


class TestReliabilityDiagram:
    def test_output_shapes(self) -> None:
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(200)
        labels = rng.integers(0, 2, size=200)
        diag = reliability_diagram(logits, labels, n_bins=10)
        assert diag.bin_confidences.shape == (10,)
        assert diag.bin_accuracies.shape == (10,)
        assert diag.bin_counts.shape == (10,)
        assert diag.bin_edges.shape == (11,)

    def test_bin_counts_sum_to_n(self) -> None:
        rng = np.random.default_rng(99)
        n = 500
        logits = rng.standard_normal(n)
        labels = rng.integers(0, 2, size=n)
        diag = reliability_diagram(logits, labels, n_bins=15)
        assert diag.bin_counts.sum() == n

    def test_perfectly_calibrated_logits(self) -> None:
        """Logits whose sigmoid equals the label mean per bin => diagonal."""
        rng = np.random.default_rng(0)
        n = 50_000
        true_probs = rng.uniform(0.05, 0.95, size=n)
        labels = (rng.uniform(size=n) < true_probs).astype(int)
        logits = np.log(true_probs / (1.0 - true_probs))

        diag = reliability_diagram(logits, labels, n_bins=10)
        nonempty = diag.bin_counts > 0
        gaps = np.abs(diag.bin_accuracies[nonempty] - diag.bin_confidences[nonempty])
        assert np.all(gaps < 0.05), f"Max gap {gaps.max():.4f} >= 0.05"

    def test_systematically_overconfident(self) -> None:
        """Predicted probs systematically higher than actual accuracy."""
        rng = np.random.default_rng(1)
        n = 10_000
        true_p = 0.2
        labels = rng.binomial(1, true_p, size=n)
        predicted_p = 0.8
        logits = np.full(n, _logit(predicted_p))

        diag = reliability_diagram(logits, labels, n_bins=10)
        nonempty = diag.bin_counts > 0
        assert nonempty.sum() == 1
        conf = diag.bin_confidences[nonempty][0]
        acc = diag.bin_accuracies[nonempty][0]
        assert conf > acc + 0.3

    def test_empty_bins_are_nan(self) -> None:
        """Bins with no samples get NaN for confidence and accuracy."""
        logits = np.full(100, _logit(0.1))
        labels = np.zeros(100, dtype=int)
        diag = reliability_diagram(logits, labels, n_bins=10)
        empty = diag.bin_counts == 0
        assert empty.sum() > 0
        assert np.all(np.isnan(diag.bin_confidences[empty]))
        assert np.all(np.isnan(diag.bin_accuracies[empty]))

    def test_single_bin(self) -> None:
        logits = np.array([0.0, 1.0, -1.0, 2.0])
        labels = np.array([0, 1, 0, 1])
        diag = reliability_diagram(logits, labels, n_bins=1)
        assert diag.bin_counts[0] == 4
        assert diag.bin_accuracies[0] == pytest.approx(0.5)

    def test_all_positive_labels(self) -> None:
        logits = np.linspace(-2, 2, 100)
        labels = np.ones(100, dtype=int)
        diag = reliability_diagram(logits, labels, n_bins=5)
        nonempty = diag.bin_counts > 0
        assert np.all(diag.bin_accuracies[nonempty] == 1.0)

    def test_all_negative_labels(self) -> None:
        logits = np.linspace(-2, 2, 100)
        labels = np.zeros(100, dtype=int)
        diag = reliability_diagram(logits, labels, n_bins=5)
        nonempty = diag.bin_counts > 0
        assert np.all(diag.bin_accuracies[nonempty] == 0.0)

    def test_bin_edges_span_unit_interval(self) -> None:
        logits = np.array([0.0])
        labels = np.array([1])
        diag = reliability_diagram(logits, labels, n_bins=7)
        assert diag.bin_edges[0] == pytest.approx(0.0)
        assert diag.bin_edges[-1] == pytest.approx(1.0)

    def test_extreme_logits(self) -> None:
        """Very large / very small logits don't cause overflow."""
        logits = np.array([100.0, -100.0, 50.0, -50.0])
        labels = np.array([1, 0, 1, 0])
        diag = reliability_diagram(logits, labels, n_bins=10)
        assert np.all(np.isfinite(diag.bin_confidences[diag.bin_counts > 0]))


class TestReliabilityDiagramValidation:
    def test_empty_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            reliability_diagram(np.array([]), np.array([]))

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            reliability_diagram(np.array([0.0, 1.0]), np.array([1]))

    def test_non_binary_labels(self) -> None:
        with pytest.raises(ValueError, match="binary"):
            reliability_diagram(np.array([0.0, 1.0]), np.array([0, 2]))

    def test_n_bins_zero(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            reliability_diagram(np.array([0.0]), np.array([1]), n_bins=0)


# expected_calibration_error


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_near_zero(self) -> None:
        """Perfectly calibrated logits => ECE ≈ 0."""
        rng = np.random.default_rng(42)
        n = 50_000
        true_probs = rng.uniform(0.05, 0.95, size=n)
        labels = (rng.uniform(size=n) < true_probs).astype(int)
        logits = np.log(true_probs / (1.0 - true_probs))

        ece = expected_calibration_error(logits, labels, n_bins=15)
        assert ece < 0.02, f"ECE {ece:.4f} >= 0.02 for calibrated logits"

    def test_overconfident_positive_ece(self) -> None:
        """Systematically biased logits => ECE > 0."""
        rng = np.random.default_rng(1)
        n = 10_000
        labels = rng.binomial(1, 0.2, size=n)
        logits = np.full(n, _logit(0.8))

        ece = expected_calibration_error(logits, labels, n_bins=10)
        assert ece > 0.5, f"ECE {ece:.4f} should be > 0.5 for 0.8 vs 0.2"

    def test_ece_bounded_zero_one(self) -> None:
        rng = np.random.default_rng(7)
        logits = rng.standard_normal(1000)
        labels = rng.integers(0, 2, size=1000)
        ece = expected_calibration_error(logits, labels)
        assert 0.0 <= ece <= 1.0

    def test_constant_prediction_matches_manual(self) -> None:
        """All predictions in one bin => ECE = |acc - conf|."""
        n = 1000
        p_pred = 0.7
        p_true = 0.3
        rng = np.random.default_rng(55)
        labels = rng.binomial(1, p_true, size=n)
        logits = np.full(n, _logit(p_pred))

        ece = expected_calibration_error(logits, labels, n_bins=10)
        actual_acc = labels.mean()
        expected = abs(actual_acc - p_pred)
        assert ece == pytest.approx(expected, abs=0.01)

    def test_ece_decreases_with_better_calibration(self) -> None:
        """Moving predictions toward true probabilities lowers ECE."""
        rng = np.random.default_rng(12)
        n = 10_000
        true_probs = rng.uniform(0.1, 0.9, size=n)
        labels = (rng.uniform(size=n) < true_probs).astype(int)

        logits_good = np.log(true_probs / (1.0 - true_probs))
        shifted = np.clip(true_probs + 0.3, 0.01, 0.99)
        logits_bad = np.log(shifted / (1.0 - shifted))

        ece_good = expected_calibration_error(logits_good, labels, n_bins=15)
        ece_bad = expected_calibration_error(logits_bad, labels, n_bins=15)
        assert ece_good < ece_bad

    def test_n_bins_affects_result(self) -> None:
        """Different n_bins produce different ECE values (not identical)."""
        rng = np.random.default_rng(3)
        logits = rng.standard_normal(5000)
        labels = rng.integers(0, 2, size=5000)
        ece_5 = expected_calibration_error(logits, labels, n_bins=5)
        ece_50 = expected_calibration_error(logits, labels, n_bins=50)
        assert ece_5 != pytest.approx(ece_50, abs=1e-10)
