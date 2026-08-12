"""Tests for evaluation.stats - McNemar, Wilson, adaptive stopping."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evaluation.stats import (
    FINAL_ALPHA,
    INTERIM_ALPHA,
    EvalOutcome,
    McNemarResult,
    adaptive_stop,
    mcnemar_test,
    per_round_ler,
    wilson_interval,
)


class TestMcNemarTest:
    """McNemar's test on synthetic disagreement matrices."""

    def test_perfect_agreement(self):
        """All shots agree: both correct or both wrong => p=1."""
        gnn = np.array([True, True, False, False, True])
        baseline = np.array([True, True, False, False, True])
        result = mcnemar_test(gnn, baseline)
        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert result.n_discordant == 0
        assert result.gnn_wins == 0
        assert result.baseline_wins == 0

    def test_all_gnn_wins(self):
        """GNN always correct, baseline always wrong - maximally different."""
        n = 200
        gnn = np.ones(n, dtype=bool)
        baseline = np.zeros(n, dtype=bool)
        result = mcnemar_test(gnn, baseline)
        # b=200, c=0: chi2 = 200^2 / 200 = 200
        assert result.statistic == pytest.approx(200.0)
        assert result.p_value < 1e-10
        assert result.gnn_wins == 200
        assert result.baseline_wins == 0

    def test_all_baseline_wins(self):
        """Baseline always correct, GNN always wrong."""
        n = 200
        gnn = np.zeros(n, dtype=bool)
        baseline = np.ones(n, dtype=bool)
        result = mcnemar_test(gnn, baseline)
        assert result.statistic == pytest.approx(200.0)
        assert result.p_value < 1e-10
        assert result.gnn_wins == 0
        assert result.baseline_wins == 200

    def test_symmetric_disagreement_parity(self):
        """Equal discordant counts => statistic=0, p=1."""
        n = 1000
        # Make exactly equal discordant pairs
        gnn_correct = np.zeros(n, dtype=bool)
        baseline_correct = np.zeros(n, dtype=bool)
        gnn_correct[:500] = True
        baseline_correct[:250] = True
        baseline_correct[500:750] = True
        # b = GNN right & baseline wrong = [250:500] = 250
        # c = GNN wrong & baseline right = [500:750] = 250
        result = mcnemar_test(gnn_correct, baseline_correct)
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert result.gnn_wins == 250
        assert result.baseline_wins == 250

    def test_known_statistic(self):
        """Hand-computed example: b=120, c=80 => chi2=8.0."""
        gnn = np.zeros(400, dtype=bool)
        baseline = np.zeros(400, dtype=bool)
        # Both correct: first 100
        gnn[:100] = True
        baseline[:100] = True
        # GNN correct, baseline wrong: next 120
        gnn[100:220] = True
        # GNN wrong, baseline correct: next 80
        baseline[220:300] = True
        # Both wrong: last 100

        result = mcnemar_test(gnn, baseline)
        assert result.gnn_wins == 120
        assert result.baseline_wins == 80
        assert result.n_discordant == 200
        expected_chi2 = (120 - 80) ** 2 / 200  # = 8.0
        assert result.statistic == pytest.approx(expected_chi2)
        # chi2(1) at 8.0 => p ≈ 0.00468
        assert 0.004 < result.p_value < 0.005

    def test_p_value_against_erfc(self):
        """p-value matches erfc(sqrt(chi2/2)) formula."""
        gnn = np.zeros(500, dtype=bool)
        baseline = np.zeros(500, dtype=bool)
        gnn[:300] = True
        baseline[:200] = True
        baseline[300:400] = True

        result = mcnemar_test(gnn, baseline)
        expected_p = math.erfc(math.sqrt(result.statistic / 2))
        assert result.p_value == pytest.approx(expected_p, rel=1e-10)

    def test_small_n_discordant(self):
        """Small discordant count still computes correctly."""
        gnn = np.array([True, True, False])
        baseline = np.array([True, False, True])
        # b=1, c=1
        result = mcnemar_test(gnn, baseline)
        assert result.statistic == 0.0
        assert result.p_value == pytest.approx(1.0)

    def test_single_discordant_pair(self):
        """One discordant pair: b=1, c=0 => chi2=1."""
        gnn = np.array([True, True])
        baseline = np.array([True, False])
        result = mcnemar_test(gnn, baseline)
        assert result.statistic == pytest.approx(1.0)
        assert result.n_discordant == 1
        # p for chi2(1) = 1.0: erfc(sqrt(0.5)) ≈ 0.3173
        assert 0.31 < result.p_value < 0.32

    def test_empty_raises(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            mcnemar_test(np.array([], dtype=bool), np.array([], dtype=bool))

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="shapes must match"):
            mcnemar_test(
                np.array([True, False]),
                np.array([True, False, True]),
            )

    def test_accepts_list_input(self):
        """Accepts Python lists, not just numpy arrays."""
        result = mcnemar_test([True, False, True], [True, True, False])
        assert isinstance(result, McNemarResult)


class TestWilsonInterval:
    """Wilson score confidence interval tests."""

    def test_reference_value_50pct(self):
        """50% rate with n=100 matches known values."""
        wi = wilson_interval(50, 100, alpha=0.05)
        assert wi.point == pytest.approx(0.5)
        # Wilson for 50/100: [0.4038, 0.5962]
        assert wi.lower == pytest.approx(0.4038, abs=0.001)
        assert wi.upper == pytest.approx(0.5962, abs=0.001)

    def test_reference_value_low_rate(self):
        """Low rate (1%) with n=10000: well-behaved near 0."""
        wi = wilson_interval(100, 10000, alpha=0.05)
        assert wi.point == pytest.approx(0.01)
        assert wi.lower > 0.0
        assert wi.upper > wi.lower
        # Wilson interval for p=0.01, n=10000: approximately [0.0082, 0.0122]
        assert 0.008 < wi.lower < 0.009
        assert 0.012 < wi.upper < 0.013

    def test_zero_errors(self):
        """Zero errors: lower bound is effectively 0, upper bound is positive."""
        wi = wilson_interval(0, 1000, alpha=0.05)
        assert wi.point == 0.0
        assert wi.lower < 1e-10  # Numerically zero (floating point)
        assert wi.upper > 0.0
        assert wi.upper < 0.01  # Reasonable upper bound for 0/1000

    def test_all_errors(self):
        """All errors: upper bound is 1, lower bound is less than 1."""
        wi = wilson_interval(1000, 1000, alpha=0.05)
        assert wi.point == pytest.approx(1.0)
        assert wi.upper == 1.0
        assert wi.lower < 1.0
        assert wi.lower > 0.99  # Reasonable lower bound for 1000/1000

    def test_interval_narrows_with_n(self):
        """Width decreases as sample size increases."""
        wi_small = wilson_interval(10, 100)
        wi_large = wilson_interval(100, 1000)
        width_small = wi_small.upper - wi_small.lower
        width_large = wi_large.upper - wi_large.lower
        assert width_large < width_small

    def test_interval_widens_with_confidence(self):
        """99% CI is wider than 95% CI."""
        wi_95 = wilson_interval(50, 1000, alpha=0.05)
        wi_99 = wilson_interval(50, 1000, alpha=0.01)
        width_95 = wi_95.upper - wi_95.lower
        width_99 = wi_99.upper - wi_99.lower
        assert width_99 > width_95

    def test_symmetry_at_half(self):
        """At p=0.5, interval is symmetric around 0.5."""
        wi = wilson_interval(500, 1000, alpha=0.05)
        center = (wi.lower + wi.upper) / 2
        assert center == pytest.approx(0.5, abs=0.001)

    def test_contains_true_rate(self):
        """95% CI from a known rate should contain that rate (single check)."""
        # 5% error rate, large sample
        wi = wilson_interval(500, 10000, alpha=0.05)
        assert wi.lower <= 0.05 <= wi.upper

    def test_invalid_n_total_raises(self):
        """n_total <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_total must be positive"):
            wilson_interval(0, 0)

    def test_negative_errors_raises(self):
        """Negative errors raises ValueError."""
        with pytest.raises(ValueError, match="n_errors must be non-negative"):
            wilson_interval(-1, 100)

    def test_errors_exceed_total_raises(self):
        """n_errors > n_total raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            wilson_interval(101, 100)

    def test_invalid_alpha_raises(self):
        """Alpha outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            wilson_interval(10, 100, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            wilson_interval(10, 100, alpha=1.0)


class TestPerRoundLER:
    """Per-round logical error rate conversion."""

    def test_zero_ler(self):
        """Zero LER => zero per-round."""
        assert per_round_ler(0.0, 5) == 0.0

    def test_one_ler(self):
        """LER=1 => per-round=1."""
        assert per_round_ler(1.0, 5) == 1.0

    def test_round_trip(self):
        """epsilon = 1 - (1 - LER)^(1/r); LER = 1 - (1 - epsilon)^r."""
        ler = 0.05
        r = 5
        eps = per_round_ler(ler, r)
        reconstructed = 1.0 - (1.0 - eps) ** r
        assert reconstructed == pytest.approx(ler, rel=1e-10)

    def test_single_round(self):
        """r=1: per-round = per-shot."""
        assert per_round_ler(0.03, 1) == pytest.approx(0.03)

    def test_invalid_rounds_raises(self):
        """rounds <= 0 raises."""
        with pytest.raises(ValueError, match="rounds must be positive"):
            per_round_ler(0.05, 0)

    def test_invalid_ler_raises(self):
        """LER outside [0, 1] raises."""
        with pytest.raises(ValueError, match="ler must be in"):
            per_round_ler(-0.01, 3)
        with pytest.raises(ValueError, match="ler must be in"):
            per_round_ler(1.01, 3)


class TestAdaptiveStop:
    """Adaptive early-stopping function tests."""

    def _make_arrays(
        self, n: int, gnn_error_rate: float, baseline_error_rate: float, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create correctness arrays with given error rates."""
        rng = np.random.default_rng(seed)
        gnn_correct = rng.random(n) >= gnn_error_rate
        baseline_correct = rng.random(n) >= baseline_error_rate
        return gnn_correct, baseline_correct

    def test_insufficient_errors_continue(self):
        """Below min errors at interim check => continue."""
        # Very low error rates: unlikely to reach 100 errors in 1000 shots
        gnn = np.ones(1000, dtype=bool)
        gnn[:50] = False  # Only 50 GNN errors
        baseline = np.ones(1000, dtype=bool)
        baseline[:50] = False  # Only 50 baseline errors

        decision = adaptive_stop(gnn, baseline, is_final=False)
        assert decision.action == "continue"
        assert decision.outcome is None
        assert decision.mcnemar is None
        assert "Insufficient" in decision.reason

    def test_insufficient_errors_final_unresolved(self):
        """Below min errors at final check => unresolved."""
        gnn = np.ones(1000, dtype=bool)
        gnn[:50] = False
        baseline = np.ones(1000, dtype=bool)
        baseline[:50] = False

        decision = adaptive_stop(gnn, baseline, is_final=True)
        assert decision.action == "stop"
        assert decision.outcome == EvalOutcome.UNRESOLVED
        assert decision.mcnemar is None

    def test_interim_boundary_crossed(self):
        """Large difference at interim => resolved-different."""
        n = 10000
        gnn = np.ones(n, dtype=bool)
        baseline = np.ones(n, dtype=bool)
        # GNN: 200 errors, baseline: 800 errors - clear difference
        gnn[:200] = False
        baseline[:800] = False

        decision = adaptive_stop(gnn, baseline, is_final=False)
        assert decision.action == "stop"
        assert decision.outcome == EvalOutcome.RESOLVED_DIFFERENT
        assert decision.mcnemar is not None
        assert decision.mcnemar.p_value < INTERIM_ALPHA

    def test_interim_not_crossed_continue(self):
        """Moderate difference at interim (p > 0.001) => continue."""
        n = 10000
        rng = np.random.default_rng(99)
        # Similar error rates: both ~5%
        gnn = rng.random(n) >= 0.05
        baseline = rng.random(n) >= 0.055

        decision = adaptive_stop(gnn, baseline, is_final=False)
        # With these similar rates, p should be > 0.001
        if decision.mcnemar is not None:
            if decision.mcnemar.p_value >= INTERIM_ALPHA:
                assert decision.action == "continue"
                assert decision.outcome is None

    def test_final_resolved_different(self):
        """Significant difference at final => resolved-different."""
        n = 50000
        gnn = np.ones(n, dtype=bool)
        baseline = np.ones(n, dtype=bool)
        # GNN: 1000 errors, baseline: 2000 errors
        gnn[:1000] = False
        baseline[:2000] = False

        decision = adaptive_stop(gnn, baseline, is_final=True)
        assert decision.action == "stop"
        assert decision.outcome == EvalOutcome.RESOLVED_DIFFERENT
        assert decision.mcnemar is not None
        assert decision.mcnemar.p_value < FINAL_ALPHA

    def test_final_resolved_parity(self):
        """No significant difference at final => resolved-parity."""
        n = 10000
        # Both decoders have ~same errors on ~same shots
        gnn = np.ones(n, dtype=bool)
        baseline = np.ones(n, dtype=bool)
        # Identical error pattern: both wrong on same 500 shots
        gnn[:500] = False
        baseline[:500] = False
        # Plus some discordance that's balanced
        gnn[500:600] = False  # Extra 100 GNN-only errors
        baseline[600:700] = False  # Extra 100 baseline-only errors
        # Total: GNN=600 errors, baseline=600 errors, b=100, c=100

        decision = adaptive_stop(gnn, baseline, is_final=True)
        assert decision.action == "stop"
        assert decision.outcome == EvalOutcome.RESOLVED_PARITY
        assert decision.mcnemar is not None
        assert decision.mcnemar.p_value >= FINAL_ALPHA

    def test_custom_thresholds(self):
        """Custom alpha and min_errors parameters are respected."""
        n = 5000
        gnn = np.ones(n, dtype=bool)
        baseline = np.ones(n, dtype=bool)
        gnn[:30] = False
        baseline[:30] = False

        # With min_errors=20, these 30 errors suffice
        decision = adaptive_stop(gnn, baseline, is_final=True, min_errors=20)
        assert decision.mcnemar is not None

        # With default min_errors=100, they don't
        decision = adaptive_stop(gnn, baseline, is_final=True)
        assert decision.mcnemar is None
        assert decision.outcome == EvalOutcome.UNRESOLVED

    def test_one_decoder_insufficient(self):
        """Only one decoder below threshold => insufficient."""
        n = 5000
        gnn = np.ones(n, dtype=bool)
        baseline = np.ones(n, dtype=bool)
        gnn[:200] = False  # 200 GNN errors - sufficient
        baseline[:50] = False  # 50 baseline errors - insufficient

        decision = adaptive_stop(gnn, baseline, is_final=False)
        assert decision.action == "continue"
        assert decision.mcnemar is None
        assert "Insufficient" in decision.reason
