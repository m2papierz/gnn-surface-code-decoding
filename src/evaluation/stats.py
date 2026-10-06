"""
Statistical tests and intervals for paired decoder evaluation.

Provides McNemar's test on the per-shot disagreement matrix, the pooled
two-proportion z-test for rates measured on independent samples, Wilson score
confidence intervals for binomial error rates, and an adaptive early-stopping
function implementing the pre-registered Haybittle-Peto boundary.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final

import numpy as np
from numpy.typing import NDArray


INTERIM_ALPHA: Final[float] = 0.001
FINAL_ALPHA: Final[float] = 0.05
CHECK_INTERVAL: Final[int] = 10_000
MIN_SHOTS: Final[int] = 10_000
MAX_SHOTS: Final[int] = 1_000_000
MIN_ERRORS_PER_DECODER: Final[int] = 100
MIN_EXPECTED_CELL_COUNT: Final[int] = 5


class EvalOutcome(enum.StrEnum):
    """Exhaustive evaluation point outcomes."""

    RESOLVED_DIFFERENT = "resolved-different"
    RESOLVED_PARITY = "resolved-parity"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """
    Result of McNemar's test on the disagreement matrix.

    Parameters
    ----------
    statistic : float
        Chi-squared test statistic (b - c)^2 / (b + c).
    p_value : float
        Two-sided p-value from the chi-squared(1) distribution.
    n_discordant : int
        Total discordant pairs (b + c).
    gnn_wins : int
        Shots where GNN correct and baseline wrong (b).
    baseline_wins : int
        Shots where baseline correct and GNN wrong (c).
    """

    statistic: float
    p_value: float
    n_discordant: int
    gnn_wins: int
    baseline_wins: int


@dataclass(frozen=True, slots=True)
class WilsonInterval:
    """
    Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    lower, upper : float
        Interval bounds.
    point : float
        Point estimate (k / n).
    n_errors : int
        Number of errors observed.
    n_total : int
        Total number of trials.
    alpha : float
        Significance level (1 - confidence).
    """

    lower: float
    upper: float
    point: float
    n_errors: int
    n_total: int
    alpha: float


@dataclass(frozen=True, slots=True)
class TwoProportionResult:
    """
    Result of the pooled two-proportion z-test on two independent samples.

    Parameters
    ----------
    statistic : float
        z statistic; positive when the first sample's rate is the higher one.
    p_value : float
        Two-sided p-value from the standard normal distribution.
    min_expected_count : float
        Smallest expected cell of the 2x2 table under the pooled rate. The
        normal approximation is conventionally trusted from
        ``MIN_EXPECTED_CELL_COUNT`` upward (Cochran's rule).
    """

    statistic: float
    p_value: float
    min_expected_count: float


@dataclass(frozen=True, slots=True)
class StoppingDecision:
    """
    Adaptive stopping decision at a check point.

    Parameters
    ----------
    outcome : EvalOutcome or None
        If not None, evaluation should stop with this outcome.
    action : str
        "stop" or "continue".
    reason : str
        Human-readable explanation.
    mcnemar : McNemarResult or None
        Test result if computed (None if preconditions not met).
    """

    outcome: EvalOutcome | None
    action: str
    reason: str
    mcnemar: McNemarResult | None


def mcnemar_test(
    gnn_correct: NDArray[np.bool_],
    baseline_correct: NDArray[np.bool_],
) -> McNemarResult:
    """
    McNemar's test on the per-shot disagreement matrix.

    Computes the chi-squared statistic on the off-diagonal cells of the 2x2
    contingency table (GNN correct vs baseline correct). No continuity
    correction - the protocol requires >=100 errors per decoder, ensuring
    sufficient off-diagonal mass for the chi-squared approximation.

    Parameters
    ----------
    gnn_correct : ndarray of bool, shape (N,)
        True where the GNN decoder produced the correct logical observable.
    baseline_correct : ndarray of bool, shape (N,)
        True where the baseline decoder produced the correct logical observable.

    Returns
    -------
    McNemarResult
        Test statistic, p-value, and disagreement counts.

    Raises
    ------
    ValueError
        If inputs have different lengths or are empty.
    """
    gnn_correct = np.asarray(gnn_correct, dtype=np.bool_)
    baseline_correct = np.asarray(baseline_correct, dtype=np.bool_)

    if gnn_correct.shape != baseline_correct.shape:
        raise ValueError(
            f"Input shapes must match: gnn_correct={gnn_correct.shape}, "
            f"baseline_correct={baseline_correct.shape}"
        )
    if gnn_correct.size == 0:
        raise ValueError("Cannot compute McNemar test on empty arrays")

    b = int(np.sum(gnn_correct & ~baseline_correct))
    c = int(np.sum(~gnn_correct & baseline_correct))

    n_discordant = b + c

    if n_discordant == 0:
        return McNemarResult(
            statistic=0.0,
            p_value=1.0,
            n_discordant=0,
            gnn_wins=0,
            baseline_wins=0,
        )

    chi2 = (b - c) ** 2 / n_discordant
    # chi2(1) survival function: P(Z^2 > t) = erfc(sqrt(t/2))
    p_value = math.erfc(math.sqrt(chi2 / 2))

    return McNemarResult(
        statistic=float(chi2),
        p_value=p_value,
        n_discordant=n_discordant,
        gnn_wins=b,
        baseline_wins=c,
    )


def two_proportion_test(
    n_errors_a: int,
    n_total_a: int,
    n_errors_b: int,
    n_total_b: int,
) -> TwoProportionResult:
    """
    Pooled two-proportion z-test of equal error rates in two independent samples.

    The unpaired counterpart of McNemar's test: it applies when the two rates
    come from different shots, such as one decoder at two code distances.

    Parameters
    ----------
    n_errors_a, n_total_a : int
        Errors and trials in the first sample.
    n_errors_b, n_total_b : int
        Errors and trials in the second sample.

    Returns
    -------
    TwoProportionResult
        z statistic, two-sided p-value, and the smallest expected cell. When
        the pooled rate is 0 or 1 the samples cannot differ: the statistic is
        0 and the p-value 1.

    Raises
    ------
    ValueError
        If a sample has no trials, or an error count outside ``[0, n_total]``.
    """
    for label, n_errors, n_total in (
        ("a", n_errors_a, n_total_a),
        ("b", n_errors_b, n_total_b),
    ):
        if n_total <= 0:
            raise ValueError(f"n_total_{label} must be positive, got {n_total}")
        if not (0 <= n_errors <= n_total):
            raise ValueError(
                f"n_errors_{label} must be in [0, {n_total}], got {n_errors}"
            )

    pooled = (n_errors_a + n_errors_b) / (n_total_a + n_total_b)
    min_expected_count = min(n_total_a, n_total_b) * min(pooled, 1.0 - pooled)
    variance = pooled * (1.0 - pooled) * (1.0 / n_total_a + 1.0 / n_total_b)
    if variance == 0.0:
        return TwoProportionResult(
            statistic=0.0, p_value=1.0, min_expected_count=min_expected_count
        )

    z = (n_errors_a / n_total_a - n_errors_b / n_total_b) / math.sqrt(variance)
    # Standard normal two-sided tail: P(|Z| > z) = erfc(|z| / sqrt(2))
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return TwoProportionResult(
        statistic=z, p_value=p_value, min_expected_count=min_expected_count
    )


def wilson_interval(
    n_errors: int,
    n_total: int,
    alpha: float = 0.05,
) -> WilsonInterval:
    """
    Wilson score confidence interval for a binomial proportion.

    Well-behaved near 0 and 1, unlike the Wald interval.

    Parameters
    ----------
    n_errors : int
        Number of observed errors (successes in binomial sense).
    n_total : int
        Total number of trials.
    alpha : float
        Significance level; the interval has coverage 1 - alpha.

    Returns
    -------
    WilsonInterval
        Interval bounds, point estimate, and input parameters.

    Raises
    ------
    ValueError
        If n_total <= 0 or n_errors < 0 or n_errors > n_total or alpha
        not in (0, 1).
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    if n_errors < 0:
        raise ValueError(f"n_errors must be non-negative, got {n_errors}")
    if n_errors > n_total:
        raise ValueError(f"n_errors ({n_errors}) cannot exceed n_total ({n_total})")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    p_hat = n_errors / n_total
    z = NormalDist().inv_cdf(1 - alpha / 2)
    z2 = z * z
    n = n_total

    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) / denom

    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)

    return WilsonInterval(
        lower=float(lower),
        upper=float(upper),
        point=p_hat,
        n_errors=n_errors,
        n_total=n_total,
        alpha=alpha,
    )


def per_round_ler(ler: float, rounds: int) -> float:
    """
    Convert per-shot LER to per-round LER (epsilon).

    epsilon = 1 - (1 - LER)^(1/r)

    Parameters
    ----------
    ler : float
        Per-shot logical error rate.
    rounds : int
        Number of syndrome measurement rounds.

    Returns
    -------
    float
        Per-round logical error rate.

    Raises
    ------
    ValueError
        If rounds <= 0 or ler not in [0, 1].
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if not (0.0 <= ler <= 1.0):
        raise ValueError(f"ler must be in [0, 1], got {ler}")

    if ler == 0.0:
        return 0.0
    if ler == 1.0:
        return 1.0

    return 1.0 - (1.0 - ler) ** (1.0 / rounds)


def adaptive_stop(
    gnn_correct: NDArray[np.bool_],
    baseline_correct: NDArray[np.bool_],
    *,
    is_final: bool = False,
    interim_alpha: float = INTERIM_ALPHA,
    final_alpha: float = FINAL_ALPHA,
    min_errors: int = MIN_ERRORS_PER_DECODER,
) -> StoppingDecision:
    """
    Evaluate the adaptive stopping criterion at a check point.

    Implements the Haybittle-Peto boundary: stop early at interim checks if
    McNemar p < interim_alpha; at the final check, decide at final_alpha.

    Parameters
    ----------
    gnn_correct : ndarray of bool, shape (N,)
        Cumulative GNN correctness vector up to this check point.
    baseline_correct : ndarray of bool, shape (N,)
        Cumulative baseline correctness vector up to this check point.
    is_final : bool
        True if this is the last check (set exhausted or max shots reached).
    interim_alpha : float
        Interim stopping boundary (default: 0.001).
    final_alpha : float
        Final analysis significance level (default: 0.05).
    min_errors : int
        Minimum errors per decoder for a valid test (default: 100).

    Returns
    -------
    StoppingDecision
        Whether to stop or continue, with outcome and reasoning.
    """
    gnn_correct = np.asarray(gnn_correct, dtype=np.bool_)
    baseline_correct = np.asarray(baseline_correct, dtype=np.bool_)

    n_shots = gnn_correct.size
    gnn_errors = int(np.sum(~gnn_correct))
    baseline_errors = int(np.sum(~baseline_correct))

    errors_sufficient = gnn_errors >= min_errors and baseline_errors >= min_errors

    if not errors_sufficient:
        if is_final:
            return StoppingDecision(
                outcome=EvalOutcome.UNRESOLVED,
                action="stop",
                reason=(
                    f"Insufficient errors at final check: "
                    f"GNN={gnn_errors}, baseline={baseline_errors} "
                    f"(need {min_errors} each), n_shots={n_shots}"
                ),
                mcnemar=None,
            )
        return StoppingDecision(
            outcome=None,
            action="continue",
            reason=(
                f"Insufficient errors: GNN={gnn_errors}, "
                f"baseline={baseline_errors} (need {min_errors} each)"
            ),
            mcnemar=None,
        )

    result = mcnemar_test(gnn_correct, baseline_correct)

    if not is_final:
        if result.p_value < interim_alpha:
            return StoppingDecision(
                outcome=EvalOutcome.RESOLVED_DIFFERENT,
                action="stop",
                reason=(
                    f"Interim boundary crossed: p={result.p_value:.2e} "
                    f"< {interim_alpha}, n_shots={n_shots}"
                ),
                mcnemar=result,
            )
        return StoppingDecision(
            outcome=None,
            action="continue",
            reason=(
                f"Interim check: p={result.p_value:.4f} >= {interim_alpha}, "
                f"n_shots={n_shots}"
            ),
            mcnemar=result,
        )

    # Final check
    if result.p_value < final_alpha:
        return StoppingDecision(
            outcome=EvalOutcome.RESOLVED_DIFFERENT,
            action="stop",
            reason=(
                f"Final analysis: p={result.p_value:.2e} < {final_alpha}, "
                f"n_shots={n_shots}"
            ),
            mcnemar=result,
        )

    return StoppingDecision(
        outcome=EvalOutcome.RESOLVED_PARITY,
        action="stop",
        reason=(
            f"Final analysis: p={result.p_value:.4f} >= {final_alpha}, "
            f"cannot reject H0, n_shots={n_shots}"
        ),
        mcnemar=result,
    )
