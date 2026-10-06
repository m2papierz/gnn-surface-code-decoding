"""The sub-threshold rule: does a decoder's error rate fall with distance?

For one operation, one decoder and one physical error rate, the distances are
independent samples: different circuits, different shots.  Each pair of
consecutive distances ``d < d'`` is a *step*, decided by a pooled
two-proportion z-test of the per-shot logical error rates, two-sided at
``FINAL_ALPHA``.  A step is *decreasing* when the test rejects and the rate
at ``d'`` is the lower one, *increasing* when it rejects the other way, and
*unresolved* when it does not reject or the smallest expected cell is below
``MIN_EXPECTED_CELL_COUNT``.

The error rate is *sub-threshold* when there is at least one step and every
step is decreasing, *above-threshold* when every step is increasing, and
*undetermined* otherwise.  A threshold belongs to a decoder, so the rule is
applied to each decoder separately.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import asdict, dataclass
from typing import Any

from evaluation.stats import (
    FINAL_ALPHA,
    MIN_EXPECTED_CELL_COUNT,
    two_proportion_test,
)


__all__ = [
    "DistanceRate",
    "StepTrend",
    "ThresholdClassification",
    "ThresholdVerdict",
    "rule_to_dict",
]


class StepTrend(enum.StrEnum):
    """How a decoder's error rate moves between two consecutive distances."""

    DECREASING = "decreasing"
    INCREASING = "increasing"
    UNRESOLVED = "unresolved"


class ThresholdVerdict(enum.StrEnum):
    """Exhaustive verdicts of the sub-threshold rule."""

    SUB_THRESHOLD = "sub-threshold"
    ABOVE_THRESHOLD = "above-threshold"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True)
class DistanceRate:
    """One decoder's error count at one distance.

    Parameters
    ----------
    distance : int
        Code distance.
    n_errors : int
        Shots with any observable decoded wrongly.
    n_shots : int
        Shots scored.
    """

    distance: int
    n_errors: int
    n_shots: int


@dataclass(frozen=True, slots=True)
class DistanceStep:
    """Test of one decoder's error rate between two consecutive distances.

    Parameters
    ----------
    from_distance, to_distance : int
        The smaller and the larger distance.
    statistic : float
        z statistic; positive when the rate is lower at the larger distance.
    p_value : float
        Two-sided p-value.
    min_expected_count : float
        Smallest expected cell under the pooled rate.
    trend : StepTrend
        Verdict of the step.
    """

    from_distance: int
    to_distance: int
    statistic: float
    p_value: float
    min_expected_count: float
    trend: StepTrend


def _distance_step(smaller: DistanceRate, larger: DistanceRate) -> DistanceStep:
    """Test the error rate at the larger distance against the smaller one."""
    test = two_proportion_test(
        smaller.n_errors, smaller.n_shots, larger.n_errors, larger.n_shots
    )
    rejects = (
        test.min_expected_count >= MIN_EXPECTED_CELL_COUNT
        and test.p_value < FINAL_ALPHA
    )
    if not rejects:
        trend = StepTrend.UNRESOLVED
    elif test.statistic > 0:
        trend = StepTrend.DECREASING
    else:
        trend = StepTrend.INCREASING
    return DistanceStep(
        from_distance=smaller.distance,
        to_distance=larger.distance,
        statistic=test.statistic,
        p_value=test.p_value,
        min_expected_count=test.min_expected_count,
        trend=trend,
    )


@dataclass(frozen=True, slots=True)
class ThresholdClassification:
    """Sub-threshold verdict for one operation, decoder and error rate.

    The record holds the counts; the steps and the verdict are derived from
    them, so a verdict cannot disagree with the counts beside it.

    Parameters
    ----------
    operation : str
        Operation the rates were measured on.
    decoder : str
        Decoder the verdict is about.
    error_prob : float
        Physical error probability.
    rates : tuple of DistanceRate
        The decoder's error counts, one per distance, ascending.
    """

    operation: str
    decoder: str
    error_prob: float
    rates: tuple[DistanceRate, ...]

    def __post_init__(self) -> None:
        distances = [rate.distance for rate in self.rates]
        if distances != sorted(set(distances)):
            raise ValueError(
                f"rates of {self.decoder} on {self.operation} at "
                f"p={self.error_prob:g} must have distinct ascending distances, "
                f"got {distances}"
            )

    @property
    def steps(self) -> tuple[DistanceStep, ...]:
        """One test per pair of consecutive distances."""
        return tuple(
            _distance_step(smaller, larger)
            for smaller, larger in itertools.pairwise(self.rates)
        )

    @property
    def verdict(self) -> ThresholdVerdict:
        """Verdict of the sub-threshold rule over all steps."""
        trends = {step.trend for step in self.steps}
        if trends == {StepTrend.DECREASING}:
            return ThresholdVerdict.SUB_THRESHOLD
        if trends == {StepTrend.INCREASING}:
            return ThresholdVerdict.ABOVE_THRESHOLD
        return ThresholdVerdict.UNDETERMINED

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "operation": self.operation,
            "decoder": self.decoder,
            "error_prob": self.error_prob,
            "rates": [asdict(rate) for rate in self.rates],
            "steps": [asdict(step) for step in self.steps],
            "verdict": self.verdict.value,
        }


def rule_to_dict() -> dict[str, Any]:
    """State the rule and its registered parameters, for a result artifact."""
    return {
        "metric": "per-shot logical error rate, any observable wrong",
        "step_test": (
            "pooled two-proportion z-test between consecutive distances, two-sided"
        ),
        "alpha": FINAL_ALPHA,
        "min_expected_cell_count": MIN_EXPECTED_CELL_COUNT,
        "verdicts": {
            ThresholdVerdict.SUB_THRESHOLD.value: "at least one step, all decreasing",
            ThresholdVerdict.ABOVE_THRESHOLD.value: "at least one step, all increasing",
            ThresholdVerdict.UNDETERMINED.value: "anything else",
        },
    }
