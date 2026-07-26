"""Evaluation utilities: statistical tests, intervals, adaptive stopping, harness."""

from __future__ import annotations

from evaluation.evaluator import EvalPointResult, EvalSet, evaluate_point
from evaluation.stats import EvalOutcome, WilsonInterval, mcnemar_test, wilson_interval


__all__ = [
    "EvalOutcome",
    "EvalPointResult",
    "EvalSet",
    "WilsonInterval",
    "evaluate_point",
    "mcnemar_test",
    "wilson_interval",
]
