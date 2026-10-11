"""
Calibration diagnostics: reliability diagram binning and Expected Calibration Error.

Pure functions operating on raw logits and binary labels. No model or dataset
dependencies - the plotting script handles loading and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluation.stats import WilsonInterval, wilson_interval


@dataclass(frozen=True, slots=True)
class ReliabilityDiagram:
    """Binned calibration data for a reliability diagram.

    Parameters
    ----------
    bin_confidences : ndarray, shape ``(n_bins,)``
        Mean predicted probability in each bin (NaN for empty bins).
    bin_accuracies : ndarray, shape ``(n_bins,)``
        Fraction of positive labels in each bin (NaN for empty bins).
    bin_counts : ndarray, shape ``(n_bins,)``
        Number of samples in each bin.
    bin_positives : ndarray, shape ``(n_bins,)``
        Number of positive labels in each bin.
    bin_edges : ndarray, shape ``(n_bins + 1,)``
        Bin boundary positions in [0, 1].
    """

    bin_confidences: NDArray[np.floating]
    bin_accuracies: NDArray[np.floating]
    bin_counts: NDArray[np.intp]
    bin_positives: NDArray[np.intp]
    bin_edges: NDArray[np.floating]

    @property
    def ece(self) -> float:
        """Expected Calibration Error: the count-weighted mean bin gap."""
        nonempty = self.bin_counts > 0
        weights = self.bin_counts[nonempty] / self.bin_counts.sum()
        gaps = np.abs(self.bin_accuracies[nonempty] - self.bin_confidences[nonempty])
        return float(weights @ gaps)

    def accuracy_interval(self, bin_index: int) -> WilsonInterval:
        """Wilson 95% interval of a non-empty bin's fraction of positives."""
        return wilson_interval(
            int(self.bin_positives[bin_index]), int(self.bin_counts[bin_index])
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Returns
        -------
        dict
            Sample and positive counts, the ECE, and every non-empty bin with
            its edges, counts, mean confidence and fraction of positives with
            its Wilson 95% interval.  Empty bins are left out.
        """
        bins = []
        for b in np.flatnonzero(self.bin_counts):
            interval = self.accuracy_interval(b)
            bins.append(
                {
                    "lower": float(self.bin_edges[b]),
                    "upper": float(self.bin_edges[b + 1]),
                    "n_samples": int(self.bin_counts[b]),
                    "n_positives": int(self.bin_positives[b]),
                    "mean_confidence": float(self.bin_confidences[b]),
                    "positive_fraction": float(self.bin_accuracies[b]),
                    "positive_fraction_ci_95": [interval.lower, interval.upper],
                }
            )
        return {
            "n_samples": int(self.bin_counts.sum()),
            "n_positives": int(self.bin_positives.sum()),
            "ece": self.ece,
            "bins": bins,
        }


def reliability_diagram(
    logits: NDArray[np.floating],
    labels: NDArray[np.integer],
    n_bins: int = 15,
) -> ReliabilityDiagram:
    """Compute binned calibration data for a reliability diagram.

    Bins predicted probabilities (sigmoid of logits) into ``n_bins``
    equal-width bins over [0, 1]. For each non-empty bin, reports the
    mean predicted probability (confidence) and the observed positive
    fraction (accuracy). Empty bins receive NaN.

    Parameters
    ----------
    logits : ndarray, shape ``(N,)``
        Raw model logits (pre-sigmoid).
    labels : ndarray, shape ``(N,)``
        Binary ground-truth labels in {0, 1}.
    n_bins : int
        Number of equal-width bins (default: 15).

    Returns
    -------
    ReliabilityDiagram
        Binned confidences, accuracies, counts, and bin edges.

    Raises
    ------
    ValueError
        If inputs are empty, shapes mismatch, a logit is not finite, labels
        are not binary, or n_bins < 1.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()

    if logits.size == 0:
        raise ValueError("Cannot compute reliability diagram on empty arrays")
    if logits.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, labels={labels.shape}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1)")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    # sigmoid(x) = exp(-log(1 + exp(-x))), finite for every finite x.
    probs = np.exp(-np.logaddexp(0.0, -logits))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1])
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_positives = np.bincount(bin_indices[labels == 1], minlength=n_bins)
    confidence_sums = np.bincount(bin_indices, weights=probs, minlength=n_bins)

    nonempty = bin_counts > 0
    bin_confidences = np.full(n_bins, np.nan)
    bin_accuracies = np.full(n_bins, np.nan)
    np.divide(confidence_sums, bin_counts, out=bin_confidences, where=nonempty)
    np.divide(bin_positives, bin_counts, out=bin_accuracies, where=nonempty)

    return ReliabilityDiagram(
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        bin_positives=bin_positives,
        bin_edges=bin_edges,
    )


def expected_calibration_error(
    logits: NDArray[np.floating],
    labels: NDArray[np.integer],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE).

    Weighted average of per-bin |accuracy - confidence|, where weights
    are the fraction of total samples in each bin. Empty bins contribute
    zero.

    Parameters
    ----------
    logits : ndarray, shape ``(N,)``
        Raw model logits (pre-sigmoid).
    labels : ndarray, shape ``(N,)``
        Binary ground-truth labels in {0, 1}.
    n_bins : int
        Number of equal-width bins (default: 15).

    Returns
    -------
    float
        ECE in [0, 1].
    """
    return reliability_diagram(logits, labels, n_bins=n_bins).ece
