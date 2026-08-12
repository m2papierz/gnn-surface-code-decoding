"""
Calibration diagnostics: reliability diagram binning and Expected Calibration Error.

Pure functions operating on raw logits and binary labels. No model or dataset
dependencies - the plotting script handles loading and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
    bin_edges : ndarray, shape ``(n_bins + 1,)``
        Bin boundary positions in [0, 1].
    """

    bin_confidences: NDArray[np.floating]
    bin_accuracies: NDArray[np.floating]
    bin_counts: NDArray[np.intp]
    bin_edges: NDArray[np.floating]


def _sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Numerically stable sigmoid."""
    pos = x >= 0
    z = np.empty_like(x, dtype=np.float64)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    z[~pos] = exp_x / (1.0 + exp_x)
    return z


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
        If inputs are empty, shapes mismatch, labels are not binary,
        or n_bins < 1.
    """
    logits = np.asarray(logits, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel()

    if logits.size == 0:
        raise ValueError("Cannot compute reliability diagram on empty arrays")
    if logits.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, labels={labels.shape}"
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1)")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    labels = labels.astype(np.float64)
    probs = _sigmoid(logits)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges[1:-1])

    bin_confidences = np.full(n_bins, np.nan, dtype=np.float64)
    bin_accuracies = np.full(n_bins, np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.intp)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(mask.sum())
        bin_counts[b] = count
        if count > 0:
            bin_confidences[b] = probs[mask].mean()
            bin_accuracies[b] = labels[mask].mean()

    return ReliabilityDiagram(
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
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
    diag = reliability_diagram(logits, labels, n_bins=n_bins)
    n_total = int(diag.bin_counts.sum())
    nonempty = diag.bin_counts > 0
    weights = diag.bin_counts[nonempty] / n_total
    gaps = np.abs(diag.bin_accuracies[nonempty] - diag.bin_confidences[nonempty])
    return float((weights * gaps).sum())
