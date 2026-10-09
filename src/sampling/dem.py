"""Error mechanisms of a decomposed detector error model.

``stim`` lists each mechanism of a decomposed DEM as one or more graphlike
components separated by ``^``.  This module reads that listing once into
arrays, so every consumer - the ``dem_weight`` edge feature, the scaling
diagnostics - sees the same mechanisms through the same reading.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import numpy as np
import stim
from numpy.typing import NDArray


__all__ = [
    "DemMechanisms",
    "MechanismClass",
    "error_instructions",
    "read_dem_mechanisms",
]


class MechanismClass(enum.StrEnum):
    """Structural class of one error mechanism in a decomposed DEM."""

    TWO_DETECTOR = "two_detector"
    """One graphlike component flipping two detectors."""

    BOUNDARY = "boundary"
    """One graphlike component flipping a single detector."""

    DECOMPOSED = "decomposed"
    """Two or more graphlike components: a hyperedge stim has decomposed."""


@dataclass(frozen=True, slots=True)
class DemMechanisms:
    """Error mechanisms of a decomposed DEM, one row per mechanism.

    Rows keep the order of the DEM's error instructions.  Detectors are
    stored ragged: mechanism ``m`` flips
    ``detectors[detector_offsets[m]:detector_offsets[m + 1]]``, in the order
    the DEM lists them across all of its components.

    Parameters
    ----------
    probability : ndarray, shape ``(M,)``, float64
        Probability that the mechanism occurs.
    num_components : ndarray, shape ``(M,)``, intp
        Graphlike components the mechanism is listed as; 1 means the
        mechanism was already graphlike.
    detector_offsets : ndarray, shape ``(M + 1,)``, intp
        Row boundaries into ``detectors``.
    detectors : ndarray, shape ``(detector_offsets[-1],)``, intp
        Detector indices of all mechanisms, concatenated.
    """

    probability: NDArray[np.float64]
    num_components: NDArray[np.intp]
    detector_offsets: NDArray[np.intp]
    detectors: NDArray[np.intp]

    def __post_init__(self) -> None:
        num_mechanisms = self.probability.shape[0]
        if self.probability.shape != (num_mechanisms,):
            raise ValueError(
                f"probability must have shape (M,), got {self.probability.shape}"
            )
        if self.num_components.shape != (num_mechanisms,):
            raise ValueError(
                f"num_components shape {self.num_components.shape} != "
                f"expected ({num_mechanisms},)"
            )
        if self.detector_offsets.shape != (num_mechanisms + 1,):
            raise ValueError(
                f"detector_offsets shape {self.detector_offsets.shape} != "
                f"expected ({num_mechanisms + 1},)"
            )
        if self.detectors.shape != (int(self.detector_offsets[-1]),):
            raise ValueError(
                f"detectors shape {self.detectors.shape} != expected "
                f"({int(self.detector_offsets[-1])},) from detector_offsets"
            )

    @property
    def num_detectors(self) -> NDArray[np.intp]:
        """Detectors each mechanism flips, shape ``(M,)``."""
        return np.diff(self.detector_offsets)

    def class_masks(self) -> dict[MechanismClass, NDArray[np.bool_]]:
        """Partition the mechanisms by structural class.

        Returns
        -------
        dict[MechanismClass, ndarray]
            One boolean mask of shape ``(M,)`` per class.  Every mechanism is
            in exactly one.

        Raises
        ------
        ValueError
            If an undecomposed mechanism flips neither one nor two detectors.
            Such a mechanism is not graphlike, so the model it came from is
            not a decomposed DEM of a circuit a matching decoder can read.
        """
        undecomposed = self.num_components == 1
        num_detectors = self.num_detectors
        masks = {
            MechanismClass.TWO_DETECTOR: undecomposed & (num_detectors == 2),
            MechanismClass.BOUNDARY: undecomposed & (num_detectors == 1),
            MechanismClass.DECOMPOSED: ~undecomposed,
        }
        unclassified = np.flatnonzero(~np.logical_or.reduce(list(masks.values())))
        if unclassified.size:
            first = int(unclassified[0])
            raise ValueError(
                f"{unclassified.size} undecomposed mechanism(s) are not graphlike; "
                f"the first is mechanism {first}, flipping "
                f"{int(num_detectors[first])} detectors with probability "
                f"{float(self.probability[first])!r}"
            )
        return masks


def error_instructions(
    dem: stim.DetectorErrorModel,
) -> Iterator[tuple[float, list[stim.DemTarget]]]:
    """Yield ``(probability, targets)`` per error instruction of the flattened model.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Any detector error model; repeat blocks are unrolled.

    Returns
    -------
    Iterator[tuple[float, list[stim.DemTarget]]]
        One item per ``error`` instruction, in order.
    """
    for instruction in dem.flattened():
        assert isinstance(instruction, stim.DemInstruction), "flattened has no blocks"
        if instruction.type == "error":
            # The stub's ``int`` targets belong to ``shift_detectors``; an error
            # instruction targets only detectors, observables and separators.
            targets = cast(list[stim.DemTarget], instruction.targets_copy())
            yield instruction.args_copy()[0], targets


def read_dem_mechanisms(dem: stim.DetectorErrorModel) -> DemMechanisms:
    """Read the error mechanisms of a decomposed detector error model.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Model produced with ``decompose_errors=True``.

    Returns
    -------
    DemMechanisms
        One row per error instruction of the flattened model, in order.
    """
    probabilities: list[float] = []
    component_counts: list[int] = []
    detectors: list[int] = []
    offsets: list[int] = [0]
    for probability, targets in error_instructions(dem):
        probabilities.append(probability)
        component_counts.append(1 + sum(t.is_separator() for t in targets))
        detectors.extend(t.val for t in targets if t.is_relative_detector_id())
        offsets.append(len(detectors))
    return DemMechanisms(
        probability=np.array(probabilities, dtype=np.float64),
        num_components=np.array(component_counts, dtype=np.intp),
        detector_offsets=np.array(offsets, dtype=np.intp),
        detectors=np.array(detectors, dtype=np.intp),
    )
