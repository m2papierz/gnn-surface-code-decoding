"""Metadata of committed circuits: the extractor each operation registers.

An extractor takes the path of a committed ``.stim`` circuit and returns the
``CircuitMetadata`` its representation's builder reads.  Distance and rounds
come from the manifest beside the circuit, never from the caller, and so does
a logical operation's layout: the phase windows, seam detectors and patch ids
its generator derived and validated.

``build_layout`` is the one place that turns those declarations into a
per-detector layout.  The generation gate calls it before a circuit is
written, and ``read_logical_op_metadata`` calls it on every manifest it reads,
so a manifest that no longer matches its circuit is rejected at load rather
than trained on.  Nothing here imports tqec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import stim
from numpy.typing import NDArray

from sampling.experiment import (
    ExperimentKey,
    circuit_key,
    circuit_manifest_path,
    manifest_key,
    read_circuit_manifest,
)
from sampling.graph import (
    MERGE_PHASE,
    CircuitMetadata,
    LogicalOperationLayout,
    extract_circuit_metadata,
)


__all__ = [
    "PhaseWindow",
    "assign_phase_ids",
    "build_layout",
    "read_circuit_metadata",
    "read_logical_op_metadata",
]


# Manifest fields a logical-operation circuit declares its layout through.
_LAYOUT_FIELDS: Final[tuple[str, ...]] = (
    "phase_windows",
    "seam_detector_indices",
    "patch_ids",
    "num_blocks",
)


@dataclass(frozen=True, slots=True)
class PhaseWindow:
    """A named half-open window ``[t_start, t_end)`` over detector time.

    Parameters
    ----------
    name : str
        Phase label (``"memory"``, ``"merge"``).
    t_start : float
        Inclusive lower bound.
    t_end : float
        Exclusive upper bound.
    """

    name: str
    t_start: float
    t_end: float

    def __post_init__(self) -> None:
        if not self.t_end > self.t_start:
            raise ValueError(
                f"PhaseWindow {self.name!r}: t_end ({self.t_end}) must be > "
                f"t_start ({self.t_start})"
            )

    def contains_time(self, t: float) -> bool:
        """Return whether *t* falls inside the window."""
        return self.t_start <= t < self.t_end


def assign_phase_ids(
    phase_windows: Sequence[PhaseWindow],
    detector_times: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Assign every detector the one phase window containing its time.

    The windows must partition a contiguous timeline: non-empty, each
    starting where the previous one ends.  Contiguity makes the boundaries
    sorted, so window membership is a single ``searchsorted`` and no detector
    can lie in two windows.

    Parameters
    ----------
    phase_windows : sequence of PhaseWindow
        Ordered, contiguous windows.
    detector_times : ndarray, shape ``(D,)``, float64
        Time coordinate of each detector.

    Returns
    -------
    ndarray, shape ``(D,)``, intp
        Index into ``phase_windows`` of each detector's window.

    Raises
    ------
    ValueError
        If no window is declared, if two consecutive windows do not meet, or
        if a detector lies outside the timeline (naming its index and time).
    """
    if not phase_windows:
        raise ValueError("no phase windows declared")
    for i, (window, following) in enumerate(pairwise(phase_windows)):
        if window.t_end != following.t_start:
            raise ValueError(
                f"phase windows are not contiguous: window {i} ends at "
                f"{window.t_end}, window {i + 1} starts at {following.t_start}"
            )

    boundaries = np.array(
        [phase_windows[0].t_start, *(window.t_end for window in phase_windows)]
    )
    phase_ids = np.searchsorted(boundaries, detector_times, side="right") - 1
    # NaN sorts past every boundary, so a detector without a time is caught here.
    uncovered = np.flatnonzero((phase_ids < 0) | (phase_ids >= len(phase_windows)))
    if uncovered.size:
        first = int(uncovered[0])
        raise ValueError(
            f"detector {first} at t={detector_times[first]} is not covered by any "
            f"phase window; the windows span [{boundaries[0]}, {boundaries[-1]}) "
            f"and {uncovered.size} detector(s) fall outside"
        )
    return phase_ids.astype(np.intp, copy=False)


def build_layout(
    phase_windows: Sequence[PhaseWindow],
    *,
    seam_detector_indices: Sequence[int],
    patch_ids: Sequence[int],
    num_blocks: int,
    detector_times: NDArray[np.float64],
) -> LogicalOperationLayout:
    """Validate a logical operation's declared structure and build its layout.

    Parameters
    ----------
    phase_windows : sequence of PhaseWindow
        Ordered, contiguous windows partitioning the detector timeline.
    seam_detector_indices : sequence of int
        Detectors on the merge boundary.
    patch_ids : sequence of int, length ``D``
        Patch (code-block) of each detector.
    num_blocks : int
        Number of patches; ``patch_ids`` takes exactly the values
        ``0 .. num_blocks - 1``.
    detector_times : ndarray, shape ``(D,)``, float64
        Time coordinate of each detector.

    Returns
    -------
    LogicalOperationLayout

    Raises
    ------
    ValueError
        If any detector is outside every window, if the windows are not
        contiguous, if a merge phase has no seam or a seam detector is out of
        range or outside a merge phase, if ``patch_ids`` does not cover
        exactly ``num_blocks`` patches, one per detector, or if a phase name
        is outside ``PHASE_TYPES``.
    """
    phase_ids = assign_phase_ids(phase_windows, detector_times)
    phase_names = tuple(window.name for window in phase_windows)
    return LogicalOperationLayout(
        phase_ids=phase_ids,
        phase_names=phase_names,
        patch_ids=_patch_ids(patch_ids, num_blocks, len(detector_times)),
        seam_mask=_seam_mask(
            _index_array(seam_detector_indices, "seam_detector_indices"),
            phase_ids,
            phase_names,
            detector_times,
        ),
    )


def _index_array(values: Sequence[int], field: str) -> NDArray[np.intp]:
    """Read a 1-D sequence of integers, refusing any other element type."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{field} must be 1-D, got shape {array.shape}")
    if array.size and array.dtype.kind not in "iu":
        raise ValueError(f"{field} must hold integers, got dtype {array.dtype}")
    return array.astype(np.intp, copy=False)


def _seam_mask(
    seam: NDArray[np.intp],
    phase_ids: NDArray[np.intp],
    phase_names: tuple[str, ...],
    detector_times: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Mark the seam detectors, each of which must lie in a merge phase."""
    num_detectors = len(phase_ids)
    if MERGE_PHASE in phase_names and not seam.size:
        raise ValueError("merge phase(s) present but seam_detector_indices is empty")
    out_of_range = seam[(seam < 0) | (seam >= num_detectors)]
    if out_of_range.size:
        raise ValueError(
            f"seam detector index {out_of_range[0]} out of range [0, {num_detectors})"
        )
    outside_merge = seam[np.asarray(phase_names)[phase_ids[seam]] != MERGE_PHASE]
    if outside_merge.size:
        first = int(outside_merge[0])
        raise ValueError(
            f"seam detector {first} at t={detector_times[first]} is not inside any "
            f"merge window"
        )
    seam_mask = np.zeros(num_detectors, dtype=np.bool_)
    seam_mask[seam] = True
    return seam_mask


def _patch_ids(
    patch_ids: Sequence[int], num_blocks: int, num_detectors: int
) -> NDArray[np.intp]:
    """Read one patch id per detector, covering exactly ``0 .. num_blocks - 1``.

    The phased builder indexes patch centres by id, so a gap or a negative id
    would silently pick another patch's centre.
    """
    patches = _index_array(patch_ids, "patch_ids")
    if len(patches) != num_detectors:
        raise ValueError(
            f"patch_ids length ({len(patches)}) != num_detectors ({num_detectors})"
        )
    distinct = np.unique(patches)
    if not np.array_equal(distinct, np.arange(num_blocks)):
        raise ValueError(
            f"patch_ids has {distinct.size} distinct patches {distinct.tolist()}, "
            f"expected exactly 0..{num_blocks - 1}"
        )
    return patches


def read_circuit_metadata(circuit_path: Path) -> CircuitMetadata:
    """Metadata of a committed circuit with no logical-operation layout.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit with a manifest beside it.

    Returns
    -------
    CircuitMetadata
        Built at the distance and rounds the manifest declares; ``layout`` is
        ``None``.

    Raises
    ------
    FileNotFoundError
        If the circuit carries no manifest.
    ValueError
        If the manifest is unreadable or declares an invalid key.
    """
    return _extract_at_key(circuit_path, circuit_key(circuit_path))


def read_logical_op_metadata(circuit_path: Path) -> CircuitMetadata:
    """Metadata of a committed logical-operation circuit, layout included.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit with a manifest beside it.

    Returns
    -------
    CircuitMetadata
        Built at the distance and rounds the manifest declares, carrying the
        layout the manifest declares.

    Raises
    ------
    FileNotFoundError
        If the circuit carries no manifest.
    ValueError
        If the manifest is unreadable, declares an invalid key, lacks a layout
        field, or declares a layout its circuit's detectors contradict; the
        message names the key and the manifest.
    """
    manifest_path = circuit_manifest_path(circuit_path)
    manifest = read_circuit_manifest(circuit_path)
    key = manifest_key(manifest, manifest_path)
    metadata = _extract_at_key(circuit_path, key)
    missing = [field for field in _LAYOUT_FIELDS if field not in manifest]
    if missing:
        raise ValueError(
            f"{key}: manifest {manifest_path} declares no {missing}; a "
            f"logical-operation circuit declares its layout in its manifest"
        )
    try:
        layout = _layout_from_manifest(manifest, metadata.detector_coords[:, 2])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{key}: manifest {manifest_path} declares an invalid layout: {exc}"
        ) from exc
    return replace(metadata, layout=layout)


def _layout_from_manifest(
    manifest: Mapping[str, Any], detector_times: NDArray[np.float64]
) -> LogicalOperationLayout:
    """Build the layout a manifest's layout fields declare."""
    return build_layout(
        [
            PhaseWindow(
                name=str(window["name"]),
                t_start=float(window["t_start"]),
                t_end=float(window["t_end"]),
            )
            for window in manifest["phase_windows"]
        ],
        seam_detector_indices=manifest["seam_detector_indices"],
        patch_ids=manifest["patch_ids"],
        num_blocks=int(manifest["num_blocks"]),
        detector_times=detector_times,
    )


def _extract_at_key(circuit_path: Path, key: ExperimentKey) -> CircuitMetadata:
    """Load a circuit and extract its metadata at the key's distance and rounds."""
    circuit = stim.Circuit.from_file(str(circuit_path))
    return extract_circuit_metadata(circuit, key.distance, key.rounds)
