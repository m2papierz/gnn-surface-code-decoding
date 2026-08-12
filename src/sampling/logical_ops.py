"""Logical-operation circuit source: tqec BlockGraph to committed artifact.

Emits a ``stim.Circuit`` from a tqec ``BlockGraph``, derives phase/seam/patch
metadata from the compiled circuit, validates every invariant, and writes the
circuit plus its manifest.  tqec is imported here and only here: nothing else
under ``src/`` may import it, and a test enforces this.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import stim
import tqec
from tqec import BlockGraph, NoiseModel
from tqec import compile_block_graph as _tqec_compile
from tqec.utils.noise_model import NoiseRule
from tqec.utils.scale import LinearFunction

from sampling.experiment import ExperimentKey, write_circuit_manifest


if TYPE_CHECKING:
    from sampling.graph import CircuitMetadata


__all__ = [
    "CircuitValidationError",
    "LogicalOperationMetadata",
    "PhaseWindow",
    "compile_block_graph_circuit",
    "derive_circuit_metadata",
    "extract_logical_op_circuit_metadata",
    "generate_and_write_circuit",
    "validate_logical_op_circuit",
]

logger = logging.getLogger(__name__)

_DEFAULT_TEMPORAL_HEIGHT: Final = LinearFunction(2, -1)  # 2k − 1

# tqec applies idle depolarization every tick a qubit is idle; stim applies it
# once per round on data qubits only.  This factor equalises the total DEM
# error budget across d ∈ {3, 5, 7}.
_IDLE_DEPOLARIZATION_FACTOR: Final[float] = 0.26

_MIN_COORD_DIMS: Final[int] = 3


class CircuitValidationError(Exception):
    """A logical-operation circuit failed the validation gate."""


@dataclass(frozen=True, slots=True)
class PhaseWindow:
    """A named contiguous window over detector time.

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
        if self.t_end <= self.t_start:
            raise ValueError(
                f"PhaseWindow {self.name!r}: t_end ({self.t_end}) must be > "
                f"t_start ({self.t_start})"
            )

    def contains_time(self, t: float) -> bool:
        """Return whether *t* falls inside the window."""
        return self.t_start <= t < self.t_end


@dataclass(frozen=True, slots=True)
class LogicalOperationMetadata:
    """Typed metadata record for a logical-operation circuit.

    Carries the phase/patch axis and the observable set, owned by the
    circuit source and consumed by the graph builder.  Derived once at
    extraction time and never re-derived inside a worker.

    Parameters
    ----------
    phase_windows : tuple of PhaseWindow
        Contiguous, ordered windows over detector time.
    seam_detector_indices : tuple of int
        Detector indices on the merge boundary (sorted).
    patch_ids : tuple of int
        Per-detector patch (code-block) assignment.
    num_blocks : int
        Expected number of distinct code blocks.
    observables : tuple of str
        Ordered observable names.
    """

    phase_windows: tuple[PhaseWindow, ...]
    seam_detector_indices: tuple[int, ...]
    patch_ids: tuple[int, ...]
    num_blocks: int
    observables: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.phase_windows:
            raise ValueError("phase_windows must be non-empty")
        for i in range(len(self.phase_windows) - 1):
            if (
                abs(self.phase_windows[i].t_end - self.phase_windows[i + 1].t_start)
                > 1e-9
            ):
                raise ValueError(
                    f"phase_windows are not contiguous: window {i} ends at "
                    f"{self.phase_windows[i].t_end}, window {i + 1} starts at "
                    f"{self.phase_windows[i + 1].t_start}"
                )
        if self.num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {self.num_blocks}")
        if not self.observables:
            raise ValueError("observables must be non-empty")
        actual_blocks = len(set(self.patch_ids)) if self.patch_ids else 0
        if actual_blocks != self.num_blocks:
            raise ValueError(
                f"patch_ids has {actual_blocks} distinct values, "
                f"expected num_blocks={self.num_blocks}"
            )
        for idx in self.seam_detector_indices:
            if idx < 0:
                raise ValueError(f"seam_detector_indices contains negative index {idx}")

    def validate_against_coords(self, detector_coords: np.ndarray) -> None:
        """Re-assert coordinate-dependent gate invariants.

        Parameters
        ----------
        detector_coords : ndarray, shape ``(D, >=3)``
            Detector coordinates with at least ``(x, y, t)``.

        Raises
        ------
        ValueError
            If any invariant is violated.
        """
        n_det = detector_coords.shape[0]
        if len(self.patch_ids) != n_det:
            raise ValueError(
                f"patch_ids length ({len(self.patch_ids)}) != num_detectors ({n_det})"
            )
        for det_idx in range(n_det):
            t = detector_coords[det_idx, 2]
            if not any(pw.contains_time(t) for pw in self.phase_windows):
                raise ValueError(
                    f"detector {det_idx} at t={t} is not covered by any phase window"
                )
        merge_windows = [pw for pw in self.phase_windows if pw.name == "merge"]
        has_merge = bool(merge_windows)
        if has_merge and not self.seam_detector_indices:
            raise ValueError(
                "merge phase(s) present but seam_detector_indices is empty"
            )
        for idx in self.seam_detector_indices:
            if idx >= n_det:
                raise ValueError(f"seam detector index {idx} out of range [0, {n_det})")
            t = detector_coords[idx, 2]
            if not any(mw.contains_time(t) for mw in merge_windows):
                raise ValueError(
                    f"seam detector {idx} at t={t} is not inside any merge window"
                )

    def to_circuit_metadata_fields(self, detector_coords: np.ndarray) -> dict[str, Any]:
        """Convert to the arrays ``CircuitMetadata`` carries.

        Parameters
        ----------
        detector_coords : ndarray, shape ``(D, >=3)``

        Returns
        -------
        dict
            Keys: ``phase_ids``, ``phase_names``, ``patch_ids``, ``seam_mask``.
        """
        n_det = detector_coords.shape[0]

        phase_ids = np.empty(n_det, dtype=np.intp)
        for det_idx in range(n_det):
            t = detector_coords[det_idx, 2]
            for pw_idx, pw in enumerate(self.phase_windows):
                if pw.contains_time(t):
                    phase_ids[det_idx] = pw_idx
                    break

        phase_names = tuple(pw.name for pw in self.phase_windows)

        seam_set = frozenset(self.seam_detector_indices)
        seam_mask = np.array([i in seam_set for i in range(n_det)], dtype=np.bool_)

        patch_id_arr = np.array(self.patch_ids, dtype=np.intp)

        return {
            "phase_ids": phase_ids,
            "phase_names": phase_names,
            "patch_ids": patch_id_arr,
            "seam_mask": seam_mask,
        }

    @classmethod
    def from_manifest(
        cls,
        manifest: dict[str, Any],
        *,
        observables: tuple[str, ...],
        detector_coords: np.ndarray,
    ) -> LogicalOperationMetadata:
        """Build from a circuit manifest, re-asserting gate invariants.

        Parameters
        ----------
        manifest : dict
            Circuit manifest as read from JSON.
        observables : tuple of str
            Ordered observable names (from the operation profile).
        detector_coords : ndarray, shape ``(D, >=3)``
            Detector coordinates for coordinate-dependent validation.

        Returns
        -------
        LogicalOperationMetadata

        Raises
        ------
        ValueError
            If the manifest is missing required fields or any gate invariant
            is violated.
        KeyError
            If a required field is absent from the manifest.
        """
        raw_windows = manifest["phase_windows"]
        phase_windows = tuple(
            PhaseWindow(name=pw["name"], t_start=pw["t_start"], t_end=pw["t_end"])
            for pw in raw_windows
        )
        seam_detector_indices = tuple(manifest["seam_detector_indices"])
        patch_ids = tuple(manifest["patch_ids"])
        num_blocks = int(manifest["num_blocks"])

        meta = cls(
            phase_windows=phase_windows,
            seam_detector_indices=seam_detector_indices,
            patch_ids=patch_ids,
            num_blocks=num_blocks,
            observables=observables,
        )
        meta.validate_against_coords(detector_coords)

        n_obs = manifest.get("num_observables")
        if n_obs is not None and int(n_obs) != len(observables):
            raise ValueError(
                f"manifest declares {n_obs} observables but profile "
                f"provides {len(observables)} names: {observables}"
            )

        return meta


def extract_logical_op_circuit_metadata(
    circuit: stim.Circuit,
    distance: int,
    rounds: int,
    *,
    logical_op_metadata: LogicalOperationMetadata,
) -> "CircuitMetadata":
    """Build a ``CircuitMetadata`` with logical-operation fields populated.

    Parameters
    ----------
    circuit : stim.Circuit
        Compiled circuit.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    logical_op_metadata : LogicalOperationMetadata
        Pre-validated metadata record.

    Returns
    -------
    CircuitMetadata
    """
    from sampling.graph import CircuitMetadata, extract_circuit_metadata

    base = extract_circuit_metadata(circuit, distance, rounds)
    logical_op_metadata.validate_against_coords(base.detector_coords)
    fields = logical_op_metadata.to_circuit_metadata_fields(base.detector_coords)

    return CircuitMetadata(
        detector_coords=base.detector_coords,
        distance=base.distance,
        rounds=base.rounds,
        num_detectors=base.num_detectors,
        dem_edge_weights=base.dem_edge_weights,
        phase_ids=fields["phase_ids"],
        phase_names=fields["phase_names"],
        patch_ids=fields["patch_ids"],
        seam_mask=fields["seam_mask"],
    )


def _stim_compatible_noise(error_prob: float) -> NoiseModel:
    """Build a noise model matching stim's 4-channel uniform convention.

    stim's ``Circuit.generated`` applies four noise channels at level *p*:
    ``after_clifford_depolarization``, ``after_reset_flip_probability``,
    ``before_measure_flip_probability``, and
    ``before_round_data_depolarization``.

    tqec uses a different gate decomposition (CZ instead of H + CX) and a
    finer-grained idle model (per-tick on every idle qubit rather than
    per-round on data qubits only).  ``_IDLE_DEPOLARIZATION_FACTOR`` scales
    the per-tick idle rate so that the total detector-error-model weight
    matches the stim convention across distances.

    Parameters
    ----------
    error_prob : float
        Physical error probability *p*.

    Returns
    -------
    NoiseModel
    """
    p = error_prob
    no_noise = NoiseRule(after={}, flip_result=0)
    return NoiseModel(
        idle_depolarization=_IDLE_DEPOLARIZATION_FACTOR * p,
        gate_rules={
            "RX": NoiseRule(after={"Z_ERROR": p}),
            "RY": NoiseRule(after={"X_ERROR": p}),
            "R": NoiseRule(after={"X_ERROR": p}),
            "MR": no_noise,
        },
        measure_rules={
            "X": NoiseRule(after={}, flip_result=p),
            "Y": NoiseRule(after={}, flip_result=p),
            "Z": NoiseRule(after={}, flip_result=p),
            "XX": NoiseRule(after={}, flip_result=p),
            "YY": NoiseRule(after={}, flip_result=p),
            "ZZ": NoiseRule(after={}, flip_result=p),
        },
        any_clifford_1q_rule=NoiseRule(after={"DEPOLARIZE1": p}),
        any_clifford_2q_rule=NoiseRule(after={"DEPOLARIZE2": p}),
    )


def _align_detector_times(circuit: stim.Circuit) -> stim.Circuit:
    """Shift final boundary detectors to their own time step.

    tqec merges final-measurement boundary detectors into the last syndrome
    round (same ``t`` coordinate), producing duplicate ``(x, y)`` positions.
    stim places them at ``t_max + 1``.  This function separates them so the
    temporal structure matches stim's convention.

    Parameters
    ----------
    circuit : stim.Circuit

    Returns
    -------
    stim.Circuit
        Circuit with boundary detectors shifted to ``t_max + 1``.  Returned
        unchanged when no duplicate boundary positions are detected.
    """
    coords = circuit.get_detector_coordinates()
    if not coords:
        return circuit

    by_t: dict[float, list[tuple[int, tuple[float, float]]]] = defaultdict(list)
    for idx, c in coords.items():
        by_t[c[2]].append((idx, (c[0], c[1])))

    max_t = max(by_t)
    t0_positions = {pos for _, pos in by_t[0.0]}

    from collections import Counter

    max_t_counts = Counter(pos for _, pos in by_t[max_t])
    dup_positions = {pos for pos, cnt in max_t_counts.items() if cnt > 1}

    if not dup_positions or dup_positions != t0_positions:
        return circuit

    shift_set: set[int] = set()
    pos_groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for idx, pos in by_t[max_t]:
        if pos in dup_positions:
            pos_groups[pos].append(idx)

    for pos, indices in pos_groups.items():
        if len(indices) != 2:
            raise ValueError(
                f"Expected exactly 2 detectors at boundary position {pos} "
                f"at t={max_t}, got {len(indices)}"
            )
        shift_set.add(max(indices))

    new_t = max_t + 1
    new_circuit = stim.Circuit()
    det_idx = 0
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            if det_idx in shift_set:
                args = list(inst.gate_args_copy())
                args[2] = new_t
                new_circuit.append("DETECTOR", inst.targets_copy(), args)
            else:
                new_circuit.append(inst)
            det_idx += 1
        else:
            new_circuit.append(inst)

    return new_circuit


def compile_block_graph_circuit(
    block_graph: BlockGraph,
    *,
    k: int,
    error_prob: float,
    block_temporal_height: LinearFunction = _DEFAULT_TEMPORAL_HEIGHT,
) -> stim.Circuit:
    """Compile a tqec ``BlockGraph`` into a ``stim.Circuit`` with noise.

    Parameters
    ----------
    block_graph : BlockGraph
        The logical-operation block graph.
    k : int
        Scale parameter.  Code distance is ``2k + 1``.
    error_prob : float
        Physical error probability for the uniform depolarizing model.
    block_temporal_height : LinearFunction
        Rounds of stabilizer measurement per temporal block (default ``2k−1``).

    Returns
    -------
    stim.Circuit
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not (0 < error_prob < 1):
        raise ValueError(f"error_prob must be in (0, 1), got {error_prob}")

    tcg = _tqec_compile(
        block_graph,
        block_temporal_height=block_temporal_height,
    )
    noise = _stim_compatible_noise(error_prob)
    return tcg.generate_stim_circuit(k=k, noise_model=noise)


def _detector_coords_array(circuit: stim.Circuit) -> np.ndarray:
    """Return detector coordinates as a ``(num_detectors, D)`` float64 array."""
    raw = circuit.get_detector_coordinates()
    n = circuit.num_detectors
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    ndim = len(raw[0])
    arr = np.empty((n, ndim), dtype=np.float64)
    for i in range(n):
        arr[i] = raw[i]
    return arr


def _z_levels(block_graph: BlockGraph) -> list[int]:
    """Sorted unique z-positions of cubes in the block graph."""
    return sorted({cube.position.z for cube in block_graph.cubes})


def _cubes_at_z(block_graph: BlockGraph, z: int) -> list[tuple[int, int]]:
    """Return the (x, y) positions of cubes at a given z-level."""
    return sorted(
        (cube.position.x, cube.position.y)
        for cube in block_graph.cubes
        if cube.position.z == z
    )


def _has_spatial_pipes_at_z(block_graph: BlockGraph, z: int) -> bool:
    """Check whether any spatial (X/Y) pipe exists at z-level *z*."""
    cubes_at = set(_cubes_at_z(block_graph, z))
    for pipe in block_graph.pipes:
        if pipe.direction.name in ("X", "Y"):
            u = pipe.u
            v = pipe.v
            u_xy = (u.position.x, u.position.y)
            v_xy = (v.position.x, v.position.y)
            u_z = u.position.z
            v_z = v.position.z
            if (u_z == z and u_xy in cubes_at) or (v_z == z and v_xy in cubes_at):
                return True
    return False


def derive_circuit_metadata(
    block_graph: BlockGraph,
    circuit: stim.Circuit,
    *,
    k: int,
    block_temporal_height: LinearFunction = _DEFAULT_TEMPORAL_HEIGHT,
) -> dict[str, Any]:
    """Derive phase windows, seam detectors, and patch IDs.

    Parameters
    ----------
    block_graph : BlockGraph
        Source block graph.
    circuit : stim.Circuit
        Compiled circuit.
    k : int
        Scale parameter.
    block_temporal_height : LinearFunction
        Temporal height used during compilation.

    Returns
    -------
    dict
        Keys: ``phase_windows``, ``seam_detector_indices``, ``patch_ids``,
        ``num_blocks``.
    """
    coords = _detector_coords_array(circuit)
    n_det = circuit.num_detectors
    z_levels = _z_levels(block_graph)
    n_z = len(z_levels)

    if n_det == 0 or n_z == 0:
        return {
            "phase_windows": [],
            "seam_detector_indices": [],
            "patch_ids": [],
            "num_blocks": 0,
        }

    max_t = float(coords[:, 2].max())
    stride = (max_t + 1.0) / n_z

    phase_windows: list[PhaseWindow] = []
    for i, z in enumerate(z_levels):
        t_start = i * stride
        t_end = (i + 1) * stride
        has_merge = _has_spatial_pipes_at_z(block_graph, z)
        name = "merge" if has_merge else "memory"
        phase_windows.append(PhaseWindow(name=name, t_start=t_start, t_end=t_end))

    unique_xy = sorted({(c.position.x, c.position.y) for c in block_graph.cubes})
    num_blocks = len(unique_xy)
    xy_to_patch: dict[tuple[int, int], int] = {
        xy: idx for idx, xy in enumerate(unique_xy)
    }

    cube_centers = _compute_all_cube_centers(
        coords,
        block_graph,
        z_levels,
        stride,
    )

    patch_ids = _assign_patch_ids(
        coords,
        block_graph,
        cube_centers,
        xy_to_patch,
        z_levels,
        stride,
    )

    memory_xy_positions: set[tuple[float, float]] = set()
    for i, z in enumerate(z_levels):
        if not _has_spatial_pipes_at_z(block_graph, z):
            t_lo = i * stride
            t_hi = (i + 1) * stride
            mask = (coords[:, 2] >= t_lo - 0.5) & (coords[:, 2] < t_hi - 0.5)
            for row in coords[mask]:
                memory_xy_positions.add((round(row[0], 6), round(row[1], 6)))

    seam_detector_indices: list[int] = []
    for i, z in enumerate(z_levels):
        if _has_spatial_pipes_at_z(block_graph, z):
            t_lo = i * stride
            t_hi = (i + 1) * stride
            for det_idx in range(n_det):
                t = coords[det_idx, 2]
                if t_lo <= t < t_hi:
                    xy = (round(coords[det_idx, 0], 6), round(coords[det_idx, 1], 6))
                    if xy not in memory_xy_positions:
                        seam_detector_indices.append(det_idx)

    return {
        "phase_windows": phase_windows,
        "seam_detector_indices": sorted(seam_detector_indices),
        "patch_ids": patch_ids,
        "num_blocks": num_blocks,
    }


def _compute_all_cube_centers(
    coords: np.ndarray,
    block_graph: BlockGraph,
    z_levels: list[int],
    stride: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute detector-space centroids for every unique (x, y) cube position.

    Cubes that only exist at merge z-levels (ancilla blocks) are handled by
    computing centers from the first z-level where they appear, after
    subtracting the detectors that belong to already-known patches.
    """
    all_xy = sorted({(c.position.x, c.position.y) for c in block_graph.cubes})

    if len(all_xy) == 1:
        centroid = coords[:, :2].mean(axis=0)
        return {all_xy[0]: centroid}

    centers: dict[tuple[int, int], np.ndarray] = {}
    remaining_xy = set(all_xy)

    for z_idx, z in enumerate(z_levels):
        if not remaining_xy:
            break
        cubes_here = set(_cubes_at_z(block_graph, z))
        new_cubes = cubes_here & remaining_xy
        if not new_cubes:
            continue

        t_lo = z_idx * stride
        t_hi = (z_idx + 1) * stride
        mask = (coords[:, 2] >= t_lo - 0.5) & (coords[:, 2] < t_hi - 0.5)
        slice_coords = coords[mask, :2]

        if not centers:
            _seed_centers_from_slice(centers, slice_coords, list(cubes_here))
        else:
            known_in_slice = cubes_here - new_cubes
            if known_in_slice:
                assigned = np.zeros(len(slice_coords), dtype=bool)
                for pt_idx, pt in enumerate(slice_coords):
                    best = min(
                        known_in_slice,
                        key=lambda k: float(np.sum((pt - centers[k]) ** 2)),
                    )
                    if float(np.sum((pt - centers[best]) ** 2)) < (stride * 4) ** 2:
                        assigned[pt_idx] = True
                unassigned = slice_coords[~assigned]
            else:
                unassigned = slice_coords

            if len(unassigned) > 0 and len(new_cubes) == 1:
                nc = next(iter(new_cubes))
                centers[nc] = unassigned.mean(axis=0)
            elif len(unassigned) > 0 and len(new_cubes) > 1:
                _seed_centers_from_slice(centers, unassigned, list(new_cubes))

        remaining_xy -= new_cubes

    return centers


def _seed_centers_from_slice(
    centers: dict[tuple[int, int], np.ndarray],
    slice_coords: np.ndarray,
    cubes_xy: list[tuple[int, int]],
) -> None:
    """Seed cube centers from a coordinate slice using iterative refinement."""
    cube_arr = np.array(cubes_xy, dtype=np.float64)
    cube_min = cube_arr.min(axis=0)
    cube_max = cube_arr.max(axis=0)
    n_per_axis = cube_max - cube_min + 1.0

    det_min = slice_coords.min(axis=0)
    det_range = slice_coords.max(axis=0) - det_min
    stride_per_cube = det_range / np.where(n_per_axis > 1, n_per_axis, 1.0)

    for cx, cy in cubes_xy:
        centers[(cx, cy)] = det_min + np.array(
            [
                (cx - cube_min[0] + 0.5) * stride_per_cube[0],
                (cy - cube_min[1] + 0.5) * stride_per_cube[1],
            ]
        )

    all_keys = {(cx, cy) for cx, cy in cubes_xy}
    for _ in range(5):
        groups: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
        for pt in slice_coords:
            best_key = min(
                all_keys, key=lambda kk: float(np.sum((pt - centers[kk]) ** 2))
            )
            groups[best_key].append(pt)
        for key in all_keys:
            if groups.get(key):
                centers[key] = np.mean(groups[key], axis=0)


def _assign_patch_ids(
    coords: np.ndarray,
    block_graph: BlockGraph,
    cube_centers: dict[tuple[int, int], np.ndarray],
    xy_to_patch: dict[tuple[int, int], int],
    z_levels: list[int],
    stride: float,
) -> list[int]:
    """Assign a patch ID to every detector."""
    n_det = coords.shape[0]
    patch_ids = [0] * n_det

    for det_idx in range(n_det):
        t = coords[det_idx, 2]
        z_idx = min(int(t / stride), len(z_levels) - 1)
        z = z_levels[z_idx]
        cubes_xy = _cubes_at_z(block_graph, z)

        available_centers = {
            xy: cube_centers.get(xy, np.array([0.0, 0.0])) for xy in cubes_xy
        }
        if not available_centers:
            available_centers = cube_centers

        pt = coords[det_idx, :2]
        best_xy = min(
            available_centers, key=lambda k: np.sum((pt - available_centers[k]) ** 2)
        )
        patch_ids[det_idx] = xy_to_patch[best_xy]

    return patch_ids


def _gate_fail(
    operation: str,
    distance: int,
    error_prob: float,
    invariant: str,
) -> CircuitValidationError:
    return CircuitValidationError(
        f"{operation} (d={distance}, p={error_prob}): {invariant}"
    )


def validate_logical_op_circuit(
    circuit: stim.Circuit,
    *,
    operation: str,
    distance: int,
    error_prob: float,
    phase_windows: list[PhaseWindow],
    seam_detector_indices: list[int],
    patch_ids: list[int],
    num_blocks: int,
    expected_observable_count: int,
) -> dict[str, Any]:
    """Run the validation gate on a compiled logical-operation circuit.

    Returns a provenance dict on success.  Raises
    :class:`CircuitValidationError` on any violation.

    Parameters
    ----------
    circuit : stim.Circuit
        The compiled circuit.
    operation : str
        Logical operation name.
    distance : int
        Code distance (``2k + 1``).
    error_prob : float
        Physical error probability.
    phase_windows : list[PhaseWindow]
        Expected phase windows that must partition the detector timeline.
    seam_detector_indices : list[int]
        Indices of detectors on the merge boundary.
    patch_ids : list[int]
        Patch assignment per detector.
    num_blocks : int
        Expected number of distinct code blocks.
    expected_observable_count : int
        Number of correlation surfaces expected.

    Returns
    -------
    dict
        Provenance fields for the manifest.
    """
    n_det = circuit.num_detectors

    def fail(msg: str) -> CircuitValidationError:
        return _gate_fail(operation, distance, error_prob, msg)

    # 1. DEM decomposes
    try:
        dem = circuit.detector_error_model(decompose_errors=True)
    except Exception as exc:
        raise fail(
            f"detector_error_model(decompose_errors=True) raised: {exc}"
        ) from exc

    # Record whether tqec pre-decomposed
    try:
        dem_raw = circuit.detector_error_model(decompose_errors=False)
        tqec_predecomposes = str(dem_raw) == str(dem)
    except Exception:
        tqec_predecomposes = False

    # 2. Coordinates are 3-D or wider
    coords = _detector_coords_array(circuit)
    if n_det > 0:
        if coords.shape[1] < _MIN_COORD_DIMS:
            raise fail(
                f"detector coordinates have {coords.shape[1]} dimensions, "
                f"need >= {_MIN_COORD_DIMS}"
            )

    # 3. Phase windows partition the detector timeline
    if not phase_windows:
        raise fail("no phase windows declared")

    for i in range(len(phase_windows) - 1):
        if abs(phase_windows[i].t_end - phase_windows[i + 1].t_start) > 1e-9:
            raise fail(
                f"phase windows are not contiguous: window {i} ends at "
                f"{phase_windows[i].t_end}, window {i + 1} starts at "
                f"{phase_windows[i + 1].t_start}"
            )

    for det_idx in range(n_det):
        t = coords[det_idx, 2]
        covered = any(pw.contains_time(t) for pw in phase_windows)
        if not covered:
            win_repr = [(pw.name, pw.t_start, pw.t_end) for pw in phase_windows]
            raise fail(
                f"detector {det_idx} at t={t} is not covered "
                f"by any phase window (windows: {win_repr})"
            )

    # 4. Seam detectors
    has_merge = any(pw.name == "merge" for pw in phase_windows)
    if has_merge and not seam_detector_indices:
        raise fail("operation has merge phase(s) but seam_detector_indices is empty")

    merge_windows = [pw for pw in phase_windows if pw.name == "merge"]
    for idx in seam_detector_indices:
        if idx < 0 or idx >= n_det:
            raise fail(f"seam detector index {idx} out of range [0, {n_det})")
        t = coords[idx, 2]
        in_merge = any(mw.contains_time(t) for mw in merge_windows)
        if not in_merge:
            raise fail(f"seam detector {idx} at t={t} is not inside any merge window")

    # 5. Patch IDs
    if len(patch_ids) != n_det:
        raise fail(f"patch_ids length ({len(patch_ids)}) != num_detectors ({n_det})")
    actual_blocks = len(set(patch_ids)) if patch_ids else 0
    if actual_blocks != num_blocks:
        raise fail(
            f"patch_ids has {actual_blocks} distinct patches, expected {num_blocks}"
        )

    # 6. Observable count
    if circuit.num_observables != expected_observable_count:
        raise fail(
            f"circuit has {circuit.num_observables} observables, "
            f"expected {expected_observable_count}"
        )

    # 7. Each observable is non-trivial
    observable_has_error = [False] * circuit.num_observables
    for instruction in dem.flattened():
        if instruction.type == "error":
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    pass
                elif target.is_logical_observable_id():
                    obs_idx = target.val
                    if 0 <= obs_idx < circuit.num_observables:
                        observable_has_error[obs_idx] = True
    for obs_idx, has in enumerate(observable_has_error):
        if not has:
            raise fail(f"observable {obs_idx} is trivial (no error mechanism flips it)")

    return {
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "tqec_predecomposes": tqec_predecomposes,
    }


def generate_and_write_circuit(
    block_graph: BlockGraph,
    *,
    operation: str,
    k: int,
    error_prob: float,
    output_dir: Path,
    block_temporal_height: LinearFunction = _DEFAULT_TEMPORAL_HEIGHT,
    generating_command: str = "",
) -> Path:
    """Compile, derive metadata, validate, and write a circuit + manifest.

    Parameters
    ----------
    block_graph : BlockGraph
        Source block graph.
    operation : str
        Logical operation name (e.g. ``"memory"``, ``"zz_merge_split"``).
    k : int
        Scale parameter (distance = ``2k + 1``).
    error_prob : float
        Physical error probability.
    output_dir : Path
        Directory to write the ``.stim`` and ``.manifest.json`` files.
    block_temporal_height : LinearFunction
        Rounds per temporal block (default ``2k − 1``).
    generating_command : str
        Shell command that reproduces this generation.

    Returns
    -------
    Path
        Path to the written ``.stim`` file.
    """
    distance = 2 * k + 1
    correlation_surfaces = block_graph.find_correlation_surfaces()
    expected_obs = len(correlation_surfaces)

    circuit = compile_block_graph_circuit(
        block_graph,
        k=k,
        error_prob=error_prob,
        block_temporal_height=block_temporal_height,
    )
    circuit = _align_detector_times(circuit)

    metadata = derive_circuit_metadata(
        block_graph,
        circuit,
        k=k,
        block_temporal_height=block_temporal_height,
    )

    provenance = validate_logical_op_circuit(
        circuit,
        operation=operation,
        distance=distance,
        error_prob=error_prob,
        phase_windows=metadata["phase_windows"],
        seam_detector_indices=metadata["seam_detector_indices"],
        patch_ids=metadata["patch_ids"],
        num_blocks=metadata["num_blocks"],
        expected_observable_count=expected_obs,
    )

    rounds_per_block = block_temporal_height(k) + 2
    n_z = len(_z_levels(block_graph))
    total_rounds = n_z * rounds_per_block

    p_tag = f"p{error_prob:.6g}".replace(".", "_")
    filename = f"d{distance}_r{total_rounds}_{p_tag}.stim"
    circuit_path = output_dir / filename

    output_dir.mkdir(parents=True, exist_ok=True)
    circuit.to_file(str(circuit_path))

    circuit_sha256 = hashlib.sha256(circuit_path.read_bytes()).hexdigest()

    key = ExperimentKey(
        operation=operation,
        distance=distance,
        rounds=total_rounds,
        error_prob=error_prob,
    )
    manifest_provenance: dict[str, Any] = {
        "tqec_version": tqec.__version__,
        "stim_version": stim.__version__,
        "k": k,
        "circuit_file": str(circuit_path),
        "circuit_sha256": circuit_sha256,
        "num_detectors": provenance["num_detectors"],
        "num_observables": provenance["num_observables"],
        "tqec_predecomposes": provenance["tqec_predecomposes"],
        "phase_windows": [
            {"name": pw.name, "t_start": pw.t_start, "t_end": pw.t_end}
            for pw in metadata["phase_windows"]
        ],
        "seam_detector_indices": metadata["seam_detector_indices"],
        "patch_ids": metadata["patch_ids"],
        "num_blocks": metadata["num_blocks"],
        "generation_command": generating_command,
    }
    write_circuit_manifest(circuit_path, key, provenance=manifest_provenance)

    logger.info(
        "Wrote %s  (%s, detectors=%d, observables=%d)",
        circuit_path.name,
        key,
        provenance["num_detectors"],
        provenance["num_observables"],
    )
    return circuit_path
