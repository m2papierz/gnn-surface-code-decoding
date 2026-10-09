"""Detector graph construction for GNN-based QEC decoding.

``FiredDetectorGraph`` / ``build_fired_detector_graph``: fired detectors
only, complete graph with learned features. Primary representation for
GNN training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import stim
from numpy.typing import NDArray

from sampling.dem import DemMechanisms, read_dem_mechanisms


NODE_DIM: int = 6
"""Node feature dimensionality: x_norm, y_norm, t_norm, d_x, d_y, basis."""

EDGE_DIM: int = 6
"""Edge feature dimensionality: dx, dy, dt, euclidean, chebyshev, dem_weight."""

COORDINATE_DIM: Final[int] = 3
"""Coordinates every detector declares: ``(x, y, t)``."""


def complete_graph_edge_count[CountT: (int, NDArray[np.int64])](
    num_fired: CountT,
) -> CountT:
    """Directed edges of the complete graph over the fired detectors.

    The one definition of the graph's size law: ``FiredDetectorGraph``
    validates against it, and anything that sizes a graph without building
    it calls it.

    Parameters
    ----------
    num_fired : int or ndarray of int64
        Fired-detector count of one shot, or of each shot in a batch.

    Returns
    -------
    int or ndarray of int64
        ``N * (N - 1)``, in the form the count was given.
    """
    return num_fired * (num_fired - 1)


MERGE_PHASE: Final[str] = "merge"
"""Phase during which the patches are merged; its detectors carry the seam."""

PHASE_TYPES: tuple[str, ...] = ("memory", MERGE_PHASE, "split")
"""Fixed vocabulary of phase types for the one-hot encoding."""


@dataclass(frozen=True, slots=True)
class LogicalOperationLayout:
    """Per-detector structure of a logical-operation circuit.

    Parameters
    ----------
    phase_ids : ndarray, shape ``(D,)``, intp
        Phase-window index of each detector.
    phase_names : tuple of str
        Ordered phase-window names, each in ``PHASE_TYPES``;
        ``phase_names[phase_ids[i]]`` is the phase of detector *i*.
    patch_ids : ndarray, shape ``(D,)``, intp
        Patch (code-block) index of each detector.
    seam_mask : ndarray, shape ``(D,)``, bool
        ``True`` for detectors on the merge boundary.
    """

    phase_ids: NDArray[np.intp]
    phase_names: tuple[str, ...]
    patch_ids: NDArray[np.intp]
    seam_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.phase_ids.ndim != 1:
            raise ValueError(f"phase_ids must be 1-D, got shape {self.phase_ids.shape}")
        if not self.phase_names:
            raise ValueError("phase_names must be non-empty")
        # A name outside the vocabulary would encode as an all-zero one-hot.
        unknown = sorted(set(self.phase_names) - set(PHASE_TYPES))
        if unknown:
            raise ValueError(
                f"phase_names {unknown} are outside the phase vocabulary {PHASE_TYPES}"
            )
        n_phases = len(self.phase_names)
        if self.phase_ids.min() < 0 or self.phase_ids.max() >= n_phases:
            raise ValueError(
                f"phase_ids values must be in [0, {n_phases}), "
                f"got [{self.phase_ids.min()}, {self.phase_ids.max()}]"
            )
        for name, values in (
            ("patch_ids", self.patch_ids),
            ("seam_mask", self.seam_mask),
        ):
            if values.shape != self.phase_ids.shape:
                raise ValueError(
                    f"{name} shape {values.shape} != phase_ids shape "
                    f"{self.phase_ids.shape}"
                )

    @property
    def num_detectors(self) -> int:
        """Detectors the layout covers."""
        return self.phase_ids.shape[0]


@dataclass(frozen=True, slots=True)
class CircuitMetadata:
    """Precomputed per-circuit metadata for the graph builder.

    Extracted once per circuit by its operation's ``MetadataExtractor`` and
    shared read-only across DataLoader workers.

    Parameters
    ----------
    detector_coords : ndarray, shape ``(D, 3)``, float64
        ``(x, y, t)`` coordinates for each detector.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    num_detectors : int
        Total detector count (must equal ``detector_coords.shape[0]``).
    dem_edge_weights : ndarray, shape ``(D, D)``, float64
        Pairwise detector error probabilities from the detector error
        model.  Entry ``[i, j]`` is the combined probability that
        detectors *i* and *j* are both flipped by the same error
        mechanism.  Zero for pairs with no shared error.
    layout : LogicalOperationLayout or None
        Phase, patch and seam structure of a logical-operation circuit;
        ``None`` for operations without one (memory).
    """

    detector_coords: np.ndarray
    distance: int
    rounds: int
    num_detectors: int
    dem_edge_weights: np.ndarray
    layout: LogicalOperationLayout | None = None

    def __post_init__(self) -> None:
        D = self.num_detectors
        if self.detector_coords.shape != (D, COORDINATE_DIM):
            raise ValueError(
                f"detector_coords shape {self.detector_coords.shape} != "
                f"expected ({D}, {COORDINATE_DIM})"
            )
        if self.dem_edge_weights.shape != (D, D):
            raise ValueError(
                f"dem_edge_weights shape {self.dem_edge_weights.shape} != "
                f"expected ({D}, {D})"
            )
        if self.distance < 1:
            raise ValueError(f"distance must be >= 1, got {self.distance}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")

        if self.layout is not None and self.layout.num_detectors != D:
            raise ValueError(
                f"layout phase_ids shape {self.layout.phase_ids.shape} != "
                f"expected ({D},)"
            )


class MetadataExtractor(Protocol):
    """Reads a committed circuit into the metadata its builder reads.

    Distance, rounds and any layout come from the manifest beside the circuit.
    ``num_observables`` is the label width of the contract the caller runs
    under; a circuit with any other observable count is refused, so its shots
    can never be labelled under a contract of another width.  Each operation
    names its own extractor.
    """

    def __call__(
        self, circuit_path: Path, /, *, num_observables: int
    ) -> CircuitMetadata: ...


@dataclass(frozen=True, slots=True)
class FiredDetectorGraph:
    """Fired-detector complete graph with learned features.

    Parameters
    ----------
    node_features : ndarray, shape ``(N, F_node)``, float32
        Per-node features.  Width is ``node_dim`` (default ``NODE_DIM``
        for the spatial representation).
    edge_index : ndarray, shape ``(2, E)``, int64
        Directed COO edges for the complete graph (both directions).
    edge_features : ndarray, shape ``(E, F_edge)``, float32
        Per-edge features.  Width is ``edge_dim`` (default ``EDGE_DIM``
        for the spatial representation).
    num_fired : int
        Number of fired detectors (nodes in the graph).
    fired_indices : ndarray, shape ``(N,)``, int64
        Original detector indices that fired.
    node_dim : int
        Expected node feature width (validated in ``__post_init__``).
    edge_dim : int
        Expected edge feature width (validated in ``__post_init__``).
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    num_fired: int
    fired_indices: np.ndarray
    node_dim: int = NODE_DIM
    edge_dim: int = EDGE_DIM

    def __post_init__(self) -> None:
        N = self.num_fired
        E = complete_graph_edge_count(N)

        if self.node_features.shape != (N, self.node_dim):
            raise ValueError(
                f"node_features shape {self.node_features.shape} != "
                f"expected ({N}, {self.node_dim})"
            )
        if self.edge_index.shape != (2, E):
            raise ValueError(
                f"edge_index shape {self.edge_index.shape} != expected (2, {E})"
            )
        if self.edge_features.shape != (E, self.edge_dim):
            raise ValueError(
                f"edge_features shape {self.edge_features.shape} != "
                f"expected ({E}, {self.edge_dim})"
            )
        if self.fired_indices.shape != (N,):
            raise ValueError(
                f"fired_indices shape {self.fired_indices.shape} != expected ({N},)"
            )


def dem_weight_mechanisms(mechanisms: DemMechanisms) -> NDArray[np.bool_]:
    """Select the mechanisms the ``dem_weight`` edge feature is built from.

    A mechanism contributes when it flips exactly two detectors in total.
    That admits every undecomposed two-detector mechanism and, among the
    decomposed ones, only those listed as two single-detector components.
    Single-detector mechanisms and every wider decomposition contribute
    nothing.

    Parameters
    ----------
    mechanisms : DemMechanisms
        Mechanisms of a decomposed detector error model.

    Returns
    -------
    ndarray, shape ``(M,)``, bool
    """
    return mechanisms.num_detectors == 2


def _extract_dem_weights(circuit: stim.Circuit) -> np.ndarray:
    """Extract pairwise detector error probabilities from the DEM.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit to extract the detector error model from.

    Returns
    -------
    ndarray, shape ``(D, D)``, float64
        Symmetric matrix where entry ``[i, j]`` is the combined
        probability that detectors *i* and *j* are flipped by the same
        error mechanism.  Multiple mechanisms affecting the same pair
        are combined via ``p = 1 - prod(1 - p_k)``.
    """
    mechanisms = read_dem_mechanisms(
        circuit.detector_error_model(decompose_errors=True)
    )
    selected = dem_weight_mechanisms(mechanisms)
    start = mechanisms.detector_offsets[:-1][selected]
    first = mechanisms.detectors[start]
    second = mechanisms.detectors[start + 1]
    log_survival = np.log1p(-mechanisms.probability[selected])

    num_det = circuit.num_detectors
    log_complement = np.zeros((num_det, num_det), dtype=np.float64)
    # ``np.add.at`` accumulates in mechanism order, so each cell sums its
    # terms in the order the DEM lists them.
    np.add.at(log_complement, (first, second), log_survival)
    np.add.at(log_complement, (second, first), log_survival)

    return 1.0 - np.exp(log_complement)


def read_detector_coordinates(circuit: stim.Circuit) -> NDArray[np.float64]:
    """Read the ``(x, y, t)`` every detector of a circuit declares.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit whose detectors are annotated with coordinates.

    Returns
    -------
    ndarray, shape ``(D, 3)``, float64
        Row *i* holds detector *i*'s coordinates, ``SHIFT_COORDS`` applied.

    Raises
    ------
    ValueError
        If any detector declares other than three coordinates, naming the
        first such detector and what it declares.
    """
    by_index = circuit.get_detector_coordinates()
    declared = [by_index[i] for i in range(circuit.num_detectors)]
    arity = np.fromiter(map(len, declared), dtype=np.intp, count=len(declared))
    malformed = np.flatnonzero(arity != COORDINATE_DIM)
    if malformed.size:
        first = int(malformed[0])
        raise ValueError(
            f"detector {first} declares {arity[first]} coordinate dimensions "
            f"{tuple(declared[first])}, expected {COORDINATE_DIM}: (x, y, t); "
            f"{malformed.size} of {len(declared)} detectors are malformed"
        )
    return np.array(declared, dtype=np.float64).reshape(len(declared), COORDINATE_DIM)


def extract_circuit_metadata(
    circuit: stim.Circuit,
    distance: int,
    rounds: int,
) -> CircuitMetadata:
    """Extract graph builder metadata from a Stim circuit.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit whose detectors are annotated with ``(x, y, t)``.
    distance : int
        Code distance (used for spatial normalization as ``2 * d``).
    rounds : int
        Number of syndrome measurement rounds (temporal normalization).

    Returns
    -------
    CircuitMetadata

    Raises
    ------
    ValueError
        If a detector does not declare exactly ``(x, y, t)``.
    """
    return CircuitMetadata(
        detector_coords=read_detector_coordinates(circuit),
        distance=distance,
        rounds=rounds,
        num_detectors=circuit.num_detectors,
        dem_edge_weights=_extract_dem_weights(circuit),
    )


_EMPTY_NODE_FEATURES = np.zeros((0, NODE_DIM), dtype=np.float32)
_EMPTY_EDGE_INDEX = np.zeros((2, 0), dtype=np.int64)
_EMPTY_EDGE_FEATURES = np.zeros((0, EDGE_DIM), dtype=np.float32)
_EMPTY_FIRED = np.zeros((0,), dtype=np.int64)


def build_fired_detector_graph(
    syndrome: np.ndarray,
    metadata: CircuitMetadata,
) -> FiredDetectorGraph:
    """Build a fired-detector complete graph from a syndrome vector.

    Parameters
    ----------
    syndrome : ndarray, shape ``(D,)``
        Binary syndrome bit-vector (1 = fired).
    metadata : CircuitMetadata
        Precomputed circuit metadata.

    Returns
    -------
    FiredDetectorGraph
        Complete graph over fired detectors with learned features.
        For empty syndromes (zero fired detectors), returns a graph
        with ``num_fired=0`` - the caller short-circuits to no-flip.
    """
    fired = np.flatnonzero(syndrome)
    N = len(fired)

    if N == 0:
        return FiredDetectorGraph(
            node_features=_EMPTY_NODE_FEATURES,
            edge_index=_EMPTY_EDGE_INDEX,
            edge_features=_EMPTY_EDGE_FEATURES,
            num_fired=0,
            fired_indices=_EMPTY_FIRED,
        )

    coords = metadata.detector_coords[fired]  # (N, 3)
    x = coords[:, 0]
    y = coords[:, 1]
    t = coords[:, 2]

    d = metadata.distance
    r = metadata.rounds
    spatial_scale = 2.0 * d
    temporal_scale = float(r)

    # Normalized coordinates
    x_norm = x / spatial_scale
    y_norm = y / spatial_scale
    t_norm = t / temporal_scale

    # Signed boundary distances: -1 at min boundary, +1 at max boundary
    d_x = (x - d) / d
    d_y = (y - d) / d

    # Basis flag: distinguishes X-check vs Z-check stabilizers
    basis = (((x + y) / 2).astype(np.intp) % 2).astype(np.float32)

    node_features = np.column_stack([x_norm, y_norm, t_norm, d_x, d_y, basis]).astype(
        np.float32
    )

    if N == 1:
        return FiredDetectorGraph(
            node_features=node_features,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, EDGE_DIM), dtype=np.float32),
            num_fired=1,
            fired_indices=fired.astype(np.int64),
        )

    # Complete graph edges via broadcasting (vectorized, no Python loop)
    idx = np.arange(N, dtype=np.int64)
    all_src = np.repeat(idx, N)  # (N*N,)
    all_dst = np.tile(idx, N)  # (N*N,)
    not_self = all_src != all_dst  # mask out diagonal
    src = all_src[not_self]  # (N*(N-1),)
    dst = all_dst[not_self]  # (N*(N-1),)
    edge_index = np.stack([src, dst], axis=0)

    # Edge features from normalized coordinates (vectorized)
    norm_coords = node_features[:, :3]  # (N, 3): x_norm, y_norm, t_norm
    src_coords = norm_coords[src]  # (E, 3)
    dst_coords = norm_coords[dst]  # (E, 3)
    delta = dst_coords - src_coords  # (E, 3): signed (dx, dy, dt)

    abs_delta = np.abs(delta)
    euclidean = np.sqrt((delta**2).sum(axis=1))
    chebyshev = abs_delta.max(axis=1)

    # DEM edge weights for each (src, dst) pair in the complete graph
    dem_w = metadata.dem_edge_weights[fired[src], fired[dst]]

    edge_features = np.column_stack([delta, euclidean, chebyshev, dem_w]).astype(
        np.float32
    )

    return FiredDetectorGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        num_fired=N,
        fired_indices=fired.astype(np.int64),
    )


PHASED_NODE_DIM: int = 11
"""Phased node features: spatial 6 + phase one-hot 3 + patch_id 1 + is_seam 1."""

PHASED_EDGE_DIM: int = EDGE_DIM
"""Phased edge features are identical to spatial."""

_EMPTY_PHASED_NODE_FEATURES = np.zeros((0, PHASED_NODE_DIM), dtype=np.float32)
_EMPTY_PHASED_EDGE_FEATURES = np.zeros((0, PHASED_EDGE_DIM), dtype=np.float32)


def _compute_complete_graph_edges(
    norm_coords: np.ndarray,
    N: int,
    dem_edge_weights: np.ndarray,
    fired: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build complete-graph edge index and edge features.

    Parameters
    ----------
    norm_coords : ndarray, shape ``(N, 3)``
        Normalized ``(x, y, t)`` coordinates.
    N : int
        Number of nodes.
    dem_edge_weights : ndarray, shape ``(D, D)``
        Full pairwise DEM weight matrix.
    fired : ndarray, shape ``(N,)``
        Original detector indices of the fired detectors.

    Returns
    -------
    edge_index : ndarray, shape ``(2, N*(N-1))``, int64
    edge_features : ndarray, shape ``(N*(N-1), EDGE_DIM)``, float32
    """
    idx = np.arange(N, dtype=np.int64)
    all_src = np.repeat(idx, N)
    all_dst = np.tile(idx, N)
    not_self = all_src != all_dst
    src = all_src[not_self]
    dst = all_dst[not_self]
    edge_index = np.stack([src, dst], axis=0)

    src_coords = norm_coords[src]
    dst_coords = norm_coords[dst]
    delta = dst_coords - src_coords

    abs_delta = np.abs(delta)
    euclidean = np.sqrt((delta**2).sum(axis=1))
    chebyshev = abs_delta.max(axis=1)

    dem_w = dem_edge_weights[fired[src], fired[dst]]

    edge_features = np.column_stack([delta, euclidean, chebyshev, dem_w]).astype(
        np.float32
    )
    return edge_index, edge_features


def _patch_geometry(
    all_coords_xy: np.ndarray,
    patch_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Compute per-patch spatial centers.

    Parameters
    ----------
    all_coords_xy : ndarray, shape ``(D, 2)``
        Detector ``(x, y)`` coordinates for all detectors.
    patch_ids : ndarray, shape ``(D,)``
        Patch assignment per detector.

    Returns
    -------
    patch_centers : ndarray, shape ``(max_patch + 1, 2)``
        Centroid of each patch's detectors.
    num_patches : int
        Number of distinct patches.
    """
    unique_patches = np.unique(patch_ids)
    num_patches = len(unique_patches)
    max_id = int(unique_patches.max()) if num_patches > 0 else 0
    centers = np.zeros((max_id + 1, 2), dtype=np.float64)
    for p in unique_patches:
        mask = patch_ids == p
        centers[p] = all_coords_xy[mask].mean(axis=0)
    return centers, num_patches


def _merge_geometry(
    patch_centers: np.ndarray,
    num_patches: int,
    distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute merged-block center and scale for merge phases.

    Parameters
    ----------
    patch_centers : ndarray, shape ``(P, 2)``
    num_patches : int
    distance : int

    Returns
    -------
    merge_center : ndarray, shape ``(2,)``
    merge_scale : ndarray, shape ``(2,)``
        Per-axis scale for boundary distance in the merged block.
    """
    d = float(distance)
    merge_center = patch_centers[:num_patches].mean(axis=0)
    merge_scale = np.array([d, d], dtype=np.float64)

    if num_patches >= 2:
        pc = patch_centers[:num_patches]
        spread = pc.max(axis=0) - pc.min(axis=0)
        merge_axis = 0 if spread[0] >= spread[1] else 1
        merge_scale[merge_axis] = 2.0 * d

    return merge_center, merge_scale


def build_phased_detector_graph(
    syndrome: np.ndarray,
    metadata: CircuitMetadata,
) -> FiredDetectorGraph:
    """Build a phased fired-detector complete graph from a syndrome vector.

    Extends the spatial features with phase one-hot, patch identity, seam
    flag, and dynamic boundary distances that reflect the merged geometry
    during merge phases.

    Parameters
    ----------
    syndrome : ndarray, shape ``(D,)``
        Binary syndrome bit-vector (1 = fired).
    metadata : CircuitMetadata
        Precomputed circuit metadata carrying a ``layout``.

    Returns
    -------
    FiredDetectorGraph
        Complete graph over fired detectors with phased features.

    Raises
    ------
    ValueError
        If metadata carries no logical-operation layout.
    """
    layout = metadata.layout
    if layout is None:
        raise ValueError(
            "build_phased_detector_graph requires CircuitMetadata with "
            "logical-operation fields (a LogicalOperationLayout); got None - use "
            "the spatial builder for memory circuits"
        )

    fired = np.flatnonzero(syndrome)
    N = len(fired)

    if N == 0:
        return FiredDetectorGraph(
            node_features=_EMPTY_PHASED_NODE_FEATURES,
            edge_index=_EMPTY_EDGE_INDEX,
            edge_features=_EMPTY_PHASED_EDGE_FEATURES,
            num_fired=0,
            fired_indices=_EMPTY_FIRED,
            node_dim=PHASED_NODE_DIM,
            edge_dim=PHASED_EDGE_DIM,
        )

    all_coords = metadata.detector_coords
    coords = all_coords[fired]  # (N, 3)
    x = coords[:, 0]
    y = coords[:, 1]
    t = coords[:, 2]

    d = metadata.distance
    r = metadata.rounds
    spatial_scale = 2.0 * d
    temporal_scale = float(r)

    x_norm = x / spatial_scale
    y_norm = y / spatial_scale
    t_norm = t / temporal_scale

    patch_centers, num_patches = _patch_geometry(all_coords[:, :2], layout.patch_ids)
    merge_center, merge_scale = _merge_geometry(patch_centers, num_patches, d)

    fired_patch_ids = layout.patch_ids[fired]
    fired_centers = patch_centers[fired_patch_ids]  # (N, 2)

    d_x = (x - fired_centers[:, 0]) / float(d)
    d_y = (y - fired_centers[:, 1]) / float(d)

    phase_name_arr = np.array(layout.phase_names)
    fired_phase_names = phase_name_arr[layout.phase_ids[fired]]
    merge_mask = fired_phase_names == MERGE_PHASE

    if merge_mask.any():
        d_x[merge_mask] = (x[merge_mask] - merge_center[0]) / merge_scale[0]
        d_y[merge_mask] = (y[merge_mask] - merge_center[1]) / merge_scale[1]

    basis = (((x + y) / 2).astype(np.intp) % 2).astype(np.float32)

    phase_onehot = np.zeros((N, len(PHASE_TYPES)), dtype=np.float32)
    for type_idx, phase_type in enumerate(PHASE_TYPES):
        phase_onehot[fired_phase_names == phase_type, type_idx] = 1.0

    max_patch = max(1, num_patches - 1)
    patch_id_norm = fired_patch_ids.astype(np.float32) / max_patch

    is_seam = layout.seam_mask[fired].astype(np.float32)

    node_features = np.column_stack(
        [
            x_norm,
            y_norm,
            t_norm,
            d_x,
            d_y,
            basis,
            phase_onehot,
            patch_id_norm,
            is_seam,
        ]
    ).astype(np.float32)

    norm_coords = node_features[:, :3]

    if N == 1:
        return FiredDetectorGraph(
            node_features=node_features,
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_features=np.zeros((0, PHASED_EDGE_DIM), dtype=np.float32),
            num_fired=1,
            fired_indices=fired.astype(np.int64),
            node_dim=PHASED_NODE_DIM,
            edge_dim=PHASED_EDGE_DIM,
        )

    edge_index, edge_features = _compute_complete_graph_edges(
        norm_coords, N, metadata.dem_edge_weights, fired
    )

    return FiredDetectorGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        num_fired=N,
        fired_indices=fired.astype(np.int64),
        node_dim=PHASED_NODE_DIM,
        edge_dim=PHASED_EDGE_DIM,
    )
