"""Detector graph construction for GNN-based QEC decoding.

``FiredDetectorGraph`` / ``build_fired_detector_graph``: fired detectors
only, complete graph with learned features. Primary representation for
GNN training and inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


NODE_DIM: int = 6
"""Node feature dimensionality: x_norm, y_norm, t_norm, d_x, d_y, basis."""

EDGE_DIM: int = 6
"""Edge feature dimensionality: dx, dy, dt, euclidean, chebyshev, dem_weight."""


@dataclass(frozen=True, slots=True)
class CircuitMetadata:
    """Precomputed per-circuit metadata for the graph builder.

    Extracted once from a ``stim.Circuit`` via ``extract_circuit_metadata``
    and shared read-only across DataLoader workers.

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
    phase_ids : ndarray or None, shape ``(D,)``, intp
        Per-detector phase-window index (0-based).  ``None`` for operations
        without phase structure (e.g. memory).
    phase_names : tuple of str or None
        Ordered phase-window names, one per window.
        ``phase_names[phase_ids[i]]`` gives the phase of detector *i*.
    patch_ids : ndarray or None, shape ``(D,)``, intp
        Per-detector patch (code-block) index.
    seam_mask : ndarray or None, shape ``(D,)``, bool
        ``True`` for detectors on the merge boundary.
    """

    detector_coords: np.ndarray
    distance: int
    rounds: int
    num_detectors: int
    dem_edge_weights: np.ndarray
    phase_ids: np.ndarray | None = None
    phase_names: tuple[str, ...] | None = None
    patch_ids: np.ndarray | None = None
    seam_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.detector_coords.ndim != 2 or self.detector_coords.shape[1] < 3:
            raise ValueError(
                f"detector_coords must have shape (D, >=3), "
                f"got {self.detector_coords.shape}"
            )
        if self.detector_coords.shape[0] != self.num_detectors:
            raise ValueError(
                f"detector_coords rows {self.detector_coords.shape[0]} != "
                f"num_detectors {self.num_detectors}"
            )
        D = self.num_detectors
        if self.dem_edge_weights.shape != (D, D):
            raise ValueError(
                f"dem_edge_weights shape {self.dem_edge_weights.shape} != "
                f"expected ({D}, {D})"
            )
        if self.distance < 1:
            raise ValueError(f"distance must be >= 1, got {self.distance}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")

        lo_fields = (self.phase_ids, self.phase_names, self.patch_ids, self.seam_mask)
        any_set = any(f is not None for f in lo_fields)
        all_set = all(f is not None for f in lo_fields)
        if any_set and not all_set:
            present = [
                n
                for n, f in zip(
                    ("phase_ids", "phase_names", "patch_ids", "seam_mask"),
                    lo_fields,
                    strict=True,
                )
                if f is not None
            ]
            raise ValueError(
                f"Logical-operation fields are all-or-nothing: got only {present}"
            )

        if self.phase_ids is not None:
            if self.phase_ids.shape != (D,):
                raise ValueError(
                    f"phase_ids shape {self.phase_ids.shape} != expected ({D},)"
                )
            if not self.phase_names:
                raise ValueError("phase_names must be non-empty when phase_ids is set")
            n_phases = len(self.phase_names)
            if self.phase_ids.max() >= n_phases or self.phase_ids.min() < 0:
                raise ValueError(
                    f"phase_ids values must be in [0, {n_phases}), "
                    f"got [{self.phase_ids.min()}, {self.phase_ids.max()}]"
                )
        if self.patch_ids is not None and self.patch_ids.shape != (D,):
            raise ValueError(
                f"patch_ids shape {self.patch_ids.shape} != expected ({D},)"
            )
        if self.seam_mask is not None and self.seam_mask.shape != (D,):
            raise ValueError(
                f"seam_mask shape {self.seam_mask.shape} != expected ({D},)"
            )


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
        E = N * (N - 1) if N > 1 else 0

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
    dem = circuit.detector_error_model(decompose_errors=True)
    num_det = circuit.num_detectors
    log_complement = np.zeros((num_det, num_det), dtype=np.float64)

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        p = instruction.args_copy()[0]
        if p == 0.0:
            continue
        dets = [
            t.val for t in instruction.targets_copy() if t.is_relative_detector_id()
        ]
        if len(dets) == 2:
            i, j = dets
            lp = np.log1p(-p)
            log_complement[i, j] += lp
            log_complement[j, i] += lp

    return 1.0 - np.exp(log_complement)


def extract_circuit_metadata(
    circuit: stim.Circuit,
    distance: int,
    rounds: int,
) -> CircuitMetadata:
    """Extract graph builder metadata from a Stim circuit.

    Parameters
    ----------
    circuit : stim.Circuit
        Circuit with detector coordinates annotated.
    distance : int
        Code distance (used for spatial normalization as ``2 * d``).
    rounds : int
        Number of syndrome measurement rounds (temporal normalization).

    Returns
    -------
    CircuitMetadata
    """
    coord_dict = circuit.get_detector_coordinates()
    num_det = circuit.num_detectors

    coords = np.zeros((num_det, 3), dtype=np.float64)
    for det_id, c in coord_dict.items():
        if 0 <= det_id < num_det:
            coords[det_id, : min(len(c), 3)] = c[:3]

    dem_weights = _extract_dem_weights(circuit)

    return CircuitMetadata(
        detector_coords=coords,
        distance=distance,
        rounds=rounds,
        num_detectors=num_det,
        dem_edge_weights=dem_weights,
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


PHASE_TYPES: tuple[str, ...] = ("memory", "merge", "split")
"""Fixed vocabulary of phase types for the one-hot encoding."""

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
    all_coords_xy: np.ndarray,
    phase_ids: np.ndarray,
    phase_names: tuple[str, ...],
    patch_centers: np.ndarray,
    num_patches: int,
    distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute merged-block center and scale for merge phases.

    Parameters
    ----------
    all_coords_xy : ndarray, shape ``(D, 2)``
    phase_ids : ndarray, shape ``(D,)``
    phase_names : tuple of str
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
        Precomputed circuit metadata **with logical-operation fields
        populated** (``phase_ids``, ``phase_names``, ``patch_ids``,
        ``seam_mask``).

    Returns
    -------
    FiredDetectorGraph
        Complete graph over fired detectors with phased features.

    Raises
    ------
    ValueError
        If metadata lacks logical-operation fields.
    """
    if metadata.phase_ids is None:
        raise ValueError(
            "build_phased_detector_graph requires CircuitMetadata with "
            "logical-operation fields (phase_ids, phase_names, patch_ids, "
            "seam_mask); got None - use the spatial builder for memory circuits"
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

    patch_centers, num_patches = _patch_geometry(all_coords[:, :2], metadata.patch_ids)
    merge_center, merge_scale = _merge_geometry(
        all_coords[:, :2],
        metadata.phase_ids,
        metadata.phase_names,
        patch_centers,
        num_patches,
        d,
    )

    fired_patch_ids = metadata.patch_ids[fired]
    fired_centers = patch_centers[fired_patch_ids]  # (N, 2)

    d_x = (x - fired_centers[:, 0]) / float(d)
    d_y = (y - fired_centers[:, 1]) / float(d)

    phase_name_arr = np.array(metadata.phase_names)
    fired_phase_names = phase_name_arr[metadata.phase_ids[fired]]
    merge_mask = fired_phase_names == "merge"

    if merge_mask.any():
        d_x[merge_mask] = (x[merge_mask] - merge_center[0]) / merge_scale[0]
        d_y[merge_mask] = (y[merge_mask] - merge_center[1]) / merge_scale[1]

    basis = (((x + y) / 2).astype(np.intp) % 2).astype(np.float32)

    phase_onehot = np.zeros((N, len(PHASE_TYPES)), dtype=np.float32)
    for type_idx, phase_type in enumerate(PHASE_TYPES):
        phase_onehot[fired_phase_names == phase_type, type_idx] = 1.0

    max_patch = max(1, num_patches - 1)
    patch_id_norm = fired_patch_ids.astype(np.float32) / max_patch

    is_seam = metadata.seam_mask[fired].astype(np.float32)

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
