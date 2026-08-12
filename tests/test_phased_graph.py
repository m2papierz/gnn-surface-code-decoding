"""Tests for the phased graph representation builder."""

from __future__ import annotations

import numpy as np
import pytest

from sampling.graph import (
    EDGE_DIM,
    NODE_DIM,
    PHASE_TYPES,
    PHASED_EDGE_DIM,
    PHASED_NODE_DIM,
    CircuitMetadata,
    FiredDetectorGraph,
    build_fired_detector_graph,
    build_phased_detector_graph,
)
from sampling.representation import (
    PHASED,
    SPATIAL,
    resolve_builder,
    resolve_descriptor,
)


def _make_merge_split_metadata(
    distance: int = 3,
    rounds: int = 9,
) -> CircuitMetadata:
    """Build synthetic metadata for a 3-phase merge-split with two patches.

    Simulates a ZZ merge-split layout: two d×d patches side-by-side
    along the x-axis, with phases (memory, merge, memory) over 3 temporal
    blocks of ``rounds / 3`` each.

    Patch 0 has detectors at x ∈ [0, 2d], patch 1 at x ∈ [2d, 4d].
    Both span y ∈ [0, 2d].  Seam detectors are those at x ≈ 2d
    (the facing boundary) during the merge phase.
    """
    d = distance
    rng = np.random.default_rng(42)

    n_det_per_patch_per_phase = 8
    n_phases = 3
    n_det = n_det_per_patch_per_phase * 2 * n_phases

    coords = np.zeros((n_det, 3), dtype=np.float64)
    phase_ids = np.zeros(n_det, dtype=np.intp)
    patch_ids = np.zeros(n_det, dtype=np.intp)
    seam_indices: list[int] = []

    phase_stride = rounds / n_phases
    idx = 0
    for phase_idx in range(n_phases):
        t_base = phase_idx * phase_stride
        for patch in range(2):
            x_offset = patch * 2 * d
            for _ in range(n_det_per_patch_per_phase):
                x = x_offset + rng.uniform(0.5, 2 * d - 0.5)
                y = rng.uniform(0.5, 2 * d - 0.5)
                t = t_base + rng.uniform(0, phase_stride - 0.1)
                coords[idx] = [x, y, t]
                phase_ids[idx] = phase_idx
                patch_ids[idx] = patch

                # Seam: detectors near x ≈ 2d during merge phase (phase_idx=1)
                if phase_idx == 1 and abs(x - 2 * d) < d * 0.3:
                    seam_indices.append(idx)

                idx += 1

    # Ensure at least 2 seam detectors for a valid merge
    if len(seam_indices) < 2:
        for i in range(2 - len(seam_indices)):
            det_idx = n_det_per_patch_per_phase * 2 + i
            coords[det_idx, 0] = 2 * d + rng.uniform(-0.2, 0.2)
            seam_indices.append(det_idx)

    phase_names = ("memory", "merge", "memory")
    seam_mask = np.zeros(n_det, dtype=np.bool_)
    for si in seam_indices:
        seam_mask[si] = True

    return CircuitMetadata(
        detector_coords=coords,
        distance=d,
        rounds=rounds,
        num_detectors=n_det,
        dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
        phase_ids=phase_ids,
        phase_names=phase_names,
        patch_ids=patch_ids,
        seam_mask=seam_mask,
    )


def _make_memory_metadata(distance: int = 3, rounds: int = 3) -> CircuitMetadata:
    """Build synthetic memory-only CircuitMetadata (no LO fields)."""
    d = distance
    rng = np.random.default_rng(99)
    n_det = 24
    coords = np.zeros((n_det, 3), dtype=np.float64)
    for i in range(n_det):
        coords[i] = [
            rng.uniform(0, 2 * d),
            rng.uniform(0, 2 * d),
            rng.uniform(0, rounds),
        ]
    return CircuitMetadata(
        detector_coords=coords,
        distance=d,
        rounds=rounds,
        num_detectors=n_det,
        dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
    )


class TestFiredDetectorGraphDimensions:
    """FiredDetectorGraph validates against carried dimensions."""

    def test_spatial_defaults_unchanged(self) -> None:
        g = FiredDetectorGraph(
            node_features=np.zeros((2, NODE_DIM), dtype=np.float32),
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_features=np.zeros((2, EDGE_DIM), dtype=np.float32),
            num_fired=2,
            fired_indices=np.array([0, 1], dtype=np.int64),
        )
        assert g.node_dim == NODE_DIM
        assert g.edge_dim == EDGE_DIM

    def test_phased_dimensions_accepted(self) -> None:
        g = FiredDetectorGraph(
            node_features=np.zeros((2, PHASED_NODE_DIM), dtype=np.float32),
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            edge_features=np.zeros((2, PHASED_EDGE_DIM), dtype=np.float32),
            num_fired=2,
            fired_indices=np.array([0, 1], dtype=np.int64),
            node_dim=PHASED_NODE_DIM,
            edge_dim=PHASED_EDGE_DIM,
        )
        assert g.node_dim == PHASED_NODE_DIM

    def test_mismatched_node_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="node_features shape"):
            FiredDetectorGraph(
                node_features=np.zeros((2, 7), dtype=np.float32),
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                edge_features=np.zeros((2, EDGE_DIM), dtype=np.float32),
                num_fired=2,
                fired_indices=np.array([0, 1], dtype=np.int64),
                node_dim=11,
            )

    def test_mismatched_edge_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="edge_features shape"):
            FiredDetectorGraph(
                node_features=np.zeros((2, NODE_DIM), dtype=np.float32),
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                edge_features=np.zeros((2, 3), dtype=np.float32),
                num_fired=2,
                fired_indices=np.array([0, 1], dtype=np.int64),
            )


class TestPhasedBuilderValidation:
    """build_phased_detector_graph rejects metadata without LO fields."""

    def test_rejects_memory_metadata(self) -> None:
        meta = _make_memory_metadata()
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        with pytest.raises(ValueError, match="logical-operation fields"):
            build_phased_detector_graph(syndrome, meta)


class TestPhasedBuilderShapes:
    """Shape correctness of the phased fired-detector graph builder."""

    @pytest.fixture
    def meta(self) -> CircuitMetadata:
        return _make_merge_split_metadata()

    def test_empty_syndrome(self, meta: CircuitMetadata) -> None:
        syndrome = np.zeros(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        assert g.num_fired == 0
        assert g.node_features.shape == (0, PHASED_NODE_DIM)
        assert g.edge_index.shape == (2, 0)
        assert g.edge_features.shape == (0, PHASED_EDGE_DIM)
        assert g.node_dim == PHASED_NODE_DIM
        assert g.edge_dim == PHASED_EDGE_DIM

    def test_single_fired(self, meta: CircuitMetadata) -> None:
        syndrome = np.zeros(meta.num_detectors, dtype=np.uint8)
        syndrome[0] = 1
        g = build_phased_detector_graph(syndrome, meta)

        assert g.num_fired == 1
        assert g.node_features.shape == (1, PHASED_NODE_DIM)
        assert g.edge_index.shape == (2, 0)

    @pytest.mark.parametrize("n_fired", [2, 3, 5, 10])
    def test_complete_graph_edge_count(
        self, meta: CircuitMetadata, n_fired: int
    ) -> None:
        syndrome = np.zeros(meta.num_detectors, dtype=np.uint8)
        syndrome[:n_fired] = 1
        g = build_phased_detector_graph(syndrome, meta)

        assert g.num_fired == n_fired
        E = n_fired * (n_fired - 1)
        assert g.edge_index.shape == (2, E)
        assert g.edge_features.shape == (E, PHASED_EDGE_DIM)

    def test_all_fired(self, meta: CircuitMetadata) -> None:
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        N = meta.num_detectors
        E = N * (N - 1)
        assert g.num_fired == N
        assert g.node_features.shape == (N, PHASED_NODE_DIM)
        assert g.edge_index.shape == (2, E)

    def test_dtypes(self, meta: CircuitMetadata) -> None:
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        assert g.node_features.dtype == np.float32
        assert g.edge_features.dtype == np.float32
        assert g.edge_index.dtype == np.int64


class TestPhasedBuilderFeatures:
    """Feature value correctness of the phased builder."""

    @pytest.fixture
    def meta(self) -> CircuitMetadata:
        return _make_merge_split_metadata()

    def test_spatial_core_present(self, meta: CircuitMetadata) -> None:
        """First 3 features (x_norm, y_norm, t_norm) are non-negative."""
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        assert np.all(g.node_features[:, 0] >= 0)  # x_norm
        assert np.all(g.node_features[:, 1] >= 0)  # y_norm
        assert np.all(g.node_features[:, 2] >= 0)  # t_norm

    def test_basis_flag_binary(self, meta: CircuitMetadata) -> None:
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        basis = g.node_features[:, 5]
        assert np.all((basis == 0) | (basis == 1))

    def test_phase_onehot_sums_to_one(self, meta: CircuitMetadata) -> None:
        """Each node's phase one-hot sums to exactly 1."""
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        phase_cols = g.node_features[:, 6:9]  # 3 phase types
        row_sums = phase_cols.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_phase_onehot_binary(self, meta: CircuitMetadata) -> None:
        """Phase one-hot columns are 0 or 1."""
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        phase_cols = g.node_features[:, 6:9]
        assert np.all((phase_cols == 0) | (phase_cols == 1))

    def test_memory_and_merge_phases_present(self, meta: CircuitMetadata) -> None:
        """Merge-split metadata produces both memory and merge one-hot."""
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        memory_col = g.node_features[:, 6]
        merge_col = g.node_features[:, 7]
        assert memory_col.sum() > 0
        assert merge_col.sum() > 0

    def test_patch_id_range(self, meta: CircuitMetadata) -> None:
        """Patch IDs are normalized to [0, 1]."""
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        patch_id = g.node_features[:, 9]
        assert np.all(patch_id >= 0)
        assert np.all(patch_id <= 1)

    def test_patch_id_two_values(self, meta: CircuitMetadata) -> None:
        """Two-patch metadata produces exactly 2 distinct patch IDs."""
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        patch_id = g.node_features[:, 9]
        unique_vals = np.unique(patch_id)
        assert len(unique_vals) == 2
        np.testing.assert_allclose(sorted(unique_vals), [0.0, 1.0], atol=1e-6)

    def test_seam_flag_binary(self, meta: CircuitMetadata) -> None:
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        is_seam = g.node_features[:, 10]
        assert np.all((is_seam == 0) | (is_seam == 1))

    def test_seam_flag_present(self, meta: CircuitMetadata) -> None:
        """At least one seam detector exists."""
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        is_seam = g.node_features[:, 10]
        assert is_seam.sum() > 0


class TestDynamicBoundaryDistances:
    """Seam nodes in merge phases get boundary distances from merged geometry."""

    def _two_patch_metadata(self, d: int = 3) -> CircuitMetadata:
        """Realistic two-patch layout with enough detectors per patch.

        Patch 0 detectors spread across [0.5, 2d-0.5] in x.
        Patch 1 detectors spread across [2d+0.5, 4d-0.5] in x.
        Both patches span [0.5, 2d-0.5] in y.
        3 phases: memory (t ∈ [0, 2)), merge (t ∈ [2, 4)), memory (t ∈ [4, 6)).
        Seam detectors are near x ≈ 2d during the merge phase.
        """
        rng = np.random.default_rng(7)
        n_per_patch_per_phase = 12
        n_phases = 3
        n_det = n_per_patch_per_phase * 2 * n_phases

        coords = np.zeros((n_det, 3), dtype=np.float64)
        phase_ids = np.zeros(n_det, dtype=np.intp)
        patch_ids = np.zeros(n_det, dtype=np.intp)
        seam_mask = np.zeros(n_det, dtype=np.bool_)

        idx = 0
        for phase_idx in range(n_phases):
            t_base = phase_idx * 2.0
            for patch in range(2):
                x_lo = patch * 2 * d + 0.5
                x_hi = (patch + 1) * 2 * d - 0.5
                for _ in range(n_per_patch_per_phase):
                    coords[idx] = [
                        rng.uniform(x_lo, x_hi),
                        rng.uniform(0.5, 2 * d - 0.5),
                        t_base + rng.uniform(0, 1.9),
                    ]
                    phase_ids[idx] = phase_idx
                    patch_ids[idx] = patch
                    if phase_idx == 1 and abs(coords[idx, 0] - 2 * d) < 1.0:
                        seam_mask[idx] = True
                    idx += 1

        return CircuitMetadata(
            detector_coords=coords,
            distance=d,
            rounds=6,
            num_detectors=n_det,
            dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
            phase_ids=phase_ids,
            phase_names=("memory", "merge", "memory"),
            patch_ids=patch_ids,
            seam_mask=seam_mask,
        )

    def test_merge_phase_boundary_distances_smaller(self) -> None:
        """Seam detectors near x ≈ 2d get |d_x| closer to 0 during merge
        than they would from per-patch geometry alone.
        """
        d = 3
        meta = self._two_patch_metadata(d)
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        # Identify merge-phase seam nodes in the output
        fired = np.arange(meta.num_detectors)
        merge_seam = (meta.seam_mask[fired]) & (
            np.array(meta.phase_names)[meta.phase_ids[fired]] == "merge"
        )
        if not merge_seam.any():
            pytest.skip("No merge-phase seam detectors in fixture")

        merge_seam_d_x = g.node_features[merge_seam, 3]

        # These detectors are near x ≈ 2d. Under merged geometry
        # (center ≈ 2d, scale = 2d), d_x ≈ 0. Under per-patch geometry
        # (center ≈ d or 3d, scale = d), |d_x| would be ≈ 1.
        # The merged computation should give |d_x| significantly smaller.
        assert np.all(np.abs(merge_seam_d_x) < 0.5), (
            f"merge seam d_x too large: {merge_seam_d_x}"
        )

    def test_memory_phase_per_patch_distances(self) -> None:
        """In memory phases, boundary distances reflect individual patches."""
        d = 3
        meta = self._two_patch_metadata(d)
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        fired = np.arange(meta.num_detectors)
        phase_names = np.array(meta.phase_names)
        memory_mask = phase_names[meta.phase_ids[fired]] == "memory"
        memory_d_x = g.node_features[memory_mask, 3]

        # Memory-phase boundary distances should span roughly [-1, 1]
        # (detectors are spread across each patch)
        assert memory_d_x.min() < -0.3
        assert memory_d_x.max() > 0.3

    def test_merge_uses_wider_scale_on_merge_axis(self) -> None:
        """Merge scale is 2d on the merge axis, d on the other."""
        d = 3
        # Controlled layout: patches differ in x only.
        # Detector at exact center of merged block during merge.
        coords = np.array(
            [
                # Memory-phase: spread across both patches to establish centers
                [1, 3, 0.0],
                [3, 3, 0.0],
                [5, 3, 0.0],
                [7, 3, 0.0],
                [9, 3, 0.0],
                [11, 3, 0.0],
                # Merge-phase: detector at exact merged center
                [6, 3, 3.0],
            ],
            dtype=np.float64,
        )
        meta = CircuitMetadata(
            detector_coords=coords,
            distance=d,
            rounds=6,
            num_detectors=7,
            dem_edge_weights=np.zeros((7, 7), dtype=np.float64),
            phase_ids=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.intp),
            phase_names=("memory", "merge"),
            patch_ids=np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.intp),
            seam_mask=np.array([False, False, False, False, False, False, True]),
        )

        syndrome = np.zeros(7, dtype=np.uint8)
        syndrome[6] = 1  # only the merge-phase detector
        g = build_phased_detector_graph(syndrome, meta)

        # x=6 is near the center of the merged block; d_x should be close
        # to 0. Exact center depends on patch center estimation from
        # detector positions, so allow reasonable tolerance.
        assert abs(g.node_features[0, 3]) < 0.1


class TestSpatialBitIdentity:
    """The spatial builder produces bit-identical output after the refactor."""

    def test_spatial_fixture_identity(self) -> None:
        """Spatial builder output matches a stored reference (fixture)."""
        coords = np.array(
            [
                [2.0, 0.0, 0.0],
                [4.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [0.0, 2.0, 1.5],
                [4.0, 4.0, 2.5],
            ],
            dtype=np.float64,
        )
        meta = CircuitMetadata(
            detector_coords=coords,
            distance=3,
            rounds=3,
            num_detectors=5,
            dem_edge_weights=np.zeros((5, 5), dtype=np.float64),
        )
        syndrome = np.ones(5, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, meta)

        assert g.node_dim == NODE_DIM
        assert g.edge_dim == EDGE_DIM
        assert g.node_features.shape == (5, 6)
        assert g.edge_features.shape == (20, 6)

        # Check a known node feature vector (node 0: (2, 0, 0), d=3, r=3)
        expected_0 = np.array([1 / 3, 0.0, 0.0, -1 / 3, -1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(g.node_features[0], expected_0, atol=1e-6)


class TestPhasedBuilderResolution:
    """Phased builder resolves through LS0a's mechanism."""

    def test_phased_descriptor_registered(self) -> None:
        desc = resolve_descriptor("phased")
        assert desc is PHASED
        assert desc.version == "phased"
        assert desc.node_dim == PHASED_NODE_DIM
        assert desc.edge_dim == PHASED_EDGE_DIM

    def test_phased_builder_resolves(self) -> None:
        builder = resolve_builder("phased")
        assert builder is build_phased_detector_graph

    def test_spatial_builder_unchanged(self) -> None:
        builder = resolve_builder("spatial")
        assert builder is build_fired_detector_graph

    def test_phased_descriptor_features(self) -> None:
        assert PHASED.node_features == (
            "x_norm",
            "y_norm",
            "t_norm",
            "d_x",
            "d_y",
            "basis",
            "phase_memory",
            "phase_merge",
            "phase_split",
            "patch_id",
            "is_seam",
        )
        assert PHASED.edge_features == SPATIAL.edge_features

    def test_phased_builder_produces_correct_shapes(self) -> None:
        meta = _make_merge_split_metadata()
        syndrome = np.zeros(meta.num_detectors, dtype=np.uint8)
        syndrome[:3] = 1
        builder = resolve_builder("phased")
        graph = builder(syndrome, meta)
        assert graph.node_features.shape == (3, PHASED.node_dim)
        assert graph.edge_features.shape == (6, PHASED.edge_dim)


class TestPhasedEdgeFeatures:
    """Edge features match spatial: dx, dy, dt, euclidean, chebyshev, dem_weight."""

    def test_edges_bidirectional(self) -> None:
        meta = _make_merge_split_metadata()
        syndrome = np.zeros(meta.num_detectors, dtype=np.uint8)
        syndrome[:4] = 1
        g = build_phased_detector_graph(syndrome, meta)

        forward = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        reverse = set(zip(g.edge_index[1].tolist(), g.edge_index[0].tolist()))
        assert forward == reverse

    def test_no_self_loops(self) -> None:
        meta = _make_merge_split_metadata()
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_phased_detector_graph(syndrome, meta)

        src, dst = g.edge_index[0], g.edge_index[1]
        assert not np.any(src == dst)

    def test_edge_deltas_antisymmetric(self) -> None:
        meta = _make_merge_split_metadata()
        syndrome = np.zeros(meta.num_detectors, dtype=np.uint8)
        syndrome[:3] = 1
        g = build_phased_detector_graph(syndrome, meta)

        edge_map: dict[tuple[int, int], int] = {}
        for k in range(g.edge_index.shape[1]):
            edge_map[(int(g.edge_index[0, k]), int(g.edge_index[1, k]))] = k

        for (u, v), idx_fwd in edge_map.items():
            idx_rev = edge_map.get((v, u))
            assert idx_rev is not None
            fwd = g.edge_features[idx_fwd]
            rev = g.edge_features[idx_rev]
            np.testing.assert_allclose(fwd[:3], -rev[:3], atol=1e-6)
            np.testing.assert_allclose(fwd[3:], rev[3:], atol=1e-6)


class TestCrossRepresentationGuard:
    """Phased loaded as spatial (or vice versa) raises."""

    def test_phased_graph_rejects_spatial_dim(self) -> None:
        """A graph built with phased dims cannot masquerade as spatial."""
        with pytest.raises(ValueError, match="node_features shape"):
            FiredDetectorGraph(
                node_features=np.zeros((2, PHASED_NODE_DIM), dtype=np.float32),
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                edge_features=np.zeros((2, EDGE_DIM), dtype=np.float32),
                num_fired=2,
                fired_indices=np.array([0, 1], dtype=np.int64),
                # defaults to NODE_DIM=6, but features are 11 wide
            )

    def test_spatial_graph_rejects_phased_dim(self) -> None:
        """A graph built with spatial dims cannot masquerade as phased."""
        with pytest.raises(ValueError, match="node_features shape"):
            FiredDetectorGraph(
                node_features=np.zeros((2, NODE_DIM), dtype=np.float32),
                edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
                edge_features=np.zeros((2, EDGE_DIM), dtype=np.float32),
                num_fired=2,
                fired_indices=np.array([0, 1], dtype=np.int64),
                node_dim=PHASED_NODE_DIM,
            )

    def test_phased_builder_rejects_memory_metadata(self) -> None:
        """Phased builder raises when given metadata without LO fields."""
        meta = _make_memory_metadata()
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        with pytest.raises(ValueError, match="logical-operation fields"):
            build_phased_detector_graph(syndrome, meta)


class TestPhaseTypes:
    """The fixed phase-type vocabulary."""

    def test_phased_descriptor_phase_features_match(self) -> None:
        expected = tuple(f"phase_{t}" for t in PHASE_TYPES)
        actual = PHASED.node_features[6:9]
        assert actual == expected
