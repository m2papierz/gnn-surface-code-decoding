"""Tests for the data generation and packaging pipeline."""

from pathlib import Path

import numpy as np
import pytest
import stim
import torch

from model.dataset import StreamingSurfaceCodeDataset
from sampling.graph import (
    EDGE_DIM,
    NODE_DIM,
    CircuitMetadata,
    FiredDetectorGraph,
    build_fired_detector_graph,
    extract_circuit_metadata,
)
from sampling.sampler import CircuitSetting, WorkerSampler, settings_from_circuit_dir


def _make_metadata(coords: np.ndarray, distance: int, rounds: int) -> CircuitMetadata:
    """Build synthetic CircuitMetadata for testing without Stim."""
    n = coords.shape[0]
    return CircuitMetadata(
        detector_coords=coords.astype(np.float64),
        distance=distance,
        rounds=rounds,
        num_detectors=n,
        dem_edge_weights=np.zeros((n, n), dtype=np.float64),
    )


class TestFiredDetectorGraph:
    """Fired-detector graph builder: fired detectors => complete graph."""

    @pytest.fixture
    def d3_metadata(self) -> CircuitMetadata:
        c = stim.Circuit.from_file("data/circuits/memory/d3_r3_p0_01.stim")
        return extract_circuit_metadata(c, distance=3, rounds=3)

    @pytest.fixture
    def d5_metadata(self) -> CircuitMetadata:
        c = stim.Circuit.from_file("data/circuits/memory/d5_r5_p0_01.stim")
        return extract_circuit_metadata(c, distance=5, rounds=5)

    @pytest.fixture
    def d7_metadata(self) -> CircuitMetadata:
        c = stim.Circuit.from_file("data/circuits/memory/d7_r7_p0_01.stim")
        return extract_circuit_metadata(c, distance=7, rounds=7)

    def test_empty_syndrome_returns_zero_nodes(
        self, d3_metadata: CircuitMetadata
    ) -> None:
        syndrome = np.zeros(d3_metadata.num_detectors, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, d3_metadata)

        assert g.num_fired == 0
        assert g.node_features.shape == (0, NODE_DIM)
        assert g.edge_index.shape == (2, 0)
        assert g.edge_features.shape == (0, EDGE_DIM)
        assert g.fired_indices.shape == (0,)

    def test_single_fired_no_edges(self, d3_metadata: CircuitMetadata) -> None:
        syndrome = np.zeros(d3_metadata.num_detectors, dtype=np.uint8)
        syndrome[5] = 1
        g = build_fired_detector_graph(syndrome, d3_metadata)

        assert g.num_fired == 1
        assert g.node_features.shape == (1, NODE_DIM)
        assert g.edge_index.shape == (2, 0)
        assert g.edge_features.shape == (0, EDGE_DIM)
        assert g.fired_indices[0] == 5

    @pytest.mark.parametrize("n_fired", [2, 3, 5, 10])
    def test_complete_graph_edge_count(self, n_fired: int) -> None:
        """N fired detectors => N*(N-1) directed edges."""
        coords = np.random.default_rng(42).uniform(0, 10, (20, 3))
        meta = _make_metadata(coords, distance=5, rounds=5)
        syndrome = np.zeros(20, dtype=np.uint8)
        syndrome[:n_fired] = 1

        g = build_fired_detector_graph(syndrome, meta)

        assert g.num_fired == n_fired
        expected_edges = n_fired * (n_fired - 1)
        assert g.edge_index.shape == (2, expected_edges)
        assert g.edge_features.shape == (expected_edges, EDGE_DIM)

    def test_node_features_shape_and_dtype(self, d3_metadata: CircuitMetadata) -> None:
        syndrome = np.zeros(d3_metadata.num_detectors, dtype=np.uint8)
        syndrome[0] = syndrome[5] = syndrome[10] = 1
        g = build_fired_detector_graph(syndrome, d3_metadata)

        assert g.node_features.shape == (3, NODE_DIM)
        assert g.node_features.dtype == np.float32
        assert g.edge_features.dtype == np.float32
        assert g.edge_index.dtype == np.int64

    def test_normalized_coords_range(self, d3_metadata: CircuitMetadata) -> None:
        """Normalized (x, y, t) should be in [0, 1]."""
        syndrome = np.ones(d3_metadata.num_detectors, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, d3_metadata)

        x_norm = g.node_features[:, 0]
        y_norm = g.node_features[:, 1]
        t_norm = g.node_features[:, 2]

        assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
        assert np.all(y_norm >= 0) and np.all(y_norm <= 1)
        assert np.all(t_norm >= 0) and np.all(t_norm <= 1)

    def test_boundary_distances_range(self, d3_metadata: CircuitMetadata) -> None:
        """Signed boundary distances d_x, d_y should be in [-1, 1]."""
        syndrome = np.ones(d3_metadata.num_detectors, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, d3_metadata)

        d_x = g.node_features[:, 3]
        d_y = g.node_features[:, 4]

        assert np.all(d_x >= -1) and np.all(d_x <= 1)
        assert np.all(d_y >= -1) and np.all(d_y <= 1)

    def test_basis_flag_binary(self, d3_metadata: CircuitMetadata) -> None:
        syndrome = np.ones(d3_metadata.num_detectors, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, d3_metadata)

        basis = g.node_features[:, 5]
        assert np.all((basis == 0) | (basis == 1))

    def test_basis_flag_both_types_present(self, d3_metadata: CircuitMetadata) -> None:
        """All detectors fired => both stabilizer types present."""
        syndrome = np.ones(d3_metadata.num_detectors, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, d3_metadata)

        basis = g.node_features[:, 5]
        assert 0.0 in basis and 1.0 in basis

    def test_known_node_features(self) -> None:
        """Hand-computed features for a known detector configuration."""
        coords = np.array(
            [[2.0, 0.0, 0.0], [4.0, 2.0, 1.0], [2.0, 4.0, 2.0]],
            dtype=np.float64,
        )
        meta = _make_metadata(coords, distance=3, rounds=3)
        syndrome = np.ones(3, dtype=np.uint8)

        g = build_fired_detector_graph(syndrome, meta)

        # Node 0: (2, 0, 0), d=3, r=3
        # x_norm = 2/6 = 1/3, y_norm = 0/6 = 0, t_norm = 0/3 = 0
        # d_x = (2-3)/3 = -1/3, d_y = (0-3)/3 = -1
        # basis = int((2+0)/2) % 2 = 1 % 2 = 1
        expected_0 = np.array([1 / 3, 0.0, 0.0, -1 / 3, -1.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(g.node_features[0], expected_0, atol=1e-6)

        # Node 1: (4, 2, 1)
        # x_norm = 4/6 = 2/3, y_norm = 2/6 = 1/3, t_norm = 1/3
        # d_x = (4-3)/3 = 1/3, d_y = (2-3)/3 = -1/3
        # basis = int((4+2)/2) % 2 = 3 % 2 = 1
        expected_1 = np.array(
            [2 / 3, 1 / 3, 1 / 3, 1 / 3, -1 / 3, 1.0], dtype=np.float32
        )
        np.testing.assert_allclose(g.node_features[1], expected_1, atol=1e-6)

    def test_known_edge_features(self) -> None:
        """Hand-computed edge features for two nodes."""
        coords = np.array([[0.0, 0.0, 0.0], [6.0, 6.0, 3.0]], dtype=np.float64)
        meta = _make_metadata(coords, distance=3, rounds=3)
        syndrome = np.ones(2, dtype=np.uint8)

        g = build_fired_detector_graph(syndrome, meta)

        # Edge 0=>1: dx = 1-0 = 1.0, dy = 1-0 = 1.0, dt = 1-0 = 1.0
        # (normalized: 6/6=1, 6/6=1, 3/3=1)
        # euclidean = sqrt(3), chebyshev = 1
        e01_idx = None
        for k in range(g.edge_index.shape[1]):
            if g.edge_index[0, k] == 0 and g.edge_index[1, k] == 1:
                e01_idx = k
                break
        assert e01_idx is not None

        ef = g.edge_features[e01_idx]
        np.testing.assert_allclose(ef[0], 1.0, atol=1e-6)  # dx
        np.testing.assert_allclose(ef[1], 1.0, atol=1e-6)  # dy
        np.testing.assert_allclose(ef[2], 1.0, atol=1e-6)  # dt
        np.testing.assert_allclose(ef[3], np.sqrt(3), atol=1e-5)  # euclidean
        np.testing.assert_allclose(ef[4], 1.0, atol=1e-6)  # chebyshev

        # Edge 1=>0: opposite sign deltas
        e10_idx = None
        for k in range(g.edge_index.shape[1]):
            if g.edge_index[0, k] == 1 and g.edge_index[1, k] == 0:
                e10_idx = k
                break
        assert e10_idx is not None

        ef_rev = g.edge_features[e10_idx]
        np.testing.assert_allclose(ef_rev[:3], -ef[:3], atol=1e-6)
        np.testing.assert_allclose(ef_rev[3], ef[3], atol=1e-6)  # same dist
        np.testing.assert_allclose(ef_rev[4], ef[4], atol=1e-6)

    def test_edges_are_bidirectional(self, d3_metadata: CircuitMetadata) -> None:
        syndrome = np.zeros(d3_metadata.num_detectors, dtype=np.uint8)
        syndrome[0] = syndrome[5] = syndrome[10] = syndrome[15] = 1
        g = build_fired_detector_graph(syndrome, d3_metadata)

        forward = set(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
        reverse = set(zip(g.edge_index[1].tolist(), g.edge_index[0].tolist()))
        assert forward == reverse

    def test_no_self_loops(self, d3_metadata: CircuitMetadata) -> None:
        syndrome = np.ones(d3_metadata.num_detectors, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, d3_metadata)

        src, dst = g.edge_index[0], g.edge_index[1]
        assert not np.any(src == dst)

    def test_edge_deltas_antisymmetric(self, d3_metadata: CircuitMetadata) -> None:
        """delta(i=>j) = -delta(j=>i) for directional edge features."""
        syndrome = np.zeros(d3_metadata.num_detectors, dtype=np.uint8)
        syndrome[0] = syndrome[5] = syndrome[10] = 1
        g = build_fired_detector_graph(syndrome, d3_metadata)

        edge_map: dict[tuple[int, int], int] = {}
        for k in range(g.edge_index.shape[1]):
            edge_map[(int(g.edge_index[0, k]), int(g.edge_index[1, k]))] = k

        for (u, v), idx_fwd in edge_map.items():
            idx_rev = edge_map.get((v, u))
            assert idx_rev is not None
            fwd = g.edge_features[idx_fwd]
            rev = g.edge_features[idx_rev]
            np.testing.assert_allclose(
                fwd[:3],
                -rev[:3],
                atol=1e-6,
                err_msg=f"Deltas not antisymmetric for ({u},{v})",
            )
            np.testing.assert_allclose(
                fwd[3:],
                rev[3:],
                atol=1e-6,
                err_msg=f"Distances not symmetric for ({u},{v})",
            )

    def test_fired_indices_match_syndrome(self, d3_metadata: CircuitMetadata) -> None:
        syndrome = np.zeros(d3_metadata.num_detectors, dtype=np.uint8)
        fired = [2, 7, 13, 20]
        for i in fired:
            syndrome[i] = 1
        g = build_fired_detector_graph(syndrome, d3_metadata)

        np.testing.assert_array_equal(sorted(g.fired_indices), sorted(fired))

    @pytest.mark.parametrize(
        "fixture_name,d,r",
        [("d3_metadata", 3, 3), ("d5_metadata", 5, 5), ("d7_metadata", 7, 7)],
    )
    def test_all_fired_shape(
        self,
        fixture_name: str,
        d: int,
        r: int,
        request: pytest.FixtureRequest,
    ) -> None:
        """All detectors fired => correct shapes for d=3, 5, 7."""
        meta: CircuitMetadata = request.getfixturevalue(fixture_name)
        syndrome = np.ones(meta.num_detectors, dtype=np.uint8)
        g = build_fired_detector_graph(syndrome, meta)

        N = meta.num_detectors
        E = N * (N - 1)
        assert g.num_fired == N
        assert g.node_features.shape == (N, NODE_DIM)
        assert g.edge_index.shape == (2, E)
        assert g.edge_features.shape == (E, EDGE_DIM)

    def test_normalization_scale_invariance(self) -> None:
        """Same fractional position gives same normalized coords
        regardless of distance."""
        # Center detector at (d, d, r/2) for d=3 and d=7
        coords_d3 = np.array([[3.0, 3.0, 1.5]], dtype=np.float64)
        coords_d7 = np.array([[7.0, 7.0, 3.5]], dtype=np.float64)
        meta_d3 = _make_metadata(coords_d3, distance=3, rounds=3)
        meta_d7 = _make_metadata(coords_d7, distance=7, rounds=7)

        s = np.ones(1, dtype=np.uint8)
        g3 = build_fired_detector_graph(s, meta_d3)
        g7 = build_fired_detector_graph(s, meta_d7)

        # Normalized coords and boundary distances should be identical
        np.testing.assert_allclose(
            g3.node_features[0, :5], g7.node_features[0, :5], atol=1e-6
        )

    def test_dataclass_validation_catches_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="node_features shape"):
            FiredDetectorGraph(
                node_features=np.zeros((2, 3), dtype=np.float32),
                edge_index=np.zeros((2, 2), dtype=np.int64),
                edge_features=np.zeros((2, EDGE_DIM), dtype=np.float32),
                num_fired=2,
                fired_indices=np.array([0, 1], dtype=np.int64),
            )

    def test_metadata_validation_catches_bad_coords(self) -> None:
        with pytest.raises(ValueError, match="detector_coords"):
            CircuitMetadata(
                detector_coords=np.zeros((5,), dtype=np.float64),
                distance=3,
                rounds=3,
                num_detectors=5,
                dem_edge_weights=np.zeros((5, 5), dtype=np.float64),
            )

    def test_extract_metadata_d3(self) -> None:
        c = stim.Circuit.from_file("data/circuits/memory/d3_r3_p0_01.stim")
        meta = extract_circuit_metadata(c, distance=3, rounds=3)

        assert meta.num_detectors == 24
        assert meta.distance == 3
        assert meta.rounds == 3
        assert meta.detector_coords.shape == (24, 3)

    def test_extract_metadata_d7(self) -> None:
        c = stim.Circuit.from_file("data/circuits/memory/d7_r7_p0_01.stim")
        meta = extract_circuit_metadata(c, distance=7, rounds=7)

        assert meta.num_detectors == 336
        assert meta.distance == 7
        assert meta.rounds == 7


CIRCUITS_DIR = Path("data/circuits/memory")


class TestCircuitSetting:
    """CircuitSetting dataclass validation."""

    def test_invalid_distance(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            CircuitSetting(
                circuit_path=CIRCUITS_DIR / "d3_r3_p0_01.stim",
                distance=0,
                rounds=3,
                error_prob=0.01,
            )

    def test_invalid_error_prob_zero(self) -> None:
        with pytest.raises(ValueError, match="error_prob"):
            CircuitSetting(
                circuit_path=CIRCUITS_DIR / "d3_r3_p0_01.stim",
                distance=3,
                rounds=3,
                error_prob=0.0,
            )

    def test_invalid_error_prob_one(self) -> None:
        with pytest.raises(ValueError, match="error_prob"):
            CircuitSetting(
                circuit_path=CIRCUITS_DIR / "d3_r3_p0_01.stim",
                distance=3,
                rounds=3,
                error_prob=1.0,
            )


# settings_from_circuit_dir


class TestSettingsFromCircuitDir:
    """Discovery of circuit settings from committed .stim files."""

    def test_discovers_all_12(self) -> None:
        settings = settings_from_circuit_dir(CIRCUITS_DIR)
        assert len(settings) == 12

    def test_sorted_by_d_r_p(self) -> None:
        settings = settings_from_circuit_dir(CIRCUITS_DIR)
        keys = [(s.distance, s.rounds, s.error_prob) for s in settings]
        assert keys == sorted(keys)

    def test_filter_by_distance(self) -> None:
        settings = settings_from_circuit_dir(CIRCUITS_DIR, distances=[3])
        assert all(s.distance == 3 for s in settings)
        assert len(settings) == 4

    def test_filter_by_error_prob(self) -> None:
        settings = settings_from_circuit_dir(CIRCUITS_DIR, error_probs=[0.01])
        assert all(abs(s.error_prob - 0.01) < 1e-12 for s in settings)
        assert len(settings) == 3

    def test_filter_both(self) -> None:
        settings = settings_from_circuit_dir(
            CIRCUITS_DIR, distances=[3, 5], error_probs=[0.003, 0.01]
        )
        assert len(settings) == 4

    def test_no_match_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No matching"):
            settings_from_circuit_dir(tmp_path)

    def test_circuit_paths_exist(self) -> None:
        settings = settings_from_circuit_dir(CIRCUITS_DIR)
        for s in settings:
            assert s.circuit_path.exists(), f"Missing: {s.circuit_path}"


def _d3_settings() -> list[CircuitSetting]:
    """d=3 settings across all error probs."""
    return settings_from_circuit_dir(CIRCUITS_DIR, distances=[3])


class TestWorkerSampler:
    """Per-worker streaming sampler with Stim backends."""

    def test_deterministic(self) -> None:
        """Same seed produces identical sample sequences."""
        settings = _d3_settings()
        s1 = WorkerSampler(settings, worker_seed=12345)
        s2 = WorkerSampler(settings, worker_seed=12345)

        for _ in range(20):
            syn1, obs1, meta1, p1 = s1.sample()
            syn2, obs2, meta2, p2 = s2.sample()

            np.testing.assert_array_equal(syn1, syn2)
            np.testing.assert_array_equal(obs1, obs2)
            assert p1 == p2

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different sequences."""
        settings = _d3_settings()
        s1 = WorkerSampler(settings, worker_seed=111)
        s2 = WorkerSampler(settings, worker_seed=222)

        syndromes_1 = [s1.sample()[0].tobytes() for _ in range(50)]
        syndromes_2 = [s2.sample()[0].tobytes() for _ in range(50)]

        assert syndromes_1 != syndromes_2

    def test_output_shapes(self) -> None:
        settings = _d3_settings()
        sampler = WorkerSampler(settings, worker_seed=42)
        syn, obs, meta, p = sampler.sample()

        assert syn.dtype == np.uint8
        assert obs.dtype == np.uint8
        assert syn.shape == (meta.num_detectors,)
        assert obs.ndim == 1
        assert isinstance(meta, CircuitMetadata)
        assert isinstance(p, float)

    def test_uniform_setting_selection(self) -> None:
        """All settings sampled roughly equally over many shots."""
        settings = _d3_settings()
        sampler = WorkerSampler(settings, worker_seed=99)

        p_counts: dict[float, int] = {}
        n = 400
        for _ in range(n):
            _, _, _, p = sampler.sample()
            p_counts[p] = p_counts.get(p, 0) + 1

        assert len(p_counts) == 4
        for count in p_counts.values():
            assert count > n // 8, f"Under-represented: {p_counts}"

    def test_empty_settings_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            WorkerSampler([], worker_seed=42)


class TestStreamingSurfaceCodeDataset:
    """Streaming IterableDataset with fired-detector graph builder."""

    @pytest.fixture
    def d3_dataset(self) -> StreamingSurfaceCodeDataset:
        return StreamingSurfaceCodeDataset(
            settings=_d3_settings(),
            master_seed=42,
        )

    def test_yields_data_objects(self, d3_dataset: StreamingSurfaceCodeDataset) -> None:
        it = iter(d3_dataset)
        for _ in range(10):
            data = next(it)
            assert isinstance(data, torch.Tensor) is False
            assert hasattr(data, "x")
            assert hasattr(data, "edge_index")
            assert hasattr(data, "edge_attr")

    def test_feature_dims(self, d3_dataset: StreamingSurfaceCodeDataset) -> None:
        """Output shapes match feature dimensions."""
        it = iter(d3_dataset)
        for _ in range(20):
            data = next(it)
            N = data.num_fired.item()

            assert data.x.shape == (N, NODE_DIM)
            assert data.x.dtype == torch.float32

            E = N * (N - 1) if N > 1 else 0
            assert data.edge_index.shape == (2, E)
            assert data.edge_index.dtype == torch.int64

            assert data.edge_attr.shape == (E, EDGE_DIM)
            assert data.edge_attr.dtype == torch.float32

    def test_labels_present(self, d3_dataset: StreamingSurfaceCodeDataset) -> None:
        it = iter(d3_dataset)
        data = next(it)

        assert data.y.ndim == 1
        assert data.y.dtype == torch.float32
        assert data.logical.ndim == 1
        torch.testing.assert_close(data.y, data.logical)

    def test_deterministic(self) -> None:
        """Same master_seed produces identical sample sequences."""
        settings = _d3_settings()
        ds1 = StreamingSurfaceCodeDataset(settings=settings, master_seed=77)
        ds2 = StreamingSurfaceCodeDataset(settings=settings, master_seed=77)

        it1, it2 = iter(ds1), iter(ds2)
        for _ in range(20):
            d1, d2 = next(it1), next(it2)
            torch.testing.assert_close(d1.x, d2.x)
            torch.testing.assert_close(d1.edge_attr, d2.edge_attr)
            torch.testing.assert_close(d1.y, d2.y)
            assert d1.num_fired == d2.num_fired

    def test_different_seeds_differ(self) -> None:
        settings = _d3_settings()
        ds1 = StreamingSurfaceCodeDataset(settings=settings, master_seed=1)
        ds2 = StreamingSurfaceCodeDataset(settings=settings, master_seed=2)

        it1, it2 = iter(ds1), iter(ds2)
        xs_1 = [next(it1).x.numpy().tobytes() for _ in range(50)]
        xs_2 = [next(it2).x.numpy().tobytes() for _ in range(50)]
        assert xs_1 != xs_2

    def test_empty_syndrome_handling(self) -> None:
        """Empty syndromes yield Data with num_fired=0 and empty tensors."""
        settings = _d3_settings()
        ds = StreamingSurfaceCodeDataset(settings=settings, master_seed=42)
        it = iter(ds)

        found_empty = False
        for _ in range(500):
            data = next(it)
            if data.num_fired.item() == 0:
                found_empty = True
                assert data.x.shape == (0, NODE_DIM)
                assert data.edge_index.shape == (2, 0)
                assert data.edge_attr.shape == (0, EDGE_DIM)
                break

        assert found_empty, "No empty syndrome in 500 samples at d=3"

    def test_p_feature_absent_by_default(
        self, d3_dataset: StreamingSurfaceCodeDataset
    ) -> None:
        data = next(iter(d3_dataset))
        assert not hasattr(data, "p") or data.p is None

    def test_p_feature_present_when_enabled(self) -> None:
        ds = StreamingSurfaceCodeDataset(
            settings=_d3_settings(),
            master_seed=42,
            include_p_feature=True,
        )
        data = next(iter(ds))

        assert hasattr(data, "p")
        assert data.p.dtype == torch.float32
        p_val = data.p.item()
        assert any(abs(p_val - ref) < 1e-6 for ref in (0.003, 0.005, 0.008, 0.01))

    def test_empty_settings_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            StreamingSurfaceCodeDataset(settings=[], master_seed=42)


class TestCustomMetadataExtractor:
    """WorkerSampler and StreamingSurfaceCodeDataset accept a profile's
    metadata extractor, so any registered operation can be served by the
    same dataset class."""

    def test_worker_sampler_calls_custom_extractor(self) -> None:
        """WorkerSampler invokes the supplied extractor for every setting."""
        settings = _d3_settings()
        call_log: list[tuple[int, int]] = []

        def tracking_extractor(
            circuit: stim.Circuit, distance: int, rounds: int
        ) -> CircuitMetadata:
            call_log.append((distance, rounds))
            return extract_circuit_metadata(circuit, distance, rounds)

        sampler = WorkerSampler(
            settings, worker_seed=42, metadata_extractor=tracking_extractor
        )
        assert len(call_log) == len(settings)
        for (d, r), s in zip(call_log, settings, strict=True):
            assert d == s.distance
            assert r == s.rounds

        syn, obs, meta, p = sampler.sample()
        assert isinstance(meta, CircuitMetadata)

    def test_worker_sampler_default_extractor_matches_explicit(self) -> None:
        """Omitting metadata_extractor is identical to passing
        extract_circuit_metadata."""
        settings = _d3_settings()
        s_default = WorkerSampler(settings, worker_seed=77)
        s_explicit = WorkerSampler(
            settings,
            worker_seed=77,
            metadata_extractor=extract_circuit_metadata,
        )

        for _ in range(20):
            syn_d, obs_d, meta_d, p_d = s_default.sample()
            syn_e, obs_e, meta_e, p_e = s_explicit.sample()
            np.testing.assert_array_equal(syn_d, syn_e)
            np.testing.assert_array_equal(obs_d, obs_e)
            assert p_d == p_e

    def test_dataset_passes_extractor_to_sampler(self) -> None:
        """StreamingSurfaceCodeDataset threads the metadata_extractor
        through to the WorkerSampler it creates in __iter__."""
        settings = _d3_settings()
        calls: list[int] = []

        def counting_extractor(
            circuit: stim.Circuit, distance: int, rounds: int
        ) -> CircuitMetadata:
            calls.append(1)
            return extract_circuit_metadata(circuit, distance, rounds)

        ds = StreamingSurfaceCodeDataset(
            settings=settings,
            master_seed=42,
            metadata_extractor=counting_extractor,
        )
        it = iter(ds)
        next(it)
        assert len(calls) == len(settings)

    def test_dataset_with_profile_contract_and_extractor(self) -> None:
        """Dataset parameterized by a profile's contract and extractor
        produces correct feature dimensions and label shape."""
        from sampling.profile import MEMORY_PROFILE

        settings = _d3_settings()
        profile = MEMORY_PROFILE
        ds = StreamingSurfaceCodeDataset(
            settings=settings,
            master_seed=42,
            contract=profile.data_contract,
            metadata_extractor=profile.metadata_extractor,
        )
        assert ds.contract == profile.data_contract

        data = next(iter(ds))
        assert data.x.shape[1] == profile.data_contract.node_dim
        assert data.edge_attr.shape[1] == profile.data_contract.edge_dim
        assert data.y.shape == (profile.data_contract.num_observables,)

    def test_every_registered_operation_constructs_dataset(self) -> None:
        """For each registered operation with committed circuits, a
        dataset can be constructed from its profile's contract and
        metadata extractor and yields correctly shaped samples."""
        from sampling.profile import registered_operations, resolve_profile
        from sampling.sampler import settings_from_circuit_dir

        repo_root = Path(__file__).resolve().parent.parent
        for op in registered_operations():
            profile = resolve_profile(op)
            root = repo_root / profile.circuit_root
            if not root.is_dir():
                pytest.skip(f"Circuit root {root} not present")
            settings = settings_from_circuit_dir(root)
            contract = profile.data_contract
            ds = StreamingSurfaceCodeDataset(
                settings=settings,
                master_seed=42,
                contract=contract,
                metadata_extractor=profile.metadata_extractor,
            )
            data = next(iter(ds))
            assert data.x.shape[1] == contract.node_dim
            assert data.edge_attr.shape[1] == contract.edge_dim
            assert data.y.shape == (contract.num_observables,)
            assert data.logical.shape == (contract.num_observables,)
