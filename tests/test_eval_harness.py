"""Tests for the evaluation harness: evaluator + decoder wrappers.

Exercises the core evaluate_point function, eval set loading, discovery,
and the dry-run path on the committed CI shard - all CPU-only, no trained
model required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from decoders import Decoder, GNNDecoder, PyMatchingDecoder
from evaluation.evaluator import (
    EvalPointResult,
    EvalReport,
    EvalSet,
    discover_eval_sets,
    evaluate_point,
    load_eval_set,
)
from evaluation.stats import EvalOutcome, WilsonInterval
from model.decoder import build_model
from sampling.graph import CircuitMetadata


CI_SHARD_DIR = Path(__file__).resolve().parent.parent / "data" / "ci_shard" / "memory"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "memory"
CIRCUIT_DIR = Path(__file__).resolve().parent.parent / "data" / "circuits" / "memory"


@pytest.fixture(scope="module")
def ci_shard_eval_set() -> EvalSet:
    """Load CI shard as an EvalSet."""
    if not (CI_SHARD_DIR / "manifest.json").exists():
        pytest.skip("CI shard not found")
    return load_eval_set(CI_SHARD_DIR)


@pytest.fixture(scope="module")
def circuit_metadata(ci_shard_eval_set: EvalSet) -> CircuitMetadata:
    """Build CircuitMetadata from the CI shard."""
    n_det = ci_shard_eval_set.syndromes.shape[1]
    return CircuitMetadata(
        detector_coords=ci_shard_eval_set.detector_coords,
        distance=ci_shard_eval_set.distance,
        rounds=ci_shard_eval_set.rounds,
        num_detectors=n_det,
        dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
    )


class FakeDecoder:
    """Deterministic decoder for testing: predicts all-zero (no flip)."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return np.zeros((syndromes.shape[0], 1), dtype=np.uint8)


class PerfectDecoder:
    """Decoder that always returns the true observable (for testing)."""

    def __init__(self, observables: np.ndarray, name: str = "perfect") -> None:
        self._observables = observables
        self._name = name
        self._offset = 0

    @property
    def name(self) -> str:
        return self._name

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        n = syndromes.shape[0]
        result = self._observables[self._offset : self._offset + n]
        self._offset += n
        return result


class ErrorDecoder:
    """Decoder that always predicts the opposite of truth (for testing)."""

    def __init__(self, observables: np.ndarray, name: str = "error") -> None:
        self._observables = observables
        self._name = name
        self._offset = 0

    @property
    def name(self) -> str:
        return self._name

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        n = syndromes.shape[0]
        result = 1 - self._observables[self._offset : self._offset + n]
        self._offset += n
        return result


class TestLoadEvalSet:
    """Test eval set loading from both npy and npz formats."""

    def test_load_ci_shard_npy_format(self, ci_shard_eval_set: EvalSet) -> None:
        es = ci_shard_eval_set
        assert es.distance == 3
        assert es.rounds == 3
        assert es.error_prob == 0.01
        assert es.num_shots == 256
        assert es.syndromes.shape == (256, 24)
        assert es.observables.shape == (256, 1)
        assert es.detector_coords.shape == (24, 3)

    def test_load_npz_format(self) -> None:
        """Load an eval set in npz format if available."""
        sample_dir = EVAL_DIR / "d3_p0_0100"
        if not sample_dir.exists():
            pytest.skip("Eval set d3_p0_0100 not found")

        es = load_eval_set(sample_dir)
        assert es.distance == 3
        assert es.rounds == 3
        assert es.error_prob == 0.01
        assert es.syndromes.shape[0] == es.num_shots
        assert es.syndromes.shape[1] == 24
        assert es.observables.shape == (es.num_shots, 1)

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            load_eval_set(tmp_path)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.json").write_text('{"distance": 3}')
        np.savez(
            tmp_path / "data.npz",
            syndromes=np.zeros((10, 4), dtype=np.uint8),
            observables=np.zeros((10, 1), dtype=np.uint8),
            detector_coords=np.zeros((4, 3)),
        )
        with pytest.raises(ValueError, match="required field"):
            load_eval_set(tmp_path)


class TestDiscoverEvalSets:
    def test_discovers_existing_sets(self) -> None:
        if not EVAL_DIR.exists():
            pytest.skip("eval dir not found")
        dirs = discover_eval_sets(EVAL_DIR)
        assert len(dirs) >= 1

    def test_filter_by_distance(self) -> None:
        if not EVAL_DIR.exists():
            pytest.skip("eval dir not found")
        dirs = discover_eval_sets(EVAL_DIR, distances=[3])
        for d in dirs:
            manifest = json.loads((d / "manifest.json").read_text())
            assert manifest["distance"] == 3

    def test_filter_by_error_prob(self) -> None:
        if not EVAL_DIR.exists():
            pytest.skip("eval dir not found")
        dirs = discover_eval_sets(EVAL_DIR, error_probs=[0.01])
        for d in dirs:
            manifest = json.loads((d / "manifest.json").read_text())
            assert manifest["error_prob"] == 0.01

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        assert discover_eval_sets(tmp_path) == []

    def test_synthetic_discover_all(self, tmp_path: Path) -> None:
        """Discover with synthetic directories (no live data needed)."""
        for d, p in [(3, 0.01), (5, 0.005), (7, 0.01)]:
            p_str = f"{p:.4f}".replace(".", "_")
            sub = tmp_path / f"d{d}_p{p_str}"
            sub.mkdir()
            (sub / "manifest.json").write_text(
                json.dumps({"distance": d, "error_prob": p})
            )

        dirs = discover_eval_sets(tmp_path)
        assert len(dirs) == 3

    def test_synthetic_filter_by_distance(self, tmp_path: Path) -> None:
        for d, p in [(3, 0.01), (5, 0.005)]:
            p_str = f"{p:.4f}".replace(".", "_")
            sub = tmp_path / f"d{d}_p{p_str}"
            sub.mkdir()
            (sub / "manifest.json").write_text(
                json.dumps({"distance": d, "error_prob": p})
            )

        dirs = discover_eval_sets(tmp_path, distances=[3])
        assert len(dirs) == 1
        manifest = json.loads((dirs[0] / "manifest.json").read_text())
        assert manifest["distance"] == 3

    def test_synthetic_filter_by_error_prob(self, tmp_path: Path) -> None:
        for d, p in [(3, 0.01), (5, 0.005), (7, 0.01)]:
            p_str = f"{p:.4f}".replace(".", "_")
            sub = tmp_path / f"d{d}_p{p_str}"
            sub.mkdir()
            (sub / "manifest.json").write_text(
                json.dumps({"distance": d, "error_prob": p})
            )

        dirs = discover_eval_sets(tmp_path, error_probs=[0.01])
        assert len(dirs) == 2


class TestEvaluatePoint:
    """Core evaluation function with fake decoders."""

    @pytest.fixture()
    def simple_eval_set(self) -> EvalSet:
        """Small synthetic eval set for testing."""
        n = 200
        n_det = 10
        rng = np.random.default_rng(42)
        syndromes = rng.integers(0, 2, size=(n, n_det), dtype=np.uint8)
        observables = rng.integers(0, 2, size=(n, 1), dtype=np.uint8)
        coords = rng.random((n_det, 3))

        return EvalSet(
            syndromes=syndromes,
            observables=observables,
            detector_coords=coords,
            distance=3,
            rounds=3,
            error_prob=0.01,
            num_shots=n,
            circuit_file="data/circuits/memory/d3_r3_p0_01.stim",
            manifest={
                "distance": 3,
                "rounds": 3,
                "error_prob": 0.01,
                "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
            },
        )

    def test_two_identical_decoders_resolved_parity(
        self, simple_eval_set: EvalSet
    ) -> None:
        """Two decoders that produce identical predictions => parity."""
        decoders = {"a": FakeDecoder("a"), "b": FakeDecoder("b")}
        result = evaluate_point(
            simple_eval_set,
            decoders,
            reference_decoder="a",
            check_interval=simple_eval_set.num_shots,
        )
        assert result.outcome in (
            EvalOutcome.RESOLVED_PARITY,
            EvalOutcome.UNRESOLVED,
        )
        assert result.n_shots_used == simple_eval_set.num_shots
        assert "a" in result.decoder_results
        assert "b" in result.decoder_results

    def test_perfect_vs_error_resolved_different(
        self, simple_eval_set: EvalSet
    ) -> None:
        """Perfect vs always-wrong decoder => resolved-different."""
        obs = simple_eval_set.observables
        decoders = {
            "perfect": PerfectDecoder(obs, "perfect"),
            "error": ErrorDecoder(obs, "error"),
        }
        result = evaluate_point(
            simple_eval_set,
            decoders,
            reference_decoder="perfect",
            check_interval=simple_eval_set.num_shots,
        )
        # Perfect decoder has 0 errors; error decoder has ~100% errors
        assert result.decoder_results["perfect"].n_errors == 0
        assert result.decoder_results["error"].n_errors > 0

    def test_result_has_wilson_intervals(self, simple_eval_set: EvalSet) -> None:
        decoders = {"a": FakeDecoder("a"), "b": FakeDecoder("b")}
        result = evaluate_point(
            simple_eval_set,
            decoders,
            reference_decoder="a",
            check_interval=simple_eval_set.num_shots,
        )
        for dr in result.decoder_results.values():
            assert isinstance(dr.ler_interval, WilsonInterval)
            assert 0.0 <= dr.ler_interval.lower <= dr.ler_interval.upper <= 1.0
            assert isinstance(dr.per_round_interval, WilsonInterval)

    def test_result_has_mcnemar(self, simple_eval_set: EvalSet) -> None:
        decoders = {"a": FakeDecoder("a"), "b": FakeDecoder("b")}
        result = evaluate_point(
            simple_eval_set,
            decoders,
            reference_decoder="a",
            check_interval=simple_eval_set.num_shots,
        )
        assert "b" in result.mcnemar_results
        mr = result.mcnemar_results["b"]
        assert mr.p_value >= 0.0
        assert mr.n_discordant >= 0

    def test_per_round_ler_reported(self, simple_eval_set: EvalSet) -> None:
        decoders = {"a": FakeDecoder("a"), "b": FakeDecoder("b")}
        result = evaluate_point(
            simple_eval_set,
            decoders,
            reference_decoder="a",
            check_interval=simple_eval_set.num_shots,
        )
        for dr in result.decoder_results.values():
            assert dr.per_round_ler >= 0.0

    def test_reference_decoder_must_exist(self, simple_eval_set: EvalSet) -> None:
        decoders = {"a": FakeDecoder("a"), "b": FakeDecoder("b")}
        with pytest.raises(ValueError, match="reference_decoder"):
            evaluate_point(simple_eval_set, decoders, reference_decoder="missing")

    def test_needs_at_least_two_decoders(self, simple_eval_set: EvalSet) -> None:
        with pytest.raises(ValueError, match="at least 2 decoders"):
            evaluate_point(
                simple_eval_set,
                {"a": FakeDecoder("a")},
                reference_decoder="a",
            )

    def test_all_shots_processed_with_non_divisible_interval(self) -> None:
        """Tail shots must not be truncated when n_shots % check_interval != 0."""
        n = 250
        rng = np.random.default_rng(7)
        es = EvalSet(
            syndromes=rng.integers(0, 2, size=(n, 10), dtype=np.uint8),
            observables=rng.integers(0, 2, size=(n, 1), dtype=np.uint8),
            detector_coords=rng.random((10, 3)),
            distance=3,
            rounds=3,
            error_prob=0.01,
            num_shots=n,
            circuit_file="data/circuits/memory/d3_r3_p0_01.stim",
            manifest={
                "distance": 3,
                "rounds": 3,
                "error_prob": 0.01,
                "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
            },
        )
        result = evaluate_point(
            es,
            {"a": FakeDecoder("a"), "b": FakeDecoder("b")},
            reference_decoder="a",
            check_interval=100,
        )
        assert result.n_shots_used == n

    def test_stopping_baseline_ignores_other_decoders(self) -> None:
        """Stopping is driven only by the stopping_baseline, not other decoders."""
        n = 500
        rng = np.random.default_rng(99)
        observables = rng.integers(0, 2, size=(n, 1), dtype=np.uint8)
        es = EvalSet(
            syndromes=rng.integers(0, 2, size=(n, 10), dtype=np.uint8),
            observables=observables,
            detector_coords=rng.random((10, 3)),
            distance=3,
            rounds=3,
            error_prob=0.01,
            num_shots=n,
            circuit_file="data/circuits/memory/d3_r3_p0_01.stim",
            manifest={
                "distance": 3,
                "rounds": 3,
                "error_prob": 0.01,
                "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
            },
        )
        # 'ref': always predicts correctly (0 errors)
        # 'baseline': always predicts correctly too (parity with ref)
        # 'loud': always wrong (would trigger resolved-different immediately)
        decoders = {
            "ref": PerfectDecoder(observables.copy(), "ref"),
            "baseline": PerfectDecoder(observables.copy(), "baseline"),
            "loud": ErrorDecoder(observables.copy(), "loud"),
        }
        result = evaluate_point(
            es,
            decoders,
            reference_decoder="ref",
            stopping_baseline="baseline",
            check_interval=n,
        )
        # Stopping is driven by ref vs baseline (parity), not ref vs loud
        assert result.outcome in (
            EvalOutcome.RESOLVED_PARITY,
            EvalOutcome.UNRESOLVED,
        )
        assert result.n_shots_used == n
        # McNemar for ref vs loud is still computed
        assert "loud" in result.mcnemar_results

    def test_stopping_baseline_validation(self, simple_eval_set: EvalSet) -> None:
        """Invalid stopping_baseline raises ValueError."""
        decoders = {"a": FakeDecoder("a"), "b": FakeDecoder("b")}
        with pytest.raises(ValueError, match="stopping_baseline"):
            evaluate_point(
                simple_eval_set,
                decoders,
                reference_decoder="a",
                stopping_baseline="missing",
            )
        with pytest.raises(ValueError, match="stopping_baseline must differ"):
            evaluate_point(
                simple_eval_set,
                decoders,
                reference_decoder="a",
                stopping_baseline="a",
            )


class TestEvalReport:
    def test_save_and_load(self, tmp_path: Path) -> None:
        report = EvalReport(metadata={"mode": "test"})
        out = tmp_path / "results.json"
        report.save(out)
        loaded = json.loads(out.read_text())
        assert loaded["metadata"]["mode"] == "test"


class TestPyMatchingDecoder:
    def test_satisfies_protocol(self) -> None:
        circuit_path = CIRCUIT_DIR / "d3_r3_p0_01.stim"
        if not circuit_path.exists():
            pytest.skip("Circuit file not found")
        decoder = PyMatchingDecoder(circuit_path)
        assert isinstance(decoder, Decoder)
        assert decoder.name == "mwpm"

    def test_decode_batch_shapes(self) -> None:
        circuit_path = CIRCUIT_DIR / "d3_r3_p0_01.stim"
        if not circuit_path.exists():
            pytest.skip("Circuit file not found")
        decoder = PyMatchingDecoder(circuit_path)
        syndromes = np.zeros((10, 24), dtype=np.uint8)
        result = decoder.decode_batch(syndromes)
        assert result.shape == (10, 1)
        assert result.dtype == np.uint8

    def test_decode_batch_binary_output(self) -> None:
        circuit_path = CIRCUIT_DIR / "d3_r3_p0_01.stim"
        if not circuit_path.exists():
            pytest.skip("Circuit file not found")
        decoder = PyMatchingDecoder(circuit_path)
        rng = np.random.default_rng(0)
        syndromes = rng.integers(0, 2, size=(50, 24), dtype=np.uint8)
        result = decoder.decode_batch(syndromes)
        assert set(np.unique(result)).issubset({0, 1})


class TestGNNDecoder:
    @pytest.fixture()
    def gnn_decoder(self, circuit_metadata: CircuitMetadata) -> GNNDecoder:
        model = build_model(hidden_dim=32, num_layers=2, dropout=0.0)
        return GNNDecoder.from_metadata(
            model=model,
            metadata=circuit_metadata,
            threshold=0.0,
            device=torch.device("cpu"),
            batch_size=32,
        )

    def test_satisfies_protocol(self, gnn_decoder: GNNDecoder) -> None:
        assert isinstance(gnn_decoder, Decoder)
        assert gnn_decoder.name == "gnn"

    def test_decode_batch_shapes(self, gnn_decoder: GNNDecoder) -> None:
        syndromes = np.zeros((10, 24), dtype=np.uint8)
        result = gnn_decoder.decode_batch(syndromes)
        assert result.shape == (10, 1)
        assert result.dtype == np.uint8

    def test_empty_syndrome_predicts_no_flip(self, gnn_decoder: GNNDecoder) -> None:
        syndromes = np.zeros((5, 24), dtype=np.uint8)
        result = gnn_decoder.decode_batch(syndromes)
        assert np.all(result == 0)

    def test_decode_batch_binary_output(
        self, gnn_decoder: GNNDecoder, ci_shard_eval_set: EvalSet
    ) -> None:
        syndromes = ci_shard_eval_set.syndromes[:20]
        result = gnn_decoder.decode_batch(syndromes)
        assert set(np.unique(result)).issubset({0, 1})


class TestDryRunEndToEnd:
    """Full pipeline on CI shard: load => decode => evaluate => report."""

    def test_dry_run_pipeline(
        self,
        ci_shard_eval_set: EvalSet,
        circuit_metadata: CircuitMetadata,
    ) -> None:
        """Dry-run produces valid results, MWPM has errors, report serializes."""
        model = build_model(hidden_dim=32, num_layers=2, dropout=0.0)
        gnn_decoder = GNNDecoder.from_metadata(
            model=model,
            metadata=circuit_metadata,
            threshold=0.0,
            device=torch.device("cpu"),
            batch_size=64,
        )

        circuit_path = CIRCUIT_DIR / "d3_r3_p0_01.stim"
        if not circuit_path.exists():
            pytest.skip("Circuit file not found")
        mwpm_decoder = PyMatchingDecoder(circuit_path)

        result = evaluate_point(
            ci_shard_eval_set,
            {"gnn": gnn_decoder, "mwpm": mwpm_decoder},
            reference_decoder="gnn",
            check_interval=ci_shard_eval_set.num_shots,
        )

        assert isinstance(result, EvalPointResult)
        assert result.distance == 3
        assert result.n_shots_used == 256
        assert "gnn" in result.decoder_results
        assert "mwpm" in result.decoder_results
        assert result.outcome in EvalOutcome

        mwpm_dr = result.decoder_results["mwpm"]
        assert mwpm_dr.n_errors > 0
        assert 0.0 < mwpm_dr.ler < 1.0

        report = EvalReport(results=[result], metadata={"mode": "dry-run"})
        d = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        loaded = json.loads(json_str)
        assert loaded["points"][0]["distance"] == 3
