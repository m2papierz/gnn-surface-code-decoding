"""Tests for the experiment identity carried by every point.

Covers the ``(operation, d, r, p)`` record, the rules that resolve it from a
committed artifact, and its appearance in evaluation results.  CPU-only; the
committed circuits and eval sets are the fixtures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import stim

from evaluation.evaluator import (
    EvalReport,
    EvalSet,
    discover_eval_sets,
    evaluate_point,
    load_eval_set,
)
from sampling.experiment import (
    ExperimentKey,
    circuit_manifest_path,
    eval_set_operation,
    read_circuit_manifest,
    validate_operation_name,
    write_circuit_manifest,
)
from sampling.sampler import ExperimentPoint, settings_from_circuit_dir


REPO_ROOT = Path(__file__).resolve().parent.parent
CIRCUITS_DIR = REPO_ROOT / "data" / "circuits" / "memory"
EVAL_DIR = REPO_ROOT / "data" / "eval" / "memory"
CI_SHARD_DIR = REPO_ROOT / "data" / "ci_shard" / "memory"

_MEMORY_CIRCUIT = "data/circuits/memory/d3_r3_p0_01.stim"


class ZeroDecoder:
    """Deterministic decoder predicting no flip, for exercising the evaluator."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        return np.zeros((syndromes.shape[0], 1), dtype=np.uint8)


def _write_circuit(directory: Path, name: str, distance: int) -> Path:
    """Write a small surface code circuit into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=0.01,
    )
    path = directory / name
    circuit.to_file(path)
    return path


def _synthetic_eval_set(circuit_file: str, manifest: dict) -> EvalSet:
    """A tiny eval set carrying the given circuit reference and manifest."""
    rng = np.random.default_rng(0)
    n, n_det = 40, 10
    return EvalSet(
        syndromes=rng.integers(0, 2, size=(n, n_det), dtype=np.uint8),
        observables=rng.integers(0, 2, size=(n, 1), dtype=np.uint8),
        detector_coords=rng.random((n_det, 3)),
        distance=3,
        rounds=3,
        error_prob=0.01,
        num_shots=n,
        circuit_file=circuit_file,
        manifest=manifest,
    )


class TestExperimentKey:
    @pytest.mark.parametrize("operation", ["", "Memory", "3d", "zz-merge", "zz merge"])
    def test_invalid_operation_rejected(self, operation: str) -> None:
        with pytest.raises(ValueError, match="operation"):
            ExperimentKey(operation=operation, distance=3, rounds=3, error_prob=0.01)

    def test_invalid_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            ExperimentKey(operation="memory", distance=0, rounds=3, error_prob=0.01)

    def test_invalid_rounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="rounds"):
            ExperimentKey(operation="memory", distance=3, rounds=0, error_prob=0.01)

    @pytest.mark.parametrize("error_prob", [0.0, 1.0, -0.1])
    def test_invalid_error_prob_rejected(self, error_prob: float) -> None:
        with pytest.raises(ValueError, match="error_prob"):
            ExperimentKey(
                operation="memory", distance=3, rounds=3, error_prob=error_prob
            )

    def test_is_hashable_and_comparable(self) -> None:
        a = ExperimentKey(operation="memory", distance=3, rounds=3, error_prob=0.01)
        b = ExperimentKey(operation="memory", distance=3, rounds=3, error_prob=0.01)
        c = ExperimentKey(operation="tcnot", distance=3, rounds=3, error_prob=0.01)
        assert a == b
        assert len({a, b, c}) == 2


class TestCommittedCircuitManifests:
    def test_every_committed_circuit_declares_memory(self) -> None:
        circuits = sorted(CIRCUITS_DIR.glob("*.stim"))
        assert circuits, "no committed circuits found"
        for circuit_path in circuits:
            manifest = read_circuit_manifest(circuit_path)
            assert manifest is not None, f"no manifest beside {circuit_path.name}"
            assert manifest["operation"] == "memory"

    def test_manifest_matches_the_circuit_it_describes(self) -> None:
        """A hand-edited manifest cannot drift away from its circuit."""
        for circuit_path in sorted(CIRCUITS_DIR.glob("*.stim")):
            manifest = read_circuit_manifest(circuit_path)
            assert manifest is not None
            circuit = stim.Circuit.from_file(circuit_path)
            dem = circuit.detector_error_model(decompose_errors=True)

            assert (
                manifest["circuit_sha256"]
                == hashlib.sha256(circuit_path.read_bytes()).hexdigest()
            )
            assert manifest["num_detectors"] == dem.num_detectors
            assert manifest["num_observables"] == dem.num_observables


class TestDiscoveryCarriesTheKey:
    def test_committed_circuits_resolve_to_memory(self) -> None:
        points = settings_from_circuit_dir(CIRCUITS_DIR)
        assert len(points) == 12
        for point in points:
            assert isinstance(point, ExperimentPoint)
            assert point.operation == "memory"
            assert point.key == ExperimentKey(
                operation="memory",
                distance=point.distance,
                rounds=point.rounds,
                error_prob=point.error_prob,
            )

    def test_manifest_overrides_the_filename(self, tmp_path: Path) -> None:
        """Where manifest and filename disagree, the manifest decides."""
        circuit_path = _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        write_circuit_manifest(
            circuit_path,
            ExperimentKey(
                operation="zz_merge_split", distance=5, rounds=9, error_prob=0.005
            ),
            provenance={"note": "declared identity differs from the filename"},
        )

        (point,) = settings_from_circuit_dir(tmp_path)
        assert point.operation == "zz_merge_split"
        assert (point.distance, point.rounds, point.error_prob) == (5, 9, 0.005)

    def test_operation_root_resolves(self, tmp_path: Path) -> None:
        op_root = tmp_path / "zz_merge_split"
        circuit_path = _write_circuit(op_root, "d3_r9_p0_005.stim", distance=3)
        write_circuit_manifest(
            circuit_path,
            ExperimentKey(
                operation="zz_merge_split", distance=3, rounds=9, error_prob=0.005
            ),
            provenance={},
        )

        (point,) = settings_from_circuit_dir(op_root)
        assert point.operation == "zz_merge_split"

    def test_discovery_does_not_descend_into_subdirectories(
        self, tmp_path: Path
    ) -> None:
        """One operation's root keeps its own circuits when another sits inside."""
        _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        _write_circuit(tmp_path / "zz_merge_split", "d3_r9_p0_005.stim", distance=3)

        points = settings_from_circuit_dir(tmp_path)
        assert len(points) == 1
        assert points[0].circuit_path.parent == tmp_path

    def test_unmanifested_circuit_directory_resolves_to_memory(
        self, tmp_path: Path
    ) -> None:
        """Circuits committed before manifests existed are memory circuits."""
        _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        (point,) = settings_from_circuit_dir(tmp_path)
        assert point.operation == "memory"

    def test_manifest_without_operation_is_rejected(self, tmp_path: Path) -> None:
        circuit_path = _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        circuit_manifest_path(circuit_path).write_text(
            json.dumps({"distance": 3, "rounds": 3, "error_prob": 0.01}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="operation"):
            settings_from_circuit_dir(tmp_path)

    def test_truncated_manifest_is_rejected(self, tmp_path: Path) -> None:
        """A half-written manifest fails loudly instead of parsing as something."""
        circuit_path = _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        circuit_manifest_path(circuit_path).write_text(
            '{"operation": "memory", "distance": 3, "rou', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            settings_from_circuit_dir(tmp_path)

    def test_manifest_with_invalid_key_value_names_the_manifest(
        self, tmp_path: Path
    ) -> None:
        circuit_path = _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        manifest_path = circuit_manifest_path(circuit_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "operation": "memory",
                    "distance": "three",
                    "rounds": 3,
                    "error_prob": 0.01,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=manifest_path.name):
            settings_from_circuit_dir(tmp_path)

    def test_two_circuits_claiming_one_identity_are_rejected(
        self, tmp_path: Path
    ) -> None:
        """A duplicated identity would double that point's weight in the mixture."""
        first = _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        second = _write_circuit(tmp_path, "d5_r5_p0_01.stim", distance=5)
        key = ExperimentKey(operation="memory", distance=3, rounds=3, error_prob=0.01)
        write_circuit_manifest(first, key, provenance={})
        write_circuit_manifest(second, key, provenance={})

        with pytest.raises(ValueError, match="same experiment point"):
            settings_from_circuit_dir(tmp_path)

    def test_missing_circuit_directory_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError, match="does not exist"):
            settings_from_circuit_dir(tmp_path / "absent")


class TestWriteCircuitManifest:
    def test_round_trips_and_leaves_no_temporary_behind(self, tmp_path: Path) -> None:
        circuit_path = _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        key = ExperimentKey(
            operation="zz_merge_split", distance=3, rounds=9, error_prob=0.005
        )
        manifest_path = write_circuit_manifest(
            circuit_path, key, provenance={"tqec_version": "0.0.0"}
        )

        manifest = read_circuit_manifest(circuit_path)
        assert manifest is not None
        assert manifest["operation"] == "zz_merge_split"
        assert manifest["rounds"] == 9
        assert manifest["tqec_version"] == "0.0.0"
        assert manifest_path.read_text(encoding="utf-8").endswith("\n")
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "d3_r3_p0_01.manifest.json",
            "d3_r3_p0_01.stim",
        ]

    def test_provenance_cannot_redeclare_a_key_field(self, tmp_path: Path) -> None:
        circuit_path = _write_circuit(tmp_path, "d3_r3_p0_01.stim", distance=3)
        with pytest.raises(ValueError, match="operation"):
            write_circuit_manifest(
                circuit_path,
                ExperimentKey(
                    operation="memory", distance=3, rounds=3, error_prob=0.01
                ),
                provenance={"operation": "zz_merge_split"},
            )


class TestArtifactRoots:
    """Every artifact tree is partitioned by operation at its root."""

    @pytest.mark.parametrize("tree", ["circuits", "eval", "ci_shard"])
    def test_root_holds_only_operation_directories(self, tree: str) -> None:
        root = REPO_ROOT / "data" / tree
        entries = [p for p in root.iterdir() if p.name != ".gitignore"]
        assert entries, f"{root} is empty"
        for entry in entries:
            assert entry.is_dir(), f"{entry} sits beside the operation roots"
            validate_operation_name(entry.name)

    def test_every_eval_set_names_a_circuit_that_exists(self) -> None:
        """A relocation that missed a reference shows up here, not at eval time."""
        manifests = sorted((REPO_ROOT / "data" / "eval").glob("*/*/manifest.json"))
        manifests += sorted((REPO_ROOT / "data" / "ci_shard").glob("*/manifest.json"))
        assert manifests, "no committed shot manifests found"
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            circuit_path = REPO_ROOT / manifest["circuit_file"]
            assert circuit_path.exists(), (
                f"{manifest_path} names a missing circuit {manifest['circuit_file']}"
            )

    def test_discovery_does_not_span_operations(self) -> None:
        """The tree root is not an operation root, so a sweep cannot span two."""
        assert discover_eval_sets(REPO_ROOT / "data" / "eval") == []
        assert discover_eval_sets(EVAL_DIR)


class TestEvalSetOperation:
    def test_committed_eval_set_resolves_without_declaring(self) -> None:
        eval_dir = EVAL_DIR / "d3_p0_0100"
        if not eval_dir.exists():
            pytest.skip("eval set d3_p0_0100 not found")
        eval_set = load_eval_set(eval_dir)
        assert "operation" not in eval_set.manifest
        assert eval_set.operation == "memory"

    def test_ci_shard_resolves_without_declaring(self) -> None:
        eval_set = load_eval_set(CI_SHARD_DIR)
        assert "operation" not in eval_set.manifest
        assert eval_set.operation == "memory"

    def test_declared_operation_is_used_when_it_agrees(self) -> None:
        assert eval_set_operation({"operation": "memory"}, _MEMORY_CIRCUIT) == "memory"

    def test_declared_operation_disagreeing_with_its_circuit_raises(self) -> None:
        with pytest.raises(ValueError, match="zz_merge_split"):
            eval_set_operation({"operation": "zz_merge_split"}, _MEMORY_CIRCUIT)

    def test_unreachable_circuit_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="no_such_circuit"):
            eval_set_operation({}, "data/circuits/no_such_circuit.stim")

    def test_resolved_once_at_construction(self) -> None:
        """A set whose identity cannot be established fails before any decode."""
        with pytest.raises(FileNotFoundError, match="no_such_circuit"):
            _synthetic_eval_set(
                "data/circuits/no_such_circuit.stim",
                {"distance": 3, "rounds": 3, "error_prob": 0.01},
            )


class TestResultsCarryTheOperation:
    def test_point_result_carries_the_key(self) -> None:
        eval_set = _synthetic_eval_set(
            _MEMORY_CIRCUIT,
            {
                "distance": 3,
                "rounds": 3,
                "error_prob": 0.01,
                "circuit_file": _MEMORY_CIRCUIT,
            },
        )
        result = evaluate_point(
            eval_set,
            {"a": ZeroDecoder("a"), "b": ZeroDecoder("b")},
            reference_decoder="a",
            check_interval=eval_set.num_shots,
        )
        assert result.key == ExperimentKey(
            operation="memory", distance=3, rounds=3, error_prob=0.01
        )
        assert result.operation == "memory"

    def test_serialized_report_carries_the_operation(self) -> None:
        eval_set = _synthetic_eval_set(
            _MEMORY_CIRCUIT,
            {
                "distance": 3,
                "rounds": 3,
                "error_prob": 0.01,
                "circuit_file": _MEMORY_CIRCUIT,
            },
        )
        result = evaluate_point(
            eval_set,
            {"a": ZeroDecoder("a"), "b": ZeroDecoder("b")},
            reference_decoder="a",
            check_interval=eval_set.num_shots,
        )
        point = EvalReport(results=[result]).to_dict()["points"][0]
        assert point["operation"] == "memory"
        assert point["distance"] == 3
