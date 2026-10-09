"""Tests for the representation-aware evaluation path.

Verifies that a loaded eval set takes its metadata from its circuit through
the operation's extractor, that the record refuses shots its metadata does
not describe, and that GNNDecoder rejects mismatched representations.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import stim

from decoders import GNNDecoder
from evaluation.evaluator import EvalSet, load_eval_set
from model.decoder import build_model
from sampling.experiment import circuit_key
from sampling.graph import (
    CircuitMetadata,
    LogicalOperationLayout,
    extract_circuit_metadata,
)
from sampling.profile import resolve_profile
from sampling.representation import (
    PHASED,
    DataContract,
    LabelSpec,
)
from synthetic_eval_sets import MEMORY_D3, synthetic_eval_set


REPO_ROOT = Path(__file__).resolve().parent.parent
CI_SHARD_DIR = REPO_ROOT / "data" / "ci_shard" / "memory"
FROZEN_SETS = sorted(
    [
        *(REPO_ROOT / "data/eval").glob("*/*/"),
        *(REPO_ROOT / "data/ci_shard").glob("*/"),
    ],
    key=str,
)


def _frozen_arrays(frozen_dir: Path) -> dict[str, np.ndarray]:
    """Every array a frozen set or CI shard stored, by name."""
    if (frozen_dir / "data.npz").exists():
        with np.load(frozen_dir / "data.npz") as archive:
            return dict(archive)
    return {path.stem: np.load(path) for path in frozen_dir.glob("*.npy")}


def _ci_shard_copy(tmp_path: Path) -> Path:
    """A writable copy of the CI shard, still bound to the committed circuit."""
    copy = tmp_path / "memory"
    shutil.copytree(CI_SHARD_DIR, copy)
    return copy


class TestLoadedMetadataComesFromTheCircuit:
    @pytest.mark.parametrize(
        "frozen_dir", FROZEN_SETS, ids=lambda path: f"{path.parent.name}/{path.name}"
    )
    def test_committed_set_loads_the_metadata_it_froze(self, frozen_dir: Path) -> None:
        """The circuit's metadata equals what generation stored beside the shots."""
        eval_set = load_eval_set(frozen_dir)
        frozen = _frozen_arrays(frozen_dir)
        metadata = eval_set.circuit_metadata
        profile = resolve_profile(eval_set.operation)

        assert eval_set.key == circuit_key(eval_set.circuit_file)
        assert eval_set.observable_names == profile.observable_names
        assert metadata.num_detectors == frozen["syndromes"].shape[1]
        np.testing.assert_array_equal(
            metadata.detector_coords, frozen["detector_coords"], strict=True
        )
        if "phase_ids" not in frozen:
            assert metadata.layout is None
            return
        assert metadata.layout is not None
        for name in ("phase_ids", "patch_ids", "seam_mask"):
            np.testing.assert_array_equal(
                getattr(metadata.layout, name), frozen[name], strict=True
            )
        manifest = json.loads((frozen_dir / "manifest.json").read_text())
        assert metadata.layout.phase_names == tuple(manifest["phase_names"])

    def test_metadata_arrays_stored_beside_the_shots_are_not_read(
        self, tmp_path: Path
    ) -> None:
        shard = _ci_shard_copy(tmp_path)
        np.save(shard / "detector_coords.npy", np.full((24, 3), np.nan))

        loaded = load_eval_set(shard).circuit_metadata

        np.testing.assert_array_equal(
            loaded.detector_coords,
            load_eval_set(CI_SHARD_DIR).circuit_metadata.detector_coords,
        )

    def test_dem_weights_come_from_the_circuit(self) -> None:
        eval_set = load_eval_set(CI_SHARD_DIR)
        circuit = stim.Circuit.from_file(eval_set.circuit_file)
        expected = extract_circuit_metadata(circuit, distance=3, rounds=3)

        assert expected.dem_edge_weights.any()
        np.testing.assert_array_equal(
            eval_set.circuit_metadata.dem_edge_weights,
            expected.dem_edge_weights,
            strict=True,
        )

    def test_missing_circuit_raises_naming_set_and_circuit(
        self, tmp_path: Path
    ) -> None:
        shard = _ci_shard_copy(tmp_path)
        manifest = json.loads((shard / "manifest.json").read_text())
        manifest["circuit_file"] = "data/circuits/memory/no_such_circuit.stim"
        (shard / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(FileNotFoundError, match=f"{shard}.*no_such_circuit"):
            load_eval_set(shard)

    def test_detector_count_mismatch_raises_naming_set_and_circuit(
        self, tmp_path: Path
    ) -> None:
        shard = _ci_shard_copy(tmp_path)
        np.save(shard / "syndromes.npy", np.load(shard / "syndromes.npy")[:, :-1])

        with pytest.raises(
            ValueError, match=f"{shard}.*d3_r3_p0_01.stim.*23 detectors.*24"
        ):
            load_eval_set(shard)

    def test_shots_without_any_array_raise_naming_the_set(self, tmp_path: Path) -> None:
        shard = _ci_shard_copy(tmp_path)
        (shard / "observables.npy").unlink()

        with pytest.raises(FileNotFoundError, match=f"{shard} holds neither"):
            load_eval_set(shard)


class TestEvalSetRefusesShotsItsMetadataDoesNotDescribe:
    @pytest.fixture
    def eval_set(self) -> EvalSet:
        return synthetic_eval_set(
            np.zeros((4, 6), dtype=np.uint8),
            np.zeros((4, 1), dtype=np.uint8),
            key=MEMORY_D3,
            observable_names=("logical_observable",),
        )

    def test_non_uint8_shots_raise(self, eval_set: EvalSet) -> None:
        with pytest.raises(ValueError, match="syndromes must be a 2-D uint8"):
            replace(eval_set, syndromes=eval_set.syndromes.astype(np.float32))

    def test_one_dimensional_observables_raise(self, eval_set: EvalSet) -> None:
        with pytest.raises(ValueError, match="observables must be a 2-D uint8"):
            replace(eval_set, observables=eval_set.observables[:, 0])

    def test_no_shots_raise(self, eval_set: EvalSet) -> None:
        with pytest.raises(ValueError, match="N >= 1"):
            replace(
                eval_set,
                syndromes=eval_set.syndromes[:0],
                observables=eval_set.observables[:0],
            )

    def test_shot_counts_disagreeing_raise(self, eval_set: EvalSet) -> None:
        with pytest.raises(ValueError, match="4 shot.*3"):
            replace(eval_set, observables=eval_set.observables[:3])

    def test_observable_names_disagreeing_with_columns_raise(
        self, eval_set: EvalSet
    ) -> None:
        with pytest.raises(ValueError, match="1 observable.*'Z_0', 'Z_1'"):
            replace(eval_set, observable_names=("Z_0", "Z_1"))

    def test_metadata_at_another_distance_raises(self, eval_set: EvalSet) -> None:
        with pytest.raises(ValueError, match=r"\(5, 3\)"):
            replace(
                eval_set,
                circuit_metadata=replace(eval_set.circuit_metadata, distance=5),
            )


class TestGNNDecoderRepresentationValidation:
    """GNNDecoder validates contract representation against metadata."""

    def test_phased_contract_rejects_spatial_metadata(self) -> None:
        """A phased checkpoint cannot be scored on spatial-only metadata."""
        rng = np.random.default_rng(42)
        n_det = 10
        metadata = CircuitMetadata(
            detector_coords=rng.random((n_det, 3)),
            distance=3,
            rounds=3,
            num_detectors=n_det,
            dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
        )

        phased_contract = DataContract(
            representation=PHASED,
            labels=LabelSpec(
                num_observables=1,
                observable_names=("logical_observable",),
            ),
        )
        model = build_model(
            node_dim=PHASED.node_dim,
            edge_dim=PHASED.edge_dim,
            hidden_dim=32,
            num_layers=2,
            num_observables=1,
            dropout=0.0,
        )

        with pytest.raises(ValueError, match="lacks phased fields"):
            GNNDecoder(
                model=model,
                metadata=metadata,
                contract=phased_contract,
            )

    def test_spatial_contract_accepts_spatial_metadata(self) -> None:
        """A spatial checkpoint works with spatial-only metadata."""
        rng = np.random.default_rng(42)
        n_det = 10
        metadata = CircuitMetadata(
            detector_coords=rng.random((n_det, 3)),
            distance=3,
            rounds=3,
            num_detectors=n_det,
            dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
        )
        model = build_model(
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            node_dim=6,
            edge_dim=6,
            num_observables=1,
        )

        decoder = GNNDecoder(
            model=model,
            metadata=metadata,
            contract=resolve_profile("memory").data_contract,
        )
        assert decoder.name == "gnn"

    def test_spatial_contract_accepts_phased_metadata(self) -> None:
        """A spatial checkpoint can be scored on phased metadata (downgrade)."""
        rng = np.random.default_rng(42)
        n_det = 10
        metadata = CircuitMetadata(
            detector_coords=rng.random((n_det, 3)),
            distance=3,
            rounds=3,
            num_detectors=n_det,
            dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
            layout=LogicalOperationLayout(
                phase_ids=np.zeros(n_det, dtype=np.intp),
                phase_names=("memory",),
                patch_ids=np.zeros(n_det, dtype=np.intp),
                seam_mask=np.zeros(n_det, dtype=np.bool_),
            ),
        )
        model = build_model(
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
            node_dim=6,
            edge_dim=6,
            num_observables=1,
        )

        decoder = GNNDecoder(
            model=model,
            metadata=metadata,
            contract=resolve_profile("memory").data_contract,
        )
        assert decoder.name == "gnn"
