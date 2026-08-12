"""Tests for representation-aware evaluation path (LS08a).

Verifies that EvalSet carries CircuitMetadata, that the loader reads phased
metadata from npz files, that GNNDecoder rejects mismatched representations,
and that the spatial path remains byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from decoders import GNNDecoder
from evaluation.evaluator import EvalSet, load_eval_set
from model.decoder import build_model
from sampling.graph import CircuitMetadata
from sampling.representation import (
    PHASED,
    SPATIAL_MEMORY,
    DataContract,
    LabelSpec,
)


CI_SHARD_DIR = Path(__file__).resolve().parent.parent / "data" / "ci_shard" / "memory"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "memory"
CIRCUIT_DIR = Path(__file__).resolve().parent.parent / "data" / "circuits" / "memory"


class TestEvalSetCircuitMetadata:
    """EvalSet builds CircuitMetadata at construction."""

    def test_spatial_eval_set_has_metadata(self) -> None:
        rng = np.random.default_rng(42)
        n, n_det = 100, 10
        coords = rng.random((n_det, 3))

        es = EvalSet(
            syndromes=rng.integers(0, 2, size=(n, n_det), dtype=np.uint8),
            observables=rng.integers(0, 2, size=(n, 1), dtype=np.uint8),
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

        assert isinstance(es.circuit_metadata, CircuitMetadata)
        assert es.circuit_metadata.distance == 3
        assert es.circuit_metadata.rounds == 3
        assert es.circuit_metadata.num_detectors == n_det
        np.testing.assert_array_equal(es.circuit_metadata.detector_coords, coords)
        assert es.circuit_metadata.phase_ids is None

    def test_phased_eval_set_has_metadata(self) -> None:
        rng = np.random.default_rng(42)
        n, n_det = 100, 10
        coords = rng.random((n_det, 3))
        phase_ids = np.zeros(n_det, dtype=np.intp)
        phase_ids[5:] = 1
        patch_ids = np.zeros(n_det, dtype=np.intp)
        patch_ids[5:] = 1
        seam_mask = np.zeros(n_det, dtype=np.bool_)
        seam_mask[4:6] = True
        phase_names = ("memory", "merge")

        es = EvalSet(
            syndromes=rng.integers(0, 2, size=(n, n_det), dtype=np.uint8),
            observables=rng.integers(0, 2, size=(n, 1), dtype=np.uint8),
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
            phase_ids=phase_ids,
            phase_names=phase_names,
            patch_ids=patch_ids,
            seam_mask=seam_mask,
        )

        assert es.circuit_metadata.phase_ids is not None
        np.testing.assert_array_equal(es.circuit_metadata.phase_ids, phase_ids)
        assert es.circuit_metadata.phase_names == phase_names
        np.testing.assert_array_equal(es.circuit_metadata.patch_ids, patch_ids)
        np.testing.assert_array_equal(es.circuit_metadata.seam_mask, seam_mask)

    def test_ci_shard_metadata_matches(self) -> None:
        """CI shard's circuit_metadata matches hand-built metadata."""
        if not (CI_SHARD_DIR / "manifest.json").exists():
            pytest.skip("CI shard not found")

        es = load_eval_set(CI_SHARD_DIR)
        md = es.circuit_metadata

        assert md.distance == es.distance
        assert md.rounds == es.rounds
        assert md.num_detectors == es.syndromes.shape[1]
        np.testing.assert_array_equal(md.detector_coords, es.detector_coords)
        assert md.phase_ids is None


# load_eval_set reads phased metadata from npz


class TestLoadEvalSetPhasedMetadata:
    """Loader reads phased arrays from npz and validates representation."""

    def test_loads_phased_npz(self, tmp_path: Path) -> None:
        n, n_det = 50, 8
        rng = np.random.default_rng(99)
        syndromes = rng.integers(0, 2, size=(n, n_det), dtype=np.uint8)
        observables = rng.integers(0, 2, size=(n, 1), dtype=np.uint8)
        coords = rng.random((n_det, 3))
        phase_ids = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.intp)
        patch_ids = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.intp)
        seam_mask = np.array(
            [False, False, False, True, True, False, False, False], dtype=np.bool_
        )

        np.savez_compressed(
            tmp_path / "data.npz",
            syndromes=syndromes,
            observables=observables,
            detector_coords=coords,
            phase_ids=phase_ids,
            patch_ids=patch_ids,
            seam_mask=seam_mask,
        )
        manifest = {
            "distance": 3,
            "rounds": 5,
            "error_prob": 0.005,
            "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
            "representation_version": "phased",
            "observable_names": ["logical_observable"],
            "phase_names": ["memory", "merge", "split"],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        es = load_eval_set(tmp_path)
        assert es.circuit_metadata.phase_ids is not None
        np.testing.assert_array_equal(es.circuit_metadata.phase_ids, phase_ids)
        assert es.circuit_metadata.phase_names == ("memory", "merge", "split")
        np.testing.assert_array_equal(es.circuit_metadata.patch_ids, patch_ids)
        np.testing.assert_array_equal(es.circuit_metadata.seam_mask, seam_mask)

    def test_spatial_npz_has_no_phased_fields(self, tmp_path: Path) -> None:
        """A spatial eval set loads without phased arrays."""
        n, n_det = 20, 4
        rng = np.random.default_rng(7)
        np.savez_compressed(
            tmp_path / "data.npz",
            syndromes=rng.integers(0, 2, size=(n, n_det), dtype=np.uint8),
            observables=rng.integers(0, 2, size=(n, 1), dtype=np.uint8),
            detector_coords=rng.random((n_det, 3)),
        )
        manifest = {
            "distance": 3,
            "rounds": 3,
            "error_prob": 0.01,
            "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        es = load_eval_set(tmp_path)
        assert es.circuit_metadata.phase_ids is None

    def test_rejects_phased_representation_without_data(self, tmp_path: Path) -> None:
        """A manifest declaring phased representation but no phased arrays raises."""
        n, n_det = 20, 4
        rng = np.random.default_rng(7)
        np.savez_compressed(
            tmp_path / "data.npz",
            syndromes=rng.integers(0, 2, size=(n, n_det), dtype=np.uint8),
            observables=rng.integers(0, 2, size=(n, 1), dtype=np.uint8),
            detector_coords=rng.random((n_det, 3)),
        )
        manifest = {
            "distance": 3,
            "rounds": 3,
            "error_prob": 0.01,
            "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
            "representation_version": "phased",
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="lacks the phased metadata"):
            load_eval_set(tmp_path)

    def test_rejects_phase_ids_without_phase_names(self, tmp_path: Path) -> None:
        """Phase arrays in npz but no phase_names in manifest raises."""
        n, n_det = 20, 4
        rng = np.random.default_rng(7)
        np.savez_compressed(
            tmp_path / "data.npz",
            syndromes=rng.integers(0, 2, size=(n, n_det), dtype=np.uint8),
            observables=rng.integers(0, 2, size=(n, 1), dtype=np.uint8),
            detector_coords=rng.random((n_det, 3)),
            phase_ids=np.zeros(n_det, dtype=np.intp),
            patch_ids=np.zeros(n_det, dtype=np.intp),
            seam_mask=np.zeros(n_det, dtype=np.bool_),
        )
        manifest = {
            "distance": 3,
            "rounds": 3,
            "error_prob": 0.01,
            "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="phase_names"):
            load_eval_set(tmp_path)


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
        model = build_model(hidden_dim=32, num_layers=2, dropout=0.0)

        decoder = GNNDecoder(
            model=model,
            metadata=metadata,
            contract=SPATIAL_MEMORY,
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
            phase_ids=np.zeros(n_det, dtype=np.intp),
            phase_names=("memory",),
            patch_ids=np.zeros(n_det, dtype=np.intp),
            seam_mask=np.zeros(n_det, dtype=np.bool_),
        )
        model = build_model(hidden_dim=32, num_layers=2, dropout=0.0)

        decoder = GNNDecoder(
            model=model,
            metadata=metadata,
            contract=SPATIAL_MEMORY,
        )
        assert decoder.name == "gnn"


class TestSpatialPathIdentity:
    """The spatial path produces identical results with circuit_metadata."""

    def test_ci_shard_gnn_decode_unchanged(self) -> None:
        """GNN decode on CI shard via circuit_metadata matches direct metadata."""
        if not (CI_SHARD_DIR / "manifest.json").exists():
            pytest.skip("CI shard not found")

        es = load_eval_set(CI_SHARD_DIR)

        # Build metadata the old way (hand-constructed)
        n_det = es.syndromes.shape[1]
        old_metadata = CircuitMetadata(
            detector_coords=es.detector_coords,
            distance=es.distance,
            rounds=es.rounds,
            num_detectors=n_det,
            dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
        )

        model = build_model(hidden_dim=32, num_layers=2, dropout=0.0)
        torch.manual_seed(0)

        decoder_old = GNNDecoder.from_metadata(
            model=model,
            metadata=old_metadata,
            threshold=0.0,
            device=torch.device("cpu"),
            batch_size=64,
        )
        result_old = decoder_old.decode_batch(es.syndromes[:20])

        decoder_new = GNNDecoder.from_metadata(
            model=model,
            metadata=es.circuit_metadata,
            threshold=0.0,
            device=torch.device("cpu"),
            batch_size=64,
        )
        result_new = decoder_new.decode_batch(es.syndromes[:20])

        np.testing.assert_array_equal(result_old, result_new)


# generate_eval_sets stores representation metadata


class TestEvalSetManifestRepresentation:
    """Eval set manifests carry representation_version and observable_names."""

    def test_existing_memory_manifest_loads_as_spatial(self) -> None:
        """An existing memory eval set (no representation_version) defaults spatial."""
        sample_dir = EVAL_DIR / "d3_p0_0100"
        if not sample_dir.exists():
            pytest.skip("Eval set d3_p0_0100 not found")

        es = load_eval_set(sample_dir)
        assert es.manifest.get("representation_version", "spatial") == "spatial"
        assert es.circuit_metadata.phase_ids is None
