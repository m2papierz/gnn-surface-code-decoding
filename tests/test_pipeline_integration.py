"""End-to-end pipeline integration tests using the committed CI shard.

Loads pre-sampled syndromes and labels from ``data/ci_shard/memory/`` - no GPU
and no Stim dependency required.  Exercises the full path:
shard => graph builder => PyG batching => model forward => loss backward.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from model.decoder import QECDecoder, build_model
from sampling.graph import (
    EDGE_DIM,
    NODE_DIM,
    CircuitMetadata,
    build_fired_detector_graph,
)
from training import FocalBCEWithLogitsLoss


CI_SHARD_DIR = Path(__file__).resolve().parent.parent / "data" / "ci_shard" / "memory"


@pytest.fixture(scope="module")
def shard() -> dict[str, np.ndarray | dict]:
    """Load CI shard arrays and manifest once per module."""
    manifest_path = CI_SHARD_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("CI shard not found - run scripts/generate_ci_shard_memory.py")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    syndromes = np.load(CI_SHARD_DIR / "syndromes.npy")
    observables = np.load(CI_SHARD_DIR / "observables.npy")
    detector_coords = np.load(CI_SHARD_DIR / "detector_coords.npy")

    assert syndromes.shape == tuple(manifest["syndromes_shape"])
    assert observables.shape == tuple(manifest["observables_shape"])

    return {
        "syndromes": syndromes,
        "observables": observables,
        "detector_coords": detector_coords,
        "manifest": manifest,
    }


@pytest.fixture(scope="module")
def circuit_metadata(shard: dict) -> CircuitMetadata:
    """Reconstruct CircuitMetadata from stored detector coordinates."""
    m = shard["manifest"]
    n_det = m["num_detectors"]
    return CircuitMetadata(
        detector_coords=shard["detector_coords"],
        distance=m["distance"],
        rounds=m["rounds"],
        num_detectors=n_det,
        dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
    )


@pytest.fixture(scope="module")
def shard_graphs(
    shard: dict,
    circuit_metadata: CircuitMetadata,
) -> list[Data]:
    """Build graphs and PyG Data objects for the full shard."""
    syndromes = shard["syndromes"]
    observables = shard["observables"]
    data_list: list[Data] = []

    for i in range(syndromes.shape[0]):
        graph = build_fired_detector_graph(syndromes[i], circuit_metadata)
        data = Data(
            x=torch.from_numpy(graph.node_features),
            edge_index=torch.from_numpy(graph.edge_index),
            edge_attr=torch.from_numpy(graph.edge_features),
            y=torch.from_numpy(observables[i].astype(np.float32)),
            num_fired=torch.tensor(graph.num_fired, dtype=torch.long),
        )
        data_list.append(data)

    return data_list


class TestCIShardIntegrity:
    """Verify the committed shard is well-formed and matches its manifest."""

    def test_manifest_fields(self, shard: dict) -> None:
        m = shard["manifest"]
        required = {
            "circuit_file",
            "circuit_sha256",
            "stim_version",
            "seed",
            "num_shots",
            "distance",
            "rounds",
            "error_prob",
            "num_detectors",
            "num_observables",
            "positive_count",
            "generation_command",
        }
        assert required.issubset(m.keys())

    def test_shapes_match_manifest(self, shard: dict) -> None:
        m = shard["manifest"]
        assert shard["syndromes"].shape == (m["num_shots"], m["num_detectors"])
        assert shard["observables"].shape == (m["num_shots"], m["num_observables"])
        assert shard["detector_coords"].shape[0] == m["num_detectors"]
        assert shard["detector_coords"].shape[1] >= 3

    def test_binary_values(self, shard: dict) -> None:
        assert set(np.unique(shard["syndromes"])).issubset({0, 1})
        assert set(np.unique(shard["observables"])).issubset({0, 1})

    def test_positive_count(self, shard: dict) -> None:
        m = shard["manifest"]
        actual = int(shard["observables"].any(axis=1).sum())
        assert actual == m["positive_count"]
        assert actual > 0, "shard must contain at least one logical error"


class TestGraphBuilding:
    """Graph builder on shard syndromes - no Stim required."""

    def test_all_graphs_have_valid_shapes(self, shard_graphs: list[Data]) -> None:
        for data in shard_graphs:
            n = data.num_fired.item()
            e = n * (n - 1) if n > 1 else 0
            assert data.x.shape == (n, NODE_DIM)
            assert data.edge_index.shape == (2, e)
            assert data.edge_attr.shape == (e, EDGE_DIM)

    def test_empty_syndromes_present(self, shard_graphs: list[Data]) -> None:
        """At p=0.01, d=3, many shots should have zero fired detectors."""
        empties = [d for d in shard_graphs if d.num_fired.item() == 0]
        assert len(empties) > 0, "expected some empty-syndrome samples at p=0.01"

    def test_nonempty_syndromes_present(self, shard_graphs: list[Data]) -> None:
        nonempty = [d for d in shard_graphs if d.num_fired.item() > 0]
        assert len(nonempty) > 0, "expected some fired-detector samples"

    def test_node_features_finite(self, shard_graphs: list[Data]) -> None:
        for data in shard_graphs:
            if data.x.numel() > 0:
                assert torch.isfinite(data.x).all()

    def test_edge_features_finite(self, shard_graphs: list[Data]) -> None:
        for data in shard_graphs:
            if data.edge_attr.numel() > 0:
                assert torch.isfinite(data.edge_attr).all()


class TestEndToEndForwardBackward:
    """Full pipeline: shard => batch => model forward => loss => backward."""

    @pytest.fixture()
    def model(self) -> QECDecoder:
        return build_model(
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            hidden_dim=32,
            num_layers=2,
            num_observables=1,
            dropout=0.0,
        )

    @pytest.fixture()
    def criterion(self) -> FocalBCEWithLogitsLoss:
        return FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0)

    def _make_batch(self, data_list: list[Data], max_items: int = 32) -> Batch:
        """Build a PyG batch from the first max_items nonempty graphs."""
        nonempty = [d for d in data_list if d.num_fired.item() > 0][:max_items]
        assert len(nonempty) > 0
        return Batch.from_data_list(nonempty)

    def test_forward_produces_logits(
        self,
        model: QECDecoder,
        shard_graphs: list[Data],
    ) -> None:
        batch = self._make_batch(shard_graphs)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (batch.num_graphs, 1)
        assert torch.isfinite(logits).all()

    def test_backward_updates_parameters(
        self,
        model: QECDecoder,
        criterion: FocalBCEWithLogitsLoss,
        shard_graphs: list[Data],
    ) -> None:
        batch = self._make_batch(shard_graphs)
        model.train()

        logits = model(batch)
        loss = criterion(logits.view(-1), batch.y)

        loss.backward()

        grads = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(grads) > 0, "no gradients computed"
        assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"

    def test_loss_is_scalar_and_positive(
        self,
        model: QECDecoder,
        criterion: FocalBCEWithLogitsLoss,
        shard_graphs: list[Data],
    ) -> None:
        batch = self._make_batch(shard_graphs)
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            loss = criterion(logits.view(-1), batch.y)
        assert loss.dim() == 0
        assert loss.item() > 0.0

    def test_multiple_batches_independent(
        self,
        model: QECDecoder,
        criterion: FocalBCEWithLogitsLoss,
        shard_graphs: list[Data],
    ) -> None:
        """Two batches from different slices produce independent losses."""
        nonempty = [d for d in shard_graphs if d.num_fired.item() > 0]
        if len(nonempty) < 8:
            pytest.skip("not enough nonempty graphs for two batches")

        mid = len(nonempty) // 2
        batch_a = Batch.from_data_list(nonempty[:mid])
        batch_b = Batch.from_data_list(nonempty[mid:])

        model.eval()
        with torch.no_grad():
            loss_a = criterion(model(batch_a).view(-1), batch_a.y)
            loss_b = criterion(model(batch_b).view(-1), batch_b.y)

        assert loss_a.item() != pytest.approx(loss_b.item(), abs=1e-8)

    def test_mixed_empty_and_nonempty_batch(
        self,
        model: QECDecoder,
        shard_graphs: list[Data],
    ) -> None:
        """Batch mixing empty (num_fired=0) and nonempty graphs.

        Empty graphs contribute zero nodes to the batch vector.
        ``LogicalHead`` uses ``num_graphs`` to produce output rows for all
        graphs, so empty-graph rows should get zero-pooling predictions.
        """
        empties = [d for d in shard_graphs if d.num_fired.item() == 0][:4]
        nonempty = [d for d in shard_graphs if d.num_fired.item() > 0][:4]
        if not empties or not nonempty:
            pytest.skip("shard needs both empty and nonempty syndromes")

        mixed = nonempty[:2] + empties[:2] + nonempty[2:]
        batch = Batch.from_data_list(mixed)

        model.eval()
        with torch.no_grad():
            logits = model(batch)

        assert logits.shape == (len(mixed), 1)
        assert torch.isfinite(logits).all()

    def test_gradient_reaches_all_encoder_layers(
        self,
        model: QECDecoder,
        criterion: FocalBCEWithLogitsLoss,
        shard_graphs: list[Data],
    ) -> None:
        batch = self._make_batch(shard_graphs)
        model.train()
        model.zero_grad()

        logits = model(batch)
        loss = criterion(logits.view(-1), batch.y)
        loss.backward()

        last_layer_idx = model.encoder.num_layers - 1
        last_edge_prefix = f"encoder.layers.{last_layer_idx}.edge_"

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # LogicalHead does not consume edge embeddings, so the last
            # layer's edge_update / edge_norm have no gradient path.
            if name.startswith(last_edge_prefix):
                continue
            assert param.grad is not None, f"no gradient for {name}"
            assert param.grad.abs().sum() > 0, f"zero gradient for {name}"
