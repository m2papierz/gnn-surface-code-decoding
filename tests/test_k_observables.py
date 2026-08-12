"""Tests for K-observable plumbing end to end.

Verifies that the observable count travels from the data contract through
model construction, training primitives, checkpoint I/O, decoder allocation,
and threshold sweep - and that K=1 reproduces the pre-existing single-
observable path exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from decoders import GNNDecoder
from model.decoder import build_model
from sampling.graph import CircuitMetadata
from sampling.representation import (
    SPATIAL,
    SPATIAL_MEMORY,
    DataContract,
    LabelSpec,
)
from training.loss import FocalBCEWithLogitsLoss
from training.primitives import (
    TrainingState,
    build_training_state,
    compute_val_metrics,
    load_checkpoint,
    sweep_threshold,
    write_checkpoint,
)


SPATIAL_K2 = DataContract(
    representation=SPATIAL,
    labels=LabelSpec(
        num_observables=2,
        observable_names=("obs_XX_XI", "obs_ZI_ZI"),
    ),
)


def _make_metadata(n_det: int = 24) -> CircuitMetadata:
    rng = np.random.default_rng(0)
    return CircuitMetadata(
        detector_coords=rng.random((n_det, 3)).astype(np.float64),
        distance=3,
        rounds=3,
        num_detectors=n_det,
        dem_edge_weights=np.zeros((n_det, n_det), dtype=np.float64),
    )


def _make_batch(
    n_graphs: int, num_observables: int, *, node_dim: int = 6, edge_dim: int = 6
) -> Batch:
    """Synthetic batch with K-vector labels."""
    data_list: list[Data] = []
    for _ in range(n_graphs):
        n = np.random.randint(3, 8)
        e = np.random.randint(4, 12)
        data_list.append(
            Data(
                x=torch.randn(n, node_dim),
                edge_index=torch.randint(0, n, (2, e)),
                edge_attr=torch.randn(e, edge_dim),
                y=torch.randint(0, 2, (num_observables,)).float(),
                num_fired=torch.tensor(n, dtype=torch.long),
            )
        )
    return Batch.from_data_list(data_list)


def _make_training_state(
    contract: DataContract, device: torch.device | None = None
) -> TrainingState:
    """Minimal training state for testing."""
    from training.config import TrainConfig

    device = device or torch.device("cpu")
    cfg = TrainConfig(
        sample_budget=1000,
        batch_size=64,
        val_interval_samples=500,
        val_size=100,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        num_workers=0,
    )
    return build_training_state(
        cfg, contract=contract, total_budget=1000, device=device
    )


class TestDecodeBatchAllocation:
    """GNNDecoder.decode_batch output shape matches K from the contract."""

    def test_k1_shape(self) -> None:
        meta = _make_metadata()
        model = build_model(hidden_dim=16, num_layers=2, dropout=0.0, num_observables=1)
        decoder = GNNDecoder.from_metadata(
            model=model,
            metadata=meta,
            contract=SPATIAL_MEMORY,
            device=torch.device("cpu"),
        )
        syndromes = np.zeros((10, 24), dtype=np.uint8)
        result = decoder.decode_batch(syndromes)
        assert result.shape == (10, 1)

    def test_k2_shape(self) -> None:
        meta = _make_metadata()
        model = build_model(hidden_dim=16, num_layers=2, dropout=0.0, num_observables=2)
        decoder = GNNDecoder.from_metadata(
            model=model,
            metadata=meta,
            contract=SPATIAL_K2,
            device=torch.device("cpu"),
        )
        syndromes = np.zeros((10, 24), dtype=np.uint8)
        result = decoder.decode_batch(syndromes)
        assert result.shape == (10, 2)

    def test_k2_nonempty_syndromes(self) -> None:
        meta = _make_metadata()
        model = build_model(hidden_dim=16, num_layers=2, dropout=0.0, num_observables=2)
        decoder = GNNDecoder.from_metadata(
            model=model,
            metadata=meta,
            contract=SPATIAL_K2,
            device=torch.device("cpu"),
            batch_size=4,
        )
        rng = np.random.default_rng(42)
        syndromes = rng.integers(0, 2, size=(10, 24), dtype=np.uint8)
        result = decoder.decode_batch(syndromes)
        assert result.shape == (10, 2)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1})


class TestCheckpointKValidation:
    """load_checkpoint rejects mismatched num_observables."""

    def test_mismatched_num_observables_raises(self, tmp_path) -> None:
        state_k1 = _make_training_state(SPATIAL_MEMORY)
        ckpt_path = tmp_path / "ckpt_k1.pt"
        write_checkpoint(
            ckpt_path,
            state_k1,
            samples_consumed=100,
            best_metric=0.5,
            decision_threshold=0.0,
            contract=SPATIAL_MEMORY,
            cfg=state_k1._cfg if hasattr(state_k1, "_cfg") else _default_cfg(),
        )

        state_k2 = _make_training_state(SPATIAL_K2)
        with pytest.raises(ValueError, match="Observable count mismatch"):
            load_checkpoint(
                ckpt_path,
                state_k2,
                _default_cfg(),
                contract=SPATIAL_K2,
            )

    def test_matching_num_observables_passes(self, tmp_path) -> None:
        state = _make_training_state(SPATIAL_MEMORY)
        ckpt_path = tmp_path / "ckpt.pt"
        write_checkpoint(
            ckpt_path,
            state,
            samples_consumed=100,
            best_metric=0.5,
            decision_threshold=0.0,
            contract=SPATIAL_MEMORY,
            cfg=_default_cfg(),
        )
        ckpt = load_checkpoint(
            ckpt_path,
            state,
            _default_cfg(),
            contract=SPATIAL_MEMORY,
        )
        assert ckpt["samples_consumed"] == 100


def _default_cfg():
    from training.config import TrainConfig

    return TrainConfig(
        sample_budget=1000,
        batch_size=64,
        val_interval_samples=500,
        val_size=100,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        num_workers=0,
    )


class TestLossKObservables:
    """Focal loss aggregates per-element across all observables."""

    def test_loss_k1_matches_scalar(self) -> None:
        """K=1: .view(-1) on (B,1) logits matches inherently-1D input."""
        loss_fn = FocalBCEWithLogitsLoss(alpha=0.75, gamma=1.0)
        logits_2d = torch.randn(8, 1)
        targets_2d = torch.randint(0, 2, (8, 1)).float()
        logits_1d = logits_2d.squeeze(1)
        targets_1d = targets_2d.squeeze(1)
        loss_from_2d = loss_fn(logits_2d.view(-1), targets_2d.view(-1))
        loss_from_1d = loss_fn(logits_1d, targets_1d)
        assert loss_from_2d.item() == pytest.approx(loss_from_1d.item())

    def test_loss_k2_element_wise(self) -> None:
        """K=2: loss treats each (graph, observable) pair independently."""
        loss_fn = FocalBCEWithLogitsLoss(alpha=0.75, gamma=1.0)
        logits = torch.randn(4, 2)
        targets = torch.randint(0, 2, (4, 2)).float()
        loss = loss_fn(logits.view(-1), targets.view(-1))
        assert loss.item() > 0.0
        assert torch.isfinite(loss)

    def test_loss_gradient_all_elements(self) -> None:
        """Gradient flows to every element for K>1."""
        loss_fn = FocalBCEWithLogitsLoss(alpha=0.75, gamma=1.0)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 2, (4, 3)).float()
        loss = loss_fn(logits.view(-1), targets.view(-1))
        loss.backward()
        assert logits.grad is not None
        assert (logits.grad.abs() > 0).all()


class TestSweepThresholdK:
    """sweep_threshold produces one global scalar for all K observables."""

    def test_k2_returns_scalar(self) -> None:
        state = _make_training_state(SPATIAL_K2)
        batches = [_make_batch(8, num_observables=2)]
        for b in batches:
            b.to(torch.device("cpu"))
        thr = sweep_threshold(state, batches)
        assert isinstance(thr, float)

    def test_k1_returns_scalar(self) -> None:
        state = _make_training_state(SPATIAL_MEMORY)
        batches = [_make_batch(8, num_observables=1)]
        thr = sweep_threshold(state, batches)
        assert isinstance(thr, float)


class TestValMetricsK:
    """compute_val_metrics uses any(dim=1) for multi-observable LER."""

    def test_k2_metrics(self) -> None:
        state = _make_training_state(SPATIAL_K2)
        batches = [_make_batch(8, num_observables=2)]
        metrics = compute_val_metrics(state, batches)
        assert "loss" in metrics
        assert "ler" in metrics
        assert 0.0 <= metrics["ler"] <= 1.0

    def test_k1_metrics(self) -> None:
        state = _make_training_state(SPATIAL_MEMORY)
        batches = [_make_batch(8, num_observables=1)]
        metrics = compute_val_metrics(state, batches)
        assert "loss" in metrics
        assert "ler" in metrics
        assert 0.0 <= metrics["ler"] <= 1.0
