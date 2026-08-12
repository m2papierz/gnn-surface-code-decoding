"""Tests for decoding heads and QECDecoder."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from model.decoder import LogicalHead, build_model


def _make_graph(num_nodes: int = 6, num_edges: int = 10) -> Data:
    """Create a minimal detector graph for testing (enriched features)."""
    return Data(
        x=torch.rand(num_nodes, 6),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.rand(num_edges, 6),
        y=torch.zeros(1),
        logical=torch.zeros(1),
        setting_id=torch.tensor(0),
    )


@pytest.fixture
def single_graph() -> Data:
    return _make_graph(num_nodes=6, num_edges=10)


@pytest.fixture
def batched() -> Batch:
    """Batch of 3 graphs with different sizes."""
    g1 = _make_graph(num_nodes=6, num_edges=10)
    g2 = _make_graph(num_nodes=10, num_edges=18)
    g3 = _make_graph(num_nodes=8, num_edges=14)
    return Batch.from_data_list([g1, g2, g3])


class TestLogicalHead:
    """Tests for graph-level observable prediction."""

    def test_output_shape_single_observable(self) -> None:
        head = LogicalHead(hidden_dim=32, num_observables=1, dropout=0.0)
        h = torch.randn(18, 32)
        batch = torch.tensor([0] * 6 + [1] * 6 + [2] * 6)
        logits = head(h, batch)
        assert logits.shape == (3, 1)

    def test_output_shape_multi_observable(self) -> None:
        head = LogicalHead(hidden_dim=32, num_observables=3, dropout=0.0)
        h = torch.randn(10, 32)
        batch = torch.tensor([0] * 4 + [1] * 6)
        logits = head(h, batch)
        assert logits.shape == (2, 3)

    def test_ignores_extra_kwargs(self) -> None:
        head = LogicalHead(hidden_dim=16, dropout=0.0)
        h = torch.randn(5, 16)
        batch = torch.zeros(5, dtype=torch.long)
        logits = head(h, batch, edge_index=None, edge_h=None, edge_attr=None)
        assert logits.shape == (1, 1)

    def test_sum_pooling_gradient_flow(self) -> None:
        """Sum pooling guarantees every node receives gradient."""
        head = LogicalHead(hidden_dim=16, dropout=0.0)
        h = torch.randn(8, 16, requires_grad=True)
        batch = torch.tensor([0] * 4 + [1] * 4)
        logits = head(h, batch)
        logits.sum().backward()
        assert h.grad is not None
        per_node_grad_norm = h.grad.abs().sum(dim=-1)
        assert (per_node_grad_norm > 0).all()


class TestQECDecoder:
    """Tests for the full encoder + head pipeline."""

    def test_output_shape(self, batched: Batch) -> None:
        model = build_model(
            node_dim=6,
            edge_dim=6,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            logits = model(batched)
        assert logits.shape == (3, 1)

    def test_backward_pass(self, batched: Batch) -> None:
        model = build_model(
            node_dim=6,
            edge_dim=6,
            hidden_dim=32,
            num_layers=2,
            dropout=0.0,
        )
        logits = model(batched)
        logits.sum().backward()
        last = model.encoder.num_layers - 1
        for name, param in model.named_parameters():
            # LogicalHead pools only node embeddings, so the final
            # layer's edge_update and edge_norm have no gradient path.
            if f"layers.{last}.edge_update" in name:
                assert param.grad is None, f"Unexpected gradient for {name}"
                continue
            if f"layers.{last}.edge_norm" in name:
                assert param.grad is None, f"Unexpected gradient for {name}"
                continue
            assert param.grad is not None, f"No gradient for {name}"
