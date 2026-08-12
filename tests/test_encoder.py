"""Tests for the detector graph encoder."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from model.encoder import DetectorGraphEncoder
from sampling.graph import EDGE_DIM, NODE_DIM


@pytest.fixture
def encoder() -> DetectorGraphEncoder:
    """Small encoder for fast tests."""
    return DetectorGraphEncoder(
        node_dim=1,
        edge_dim=2,
        hidden_dim=32,
        num_layers=3,
        dropout=0.0,
    )


@pytest.fixture
def small_graph() -> Data:
    """Minimal detector graph: 5 detector nodes + 1 boundary, 8 edges."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    x = torch.tensor([[1], [0], [1], [0], [0], [0]], dtype=torch.float32)
    edge_attr = torch.rand(8, 2)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TestOutputShapes:
    """Verify encoder produces correctly shaped embeddings."""

    def test_single_graph(
        self,
        encoder: DetectorGraphEncoder,
        small_graph: Data,
    ) -> None:
        h, edge_h = encoder(
            small_graph.x, small_graph.edge_index, small_graph.edge_attr
        )
        assert h.shape == (6, 32)
        assert edge_h.shape == (8, 32)

    def test_hidden_dim_propagates(self) -> None:
        enc = DetectorGraphEncoder(hidden_dim=64, num_layers=2)
        x = torch.randn(10, NODE_DIM)
        ei = torch.randint(0, 10, (2, 20))
        ea = torch.randn(20, EDGE_DIM)
        h, edge_h = enc(x, ei, ea)
        assert h.shape == (10, 64)
        assert edge_h.shape == (20, 64)

    def test_batched_graphs(
        self,
        encoder: DetectorGraphEncoder,
        small_graph: Data,
    ) -> None:
        batch = Batch.from_data_list([small_graph, small_graph, small_graph])
        h, edge_h = encoder(batch.x, batch.edge_index, batch.edge_attr)
        assert h.shape == (18, 32)
        assert edge_h.shape == (24, 32)


class TestGradients:
    """Verify gradients flow through the encoder."""

    def test_backward_pass(
        self,
        encoder: DetectorGraphEncoder,
        small_graph: Data,
    ) -> None:
        h, edge_h = encoder(
            small_graph.x, small_graph.edge_index, small_graph.edge_attr
        )
        loss = h.sum() + edge_h.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_node_features_receive_grad(
        self,
        encoder: DetectorGraphEncoder,
        small_graph: Data,
    ) -> None:
        x = small_graph.x.clone().requires_grad_(True)
        h, edge_h = encoder(x, small_graph.edge_index, small_graph.edge_attr)
        (h.sum() + edge_h.sum()).backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestConfiguration:
    """Verify constructor validation and edge cases."""

    def test_single_layer(self) -> None:
        enc = DetectorGraphEncoder(num_layers=1, hidden_dim=16)
        x = torch.randn(4, NODE_DIM)
        ei = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        ea = torch.randn(3, EDGE_DIM)
        h, edge_h = enc(x, ei, ea)
        assert h.shape == (4, 16)
        assert edge_h.shape == (3, 16)

    def test_custom_input_dims(self) -> None:
        enc = DetectorGraphEncoder(
            node_dim=3,
            edge_dim=6,
            hidden_dim=16,
            num_layers=2,
        )
        x = torch.randn(8, 3)
        ei = torch.randint(0, 8, (2, 12))
        ea = torch.randn(12, 6)
        h, edge_h = enc(x, ei, ea)
        assert h.shape == (8, 16)
        assert edge_h.shape == (12, 16)

    def test_invalid_num_layers(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            DetectorGraphEncoder(num_layers=0)

    def test_invalid_hidden_dim(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim"):
            DetectorGraphEncoder(hidden_dim=0)


class TestDeterminism:
    """Verify reproducible outputs in eval mode."""

    def test_eval_deterministic(
        self,
        encoder: DetectorGraphEncoder,
        small_graph: Data,
    ) -> None:
        encoder.eval()
        with torch.no_grad():
            h1, e1 = encoder(
                small_graph.x, small_graph.edge_index, small_graph.edge_attr
            )
            h2, e2 = encoder(
                small_graph.x, small_graph.edge_index, small_graph.edge_attr
            )
        assert torch.allclose(h1, h2)
        assert torch.allclose(e1, e2)
