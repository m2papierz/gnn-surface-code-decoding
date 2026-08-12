"""Verify model.ops functions match their reference implementations.

The CUDA equivalence suite is a two-tier ladder (see the ``cuda-fast-path``
decision graph):

tier 1
    float32 fused kernel against eager PyTorch at atol 1e-5.  The fused path is
    float32 throughout, so a deviation beyond that bound is a structural defect
    - wrong weight block, transposed operand, bad gather, off-by-one edge index
    - with no precision noise to hide behind.
tier 2
    LER re-measurement on the frozen eval sets, required only before the fast
    path serves a reported number.  Lives in the evaluation harness, not here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model.ops import (
    Backend,
    EdgeUpdateWeights,
    edge_update,
    edge_update_reference,
    fused_norm_residual_dropout,
    get_backend,
    set_backend,
)


def _make_graph(n_nodes: int = 10, n_edges: int = 20, hidden: int = 16):
    x = torch.randn(n_nodes, hidden)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_h = torch.randn(n_edges, hidden)
    return x, edge_index, edge_h


def _make_weights(hidden: int):
    linear = nn.Linear(3 * hidden, hidden)
    return linear, EdgeUpdateWeights.from_linear(linear)


def _make_cuda_weights(hidden: int):
    linear = nn.Linear(3 * hidden, hidden).cuda()
    return linear, EdgeUpdateWeights.from_linear(linear)


class TestEdgeUpdateReference:
    """The reference is the normative definition; pin it against the algebra."""

    def test_matches_concat_then_linear(self) -> None:
        x, ei, eh = _make_graph()
        linear, weights = _make_weights(x.shape[1])

        out = edge_update_reference(x, ei, eh, weights)

        h_src, h_dst = x[ei[0]], x[ei[1]]
        concat = torch.cat([h_src + h_dst, (h_src - h_dst).abs(), eh], dim=-1)
        torch.testing.assert_close(out, linear(concat), atol=1e-6, rtol=1e-6)

    def test_shape(self) -> None:
        x, ei, eh = _make_graph()
        _, weights = _make_weights(x.shape[1])
        assert edge_update_reference(x, ei, eh, weights).shape == (ei.shape[1], 16)

    def test_symmetric_under_edge_reversal_when_edge_h_is_shared(self) -> None:
        """Sum and |diff| blocks are symmetric, so reversal changes nothing."""
        x, ei, eh = _make_graph()
        _, weights = _make_weights(x.shape[1])
        zero_eh = torch.zeros_like(eh)

        forward = edge_update_reference(x, ei, zero_eh, weights)
        reverse = edge_update_reference(
            x, torch.stack([ei[1], ei[0]]), zero_eh, weights
        )
        torch.testing.assert_close(forward, reverse, atol=1e-6, rtol=1e-6)

    def test_zero_edges(self) -> None:
        x = torch.randn(5, 8)
        _, weights = _make_weights(8)
        out = edge_update_reference(
            x, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 8), weights
        )
        assert out.shape == (0, 8)

    def test_self_loops_zero_the_diff_block(self) -> None:
        x = torch.randn(4, 8)
        ei = torch.tensor([[0, 1, 2], [0, 1, 2]])
        eh = torch.zeros(3, 8)
        _, weights = _make_weights(8)

        out = edge_update_reference(x, ei, eh, weights)

        # src == dst, so the |diff| block vanishes and only 2*x[i] survives.
        expected = torch.stack(
            [torch.nn.functional.linear(2 * x[i], weights.w_sum) for i in range(3)]
        )
        if weights.bias is not None:
            expected = expected + weights.bias
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestEdgeUpdateWeights:
    def test_blocks_reconstruct_the_original_weight(self) -> None:
        linear, weights = _make_weights(16)
        rebuilt = torch.cat([weights.w_sum, weights.w_tail], dim=1)
        torch.testing.assert_close(rebuilt, linear.weight.detach())

    def test_block_shapes(self) -> None:
        _, weights = _make_weights(16)
        assert weights.w_sum.shape == (16, 16)
        assert weights.w_tail.shape == (16, 32)
        assert weights.out_dim == 16

    def test_every_block_stays_float32(self) -> None:
        _, weights = _make_weights(16)
        assert weights.w_sum.dtype == torch.float32
        assert weights.w_tail.dtype == torch.float32
        assert weights.bias is not None and weights.bias.dtype == torch.float32

    def test_rejects_non_triple_width(self) -> None:
        with pytest.raises(ValueError, match="three equal blocks"):
            EdgeUpdateWeights.from_linear(nn.Linear(17, 8))

    def test_blocks_are_contiguous(self) -> None:
        _, weights = _make_weights(16)
        assert weights.w_sum.is_contiguous()
        assert weights.w_tail.is_contiguous()


class TestFusedNormResidualDropout:
    def test_eval_matches_norm_plus_residual(self) -> None:
        x = torch.randn(8, 16)
        residual = torch.randn(8, 16)
        norm = nn.LayerNorm(16)
        dropout = nn.Dropout(0.1)

        result = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        torch.testing.assert_close(result, norm(x) + residual)

    def test_training_mode_has_dropout_effect(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(100, 32)
        residual = torch.zeros(100, 32)
        norm = nn.LayerNorm(32)
        dropout = nn.Dropout(0.5)

        result = fused_norm_residual_dropout(x, residual, norm, dropout, training=True)
        assert (result - norm(x)).abs().sum().item() > 0, "Dropout had no effect"


class TestBackendManagement:
    def test_default_is_pytorch(self) -> None:
        set_backend("pytorch")
        assert get_backend() == Backend.PYTORCH

    def test_set_compiled(self) -> None:
        set_backend("compiled")
        assert get_backend() == Backend.COMPILED
        set_backend("pytorch")

    def test_cuda_request_never_silently_downgrades(self) -> None:
        """A mislabelled backend is worse than a missing one.

        A silent downgrade to pytorch would let a benchmark row or an
        equivalence test carry the name of a backend that never ran.
        """
        try:
            import kernels

            built = kernels.AVAILABLE
        except ImportError:
            built = False

        set_backend("pytorch")
        if built:
            set_backend("cuda")
            assert get_backend() == Backend.CUDA
            set_backend("pytorch")
        else:
            with pytest.raises(RuntimeError, match="not importable"):
                set_backend("cuda")
            # The failed request left the active backend untouched.
            assert get_backend() == Backend.PYTORCH

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            set_backend("nonexistent")

    def test_case_insensitive(self) -> None:
        set_backend("PyTorch")
        assert get_backend() == Backend.PYTORCH
        set_backend("COMPILED")
        assert get_backend() == Backend.COMPILED
        set_backend("pytorch")


def _cuda_ops_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import kernels

        return kernels.AVAILABLE
    except ImportError:
        return False


pytestmark_cuda = pytest.mark.skipif(
    not _cuda_ops_available(), reason="CUDA ops not built or no GPU"
)


@pytestmark_cuda
class TestEdgeUpdateStructuralEquivalence:
    """Tier 1 - float32 fused kernel against eager PyTorch.

    Every shape here also exercises the tiling: 64x128 output tiles over a
    K=2H march, so the cases below straddle single, partial and multiple tiles
    on both axes.
    """

    @pytest.mark.parametrize(
        ("n_nodes", "n_edges", "hidden"),
        [
            (10, 20, 16),  # sub-tile on both axes
            (24, 90, 64),  # partial M tile, exact N tile
            (40, 256, 128),  # multiple M tiles, exact N tile
            (64, 300, 128),  # M tail
            (8, 1, 128),  # single edge
            (12, 65, 32),  # M just past one tile
        ],
    )
    def test_matches_reference(self, n_nodes: int, n_edges: int, hidden: int) -> None:
        torch.manual_seed(0)
        x, ei, eh = _make_graph(n_nodes, n_edges, hidden)
        x, ei, eh = x.cuda(), ei.cuda(), eh.cuda()
        linear, weights = _make_cuda_weights(hidden)

        set_backend("pytorch")
        with torch.no_grad():
            ref = edge_update(x, ei, eh, linear, weights)

        set_backend("cuda")
        out = edge_update(x, ei, eh, linear, weights)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_zero_edges(self) -> None:
        x = torch.randn(5, 32).cuda()
        linear, weights = _make_cuda_weights(32)

        set_backend("cuda")
        out = edge_update(
            x,
            torch.zeros(2, 0, dtype=torch.long).cuda(),
            torch.zeros(0, 32).cuda(),
            linear,
            weights,
        )
        set_backend("pytorch")

        assert out.shape == (0, 32)

    def test_self_loops(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(4, 64).cuda()
        ei = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long).cuda()
        eh = torch.randn(3, 64).cuda()
        linear, weights = _make_cuda_weights(64)

        set_backend("pytorch")
        with torch.no_grad():
            ref = edge_update(x, ei, eh, linear, weights)
        set_backend("cuda")
        out = edge_update(x, ei, eh, linear, weights)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytestmark_cuda
class TestFusedNormResidualDropoutEquivalence:
    @pytest.mark.parametrize(("rows", "hidden"), [(8, 16), (1, 32), (4, 7), (128, 128)])
    def test_matches_reference(self, rows: int, hidden: int) -> None:
        x = torch.randn(rows, hidden).cuda()
        residual = torch.randn(rows, hidden).cuda()
        norm = nn.LayerNorm(hidden).cuda()
        dropout = nn.Dropout(0.0)

        set_backend("pytorch")
        ref = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        set_backend("cuda")
        out = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_matches_the_encoders_own_norm_layer(self) -> None:
        """Guards the kernel against the layer the encoder actually uses.

        The kernel normalises each row over its own channels.  PyG's
        ``LayerNorm`` defaults to ``mode="graph"``, which takes statistics
        across the whole tensor; comparing only against ``nn.LayerNorm`` would
        let that mismatch pass unnoticed.
        """
        from torch_geometric.nn import LayerNorm

        from model.encoder import _GINEBlock

        block = _GINEBlock(hidden_dim=32, edge_dim=6, dropout=0.0)
        assert isinstance(block.norm, LayerNorm)
        assert block.norm.mode == "node"

        norm = block.norm.cuda()
        dropout = nn.Dropout(0.0)
        x = torch.randn(12, 32).cuda()
        residual = torch.randn(12, 32).cuda()

        set_backend("pytorch")
        ref = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        set_backend("cuda")
        out = fused_norm_residual_dropout(x, residual, norm, dropout, training=False)
        set_backend("pytorch")

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytestmark_cuda
class TestEncoderEquivalence:
    """Every kernel in its real call site, which is where wiring bugs show."""

    def test_forward_matches_across_backends(self) -> None:
        from model.encoder import DetectorGraphEncoder

        torch.manual_seed(0)
        encoder = DetectorGraphEncoder(hidden_dim=64, num_layers=3, dropout=0.0)
        encoder = encoder.cuda().eval()

        x = torch.randn(24, encoder.node_dim).cuda()
        edge_index = torch.randint(0, 24, (2, 90)).cuda()
        edge_attr = torch.randn(90, encoder.edge_dim).cuda()

        set_backend("pytorch")
        with torch.no_grad():
            ref_h, ref_e = encoder(x, edge_index, edge_attr)

        set_backend("cuda")
        with torch.no_grad():
            out_h, out_e = encoder(x, edge_index, edge_attr)
        set_backend("pytorch")

        torch.testing.assert_close(out_h, ref_h, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_e, ref_e, atol=1e-5, rtol=1e-5)

    def test_weight_cache_invalidates_on_weight_mutation(self) -> None:
        from model.encoder import _GINEBlock

        block = _GINEBlock(hidden_dim=32, edge_dim=6, dropout=0.0).cuda()

        set_backend("cuda")
        try:
            before = block._fused_edge_weights().w_sum.clone()
            with torch.no_grad():
                block.edge_update[0].weight.add_(1.0)
            after = block._fused_edge_weights().w_sum
        finally:
            set_backend("pytorch")

        assert not torch.equal(before, after)

    def test_no_weight_split_off_the_fast_path(self) -> None:
        """Splitting during training would strand the parameter's gradient."""
        from model.encoder import _GINEBlock

        block = _GINEBlock(hidden_dim=32, edge_dim=6, dropout=0.0)

        set_backend("pytorch")
        assert block._fused_edge_weights() is None
        set_backend("compiled")
        assert block._fused_edge_weights() is None
        set_backend("pytorch")
