"""CUDA kernel wrappers matching ``model.ops`` signatures.

Each wrapper dispatches on an explicit, testable condition — CUDA device and
contiguous layout — and otherwise defers to the PyTorch reference in
``model.ops``.  Failures inside a kernel propagate; they are never converted
into a quiet switch back to PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kernels._C import fused_edge_update as _cuda_edge_update
from kernels._C import fused_norm_residual_dropout as _cuda_norm_res_drop
from model.ops import EdgeUpdateWeights, edge_update_reference


def edge_update(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_h: torch.Tensor,
    weights: EdgeUpdateWeights,
) -> torch.Tensor:
    """CUDA fused symmetric edge features + linear projection.

    Computes ``[h_src + h_dst | |h_src - h_dst| | edge_h] @ W.T + b`` without
    ever materialising the ``(E, 3H)`` concatenation.  The sum block is
    linear, so it is evaluated once per *node* and gathered, leaving a GEMM
    of inner dimension ``2H`` over the edges.

    Parameters
    ----------
    x : Tensor, shape ``(N, H)``
    edge_index : Tensor, shape ``(2, E)``
    edge_h : Tensor, shape ``(E, H)``
    weights : EdgeUpdateWeights
        Pre-split weight blocks; see :class:`model.ops.EdgeUpdateWeights`.

    Returns
    -------
    Tensor, shape ``(E, H_out)``
    """
    if not (x.is_cuda and x.is_contiguous() and edge_h.is_contiguous()):
        return edge_update_reference(x, edge_index, edge_h, weights)

    # (N, H_out) — the linear block, evaluated per node rather than per edge.
    node_term = F.linear(x, weights.w_sum)
    return _cuda_edge_update(
        node_term=node_term.contiguous(),
        node_h=x,
        edge_h=edge_h,
        weight=weights.w_tail,
        bias=weights.bias,
        edge_index=edge_index.contiguous(),
    )


def fused_norm_residual_dropout(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """CUDA fused row-wise LayerNorm + residual + dropout.

    Parameters
    ----------
    x, residual : Tensor, shape ``(N, H)``
    gamma, beta : Tensor, shape ``(H,)``
        LayerNorm affine parameters.
    dropout_p : float
        Dropout probability, applied only when *training*.
    training : bool

    Returns
    -------
    Tensor, shape ``(N, H)``
    """
    if x.is_cuda and x.is_contiguous():
        return _cuda_norm_res_drop(x, residual, gamma, beta, dropout_p, training)

    normed = F.layer_norm(x, (x.size(-1),), gamma, beta)
    return F.dropout(normed, dropout_p, training) + residual
