"""
Swappable compute operations for GNN encoder and heads.

Every function here isolates a compute-intensive pattern that appears in
the encoder or head forward passes. The active backend is selected via
:func:`set_backend` or the ``QECDEC_BACKEND`` environment variable.

Backends
--------
``"pytorch"``
    Pure PyTorch reference implementations (default, always available).
    Full autograd support - safe for training and inference.
``"compiled"``
    ``torch.compile``-wrapped PyTorch - identical numerics, kernel fusion
    handled by the compiler.  Full autograd support - recommended for
    training on GPU.
``"cuda"``
    Hand-written CUDA kernels loaded from ``kernels`` (requires build).
    **Inference only** - these are forward-pass kernels without autograd
    backward implementations.  Using this backend for training will
    silently break gradient propagation.

A kernel that raises propagates. The single permitted fallback lives in
:mod:`kernels.ops`, which routes non-CUDA and non-contiguous inputs to the
PyTorch reference on an explicit, testable condition; catching failures
here instead would let a broken kernel report numbers it never produced.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class Backend(Enum):
    """Available compute backends."""

    PYTORCH = "pytorch"
    COMPILED = "compiled"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class EdgeUpdateWeights:
    """Column blocks of a trained ``nn.Linear(3H, H_out)`` edge-update weight.

    ``F.linear`` computes ``input @ W.T``, so with
    ``input = [h_src + h_dst | |h_src - h_dst| | edge_h]`` the product splits
    exactly into three column blocks of ``W``.  The first block multiplies a
    *linear* function of the node embeddings, so it is evaluated per node and
    gathered; the remaining two stay on the edges and are packed together so
    the fused kernel runs one GEMM of inner dimension ``2H``.

    Parameters
    ----------
    w_sum : Tensor, shape ``(H_out, H)``, float32
        Block multiplying ``h_src + h_dst``.
    w_tail : Tensor, shape ``(H_out, 2H)``, float32
        ``[w_diff | w_edge]``, contiguous.
    bias : Tensor or None, shape ``(H_out,)``, float32
    """

    w_sum: torch.Tensor
    w_tail: torch.Tensor
    bias: torch.Tensor | None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> Self:
        """Split a trained edge-update linear into its three column blocks.

        Parameters
        ----------
        linear : nn.Linear
            Must have ``in_features == 3 * out_features_of_the_hidden_dim``.

        Returns
        -------
        EdgeUpdateWeights
        """
        out_dim, in_dim = linear.weight.shape
        if in_dim % 3 != 0:
            raise ValueError(
                f"edge-update weight has in_features={in_dim}, which is not "
                "three equal blocks"
            )
        hidden = in_dim // 3
        weight = linear.weight.detach()
        return cls(
            w_sum=weight[:, :hidden].contiguous(),
            w_tail=weight[:, hidden:].contiguous(),
            bias=None if linear.bias is None else linear.bias.detach(),
        )

    @property
    def out_dim(self) -> int:
        """Output width ``H_out``."""
        return self.w_sum.shape[0]


_active_backend: Backend = Backend.PYTORCH
_cuda_module = None  # lazily imported and cached


def get_backend() -> Backend:
    """Return the currently active compute backend."""
    return _active_backend


def set_backend(backend: str | Backend) -> None:
    """Set the compute backend globally.

    Asking for ``"cuda"`` when the extension is not built raises.  A silent
    downgrade to ``"pytorch"`` would let a caller label a benchmark row, or an
    equivalence test, with a backend that never ran - the same failure the
    removed ``try/except`` in this module used to hide.  Callers that legitimately
    tolerate a missing extension catch this and skip, rather than being handed a
    different backend under the name they asked for.

    Parameters
    ----------
    backend : str or Backend
        One of ``"pytorch"``, ``"compiled"``, ``"cuda"``.

    Raises
    ------
    ValueError
        If *backend* is not a known backend name.
    RuntimeError
        If ``"cuda"`` is requested but the CUDA extension is not importable.
    """
    global _active_backend, _cuda_module

    if isinstance(backend, str):
        backend = Backend(backend.lower())

    if backend == Backend.CUDA:
        try:
            import kernels.ops as _mod
        except ImportError as exc:
            raise RuntimeError(
                "cuda backend requested but the CUDA extension is not importable; "
                "build it with `uv run python scripts/build_kernels.py build_ext "
                f"--inplace` or select another backend ({exc})"
            ) from exc
        _cuda_module = _mod

    _active_backend = backend
    logger.info("Compute backend set to: %s", backend.value)


def _init_backend_from_env() -> None:
    """Initialise the backend from ``QECDEC_BACKEND`` env var (if set).

    Import time is the one place a degrade is right: importing this module must
    not explode because an env var names a backend this machine cannot serve.
    Every explicit :func:`set_backend` call still raises.
    """
    env = os.environ.get("QECDEC_BACKEND", "pytorch").lower()
    try:
        set_backend(env)
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "QECDEC_BACKEND='%s' unusable (%s) - starting on pytorch", env, exc
        )
        set_backend(Backend.PYTORCH)


def edge_update(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_h: torch.Tensor,
    linear: nn.Linear,
    weights: EdgeUpdateWeights | None = None,
) -> torch.Tensor:
    """Symmetric edge features followed by their linear projection.

    Computes ``[h_src + h_dst | |h_src - h_dst| | edge_h] @ W.T + b``.  The
    two steps are one operation because the concatenation is never worth
    materialising: it is three times the width of its only consumer, and the
    ``cuda`` backend folds it into the GEMM entirely.

    The ``pytorch`` and ``compiled`` backends run *linear* directly, so the
    parameter stays on the autograd graph and training is unaffected.  The
    split blocks are detached and therefore only ever reach the inference-only
    ``cuda`` backend.

    Parameters
    ----------
    x : Tensor, shape ``(N, H)``
        Node embeddings.
    edge_index : Tensor, shape ``(2, E)``
        Source/destination indices in COO format.
    edge_h : Tensor, shape ``(E, H)``
        Current edge embeddings.
    linear : nn.Linear
        The trained ``(3H -> H_out)`` projection; the source of truth.
    weights : EdgeUpdateWeights or None
        Cached column blocks of *linear*.  Read only by the ``cuda`` backend,
        which derives them itself when None.

    Returns
    -------
    Tensor, shape ``(E, H_out)``
    """
    if _active_backend == Backend.CUDA and _cuda_module is not None:
        if weights is None:
            weights = EdgeUpdateWeights.from_linear(linear)
        return _cuda_module.edge_update(x, edge_index, edge_h, weights)

    h_src = x[edge_index[0]]
    h_dst = x[edge_index[1]]
    return linear(torch.cat([h_src + h_dst, (h_src - h_dst).abs(), edge_h], dim=-1))


def edge_update_reference(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_h: torch.Tensor,
    weights: EdgeUpdateWeights,
) -> torch.Tensor:
    """Pure-PyTorch edge update - the normative definition of the operation.

    Deliberately free of backend dispatch: :mod:`kernels.ops` falls back to
    *this*, not to :func:`edge_update`, because falling back to the dispatcher
    while the ``cuda`` backend is active would recurse forever.

    Parameters
    ----------
    x : Tensor, shape ``(N, H)``
    edge_index : Tensor, shape ``(2, E)``
    edge_h : Tensor, shape ``(E, H)``
    weights : EdgeUpdateWeights

    Returns
    -------
    Tensor, shape ``(E, H_out)``
    """
    h_src = x[edge_index[0]]
    h_dst = x[edge_index[1]]
    concat = torch.cat([h_src + h_dst, (h_src - h_dst).abs(), edge_h], dim=-1)
    weight = torch.cat([weights.w_sum, weights.w_tail], dim=1)
    return F.linear(concat, weight, weights.bias)


def fused_norm_residual_dropout(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
    dropout: nn.Module,
    training: bool,
) -> torch.Tensor:
    """Normalise, optionally drop, and add residual.

    Applies ``norm(x)``, then dropout (only when *training*), then adds
    the *residual* connection.

    Parameters
    ----------
    x : Tensor, shape ``(N, H)``
        Input to normalise.
    residual : Tensor, shape ``(N, H)``
        Residual tensor to add after norm + dropout.
    norm : nn.Module
        Row-wise normalisation layer exposing ``weight``/``bias``:
        ``nn.LayerNorm`` or ``torch_geometric.nn.LayerNorm(mode="node")``.
        The ``cuda`` backend normalises each row over its own channels, so a
        layer taking statistics across rows would not be equivalent.
    dropout : nn.Module
        Dropout layer (used only when *training* is ``True``).
    training : bool
        Whether the model is in training mode.

    Returns
    -------
    Tensor, shape ``(N, H)``
    """
    if _active_backend == Backend.CUDA and _cuda_module is not None:
        return _cuda_module.fused_norm_residual_dropout(
            x,
            residual,
            norm.weight,
            norm.bias,
            float(dropout.p),
            training,
        )

    x = norm(x)
    if training:
        x = dropout(x)
    return x + residual


_init_backend_from_env()
