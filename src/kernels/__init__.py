"""Custom CUDA kernels for the GNN-QEC inference fast path.

Forward-pass only — none of these ops implements an autograd backward, so
reaching one from a training path yields silently wrong gradients.

- ``fused_edge_update``: gather, symmetric combine and GEMM in one launch,
  never materialising the ``(E, 3H)`` concatenation
- ``fused_norm_residual_dropout``: row-wise LayerNorm + dropout + residual
- ``fired_detector_node_features`` / ``fired_detector_edges``: batched
  syndromes to device-side complete graphs, bit-identical to the numpy builder
"""

from __future__ import annotations

# `_C` links against libc10/libtorch.  Importing it before torch has loaded
# those shared objects fails with `ImportError: libc10.so`, which the try
# below would silently report as "kernels not built".  Importing torch first
# is what makes AVAILABLE mean what it says.
import torch  # noqa: F401


try:
    from kernels._C import (  # noqa: F401
        fired_detector_edges,
        fired_detector_node_features,
        fused_edge_update,
        fused_norm_residual_dropout,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False
