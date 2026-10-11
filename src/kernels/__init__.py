"""Custom CUDA kernels for the GNN-QEC inference fast path.

Forward-pass only - none of these ops implements an autograd backward, so
reaching one from a training path yields silently wrong gradients.

- ``fused_edge_update``: gather, symmetric combine and GEMM in one launch,
  never materialising the ``(E, 3H)`` concatenation
- ``fused_norm_residual_dropout``: row-wise LayerNorm + dropout + residual
- ``fired_detector_node_features`` / ``fired_detector_edges``: batched
  syndromes to device-side complete graphs, bit-identical to the numpy builder

The compiled extension is reached only through :func:`kernels.extension.load`,
which refuses a build of anything but the sources in ``cpp/``.
"""
