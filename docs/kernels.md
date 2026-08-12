# Custom CUDA Kernels

Fused CUDA kernels for the GNN encoder and graph construction, targeting NVIDIA Ampere (sm_80) and Ada Lovelace (sm_89) architectures.

## Kernels

| Kernel | Replaces | Key optimisation |
|--------|----------|------------------|
| `fused_edge_update` | Materialise-then-GEMM: gather + concat (E, 3H) + matmul => single launch | Algebraic split: node-side term via cuBLAS on (N, H), edge-side GEMM with K=2H in registers |
| `fused_norm_residual` | LayerNorm + Dropout + residual add (3 launches => 1) | 2-pass fused stats (sum+sum_sq), shared gamma/beta cache, template dropout elimination |
| `fired_detector_node_features` | CPU graph construction per shot | Compacted fired detectors => (N, 6) node features on device |
| `fired_detector_edges` | CPU all-pairs edge construction | Node features => all-pairs (2, E) edge index + (E, 6) edge features on device |

## Prerequisites

- CUDA Toolkit >= 12.0
- PyTorch >= 2.5 with CUDA support
- A GPU with compute capability >= 8.0 (Ampere or newer)

## Build

### Option A: Ahead-of-time (recommended)

From the **project root**:

```bash
uv run scripts/build_kernels.py build_ext --inplace
```

Verify:

```bash
uv run python -c "import kernels; print(kernels.AVAILABLE)"  # True
```

### Option B: JIT compilation (development)

```python
from kernels.build import build
module = build()  # compiles on first call, cached afterwards
```

JIT is slower on first run but doesn't require the build step.

## Usage

> [!IMPORTANT]
> The `"cuda"` backend is **inference and benchmarking only**. The custom kernels do not implement autograd backward passes. For training, use `"pytorch"` or `"compiled"`.

Kernels activate automatically when the compute backend is set to `"cuda"`:

```python
from model.ops import set_backend
set_backend("cuda")  # inference only - no autograd backward
```

Or via environment variable:

```bash
QECDEC_BACKEND=cuda uv run scripts/eval_gnn.py \
    -c configs/eval_memory_d3_direct.yaml
```

Or in benchmarking:

```bash
uv run scripts/benchmark_all.py --distance 7 --batch-size 1
```

If the kernels are not built, `set_backend("cuda")` **raises**. It does not
downgrade to PyTorch: a silent downgrade would let a benchmark row or an
equivalence test carry the name of a backend that never ran. The one exception
is `QECDEC_BACKEND`, which is read at import time and degrades with a warning
so that importing `model.ops` cannot fail over an env var.

## Testing

Equivalence tests verify CUDA kernels match PyTorch output within `atol=1e-5`:

```bash
uv run scripts/build_kernels.py build_ext --inplace
uv run pytest tests/test_ops.py tests/test_bucketed_forward.py -v
```

Tests are automatically skipped without a GPU or without built kernels.

## Architecture notes

### `fused_edge_update`

Algebraic decomposition eliminates the (E, 3H) concat buffer entirely.  `W·[a|b|c] = W_sum·a + W_diff·b + W_edge·c` where the symmetric `W_sum·(h_src + h_dst)` moves off edges onto nodes via a single cuBLAS call on (N, H).  The per-edge kernel then GEMMs with K=2H (|h_src - h_dst| and edge embedding) and adds the node-side term in the epilogue.

- **Template dispatch**: `<bool UseVec4>` selected at compile time - float4 vectorised path (when `hidden_dim % 4 == 0`) or scalar fallback, with zero branch cost in the hot loop.
- **`__launch_bounds__(256)`** for register allocation hints.
- High parallelism from massive block count (batch x edges), not from wide blocks.

### `fused_norm_residual`

One thread-block per row (node or edge embedding).

- **2-pass instead of 3**: mean and variance computed together via `sum + sum_sq` in a single read of the input row, then `var = E[x^2] - E[x]^2`. Saves ~14% bandwidth vs the naive mean-then-variance approach. Numerically sufficient for float32 at `hidden_dim <= 512` (verified: max error 2.4e-7 vs Welford gold standard).
- **Shared memory cache** for gamma/beta vectors - loaded cooperatively once per block, eliminating repeated global reads in the normalise+scale pass.
- **Template `<bool Training>`**: inference path compiled without curand state allocation (~40 registers freed), dropout branch, or scale computation.
- **Reproducible seed** from PyTorch's default CUDA generator (`at::cuda::detail::getDefaultCUDAGenerator()`) instead of non-deterministic `steady_clock::now()`.

### `fired_detector_node_features` / `fired_detector_edges`

Batched graph construction reproducing `sampling.graph.build_fired_detector_graph` on device.  Both kernels write into caller-owned output buffers addressed through per-shot prefix sums, enabling the full syndrome-to-graph pipeline without host round-trips.

### Common

All kernels use:
- `at::cuda::getCurrentCUDAStream()` - correct behaviour with AMP, DataParallel, and multi-stream pipelines.
- `C10_CUDA_KERNEL_LAUNCH_CHECK()` - catches asynchronous kernel launch failures.
- `__restrict__` pointer hints on all kernel arguments.
- Anonymous namespaces for internal symbols.

## File layout

```
src/kernels/
├── __init__.py          # AVAILABLE flag
├── ops.py               # Python wrappers (fallback to PyTorch on CPU)
├── build.py             # JIT build config
├── bucketed.py          # Bucketed CUDA-Graphs fast path
├── graph_build.py       # Python wrapper for graph construction kernels
└── cpp/
    ├── fused_edge_update.cu     # Fused gather + symmetric combine + GEMM
    ├── fused_norm_residual.cu   # Fused LayerNorm + residual + dropout
    ├── graph_build.cu           # Batched fired-detector graph construction
    └── bindings.cpp             # pybind11 => kernels._C
```
