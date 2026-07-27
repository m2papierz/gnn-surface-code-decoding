# Custom CUDA Kernels

Fused CUDA kernels for the GNN encoder and loss computation, targeting NVIDIA Ampere (sm_80) and Ada Lovelace (sm_89) architectures.

## Kernels

| Kernel | Replaces | Key optimisation |
|--------|----------|------------------|
| `fused_edge_features` | 2x gather + add + sub+abs + concat (5 launches => 1) | Template-dispatched float4 vectorised loads |
| `fused_norm_residual` | LayerNorm + Dropout + residual add (3 launches => 1) | 2-pass fused stats (sum+sum_sq), shared gamma/beta cache, template dropout elimination |
| `graph_norm_bce` | BCE + scatter_add + normalize (3 launches => 2) | Warp-cooperative segmented prefix sum (64× fewer atomicAdd vs naive scatter) |

## Prerequisites

- CUDA Toolkit ≥ 12.0
- PyTorch ≥ 2.5 with CUDA support
- A GPU with compute capability ≥ 8.0 (Ampere or newer)

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
QECDEC_BACKEND=cuda uv run scripts/eval_harness.py \
    --checkpoint outputs/runs/direct/best.pt
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

### `fused_edge_features`

One thread-block per edge, each block processes one edge's hidden dimension.

- **Template dispatch**: `<bool UseVec4>` selected at compile time - float4 vectorised path (when `hidden_dim % 4 == 0`) or scalar fallback, with zero branch cost in the hot loop.
- **`__launch_bounds__(256)`** for register allocation hints.
- High parallelism from massive block count (batch×edges ≈ 100k+ blocks), not from wide blocks.

### `fused_norm_residual`

One thread-block per row (node or edge embedding).

- **2-pass instead of 3**: mean and variance computed together via `sum + sum_sq` in a single read of the input row, then `var = E[x²] - E[x]²`. Saves ~14% bandwidth vs the naive mean-then-variance approach. Numerically sufficient for float32 at `hidden_dim ≤ 512` (verified: max error 2.4e-7 vs Welford gold standard).
- **Shared memory cache** for gamma/beta vectors - loaded cooperatively once per block, eliminating repeated global reads in the normalise+scale pass.
- **Template `<bool Training>`**: inference path compiled without curand state allocation (~40 registers freed), dropout branch, or scale computation.
- **Reproducible seed** from PyTorch's default CUDA generator (`at::cuda::detail::getDefaultCUDAGenerator()`) instead of non-deterministic `steady_clock::now()`.

### `graph_norm_bce`

Two-kernel pipeline: scatter (BCE => per-graph accumulators) then reduce (mean-of-means).

- **Warp-cooperative segmented prefix sum**: exploits the fact that PyG `Batch` produces sorted (monotonic) `edge_graph` indices. Within each warp of 32 threads, edges from the same graph form contiguous segments. `__shfl_up_sync` with graph-ID equality check accumulates BCE values within each segment; only the segment-end lane emits a single `atomicAdd`. Result: **~32× fewer atomics** than per-thread scatter.
- **Precomputed graph counts**: `torch::bincount(edge_graph)` replaces the second set of `atomicAdd` for counting edges. Combined: **~64× fewer atomics** than the original (verified: d=7 batch=128 => 797k => 12k atomicAdd).
- Boundary handling: warps that straddle graph boundaries correctly emit separate atomics per segment (at most 2 per warp).

### Common

All kernels use:
- `at::cuda::getCurrentCUDAStream()` - correct behaviour with AMP, DataParallel, and multi-stream pipelines.
- `C10_CUDA_KERNEL_LAUNCH_CHECK()` - catches asynchronous kernel launch failures.
- `__restrict__` pointer hints on all kernel arguments.
- Anonymous namespaces for internal symbols.

## File layout

```
src/kernels/
├── README.md
├── __init__.py          # AVAILABLE flag
├── ops.py               # Python wrappers (fallback to PyTorch on CPU)
├── build.py             # JIT build config
└── cpp/
    ├── fused_edge_features.cu   # Kernel #1
    ├── fused_norm_residual.cu   # Kernel #2
    ├── graph_norm_bce.cu        # Kernel #3
    └── bindings.cpp             # pybind11 => kernels._C
```
