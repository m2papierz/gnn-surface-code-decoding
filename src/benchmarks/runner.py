"""Inference benchmark harness for GNN QEC decoders.

Autodiscovers trained checkpoints, benchmarks each across backends and
batch sizes, and writes a consolidated JSON report with hardware metadata.

The project has three compute backends:

- ``pytorch`` — pure PyTorch eager ops
- ``compiled`` — PyTorch ops + ``torch.compile`` model wrapper
- ``cuda`` — hand-written CUDA kernels (requires ``build_kernels.py``)

The ``cuda`` backend operates at the ops level (``model.ops``),
not at the ``torch.compile`` level.  This module sets the correct ops
backend before loading the model for each benchmark.
"""

from __future__ import annotations

import json
import logging
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from inference import InferenceEngine, load_model_from_checkpoint, make_synthetic_batch


logger = logging.getLogger(__name__)

# Approximate d=5 rotated surface code geometry.
_DEFAULT_NODES: int = 66
_DEFAULT_EDGES: int = 192

# Maps project backend name => (ops_backend, engine_backend).
# ops_backend is passed to model.ops.set_backend().
# engine_backend is passed to inference.InferenceEngine.
_BACKEND_MAP: dict[str, tuple[str, str]] = {
    "pytorch": ("pytorch", "pytorch"),
    "compiled": ("compiled", "compiled"),
    "cuda": ("cuda", "pytorch"),
}


@dataclass
class BenchmarkReport:
    """Full benchmark report with hardware metadata."""

    hardware: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Write report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"hardware": self.hardware, "results": self.results},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Benchmark report saved to %s", path)


def _collect_hardware_info() -> dict[str, Any]:
    """Gather hardware and software metadata."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        info["cuda_version"] = torch.version.cuda or "unknown"
    return info


def detect_available_backends() -> list[str]:
    """Return list of backends that are actually usable.

    Always includes ``pytorch``.  Adds ``compiled`` if CUDA is available.
    Adds ``cuda`` if the CUDA kernels extension is built.
    """
    available = ["pytorch"]

    if torch.cuda.is_available():
        available.append("compiled")

    try:
        import kernels

        if kernels.AVAILABLE:
            available.append("cuda")
            logger.info("CUDA kernels detected — 'cuda' backend available")
        else:
            logger.info("CUDA kernels not built — 'cuda' backend skipped")
    except ImportError:
        logger.info("kernels package not found — 'cuda' backend skipped")

    return available


def discover_checkpoints(runs_dir: Path) -> dict[str, Path]:
    """Find best.pt checkpoints under runs_dir/direct/."""
    found: dict[str, Path] = {}
    ckpt = runs_dir / "direct" / "best.pt"
    if ckpt.is_file():
        found["direct"] = ckpt
        logger.info("Found checkpoint: %s", ckpt)
    else:
        logger.warning("No checkpoint at %s", ckpt)
    return found


def _peak_memory_mb() -> float:
    """Return peak CUDA memory allocated in MB (0.0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_checkpoint(
    checkpoint: Path,
    *,
    backends: Sequence[str] = ("pytorch", "compiled", "cuda"),
    batch_sizes: Sequence[int] = (1, 16, 64, 128),
    n_iters: int = 100,
    warmup_iters: int = 10,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Benchmark a single checkpoint across backends and batch sizes.

    For each backend, reloads the model fresh with the correct ops
    backend to avoid stale compiled caches.
    """
    from model.ops import set_backend

    results: list[dict[str, Any]] = []

    for backend in backends:
        if backend not in _BACKEND_MAP:
            logger.warning("Unknown backend %r, skipping", backend)
            continue

        ops_backend, engine_backend = _BACKEND_MAP[backend]

        # Reload model with the right ops backend active.
        set_backend(ops_backend)
        model, cfg = load_model_from_checkpoint(checkpoint, device=device)

        for bs in batch_sizes:
            label = f"direct/{backend}/bs={bs}"
            logger.info("Benchmarking %s ...", label)

            batch = make_synthetic_batch(
                n_graphs=bs,
                n_nodes_per_graph=_DEFAULT_NODES,
                n_edges_per_graph=_DEFAULT_EDGES,
                device=device,
            )

            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                engine = InferenceEngine(
                    model,
                    backend=engine_backend,
                    device=device,
                    warmup_iters=warmup_iters,
                )
                metrics = engine.benchmark(batch, n_iters=n_iters)
                peak_mem = _peak_memory_mb()

                result = {
                    "checkpoint": str(checkpoint),
                    "case": "direct",
                    "backend": backend,
                    "batch_size": bs,
                    "n_iters": n_iters,
                    "mean_ms": round(metrics["mean_ms"], 4),
                    "std_ms": round(metrics["std_ms"], 4),
                    "median_ms": round(metrics["median_ms"], 4),
                    "min_ms": round(metrics["min_ms"], 4),
                    "max_ms": round(metrics["max_ms"], 4),
                    "throughput_graphs_per_sec": round(
                        metrics["throughput_graphs_per_sec"], 1
                    ),
                    "peak_memory_mb": round(peak_mem, 1),
                }
                results.append(result)

                logger.info(
                    "  %s: %.2f ms  (%.0f graphs/s, %.0f MB peak)",
                    label,
                    metrics["mean_ms"],
                    metrics["throughput_graphs_per_sec"],
                    peak_mem,
                )

            except Exception:
                logger.exception("  %s: FAILED", label)

    return results


def run_all(
    runs_dir: Path,
    output_path: Path,
    *,
    backends: Sequence[str] | None = None,
    batch_sizes: Sequence[int] = (1, 16, 64, 128),
    n_iters: int = 100,
    warmup_iters: int = 10,
) -> BenchmarkReport:
    """Benchmark all discovered checkpoints and save report.

    Parameters
    ----------
    backends : sequence of str or None
        Backends to test.  None = auto-detect all available.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = discover_checkpoints(runs_dir)

    if not checkpoints:
        logger.error("No checkpoints found in %s", runs_dir)
        return BenchmarkReport()

    if backends is None:
        backends = detect_available_backends()
        logger.info("Auto-detected backends: %s", backends)

    report = BenchmarkReport(hardware=_collect_hardware_info())

    for _, ckpt in checkpoints.items():
        results = benchmark_checkpoint(
            ckpt,
            backends=backends,
            batch_sizes=batch_sizes,
            n_iters=n_iters,
            warmup_iters=warmup_iters,
            device=device,
        )
        report.results.extend(results)

    report.save(output_path)
    return report
