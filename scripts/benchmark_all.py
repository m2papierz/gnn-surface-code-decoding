"""Latency, throughput, and memory benchmark across all inference backends.

Measures p50/p95/p99 latency, per-shot throughput, and peak GPU memory for
each backend on representative v2 batch shapes derived from Stim-sampled
syndromes.

Backends tested (where available):

``pytorch``
    Vanilla eager-mode PyTorch.
``compiled``
    ``torch.compile(mode="default")`` — fuses ops via Triton.
``cuda``
    Custom CUDA kernels (fused edge update, fused norm/residual), float32
    throughout.
``bucketed``
    CUDA-Graphs-captured forward, bucketed on node and edge totals.
``e2e_gpu+bucketed``
    End-to-end: GPU graph build → bucketed CUDA-Graphs forward.  This is the
    deployed latency path — timing starts at syndromes already in device
    memory.

Latency, not throughput, is the figure of merit: spec §9 frames this against
µs-scale syndrome cycles, so the batch-1 row is the one that matters.

Usage::

    uv run python scripts/benchmark_all.py
    uv run python scripts/benchmark_all.py \\
        --distance 5 --batch-size 128 -o outputs/bench.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import stim
import torch
from torch_geometric.data import Batch, Data

from model.decoder import QECDecoder, build_model
from model.ops import set_backend
from sampling.graph import (
    CircuitMetadata,
    build_fired_detector_graph,
    extract_circuit_metadata,
)


logger = logging.getLogger(__name__)

CIRCUIT_DIR = Path("data/circuits")
DEFAULT_N_ITERS = 200
WARMUP_ITERS = 30


def _load(distance: int, error_prob: str) -> tuple[stim.Circuit, CircuitMetadata]:
    path = CIRCUIT_DIR / f"d{distance}_r{distance}_p{error_prob}.stim"
    circuit = stim.Circuit.from_file(str(path))
    return circuit, extract_circuit_metadata(
        circuit, distance=distance, rounds=distance
    )


def _percentiles(timings_ms: Sequence[float]) -> dict[str, float]:
    t = np.asarray(timings_ms, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(t, 50)), 4),
        "p95_ms": round(float(np.percentile(t, 95)), 4),
        "p99_ms": round(float(np.percentile(t, 99)), 4),
    }


def _make_pyg_batch(
    syndromes: np.ndarray, metadata: CircuitMetadata, device: torch.device
) -> Batch:
    graphs = [build_fired_detector_graph(s, metadata) for s in syndromes]
    return Batch.from_data_list(
        [
            Data(
                x=torch.from_numpy(g.node_features),
                edge_index=torch.from_numpy(g.edge_index),
                edge_attr=torch.from_numpy(g.edge_features),
            )
            for g in graphs
        ]
    ).to(device)


def _time_cuda(fn: Callable[[], object], n_iters: int) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(n_iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return timings


def _time_cpu(fn: Callable[[], object], n_iters: int) -> list[float]:
    for _ in range(WARMUP_ITERS):
        fn()
    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def _peak_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _format_row(result: dict, batch_size: int) -> str:
    p50 = result["p50_ms"]
    per_shot_us = p50 * 1000.0 / batch_size if p50 > 0 else 0
    shots_s = batch_size / (p50 / 1000.0) if p50 > 0 else 0
    return (
        f"{result['backend']:<24}"
        f"{result['p50_ms']:>10.4f}"
        f"{result['p95_ms']:>10.4f}"
        f"{result['p99_ms']:>10.4f}"
        f"{per_shot_us:>10.2f}"
        f"{shots_s:>12.0f}"
        f"{result['peak_mem_mb']:>10.1f}"
    )


def _run_backend(
    name: str,
    model: QECDecoder,
    batch: Batch,
    n_iters: int,
    *,
    ops_backend: str = "pytorch",
    compile_model: bool = False,
) -> dict | None:
    try:
        set_backend(ops_backend)
    except RuntimeError as exc:
        # Skip the row rather than measure a different backend under this name.
        logger.warning("%s unavailable: %s", name, exc)
        return None

    torch.cuda.reset_peak_memory_stats()

    m = model
    if compile_model:
        try:
            m = torch.compile(model, mode="default", fullgraph=False)
        except Exception as exc:
            logger.warning("%s unavailable: %s", name, exc)
            return None

    @torch.no_grad()
    def fn() -> None:
        m(batch)

    timings = _time_cuda(fn, n_iters)
    stats = _percentiles(timings)
    set_backend("pytorch")
    return {"backend": name, **stats, "peak_mem_mb": round(_peak_memory_mb(), 1)}


def _run_device_path(
    model: QECDecoder,
    syndromes: np.ndarray,
    metadata: CircuitMetadata,
    device: torch.device,
    n_iters: int,
    *,
    end_to_end: bool,
) -> dict | None:
    """Time the bucketed forward, optionally including the graph build.

    With *end_to_end* the timed region starts at syndromes already resident in
    device memory, which is the shape of the deployed latency path.
    """
    try:
        import kernels

        if not kernels.AVAILABLE:
            return None
        from kernels.bucketed import BucketedGraphRunner
        from kernels.graph_build import (
            build_fired_detector_graphs,
            detector_coords_to_device,
        )
    except ImportError:
        return None

    set_backend("cuda")
    coords = detector_coords_to_device(metadata, device)
    device_syn = torch.as_tensor(syndromes, device=device)

    def build():
        return build_fired_detector_graphs(
            device_syn, coords, distance=metadata.distance, rounds=metadata.rounds
        )

    prebuilt = build()
    runner = BucketedGraphRunner(model, batch_size=len(syndromes))
    torch.cuda.reset_peak_memory_stats()

    def fn() -> None:
        runner.forward_from_batch(build() if end_to_end else prebuilt)

    timings = _time_cuda(fn, n_iters)
    stats = _percentiles(timings)
    set_backend("pytorch")

    return {
        "backend": "e2e_gpu+bucketed" if end_to_end else "bucketed",
        **stats,
        "peak_mem_mb": round(_peak_memory_mb(), 1),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", type=int, default=7, choices=[3, 5, 7])
    parser.add_argument("--error-prob", default="0_01")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--n-iters", type=int, default=DEFAULT_N_ITERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-compiled",
        action="store_true",
        help="Omit the torch.compile row (compilation costs a minute or two).",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("Benchmarks require a CUDA device.")

    device = torch.device("cuda")
    circuit, metadata = _load(args.distance, args.error_prob)
    sampler = circuit.compile_detector_sampler(seed=args.seed)
    syndromes = sampler.sample(shots=args.batch_size, bit_packed=False).astype(np.uint8)

    fired = syndromes.sum(axis=1)
    logger.info(
        "d=%d  p=%s  detectors=%d  batch=%d",
        args.distance,
        args.error_prob.replace("_", "."),
        metadata.num_detectors,
        args.batch_size,
    )
    logger.info(
        "fired: mean=%.1f  p99=%.0f  max=%d",
        fired.mean(),
        np.percentile(fired, 99),
        fired.max(),
    )
    logger.info("GPU: %s\n", torch.cuda.get_device_name(0))

    model = (
        build_model(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=0.0,
        )
        .to(device)
        .eval()
    )
    pyg_batch = _make_pyg_batch(syndromes, metadata, device)

    header = (
        f"{'backend':<24}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}"
        f"{'us/shot':>10}{'shots/s':>12}{'mem MB':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    rows: list[dict] = []

    backends: list[tuple[str, Callable[[], dict | None]]] = [
        (
            "pytorch",
            lambda: _run_backend("pytorch", model, pyg_batch, args.n_iters),
        ),
        (
            "compiled",
            lambda: _run_backend(
                "compiled", model, pyg_batch, args.n_iters, compile_model=True
            ),
        ),
        (
            "cuda",
            lambda: _run_backend(
                "cuda", model, pyg_batch, args.n_iters, ops_backend="cuda"
            ),
        ),
        *[
            (
                "e2e_gpu+bucketed" if e2e else "bucketed",
                lambda e2e=e2e: _run_device_path(
                    model,
                    syndromes,
                    metadata,
                    device,
                    args.n_iters,
                    end_to_end=e2e,
                ),
            )
            for e2e in (False, True)
        ],
    ]

    skipped = set()
    if args.skip_compiled:
        skipped.add("compiled")

    for name, run_fn in backends:
        if name in skipped:
            logger.info("%-24s skipped (--skip-%s)", name, name)
            continue
        result = run_fn()
        if result is None:
            continue
        p50 = result["p50_ms"]
        result["per_shot_p50_us"] = (
            round(p50 * 1000.0 / args.batch_size, 2) if p50 > 0 else 0
        )
        result["shots_per_sec"] = (
            round(args.batch_size / (p50 / 1000.0), 0) if p50 > 0 else 0
        )
        rows.append(result)
        logger.info("%s", _format_row(result, args.batch_size))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "distance": args.distance,
                    "error_prob": args.error_prob.replace("_", "."),
                    "batch_size": args.batch_size,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "results": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("\nWrote %s", args.output)


if __name__ == "__main__":
    main()
