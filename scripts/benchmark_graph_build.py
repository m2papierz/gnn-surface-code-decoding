"""Benchmark device-side graph construction against the numpy builder.

Measures the latency of turning a batch of syndromes into a model-ready
batched graph, on the three paths that can actually be deployed:

``numpy``
    ``build_fired_detector_graph`` per shot.  Graph construction alone, no
    collation and no transfer — the lower bound on the CPU path.
``numpy+collate+h2d``
    The honest CPU path: per-shot build, ``Batch.from_data_list``, then
    ``.to(device)``.  This is what an inference server pays today.
``gpu``
    ``build_fired_detector_graphs`` with syndromes already in device memory.

Latency is reported as p50/p95/p99 — never mean-only — with CUDA events on
the device path and ``perf_counter`` on the host path.

Examples
--------
    uv run python scripts/benchmark_graph_build.py
    uv run python scripts/benchmark_graph_build.py --distance 5 --batch-sizes 1 32
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

from sampling.graph import (
    CircuitMetadata,
    build_fired_detector_graph,
    extract_circuit_metadata,
)


logger = logging.getLogger(__name__)

CIRCUIT_DIR = Path("data/circuits")
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 8, 32, 128, 512)
DEFAULT_ITERS = 200
WARMUP_ITERS = 20


def _load(distance: int, error_prob: str) -> tuple[stim.Circuit, CircuitMetadata]:
    path = CIRCUIT_DIR / f"d{distance}_r{distance}_p{error_prob}.stim"
    circuit = stim.Circuit.from_file(str(path))
    metadata = extract_circuit_metadata(circuit, distance=distance, rounds=distance)
    return circuit, metadata


def _percentiles(timings_ms: Sequence[float]) -> dict[str, float]:
    t = np.asarray(timings_ms, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(t, 50)),
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
    }


def _time_host(fn: Callable[[], object], n_iters: int) -> list[float]:
    for _ in range(WARMUP_ITERS):
        fn()
    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def _time_device(fn: Callable[[], object], n_iters: int) -> list[float]:
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


def _numpy_batch(syndromes: np.ndarray, metadata: CircuitMetadata) -> Batch:
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
    )


def _measure(
    syndromes: np.ndarray,
    metadata: CircuitMetadata,
    device: torch.device,
    n_iters: int,
) -> list[dict[str, object]]:
    """Measure all three paths on one fixed batch of syndromes."""
    from kernels.graph_build import (
        build_fired_detector_graphs,
        detector_coords_to_device,
    )

    batch_size = len(syndromes)
    coords = detector_coords_to_device(metadata, device)
    device_syndromes = torch.as_tensor(syndromes, device=device)

    paths: list[tuple[str, Callable[[], object], bool]] = [
        (
            "numpy",
            lambda: [build_fired_detector_graph(s, metadata) for s in syndromes],
            False,
        ),
        (
            "numpy+collate+h2d",
            lambda: _numpy_batch(syndromes, metadata).to(device),
            False,
        ),
        (
            "gpu",
            lambda: build_fired_detector_graphs(
                device_syndromes,
                coords,
                distance=metadata.distance,
                rounds=metadata.rounds,
            ),
            True,
        ),
    ]

    rows: list[dict[str, object]] = []
    for name, fn, on_device in paths:
        timings = _time_device(fn, n_iters) if on_device else _time_host(fn, n_iters)
        stats = _percentiles(timings)
        rows.append(
            {
                "path": name,
                "batch_size": batch_size,
                "n_iters": n_iters,
                **{k: round(v, 4) for k, v in stats.items()},
                "per_shot_p50_us": round(stats["p50_ms"] * 1000.0 / batch_size, 3),
                "shots_per_sec": round(batch_size / (stats["p50_ms"] / 1000.0), 1),
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--distance", type=int, default=7, choices=[3, 5, 7])
    parser.add_argument("--error-prob", default="0_01", help="Circuit filename suffix.")
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--n-iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("GPU graph builder requires a CUDA device; none available.")

    device = torch.device("cuda")
    circuit, metadata = _load(args.distance, args.error_prob)
    sampler = circuit.compile_detector_sampler(seed=args.seed)
    pool = sampler.sample(shots=max(args.batch_sizes), bit_packed=False).astype(
        np.uint8
    )

    fired = pool.sum(axis=1)
    logger.info(
        "d=%d p=%s  detectors=%d  fired: mean=%.1f p99=%.0f max=%d",
        args.distance,
        args.error_prob.replace("_", "."),
        metadata.num_detectors,
        fired.mean(),
        np.percentile(fired, 99),
        fired.max(),
    )
    logger.info("GPU: %s\n", torch.cuda.get_device_name(0))

    header = f"{'path':<20}{'batch':>7}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}"
    header += f"{'us/shot':>10}{'shots/s':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    rows: list[dict[str, object]] = []
    for batch_size in sorted(args.batch_sizes):
        for row in _measure(pool[:batch_size], metadata, device, args.n_iters):
            rows.append(row)
            logger.info(
                "%-20s%7d%10.4f%10.4f%10.4f%10.2f%12.0f",
                row["path"],
                row["batch_size"],
                row["p50_ms"],
                row["p95_ms"],
                row["p99_ms"],
                row["per_shot_p50_us"],
                row["shots_per_sec"],
            )
        logger.info("")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "distance": args.distance,
                    "error_prob": args.error_prob.replace("_", "."),
                    "results": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
