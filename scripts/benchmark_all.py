"""Latency, throughput, and memory benchmark across inference backends.

Measures p50/p95/p99 latency, per-shot throughput, and peak GPU memory for
GNN backends and classical decoders (Belief-Matching) on representative batch
shapes derived from Stim-sampled syndromes.

``--operation`` selects the experiment axis and determines the circuit root,
graph representation, and which backends are available.  The CUDA fast path
(custom kernels, bucketed CUDA-Graphs) is spatial-only; for non-spatial
representations (e.g. phased), only the PyTorch and compiled backends are
measured, and the output carries a comparability caveat.  For a spatial
representation the fast-path rows are required: an extension that does not
load fails the run instead of leaving them out of the table.

GNN backends tested (where available):

``pytorch``
    Vanilla eager-mode PyTorch.
``compiled``
    ``torch.compile(mode="default")`` - fuses ops via Triton.
``cuda``
    Custom CUDA kernels (fused edge update, fused norm/residual), float32
    throughout.  Spatial only.
``bucketed``
    CUDA-Graphs-captured forward, bucketed on node and edge totals.
    Spatial only.
``e2e_gpu+bucketed``
    End-to-end: GPU graph build => bucketed CUDA-Graphs forward.  This is the
    deployed latency path - timing starts at syndromes already in device
    memory.  Spatial only.

Classical decoders tested:

``belief_matching``
    Belief-Matching (BP re-weighted matching), CPU, shot by shot.  Timed with
    ``time.perf_counter()``, same warmup and iteration count as GNN.

Latency, not throughput, is the figure of merit: a real-time decoder keeps pace
with µs-scale syndrome cycles, so the batch-1 row is the one that matters.

Usage::

    uv run python scripts/benchmark_all.py --operation memory
    uv run python scripts/benchmark_all.py --operation zz_merge_split \\
        --distance 3 --batch-size 128 -o outputs/results/latency_zz_merge_split_d3.json
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import stim
import torch
from torch_geometric.data import Batch, Data

from artifact_file import write_json
from decoders import BeliefMatchingDecoder
from model.decoder import QECDecoder, build_model, collate_graphs
from model.ops import set_backend
from sampling.graph import CircuitMetadata
from sampling.profile import OperationProfile, resolve_profile
from sampling.representation import (
    SUPPORTED_FAST_PATH_VERSIONS,
    DataContract,
    resolve_builder,
)
from sampling.sampler import settings_from_circuit_dir


logger = logging.getLogger(__name__)

DEFAULT_N_ITERS = 200
WARMUP_ITERS = 30


def _load(
    profile: OperationProfile, distance: int, error_prob: float
) -> tuple[Path, stim.Circuit, CircuitMetadata]:
    """Read the operation's one committed circuit at ``(distance, error_prob)``.

    Raises
    ------
    ValueError
        If the root holds no circuit at that point, or more than one (at
        different rounds).
    """
    points = settings_from_circuit_dir(
        profile.circuit_root,
        operation=profile.operation,
        distances=[distance],
        error_probs=[error_prob],
    )
    if len(points) != 1:
        raise ValueError(
            f"{profile.circuit_root} holds {len(points)} circuits at d={distance}, "
            f"p={error_prob}: {[str(point.circuit_path) for point in points]}"
        )
    (point,) = points
    circuit = stim.Circuit.from_file(str(point.circuit_path))
    metadata = profile.metadata_extractor(
        point.circuit_path, num_observables=profile.data_contract.num_observables
    )
    return point.circuit_path, circuit, metadata


def _percentiles(timings_ms: Sequence[float]) -> dict[str, float]:
    t = np.asarray(timings_ms, dtype=np.float64)
    return {
        "p50_ms": round(float(np.percentile(t, 50)), 4),
        "p95_ms": round(float(np.percentile(t, 95)), 4),
        "p99_ms": round(float(np.percentile(t, 99)), 4),
    }


def _make_pyg_batch(
    syndromes: np.ndarray,
    metadata: CircuitMetadata,
    device: torch.device,
    builder: Callable,
) -> Batch:
    graphs = [builder(s, metadata) for s in syndromes]
    return collate_graphs(
        [
            Data(
                x=torch.from_numpy(g.node_features),
                edge_index=torch.from_numpy(g.edge_index),
                edge_attr=torch.from_numpy(g.edge_features),
            )
            for g in graphs
        ],
        device,
    )


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
        f"{result.get('peak_mem_mb', 0):>10.1f}"
    )


def _run_backend(
    name: str,
    model: QECDecoder,
    batch: Batch,
    n_iters: int,
    *,
    ops_backend: str = "pytorch",
    compile_model: bool = False,
) -> dict:
    set_backend(ops_backend)

    torch.cuda.reset_peak_memory_stats()

    m = (
        torch.compile(model, mode="default", fullgraph=False)
        if compile_model
        else model
    )

    @torch.no_grad()
    def fn() -> None:
        m(batch)

    timings = _time_cuda(fn, n_iters)
    stats = _percentiles(timings)
    set_backend("pytorch")
    return {
        "backend": name,
        "timing_method": "cuda_events",
        **stats,
        "peak_mem_mb": round(_peak_memory_mb(), 1),
    }


def _run_belief_matching(
    circuit_path: Path,
    syndromes: np.ndarray,
    n_iters: int,
) -> dict:
    """Time the Belief-Matching decoder on CPU."""
    decoder = BeliefMatchingDecoder(circuit_path)

    def fn() -> None:
        decoder.decode_batch(syndromes)

    timings = _time_cpu(fn, n_iters)
    stats = _percentiles(timings)
    return {"backend": "belief_matching", "timing_method": "perf_counter", **stats}


def _run_device_path(
    model: QECDecoder,
    syndromes: np.ndarray,
    metadata: CircuitMetadata,
    device: torch.device,
    n_iters: int,
    *,
    contract: DataContract,
    end_to_end: bool,
) -> dict:
    """Time the bucketed forward, optionally including the graph build.

    With *end_to_end* the timed region starts at syndromes already resident in
    device memory, which is the shape of the deployed latency path.
    """
    # Imported here because these modules load the compiled extension, which
    # only spatial representations need; for them a missing build fails here.
    from kernels.bucketed import BucketedGraphRunner
    from kernels.graph_build import build_fired_detector_graphs, metadata_to_device

    set_backend("cuda")
    dev_meta = metadata_to_device(metadata, device)
    device_syn = torch.as_tensor(syndromes, device=device)

    def build():
        return build_fired_detector_graphs(device_syn, dev_meta)

    prebuilt = build()
    runner = BucketedGraphRunner(model, contract=contract, batch_size=len(syndromes))
    torch.cuda.reset_peak_memory_stats()

    def fn() -> None:
        runner.forward_from_batch(build() if end_to_end else prebuilt)

    timings = _time_cuda(fn, n_iters)
    stats = _percentiles(timings)
    set_backend("pytorch")

    return {
        "backend": "e2e_gpu+bucketed" if end_to_end else "bucketed",
        "timing_method": "cuda_events",
        **stats,
        "peak_mem_mb": round(_peak_memory_mb(), 1),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--operation",
        required=True,
        help="Operation to benchmark (e.g. memory, zz_merge_split).",
    )
    parser.add_argument("--distance", type=int, default=7, choices=[3, 5, 7])
    parser.add_argument("--error-prob", type=float, default=0.01)
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
    parser.add_argument(
        "--skip-belief-matching",
        action="store_true",
        help="Omit the Belief-Matching decoder row.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not torch.cuda.is_available():
        raise SystemExit("Benchmarks require a CUDA device.")

    profile = resolve_profile(args.operation)
    contract = profile.data_contract
    representation = profile.representation_version
    fast_path_available = representation in SUPPORTED_FAST_PATH_VERSIONS

    device = torch.device("cuda")
    circuit_path, circuit, metadata = _load(profile, args.distance, args.error_prob)
    sampler = circuit.compile_detector_sampler(seed=args.seed)
    dets, _ = sampler.sample(
        shots=args.batch_size, separate_observables=True, bit_packed=False
    )
    syndromes = dets.astype(np.uint8)

    fired = syndromes.sum(axis=1)
    logger.info(
        "operation=%s  representation=%s  d=%d  p=%s  detectors=%d  batch=%d",
        args.operation,
        representation,
        args.distance,
        args.error_prob,
        metadata.num_detectors,
        args.batch_size,
    )
    logger.info(
        "fired: mean=%.1f  p99=%.0f  max=%d",
        fired.mean(),
        np.percentile(fired, 99),
        fired.max(),
    )
    if not fast_path_available:
        logger.info(
            "fast-path backends skipped: representation %r is not spatial",
            representation,
        )
    logger.info("GPU: %s\n", torch.cuda.get_device_name(0))

    builder = resolve_builder(representation)
    model = (
        build_model(
            node_dim=contract.node_dim,
            edge_dim=contract.edge_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_observables=contract.num_observables,
            dropout=0.0,
        )
        .to(device)
        .eval()
    )
    pyg_batch = _make_pyg_batch(syndromes, metadata, device, builder)

    header = (
        f"{'backend':<24}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}"
        f"{'us/shot':>10}{'shots/s':>12}{'mem MB':>10}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    rows: list[dict] = []

    gnn_backends: list[tuple[str, Callable[[], dict]]] = [
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
    ]

    if fast_path_available:
        gnn_backends.extend(
            [
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
                            contract=contract,
                            end_to_end=e2e,
                        ),
                    )
                    for e2e in (False, True)
                ],
            ]
        )

    classical_backends: list[tuple[str, Callable[[], dict]]] = []
    if not args.skip_belief_matching:
        classical_backends.append(
            (
                "belief_matching",
                lambda: _run_belief_matching(
                    circuit_path,
                    syndromes,
                    args.n_iters,
                ),
            )
        )

    skipped = set()
    if args.skip_compiled:
        skipped.add("compiled")

    for name, run_fn in gnn_backends + classical_backends:
        if name in skipped:
            logger.info("%-24s skipped (--skip-%s)", name, name)
            continue
        result = run_fn()
        p50 = result["p50_ms"]
        result["per_shot_p50_us"] = (
            round(p50 * 1000.0 / args.batch_size, 2) if p50 > 0 else 0
        )
        result["shots_per_sec"] = (
            round(args.batch_size / (p50 / 1000.0), 0) if p50 > 0 else 0
        )
        rows.append(result)
        logger.info("%s", _format_row(result, args.batch_size))

    caveats: list[str] = []
    if not fast_path_available:
        caveats.append(
            f"Representation {representation!r} cannot use the CUDA fast path "
            f"(spatial-only). GNN latency here is from the PyTorch/compiled "
            f"path and is NOT directly comparable to memory-track fast-path "
            f"numbers."
        )
    if any(r["backend"] == "belief_matching" for r in rows):
        caveats.append(
            "Belief-Matching runs on CPU (shot-by-shot Python loop, timed "
            "with time.perf_counter); GNN runs on GPU (timed with CUDA "
            "events). The comparison measures deployed latency, not "
            "algorithmic cost."
        )

    if args.output is not None:
        payload: dict = {
            "operation": args.operation,
            "representation": representation,
            "gpu": torch.cuda.get_device_name(0),
            "distance": args.distance,
            "error_prob": args.error_prob,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "n_iters": args.n_iters,
            "warmup_iters": WARMUP_ITERS,
            "fast_path_available": fast_path_available,
            "results": rows,
        }
        if caveats:
            payload["caveats"] = caveats
        write_json(args.output, payload)
        logger.info("\nWrote %s", args.output)


if __name__ == "__main__":
    main()
