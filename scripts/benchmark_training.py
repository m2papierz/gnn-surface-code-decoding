"""Micro-benchmark for GPU training throughput.

Measures real training throughput (forward + backward + optimizer step)
using the actual model, optimizer, loss, and data pipeline. Reports
per-batch latency (p50/p95) and samples/sec for each configuration.

Configurations tested:
  - Eager vs torch.compile(mode="default", dynamic=True)
  - FP32 vs AMP (bf16)
  - Batch size sweep (64, 128, 256, 512)

Also generates a torch.profiler trace for the baseline configuration.

Examples
--------
    uv run python scripts/benchmark_training.py
    uv run python scripts/benchmark_training.py --no-profile
    uv run python scripts/benchmark_training.py --batch-sizes 64 128 256
"""

from __future__ import annotations

import argparse
import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.data import Batch, Data

from model.decoder import build_model
from sampling.graph import build_fired_detector_graph, extract_circuit_metadata
from training import FocalBCEWithLogitsLoss


logger = logging.getLogger(__name__)

CIRCUIT_DIR = Path("data/circuits")
CI_SHARD_DIR = Path("data/ci_shard")
DEFAULT_BATCH_SIZES = [64, 128, 256, 512]
N_WARMUP = 10
N_MEASURE = 40
N_POOL = 20


@dataclass(frozen=True)
class BenchResult:
    """Result of a single benchmark configuration."""

    label: str
    distance: int
    batch_size: int
    compiled: bool
    amp: bool
    latency_ms_p50: float
    latency_ms_p95: float
    samples_per_sec: float
    peak_mem_mb: float


def _build_batches_from_stim(
    circuit_path: Path,
    distance: int,
    rounds: int,
    batch_size: int,
    n_batches: int,
    seed: int = 42,
) -> list[Batch]:
    """Build GPU-resident batches from Stim-sampled syndromes.

    Samples syndromes, builds fired-detector graphs, creates PyG
    batches, and moves them to CUDA. Returns a list of pre-built
    batches ready for the training loop.
    """
    circuit = stim.Circuit.from_file(str(circuit_path))
    meta = extract_circuit_metadata(circuit, distance=distance, rounds=rounds)
    sampler = circuit.compile_detector_sampler(seed=seed)

    total_shots = batch_size * n_batches
    raw = sampler.sample(shots=total_shots, bit_packed=False, append_observables=True)
    n_det = meta.num_detectors
    syndromes = raw[:, :n_det].astype(np.uint8)
    observables = raw[:, n_det:].astype(np.float32)

    batches = []
    for b in range(n_batches):
        data_list = []
        start = b * batch_size
        for i in range(start, start + batch_size):
            graph = build_fired_detector_graph(syndromes[i], meta)
            data = Data(
                x=torch.from_numpy(graph.node_features),
                edge_index=torch.from_numpy(graph.edge_index),
                edge_attr=torch.from_numpy(graph.edge_features),
                y=torch.from_numpy(observables[i]),
                num_fired=torch.tensor(graph.num_fired, dtype=torch.long),
            )
            data_list.append(data)
        batches.append(Batch.from_data_list(data_list).to("cuda"))
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f"    {b + 1}/{n_batches} batches built", flush=True)
    return batches


def _build_batches_from_ci_shard(
    batch_size: int,
    n_batches: int,
) -> list[Batch]:
    """Build GPU-resident batches from the committed CI shard (d=3)."""
    syndromes = np.load(CI_SHARD_DIR / "syndromes.npy")
    observables = np.load(CI_SHARD_DIR / "observables.npy").astype(np.float32)
    coords = np.load(CI_SHARD_DIR / "detector_coords.npy")

    from sampling.graph import CircuitMetadata

    meta = CircuitMetadata(
        detector_coords=coords,
        distance=3,
        rounds=3,
        num_detectors=coords.shape[0],
    )

    n_available = len(syndromes)
    batches = []
    for b in range(n_batches):
        data_list = []
        for i in range(batch_size):
            idx = (b * batch_size + i) % n_available
            graph = build_fired_detector_graph(syndromes[idx], meta)
            data = Data(
                x=torch.from_numpy(graph.node_features),
                edge_index=torch.from_numpy(graph.edge_index),
                edge_attr=torch.from_numpy(graph.edge_features),
                y=torch.from_numpy(observables[idx]),
                num_fired=torch.tensor(graph.num_fired, dtype=torch.long),
            )
            data_list.append(data)
        batches.append(Batch.from_data_list(data_list).to("cuda"))
        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(f"    {b + 1}/{n_batches} batches built", flush=True)
    return batches


def _run_config(
    batches: list[Batch],
    distance: int,
    batch_size: int,
    use_compile: bool,
    use_amp: bool,
    hidden_dim: int = 128,
    num_layers: int = 6,
    lr: float = 1.5e-4,
    weight_decay: float = 1e-5,
) -> BenchResult:
    """Benchmark a single configuration.

    Builds a fresh model, runs warmup, then measures N_MEASURE steps
    with CUDA events.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = build_model(
        node_dim=6,
        edge_dim=5,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to("cuda")

    if use_compile:
        model = torch.compile(model, mode="default", dynamic=True, fullgraph=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    label_parts = []
    label_parts.append("compiled" if use_compile else "eager")
    label_parts.append("bf16" if use_amp else "fp32")
    label = "+".join(label_parts)

    def _step(batch: Batch) -> None:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", enabled=use_amp, dtype=torch.bfloat16
        ):
            logits = model(batch)
            loss = criterion(logits.view(-1), batch.y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    n_pool = len(batches)

    for i in range(N_WARMUP):
        _step(batches[i % n_pool])
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(N_MEASURE)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(N_MEASURE)]

    for i in range(N_MEASURE):
        b = batches[i % n_pool]
        start_events[i].record()
        _step(b)
        end_events[i].record()

    torch.cuda.synchronize()

    latencies_ms = np.array(
        [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    )

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    samples_per_sec = batch_size / (p50 / 1000.0)

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()

    return BenchResult(
        label=label,
        distance=distance,
        batch_size=batch_size,
        compiled=use_compile,
        amp=use_amp,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        samples_per_sec=samples_per_sec,
        peak_mem_mb=peak_mem_mb,
    )


def _run_profiler(
    batches: list[Batch],
    output_dir: Path,
    use_amp: bool = True,
    use_compile: bool = False,
    hidden_dim: int = 128,
    num_layers: int = 6,
) -> None:
    """Generate a torch.profiler trace for the baseline configuration."""
    torch.cuda.empty_cache()
    gc.collect()

    model = build_model(
        node_dim=6,
        edge_dim=5,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to("cuda")

    if use_compile:
        model = torch.compile(model, mode="default", dynamic=True, fullgraph=False)

    optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-5)
    criterion = FocalBCEWithLogitsLoss(alpha=0.25, gamma=2.0)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    n_batches = len(batches)

    for i in range(N_WARMUP):
        b = batches[i % n_batches]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda", enabled=use_amp, dtype=torch.bfloat16
        ):
            logits = model(b)
            loss = criterion(logits.view(-1), b.y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "compiled" if use_compile else "eager"
    amp_tag = "bf16" if use_amp else "fp32"

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(output_dir / f"profile_{tag}_{amp_tag}")
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(15):
            b = batches[i % n_batches]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=torch.bfloat16
            ):
                logits = model(b)
                loss = criterion(logits.view(-1), b.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    del model, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()


def _print_results_table(results: list[BenchResult]) -> None:
    """Print a formatted results table."""
    print()
    print(
        f"{'Config':<20} {'d':>2} {'BS':>4} "
        f"{'p50 (ms)':>10} {'p95 (ms)':>10} "
        f"{'samp/sec':>10} {'Mem (MB)':>10}"
    )
    print("-" * 80)

    for r in sorted(results, key=lambda x: (x.distance, x.batch_size, x.label)):
        print(
            f"{r.label:<20} {r.distance:>2} {r.batch_size:>4} "
            f"{r.latency_ms_p50:>10.1f} {r.latency_ms_p95:>10.1f} "
            f"{r.samples_per_sec:>10.0f} {r.peak_mem_mb:>10.0f}"
        )

    print()

    print("=== Optimization deltas (vs eager+fp32 at same d, BS) ===")
    print()
    baseline_key = {}
    for r in results:
        if r.label == "eager+fp32":
            baseline_key[(r.distance, r.batch_size)] = r

    for r in sorted(results, key=lambda x: (x.distance, x.batch_size, x.label)):
        base = baseline_key.get((r.distance, r.batch_size))
        if base is None or r.label == "eager+fp32":
            continue
        speedup = r.samples_per_sec / base.samples_per_sec
        latency_delta = (
            (r.latency_ms_p50 - base.latency_ms_p50) / base.latency_ms_p50 * 100
        )
        mem_delta = (r.peak_mem_mb - base.peak_mem_mb) / base.peak_mem_mb * 100
        print(
            f"  d={r.distance} BS={r.batch_size:>3} {r.label:<20} "
            f"speedup={speedup:.2f}x  "
            f"latency={latency_delta:+.1f}%  "
            f"mem={mem_delta:+.1f}%"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to test (default: %(default)s)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="Skip torch.profiler trace generation",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("outputs/profiler"),
        help="Directory for profiler traces (default: outputs/profiler)",
    )
    parser.add_argument(
        "--skip-d3",
        action="store_true",
        help="Skip d=3 benchmarks",
    )
    parser.add_argument(
        "--skip-d7",
        action="store_true",
        help="Skip d=7 benchmarks",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return

    torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print("GPU Training Throughput Micro-Benchmark")
    print("=" * 80)
    print(f"  Device:    {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  BF16:      {torch.cuda.is_bf16_supported()}")
    print(f"  TF32:      enabled (set_float32_matmul_precision='high')")
    print(f"  Model:     6-layer GINEConv, hidden=128")
    print(f"  Loss:      FocalBCE(alpha=0.25, gamma=2.0)")
    print(f"  Optimizer: AdamW(lr=1.5e-4, wd=1e-5)")
    print(f"  Pool:      {N_POOL} batches, Warmup: {N_WARMUP}, Measure: {N_MEASURE}")
    print()

    max_bs = max(args.batch_sizes)
    results: list[BenchResult] = []

    if not args.skip_d3:
        print("--- Building d=3 batches from CI shard ---")
        d3_batches = _build_batches_from_ci_shard(max_bs, N_POOL)

        n_nodes = sum(b.num_nodes for b in d3_batches) / len(d3_batches)
        n_edges = sum(b.edge_index.shape[1] for b in d3_batches) / len(d3_batches)
        print(
            f"  Built {len(d3_batches)} batches, avg nodes/batch={n_nodes:.0f}, edges/batch={n_edges:.0f}"
        )
        print()

        for bs in args.batch_sizes:
            if bs > max_bs:
                continue
            bs_batches = d3_batches
            if bs < max_bs:
                bs_batches = _build_batches_from_ci_shard(bs, N_POOL)

            for use_compile in [False, True]:
                for use_amp in [False, True]:
                    tag = (
                        ("compiled" if use_compile else "eager")
                        + "+"
                        + ("bf16" if use_amp else "fp32")
                    )
                    print(f"  Benchmarking d=3, BS={bs}, {tag}...", end="", flush=True)
                    r = _run_config(
                        bs_batches,
                        distance=3,
                        batch_size=bs,
                        use_compile=use_compile,
                        use_amp=use_amp,
                    )
                    results.append(r)
                    print(
                        f"  {r.samples_per_sec:.0f} samp/s, p50={r.latency_ms_p50:.1f}ms"
                    )

            if bs < max_bs:
                del bs_batches
                torch.cuda.empty_cache()

        del d3_batches
        torch.cuda.empty_cache()
        gc.collect()

    if not args.skip_d7:
        print("\n--- Building d=7 batches from Stim sampling ---")
        circuit_path = CIRCUIT_DIR / "d7_r7_p0_01.stim"
        d7_batches = _build_batches_from_stim(
            circuit_path, distance=7, rounds=7, batch_size=max_bs, n_batches=N_POOL
        )

        n_nodes = sum(b.num_nodes for b in d7_batches) / len(d7_batches)
        n_edges = sum(b.edge_index.shape[1] for b in d7_batches) / len(d7_batches)
        print(
            f"  Built {len(d7_batches)} batches, avg nodes/batch={n_nodes:.0f}, edges/batch={n_edges:.0f}"
        )
        print()

        for bs in args.batch_sizes:
            if bs > max_bs:
                continue
            bs_batches = d7_batches
            if bs < max_bs:
                bs_batches = _build_batches_from_stim(
                    circuit_path, distance=7, rounds=7, batch_size=bs, n_batches=N_POOL
                )

            for use_compile in [False, True]:
                for use_amp in [False, True]:
                    tag = (
                        ("compiled" if use_compile else "eager")
                        + "+"
                        + ("bf16" if use_amp else "fp32")
                    )
                    print(f"  Benchmarking d=7, BS={bs}, {tag}...", end="", flush=True)
                    r = _run_config(
                        bs_batches,
                        distance=7,
                        batch_size=bs,
                        use_compile=use_compile,
                        use_amp=use_amp,
                    )
                    results.append(r)
                    print(
                        f"  {r.samples_per_sec:.0f} samp/s, p50={r.latency_ms_p50:.1f}ms"
                    )

            if bs < max_bs:
                del bs_batches
                torch.cuda.empty_cache()

        if not args.no_profile:
            print("\n--- Running torch.profiler (d=7 baseline) ---")
            _run_profiler(
                d7_batches,
                args.profile_dir,
                use_amp=True,
                use_compile=False,
            )

        del d7_batches
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    _print_results_table(results)


if __name__ == "__main__":
    main()
