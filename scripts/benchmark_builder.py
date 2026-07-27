"""Benchmark graph-builder throughput at d=7 scale.

Measures build_fired_detector_graph samples/sec on 1 core and on N cores
(default 8), using Stim-sampled syndromes at d=7, p=0.01 (worst-case fired
count near threshold). Compares against GPU training consumption rate to
verify the week-1 throughput gate: builder >= 2x GPU consumption.

Examples
--------
    uv run python scripts/benchmark_builder.py
    uv run python scripts/benchmark_builder.py --cores 4 --samples 50000
    uv run python scripts/benchmark_builder.py --no-gpu
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import stim

from sampling.graph import (
    CircuitMetadata,
    build_fired_detector_graph,
    extract_circuit_metadata,
)


logger = logging.getLogger(__name__)

CIRCUIT_DIR = Path("data/circuits")
DEFAULT_CIRCUIT = "d7_r7_p0_01.stim"
DEFAULT_SAMPLES = 20_000
DEFAULT_CORES = 8
WARMUP_SAMPLES = 500
BATCH_SIZE_GPU = 128


def _sample_syndromes(
    circuit_path: Path,
    n_shots: int,
    seed: int = 42,
) -> tuple[np.ndarray, CircuitMetadata]:
    """Sample syndromes from a circuit file for benchmarking.

    Parameters
    ----------
    circuit_path : Path
        Path to .stim circuit file.
    n_shots : int
        Number of syndrome shots to sample.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    syndromes : ndarray, shape (n_shots, D), uint8
    metadata : CircuitMetadata
    """
    circuit = stim.Circuit.from_file(str(circuit_path))
    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes = sampler.sample(shots=n_shots, bit_packed=False).astype(np.uint8)
    metadata = extract_circuit_metadata(circuit, distance=7, rounds=7)
    return syndromes, metadata


def _bench_single_core(
    syndromes: np.ndarray,
    metadata: CircuitMetadata,
    n_warmup: int = WARMUP_SAMPLES,
) -> float:
    """Benchmark graph builder on a single core.

    Returns
    -------
    samples_per_sec : float
    """
    n_total = len(syndromes)
    n_measure = n_total - n_warmup

    for i in range(n_warmup):
        build_fired_detector_graph(syndromes[i], metadata)

    t0 = time.perf_counter()
    for i in range(n_warmup, n_total):
        build_fired_detector_graph(syndromes[i], metadata)
    elapsed = time.perf_counter() - t0

    return n_measure / elapsed


def _worker_task(args: tuple[np.ndarray, np.ndarray, int, int]) -> float:
    """Worker function for multi-core benchmark.

    Runs build_fired_detector_graph on a slice of syndromes and returns
    elapsed time (seconds).
    """
    coords, syndromes, distance, rounds = args
    metadata = CircuitMetadata(
        detector_coords=coords,
        distance=distance,
        rounds=rounds,
        num_detectors=coords.shape[0],
    )

    t0 = time.perf_counter()
    for i in range(len(syndromes)):
        build_fired_detector_graph(syndromes[i], metadata)
    elapsed = time.perf_counter() - t0
    return elapsed


def _bench_multi_core(
    syndromes: np.ndarray,
    metadata: CircuitMetadata,
    n_cores: int,
) -> float:
    """Benchmark graph builder across multiple cores (ProcessPoolExecutor).

    Simulates DataLoader workers processing syndromes in parallel.

    Returns
    -------
    samples_per_sec : float
        Aggregate throughput across all cores.
    """
    n_total = len(syndromes)
    chunk_size = n_total // n_cores
    chunks = []
    for i in range(n_cores):
        start = i * chunk_size
        end = start + chunk_size if i < n_cores - 1 else n_total
        chunks.append(
            (
                metadata.detector_coords,
                syndromes[start:end],
                metadata.distance,
                metadata.rounds,
            )
        )

    t_wall_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_cores) as pool:
        futures = [pool.submit(_worker_task, chunk) for chunk in chunks]
        _ = [f.result() for f in futures]
    t_wall = time.perf_counter() - t_wall_start

    return n_total / t_wall


def _measure_gpu_rate() -> float | None:
    """Measure GPU training consumption rate (forward + backward).

    Builds the model, creates synthetic batches representative of d=7
    fired-detector graphs, and measures throughput at batch_size=128.

    Returns
    -------
    samples_per_sec : float or None
        None if CUDA is unavailable.
    """
    try:
        import torch
        from torch_geometric.data import Batch, Data

        from model.decoder import LogicalHead
        from model.encoder import DetectorGraphEncoder
    except ImportError:
        logger.warning("PyTorch/PyG not available; cannot measure GPU rate")
        return None

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    hidden_dim = 128
    num_layers = 6

    encoder = DetectorGraphEncoder(
        node_dim=6, edge_dim=5, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)
    head = LogicalHead(hidden_dim=hidden_dim).to(device)
    encoder.train()
    head.train()

    rng = np.random.default_rng(123)
    n_warmup = 20
    n_measure = 100
    batch_size = BATCH_SIZE_GPU
    # d=7, p=0.01: mean ~47 fired detectors (measured from Stim sampling)
    fired_mean = 47

    def _make_batch() -> Batch:
        data_list = []
        for _ in range(batch_size):
            n_nodes = max(1, int(rng.poisson(fired_mean)))
            n_edges = n_nodes * (n_nodes - 1)
            x = torch.randn(n_nodes, 6)
            ei = torch.zeros(2, n_edges, dtype=torch.long)
            if n_edges > 0:
                idx = np.arange(n_nodes, dtype=np.int64)
                src = np.repeat(idx, n_nodes)
                dst = np.tile(idx, n_nodes)
                mask = src != dst
                ei = torch.from_numpy(np.stack([src[mask], dst[mask]]))
            ea = torch.randn(n_edges, 5)
            data_list.append(Data(x=x, edge_index=ei, edge_attr=ea))
        return Batch.from_data_list(data_list).to(device)

    batches = [_make_batch() for _ in range(n_warmup + n_measure)]

    for i in range(n_warmup):
        b = batches[i]
        h, _ = encoder(b.x, b.edge_index, b.edge_attr)
        logits = head(h, b.batch)
        logits.sum().backward()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_warmup, n_warmup + n_measure):
        b = batches[i]
        h, _ = encoder(b.x, b.edge_index, b.edge_attr)
        logits = head(h, b.batch)
        logits.sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (n_measure * batch_size) / elapsed


def _print_syndrome_stats(syndromes: np.ndarray) -> None:
    """Print fired-detector count statistics."""
    fired_counts = syndromes.sum(axis=1)
    print("  Fired detector count (d=7, p=0.01):")
    print(f"    mean:   {fired_counts.mean():.1f}")
    print(f"    median: {np.median(fired_counts):.1f}")
    print(f"    p95:    {np.percentile(fired_counts, 95):.0f}")
    print(f"    p99:    {np.percentile(fired_counts, 99):.0f}")
    print(f"    p99.9:  {np.percentile(fired_counts, 99.9):.0f}")
    print(f"    max:    {fired_counts.max()}")
    print(
        f"    zero:   {(fired_counts == 0).sum()} / {len(fired_counts)} "
        f"({100 * (fired_counts == 0).mean():.1f}%)"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--circuit",
        type=Path,
        default=CIRCUIT_DIR / DEFAULT_CIRCUIT,
        help="Circuit file to benchmark (default: d7_r7_p0_01.stim)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Number of syndromes to process (default: %(default)s)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=DEFAULT_CORES,
        help="Number of parallel cores for multi-core benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU consumption rate measurement",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for syndrome sampling (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("Graph-Builder Throughput Benchmark (Week-1 Gate)")
    print("=" * 60)
    print(f"\n  Circuit: {args.circuit}")
    print(f"  Samples: {args.samples}")
    print(f"  Cores:   {args.cores}")
    print()

    # Sample syndromes
    print("Sampling syndromes from Stim...")
    syndromes, metadata = _sample_syndromes(args.circuit, args.samples, seed=args.seed)
    print(f"  Got {len(syndromes)} syndromes, {metadata.num_detectors} detectors\n")
    _print_syndrome_stats(syndromes)
    print()

    # Single-core benchmark
    print("Benchmarking single-core...")
    rate_1 = _bench_single_core(syndromes, metadata)
    print(f"  Single-core: {rate_1:,.0f} samples/sec\n")

    # Multi-core benchmark
    print(f"Benchmarking {args.cores}-core...")
    rate_n = _bench_multi_core(syndromes, metadata, args.cores)
    print(f"  {args.cores}-core:    {rate_n:,.0f} samples/sec")
    print(f"  Scaling:     {rate_n / rate_1:.2f}x (ideal: {args.cores}x)\n")

    # GPU consumption rate
    gpu_rate: float | None = None
    if not args.no_gpu:
        print("Measuring GPU training consumption rate (forward + backward)...")
        gpu_rate = _measure_gpu_rate()
        if gpu_rate is not None:
            print(
                f"  GPU rate:    {gpu_rate:,.0f} samples/sec "
                f"(batch_size={BATCH_SIZE_GPU})\n"
            )
        else:
            print("  No CUDA device available — skipping GPU measurement.\n")

    # Gate evaluation
    print("=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)
    print(f"\n  Builder ({args.cores}-core): {rate_n:,.0f} samples/sec")

    if gpu_rate is not None:
        ratio = rate_n / gpu_rate
        passed = ratio >= 2.0
        print(f"  GPU consumption:       {gpu_rate:,.0f} samples/sec")
        print(f"  Ratio (builder/GPU):   {ratio:.2f}x")
        print(f"\n  Gate (>= 2x): {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(
                f"\n  REMEDIATION NEEDED: builder must reach "
                f"{2 * gpu_rate:,.0f} samples/sec on {args.cores} cores."
            )
    else:
        print("  GPU consumption:       not measured (no CUDA)")
        print("\n  Gate cannot be evaluated without GPU measurement.")
        print("  For reference, if GPU rate were 10,000 samples/sec,")
        print(f"  ratio would be {rate_n / 10_000:.2f}x")
    print()


if __name__ == "__main__":
    main()
