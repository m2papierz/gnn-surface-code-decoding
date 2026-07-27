"""
Unified evaluation harness: run all decoders on frozen eval sets.

Single entry point for paired decoder evaluation with adaptive stopping.
Runs GNN, MWPM, and optionally BP+OSD on identical frozen shots per (d, p)
point, producing LER with Wilson 95% CIs and McNemar test results.

Usage
-----
    # Full evaluation with a trained checkpoint
    uv run python scripts/eval_harness.py --checkpoint outputs/d3_full/direct/best.pt

    # Evaluate specific distances
    uv run python scripts/eval_harness.py --checkpoint best.pt --distances 3 5

    # Dry-run on CI shard (no GPU, no trained model needed)
    uv run python scripts/eval_harness.py --dry-run

    # Custom output path
    uv run python scripts/eval_harness.py --checkpoint best.pt \
        -o outputs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import stim
import torch

from decoders import BeliefMatchingDecoder, GNNDecoder, PyMatchingDecoder
from evaluation.evaluator import (
    EvalReport,
    EvalSet,
    discover_eval_sets,
    evaluate_point,
    load_eval_set,
)
from model.decoder import build_model
from sampling.graph import CircuitMetadata, extract_circuit_metadata


logger = logging.getLogger(__name__)

EVAL_DIR = Path("data/eval")
CI_SHARD_DIR = Path("data/ci_shard")
CIRCUIT_DIR = Path("data/circuits")


def _build_dry_run_eval_set() -> EvalSet:
    """Load CI shard as a minimal eval set for pipeline validation."""
    manifest_path = CI_SHARD_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("CI shard not found at %s", CI_SHARD_DIR)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    syndromes = np.load(CI_SHARD_DIR / "syndromes.npy").astype(np.uint8)
    observables = np.load(CI_SHARD_DIR / "observables.npy").astype(np.uint8)
    detector_coords = np.load(CI_SHARD_DIR / "detector_coords.npy").astype(np.float64)

    if observables.ndim == 1:
        observables = observables[:, np.newaxis]

    return EvalSet(
        syndromes=syndromes,
        observables=observables,
        detector_coords=detector_coords,
        distance=manifest["distance"],
        rounds=manifest["rounds"],
        error_prob=manifest["error_prob"],
        num_shots=syndromes.shape[0],
        circuit_file=manifest["circuit_file"],
        manifest=manifest,
    )


def _build_gnn_decoder_dry_run(eval_set: EvalSet) -> GNNDecoder:
    """Build a randomly initialized GNN for dry-run pipeline validation."""
    metadata = CircuitMetadata(
        detector_coords=eval_set.detector_coords,
        distance=eval_set.distance,
        rounds=eval_set.rounds,
        num_detectors=eval_set.syndromes.shape[1],
    )
    model = build_model(hidden_dim=32, num_layers=2, dropout=0.0)
    return GNNDecoder.from_metadata(
        model=model,
        metadata=metadata,
        threshold=0.0,
        device=torch.device("cpu"),
        batch_size=64,
    )


def _print_results(report: EvalReport) -> None:
    """Print evaluation results as a formatted table."""
    print()
    print("=" * 100)
    print("Evaluation Results: Multi-Decoder Paired Comparison")
    print("=" * 100)

    for result in sorted(report.results, key=lambda r: (r.distance, r.error_prob)):
        print(
            f"\n--- d={result.distance} r={result.rounds} "
            f"p={result.error_prob:.4f} | "
            f"shots={result.n_shots_used} | "
            f"outcome={result.outcome.value} ---"
        )

        header = (
            f"{'Decoder':<10} {'LER':>10} {'95% CI':>24} "
            f"{'ε (per-round)':>14} {'n_errors':>10}"
        )
        print(header)
        print("-" * len(header))

        for name, dr in result.decoder_results.items():
            print(
                f"{name:<10} {dr.ler:>10.6f} "
                f"[{dr.ler_interval.lower:.6f}, {dr.ler_interval.upper:.6f}] "
                f"{dr.per_round_ler:>14.6f} "
                f"{dr.n_errors:>10}"
            )

        if result.mcnemar_results:
            print()
            print("  McNemar (reference vs comparison):")
            for comp_name, mr in result.mcnemar_results.items():
                print(
                    f"    vs {comp_name}: χ²={mr.statistic:.4f} "
                    f"p={mr.p_value:.4e} "
                    f"(discordant={mr.n_discordant}, "
                    f"gnn_wins={mr.gnn_wins}, baseline_wins={mr.baseline_wins})"
                )

        print(f"  Stopping: {result.stopping.reason}")

    print()
    print("=" * 100)


def run_dry_run() -> EvalReport:
    """Execute dry-run evaluation on the CI shard."""
    logger.info("DRY RUN: evaluating on CI shard (%s)", CI_SHARD_DIR)

    eval_set = _build_dry_run_eval_set()
    logger.info(
        "  Loaded: d=%d r=%d p=%.4f, %d shots",
        eval_set.distance,
        eval_set.rounds,
        eval_set.error_prob,
        eval_set.num_shots,
    )

    # Build decoders
    gnn_decoder = _build_gnn_decoder_dry_run(eval_set)

    circuit_path = Path(eval_set.circuit_file)
    if not circuit_path.exists():
        circuit_path = Path.cwd() / eval_set.circuit_file
    mwpm_decoder = PyMatchingDecoder(circuit_path)

    decoders = {"gnn": gnn_decoder, "mwpm": mwpm_decoder}

    # For dry-run, use a small check interval to exercise the stopping logic
    result = evaluate_point(
        eval_set,
        decoders,
        reference_decoder="gnn",
        check_interval=eval_set.num_shots,
    )

    report = EvalReport(
        results=[result],
        metadata={
            "mode": "dry-run",
            "eval_source": str(CI_SHARD_DIR),
            "device": "cpu",
        },
    )

    return report


def run_full_eval(
    checkpoint_path: Path,
    *,
    distances: list[int] | None = None,
    error_probs: list[float] | None = None,
    batch_size: int = 256,
    output_path: Path | None = None,
    include_bp_osd: bool = True,
) -> EvalReport:
    """Execute full evaluation with a trained GNN checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Discover eval sets
    eval_dirs = discover_eval_sets(
        EVAL_DIR, distances=distances, error_probs=error_probs
    )
    if not eval_dirs:
        logger.error("No eval sets found in %s", EVAL_DIR)
        sys.exit(1)

    logger.info("Found %d eval sets", len(eval_dirs))

    # Load checkpoint once
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    threshold = ckpt.get("decision_threshold", 0.0)
    logger.info(
        "Checkpoint: %s (threshold=%.4f, samples=%d)",
        checkpoint_path,
        threshold,
        ckpt.get("samples_consumed", -1),
    )

    model = build_model(
        node_dim=cfg.get("node_dim", 6),
        edge_dim=cfg.get("edge_dim", 5),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 6),
        dropout=0.0,
    ).to(device)
    state = ckpt["model_state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    report = EvalReport(
        metadata={
            "mode": "full",
            "checkpoint": str(checkpoint_path),
            "threshold": threshold,
            "device": str(device),
            "samples_consumed": ckpt.get("samples_consumed", -1),
        },
    )

    for eval_dir in eval_dirs:
        eval_set = load_eval_set(eval_dir)
        logger.info(
            "Evaluating d=%d r=%d p=%.4f (%d shots)...",
            eval_set.distance,
            eval_set.rounds,
            eval_set.error_prob,
            eval_set.num_shots,
        )

        # Build metadata for GNN
        circuit_path = Path(eval_set.circuit_file)
        if not circuit_path.exists():
            circuit_path = Path.cwd() / eval_set.circuit_file

        circuit = stim.Circuit.from_file(str(circuit_path))
        metadata = extract_circuit_metadata(circuit, eval_set.distance, eval_set.rounds)

        gnn_decoder = GNNDecoder.from_metadata(
            model=model,
            metadata=metadata,
            threshold=threshold,
            device=device,
            batch_size=batch_size,
        )
        mwpm_decoder = PyMatchingDecoder(circuit_path)

        decoders: dict = {"gnn": gnn_decoder, "mwpm": mwpm_decoder}

        if include_bp_osd:
            try:
                bm_decoder = BeliefMatchingDecoder(circuit_path)
                decoders["belief_matching"] = bm_decoder
            except ImportError:
                logger.warning(
                    "beliefmatching not available; skipping Belief-Matching decoder"
                )

        t0 = time.perf_counter()
        result = evaluate_point(
            eval_set,
            decoders,
            reference_decoder="gnn",
            stopping_baseline="mwpm",
        )
        elapsed = time.perf_counter() - t0

        report.results.append(result)
        logger.info(
            "  Done in %.1fs: outcome=%s, shots_used=%d",
            elapsed,
            result.outcome.value,
            result.n_shots_used,
        )

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained GNN checkpoint (best.pt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on CI shard with random model (pipeline validation)",
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=None,
        help="Filter to these code distances",
    )
    parser.add_argument(
        "--error-probs",
        type=float,
        nargs="+",
        default=None,
        help="Filter to these error probabilities",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--no-bp-osd",
        action="store_true",
        help="Skip Belief-Matching decoder",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unified evaluation harness."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dry_run:
        report = run_dry_run()
    elif args.checkpoint is not None:
        report = run_full_eval(
            args.checkpoint,
            distances=args.distances,
            error_probs=args.error_probs,
            batch_size=args.batch_size,
            output_path=args.output,
            include_bp_osd=not args.no_bp_osd,
        )
    else:
        logger.error("Must specify --checkpoint or --dry-run")
        sys.exit(1)

    _print_results(report)

    if args.output is not None:
        report.save(args.output)
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
