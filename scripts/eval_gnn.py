"""Evaluate a trained GNN decoder against classical baselines.

Three modes:

- **harness** (default): paired evaluation on frozen eval sets with
  adaptive stopping, Wilson CIs, and McNemar tests.
- **sanity** (``--sanity``): quick post-training check on fresh Stim
  samples - GNN vs MWPM only.
- **dry-run** (``--dry-run``): pipeline validation on the CI shard
  with a randomly initialized model (no GPU, no checkpoint).

Usage
-----
    # Full evaluation from config
    uv run scripts/eval_gnn.py -c configs/eval_memory_d3_direct.yaml

    # Sanity check on fresh samples
    uv run scripts/eval_gnn.py -c configs/eval_memory_d3_direct.yaml --sanity

    # Dry-run pipeline validation
    uv run scripts/eval_gnn.py -c configs/eval_memory_d3_direct.yaml --dry-run

    # Override checkpoint via CLI
    uv run scripts/eval_gnn.py -c configs/eval_memory_d3_direct.yaml \
        --checkpoint outputs/runs/memory/d3/direct/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import stim
import torch

from decoders import (
    BeliefMatchingDecoder,
    CorrelatedMatchingDecoder,
    Decoder,
    GNNCheckpoint,
    GNNDecoder,
    PyMatchingDecoder,
    TesseractDecoder,
    load_gnn_checkpoint,
)
from evaluation.config import EvalConfig
from evaluation.evaluator import (
    EvalReport,
    discover_eval_sets,
    evaluate_point,
    load_eval_set,
)
from evaluation.stats import wilson_interval
from model.decoder import build_model
from sampling.profile import resolve_profile
from sampling.sampler import ExperimentPoint, settings_from_circuit_dir


logger = logging.getLogger(__name__)

CI_SHARD_ROOT = Path("data/ci_shard")


def _load_checkpoint(cfg: EvalConfig, device: torch.device) -> GNNCheckpoint:
    """Load the configured checkpoint under the configured operation's contract."""
    if cfg.checkpoint is None:
        logger.error("Must specify checkpoint (via config or --checkpoint)")
        sys.exit(1)

    checkpoint = load_gnn_checkpoint(
        cfg.checkpoint,
        contract=resolve_profile(cfg.operation).data_contract,
        device=device,
    )
    logger.info(
        "Checkpoint: %s (threshold=%.4f, samples=%d)",
        checkpoint.path,
        checkpoint.decision_threshold,
        checkpoint.samples_consumed,
    )
    return checkpoint


def _require_operation(cfg: EvalConfig, operation: str, source: Path) -> None:
    """Reject an artifact that belongs to another operation than the config's."""
    if operation != cfg.operation:
        raise ValueError(
            f"{source} belongs to operation {operation!r}; the config evaluates "
            f"{cfg.operation!r}"
        )


def _print_harness_results(report: EvalReport) -> None:
    """Print harness evaluation results as a formatted table."""
    print()
    print("=" * 100)
    print("Evaluation Results: Multi-Decoder Paired Comparison")
    print("=" * 100)

    for result in sorted(
        report.results, key=lambda r: (r.operation, r.distance, r.error_prob)
    ):
        include_per_round = (
            result.metric_policy is None or result.metric_policy.include_per_round_ler
        )

        print(
            f"\n--- {result.operation} | "
            f"d={result.distance} r={result.rounds} "
            f"p={result.error_prob:.4f} | "
            f"shots={result.n_shots_used} | "
            f"outcome={result.outcome.value} ---"
        )

        if include_per_round:
            header = (
                f"{'Decoder':<10} {'LER':>10} {'95% CI':>24} "
                f"{'ε (per-round)':>14} {'n_errors':>10}"
            )
        else:
            header = f"{'Decoder':<10} {'LER':>10} {'95% CI':>24} {'n_errors':>10}"
        print(header)
        print("-" * len(header))

        for name, dr in result.decoder_results.items():
            if include_per_round:
                print(
                    f"{name:<10} {dr.ler:>10.6f} "
                    f"[{dr.ler_interval.lower:.6f}, {dr.ler_interval.upper:.6f}] "
                    f"{dr.per_round_ler:>14.6f} "
                    f"{dr.n_errors:>10}"
                )
            else:
                print(
                    f"{name:<10} {dr.ler:>10.6f} "
                    f"[{dr.ler_interval.lower:.6f}, {dr.ler_interval.upper:.6f}] "
                    f"{dr.n_errors:>10}"
                )

            if dr.per_observable and len(dr.per_observable) > 1:
                for obs in dr.per_observable:
                    print(
                        f"  {obs.name:<18} {obs.ler:>10.6f} "
                        f"[{obs.ler_interval.lower:.6f}, "
                        f"{obs.ler_interval.upper:.6f}] "
                        f"{obs.n_errors:>10}"
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


def run_dry_run(cfg: EvalConfig) -> EvalReport:
    """Execute dry-run evaluation on one operation's CI shard."""
    shard_dir = CI_SHARD_ROOT / cfg.operation
    logger.info("DRY RUN: evaluating on CI shard (%s)", shard_dir)

    eval_set = load_eval_set(shard_dir)
    _require_operation(cfg, eval_set.operation, shard_dir)
    logger.info(
        "  Loaded: d=%d r=%d p=%.4f, %d shots",
        eval_set.distance,
        eval_set.rounds,
        eval_set.error_prob,
        eval_set.num_shots,
    )

    profile = resolve_profile(cfg.operation)
    contract = profile.data_contract
    model = build_model(
        node_dim=contract.node_dim,
        edge_dim=contract.edge_dim,
        hidden_dim=32,
        num_layers=2,
        num_observables=contract.num_observables,
        dropout=0.0,
    )
    gnn_decoder = GNNDecoder.from_metadata(
        model,
        eval_set.circuit_metadata,
        contract=contract,
        threshold=0.0,
        device=torch.device("cpu"),
        batch_size=64,
    )
    mwpm_decoder = PyMatchingDecoder(Path(eval_set.circuit_file))

    result = evaluate_point(
        eval_set,
        {"gnn": gnn_decoder, "mwpm": mwpm_decoder},
        reference_decoder="gnn",
        check_interval=eval_set.num_shots,
        metric_policy=profile.metric_policy,
        observable_names=profile.observable_names,
        space_time_convention=profile.space_time_convention,
    )

    return EvalReport(
        results=[result],
        metadata={
            "mode": "dry-run",
            "eval_source": str(shard_dir),
            "device": "cpu",
        },
    )


def run_harness(cfg: EvalConfig) -> EvalReport:
    """Execute full evaluation with a trained GNN checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint = _load_checkpoint(cfg, device)
    profile = resolve_profile(cfg.operation)

    eval_dir = cfg.eval_root / cfg.operation
    eval_dirs = discover_eval_sets(
        eval_dir, distances=cfg.distances, error_probs=cfg.error_probs
    )
    if not eval_dirs:
        logger.error("No eval sets found in %s", eval_dir)
        sys.exit(1)

    logger.info("Found %d eval sets", len(eval_dirs))

    report = EvalReport(
        metadata={
            "mode": "full",
            "checkpoint": str(checkpoint.path),
            "threshold": checkpoint.decision_threshold,
            "device": str(device),
            "samples_consumed": checkpoint.samples_consumed,
        },
    )

    for ed in eval_dirs:
        eval_set = load_eval_set(ed)
        _require_operation(cfg, eval_set.operation, ed)
        logger.info(
            "Evaluating %s d=%d r=%d p=%.4f (%d shots)...",
            eval_set.operation,
            eval_set.distance,
            eval_set.rounds,
            eval_set.error_prob,
            eval_set.num_shots,
        )

        circuit_path = Path(eval_set.circuit_file)
        decoders: dict[str, Decoder] = {
            "gnn": GNNDecoder(
                checkpoint.model,
                eval_set.circuit_metadata,
                contract=checkpoint.contract,
                threshold=checkpoint.decision_threshold,
                device=device,
                batch_size=cfg.batch_size,
            ),
            "mwpm": PyMatchingDecoder(circuit_path),
            "correlated": CorrelatedMatchingDecoder(circuit_path),
        }
        if cfg.include_belief_matching:
            decoders["belief_matching"] = BeliefMatchingDecoder(circuit_path)
        if cfg.include_tesseract:
            decoders["tesseract"] = TesseractDecoder(circuit_path)

        t0 = time.perf_counter()
        result = evaluate_point(
            eval_set,
            decoders,
            reference_decoder="gnn",
            stopping_baseline="correlated",
            metric_policy=profile.metric_policy,
            observable_names=profile.observable_names,
            space_time_convention=profile.space_time_convention,
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


def _evaluate_at_setting(
    point: ExperimentPoint,
    checkpoint: GNNCheckpoint,
    cfg: EvalConfig,
    device: torch.device,
) -> dict:
    """Evaluate GNN and MWPM on fresh shots from one circuit.

    Parameters
    ----------
    point : ExperimentPoint
        Circuit to sample, of the config's operation.
    checkpoint : GNNCheckpoint
        Trained GNN, restored under that operation's contract.
    cfg : EvalConfig
        Supplies the shot count, sampler seed and inference batch size.
    device : torch.device
        Device for GNN inference.

    Returns
    -------
    dict
        Per-setting results with LER and Wilson intervals for both decoders.
    """
    circuit = stim.Circuit.from_file(str(point.circuit_path))
    metadata = resolve_profile(cfg.operation).metadata_extractor(
        point.circuit_path, num_observables=checkpoint.contract.num_observables
    )

    sampler = circuit.compile_detector_sampler(seed=cfg.seed)
    dets, obs = sampler.sample(
        shots=cfg.shots, separate_observables=True, bit_packed=False
    )
    syndromes = dets.astype(np.uint8)
    observables = obs.astype(np.uint8)

    gnn_decoder = GNNDecoder(
        checkpoint.model,
        metadata,
        contract=checkpoint.contract,
        threshold=checkpoint.decision_threshold,
        device=device,
        batch_size=cfg.batch_size,
    )
    gnn_pred = gnn_decoder.decode_batch(syndromes)
    mwpm_pred = PyMatchingDecoder(point.circuit_path).decode_batch(syndromes)
    gnn_errors = int(np.any(gnn_pred != observables, axis=1).sum())
    mwpm_errors = int(np.any(mwpm_pred != observables, axis=1).sum())

    gnn_ler = gnn_errors / cfg.shots
    mwpm_ler = mwpm_errors / cfg.shots
    gnn_ci = wilson_interval(gnn_errors, cfg.shots)
    mwpm_ci = wilson_interval(mwpm_errors, cfg.shots)

    return {
        "distance": point.distance,
        "rounds": point.rounds,
        "error_prob": point.error_prob,
        "n_shots": cfg.shots,
        "gnn_errors": gnn_errors,
        "gnn_ler": gnn_ler,
        "gnn_ci_95": [gnn_ci.lower, gnn_ci.upper],
        "mwpm_errors": mwpm_errors,
        "mwpm_ler": mwpm_ler,
        "mwpm_ci_95": [mwpm_ci.lower, mwpm_ci.upper],
        "gnn_le_mwpm": gnn_ler <= mwpm_ler,
        "threshold": checkpoint.decision_threshold,
    }


def run_sanity(cfg: EvalConfig) -> list[dict]:
    """Execute sanity evaluation on fresh Stim samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(cfg, device)

    settings = settings_from_circuit_dir(
        cfg.circuit_dir,
        operation=cfg.operation,
        distances=cfg.distances,
        error_probs=cfg.error_probs,
    )
    dist_label = ",".join(str(d) for d in cfg.distances) if cfg.distances else "all"
    logger.info("Found %d settings for d={%s}", len(settings), dist_label)

    results = []
    for s in settings:
        logger.info(
            "Evaluating d=%d r=%d p=%.4f ...",
            s.distance,
            s.rounds,
            s.error_prob,
        )
        r = _evaluate_at_setting(s, checkpoint, cfg, device)
        results.append(r)
        parity = "PASS" if r["gnn_le_mwpm"] else "FAIL"
        logger.info(
            "  GNN LER=%.6f [%.6f, %.6f]  MWPM LER=%.6f [%.6f, %.6f]  %s",
            r["gnn_ler"],
            r["gnn_ci_95"][0],
            r["gnn_ci_95"][1],
            r["mwpm_ler"],
            r["mwpm_ci_95"][0],
            r["mwpm_ci_95"][1],
            parity,
        )

    _print_sanity_results(results, cfg, checkpoint)
    return results


def _print_sanity_results(
    results: list[dict],
    cfg: EvalConfig,
    checkpoint: GNNCheckpoint,
) -> None:
    """Print sanity evaluation results as a formatted table."""
    distances_seen = sorted({r["distance"] for r in results})
    print()
    print("=" * 80)
    print(f"Sanity Evaluation (d={distances_seen}): GNN vs MWPM")
    print("=" * 80)
    print(
        f"{'d':>3} {'p':>8} {'GNN LER':>12} {'GNN 95% CI':>20} "
        f"{'MWPM LER':>12} {'MWPM 95% CI':>20} {'Parity':>8}"
    )
    print("-" * 85)
    for r in sorted(results, key=lambda x: (x["distance"], x["error_prob"])):
        parity = "PASS" if r["gnn_le_mwpm"] else "FAIL"
        print(
            f"{r['distance']:>3} {r['error_prob']:>8.4f} "
            f"{r['gnn_ler']:>12.6f} "
            f"[{r['gnn_ci_95'][0]:.6f}, {r['gnn_ci_95'][1]:.6f}] "
            f"{r['mwpm_ler']:>12.6f} "
            f"[{r['mwpm_ci_95'][0]:.6f}, {r['mwpm_ci_95'][1]:.6f}] "
            f"{parity:>8}"
        )
    print()
    print(f"Shots per setting: {cfg.shots}")
    print(f"Decision threshold: {checkpoint.decision_threshold:.4f}")
    print(f"Samples consumed during training: {checkpoint.samples_consumed}")


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[EvalConfig, str, Path | None]:
    """Parse CLI arguments into an :class:`EvalConfig`.

    CLI args override values loaded from YAML config.

    Returns
    -------
    tuple of (EvalConfig, mode, output path)
        mode is one of ``"harness"``, ``"sanity"``, ``"dry-run"``.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="YAML config file (CLI args override config values)",
    )
    parser.add_argument("--operation", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on CI shard with random model (pipeline validation)",
    )
    mode_group.add_argument(
        "--sanity",
        action="store_true",
        help="Quick sanity check on fresh Stim samples (GNN vs MWPM)",
    )

    parser.add_argument("--distances", type=int, nargs="+", default=None)
    parser.add_argument("--error-probs", type=float, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--circuit-dir", type=Path, default=None)
    parser.add_argument("--no-bp-osd", action="store_true", help="Skip Belief-Matching")
    parser.add_argument("--tesseract", action="store_true", default=None)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fields: dict[str, Any] = {}
    if args.config is not None and args.config.is_file():
        fields = EvalConfig.read_yaml_fields(args.config)

    field_map: dict[str, object] = {
        "operation": args.operation,
        "checkpoint": args.checkpoint,
        "eval_root": args.eval_root,
        "circuit_dir": args.circuit_dir,
        "distances": args.distances,
        "error_probs": args.error_probs,
        "batch_size": args.batch_size,
        "shots": args.shots,
        "seed": args.seed,
    }

    if args.no_bp_osd:
        field_map["include_belief_matching"] = False
    if args.tesseract:
        field_map["include_tesseract"] = True

    fields.update({key: value for key, value in field_map.items() if value is not None})

    if args.dry_run:
        mode = "dry-run"
    elif args.sanity:
        mode = "sanity"
    else:
        mode = "harness"

    return EvalConfig.from_fields(fields), mode, args.output


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN evaluation."""
    cfg, mode, output = parse_args(argv)

    if mode == "dry-run":
        report = run_dry_run(cfg)
        _print_harness_results(report)
        if output is not None:
            report.save(output)
            logger.info("Results saved to %s", output)

    elif mode == "sanity":
        results = run_sanity(cfg)
        out_path = output
        if out_path is None and cfg.checkpoint is not None:
            out_path = cfg.checkpoint.parent / "eval_sanity.json"
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nResults saved to {out_path}")

    else:
        report = run_harness(cfg)
        _print_harness_results(report)
        if output is not None:
            report.save(output)
            logger.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
