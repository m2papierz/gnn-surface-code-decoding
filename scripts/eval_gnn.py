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

import numpy as np
import pymatching
import stim
import torch
from torch_geometric.data import Batch, Data

from decoders import (
    BeliefMatchingDecoder,
    CorrelatedMatchingDecoder,
    GNNDecoder,
    PyMatchingDecoder,
    TesseractDecoder,
)
from evaluation.config import EvalConfig
from evaluation.evaluator import (
    EvalReport,
    EvalSet,
    discover_eval_sets,
    evaluate_point,
    load_eval_set,
)
from evaluation.stats import wilson_interval
from model.decoder import build_model
from sampling.graph import build_fired_detector_graph, extract_circuit_metadata
from sampling.representation import DataContract
from sampling.sampler import settings_from_circuit_dir


logger = logging.getLogger(__name__)

EVAL_ROOT = Path("data/eval")
CI_SHARD_ROOT = Path("data/ci_shard")


def _load_model(
    cfg: EvalConfig, device: torch.device
) -> tuple[torch.nn.Module, DataContract, float, dict]:
    """Load a trained GNN checkpoint.

    Returns
    -------
    tuple of (model, contract, threshold, raw checkpoint dict)
    """
    if cfg.checkpoint is None:
        logger.error("Must specify checkpoint (via config or --checkpoint)")
        sys.exit(1)

    ckpt = torch.load(cfg.checkpoint, weights_only=False, map_location=device)
    ckpt_cfg = ckpt["config"]
    threshold = ckpt.get("decision_threshold", 0.0)

    contract = DataContract.from_checkpoint_config(ckpt_cfg)
    model = build_model(
        node_dim=contract.node_dim,
        edge_dim=contract.edge_dim,
        hidden_dim=ckpt_cfg.get("hidden_dim", 128),
        num_layers=ckpt_cfg.get("num_layers", 6),
        num_observables=contract.num_observables,
        dropout=0.0,
    ).to(device)
    state = ckpt["model_state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    logger.info(
        "Checkpoint: %s (threshold=%.4f, samples=%d)",
        cfg.checkpoint,
        threshold,
        ckpt.get("samples_consumed", -1),
    )

    return model, contract, threshold, ckpt


def _build_dry_run_eval_set(shard_dir: Path) -> EvalSet:
    """Load a CI shard as a minimal eval set for pipeline validation."""
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("CI shard not found at %s", shard_dir)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    syndromes = np.load(shard_dir / "syndromes.npy").astype(np.uint8)
    observables = np.load(shard_dir / "observables.npy").astype(np.uint8)
    detector_coords = np.load(shard_dir / "detector_coords.npy").astype(np.float64)

    if observables.ndim == 1:
        observables = observables[:, np.newaxis]

    circuit = stim.Circuit.from_file(manifest["circuit_file"])
    base_meta = extract_circuit_metadata(
        circuit, manifest["distance"], manifest["rounds"]
    )

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
        dem_edge_weights=base_meta.dem_edge_weights,
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
    from sampling.profile import resolve_profile

    shard_dir = CI_SHARD_ROOT / cfg.operation
    logger.info("DRY RUN: evaluating on CI shard (%s)", shard_dir)

    eval_set = _build_dry_run_eval_set(shard_dir)
    logger.info(
        "  Loaded: d=%d r=%d p=%.4f, %d shots",
        eval_set.distance,
        eval_set.rounds,
        eval_set.error_prob,
        eval_set.num_shots,
    )

    profile = resolve_profile(cfg.operation)
    model = build_model(
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        num_observables=profile.data_contract.num_observables,
    )
    gnn_decoder = GNNDecoder.from_metadata(
        model=model,
        metadata=eval_set.circuit_metadata,
        threshold=0.0,
        device=torch.device("cpu"),
        batch_size=64,
        contract=profile.data_contract,
    )

    circuit_path = Path(eval_set.circuit_file)
    if not circuit_path.exists():
        circuit_path = Path.cwd() / eval_set.circuit_file
    mwpm_decoder = PyMatchingDecoder(circuit_path)

    decoders = {"gnn": gnn_decoder, "mwpm": mwpm_decoder}

    result = evaluate_point(
        eval_set,
        decoders,
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
    from sampling.profile import resolve_profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model, contract, threshold, ckpt = _load_model(cfg, device)
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
            "checkpoint": str(cfg.checkpoint),
            "threshold": threshold,
            "device": str(device),
            "samples_consumed": ckpt.get("samples_consumed", -1),
        },
    )

    for ed in eval_dirs:
        eval_set = load_eval_set(ed)
        logger.info(
            "Evaluating %s d=%d r=%d p=%.4f (%d shots)...",
            eval_set.operation,
            eval_set.distance,
            eval_set.rounds,
            eval_set.error_prob,
            eval_set.num_shots,
        )

        circuit_path = Path(eval_set.circuit_file)
        if not circuit_path.exists():
            circuit_path = Path.cwd() / eval_set.circuit_file

        gnn_decoder = GNNDecoder.from_metadata(
            model=model,
            metadata=eval_set.circuit_metadata,
            threshold=threshold,
            device=device,
            batch_size=cfg.batch_size,
            contract=contract,
        )
        mwpm_decoder = PyMatchingDecoder(circuit_path)
        correlated_decoder = CorrelatedMatchingDecoder(circuit_path)

        decoders: dict = {
            "gnn": gnn_decoder,
            "mwpm": mwpm_decoder,
            "correlated": correlated_decoder,
        }

        if cfg.include_belief_matching:
            try:
                bm_decoder = BeliefMatchingDecoder(circuit_path)
                decoders["belief_matching"] = bm_decoder
            except ImportError:
                logger.warning(
                    "beliefmatching not available; skipping Belief-Matching decoder"
                )

        if cfg.include_tesseract:
            try:
                tess_decoder = TesseractDecoder(circuit_path)
                decoders["tesseract"] = tess_decoder
            except ImportError:
                logger.warning(
                    "tesseract-decoder not available; skipping Tesseract decoder"
                )

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
    model: torch.nn.Module,
    circuit_path: Path,
    distance: int,
    rounds: int,
    error_prob: float,
    threshold: float,
    n_shots: int,
    device: torch.device,
    batch_size: int = 256,
    seed: int = 99,
) -> dict:
    """Evaluate GNN and MWPM on fresh shots from a single circuit.

    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN model in eval mode.
    circuit_path : Path
        Path to the ``.stim`` circuit file.
    distance, rounds : int
        Code distance and syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    threshold : float
        GNN decision threshold (logit-space).
    n_shots : int
        Number of shots to sample and decode.
    device : torch.device
        Device for GNN inference.
    batch_size : int
        Batch size for GNN decoding.
    seed : int
        Stim sampler seed.

    Returns
    -------
    dict
        Per-setting results with LER and Wilson intervals for both decoders.
    """
    circuit = stim.Circuit.from_file(str(circuit_path))
    dem = circuit.detector_error_model(decompose_errors=True)
    meta = extract_circuit_metadata(circuit, distance, rounds)
    matching = pymatching.Matching.from_detector_error_model(dem)

    sampler = circuit.compile_detector_sampler(seed=seed)
    raw = sampler.sample(shots=n_shots, bit_packed=False, append_observables=True)

    n_det = dem.num_detectors
    n_obs = dem.num_observables
    syndromes = raw[:, :n_det].astype(np.uint8)
    observables = raw[:, n_det : n_det + n_obs].astype(np.uint8)

    mwpm_pred = matching.decode_batch(syndromes)[:, :n_obs]
    mwpm_errors = int(np.any(mwpm_pred != observables, axis=1).sum())

    model.eval()
    gnn_errors = 0
    use_amp = device.type == "cuda"

    for start in range(0, n_shots, batch_size):
        end = min(start + batch_size, n_shots)
        data_list = []
        for i in range(start, end):
            graph = build_fired_detector_graph(syndromes[i], meta)
            data = Data(
                x=torch.from_numpy(graph.node_features),
                edge_index=torch.from_numpy(graph.edge_index),
                edge_attr=torch.from_numpy(graph.edge_features),
                y=torch.from_numpy(observables[i].astype(np.float32)),
                num_fired=torch.tensor(graph.num_fired, dtype=torch.long),
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list).to(device)
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=torch.bfloat16
            ),
        ):
            logits = model(batch)

        pred = (logits > threshold).float()
        target = batch.y.view_as(pred)
        gnn_errors += int((pred != target).any(dim=1).sum().item())

    gnn_ler = gnn_errors / n_shots
    mwpm_ler = mwpm_errors / n_shots
    gnn_ci = wilson_interval(gnn_errors, n_shots)
    mwpm_ci = wilson_interval(mwpm_errors, n_shots)

    return {
        "distance": distance,
        "rounds": rounds,
        "error_prob": error_prob,
        "n_shots": n_shots,
        "gnn_errors": gnn_errors,
        "gnn_ler": gnn_ler,
        "gnn_ci_95": [gnn_ci.lower, gnn_ci.upper],
        "mwpm_errors": mwpm_errors,
        "mwpm_ler": mwpm_ler,
        "mwpm_ci_95": [mwpm_ci.lower, mwpm_ci.upper],
        "gnn_le_mwpm": gnn_ler <= mwpm_ler,
        "threshold": threshold,
    }


def run_sanity(cfg: EvalConfig) -> list[dict]:
    """Execute sanity evaluation on fresh Stim samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _contract, threshold, ckpt = _load_model(cfg, device)

    assert cfg.circuit_dir is not None
    settings = settings_from_circuit_dir(
        cfg.circuit_dir,
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
        r = _evaluate_at_setting(
            model=model,
            circuit_path=s.circuit_path,
            distance=s.distance,
            rounds=s.rounds,
            error_prob=s.error_prob,
            threshold=threshold,
            n_shots=cfg.shots,
            device=device,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )
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

    _print_sanity_results(results, cfg, threshold, ckpt)
    return results


def _print_sanity_results(
    results: list[dict],
    cfg: EvalConfig,
    threshold: float,
    ckpt: dict,
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
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Samples consumed during training: {ckpt.get('samples_consumed', -1)}")


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

    if args.config is not None and args.config.is_file():
        cfg = EvalConfig.from_yaml(args.config)
    else:
        cfg = EvalConfig()

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

    cfg_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
    for key, value in field_map.items():
        if value is not None:
            cfg_dict[key] = value

    for key in ("checkpoint", "eval_root", "circuit_dir"):
        if key in cfg_dict and cfg_dict[key] is not None:
            cfg_dict[key] = Path(cfg_dict[key])

    if args.dry_run:
        mode = "dry-run"
    elif args.sanity:
        mode = "sanity"
    else:
        mode = "harness"

    return EvalConfig(**cfg_dict), mode, args.output


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
