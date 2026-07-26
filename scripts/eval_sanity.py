"""Post-training sanity evaluation: GNN vs MWPM on fresh Stim samples.

Loads a trained checkpoint, samples fresh shots from committed circuit files,
and decodes with both GNN and MWPM on identical syndromes. Reports per-p LER
with Wilson 95% confidence intervals.

Usage
-----
    # Evaluate d=3 (default 100k shots per setting)
    uv run python scripts/eval_sanity.py outputs/d3_full/direct/best.pt --distances 3

    # Evaluate d=5 and d=7 together
    uv run python scripts/eval_sanity.py outputs/d7_full/direct/best.pt --distances 5 7

    # All distances, custom shot count
    uv run python scripts/eval_sanity.py outputs/best.pt --shots 50000
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pymatching
import stim
import torch
from torch_geometric.data import Batch, Data

from evaluation.stats import wilson_interval
from model.decoder import build_model
from sampling.graph import build_fired_detector_graph, extract_circuit_metadata
from sampling.sampler import settings_from_circuit_dir


logger = logging.getLogger(__name__)

CIRCUIT_DIR = Path("data/circuits")


def evaluate_at_setting(
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="Path to best.pt checkpoint")
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=None,
        help="Code distances to evaluate (default: all in circuit dir)",
    )
    parser.add_argument(
        "--error-probs",
        type=float,
        nargs="+",
        default=None,
        help="Filter to these error probabilities",
    )
    parser.add_argument("--shots", type=int, default=100_000, help="Shots per setting")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--circuit-dir", type=Path, default=CIRCUIT_DIR)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <checkpoint_dir>/eval_sanity.json)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the unified sanity evaluation."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    threshold = ckpt.get("decision_threshold", 0.0)

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

    logger.info("Loaded checkpoint: %s", args.checkpoint)
    logger.info("  samples_consumed: %d", ckpt.get("samples_consumed", -1))
    logger.info("  decision_threshold: %.4f", threshold)
    logger.info("  device: %s", device)

    settings = settings_from_circuit_dir(
        args.circuit_dir,
        distances=args.distances,
        error_probs=args.error_probs,
    )
    dist_label = ",".join(str(d) for d in args.distances) if args.distances else "all"
    logger.info("Found %d settings for d={%s}", len(settings), dist_label)

    results = []
    for s in settings:
        logger.info(
            "Evaluating d=%d r=%d p=%.4f ...",
            s.distance,
            s.rounds,
            s.error_prob,
        )
        r = evaluate_at_setting(
            model=model,
            circuit_path=s.circuit_path,
            distance=s.distance,
            rounds=s.rounds,
            error_prob=s.error_prob,
            threshold=threshold,
            n_shots=args.shots,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
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
    print(f"Shots per setting: {args.shots}")
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Samples consumed during training: {ckpt.get('samples_consumed', -1)}")

    out_path = args.output
    if out_path is None:
        out_path = args.checkpoint.parent / "eval_sanity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
