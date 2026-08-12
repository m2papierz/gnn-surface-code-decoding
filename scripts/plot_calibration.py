#!/usr/bin/env python3
"""
Plot reliability diagrams and ECE for trained GNN checkpoints.

Loads a trained checkpoint, runs inference on frozen eval sets to collect
logits, then produces a multi-panel reliability diagram with per-panel ECE
annotation.

Usage
-----
    uv run python scripts/plot_calibration.py \
        --checkpoint outputs/runs/memory/d3/direct/best.pt \
        --distance 3 \
        --eval-dir data/eval/memory \
        --circuit-dir data/circuits/memory \
        -o outputs/figures/calibration_d3.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
import torch
from torch_geometric.data import Batch, Data

from evaluation.calibration import expected_calibration_error, reliability_diagram
from model.decoder import build_model
from sampling.graph import build_fired_detector_graph, extract_circuit_metadata


logger = logging.getLogger(__name__)


def _collect_logits(
    model: torch.nn.Module,
    syndromes: np.ndarray,
    observables: np.ndarray,
    metadata: object,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on syndromes and collect raw logits + matched labels.

    Empty syndromes (zero fired detectors) are excluded - they short-circuit
    to no-flip and carry no calibration signal.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    syndromes : ndarray, shape ``(N, D)``
    observables : ndarray, shape ``(N, 1)``
    metadata : CircuitMetadata
    device : torch.device
    batch_size : int

    Returns
    -------
    logits : ndarray, shape ``(M,)``
        Raw logits for non-empty syndromes.
    labels : ndarray, shape ``(M,)``
        Matched binary labels.
    """
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for start in range(0, syndromes.shape[0], batch_size):
        end = min(start + batch_size, syndromes.shape[0])
        data_list: list[Data] = []
        local_labels: list[int] = []

        for i in range(start, end):
            if int(syndromes[i].sum()) == 0:
                continue
            graph = build_fired_detector_graph(syndromes[i], metadata)
            if graph.num_fired == 0:
                continue
            data_list.append(
                Data(
                    x=torch.from_numpy(graph.node_features),
                    edge_index=torch.from_numpy(graph.edge_index),
                    edge_attr=torch.from_numpy(graph.edge_features),
                    num_fired=torch.tensor(graph.num_fired, dtype=torch.long),
                )
            )
            local_labels.append(int(observables[i, 0]))

        if not data_list:
            continue

        batch = Batch.from_data_list(data_list).to(device)
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=torch.bfloat16,
            ),
        ):
            logits = model(batch)

        all_logits.append(logits.float().cpu().numpy().ravel())
        all_labels.append(np.array(local_labels, dtype=np.int64))

    return np.concatenate(all_logits), np.concatenate(all_labels)


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, float]:
    """Load model from checkpoint.

    Returns
    -------
    model, config dict, threshold
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    threshold = ckpt.get("decision_threshold", 0.0)

    from sampling.representation import DataContract

    contract = DataContract.from_checkpoint_config(cfg)
    model = build_model(
        node_dim=contract.node_dim,
        edge_dim=contract.edge_dim,
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("num_layers", 6),
        num_observables=contract.num_observables,
        dropout=0.0,
    ).to(device)

    state = ckpt["model_state_dict"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    return model, cfg, threshold


def _discover_error_probs(eval_dir: Path, distance: int) -> list[float]:
    """Find available error probs for a distance in the eval directory."""
    prefix = f"d{distance}_p"
    probs = []
    for d in sorted(eval_dir.iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            p_str = d.name[len(prefix) :]
            probs.append(float(p_str.replace("_", ".")))
    return probs


def _plot_reliability(
    panels: list[tuple[float, np.ndarray, np.ndarray, float]],
    distance: int,
    threshold: float,
    output_path: Path,
) -> None:
    """Plot multi-panel reliability diagram.

    Parameters
    ----------
    panels : list of (error_prob, bin_confidences, bin_accuracies, ece)
    distance : int
    threshold : float
    output_path : Path
    """
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)

    for ax, (p, confs, accs, ece) in zip(axes[0], panels, strict=True):
        nonempty = ~np.isnan(confs)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
        ax.bar(
            confs[nonempty],
            accs[nonempty],
            width=1.0 / len(confs),
            alpha=0.6,
            align="center",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction of positives")
        ax.set_title(f"p = {p}")
        ax.set_aspect("equal")
        ax.text(
            0.05,
            0.92,
            f"ECE = {ece:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    fig.suptitle(
        f"Reliability diagram - d = {distance}, threshold = {threshold:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot reliability diagrams from a trained GNN checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to best.pt"
    )
    parser.add_argument("--distance", type=int, required=True, help="Code distance")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/eval/memory"),
        help="Root eval set directory",
    )
    parser.add_argument(
        "--circuit-dir",
        type=Path,
        default=Path("data/circuits/memory"),
        help="Directory with .stim circuit files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: outputs/figures/calibration_d{d}.png)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=15, help="Number of bins (default: 15)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    d = args.distance
    r = d
    output_path = args.output or Path(f"outputs/figures/calibration_d{d}.png")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, cfg, threshold = _load_model(args.checkpoint, device)
    logger.info("Loaded checkpoint: %s (threshold=%.3f)", args.checkpoint, threshold)

    error_probs = _discover_error_probs(args.eval_dir, d)
    if not error_probs:
        logger.error("No eval sets found for d=%d in %s", d, args.eval_dir)
        sys.exit(1)

    panels: list[tuple[float, np.ndarray, np.ndarray, float]] = []

    for p in error_probs:
        p_str = f"{p:.4f}".replace(".", "_")
        eval_set_dir = args.eval_dir / f"d{d}_p{p_str}"
        data = np.load(eval_set_dir / "data.npz")
        syndromes = data["syndromes"].astype(np.uint8, copy=False)
        observables = data["observables"].astype(np.uint8, copy=False)
        if observables.ndim == 1:
            observables = observables[:, np.newaxis]

        p_tag = f"p{p:.6g}".replace(".", "_")
        circuit_path = args.circuit_dir / f"d{d}_r{r}_{p_tag}.stim"
        circuit = stim.Circuit.from_file(str(circuit_path))
        metadata = extract_circuit_metadata(circuit, d, r)

        logger.info("d=%d p=%.4f: %d shots", d, p, syndromes.shape[0])
        logits, labels = _collect_logits(
            model, syndromes, observables, metadata, device
        )
        logger.info("  %d non-empty shots collected", len(logits))

        diag = reliability_diagram(logits, labels, n_bins=args.n_bins)
        ece = expected_calibration_error(logits, labels, n_bins=args.n_bins)
        logger.info("  ECE = %.4f", ece)

        panels.append((p, diag.bin_confidences, diag.bin_accuracies, ece))

    _plot_reliability(panels, d, threshold, output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
