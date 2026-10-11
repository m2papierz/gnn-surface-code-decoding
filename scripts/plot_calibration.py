#!/usr/bin/env python3
"""
Plot reliability diagrams and ECE of a trained GNN checkpoint.

Restores the checkpoint under the operation's data contract, samples fresh
shots from each of the operation's committed circuits at one distance, and
collects logits through the inference path the evaluation decodes with.  Every
observable of every point gets its own reliability diagram: each non-empty
bin's fraction of positives with its Wilson 95% interval, and the ECE with the
shots it was computed on.  Shots without a detection event never reach the
model and are excluded from the bins.  Frozen eval shots are never read, so no
calibration is ever drawn on the shots the decoders are scored on.

The figure is written to ``-o``; the binned data it is drawn from is written
beside it as ``<stem>.json``, ``{metadata, points}``, naming the checkpoint by
path and by the sha256 of the loaded bytes.

Usage
-----
    uv run scripts/plot_calibration.py --operation zz_merge_split \
        --checkpoint outputs/runs/zz_merge_split/d3/<run_id>/best.pt \
        --distance 3 -o outputs/figures/calibration_zz_merge_split_d3.png
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from artifact_file import atomic_write, write_json
from decoders import GNNCheckpoint, GNNDecoder, load_gnn_checkpoint
from evaluation.calibration import ReliabilityDiagram, reliability_diagram
from sampling.experiment import ExperimentKey
from sampling.profile import registered_operations, resolve_profile
from sampling.sampler import ExperimentPoint, sample_shots, settings_from_circuit_dir
from sampling.seeding import stable_seed


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class CalibrationSettings:
    """How every point is sampled, binned and inferred.

    Parameters
    ----------
    shots : int
        Shots sampled per point.
    seed : int
        Master seed; each point samples from its own stream derived from it.
    n_bins : int
        Equal-width probability bins per diagram.
    batch_size : int
        GNN inference batch size.

    Raises
    ------
    ValueError
        If ``shots``, ``n_bins`` or ``batch_size`` is below 1.
    """

    shots: int
    seed: int
    n_bins: int
    batch_size: int

    def __post_init__(self) -> None:
        for name in ("shots", "n_bins", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")


@dataclasses.dataclass(frozen=True, slots=True)
class PointCalibration:
    """Reliability diagrams of every observable at one experiment point.

    Parameters
    ----------
    key : ExperimentKey
        The point the shots were sampled at.
    n_shots : int
        Shots sampled.
    n_shots_without_detection_events : int
        Sampled shots with no detection event, excluded from every diagram.
    observable_names : tuple of str
        The operation's ordered observable names.
    diagrams : tuple of ReliabilityDiagram
        One diagram per observable, in ``observable_names`` order.
    """

    key: ExperimentKey
    n_shots: int
    n_shots_without_detection_events: int
    observable_names: tuple[str, ...]
    diagrams: tuple[ReliabilityDiagram, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "operation": self.key.operation,
            "distance": self.key.distance,
            "rounds": self.key.rounds,
            "error_prob": self.key.error_prob,
            "n_shots": self.n_shots,
            "n_shots_without_detection_events": self.n_shots_without_detection_events,
            "observables": [
                {"name": name, **diagram.to_dict()}
                for name, diagram in zip(
                    self.observable_names, self.diagrams, strict=True
                )
            ],
        }


def calibrate_point(
    point: ExperimentPoint, decoder: GNNDecoder, settings: CalibrationSettings
) -> PointCalibration:
    """Sample one point's circuit and bin the decoder's logits per observable.

    Parameters
    ----------
    point : ExperimentPoint
        The committed circuit to sample.
    decoder : GNNDecoder
        Decoder built on the point's circuit metadata.
    settings : CalibrationSettings
        Shots, master seed and bins.

    Returns
    -------
    PointCalibration

    Raises
    ------
    ValueError
        If no sampled shot carries a detection event.
    """
    syndromes, observables = sample_shots(
        point.circuit_path,
        shots=settings.shots,
        seed=stable_seed("calibration", str(point.key), base=settings.seed),
    )
    fired = syndromes.any(axis=1)
    n_fired = int(fired.sum())
    if n_fired == 0:
        raise ValueError(
            f"{point.key}: none of {settings.shots} shots carries a detection "
            f"event; sample more shots"
        )
    logits = decoder.logits(syndromes[fired])
    labels = observables[fired]
    observable_names = resolve_profile(point.operation).observable_names
    return PointCalibration(
        key=point.key,
        n_shots=settings.shots,
        n_shots_without_detection_events=settings.shots - n_fired,
        observable_names=observable_names,
        diagrams=tuple(
            reliability_diagram(logits[:, k], labels[:, k], n_bins=settings.n_bins)
            for k in range(len(observable_names))
        ),
    )


def calibrate(
    points: Sequence[ExperimentPoint],
    checkpoint: GNNCheckpoint,
    settings: CalibrationSettings,
    device: torch.device,
) -> list[PointCalibration]:
    """Calibrate the checkpoint at every point, each on its circuit's metadata.

    Parameters
    ----------
    points : sequence of ExperimentPoint
        Committed circuits of the checkpoint's operation.
    checkpoint : GNNCheckpoint
        Model restored under that operation's contract.
    settings : CalibrationSettings
        Shots, master seed, bins and inference batch size.
    device : torch.device
        Device the model runs on.

    Returns
    -------
    list of PointCalibration
        In ``points`` order.
    """
    calibrations = []
    for point in points:
        circuit_metadata = resolve_profile(point.operation).metadata_extractor(
            point.circuit_path, num_observables=checkpoint.contract.num_observables
        )
        decoder = GNNDecoder(
            checkpoint.model,
            circuit_metadata,
            contract=checkpoint.contract,
            threshold=checkpoint.decision_threshold,
            device=device,
            batch_size=settings.batch_size,
        )
        calibration = calibrate_point(point, decoder, settings)
        logger.info(
            "%s: ECE %s over %d of %d shots",
            point.key,
            ", ".join(f"{diagram.ece:.4f}" for diagram in calibration.diagrams),
            settings.shots - calibration.n_shots_without_detection_events,
            settings.shots,
        )
        calibrations.append(calibration)
    return calibrations


def _draw_diagram(ax: Axes, diagram: ReliabilityDiagram) -> None:
    """Draw one reliability diagram: bins with Wilson 95% bars, and the ECE."""
    nonempty = np.flatnonzero(diagram.bin_counts)
    intervals = [diagram.accuracy_interval(b) for b in nonempty]
    accuracies = diagram.bin_accuracies[nonempty]
    lower = np.array([interval.lower for interval in intervals])
    upper = np.array([interval.upper for interval in intervals])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax.errorbar(
        diagram.bin_confidences[nonempty],
        accuracies,
        yerr=np.vstack([accuracies - lower, upper - accuracies]),
        fmt="o",
        markersize=3,
        capsize=2,
        linewidth=1,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.text(
        0.05,
        0.92,
        f"ECE = {diagram.ece:.4f}\nn = {int(diagram.bin_counts.sum()):,}",
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )


def plot_calibration(
    calibrations: Sequence[PointCalibration],
    checkpoint: GNNCheckpoint,
    output_path: Path,
) -> None:
    """Draw one row per observable and one column per point, and save.

    Parameters
    ----------
    calibrations : sequence of PointCalibration
        Points of one operation, in column order.
    checkpoint : GNNCheckpoint
        The checkpoint the logits came from, named in the title.
    output_path : Path
        Figure path; the format follows its suffix.
    """
    observable_names = calibrations[0].observable_names
    fig, axes = plt.subplots(
        len(observable_names),
        len(calibrations),
        figsize=(3.2 * len(calibrations), 3.2 * len(observable_names)),
        squeeze=False,
    )
    for column, calibration in enumerate(calibrations):
        axes[0, column].set_title(f"p = {calibration.key.error_prob:g}")
        axes[-1, column].set_xlabel("Mean predicted probability")
        for row, diagram in enumerate(calibration.diagrams):
            _draw_diagram(axes[row, column], diagram)
    for row, name in enumerate(observable_names):
        axes[row, 0].set_ylabel(f"{name}\nFraction of positives")

    key = calibrations[0].key
    fig.suptitle(
        f"Reliability - {key.operation}, d = {key.distance}, r = {key.rounds} "
        f"({checkpoint.path.parent.name}, threshold = "
        f"{checkpoint.decision_threshold:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    with atomic_write(output_path) as temp:
        fig.savefig(temp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--operation", required=True, choices=sorted(registered_operations())
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--shots", type=int, default=100_000, help="Per point.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Figure path (default: "
        "outputs/figures/calibration_<operation>_d<distance>.png); the "
        "binned data is written beside it as <stem>.json.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path(
            f"outputs/figures/calibration_{args.operation}_d{args.distance}.png"
        )
    if args.output.suffix == ".json":
        parser.error(f"--output {args.output} is the figure; its data is the .json")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    settings = CalibrationSettings(
        shots=args.shots, seed=args.seed, n_bins=args.n_bins, batch_size=args.batch_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = resolve_profile(args.operation)
    checkpoint = load_gnn_checkpoint(
        args.checkpoint, contract=profile.data_contract, device=device
    )
    points = settings_from_circuit_dir(
        profile.circuit_root, operation=args.operation, distances=[args.distance]
    )
    calibrations = calibrate(points, checkpoint, settings, device)

    metadata = {
        "source": str(profile.circuit_root),
        "operation": args.operation,
        "settings": dataclasses.asdict(settings),
        "checkpoint": checkpoint.provenance(),
        "device": str(device),
    }
    write_json(
        args.output.with_suffix(".json"),
        {"metadata": metadata, "points": [c.to_dict() for c in calibrations]},
    )
    plot_calibration(calibrations, checkpoint, args.output)


if __name__ == "__main__":
    main()
