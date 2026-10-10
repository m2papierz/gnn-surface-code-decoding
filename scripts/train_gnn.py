"""Train a GNN-based QEC decoder.

The config file names the experiment and every hyperparameter; CLI flags
override single values.  Several ``distances`` train one model across them.
The run directory is named by the resolved config and its circuits, so a run
is never overwritten: training a spec that already has a run is refused, and
``--resume`` continues that run from its last checkpoint.

Examples
--------
    # Train from config
    uv run scripts/train_gnn.py -c configs/train_memory_d3.yaml

    # Override distances and budget
    uv run scripts/train_gnn.py -c configs/train.yaml \
        --distances 3 --sample-budget 2000000

    # Continue the killed run of the same config and overrides
    uv run scripts/train_gnn.py -c configs/train_memory_d3.yaml --resume
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from training import TrainConfig, Trainer


def parse_args(argv: Sequence[str] | None = None) -> tuple[TrainConfig, bool]:
    """Parse CLI arguments into a :class:`TrainConfig`.

    CLI args override values loaded from YAML config.

    Returns
    -------
    tuple of (TrainConfig, bool)
        The config, and whether to resume its run rather than start it.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="YAML config file (CLI args override config values)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["pytorch", "compiled"],
        help="Compute backend (cuda is inference-only)",
    )
    parser.add_argument("--compile-mode", type=str, default=None)
    parser.add_argument("--circuit-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--sample-budget", type=int, default=None)
    parser.add_argument("--val-interval-samples", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue this config's unfinished run from its last checkpoint",
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=None,
        help="Train on these code distances only (default: all)",
    )
    parser.add_argument("--include-p-feature", action="store_true", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fields = TrainConfig.read_yaml_fields(args.config)
    field_map = {
        "circuit_dir": args.circuit_dir,
        "output_dir": args.output_dir,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_grad_norm": args.max_grad_norm,
        "patience": args.patience,
        "sample_budget": args.sample_budget,
        "val_interval_samples": args.val_interval_samples,
        "val_size": args.val_size,
        "warmup_fraction": args.warmup_fraction,
        "seed": args.seed,
        "backend": args.backend,
        "compile_mode": args.compile_mode,
        "distances": args.distances,
        "include_p_feature": args.include_p_feature,
    }

    fields.update({key: value for key, value in field_map.items() if value is not None})
    return TrainConfig.from_fields(fields), args.resume


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN training."""
    cfg, resume = parse_args(argv)
    trainer = Trainer(cfg)
    if resume:
        trainer.resume()
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
