"""Train a GNN-based QEC decoder.

Dispatches to direct training or curriculum training based on the config:
if the YAML contains a ``curriculum:`` section, runs ``CurriculumTrainer``;
otherwise runs ``Trainer``.

Examples
--------
    # Direct training from config
    uv run scripts/train_gnn.py -c configs/train_memory_d3_direct.yaml

    # Curriculum training (detected from config)
    uv run scripts/train_gnn.py -c configs/train_memory_mixed_curriculum.yaml

    # Override distances and budget
    uv run scripts/train_gnn.py -c configs/train.yaml \
        --distances 3 --sample-budget 2000000

    # Resume from checkpoint
    uv run scripts/train_gnn.py -c configs/train.yaml \
        --resume outputs/runs/memory/d3/direct/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import yaml

from training import CurriculumConfig, CurriculumTrainer, TrainConfig, Trainer


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[TrainConfig, CurriculumConfig | None]:
    """Parse CLI arguments into a :class:`TrainConfig` and optional curriculum.

    CLI args override values loaded from YAML config.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
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
    parser.add_argument("--resume", type=Path, default=None)
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

    # Load YAML and detect curriculum
    curriculum_cfg: CurriculumConfig | None = None
    if args.config is not None and args.config.is_file():
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        cfg = TrainConfig.from_yaml(args.config)
        if "curriculum" in raw:
            curriculum_cfg = CurriculumConfig.from_yaml(raw["curriculum"])
    elif args.config is not None and not args.config.is_file():
        logging.getLogger(__name__).warning(
            "Config file %s not found, using defaults",
            args.config,
        )
        cfg = TrainConfig()
    else:
        cfg = TrainConfig()

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
        "resume": args.resume,
        "backend": args.backend,
        "compile_mode": args.compile_mode,
        "distances": args.distances,
        "include_p_feature": args.include_p_feature,
    }

    cfg_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}
    for key, value in field_map.items():
        if value is not None:
            cfg_dict[key] = value

    for key in ("circuit_dir", "output_dir"):
        if key in cfg_dict and cfg_dict[key] is not None:
            cfg_dict[key] = Path(cfg_dict[key])
    if cfg_dict.get("resume") is not None:
        cfg_dict["resume"] = Path(cfg_dict["resume"])

    return TrainConfig(**cfg_dict), curriculum_cfg


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for GNN training."""
    cfg, curriculum_cfg = parse_args(argv)

    if curriculum_cfg is not None:
        CurriculumTrainer(cfg, curriculum_cfg).fit()
    else:
        Trainer(cfg).fit()


if __name__ == "__main__":
    main()
