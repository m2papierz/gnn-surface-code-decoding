"""Train a mixed-distance GNN decoder with curriculum learning.

Examples
--------
    # Train from config (recommended)
    uv run scripts/train_curriculum.py -c configs/curriculum.yaml

    # Warm-start from a per-distance checkpoint
    uv run scripts/train_curriculum.py -c configs/curriculum.yaml \
        --resume outputs/d3_full/direct/best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import yaml

from training import CurriculumConfig, CurriculumTrainer, TrainConfig


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[TrainConfig, CurriculumConfig]:
    """Parse CLI arguments into training and curriculum configs."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("configs/curriculum.yaml"),
        help="YAML config file",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--backend", type=str, default=None, choices=["pytorch", "compiled"]
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_cfg = TrainConfig.from_yaml(args.config)
    curriculum_cfg = CurriculumConfig.from_yaml(raw["curriculum"])

    overrides = {
        "output_dir": args.output_dir,
        "resume": args.resume,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "backend": args.backend,
    }
    cfg_dict = {
        f.name: getattr(train_cfg, f.name)
        for f in train_cfg.__dataclass_fields__.values()
    }
    for key, value in overrides.items():
        if value is not None:
            cfg_dict[key] = value
    for key in ("circuit_dir", "output_dir"):
        if key in cfg_dict and cfg_dict[key] is not None:
            cfg_dict[key] = Path(cfg_dict[key])
    if cfg_dict.get("resume") is not None:
        cfg_dict["resume"] = Path(cfg_dict["resume"])

    return TrainConfig(**cfg_dict), curriculum_cfg


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for curriculum training."""
    train_cfg, curriculum_cfg = parse_args(argv)
    CurriculumTrainer(train_cfg, curriculum_cfg).fit()


if __name__ == "__main__":
    main()
