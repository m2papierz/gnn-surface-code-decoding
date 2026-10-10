"""Training package for GNN-based QEC decoders."""

from training.config import TrainConfig
from training.loss import FocalBCEWithLogitsLoss
from training.primitives import (
    TrainingState,
    build_training_state,
    compute_val_metrics,
    load_checkpoint,
    make_streaming_loader,
    sweep_threshold,
    train_step,
    write_checkpoint,
)
from training.run import RunDirectoryError, RunSpec, resolve_run_dir
from training.trainer import Trainer


__all__ = [
    "FocalBCEWithLogitsLoss",
    "RunDirectoryError",
    "RunSpec",
    "TrainConfig",
    "Trainer",
    "TrainingState",
    "build_training_state",
    "compute_val_metrics",
    "load_checkpoint",
    "make_streaming_loader",
    "resolve_run_dir",
    "sweep_threshold",
    "train_step",
    "write_checkpoint",
]
