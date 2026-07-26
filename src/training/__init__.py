"""Training package for GNN-based QEC decoders."""

from training.config import CurriculumConfig, CurriculumStage, TrainConfig
from training.curriculum import CurriculumTrainer
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
from training.trainer import Trainer


__all__ = [
    "CurriculumConfig",
    "CurriculumStage",
    "CurriculumTrainer",
    "FocalBCEWithLogitsLoss",
    "TrainConfig",
    "Trainer",
    "TrainingState",
    "build_training_state",
    "compute_val_metrics",
    "load_checkpoint",
    "make_streaming_loader",
    "sweep_threshold",
    "train_step",
    "write_checkpoint",
]
