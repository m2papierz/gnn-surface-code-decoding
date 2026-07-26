"""Shared training primitives composed by all trainers.

Provides the building blocks that every training loop needs:
state management, gradient steps, validation, threshold calibration,
checkpoint I/O, and DataLoader construction.  Trainers are thin
orchestrators that compose these functions.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)
from torch_geometric.loader import DataLoader

from model.dataset import StreamingSurfaceCodeDataset
from model.decoder import QECDecoder, build_model
from sampling.sampler import CircuitSetting
from training.config import TrainConfig
from training.loss import build_criterion


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingState:
    """Mutable bundle of training infrastructure.

    Groups the objects every training loop needs so that shared
    functions accept a single ``state`` argument.

    Parameters
    ----------
    model : QECDecoder
        The GNN decoder, already on device.
    optimizer : AdamW
        AdamW optimizer.
    scheduler : LRScheduler
        Step-level LR scheduler (SequentialLR).
    criterion : nn.Module
        Loss function.
    scaler : torch.amp.GradScaler
        Mixed-precision gradient scaler.
    amp_dtype : torch.dtype
        Autocast dtype.
    device : torch.device
        Training device.
    max_grad_norm : float
        Gradient clipping threshold.
    """

    model: QECDecoder
    optimizer: AdamW
    scheduler: LRScheduler
    criterion: nn.Module
    scaler: torch.amp.GradScaler
    amp_dtype: torch.dtype
    device: torch.device
    max_grad_norm: float


def build_training_state(
    cfg: TrainConfig,
    *,
    node_dim: int,
    edge_dim: int,
    total_budget: int,
    device: torch.device,
) -> TrainingState:
    """Build model, optimizer, scheduler, criterion, and AMP state.

    Parameters
    ----------
    cfg : TrainConfig
        Training hyperparameters.
    node_dim : int
        Input node feature dimension.
    edge_dim : int
        Input edge feature dimension.
    total_budget : int
        Total sample budget (drives scheduler parameterization).
    device : torch.device
        Target device.

    Returns
    -------
    TrainingState
    """
    from model.ops import set_backend

    set_backend(cfg.backend)

    model = build_model(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.backend == "compiled" and device.type == "cuda":
        model = torch.compile(
            model, mode=cfg.compile_mode, dynamic=True, fullgraph=False
        )
        logger.info("torch.compile enabled (mode=%s, dynamic=True)", cfg.compile_mode)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", num_params)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = max(1, total_budget // cfg.batch_size)
    warmup_steps = max(1, int(cfg.warmup_fraction * total_steps))
    eta_min = cfg.lr / 50
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
    logger.info(
        "Scheduler: LinearWarmup(%d steps) => "
        "CosineAnnealing(T_max=%d, eta_min=%.1e) [total_budget=%d]",
        warmup_steps,
        total_steps - warmup_steps,
        eta_min,
        total_budget,
    )

    criterion = build_criterion(cfg)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_dtype = getattr(torch, cfg.amp_dtype, torch.bfloat16)
    if use_amp:
        torch.set_float32_matmul_precision("high")
        logger.info("Mixed precision enabled (AMP, dtype=%s, TF32=high)", cfg.amp_dtype)

    return TrainingState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        amp_dtype=amp_dtype,
        device=device,
        max_grad_norm=cfg.max_grad_norm,
    )


def train_step(state: TrainingState, batch: Any) -> tuple[float, torch.Tensor]:
    """Execute one gradient step: forward, backward, clip, update.

    Parameters
    ----------
    state : TrainingState
        Training infrastructure (model must already be on device).
    batch : PyG Batch
        Training batch, already on device.

    Returns
    -------
    loss_value : float
        Scalar loss for this batch.
    logits : Tensor
        Model output logits (detached).
    """
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(
        device_type=state.device.type,
        enabled=state.scaler.is_enabled(),
        dtype=state.amp_dtype,
    ):
        logits = state.model(batch)
        loss = state.criterion(logits.view(-1), batch.y)

    state.scaler.scale(loss).backward()
    state.scaler.unscale_(state.optimizer)
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.max_grad_norm)
    state.scaler.step(state.optimizer)
    state.scaler.update()
    state.scheduler.step()

    return loss.item(), logits.detach()


@torch.no_grad()
def compute_val_metrics(
    state: TrainingState,
    val_batches: Iterable[Any],
) -> dict[str, float]:
    """Compute validation loss and LER over batches.

    Parameters
    ----------
    state : TrainingState
        Training infrastructure.
    val_batches : iterable of PyG Batch
        Validation batches (not yet on device).

    Returns
    -------
    dict
        Metrics with keys ``loss`` and (if any graphs) ``ler``.
    """
    state.model.eval()
    use_amp = state.scaler.is_enabled()

    total_loss = 0.0
    num_batches = 0
    total_graphs = 0
    total_errors = 0

    for batch in val_batches:
        batch = batch.to(state.device)
        with torch.amp.autocast(
            device_type=state.device.type,
            enabled=use_amp,
            dtype=state.amp_dtype,
        ):
            logits = state.model(batch)
            loss = state.criterion(logits.view(-1), batch.y)

        pred = (logits > 0.0).float()
        target_2d = batch.y.view_as(pred)
        total_graphs += pred.shape[0]
        total_errors += int((pred != target_2d).any(dim=1).sum().item())
        total_loss += loss.item()
        num_batches += 1

    metrics: dict[str, float] = {"loss": total_loss / max(num_batches, 1)}
    if total_graphs > 0:
        metrics["ler"] = total_errors / total_graphs
    return metrics


@torch.no_grad()
def sweep_threshold(
    state: TrainingState,
    val_batches: Iterable[Any],
) -> float:
    """Sweep decision thresholds to minimise LER on validation data.

    Parameters
    ----------
    state : TrainingState
        Training infrastructure.
    val_batches : iterable of PyG Batch
        Validation batches (not yet on device).

    Returns
    -------
    float
        Optimal logit threshold (0.0 = sigmoid 0.5 default).
    """
    state.model.eval()
    use_amp = state.scaler.is_enabled()

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for batch in val_batches:
        batch = batch.to(state.device)
        with torch.amp.autocast(
            device_type=state.device.type,
            enabled=use_amp,
            dtype=state.amp_dtype,
        ):
            logits = state.model(batch)
        all_logits.append(logits.cpu())
        all_targets.append(batch.y.view_as(logits).cpu())

    if not all_logits:
        return 0.0

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    thresholds = torch.linspace(-4.0, 4.0, steps=81)
    best_threshold = 0.0
    best_ler = float("inf")

    for thr in thresholds:
        pred = (logits_cat > thr.item()).float()
        errors = (pred != targets_cat).any(dim=1).sum().item()
        ler = errors / logits_cat.shape[0]
        if ler < best_ler:
            best_ler = ler
            best_threshold = thr.item()

    default_pred = (logits_cat > 0.0).float()
    default_ler = (default_pred != targets_cat).any(
        dim=1
    ).sum().item() / logits_cat.shape[0]

    logger.info(
        "Threshold calibration: default(0.0) LER=%.6f, "
        "best(%.3f) LER=%.6f (delta=%.6f)",
        default_ler,
        best_threshold,
        best_ler,
        best_ler - default_ler,
    )

    return best_threshold


def write_checkpoint(
    path: Path,
    state: TrainingState,
    *,
    samples_consumed: int,
    best_metric: float,
    decision_threshold: float,
    node_dim: int,
    edge_dim: int,
    cfg: TrainConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint file path.
    state : TrainingState
        Current training state.
    samples_consumed : int
        Cumulative training samples consumed.
    best_metric : float
        Best validation metric achieved.
    decision_threshold : float
        Decision threshold for inference.
    node_dim, edge_dim : int
        Feature dimensions (persisted for resume compatibility).
    cfg : TrainConfig
        Training config (persisted in checkpoint).
    extra : dict, optional
        Additional keys to merge into the checkpoint (e.g.
        ``curriculum_stage``, ``distance_weights``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(cfg)
    config_dict["node_dim"] = node_dim
    config_dict["edge_dim"] = edge_dim

    ckpt: dict[str, Any] = {
        "samples_consumed": samples_consumed,
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "scheduler_state_dict": state.scheduler.state_dict(),
        "best_metric": best_metric,
        "decision_threshold": decision_threshold,
        "config": config_dict,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    state: TrainingState,
    cfg: TrainConfig,
    *,
    restore_optimizer: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore training state.

    Parameters
    ----------
    path : Path
        Checkpoint file path.
    state : TrainingState
        Training state to restore into.
    cfg : TrainConfig
        Current config (checked for architecture compatibility).
    restore_optimizer : bool
        If ``True`` (default), restore optimizer and scheduler state.
        Set to ``False`` for warm-start (model weights only).

    Returns
    -------
    dict
        The raw checkpoint dict (callers extract counters).
    """
    ckpt = torch.load(path, weights_only=False)
    state.model.load_state_dict(ckpt["model_state_dict"])

    if restore_optimizer:
        if "optimizer_state_dict" in ckpt:
            state.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            state.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    ckpt_cfg = ckpt.get("config", {})
    for key in ("hidden_dim", "num_layers"):
        current = getattr(cfg, key)
        saved = ckpt_cfg.get(key)
        if saved is not None and saved != current:
            raise ValueError(
                f"Config mismatch on resume: {key}={current}, "
                f"checkpoint has {key}={saved}"
            )

    return ckpt


def make_streaming_loader(
    settings: list[CircuitSetting],
    cfg: TrainConfig,
    device: torch.device,
    *,
    distance_weights: dict[int, float] | None = None,
) -> DataLoader:
    """Create a streaming training DataLoader.

    Parameters
    ----------
    settings : list of CircuitSetting
        Circuit settings to sample from.
    cfg : TrainConfig
        Training config (seed, batch_size, num_workers, etc.).
    device : torch.device
        Target device (controls pin_memory).
    distance_weights : dict, optional
        Per-distance sampling weights for curriculum training.

    Returns
    -------
    DataLoader
    """
    ds = StreamingSurfaceCodeDataset(
        settings=settings,
        master_seed=cfg.seed,
        include_p_feature=cfg.include_p_feature,
        distance_weights=distance_weights,
    )
    pin = device.type == "cuda"
    persistent = cfg.num_workers > 0
    prefetch = 2 if cfg.num_workers > 0 else None
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )


def format_metrics(metrics: dict[str, float]) -> str:
    """Format a metrics dict for log output.

    Parameters
    ----------
    metrics : dict
        Metric name to value mapping.

    Returns
    -------
    str
    """
    parts = []
    for k, v in metrics.items():
        if k == "ler":
            parts.append(f"LER={v:.4f}")
        elif k == "loss" and abs(v) < 1e-3:
            parts.append(f"loss={v:.2e}")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)
