"""Training primitives composed by the trainer.

Provides the building blocks of the training loop: state management,
gradient steps, validation, threshold calibration, checkpoint I/O (state
snapshots, payloads, atomic writes), and DataLoader construction.  ``Trainer``
is a thin orchestrator that composes these functions, each testable without
running a fit.
"""

from __future__ import annotations

import copy
import itertools
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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

from artifact_file import atomic_write
from model.dataset import StreamingSurfaceCodeDataset
from model.decoder import QECDecoder, build_model
from sampling.profile import resolve_profile
from sampling.representation import DataContract, require_contract
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
    scaler : torch.GradScaler
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
    scaler: torch.GradScaler
    amp_dtype: torch.dtype
    device: torch.device
    max_grad_norm: float


def build_training_state(
    cfg: TrainConfig,
    *,
    contract: DataContract,
    total_budget: int,
    device: torch.device,
) -> TrainingState:
    """Build model, optimizer, scheduler, criterion, and AMP state.

    Parameters
    ----------
    cfg : TrainConfig
        Training hyperparameters.
    contract : DataContract
        Representation descriptor and label spec.
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
        node_dim=contract.node_dim,
        edge_dim=contract.edge_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_observables=contract.num_observables,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.backend == "compiled" and device.type == "cuda":
        # In place, so the module keeps its type and its state_dict keys:
        # a checkpoint does not depend on the backend that wrote it.
        model.compile(mode=cfg.compile_mode, dynamic=True, fullgraph=False)
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
    scaler = torch.GradScaler(enabled=use_amp)
    amp_dtype = cfg.amp_torch_dtype
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

    with torch.autocast(
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
        with torch.autocast(
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
        with torch.autocast(
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


def snapshot_state(state: TrainingState) -> dict[str, Any]:
    """Copy the model, optimizer, scheduler and scaler state dicts of *state*.

    The copy is independent of the live objects, so training continues while
    it holds an earlier step's state; its keys are a checkpoint's.
    """
    return copy.deepcopy(_state_dicts(state))


def restore_state(state: TrainingState, saved: Mapping[str, Any]) -> None:
    """Load the four state dicts of *saved*, a checkpoint or snapshot, into *state*."""
    state.model.load_state_dict(saved["model_state_dict"])
    state.optimizer.load_state_dict(saved["optimizer_state_dict"])
    state.scheduler.load_state_dict(saved["scheduler_state_dict"])
    state.scaler.load_state_dict(saved["scaler_state_dict"])


def checkpoint_payload(
    state: TrainingState,
    *,
    samples_consumed: int,
    best_metric: float,
    decision_threshold: float,
    contract: DataContract,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Assemble what a checkpoint file stores, referencing the live state.

    Parameters
    ----------
    state : TrainingState
        Training state whose state dicts are stored.
    samples_consumed : int
        Cumulative training samples consumed.
    best_metric : float
        Best validation metric achieved.
    decision_threshold : float
        Decision threshold for inference.
    contract : DataContract
        Representation and label contract the model was trained under.
    cfg : TrainConfig
        Training config.

    Returns
    -------
    dict
        The checkpoint, ready for :func:`write_payload`.
    """
    config = cfg.to_dict()
    config["contract"] = contract.to_dict()
    config["node_dim"] = contract.node_dim
    config["edge_dim"] = contract.edge_dim
    return {
        "samples_consumed": samples_consumed,
        "best_metric": best_metric,
        "decision_threshold": decision_threshold,
        "config": config,
        **_state_dicts(state),
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Save a checkpoint payload to *path*, atomically."""
    with atomic_write(path) as temp:
        torch.save(payload, temp)


def write_checkpoint(
    path: Path,
    state: TrainingState,
    *,
    samples_consumed: int,
    best_metric: float,
    decision_threshold: float,
    contract: DataContract,
    cfg: TrainConfig,
) -> None:
    """Save a training checkpoint, atomically.

    Parameters
    ----------
    path : Path
        Checkpoint file path.
    state, samples_consumed, best_metric, decision_threshold, contract, cfg
        As in :func:`checkpoint_payload`.
    """
    payload = checkpoint_payload(
        state,
        samples_consumed=samples_consumed,
        best_metric=best_metric,
        decision_threshold=decision_threshold,
        contract=contract,
        cfg=cfg,
    )
    write_payload(path, payload)


def load_checkpoint(
    path: Path,
    state: TrainingState,
    cfg: TrainConfig,
    *,
    contract: DataContract,
) -> dict[str, Any]:
    """Load a checkpoint and restore model, optimizer, scheduler and scaler state.

    Parameters
    ----------
    path : Path
        Checkpoint file path.
    state : TrainingState
        Training state to restore into.
    cfg : TrainConfig
        Current config (checked for architecture compatibility).
    contract : DataContract
        Contract the run trains under; the checkpoint's must equal it.

    Returns
    -------
    dict
        The raw checkpoint dict (callers extract counters).

    Raises
    ------
    KeyError
        If the checkpoint lacks a key the trainer always writes, naming it.
    ValueError
        If the checkpoint's architecture or contract differs from the run's.
    """
    ckpt = torch.load(path, weights_only=False, map_location=state.device)
    try:
        ckpt_cfg = ckpt["config"]
        saved_architecture = {
            key: ckpt_cfg[key] for key in ("hidden_dim", "num_layers")
        }
        saved_contract = DataContract.from_checkpoint_config(ckpt_cfg)
    except KeyError as exc:
        raise KeyError(f"Checkpoint {path} has no {exc.args[0]!r}") from exc

    for key, saved in saved_architecture.items():
        current = getattr(cfg, key)
        if saved != current:
            raise ValueError(
                f"Config mismatch on resume: {key}={current}, "
                f"checkpoint {path} has {key}={saved}"
            )
    require_contract(contract, saved_contract, source=path)

    try:
        restore_state(state, ckpt)
    except KeyError as exc:
        raise KeyError(f"Checkpoint {path} has no {exc.args[0]!r}") from exc
    return ckpt


def _state_dicts(state: TrainingState) -> dict[str, Any]:
    """The live state dicts of *state*, keyed as a checkpoint stores them."""
    return {
        "model_state_dict": state.model.state_dict(),
        "optimizer_state_dict": state.optimizer.state_dict(),
        "scheduler_state_dict": state.scheduler.state_dict(),
        "scaler_state_dict": state.scaler.state_dict(),
    }


def make_streaming_loader(
    settings: Sequence[CircuitSetting],
    cfg: TrainConfig,
    device: torch.device,
    *,
    master_seed: int,
) -> DataLoader:
    """Create a streaming training DataLoader.

    Parameters
    ----------
    settings : sequence of CircuitSetting
        Circuit settings to sample from.
    cfg : TrainConfig
        Training config (batch_size, num_workers, etc.).
    device : torch.device
        Target device (controls pin_memory).
    master_seed : int
        Seed of the training stream; every worker's stream derives from it.

    Returns
    -------
    DataLoader
    """
    profile = resolve_profile(cfg.operation)
    ds = StreamingSurfaceCodeDataset(
        settings=settings,
        master_seed=master_seed,
        include_p_feature=cfg.include_p_feature,
        contract=profile.data_contract,
        metadata_extractor=profile.metadata_extractor,
    )
    pin = device.type == "cuda"
    persistent = cfg.num_workers > 0
    prefetch = 2 if cfg.num_workers > 0 else None
    # PyG's annotation lists only map-style datasets; its collater batches the
    # Data an IterableDataset yields unchanged, which is what streaming needs.
    return DataLoader(
        ds,  # pyright: ignore[reportArgumentType]
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )


def make_frozen_val_samples(
    settings: Sequence[CircuitSetting],
    cfg: TrainConfig,
    *,
    master_seed: int,
) -> list[Any]:
    """Sample a frozen validation set using the operation's data contract.

    Parameters
    ----------
    settings : sequence of CircuitSetting
        Circuit settings to sample from.
    cfg : TrainConfig
        Training config (operation, include_p_feature, val_size).
    master_seed : int
        Deterministic seed for the validation stream.

    Returns
    -------
    list
        Frozen list of PyG ``Data`` objects.
    """
    profile = resolve_profile(cfg.operation)
    ds = StreamingSurfaceCodeDataset(
        settings=settings,
        master_seed=master_seed,
        include_p_feature=cfg.include_p_feature,
        contract=profile.data_contract,
        metadata_extractor=profile.metadata_extractor,
    )
    return list(itertools.islice(ds, cfg.val_size))


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
