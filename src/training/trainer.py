"""Sample-budget trainer for GNN-based QEC decoders.

Training halts after consuming ``sample_budget`` training samples;
cumulative ``samples_consumed`` is logged at every validation checkpoint
and persisted in saved checkpoints.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from sampling.representation import DataContract
from sampling.sampler import settings_from_circuit_dir
from sampling.seeding import stable_seed
from training.config import TrainConfig, seed_everything
from training.primitives import (
    TrainingState,
    build_training_state,
    compute_val_metrics,
    format_metrics,
    load_checkpoint,
    make_frozen_val_samples,
    make_streaming_loader,
    resolve_run_dir,
    sweep_threshold,
    train_step,
    write_checkpoint,
)


logger = logging.getLogger(__name__)


class Trainer:
    """GNN trainer with sample-budget semantics.

    Training halts after consuming ``cfg.sample_budget`` training
    samples.  The learning-rate scheduler is parameterized by budget
    fraction (total optimizer steps = budget / batch_size), not epochs.
    Validation runs on a frozen set pre-sampled once at training start.

    Parameters
    ----------
    cfg : TrainConfig
        Full training configuration.

    Example
    -------
    >>> trainer = Trainer(cfg)
    >>> best_path = trainer.fit()
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader: DataLoader
        self.val_loader: DataLoader

        self.samples_consumed: int = 0
        self.best_metric: float = float("inf")
        self._decision_threshold: float = 0.0
        self.history: list[dict[str, Any]] = []

        self._state: TrainingState
        self._contract: DataContract
        self._run_dir: Path
        self._best_path: Path

    def _setup_data(self) -> None:
        """Build streaming training loader and frozen validation set."""
        settings = settings_from_circuit_dir(
            self.cfg.circuit_dir, distances=self.cfg.distances
        )

        self.train_loader = make_streaming_loader(settings, self.cfg, self.device)

        train_ds = self.train_loader.dataset
        self._contract = train_ds.contract  # type: ignore[union-attr]

        val_seed = stable_seed("val", f"seed={self.cfg.seed}", base=self.cfg.seed)
        val_samples = make_frozen_val_samples(
            settings, self.cfg, self.cfg.val_size, master_seed=val_seed
        )
        self.val_loader = DataLoader(
            val_samples,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            "Training: streaming from %d settings, budget=%d samples",
            len(settings),
            self.cfg.sample_budget,
        )
        logger.info("Validation: %d frozen samples", len(val_samples))

    def _maybe_resume(self) -> None:
        """Restore training state from checkpoint if configured."""
        if self.cfg.resume is None:
            return
        ckpt = load_checkpoint(
            self.cfg.resume, self._state, self.cfg, contract=self._contract
        )
        self.samples_consumed = ckpt.get("samples_consumed", 0)
        self.best_metric = ckpt.get("best_metric", float("inf"))
        self._decision_threshold = ckpt.get("decision_threshold", 0.0)
        logger.info(
            "Resuming from %d samples (best_metric=%.6f, threshold=%.3f)",
            self.samples_consumed,
            self.best_metric,
            self._decision_threshold,
        )

    def _save_config(self) -> None:
        """Persist run configuration to JSON."""
        config_dict = asdict(self.cfg)
        config_dict["circuit_dir"] = str(self.cfg.circuit_dir)
        config_dict["output_dir"] = str(self.cfg.output_dir)
        config_dict["resume"] = str(self.cfg.resume) if self.cfg.resume else None
        config_dict["contract"] = self._contract.to_dict()
        config_dict["node_dim"] = self._contract.node_dim
        config_dict["edge_dim"] = self._contract.edge_dim
        config_dict["distances"] = self.cfg.distances

        path = self._run_dir / "config.json"
        path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    def validate(self) -> dict[str, float]:
        """Run validation on the frozen val set.

        Returns
        -------
        dict
            Validation metrics: ``loss`` and ``ler``.
        """
        return compute_val_metrics(self._state, self.val_loader)

    def calibrate_threshold(self) -> float:
        """Sweep decision thresholds on the validation set.

        Returns
        -------
        float
            Optimal logit threshold (0.0 = sigmoid 0.5 default).
        """
        return sweep_threshold(self._state, self.val_loader)

    def save_checkpoint(
        self, path: Path, samples_consumed: int, best_metric: float
    ) -> None:
        """Save a training checkpoint.

        Parameters
        ----------
        path : Path
            Checkpoint file path.
        samples_consumed : int
            Cumulative training samples consumed at this point.
        best_metric : float
            Best validation metric achieved so far.
        """
        write_checkpoint(
            path,
            self._state,
            samples_consumed=samples_consumed,
            best_metric=best_metric,
            decision_threshold=self._decision_threshold,
            contract=self._contract,
            cfg=self.cfg,
        )
        logger.debug("Saved checkpoint to %s (samples=%d)", path, samples_consumed)

    def fit(self) -> Path:
        """Execute sample-budget training.

        Consumes up to ``cfg.sample_budget`` training samples from the
        streaming dataset.  Validation runs every
        ``cfg.val_interval_samples`` samples on a frozen validation set.
        Early stopping triggers after ``cfg.patience`` consecutive
        validation checks without improvement (disabled when patience=0).

        Returns
        -------
        Path
            Path to the best model checkpoint.
        """
        seed_everything(self.cfg.seed)
        logger.info("Device: %s", self.device)

        self._run_dir = resolve_run_dir(
            self.cfg.output_dir, self.cfg.operation, self.cfg.distances, "direct"
        )
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._best_path = self._run_dir / "best.pt"

        self._setup_data()
        self._state = build_training_state(
            self.cfg,
            contract=self._contract,
            total_budget=self.cfg.sample_budget,
            device=self.device,
        )
        self._maybe_resume()
        self._save_config()

        metric_key = "ler"
        checks_without_improvement = 0

        running_loss = 0.0
        running_batches = 0
        running_graphs = 0
        running_errors = 0

        next_val_at = self.samples_consumed + self.cfg.val_interval_samples
        t0 = time.perf_counter()

        logger.info(
            "Starting training: budget=%d samples (val every %d, patience=%d)",
            self.cfg.sample_budget,
            self.cfg.val_interval_samples,
            self.cfg.patience,
        )

        train_iter = iter(self.train_loader)

        while self.samples_consumed < self.cfg.sample_budget:
            batch = next(train_iter)
            batch = batch.to(self.device)

            loss_val, logits = train_step(self._state, batch)

            self.samples_consumed += batch.num_graphs
            running_loss += loss_val
            running_batches += 1

            with torch.no_grad():
                pred = (logits > 0.0).float()
                target_2d = batch.y.view_as(pred)
                running_graphs += pred.shape[0]
                running_errors += int((pred != target_2d).any(dim=1).sum().item())

            do_val = (
                self.samples_consumed >= next_val_at
                or self.samples_consumed >= self.cfg.sample_budget
            )
            if not do_val:
                continue

            elapsed = time.perf_counter() - t0
            train_metrics: dict[str, float] = {
                "loss": running_loss / max(running_batches, 1),
            }
            if running_graphs > 0:
                train_metrics["ler"] = running_errors / running_graphs

            val_metrics = self.validate()
            current = val_metrics[metric_key]
            improved = current < self.best_metric

            if improved:
                self.best_metric = current
                checks_without_improvement = 0
                self.save_checkpoint(
                    self._best_path,
                    self.samples_consumed,
                    self.best_metric,
                )
            else:
                checks_without_improvement += 1

            lr = self._state.optimizer.param_groups[0]["lr"]
            logger.info(
                "Samples %d/%d [%.1fs, lr=%.1e]  train: %s  val: %s%s",
                self.samples_consumed,
                self.cfg.sample_budget,
                elapsed,
                lr,
                format_metrics(train_metrics),
                format_metrics(val_metrics),
                " *" if improved else "",
            )

            self.history.append(
                {
                    "samples_consumed": self.samples_consumed,
                    "lr": lr,
                    "elapsed_s": round(elapsed, 2),
                    "train": train_metrics,
                    "val": val_metrics,
                    "best": improved,
                }
            )

            running_loss = 0.0
            running_batches = 0
            running_graphs = 0
            running_errors = 0
            t0 = time.perf_counter()
            next_val_at = self.samples_consumed + self.cfg.val_interval_samples

            if (
                self.cfg.patience > 0
                and checks_without_improvement >= self.cfg.patience
            ):
                logger.info(
                    "Early stopping at %d samples (%d checks w/o improvement)",
                    self.samples_consumed,
                    self.cfg.patience,
                )
                break

        history_path = self._run_dir / "history.json"
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        logger.info(
            "Training complete. Best %s=%.6f, samples_consumed=%d",
            metric_key,
            self.best_metric,
            self.samples_consumed,
        )
        logger.info("Best checkpoint: %s", self._best_path)

        if self._best_path.exists():
            ckpt = torch.load(self._best_path, weights_only=False)
            self._state.model.load_state_dict(ckpt["model_state_dict"])

            self._decision_threshold = self.calibrate_threshold()

            self.save_checkpoint(
                self._best_path,
                ckpt["samples_consumed"],
                self.best_metric,
            )
            logger.info(
                "Checkpoint updated with decision_threshold=%.3f",
                self._decision_threshold,
            )

        return self._best_path
