"""Sample-budget trainer for GNN-based QEC decoders.

Training halts after consuming ``sample_budget`` training samples;
cumulative ``samples_consumed`` is logged at every validation and persisted
in every checkpoint.  A run lives in the directory its resolved spec names
(``training.run``); a killed run continues from its ``last.pt``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from sampling.sampler import ExperimentPoint, settings_from_circuit_dir
from sampling.seeding import stable_seed
from training.config import TrainConfig
from training.primitives import (
    TrainingState,
    build_training_state,
    compute_val_metrics,
    format_metrics,
    make_frozen_val_samples,
    make_streaming_loader,
    restore_state,
    snapshot_state,
    sweep_threshold,
    train_step,
    write_checkpoint,
)
from training.run import (
    BEST_CHECKPOINT,
    HISTORY_FILE,
    LAST_CHECKPOINT,
    BestState,
    RunProgress,
    RunSpec,
    create_run_dir,
    open_run_dir,
    read_last_checkpoint,
    require_new_run_dir,
    resolve_run_dir,
    segment_seed,
    write_history,
    write_last_checkpoint,
)


logger = logging.getLogger(__name__)


class Trainer:
    """GNN trainer with sample-budget semantics.

    Training halts after consuming ``cfg.sample_budget`` training samples,
    or earlier when ``cfg.patience`` validations pass without a new best.
    The learning-rate scheduler is parameterized by budget fraction (total
    optimizer steps = budget / batch_size), not epochs.  Validation runs on a
    frozen set pre-sampled once from its own seed, so every segment of a
    resumed run validates on the same shots.

    Parameters
    ----------
    cfg : TrainConfig
        Full training configuration.

    Example
    -------
    >>> best_path = Trainer(cfg).fit()     # a new run
    >>> best_path = Trainer(cfg).resume()  # continue the killed run of cfg
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.progress = RunProgress.start()

        self._spec: RunSpec
        self._settings: list[ExperimentPoint]
        self._state: TrainingState
        self._val_loader: DataLoader
        self._run_dir: Path

    @property
    def samples_consumed(self) -> int:
        """Training samples consumed by the run."""
        return self.progress.samples_consumed

    @property
    def history(self) -> list[dict[str, Any]]:
        """One entry per validation, across every segment of the run."""
        return self.progress.history

    def fit(self) -> Path:
        """Train a new run from sample 0.

        Returns
        -------
        Path
            The run's ``best.pt``.

        Raises
        ------
        training.run.RunDirectoryError
            If the directory of this run's spec already holds a run.
        """
        self._resolve()
        require_new_run_dir(resolve_run_dir(self._spec))
        self._prepare()
        self._run_dir = create_run_dir(self._spec, self._write_last)
        logger.info("Run directory: %s", self._run_dir)
        return self._train()

    def resume(self) -> Path:
        """Continue the unfinished run of this config from its ``last.pt``.

        Returns
        -------
        Path
            The run's ``best.pt``.

        Raises
        ------
        training.run.RunDirectoryError
            If no unfinished run of this config's spec exists.
        """
        self._resolve()
        self._run_dir = open_run_dir(self._spec)
        self._prepare()
        self.progress = read_last_checkpoint(
            self._run_dir / LAST_CHECKPOINT,
            self._state,
            self.cfg,
            contract=self._spec.contract,
        )
        self.progress.segment_starts.append(self.progress.samples_consumed)
        logger.info(
            "Resuming %s at %d samples (best LER=%.6f)",
            self._run_dir,
            self.progress.samples_consumed,
            self.progress.best_metric,
        )
        return self._train()

    def _resolve(self) -> None:
        """Discover the run's circuits and resolve its spec."""
        self._settings = settings_from_circuit_dir(
            self.cfg.circuit_dir,
            operation=self.cfg.operation,
            distances=self.cfg.distances,
        )
        self._spec = RunSpec.resolve(self.cfg, self._settings)

    def _prepare(self) -> None:
        """Sample the frozen validation set and build the training state."""
        val_seed = stable_seed("val", f"seed={self.cfg.seed}", base=self.cfg.seed)
        val_samples = make_frozen_val_samples(
            self._settings, self.cfg, master_seed=val_seed
        )
        self._val_loader = DataLoader(
            val_samples, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0
        )
        logger.info("Device: %s", self.device)
        logger.info("Validation: %d frozen samples", len(val_samples))

        # Initialisation draws from the first segment's seed; a resumed run
        # then overwrites the weights with its checkpoint's.
        torch.manual_seed(segment_seed(self.cfg.seed, 0))
        self._state = build_training_state(
            self.cfg,
            contract=self._spec.contract,
            total_budget=self.cfg.sample_budget,
            device=self.device,
        )

    def _write_last(self, path: Path) -> None:
        write_last_checkpoint(
            path, self._state, self.progress, contract=self._spec.contract, cfg=self.cfg
        )

    def _train(self) -> Path:
        """Train until the run is finished, then write its best checkpoint."""
        start = self.progress.samples_consumed
        # Torch draws dropout masks, the dataset every shot; nothing in training
        # draws from Python's or NumPy's global generator.
        seed = segment_seed(self.cfg.seed, start)
        torch.manual_seed(seed)
        train_loader = make_streaming_loader(
            self._settings, self.cfg, self.device, master_seed=seed
        )
        logger.info(
            "Training from %d of %d samples on %d settings (val every %d, patience=%d)",
            start,
            self.cfg.sample_budget,
            len(self._settings),
            self.cfg.val_interval_samples,
            self.cfg.patience,
        )

        train_iter = iter(train_loader)
        while not self.progress.is_finished(self.cfg):
            self._train_until_validation(train_iter)

        if self.cfg.patience > 0 and (
            self.progress.checks_without_improvement >= self.cfg.patience
        ):
            logger.info(
                "Early stopping at %d samples (%d checks w/o improvement)",
                self.progress.samples_consumed,
                self.cfg.patience,
            )
        return self._finish()

    def _train_until_validation(self, train_iter: Iterator[Any]) -> None:
        """Take gradient steps up to the next validation, then validate."""
        progress = self.progress
        validate_at = min(
            progress.samples_consumed + self.cfg.val_interval_samples,
            self.cfg.sample_budget,
        )
        loss_sum = 0.0
        num_batches = 0
        num_graphs = 0
        # Accumulated on the device and read once per validation, so the
        # running LER adds no host synchronisation to the step.
        num_errors = torch.zeros((), dtype=torch.long, device=self.device)
        t0 = time.perf_counter()

        while progress.samples_consumed < validate_at:
            batch = next(train_iter).to(self.device)
            loss_val, logits = train_step(self._state, batch)
            progress.samples_consumed += batch.num_graphs
            loss_sum += loss_val
            num_batches += 1
            num_graphs += batch.num_graphs
            wrong = (logits > 0.0) != batch.y.view_as(logits).bool()
            num_errors += wrong.any(dim=1).sum()

        train_metrics = {
            "loss": loss_sum / num_batches,
            "ler": num_errors.item() / num_graphs,
        }
        self._validate(train_metrics, elapsed=time.perf_counter() - t0)

    def _validate(self, train_metrics: dict[str, float], *, elapsed: float) -> None:
        """Score the frozen validation set, record it, and write ``last.pt``."""
        progress = self.progress
        val_metrics = compute_val_metrics(self._state, self._val_loader)
        improved = val_metrics["ler"] < progress.best_metric
        if improved:
            progress.best = BestState(
                samples_consumed=progress.samples_consumed,
                metric=val_metrics["ler"],
                state=snapshot_state(self._state),
            )
            progress.checks_without_improvement = 0
        else:
            progress.checks_without_improvement += 1

        lr = self._state.optimizer.param_groups[0]["lr"]
        progress.history.append(
            {
                "samples_consumed": progress.samples_consumed,
                "lr": lr,
                "elapsed_s": round(elapsed, 2),
                "train": train_metrics,
                "val": val_metrics,
                "best": improved,
            }
        )
        self._write_last(self._run_dir / LAST_CHECKPOINT)
        logger.info(
            "Samples %d/%d [%.1fs, lr=%.1e]  train: %s  val: %s%s",
            progress.samples_consumed,
            self.cfg.sample_budget,
            elapsed,
            lr,
            format_metrics(train_metrics),
            format_metrics(val_metrics),
            " *" if improved else "",
        )

    def _finish(self) -> Path:
        """Write the history, then ``best.pt``: the best state, calibrated."""
        best = self.progress.best
        # Every run validates at least once before it is finished: the budget
        # is positive and the last step before it always validates.
        assert best is not None
        write_history(self._run_dir / HISTORY_FILE, self.progress)

        restore_state(self._state, best.state)
        threshold = sweep_threshold(self._state, self._val_loader)
        best_path = self._run_dir / BEST_CHECKPOINT
        write_checkpoint(
            best_path,
            self._state,
            samples_consumed=best.samples_consumed,
            best_metric=best.metric,
            decision_threshold=threshold,
            contract=self._spec.contract,
            cfg=self.cfg,
        )
        logger.info(
            "Training complete: best LER=%.6f at %d samples, threshold=%.3f -> %s",
            best.metric,
            best.samples_consumed,
            threshold,
            best_path,
        )
        return best_path
