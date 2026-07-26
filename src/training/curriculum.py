"""Mixed-distance curriculum trainer with adaptive replay.

Trains a single GNN through a sequence of curriculum stages
(e.g. d=3 -> d=3+5 -> d=3+5+7).  Once a distance enters the
mix, it never leaves.  Sampling ratio across distances is
adaptive, proportional to each distance's current LER gap to MWPM,
re-measured every ``gap_eval_interval`` training samples.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader

from model.dataset import StreamingSurfaceCodeDataset
from sampling.sampler import CircuitSetting, settings_from_circuit_dir
from sampling.seeding import stable_seed
from training.config import (
    CurriculumConfig,
    CurriculumStage,
    TrainConfig,
    seed_everything,
)
from training.primitives import (
    TrainingState,
    build_training_state,
    compute_val_metrics,
    format_metrics,
    load_checkpoint,
    make_streaming_loader,
    sweep_threshold,
    train_step,
    write_checkpoint,
)


logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """Mixed-distance curriculum trainer with adaptive replay.

    Trains a single GNN through a sequence of curriculum stages
    (e.g. d=3 -> d=3+5 -> d=3+5+7).  Once a distance enters the
    mix, it never leaves.  Sampling ratio across distances is
    adaptive, proportional to each distance's current LER gap to MWPM,
    re-measured every ``gap_eval_interval`` training samples.

    Parameters
    ----------
    cfg : TrainConfig
        Base training hyperparameters (model, optimizer, etc.).
        ``sample_budget`` is ignored — the total budget is the sum of
        stage budgets from ``curriculum``.
    curriculum : CurriculumConfig
        Curriculum stages and gap-measurement parameters.
    """

    def __init__(self, cfg: TrainConfig, curriculum: CurriculumConfig) -> None:
        self.cfg = cfg
        self.curriculum = curriculum
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.samples_consumed: int = 0
        self.best_metric: float = float("inf")
        self._decision_threshold: float = 0.0
        self.history: list[dict[str, Any]] = []
        self.stage_history: list[dict[str, Any]] = []

        self._state: TrainingState
        self._node_dim: int
        self._edge_dim: int
        self._run_dir: Path
        self._best_path: Path
        self._stopped_early: bool = False

        self._all_settings: list[CircuitSetting] = []
        self._per_distance_val: dict[int, list[Any]] = {}
        self._distance_weights: dict[int, float] = {}

    @property
    def total_budget(self) -> int:
        """Total sample budget across all curriculum stages."""
        return self.curriculum.total_budget

    # -- Data setup (curriculum-specific) ------------------------------------

    def _setup_per_distance_val_sets(self) -> None:
        """Pre-sample frozen per-distance validation sets for gap measurement."""
        all_distances: set[int] = set()
        for stage in self.curriculum.stages:
            all_distances.update(stage.distances)

        for d in sorted(all_distances):
            d_settings = [s for s in self._all_settings if s.distance == d]
            if not d_settings:
                raise ValueError(
                    f"No circuit settings found for distance d={d} "
                    f"in {self.cfg.circuit_dir}"
                )

            val_seed = stable_seed("curriculum_val", f"d={d}", base=self.cfg.seed)
            val_ds = StreamingSurfaceCodeDataset(
                settings=d_settings,
                master_seed=val_seed,
                include_p_feature=self.cfg.include_p_feature,
            )
            self._per_distance_val[d] = list(
                itertools.islice(val_ds, self.curriculum.val_size_per_distance)
            )

        logger.info(
            "Per-distance val sets: %s",
            {d: len(v) for d, v in self._per_distance_val.items()},
        )

    def _val_batches_for(self, distances: tuple[int, ...]) -> Iterable[Any]:
        """Yield validation batches for the given distances."""
        for d in distances:
            val_samples = self._per_distance_val.get(d, [])
            if val_samples:
                yield from DataLoader(
                    val_samples,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

    def _all_val_batches(self) -> Iterable[Any]:
        """Yield validation batches across all distances."""
        for val_samples in self._per_distance_val.values():
            if val_samples:
                yield from DataLoader(
                    val_samples,
                    batch_size=self.cfg.batch_size,
                    shuffle=False,
                    num_workers=0,
                )

    # -- Gap measurement and weight computation ------------------------------

    @torch.no_grad()
    def _measure_per_distance_ler(self, distances: tuple[int, ...]) -> dict[int, float]:
        """Measure GNN LER per distance on frozen per-distance val sets."""
        self._state.model.eval()
        use_amp = self._state.scaler.is_enabled()
        result: dict[int, float] = {}

        for d in distances:
            val_samples = self._per_distance_val.get(d, [])
            if not val_samples:
                continue

            loader = DataLoader(
                val_samples,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=0,
            )

            total_graphs = 0
            total_errors = 0

            for batch in loader:
                batch = batch.to(self._state.device)
                with torch.amp.autocast(
                    device_type=self._state.device.type,
                    enabled=use_amp,
                    dtype=self._state.amp_dtype,
                ):
                    logits = self._state.model(batch)

                pred = (logits > self._decision_threshold).float()
                target_2d = batch.y.view_as(pred)
                total_graphs += pred.shape[0]
                total_errors += int((pred != target_2d).any(dim=1).sum().item())

            result[d] = total_errors / max(total_graphs, 1)

        return result

    def _compute_distance_weights(self, distances: tuple[int, ...]) -> dict[int, float]:
        """Compute per-distance sampling weights from LER gaps to MWPM."""
        per_d_ler = self._measure_per_distance_ler(distances)

        gaps: dict[int, float] = {}
        for d in distances:
            gnn_ler = per_d_ler.get(d, 0.0)
            mwpm_ler = self.curriculum.mwpm_ler.get(d, 0.0)
            gaps[d] = max(0.0, gnn_ler - mwpm_ler)

        total_gap = sum(gaps.values())
        min_w = self.curriculum.min_distance_weight

        if total_gap <= 0:
            return {d: 1.0 / len(distances) for d in distances}

        raw = {d: max(min_w, g / total_gap) for d, g in gaps.items()}
        total_raw = sum(raw.values())
        return {d: w / total_raw for d, w in raw.items()}

    # -- DataLoader factory --------------------------------------------------

    def _make_train_loader(
        self,
        distances: tuple[int, ...],
    ) -> DataLoader:
        """Create a training DataLoader for the given distances and weights."""
        stage_settings = [s for s in self._all_settings if s.distance in distances]
        return make_streaming_loader(
            stage_settings,
            self.cfg,
            self.device,
            distance_weights=self._distance_weights or None,
        )

    # -- Config persistence --------------------------------------------------

    def _save_config(self) -> None:
        config_dict = asdict(self.cfg)
        config_dict["circuit_dir"] = str(self.cfg.circuit_dir)
        config_dict["output_dir"] = str(self.cfg.output_dir)
        config_dict["resume"] = str(self.cfg.resume) if self.cfg.resume else None
        config_dict["node_dim"] = self._node_dim
        config_dict["edge_dim"] = self._edge_dim
        config_dict["curriculum"] = {
            "stages": [
                {"distances": list(s.distances), "budget": s.budget}
                for s in self.curriculum.stages
            ],
            "mwpm_ler": self.curriculum.mwpm_ler,
            "gap_eval_interval": self.curriculum.gap_eval_interval,
            "min_distance_weight": self.curriculum.min_distance_weight,
            "val_size_per_distance": self.curriculum.val_size_per_distance,
            "total_budget": self.total_budget,
        }

        path = self._run_dir / "config.json"
        path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # -- Stage execution -----------------------------------------------------

    def _run_stage(self, stage_idx: int, stage: CurriculumStage) -> None:
        """Execute a single curriculum stage."""
        logger.info(
            "=== Curriculum stage %d: distances=%s, budget=%d ===",
            stage_idx,
            stage.distances,
            stage.budget,
        )

        self._distance_weights = self._compute_distance_weights(stage.distances)
        logger.info(
            "Initial distance weights: %s",
            {d: f"{w:.3f}" for d, w in self._distance_weights.items()},
        )

        train_loader = self._make_train_loader(stage.distances)
        train_iter = iter(train_loader)

        stage_start = self.samples_consumed
        stage_end = stage_start + stage.budget
        next_gap_eval = self.samples_consumed + self.curriculum.gap_eval_interval
        next_val_at = self.samples_consumed + self.cfg.val_interval_samples

        checks_without_improvement = 0
        running_loss = 0.0
        running_batches = 0
        running_graphs = 0
        running_errors = 0
        t0 = time.perf_counter()

        while self.samples_consumed < stage_end:
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

            if self.samples_consumed >= next_gap_eval:
                self._distance_weights = self._compute_distance_weights(stage.distances)
                logger.info(
                    "Gap eval at %d samples: weights=%s",
                    self.samples_consumed,
                    {d: f"{w:.3f}" for d, w in self._distance_weights.items()},
                )
                train_loader = self._make_train_loader(stage.distances)
                train_iter = iter(train_loader)
                next_gap_eval = (
                    self.samples_consumed + self.curriculum.gap_eval_interval
                )

            do_val = (
                self.samples_consumed >= next_val_at
                or self.samples_consumed >= stage_end
            )
            if not do_val:
                continue

            elapsed = time.perf_counter() - t0
            train_metrics: dict[str, float] = {
                "loss": running_loss / max(running_batches, 1),
            }
            if running_graphs > 0:
                train_metrics["ler"] = running_errors / running_graphs

            val_metrics = compute_val_metrics(
                self._state, self._val_batches_for(stage.distances)
            )
            current = val_metrics.get("ler", float("inf"))
            improved = current < self.best_metric

            if improved:
                self.best_metric = current
                checks_without_improvement = 0
                write_checkpoint(
                    self._best_path,
                    self._state,
                    samples_consumed=self.samples_consumed,
                    best_metric=self.best_metric,
                    decision_threshold=self._decision_threshold,
                    node_dim=self._node_dim,
                    edge_dim=self._edge_dim,
                    cfg=self.cfg,
                    extra={
                        "curriculum_stage": stage_idx,
                        "distance_weights": self._distance_weights,
                    },
                )
            else:
                checks_without_improvement += 1

            per_d_ler = self._measure_per_distance_ler(stage.distances)
            lr = self._state.optimizer.param_groups[0]["lr"]
            logger.info(
                "Samples %d/%d [stage=%d, %.1fs, lr=%.1e]  "
                "train: %s  val: %s  per_d: %s%s",
                self.samples_consumed,
                self.total_budget,
                stage_idx,
                elapsed,
                lr,
                format_metrics(train_metrics),
                format_metrics(val_metrics),
                {d: f"{v:.4f}" for d, v in per_d_ler.items()},
                " *" if improved else "",
            )

            self.history.append(
                {
                    "samples_consumed": self.samples_consumed,
                    "stage": stage_idx,
                    "distances": list(stage.distances),
                    "lr": lr,
                    "elapsed_s": round(elapsed, 2),
                    "train": train_metrics,
                    "val": val_metrics,
                    "per_distance_ler": per_d_ler,
                    "distance_weights": dict(self._distance_weights),
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
                    "Early stopping at %d samples "
                    "(stage %d, %d checks w/o improvement)",
                    self.samples_consumed,
                    stage_idx,
                    self.cfg.patience,
                )
                self._stopped_early = True
                break

        stage_samples = self.samples_consumed - stage_start
        self.stage_history.append(
            {
                "stage": stage_idx,
                "distances": list(stage.distances),
                "budget": stage.budget,
                "samples_consumed": stage_samples,
                "final_weights": dict(self._distance_weights),
            }
        )
        logger.info(
            "Stage %d complete: consumed %d/%d samples",
            stage_idx,
            stage_samples,
            stage.budget,
        )

    # -- Main entry point ----------------------------------------------------

    def fit(self) -> Path:
        """Execute curriculum training.

        Returns
        -------
        Path
            Path to the best model checkpoint.
        """
        seed_everything(self.cfg.seed)
        logger.info("Device: %s", self.device)

        self._run_dir = self.cfg.output_dir / "curriculum"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._best_path = self._run_dir / "best.pt"

        self._all_settings = settings_from_circuit_dir(self.cfg.circuit_dir)
        probe_ds = StreamingSurfaceCodeDataset(
            settings=self._all_settings[:1],
            master_seed=0,
        )
        self._node_dim = probe_ds.node_dim
        self._edge_dim = probe_ds.edge_dim

        self._state = build_training_state(
            self.cfg,
            node_dim=self._node_dim,
            edge_dim=self._edge_dim,
            total_budget=self.total_budget,
            device=self.device,
        )

        if self.cfg.resume is not None:
            load_checkpoint(
                self.cfg.resume,
                self._state,
                self.cfg,
                restore_optimizer=False,
            )
            logger.info("Warm-started model from %s", self.cfg.resume)

        self._setup_per_distance_val_sets()
        self._save_config()

        logger.info(
            "Starting curriculum training: %d stages, total_budget=%d",
            len(self.curriculum.stages),
            self.total_budget,
        )

        for stage_idx, stage in enumerate(self.curriculum.stages):
            self._run_stage(stage_idx, stage)
            if self._stopped_early:
                break

        history_path = self._run_dir / "history.json"
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        stage_path = self._run_dir / "stage_history.json"
        stage_path.write_text(
            json.dumps(self.stage_history, indent=2), encoding="utf-8"
        )

        logger.info(
            "Curriculum training complete. Best LER=%.6f, samples_consumed=%d",
            self.best_metric,
            self.samples_consumed,
        )

        if self._best_path.exists():
            ckpt = torch.load(self._best_path, weights_only=False)
            self._state.model.load_state_dict(ckpt["model_state_dict"])

            self._decision_threshold = sweep_threshold(
                self._state, self._all_val_batches()
            )

            write_checkpoint(
                self._best_path,
                self._state,
                samples_consumed=ckpt["samples_consumed"],
                best_metric=self.best_metric,
                decision_threshold=self._decision_threshold,
                node_dim=self._node_dim,
                edge_dim=self._edge_dim,
                cfg=self.cfg,
                extra={
                    "curriculum_stage": ckpt.get("curriculum_stage", -1),
                    "distance_weights": self._distance_weights,
                },
            )
            logger.info(
                "Checkpoint updated with decision_threshold=%.3f",
                self._decision_threshold,
            )

        logger.info("Best checkpoint: %s", self._best_path)
        return self._best_path
