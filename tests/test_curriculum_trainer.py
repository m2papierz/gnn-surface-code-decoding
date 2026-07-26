"""Tests for mixed-distance curriculum training."""

from __future__ import annotations

import json

import numpy as np
import pytest
import stim
import torch

from sampling.sampler import WorkerSampler, settings_from_circuit_dir
from training import CurriculumConfig, CurriculumStage, CurriculumTrainer, TrainConfig


@pytest.fixture()
def circuit_dir(tmp_path):
    """Create circuit files for d=3 and d=5."""
    for d in (3, 5):
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=d,
            after_clifford_depolarization=0.01,
        )
        circuit.to_file(tmp_path / f"d{d}_r{d}_p0_01.stim")
    return tmp_path


@pytest.fixture()
def circuit_dir_3d(tmp_path):
    """Create circuit files for d=3, d=5, and d=7."""
    for d in (3, 5, 7):
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=d,
            rounds=d,
            after_clifford_depolarization=0.01,
        )
        circuit.to_file(tmp_path / f"d{d}_r{d}_p0_01.stim")
    return tmp_path


def _make_train_config(circuit_dir, output_dir, **overrides):
    defaults = dict(
        circuit_dir=circuit_dir,
        output_dir=output_dir,
        sample_budget=1000,
        batch_size=64,
        val_interval_samples=500,
        val_size=100,
        warmup_fraction=0.1,
        patience=0,
        num_workers=0,
        seed=42,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        lr=1e-3,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_curriculum_config(**overrides):
    defaults = dict(
        stages=(
            CurriculumStage(distances=(3,), budget=500),
            CurriculumStage(distances=(3, 5), budget=500),
        ),
        mwpm_ler={3: 0.05, 5: 0.09},
        gap_eval_interval=300,
        min_distance_weight=0.1,
        val_size_per_distance=50,
    )
    defaults.update(overrides)
    return CurriculumConfig(**defaults)


# ---------------------------------------------------------------------------
# CurriculumStage validation
# ---------------------------------------------------------------------------


class TestCurriculumStage:
    def test_valid_stage(self) -> None:
        stage = CurriculumStage(distances=(3, 5), budget=1000)
        assert stage.distances == (3, 5)
        assert stage.budget == 1000

    def test_empty_distances_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one distance"):
            CurriculumStage(distances=(), budget=1000)

    def test_zero_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="budget must be >= 1"):
            CurriculumStage(distances=(3,), budget=0)


# ---------------------------------------------------------------------------
# CurriculumConfig validation
# ---------------------------------------------------------------------------


class TestCurriculumConfig:
    def test_valid_config(self) -> None:
        cfg = _make_curriculum_config()
        assert len(cfg.stages) == 2
        assert cfg.total_budget == 1000

    def test_empty_stages_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CurriculumConfig(stages=(), mwpm_ler={3: 0.05})

    def test_replay_violation_rejected(self) -> None:
        with pytest.raises(ValueError, match="replay"):
            CurriculumConfig(
                stages=(
                    CurriculumStage(distances=(3, 5), budget=500),
                    CurriculumStage(distances=(5,), budget=500),
                ),
                mwpm_ler={3: 0.05, 5: 0.09},
            )

    def test_replay_extending_is_ok(self) -> None:
        cfg = CurriculumConfig(
            stages=(
                CurriculumStage(distances=(3,), budget=500),
                CurriculumStage(distances=(3, 5), budget=500),
                CurriculumStage(distances=(3, 5, 7), budget=500),
            ),
            mwpm_ler={3: 0.05, 5: 0.09, 7: 0.12},
        )
        assert cfg.total_budget == 1500

    def test_min_weight_bounds(self) -> None:
        with pytest.raises(ValueError, match="min_distance_weight"):
            _make_curriculum_config(min_distance_weight=0.0)
        with pytest.raises(ValueError, match="min_distance_weight"):
            _make_curriculum_config(min_distance_weight=1.0)

    def test_from_yaml(self) -> None:
        raw = {
            "stages": [
                {"distances": [3], "budget": 500},
                {"distances": [3, 5], "budget": 1000},
            ],
            "mwpm_ler": {3: 0.05, 5: 0.09},
            "gap_eval_interval": 200,
        }
        cfg = CurriculumConfig.from_yaml(raw)
        assert len(cfg.stages) == 2
        assert cfg.stages[0].distances == (3,)
        assert cfg.stages[1].budget == 1000
        assert cfg.mwpm_ler == {3: 0.05, 5: 0.09}
        assert cfg.gap_eval_interval == 200


# ---------------------------------------------------------------------------
# Weighted sampling in WorkerSampler
# ---------------------------------------------------------------------------


class TestWeightedSampling:
    def test_uniform_when_no_weights(self, circuit_dir) -> None:
        settings = settings_from_circuit_dir(circuit_dir)
        sampler = WorkerSampler(settings, worker_seed=42)
        counts = {s.distance: 0 for s in settings}
        for _ in range(1000):
            _, _, meta, _ = sampler.sample()
            counts[meta.distance] += 1
        assert counts[3] > 0
        assert counts[5] > 0

    def test_weights_skew_distribution(self, circuit_dir) -> None:
        settings = settings_from_circuit_dir(circuit_dir)
        weights = np.array([0.9 if s.distance == 3 else 0.1 for s in settings])
        sampler = WorkerSampler(settings, worker_seed=42, weights=weights)
        counts = {s.distance: 0 for s in settings}
        for _ in range(1000):
            _, _, meta, _ = sampler.sample()
            counts[meta.distance] += 1
        assert counts[3] > counts[5] * 3

    def test_weights_length_mismatch_rejected(self, circuit_dir) -> None:
        settings = settings_from_circuit_dir(circuit_dir)
        with pytest.raises(ValueError, match="weights length"):
            WorkerSampler(settings, worker_seed=42, weights=np.array([0.5]))

    def test_zero_weights_rejected(self, circuit_dir) -> None:
        settings = settings_from_circuit_dir(circuit_dir)
        with pytest.raises(ValueError, match="positive value"):
            WorkerSampler(settings, worker_seed=42, weights=np.zeros(len(settings)))


# ---------------------------------------------------------------------------
# CurriculumTrainer smoke tests
# ---------------------------------------------------------------------------


class TestCurriculumTrainerSmoke:
    def test_completes_and_produces_checkpoint(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config()
        trainer = CurriculumTrainer(cfg, curriculum)
        best_path = trainer.fit()

        assert trainer.samples_consumed >= 1000
        assert best_path.exists()

        ckpt = torch.load(best_path, weights_only=False)
        assert "samples_consumed" in ckpt
        assert ckpt["samples_consumed"] > 0
        assert "model_state_dict" in ckpt
        assert "curriculum_stage" in ckpt
        assert "distance_weights" in ckpt

    def test_history_logged_with_stage_info(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config()
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        assert len(trainer.history) >= 2
        for entry in trainer.history:
            assert "samples_consumed" in entry
            assert "stage" in entry
            assert "distances" in entry
            assert "distance_weights" in entry
            assert "per_distance_ler" in entry

    def test_stage_history_logged(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config()
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        assert len(trainer.stage_history) == 2
        for entry in trainer.stage_history:
            assert "stage" in entry
            assert "distances" in entry
            assert "budget" in entry
            assert "samples_consumed" in entry
            assert "final_weights" in entry

    def test_history_files_written(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config()
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        history_path = tmp_path / "out" / "curriculum" / "history.json"
        assert history_path.exists()
        history = json.loads(history_path.read_text())
        assert len(history) >= 2

        stage_path = tmp_path / "out" / "curriculum" / "stage_history.json"
        assert stage_path.exists()
        stages = json.loads(stage_path.read_text())
        assert len(stages) == 2

    def test_config_json_includes_curriculum(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config()
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        config_path = tmp_path / "out" / "curriculum" / "config.json"
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert "curriculum" in saved
        assert len(saved["curriculum"]["stages"]) == 2
        assert saved["curriculum"]["total_budget"] == 1000


# ---------------------------------------------------------------------------
# Stage transitions
# ---------------------------------------------------------------------------


class TestStageTransitions:
    def test_distances_extend_never_shrink(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config()
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        stage_distances = [set(entry["distances"]) for entry in trainer.stage_history]
        for i in range(1, len(stage_distances)):
            assert stage_distances[i - 1].issubset(stage_distances[i])

    def test_three_stage_curriculum(self, circuit_dir_3d, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir_3d, tmp_path / "out")
        curriculum = CurriculumConfig(
            stages=(
                CurriculumStage(distances=(3,), budget=300),
                CurriculumStage(distances=(3, 5), budget=300),
                CurriculumStage(distances=(3, 5, 7), budget=400),
            ),
            mwpm_ler={3: 0.05, 5: 0.09, 7: 0.12},
            gap_eval_interval=500,
            val_size_per_distance=30,
        )
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        assert len(trainer.stage_history) == 3
        assert trainer.samples_consumed >= 1000


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


class TestWeightComputation:
    def test_weights_sum_to_one(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config()
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        for entry in trainer.history:
            weights = entry["distance_weights"]
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"

    def test_uniform_when_all_at_parity(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(circuit_dir, tmp_path / "out")
        curriculum = _make_curriculum_config(
            mwpm_ler={3: 1.0, 5: 1.0},
        )
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        last_entry = trainer.history[-1]
        weights = last_entry["distance_weights"]
        if len(weights) > 1:
            values = list(weights.values())
            assert abs(values[0] - values[1]) < 0.3


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class TestCurriculumEarlyStopping:
    def test_all_stages_run_despite_nonfinal_convergence(
        self, circuit_dir, tmp_path
    ) -> None:
        cfg = _make_train_config(
            circuit_dir,
            tmp_path / "out",
            patience=2,
        )
        curriculum = _make_curriculum_config(
            stages=(
                CurriculumStage(distances=(3,), budget=50_000),
                CurriculumStage(distances=(3, 5), budget=50_000),
            ),
        )
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        assert len(trainer.stage_history) == 2
        assert trainer.samples_consumed < curriculum.total_budget

    def test_final_stage_early_stop_ends_training(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(
            circuit_dir,
            tmp_path / "out",
            patience=2,
        )
        curriculum = _make_curriculum_config(
            stages=(
                CurriculumStage(distances=(3,), budget=50_000),
                CurriculumStage(distances=(3, 5), budget=50_000),
            ),
        )
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        assert trainer._stopped_early
        assert trainer.samples_consumed < curriculum.total_budget

    def test_best_metric_resets_at_stage_transition(
        self, circuit_dir, tmp_path
    ) -> None:
        cfg = _make_train_config(
            circuit_dir,
            tmp_path / "out",
            patience=0,
        )
        curriculum = _make_curriculum_config(
            stages=(
                CurriculumStage(distances=(3,), budget=500),
                CurriculumStage(distances=(3, 5), budget=500),
            ),
        )
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        assert len(trainer.stage_history) == 2
        stage_1_entries = [e for e in trainer.history if e["stage"] == 1]
        assert any(e["best"] for e in stage_1_entries)

    def test_three_stage_all_run_with_patience(self, circuit_dir_3d, tmp_path) -> None:
        cfg = _make_train_config(
            circuit_dir_3d,
            tmp_path / "out",
            patience=2,
        )
        curriculum = CurriculumConfig(
            stages=(
                CurriculumStage(distances=(3,), budget=50_000),
                CurriculumStage(distances=(3, 5), budget=50_000),
                CurriculumStage(distances=(3, 5, 7), budget=50_000),
            ),
            mwpm_ler={3: 0.05, 5: 0.09, 7: 0.12},
            gap_eval_interval=50_000,
            val_size_per_distance=30,
        )
        trainer = CurriculumTrainer(cfg, curriculum)
        trainer.fit()

        assert len(trainer.stage_history) == 3
        assert trainer.samples_consumed < curriculum.total_budget

    def test_checkpoint_saved_in_later_stage(self, circuit_dir, tmp_path) -> None:
        cfg = _make_train_config(
            circuit_dir,
            tmp_path / "out",
            patience=0,
        )
        curriculum = _make_curriculum_config(
            stages=(
                CurriculumStage(distances=(3,), budget=500),
                CurriculumStage(distances=(3, 5), budget=500),
            ),
        )
        trainer = CurriculumTrainer(cfg, curriculum)
        best_path = trainer.fit()

        ckpt = torch.load(best_path, weights_only=False)
        assert ckpt["curriculum_stage"] == 1
