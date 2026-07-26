"""Tests for sample-budget training loop."""

from __future__ import annotations

import json

import pytest
import stim
import torch

from training import TrainConfig, Trainer


@pytest.fixture()
def circuit_dir(tmp_path):
    """Create a minimal circuit directory with one d=3 setting."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    circuit.to_file(tmp_path / "d3_r3_p0_01.stim")
    return tmp_path


def _make_config(circuit_dir, output_dir, **overrides):
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


class TestSampleBudgetSmoke:
    """Smoke test: budget=1000, batch=64, CPU."""

    def test_completes_and_produces_checkpoint(self, circuit_dir, tmp_path) -> None:
        cfg = _make_config(circuit_dir, tmp_path / "out")
        trainer = Trainer(cfg)
        best_path = trainer.fit()

        assert trainer.samples_consumed >= 1000
        assert best_path.exists()

        ckpt = torch.load(best_path, weights_only=False)
        assert "samples_consumed" in ckpt
        assert ckpt["samples_consumed"] > 0
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert "decision_threshold" in ckpt

    def test_history_logged(self, circuit_dir, tmp_path) -> None:
        cfg = _make_config(circuit_dir, tmp_path / "out")
        trainer = Trainer(cfg)
        trainer.fit()

        assert len(trainer.history) >= 2
        for entry in trainer.history:
            assert "samples_consumed" in entry
            assert entry["samples_consumed"] > 0
            assert "train" in entry
            assert "val" in entry
            assert "lr" in entry
            assert "loss" in entry["train"]
            assert "loss" in entry["val"]

        samples_seq = [e["samples_consumed"] for e in trainer.history]
        assert samples_seq == sorted(samples_seq)

    def test_history_file_written(self, circuit_dir, tmp_path) -> None:
        cfg = _make_config(circuit_dir, tmp_path / "out")
        trainer = Trainer(cfg)
        trainer.fit()

        history_path = tmp_path / "out" / "direct" / "history.json"
        assert history_path.exists()
        history = json.loads(history_path.read_text())
        assert len(history) >= 2
        assert "samples_consumed" in history[0]

    def test_config_json_written(self, circuit_dir, tmp_path) -> None:
        cfg = _make_config(circuit_dir, tmp_path / "out")
        trainer = Trainer(cfg)
        trainer.fit()

        config_path = tmp_path / "out" / "direct" / "config.json"
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert saved["sample_budget"] == 1000
        assert saved["node_dim"] == 6
        assert saved["edge_dim"] == 5


class TestEarlyStopping:
    """Early stopping triggers after configured patience."""

    def test_patience_limits_training(self, circuit_dir, tmp_path) -> None:
        cfg = _make_config(
            circuit_dir,
            tmp_path / "out",
            sample_budget=100_000,
            val_interval_samples=200,
            patience=3,
        )
        trainer = Trainer(cfg)
        trainer.fit()

        assert trainer.samples_consumed < cfg.sample_budget


class TestScheduler:
    """LR scheduler is parameterized by budget fraction."""

    def test_lr_changes_during_training(self, circuit_dir, tmp_path) -> None:
        cfg = _make_config(
            circuit_dir,
            tmp_path / "out",
            sample_budget=2000,
            val_interval_samples=500,
        )
        trainer = Trainer(cfg)
        trainer.fit()

        lrs = [e["lr"] for e in trainer.history]
        assert len(set(lrs)) > 1, "LR should change during training"


class TestConfigValidation:
    """TrainConfig validates its fields."""

    def test_sample_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="sample_budget"):
            TrainConfig(sample_budget=0)

    def test_val_interval_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="val_interval_samples"):
            TrainConfig(val_interval_samples=0)

    def test_val_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="val_size"):
            TrainConfig(val_size=0)

    def test_warmup_fraction_bounds(self) -> None:
        with pytest.raises(ValueError, match="warmup_fraction"):
            TrainConfig(warmup_fraction=0.0)
        with pytest.raises(ValueError, match="warmup_fraction"):
            TrainConfig(warmup_fraction=1.0)


class TestFromYaml:
    """TrainConfig.from_yaml parses the config file."""

    def test_loads_sample_budget_config(self, tmp_path) -> None:
        yaml_content = """\
circuit_dir: "./data/circuits"
output_dir: "./outputs"
model:
  hidden_dim: 64
  num_layers: 3
  dropout: 0.05
optimisation:
  lr: 2.0e-4
  weight_decay: 1.0e-5
  batch_size: 256
sample_budget: 5_000_000
val_interval_samples: 50_000
val_size: 5000
warmup_fraction: 0.03
patience: 5
seed: 123
"""
        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(yaml_content)

        cfg = TrainConfig.from_yaml(yaml_path)

        assert cfg.hidden_dim == 64
        assert cfg.num_layers == 3
        assert cfg.dropout == 0.05
        assert cfg.lr == 2.0e-4
        assert cfg.batch_size == 256
        assert cfg.sample_budget == 5_000_000
        assert cfg.val_interval_samples == 50_000
        assert cfg.val_size == 5000
        assert cfg.warmup_fraction == 0.03
        assert cfg.patience == 5
        assert cfg.seed == 123
