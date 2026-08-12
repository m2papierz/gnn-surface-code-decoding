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

        history_path = tmp_path / "out" / "memory" / "mixed" / "direct" / "history.json"
        assert history_path.exists()
        history = json.loads(history_path.read_text())
        assert len(history) >= 2
        assert "samples_consumed" in history[0]

    def test_config_json_written(self, circuit_dir, tmp_path) -> None:
        cfg = _make_config(circuit_dir, tmp_path / "out")
        trainer = Trainer(cfg)
        trainer.fit()

        config_path = tmp_path / "out" / "memory" / "mixed" / "direct" / "config.json"
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert saved["sample_budget"] == 1000
        assert saved["node_dim"] == 6
        assert saved["edge_dim"] == 6


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

    def test_cuda_backend_is_rejected(self) -> None:
        """The fast-path kernels are forward-only; training on them cannot work.

        ``build_training_state`` passes this field straight to
        ``model.ops.set_backend``, so without this guard a config could wire
        inference-only kernels into the training loop.
        """
        with pytest.raises(ValueError, match="inference-only"):
            TrainConfig(backend="cuda")

    def test_unknown_backend_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="backend must be one of"):
            TrainConfig(backend="tensorrt")

    def test_training_backends_are_accepted(self) -> None:
        assert TrainConfig(backend="pytorch").backend == "pytorch"
        assert TrainConfig(backend="compiled").backend == "compiled"

    def test_cuda_backend_is_rejected_from_yaml(self, tmp_path) -> None:
        """YAML is the path the CLI's own `choices` guard does not cover."""
        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text('backend: "cuda"\n', encoding="utf-8")

        with pytest.raises(ValueError, match="inference-only"):
            TrainConfig.from_yaml(yaml_path)

    def test_misspelled_amp_dtype_is_rejected(self) -> None:
        """A typo used to resolve to bfloat16 while the log echoed the typo."""
        with pytest.raises(ValueError, match="amp_dtype must be one of"):
            TrainConfig(amp_dtype="bflaot16")

    def test_torch_attribute_that_is_not_a_dtype_is_rejected(self) -> None:
        """getattr(torch, "nn") returns a module, which autocast cannot use."""
        with pytest.raises(ValueError, match="amp_dtype must be one of"):
            TrainConfig(amp_dtype="nn")

    def test_real_dtype_outside_the_amp_set_is_rejected(self) -> None:
        """float64 is a genuine dtype, and meaningless for autocast."""
        with pytest.raises(ValueError, match="amp_dtype must be one of"):
            TrainConfig(amp_dtype="float64")

    def test_amp_dtype_resolves_to_the_torch_dtype(self) -> None:
        assert TrainConfig(amp_dtype="bfloat16").amp_torch_dtype is torch.bfloat16
        assert TrainConfig(amp_dtype="float16").amp_torch_dtype is torch.float16


class TestOperationConfig:
    """TrainConfig resolves the operation to a profile."""

    def test_unknown_operation_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown operation"):
            TrainConfig(operation="nonexistent_op", circuit_dir=".")

    def test_default_operation_is_memory(self) -> None:
        cfg = TrainConfig()
        assert cfg.operation == "memory"

    def test_circuit_dir_populated_from_profile(self) -> None:
        from pathlib import Path

        cfg = TrainConfig(operation="memory")
        assert cfg.circuit_dir == Path("data/circuits/memory")

    def test_circuit_dir_override_preserved(self, circuit_dir) -> None:
        cfg = TrainConfig(operation="memory", circuit_dir=circuit_dir)
        assert cfg.circuit_dir == circuit_dir


class TestRunDir:
    """Run directory computed from (profile, strategy)."""

    def test_resolve_run_dir_layout(self) -> None:
        from pathlib import Path

        from training.primitives import resolve_run_dir

        d = resolve_run_dir(Path("outputs/runs"), "memory", [3], "direct")
        assert d == Path("outputs/runs/memory/d3/direct")

    def test_resolve_run_dir_curriculum_layout(self) -> None:
        from pathlib import Path

        from training.primitives import resolve_run_dir

        d = resolve_run_dir(Path("outputs/runs"), "memory", None, "curriculum")
        assert d == Path("outputs/runs/memory/mixed/curriculum")

    def test_resolve_run_dir_different_strategies_disjoint(self) -> None:
        from pathlib import Path

        from training.primitives import resolve_run_dir

        d1 = resolve_run_dir(Path("out"), "memory", [3], "direct")
        d2 = resolve_run_dir(Path("out"), "memory", None, "curriculum")
        assert d1 != d2

    def test_resolve_run_dir_unknown_operation_raises(self) -> None:
        from pathlib import Path

        from training.primitives import resolve_run_dir

        with pytest.raises(ValueError, match="Unknown operation"):
            resolve_run_dir(Path("out"), "nonexistent_op", [3], "direct")


class TestFromYaml:
    """TrainConfig.from_yaml parses the config file."""

    def test_loads_sample_budget_config(self, tmp_path) -> None:
        yaml_content = """\
circuit_dir: "./data/circuits/memory"
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

        assert cfg.operation == "memory"
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

    def test_from_yaml_reads_operation(self, tmp_path) -> None:
        yaml_content = """\
operation: "memory"
output_dir: "./outputs"
"""
        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(yaml_content)

        cfg = TrainConfig.from_yaml(yaml_path)
        assert cfg.operation == "memory"

    def test_from_yaml_unknown_operation_rejected(self, tmp_path) -> None:
        yaml_content = """\
operation: "bogus"
output_dir: "./outputs"
"""
        yaml_path = tmp_path / "train.yaml"
        yaml_path.write_text(yaml_content)

        with pytest.raises(ValueError, match="Unknown operation"):
            TrainConfig.from_yaml(yaml_path)
