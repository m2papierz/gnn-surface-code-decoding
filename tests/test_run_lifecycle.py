"""A run is never overwritten, is created whole, and a killed run resumes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import stim
import torch

import training.trainer as trainer_module
from artifact_file import file_sha256
from sampling.experiment import ExperimentKey, write_circuit_manifest
from sampling.sampler import settings_from_circuit_dir
from training import RunDirectoryError, RunSpec, TrainConfig, Trainer, resolve_run_dir
from training.run import segment_seed


class _Killed(BaseException):
    """Stands in for the signal that kills a training process."""


@pytest.fixture()
def circuit_dir(tmp_path: Path) -> Path:
    root = tmp_path / "circuits"
    root.mkdir()
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    circuit.to_file(root / "d3_r3_p0_01.stim")
    write_circuit_manifest(
        root / "d3_r3_p0_01.stim",
        ExperimentKey(operation="memory", distance=3, rounds=3, error_prob=0.01),
        provenance={},
    )
    return root


def _config(circuit_dir: Path, output_dir: Path, **overrides: Any) -> TrainConfig:
    fields: dict[str, Any] = dict(
        operation="memory",
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
        dropout=0.1,
        lr=1e-3,
    )
    fields.update(overrides)
    return TrainConfig(**fields)


def _run_dir(cfg: TrainConfig) -> Path:
    settings = settings_from_circuit_dir(
        cfg.circuit_dir, operation=cfg.operation, distances=cfg.distances
    )
    return resolve_run_dir(RunSpec.resolve(cfg, settings))


def _kill_after_validations(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Kill the process right after the *count*-th validation's ``last.pt``.

    The first ``last.pt`` is the one the run directory is created with.
    """
    write = trainer_module.write_last_checkpoint
    calls = 0

    def write_then_die(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        write(*args, **kwargs)
        calls += 1
        if calls == count + 1:
            raise _Killed

    monkeypatch.setattr(trainer_module, "write_last_checkpoint", write_then_die)


def _killed_run(
    monkeypatch: pytest.MonkeyPatch, cfg: TrainConfig, *, after: int
) -> Path:
    _kill_after_validations(monkeypatch, after)
    with pytest.raises(_Killed):
        Trainer(cfg).fit()
    monkeypatch.undo()
    return _run_dir(cfg)


def _raise_killed(*args: Any, **kwargs: Any) -> dict[str, float]:
    raise _Killed


def _history(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "history.json").read_text(encoding="utf-8"))


def _untimed(history: dict[str, Any]) -> dict[str, Any]:
    """The history without wall-clock times, which no seed determines."""
    validations = [
        {key: value for key, value in entry.items() if key != "elapsed_s"}
        for entry in history["validations"]
    ]
    return {**history, "validations": validations}


class TestFreshRun:
    def test_a_finished_run_holds_its_spec_state_history_and_best(
        self, circuit_dir: Path, tmp_path: Path
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        best_path = Trainer(cfg).fit()

        run_dir = _run_dir(cfg)
        assert best_path == run_dir / "best.pt"
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "best.pt",
            "config.json",
            "history.json",
            "last.pt",
        ]
        assert [p.name for p in run_dir.parent.iterdir()] == [run_dir.name]

    def test_a_directory_holding_a_run_is_refused_and_left_untouched(
        self, circuit_dir: Path, tmp_path: Path
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        best_path = Trainer(cfg).fit()
        digest = file_sha256(best_path)

        with pytest.raises(RunDirectoryError, match="never overwritten"):
            Trainer(cfg).fit()
        assert file_sha256(best_path) == digest

    def test_an_empty_directory_is_accepted(
        self, circuit_dir: Path, tmp_path: Path
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        _run_dir(cfg).mkdir(parents=True)
        assert Trainer(cfg).fit().is_file()

    def test_a_run_directory_is_created_whole_or_not_at_all(
        self, circuit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        _kill_after_validations(monkeypatch, 0)

        with pytest.raises(_Killed):
            Trainer(cfg).fit()
        assert not _run_dir(cfg).exists()
        assert list(_run_dir(cfg).parent.iterdir()) == []

    def test_best_pt_holds_the_state_of_the_best_validation(
        self, circuit_dir: Path, tmp_path: Path
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out", sample_budget=2000)
        best_path = Trainer(cfg).fit()

        validations = _history(best_path.parent)["validations"]
        best = min(validations, key=lambda entry: entry["val"]["ler"])
        ckpt = torch.load(best_path, weights_only=False)
        assert ckpt["samples_consumed"] == best["samples_consumed"]
        assert ckpt["best_metric"] == best["val"]["ler"]


class TestResume:
    def test_a_killed_run_continues_from_its_last_validation(
        self, circuit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        run_dir = _killed_run(monkeypatch, cfg, after=1)
        assert not (run_dir / "best.pt").exists()
        killed_at = torch.load(run_dir / "last.pt", weights_only=False)
        assert killed_at["samples_consumed"] == 512

        trainer = Trainer(cfg)
        best_path = trainer.resume()

        assert best_path == run_dir / "best.pt"
        assert trainer.samples_consumed >= cfg.sample_budget
        history = _history(run_dir)
        assert history["segment_starts"] == [0, 512]
        assert [e["samples_consumed"] for e in history["validations"]] == [512, 1024]

    def test_a_resumed_run_is_determined_by_where_it_was_killed(
        self, circuit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "a")
        run_dir = _killed_run(monkeypatch, cfg, after=1)
        twin_cfg = _config(circuit_dir, tmp_path / "b")
        shutil.copytree(run_dir, _run_dir(twin_cfg))
        # CUDA scatter kernels accumulate with atomics, so no GPU run is bitwise
        # reproducible, resumed or not; the property is checked on CPU, where
        # the kernels are deterministic.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        best = torch.load(Trainer(cfg).resume(), weights_only=False)
        twin = torch.load(Trainer(twin_cfg).resume(), weights_only=False)

        assert _untimed(_history(run_dir)) == _untimed(_history(_run_dir(twin_cfg)))
        for name, tensor in best["model_state_dict"].items():
            assert torch.equal(tensor, twin["model_state_dict"][name]), name

    def test_a_run_killed_before_its_first_validation_resumes_from_zero(
        self, circuit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        monkeypatch.setattr(trainer_module, "compute_val_metrics", _raise_killed)
        with pytest.raises(_Killed):
            Trainer(cfg).fit()
        monkeypatch.undo()

        Trainer(cfg).resume()
        assert _history(_run_dir(cfg))["segment_starts"] == [0, 0]

    def test_a_finished_run_is_not_resumed(
        self, circuit_dir: Path, tmp_path: Path
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        Trainer(cfg).fit()
        with pytest.raises(RunDirectoryError, match="finished run"):
            Trainer(cfg).resume()

    def test_a_spec_with_no_run_has_nothing_to_resume(
        self, circuit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        _killed_run(monkeypatch, cfg, after=1)
        changed = _config(circuit_dir, tmp_path / "out", seed=43)
        with pytest.raises(RunDirectoryError, match="No run to resume"):
            Trainer(changed).resume()

    def test_a_config_json_recording_another_spec_is_refused(
        self, circuit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = _config(circuit_dir, tmp_path / "out")
        run_dir = _killed_run(monkeypatch, cfg, after=1)
        config_path = run_dir / "config.json"
        record = json.loads(config_path.read_text(encoding="utf-8"))
        record["config"]["lr"] = 0.5
        config_path.write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(RunDirectoryError, match="records another spec"):
            Trainer(cfg).resume()


class TestSegmentSeed:
    def test_each_segment_start_draws_its_own_stream(self) -> None:
        seeds = {segment_seed(42, start) for start in (0, 512, 1024)}
        assert len(seeds) == 3
