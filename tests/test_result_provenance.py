"""Every eval report names the checkpoint it scored and the config it ran."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest
import torch

from artifact_file import file_sha256
from decoders import load_gnn_checkpoint
from sampling.profile import resolve_profile
from training import TrainConfig, build_training_state, write_checkpoint


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_gnn.py"


@pytest.fixture(scope="module")
def eval_main() -> Any:
    return runpy.run_path(str(EVAL_SCRIPT))["main"]


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    """An untrained memory checkpoint, written by the trainer's writer."""
    cfg = TrainConfig.from_fields(
        {"operation": "memory", "hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
    )
    contract = resolve_profile("memory").data_contract
    state = build_training_state(
        cfg,
        contract=contract,
        total_budget=cfg.sample_budget,
        device=torch.device("cpu"),
    )
    path = tmp_path / "run" / "best.pt"
    write_checkpoint(
        path,
        state,
        samples_consumed=128,
        best_metric=0.5,
        decision_threshold=0.25,
        contract=contract,
        cfg=cfg,
    )
    return path


def _sanity_args(checkpoint: Path, *extra: str) -> list[str]:
    return [
        "--sanity",
        "--operation",
        "memory",
        "--checkpoint",
        str(checkpoint),
        "--distances",
        "3",
        "--shots",
        "64",
        *extra,
    ]


def test_the_loaded_model_is_named_by_the_digest_of_its_file(
    checkpoint: Path,
) -> None:
    restored = load_gnn_checkpoint(
        checkpoint,
        contract=resolve_profile("memory").data_contract,
        device=torch.device("cpu"),
    )
    assert restored.sha256 == file_sha256(checkpoint)


def test_a_sanity_report_names_its_checkpoint_config_and_shots(
    eval_main: Any, checkpoint: Path, tmp_path: Path
) -> None:
    output = tmp_path / "sanity.json"
    eval_main(_sanity_args(checkpoint, "-o", str(output)))

    report = json.loads(output.read_text(encoding="utf-8"))
    metadata = report["metadata"]
    assert metadata["mode"] == "sanity"
    assert metadata["source"] == str(resolve_profile("memory").circuit_root)
    assert metadata["config"]["operation"] == "memory"
    assert metadata["config"]["shots"] == 64
    assert metadata["checkpoint"] == {
        "path": str(checkpoint),
        "sha256": file_sha256(checkpoint),
        "samples_consumed": 128,
        "decision_threshold": 0.25,
    }
    assert {point["distance"] for point in report["points"]} == {3}


def test_no_report_lands_beside_the_checkpoint(
    eval_main: Any, checkpoint: Path
) -> None:
    """A run directory is immutable; reports go where ``-o`` says, or nowhere."""
    eval_main(_sanity_args(checkpoint))
    assert [p.name for p in checkpoint.parent.iterdir()] == ["best.pt"]


def test_a_dry_run_report_names_no_checkpoint(eval_main: Any, tmp_path: Path) -> None:
    output = tmp_path / "dry_run.json"
    eval_main(["--dry-run", "--operation", "memory", "-o", str(output)])

    metadata = json.loads(output.read_text(encoding="utf-8"))["metadata"]
    assert metadata["mode"] == "dry-run"
    assert metadata["source"] == "data/ci_shard/memory"
    assert metadata["checkpoint"] is None
    assert metadata["config"]["operation"] == "memory"
