"""The calibration figure reads the operation's committed circuits and its model."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from artifact_file import file_sha256
from decoders import GNNDecoder
from model.decoder import build_model
from sampling.experiment import circuit_key
from sampling.profile import registered_operations, resolve_profile
from sampling.sampler import sample_shots, settings_from_circuit_dir
from training import TrainConfig, build_training_state, write_checkpoint


REPO_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_SCRIPT = REPO_ROOT / "scripts" / "plot_calibration.py"
DISTANCE = 3


@pytest.fixture(autouse=True)
def cpu_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Infer on CPU: CUDA scatter atomics make GPU logits differ run to run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(scope="module")
def calibration_main() -> Any:
    return runpy.run_path(str(CALIBRATION_SCRIPT))["main"]


def _write_checkpoint(path: Path, operation: str) -> Path:
    """An untrained checkpoint of *operation*, written by the trainer's writer."""
    cfg = TrainConfig.from_fields(
        {"operation": operation, "hidden_dim": 16, "num_layers": 1, "dropout": 0.0}
    )
    contract = resolve_profile(operation).data_contract
    state = build_training_state(
        cfg,
        contract=contract,
        total_budget=cfg.sample_budget,
        device=torch.device("cpu"),
    )
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


def _run(
    main: Any, checkpoint: Path, operation: str, output: Path, *, seed: int
) -> dict[str, Any]:
    main(
        [
            "--operation",
            operation,
            "--checkpoint",
            str(checkpoint),
            "--distance",
            str(DISTANCE),
            "--shots",
            "48",
            "--seed",
            str(seed),
            "--n-bins",
            "5",
            "-o",
            str(output),
        ]
    )
    return json.loads(output.with_suffix(".json").read_text())


@pytest.mark.parametrize("operation", sorted(registered_operations()))
def test_every_committed_circuit_at_the_distance_is_a_point(
    calibration_main: Any, tmp_path: Path, operation: str
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "best.pt", operation)
    figure = tmp_path / "calibration.png"
    report = _run(calibration_main, checkpoint, operation, figure, seed=0)

    profile = resolve_profile(operation)
    committed = {
        circuit_key(path)
        for path in profile.circuit_root.glob("*.stim")
        if circuit_key(path).distance == DISTANCE
    }
    reported = {
        (p["operation"], p["distance"], p["rounds"], p["error_prob"])
        for p in report["points"]
    }
    assert reported == {
        (k.operation, k.distance, k.rounds, k.error_prob) for k in committed
    }
    assert figure.stat().st_size > 0
    assert report["metadata"]["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert report["metadata"]["device"] == "cpu"
    for point in report["points"]:
        names = [obs["name"] for obs in point["observables"]]
        assert names == list(profile.observable_names)
        n_fired = point["n_shots"] - point["n_shots_without_detection_events"]
        for obs in point["observables"]:
            assert obs["n_samples"] == n_fired
            assert sum(b["n_samples"] for b in obs["bins"]) == n_fired
            for b in obs["bins"]:
                low, high = b["positive_fraction_ci_95"]
                assert low <= b["positive_fraction"] <= high


def test_the_seed_determines_the_report(calibration_main: Any, tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "best.pt", "memory")
    first = _run(calibration_main, checkpoint, "memory", tmp_path / "a.png", seed=5)
    again = _run(calibration_main, checkpoint, "memory", tmp_path / "b.png", seed=5)
    other = _run(calibration_main, checkpoint, "memory", tmp_path / "c.png", seed=6)
    assert first["points"] == again["points"]
    assert first["points"] != other["points"]


def test_a_checkpoint_of_another_operation_is_refused(
    calibration_main: Any, tmp_path: Path
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "best.pt", "memory")
    with pytest.raises(ValueError):
        _run(
            calibration_main,
            checkpoint,
            "zz_merge_split",
            tmp_path / "calibration.png",
            seed=0,
        )
    assert not (tmp_path / "calibration.json").exists()


@pytest.fixture(scope="module")
def merge_split_shots() -> tuple[Any, np.ndarray]:
    """Metadata and 64 shots of the committed d=3 merge-split circuit at the top p."""
    profile = resolve_profile("zz_merge_split")
    point = settings_from_circuit_dir(
        profile.circuit_root, operation="zz_merge_split", distances=[DISTANCE]
    )[-1]
    metadata = profile.metadata_extractor(
        point.circuit_path, num_observables=profile.data_contract.num_observables
    )
    syndromes, _ = sample_shots(point.circuit_path, shots=64, seed=11)
    syndromes[:4] = 0  # shots no model sees
    return metadata, syndromes


def _merge_split_decoder(
    metadata: Any, *, threshold: float, batch_size: int
) -> GNNDecoder:
    contract = resolve_profile("zz_merge_split").data_contract
    torch.manual_seed(0)
    model = build_model(
        node_dim=contract.node_dim,
        edge_dim=contract.edge_dim,
        hidden_dim=16,
        num_layers=2,
        num_observables=contract.num_observables,
        dropout=0.0,
    )
    return GNNDecoder(
        model,
        metadata,
        contract=contract,
        threshold=threshold,
        device=torch.device("cpu"),
        batch_size=batch_size,
    )


def test_decode_batch_thresholds_the_logits_of_shots_that_fired(
    merge_split_shots: tuple[Any, np.ndarray],
) -> None:
    metadata, syndromes = merge_split_shots
    fired = syndromes.any(axis=1)
    logits = _merge_split_decoder(metadata, threshold=0.0, batch_size=16).logits(
        syndromes[fired]
    )
    # A threshold inside the logits' range, so both decisions occur.
    threshold = float(np.median(logits))
    decoder = _merge_split_decoder(metadata, threshold=threshold, batch_size=16)

    expected = np.zeros((syndromes.shape[0], logits.shape[1]), dtype=np.uint8)
    expected[fired] = logits > threshold
    assert 0 < expected[fired].sum() < expected[fired].size
    np.testing.assert_array_equal(decoder.decode_batch(syndromes), expected)


def test_logits_do_not_depend_on_the_batch(
    merge_split_shots: tuple[Any, np.ndarray],
) -> None:
    metadata, syndromes = merge_split_shots
    fired_shots = syndromes[syndromes.any(axis=1)]
    one = _merge_split_decoder(metadata, threshold=0.0, batch_size=1)
    many = _merge_split_decoder(metadata, threshold=0.0, batch_size=25)
    # float32 CPU forward; batching changes only the summation order of the
    # matmuls, observed below 1e-6 here.
    np.testing.assert_allclose(
        one.logits(fired_shots), many.logits(fired_shots), rtol=0, atol=1e-5
    )


def test_logits_refuse_a_shot_without_a_detection_event(
    merge_split_shots: tuple[Any, np.ndarray],
) -> None:
    metadata, syndromes = merge_split_shots
    decoder = _merge_split_decoder(metadata, threshold=0.0, batch_size=16)
    with pytest.raises(ValueError, match="no detection event"):
        decoder.logits(syndromes)
