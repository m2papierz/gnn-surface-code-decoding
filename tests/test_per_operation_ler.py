"""Tests for per-operation LER: per-observable breakdown, metric policy gating,
space-time convention, and observable names on EvalSet.

Covers LS11 done-when items without modifying any existing test file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.evaluator import (
    EvalPointResult,
    EvalReport,
    EvalSet,
    _point_to_dict,
    evaluate_point,
    load_eval_set,
)
from evaluation.stats import WilsonInterval
from sampling.profile import MetricPolicy


CI_SHARD_DIR = Path(__file__).resolve().parent.parent / "data" / "ci_shard" / "memory"
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "memory"


class _FakeDecoder:
    """Deterministic decoder: returns a fixed prediction matrix."""

    def __init__(self, predictions: np.ndarray, *, name: str = "fake") -> None:
        self._predictions = predictions
        self._name = name
        self._offset = 0

    @property
    def name(self) -> str:
        return self._name

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        n = syndromes.shape[0]
        chunk = self._predictions[self._offset : self._offset + n]
        self._offset += n
        return chunk


def _make_eval_set(
    n: int,
    n_det: int,
    n_obs: int,
    *,
    seed: int = 42,
    observable_names: tuple[str, ...] | None = None,
) -> EvalSet:
    rng = np.random.default_rng(seed)
    syndromes = rng.integers(0, 2, size=(n, n_det), dtype=np.uint8)
    observables = rng.integers(0, 2, size=(n, n_obs), dtype=np.uint8)
    coords = rng.random((n_det, 3))

    return EvalSet(
        syndromes=syndromes,
        observables=observables,
        detector_coords=coords,
        distance=3,
        rounds=3,
        error_prob=0.01,
        num_shots=n,
        circuit_file="data/circuits/memory/d3_r3_p0_01.stim",
        manifest={
            "distance": 3,
            "rounds": 3,
            "error_prob": 0.01,
            "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
        },
        observable_names=observable_names,
    )


class TestPrimaryMetricKGt1:
    """Operation fails if *any* scored observable is wrong."""

    def test_any_observable_wrong_counts_as_error(self) -> None:
        n, n_det, n_obs = 200, 10, 2
        rng = np.random.default_rng(7)
        es = _make_eval_set(n, n_det, n_obs, seed=7)

        # Decoder A: gets obs 0 right always, gets obs 1 wrong 50% of the time
        preds_a = es.observables.copy()
        flip_mask = rng.random(n) < 0.5
        preds_a[flip_mask, 1] ^= 1

        # Decoder B: always predicts all-zero
        preds_b = np.zeros((n, n_obs), dtype=np.uint8)

        decoders = {
            "a": _FakeDecoder(preds_a, name="a"),
            "b": _FakeDecoder(preds_b, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )

        dr_a = result.decoder_results["a"]
        # Shot-level error = any observable wrong.  obs 0 always correct,
        # obs 1 wrong for ~50% of shots => ~50% shot-level errors for the
        # flipped portion.
        n_obs1_wrong = int(flip_mask.sum())
        # Each flipped shot has obs 1 wrong => shot is wrong
        assert dr_a.n_errors == n_obs1_wrong

    def test_all_observables_correct_means_no_error(self) -> None:
        n, n_det, n_obs = 200, 10, 3
        es = _make_eval_set(n, n_det, n_obs, seed=11)

        preds_perfect = es.observables.copy()
        preds_zero = np.zeros((n, n_obs), dtype=np.uint8)

        decoders = {
            "perfect": _FakeDecoder(preds_perfect, name="perfect"),
            "zero": _FakeDecoder(preds_zero, name="zero"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="perfect",
            check_interval=n,
        )
        assert result.decoder_results["perfect"].n_errors == 0


class TestPerObservableBreakdown:
    def test_per_observable_present_at_k_gt_1(self) -> None:
        n, n_det, n_obs = 200, 10, 2
        es = _make_eval_set(
            n,
            n_det,
            n_obs,
            observable_names=("XX", "ZZ"),
        )
        preds = np.zeros((n, n_obs), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )

        dr = result.decoder_results["a"]
        assert len(dr.per_observable) == 2
        assert dr.per_observable[0].name == "XX"
        assert dr.per_observable[1].name == "ZZ"

    def test_per_observable_has_wilson_intervals(self) -> None:
        n, n_det, n_obs = 200, 10, 2
        es = _make_eval_set(n, n_det, n_obs)
        preds = np.zeros((n, n_obs), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )

        for obs_r in result.decoder_results["a"].per_observable:
            assert isinstance(obs_r.ler_interval, WilsonInterval)
            assert obs_r.n_shots == n
            assert obs_r.n_errors >= 0

    def test_per_observable_errors_sum_ge_shot_errors(self) -> None:
        """Per-obs errors can exceed shot-level errors (double-counted shots)."""
        n, n_det, n_obs = 200, 10, 2
        es = _make_eval_set(n, n_det, n_obs, seed=99)
        preds = np.zeros((n, n_obs), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )

        dr = result.decoder_results["a"]
        sum_per_obs_errors = sum(obs.n_errors for obs in dr.per_observable)
        assert sum_per_obs_errors >= dr.n_errors

    def test_per_observable_at_k1(self) -> None:
        """Per-observable breakdown is present even at K=1."""
        n, n_det = 200, 10
        es = _make_eval_set(n, n_det, 1)
        preds = np.zeros((n, 1), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )

        dr = result.decoder_results["a"]
        assert len(dr.per_observable) == 1
        assert dr.per_observable[0].n_errors == dr.n_errors

    def test_default_observable_names_when_none(self) -> None:
        n, n_det, n_obs = 200, 10, 3
        es = _make_eval_set(n, n_det, n_obs, observable_names=None)
        preds = np.zeros((n, n_obs), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )

        names = [obs.name for obs in result.decoder_results["a"].per_observable]
        assert names == ["observable_0", "observable_1", "observable_2"]

    def test_per_observable_serialized(self) -> None:
        n, n_det, n_obs = 200, 10, 2
        es = _make_eval_set(
            n,
            n_det,
            n_obs,
            observable_names=("XX", "ZZ"),
        )
        preds = np.zeros((n, n_obs), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )
        row = _point_to_dict(result)
        per_obs = row["decoders"]["a"]["per_observable"]
        assert len(per_obs) == 2
        assert per_obs[0]["name"] == "XX"
        assert "ler" in per_obs[0]
        assert "ler_ci_95" in per_obs[0]
        assert "n_shots" in per_obs[0]
        assert "n_errors" in per_obs[0]


class TestPerRoundLerGating:
    """Per-round LER absent from results when metric_policy excludes it."""

    def _run_with_policy(self, include_per_round: bool) -> tuple[EvalPointResult, dict]:
        n, n_det = 200, 10
        es = _make_eval_set(n, n_det, 1)
        preds = np.zeros((n, 1), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }
        policy = MetricPolicy(include_per_round_ler=include_per_round)

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
            metric_policy=policy,
        )
        row = _point_to_dict(result)
        return result, row

    def test_per_round_present_when_policy_includes(self) -> None:
        result, row = self._run_with_policy(include_per_round=True)
        dec = row["decoders"]["a"]
        assert "per_round_ler" in dec
        assert "per_round_ler_ci_95" in dec

    def test_per_round_absent_when_policy_excludes(self) -> None:
        result, row = self._run_with_policy(include_per_round=False)
        dec = row["decoders"]["a"]
        assert "per_round_ler" not in dec
        assert "per_round_ler_ci_95" not in dec

    def test_per_round_present_when_no_policy(self) -> None:
        """Backward compatibility: no metric_policy => include per-round."""
        n, n_det = 200, 10
        es = _make_eval_set(n, n_det, 1)
        preds = np.zeros((n, 1), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )
        row = _point_to_dict(result)
        assert "per_round_ler" in row["decoders"]["a"]

    def test_data_still_computed_even_when_not_serialized(self) -> None:
        """The DecoderPointResult always carries per_round_ler (computation
        is not gated - only serialization is)."""
        result, _ = self._run_with_policy(include_per_round=False)
        for dr in result.decoder_results.values():
            assert dr.per_round_ler >= 0.0
            assert isinstance(dr.per_round_interval, WilsonInterval)


class TestSpaceTimeConvention:
    def test_convention_in_serialized_output(self) -> None:
        n, n_det = 200, 10
        es = _make_eval_set(n, n_det, 1)
        preds = np.zeros((n, 1), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
            space_time_convention="per_round_cycle",
        )
        row = _point_to_dict(result)
        assert row["space_time_convention"] == "per_round_cycle"

    def test_convention_absent_when_not_set(self) -> None:
        n, n_det = 200, 10
        es = _make_eval_set(n, n_det, 1)
        preds = np.zeros((n, 1), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )
        row = _point_to_dict(result)
        assert "space_time_convention" not in row


class TestObservableNamesOnEvalSet:
    def test_manifest_with_observable_names(self, tmp_path: Path) -> None:
        manifest = {
            "distance": 3,
            "rounds": 3,
            "error_prob": 0.01,
            "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
            "observable_names": ["logical_observable"],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        np.savez(
            tmp_path / "data.npz",
            syndromes=np.zeros((10, 24), dtype=np.uint8),
            observables=np.zeros((10, 1), dtype=np.uint8),
            detector_coords=np.zeros((24, 3)),
        )

        es = load_eval_set(tmp_path)
        assert es.observable_names == ("logical_observable",)

    def test_manifest_without_observable_names(self, tmp_path: Path) -> None:
        manifest = {
            "distance": 3,
            "rounds": 3,
            "error_prob": 0.01,
            "circuit_file": "data/circuits/memory/d3_r3_p0_01.stim",
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        np.savez(
            tmp_path / "data.npz",
            syndromes=np.zeros((10, 24), dtype=np.uint8),
            observables=np.zeros((10, 1), dtype=np.uint8),
            detector_coords=np.zeros((24, 3)),
        )

        es = load_eval_set(tmp_path)
        assert es.observable_names is None

    def test_observable_names_in_serialized_result(self) -> None:
        n, n_det, n_obs = 200, 10, 2
        es = _make_eval_set(
            n,
            n_det,
            n_obs,
            observable_names=("XX", "ZZ"),
        )
        preds = np.zeros((n, n_obs), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
        )
        row = _point_to_dict(result)
        assert row["observable_names"] == ["XX", "ZZ"]


class TestMemoryResultBackwardCompat:
    """A memory-track result serializes identically to before LS11,
    except for the new fields (per_observable, space_time_convention,
    observable_names) which are additive."""

    def test_memory_result_has_per_round_ler(self) -> None:
        """Memory results still carry per_round_ler - policy includes it."""
        n, n_det = 200, 10
        es = _make_eval_set(n, n_det, 1)
        preds = np.zeros((n, 1), dtype=np.uint8)
        decoders = {
            "a": _FakeDecoder(preds, name="a"),
            "b": _FakeDecoder(preds, name="b"),
        }
        policy = MetricPolicy(include_per_round_ler=True)

        result = evaluate_point(
            es,
            decoders,
            reference_decoder="a",
            check_interval=n,
            metric_policy=policy,
            space_time_convention="per_round_cycle",
        )
        row = _point_to_dict(result)

        assert "per_round_ler" in row["decoders"]["a"]
        assert "per_round_ler_ci_95" in row["decoders"]["a"]
        assert row["space_time_convention"] == "per_round_cycle"

        report = EvalReport(results=[result], metadata={"mode": "test"})
        json_str = json.dumps(report.to_dict())
        loaded = json.loads(json_str)
        assert loaded["points"][0]["operation"] == "memory"
