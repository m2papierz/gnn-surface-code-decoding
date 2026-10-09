"""Tests for the scaling diagnostics: statistics, rule, measurements and script."""

from __future__ import annotations

import json
import math
import runpy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

from evaluation.distance_scaling import (
    DistanceRate,
    StepTrend,
    ThresholdClassification,
    ThresholdVerdict,
)
from evaluation.evaluator import EvalSet, load_eval_set
from evaluation.scaling_diagnostics import (
    PointDiagnostics,
    classify_threshold,
    compare_baselines,
    dem_mass_split,
    graph_size,
    observable_flip_rates,
    report_to_dict,
)
from evaluation.stats import (
    CHECK_INTERVAL,
    FINAL_ALPHA,
    MIN_EXPECTED_CELL_COUNT,
    EvalOutcome,
    two_proportion_test,
    wilson_interval,
)
from sampling.dem import MechanismClass, read_dem_mechanisms
from sampling.experiment import ExperimentKey
from sampling.graph import complete_graph_edge_count
from sampling.profile import resolve_profile
from sampling.representation import registered_representations, resolve_builder


REPO_ROOT = Path(__file__).resolve().parent.parent
CI_SHARD_DIR = REPO_ROOT / "data" / "ci_shard" / "memory"
MERGE_SPLIT_EVAL_DIR = REPO_ROOT / "data" / "eval" / "zz_merge_split" / "d3_p0_0030"
SCRIPT = REPO_ROOT / "scripts" / "diagnose_scaling.py"

DEM_TEXT = """
error(0.1) D0 D1
error(0.2) D0
error(0.05) D1 ^ D2
error(0.15) D0 D1 ^ D2 D3
error(0.5) D2 D3 L0
"""


class _FixedErrorsDecoder:
    """Decoder that is wrong on a chosen set of shots and right on the rest."""

    def __init__(
        self, name: str, observables: np.ndarray, wrong_shots: np.ndarray
    ) -> None:
        self._name = name
        self._predictions = observables.copy()
        self._predictions[wrong_shots] ^= 1

    @property
    def name(self) -> str:
        return self._name

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        if syndromes.shape[0] != self._predictions.shape[0]:
            raise AssertionError(
                f"expected one call with all {self._predictions.shape[0]} shots, "
                f"got {syndromes.shape[0]}"
            )
        return self._predictions


def _memory_eval_set(distance: int, num_shots: int) -> EvalSet:
    """Synthetic shots attached to a committed memory circuit's identity."""
    rng = np.random.default_rng(distance)
    circuit_file = f"data/circuits/memory/d{distance}_r{distance}_p0_01.stim"
    return EvalSet(
        syndromes=rng.integers(0, 2, size=(num_shots, 8), dtype=np.uint8),
        observables=rng.integers(0, 2, size=(num_shots, 1), dtype=np.uint8),
        detector_coords=rng.random((8, 3)),
        distance=distance,
        rounds=distance,
        error_prob=0.01,
        num_shots=num_shots,
        circuit_file=circuit_file,
        manifest={
            "distance": distance,
            "rounds": distance,
            "error_prob": 0.01,
            "circuit_file": circuit_file,
        },
    )


def _point(distance: int, errors: dict[str, int], num_shots: int) -> PointDiagnostics:
    """Diagnose synthetic shots with decoders making a chosen number of errors."""
    eval_set = _memory_eval_set(distance, num_shots)
    reference, compared = (
        _FixedErrorsDecoder(name, eval_set.observables, np.arange(n_errors))
        for name, n_errors in errors.items()
    )
    return PointDiagnostics(
        key=ExperimentKey("memory", distance, distance, 0.01),
        circuit_file=eval_set.circuit_file,
        circuit_sha256="0" * 64,
        shots_sha256="0" * 64,
        graph=graph_size(eval_set.syndromes),
        observable_flips=observable_flip_rates(
            eval_set.observables, ("logical_observable",)
        ),
        dem=dem_mass_split(read_dem_mechanisms(stim.DetectorErrorModel(DEM_TEXT))),
        baselines=compare_baselines(
            eval_set,
            reference=reference,
            compared=compared,
            profile=resolve_profile("memory"),
        ),
    )


class TestTwoProportionTest:
    def test_hand_computed_value(self) -> None:
        # pooled = 0.2, variance = 0.2 * 0.8 * (1/100 + 1/100) = 0.0032,
        # z = 0.2 / sqrt(0.0032) = 5 / sqrt(2), p = erfc(2.5).
        result = two_proportion_test(30, 100, 10, 100)

        assert result.statistic == pytest.approx(5 / math.sqrt(2), rel=1e-12)
        assert result.p_value == pytest.approx(math.erfc(2.5), rel=1e-12)
        assert result.min_expected_count == pytest.approx(20.0)

    def test_swapping_samples_flips_the_sign_only(self) -> None:
        forward = two_proportion_test(30, 100, 10, 400)
        backward = two_proportion_test(10, 400, 30, 100)

        assert backward.statistic == pytest.approx(-forward.statistic)
        assert backward.p_value == pytest.approx(forward.p_value)
        assert backward.min_expected_count == forward.min_expected_count

    def test_min_expected_count_is_the_smallest_cell(self) -> None:
        # pooled = 40 / 500 = 0.08; the smallest cell is errors in the sample
        # of 100: 100 * 0.08 = 8.
        assert two_proportion_test(30, 100, 10, 400).min_expected_count == (
            pytest.approx(8.0)
        )

    def test_equal_rates_do_not_reject(self) -> None:
        result = two_proportion_test(50, 1000, 100, 2000)

        assert result.statistic == 0.0
        assert result.p_value == 1.0

    @pytest.mark.parametrize("errors", [(0, 0), (10, 20)], ids=["none", "all"])
    def test_degenerate_pooled_rate_cannot_differ(
        self, errors: tuple[int, int]
    ) -> None:
        result = two_proportion_test(errors[0], 10, errors[1], 20)

        assert (result.statistic, result.p_value) == (0.0, 1.0)
        assert result.min_expected_count == 0.0

    @pytest.mark.parametrize(
        "args",
        [(1, 0, 1, 10), (1, 10, 1, 0), (-1, 10, 1, 10), (1, 10, 11, 10)],
        ids=["empty_a", "empty_b", "negative", "errors_exceed_total"],
    )
    def test_invalid_counts_raise(self, args: tuple[int, int, int, int]) -> None:
        with pytest.raises(ValueError, match="n_(total|errors)_[ab]"):
            two_proportion_test(*args)


class TestGraphSize:
    def test_hand_computed_distributions(self) -> None:
        syndromes = np.array(
            [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=np.uint8
        )
        size = graph_size(syndromes)

        # fired = [0, 1, 3, 4]; edges = N * (N - 1) = [0, 0, 6, 12].
        assert (size.num_detectors, size.num_shots) == (4, 4)
        assert size.fired_detectors.mean == 2.0
        assert size.fired_detectors.p99 == pytest.approx(3.97)
        assert size.fired_detectors.max == 4
        assert size.edges.mean == 4.5
        assert size.edges.p99 == pytest.approx(11.82)
        assert size.edges.max == 12

    def test_edge_law_values(self) -> None:
        assert [complete_graph_edge_count(n) for n in (0, 1, 2, 5)] == [0, 0, 2, 20]
        np.testing.assert_array_equal(
            complete_graph_edge_count(np.array([0, 1, 600], dtype=np.int64)),
            [0, 0, 359_400],
        )

    @pytest.mark.parametrize(
        "version", [descriptor.version for descriptor in registered_representations()]
    )
    def test_edge_law_is_what_every_registered_builder_produces(
        self, version: str
    ) -> None:
        eval_set = load_eval_set(MERGE_SPLIT_EVAL_DIR)
        builder = resolve_builder(version)

        for syndrome in eval_set.syndromes[:32]:
            graph = builder(syndrome, eval_set.circuit_metadata)
            assert graph.edge_index.shape[1] == complete_graph_edge_count(
                int(np.count_nonzero(syndrome))
            )


class TestObservableFlipRates:
    OBSERVABLES = np.array(
        [[1, 0], [1, 1], [0, 0], [0, 1], [0, 0], [1, 0], [0, 0], [0, 0]],
        dtype=np.uint8,
    )

    def test_hand_computed_counts_and_intervals(self) -> None:
        flips = observable_flip_rates(self.OBSERVABLES, ("Z_0", "Z_1"))

        # Z_0 flips in shots 0, 1, 5; Z_1 in shots 1, 3; some observable in
        # shots 0, 1, 3, 5.
        assert flips.per_observable == {
            "Z_0": wilson_interval(3, 8),
            "Z_1": wilson_interval(2, 8),
        }
        assert flips.any_observable == wilson_interval(4, 8)

    def test_serializes_counts_rates_and_intervals_in_profile_order(self) -> None:
        serialized = observable_flip_rates(self.OBSERVABLES, ("Z_1", "Z_0")).to_dict()
        z0_interval = wilson_interval(2, 8)

        assert [entry["name"] for entry in serialized["per_observable"]] == [
            "Z_1",
            "Z_0",
        ]
        assert serialized["per_observable"][1] == {
            "name": "Z_0",
            "n_shots": 8,
            "n_flips": 2,
            "rate": 0.25,
            "rate_ci_95": [z0_interval.lower, z0_interval.upper],
        }
        assert serialized["any_observable"]["n_flips"] == 4
        json.dumps(serialized, allow_nan=False)

    def test_names_must_match_the_observable_columns(self) -> None:
        with pytest.raises(ValueError, match="2 observable column.*1 name"):
            observable_flip_rates(self.OBSERVABLES, ("Z_0",))


class TestDemMassSplit:
    def test_hand_computed_shares(self) -> None:
        split = dem_mass_split(read_dem_mechanisms(stim.DetectorErrorModel(DEM_TEXT)))

        assert split.num_mechanisms == 5
        assert split.probability_mass == pytest.approx(1.0)

        two_detector = split.classes[MechanismClass.TWO_DETECTOR]
        assert two_detector.num_mechanisms == 2
        assert two_detector.mass_share == pytest.approx(0.6)
        assert two_detector.dem_weight_mass_share == pytest.approx(0.6)

        boundary = split.classes[MechanismClass.BOUNDARY]
        assert boundary.num_mechanisms == 1
        assert boundary.mass_share == pytest.approx(0.2)
        assert boundary.dem_weight_mass_share == 0.0

        # Of the two decomposed mechanisms only ``D1 ^ D2`` flips exactly two
        # detectors, so only its 0.05 reaches the edge feature.
        decomposed = split.classes[MechanismClass.DECOMPOSED]
        assert decomposed.num_mechanisms == 2
        assert decomposed.mass_share == pytest.approx(0.2)
        assert decomposed.dem_weight_mass_share == pytest.approx(0.05)

        assert split.dem_weight_mass_share == pytest.approx(0.65)

    def test_noiseless_model_raises(self) -> None:
        mechanisms = read_dem_mechanisms(stim.DetectorErrorModel("error(0) D0 D1"))

        with pytest.raises(ValueError, match="no probability mass"):
            dem_mass_split(mechanisms)


class TestCompareBaselines:
    NUM_SHOTS = 3 * CHECK_INTERVAL

    def _compare(self, reference_wrong: np.ndarray, compared_wrong: np.ndarray):
        eval_set = _memory_eval_set(3, self.NUM_SHOTS)
        return compare_baselines(
            eval_set,
            reference=_FixedErrorsDecoder(
                "correlated", eval_set.observables, reference_wrong
            ),
            compared=_FixedErrorsDecoder("mwpm", eval_set.observables, compared_wrong),
            profile=resolve_profile("memory"),
        )

    def test_scores_every_shot_in_one_analysis(self) -> None:
        # The gap is wide enough to cross the interim boundary within the
        # first check interval; an adaptive run would stop there.
        comparison = self._compare(np.arange(200), np.arange(100, 1300))

        assert comparison.reference.n_shots == self.NUM_SHOTS
        assert comparison.compared.n_shots == self.NUM_SHOTS
        assert comparison.reference.n_errors == 200
        assert comparison.compared.n_errors == 1200

    def test_discordant_counts_are_oriented_by_role(self) -> None:
        comparison = self._compare(np.arange(200), np.arange(100, 1300))
        serialized = comparison.to_dict()

        # Shots 0-99: only the reference is wrong. Shots 200-1299: only the
        # compared decoder is wrong. Shots 100-199: both wrong.
        assert serialized["mcnemar"]["compared_only_correct"] == 100
        assert serialized["mcnemar"]["reference_only_correct"] == 1100
        assert serialized["mcnemar"]["n_discordant"] == 1200
        assert comparison.outcome is EvalOutcome.RESOLVED_DIFFERENT
        assert serialized["better"] == "correlated"
        assert (serialized["reference"], serialized["compared"]) == (
            "correlated",
            "mwpm",
        )

    def test_better_follows_the_data_not_the_role(self) -> None:
        comparison = self._compare(np.arange(100, 1300), np.arange(200))

        assert comparison.outcome is EvalOutcome.RESOLVED_DIFFERENT
        assert comparison.better == "mwpm"

    def test_parity_names_no_better_decoder(self) -> None:
        comparison = self._compare(np.arange(0, 300), np.arange(150, 450))

        assert comparison.outcome is EvalOutcome.RESOLVED_PARITY
        assert comparison.better is None

    def test_below_the_error_floor_is_unresolved(self) -> None:
        comparison = self._compare(np.arange(10), np.arange(5, 95))

        assert comparison.outcome is EvalOutcome.UNRESOLVED
        assert comparison.better is None

    def test_rates_carry_intervals_counts_and_policy_gated_metrics(self) -> None:
        entry = self._compare(np.arange(200), np.arange(100, 1300)).to_dict()[
            "decoders"
        ]["mwpm"]

        assert entry["n_shots"] == self.NUM_SHOTS
        assert entry["n_errors"] == 1200
        assert entry["ler_ci_95"][0] < entry["ler"] < entry["ler_ci_95"][1]
        # Memory's metric policy admits per-round LER.
        assert "per_round_ler_ci_95" in entry


class TestThresholdRule:
    @staticmethod
    def _classify(*rates: tuple[int, int, int]) -> ThresholdClassification:
        return ThresholdClassification(
            operation="memory",
            decoder="mwpm",
            error_prob=0.005,
            rates=tuple(DistanceRate(*rate) for rate in rates),
        )

    def test_rate_falling_at_every_step_is_sub_threshold(self) -> None:
        row = self._classify((3, 3000, 100_000), (5, 1000, 100_000), (7, 300, 100_000))

        assert [step.trend for step in row.steps] == [StepTrend.DECREASING] * 2
        assert row.verdict is ThresholdVerdict.SUB_THRESHOLD

    def test_rate_rising_at_every_step_is_above_threshold(self) -> None:
        row = self._classify((3, 300, 100_000), (5, 1000, 100_000), (7, 3000, 100_000))

        assert [step.trend for step in row.steps] == [StepTrend.INCREASING] * 2
        assert row.verdict is ThresholdVerdict.ABOVE_THRESHOLD

    def test_mixed_trends_are_undetermined(self) -> None:
        row = self._classify((3, 3000, 100_000), (5, 1000, 100_000), (7, 3000, 100_000))

        assert [step.trend for step in row.steps] == [
            StepTrend.DECREASING,
            StepTrend.INCREASING,
        ]
        assert row.verdict is ThresholdVerdict.UNDETERMINED

    def test_one_step_that_does_not_reject_is_undetermined(self) -> None:
        row = self._classify((3, 3000, 100_000), (5, 1000, 100_000), (7, 990, 100_000))

        assert row.steps[1].p_value > FINAL_ALPHA
        assert row.steps[1].trend is StepTrend.UNRESOLVED
        assert row.verdict is ThresholdVerdict.UNDETERMINED

    def test_single_distance_is_undetermined(self) -> None:
        row = self._classify((3, 3000, 100_000))

        assert row.steps == ()
        assert row.verdict is ThresholdVerdict.UNDETERMINED

    def test_small_expected_cell_blocks_a_rejecting_test(self) -> None:
        # 8/10 against 0/10 rejects (z = 3.65), but the pooled rate 0.4 puts
        # only 4 expected errors in each sample.
        row = self._classify((3, 8, 10), (5, 0, 10))
        (step,) = row.steps

        assert step.p_value < FINAL_ALPHA
        assert step.min_expected_count == pytest.approx(4.0)
        assert step.min_expected_count < MIN_EXPECTED_CELL_COUNT
        assert step.trend is StepTrend.UNRESOLVED

    def test_few_errors_at_the_larger_distance_still_resolve(self) -> None:
        # 30 errors would fail the paired test's 100-error floor; here it is
        # the evidence that the rate fell.
        row = self._classify((5, 400, 1_000_000), (7, 30, 1_000_000))

        assert row.steps[0].min_expected_count == pytest.approx(215.0)
        assert row.verdict is ThresholdVerdict.SUB_THRESHOLD

    def test_step_carries_the_two_proportion_test(self) -> None:
        (step,) = self._classify((3, 30, 100), (5, 10, 100)).steps

        assert (step.from_distance, step.to_distance) == (3, 5)
        assert step.statistic == pytest.approx(5 / math.sqrt(2))
        assert step.p_value == pytest.approx(math.erfc(2.5))

    @pytest.mark.parametrize(
        "distances", [(5, 3), (3, 3)], ids=["descending", "duplicate"]
    )
    def test_rates_must_ascend_in_distance(self, distances: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="distinct ascending distances"):
            self._classify(*((distance, 10, 1000) for distance in distances))


class TestClassifyThresholdAcrossPoints:
    NUM_SHOTS = 20_000

    @pytest.fixture(scope="class")
    def points(self) -> list[PointDiagnostics]:
        # Listed out of distance order: the ladder must not depend on it.
        errors = {
            7: {"correlated": 400, "mwpm": 4500},
            3: {"correlated": 2000, "mwpm": 2000},
            5: {"correlated": 1000, "mwpm": 3000},
        }
        return [
            _point(distance, counts, self.NUM_SHOTS)
            for distance, counts in errors.items()
        ]

    def test_each_decoder_gets_its_own_ladder(
        self, points: list[PointDiagnostics]
    ) -> None:
        rows = {row.decoder: row for row in classify_threshold(points)}

        assert [rate.distance for rate in rows["correlated"].rates] == [3, 5, 7]
        assert [rate.n_errors for rate in rows["correlated"].rates] == [
            2000,
            1000,
            400,
        ]
        assert rows["correlated"].verdict is ThresholdVerdict.SUB_THRESHOLD
        assert rows["mwpm"].verdict is ThresholdVerdict.ABOVE_THRESHOLD

    def test_report_states_verdicts_rules_and_ordered_points(
        self, points: list[PointDiagnostics]
    ) -> None:
        report = report_to_dict(points, provenance={"command": "test"})

        assert [point["distance"] for point in report["points"]] == [3, 5, 7]
        assert report["threshold_summary"] == {
            "memory": {
                "correlated": {
                    "sub-threshold": [0.01],
                    "above-threshold": [],
                    "undetermined": [],
                },
                "mwpm": {
                    "sub-threshold": [],
                    "above-threshold": [0.01],
                    "undetermined": [],
                },
            }
        }
        assert report["rules"]["threshold"]["alpha"] == FINAL_ALPHA
        assert report["rules"]["threshold"]["min_expected_cell_count"] == (
            MIN_EXPECTED_CELL_COUNT
        )
        assert [entry["version"] for entry in report["representations"]] == [
            descriptor.version for descriptor in registered_representations()
        ]
        json.dumps(report, allow_nan=False)

    def test_two_points_at_one_distance_raise(
        self, points: list[PointDiagnostics]
    ) -> None:
        with pytest.raises(ValueError, match="distinct ascending distances"):
            classify_threshold([*points, points[0]])


class TestDiagnoseScalingScript:
    @pytest.fixture(scope="class")
    def script(self) -> dict[str, Any]:
        return runpy.run_path(str(SCRIPT))

    @pytest.fixture
    def eval_root(self, tmp_path: Path) -> Path:
        """An evaluation tree holding the CI shard as memory's d=3, p=0.01 set."""
        shutil.copytree(CI_SHARD_DIR, tmp_path / "memory" / "d3_p0_0100")
        return tmp_path

    def _run(self, script: dict[str, Any], eval_root: Path, *extra: str) -> Path:
        output = eval_root / "diagnostics.json"
        script["main"](
            [
                "--operations",
                "memory",
                "--eval-root",
                str(eval_root),
                "--distances",
                "3",
                *extra,
                "-o",
                str(output),
            ]
        )
        return output

    def test_writes_the_artifact_for_the_ci_shard(
        self, script: dict[str, Any], eval_root: Path
    ) -> None:
        output = self._run(script, eval_root, "--error-probs", "0.01")
        report = json.loads(output.read_text(encoding="utf-8"))
        syndromes = np.load(CI_SHARD_DIR / "syndromes.npy")
        manifest = json.loads((CI_SHARD_DIR / "manifest.json").read_text())

        (point,) = report["points"]
        assert (point["operation"], point["distance"], point["rounds"]) == (
            "memory",
            3,
            3,
        )
        assert point["error_prob"] == 0.01
        assert point["circuit_sha256"] == manifest["circuit_sha256"]

        fired = syndromes.sum(axis=1)
        assert point["graph"]["num_detectors"] == syndromes.shape[1]
        assert point["graph"]["num_shots"] == syndromes.shape[0]
        assert point["graph"]["fired_detectors"]["mean"] == pytest.approx(fired.mean())
        assert point["graph"]["fired_detectors"]["max"] == fired.max()
        assert point["graph"]["edges"]["max"] == fired.max() * (fired.max() - 1)

        shares = [entry["mass_share"] for entry in point["dem"]["classes"].values()]
        assert sum(shares) == pytest.approx(1.0)
        assert set(point["dem"]["classes"]) == {cls.value for cls in MechanismClass}
        assert 0.0 < point["dem"]["dem_weight_mass_share"] < 1.0

        observables = np.load(CI_SHARD_DIR / "observables.npy")
        flips = point["observable_flips"]
        assert [entry["name"] for entry in flips["per_observable"]] == [
            "logical_observable"
        ]
        assert flips["per_observable"][0]["n_flips"] == observables.sum()
        assert flips["any_observable"]["n_shots"] == syndromes.shape[0]
        assert flips["any_observable"]["n_flips"] == manifest["positive_count"]

        baselines = point["baselines"]
        assert set(baselines["decoders"]) == {"correlated", "mwpm"}
        assert all(
            entry["n_shots"] == syndromes.shape[0]
            for entry in baselines["decoders"].values()
        )
        # 256 shots cannot hold 100 errors per decoder at this error rate.
        assert baselines["outcome"] == EvalOutcome.UNRESOLVED.value

        assert {row["verdict"] for row in report["threshold"]} == {
            ThresholdVerdict.UNDETERMINED.value
        }
        assert "scripts/diagnose_scaling.py" in report["provenance"]["command"]
        assert report["provenance"]["versions"]["stim"] == stim.__version__

    def test_same_inputs_give_the_same_artifact(
        self, script: dict[str, Any], eval_root: Path
    ) -> None:
        first = self._run(script, eval_root, "--error-probs", "0.01").read_bytes()
        second = self._run(script, eval_root, "--error-probs", "0.01").read_bytes()

        assert first == second

    def test_committed_circuit_without_a_frozen_set_fails_the_run(
        self, script: dict[str, Any], eval_root: Path
    ) -> None:
        # Without the error-rate filter, d=3 has four committed circuits and
        # the tree holds a set for one of them.
        with pytest.raises(ValueError, match="No frozen eval set.*p=0.003"):
            self._run(script, eval_root)

        assert not (eval_root / "diagnostics.json").exists()

    def test_circuit_that_no_longer_matches_its_set_fails_the_run(
        self, script: dict[str, Any], eval_root: Path
    ) -> None:
        manifest_path = eval_root / "memory" / "d3_p0_0100" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["circuit_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="hashes to"):
            self._run(script, eval_root, "--error-probs", "0.01")

    def test_set_naming_another_circuit_than_its_point_fails_the_run(
        self, script: dict[str, Any], eval_root: Path
    ) -> None:
        manifest_path = eval_root / "memory" / "d3_p0_0100" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["circuit_file"] = "data/circuits/memory/d3_r3_p0_008.stim"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="circuit discovery resolves"):
            self._run(script, eval_root, "--error-probs", "0.01")

    def test_shots_not_holding_the_manifest_positive_count_fail_the_run(
        self, script: dict[str, Any], eval_root: Path
    ) -> None:
        manifest_path = eval_root / "memory" / "d3_p0_0100" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["positive_count"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="positive_count=42"):
            self._run(script, eval_root, "--error-probs", "0.01")

        assert not (eval_root / "diagnostics.json").exists()
