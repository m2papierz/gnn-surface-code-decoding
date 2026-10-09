"""Scaling and baseline diagnostics on frozen evaluation sets.

Four measurements per experiment point - the size of the graphs its shots
build, how often each logical observable flips over them, how the probability
mass of its detector error model splits by mechanism class, and how two
classical baselines score on every frozen shot - and one rule across points,
defined in ``evaluation.distance_scaling``: for which physical error rates a
decoder's logical error rate falls with distance.

Nothing here samples or reads a file, so every number is a function of the
frozen shots and the circuit they came from.  The caller loads the sets and
supplies the decoders.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from decoders import Decoder
from evaluation.distance_scaling import (
    DistanceRate,
    ThresholdClassification,
    ThresholdVerdict,
    rule_to_dict,
)
from evaluation.evaluator import (
    DecoderPointResult,
    EvalSet,
    decoder_result_to_dict,
    evaluate_point,
)
from evaluation.stats import (
    FINAL_ALPHA,
    MIN_ERRORS_PER_DECODER,
    EvalOutcome,
    McNemarResult,
    WilsonInterval,
    wilson_interval,
)
from sampling.dem import DemMechanisms, MechanismClass
from sampling.experiment import ExperimentKey
from sampling.graph import complete_graph_edge_count, dem_weight_mechanisms
from sampling.profile import OperationProfile
from sampling.representation import registered_representations


__all__ = [
    "BaselineComparison",
    "DemMassSplit",
    "GraphSize",
    "ObservableFlipRates",
    "PointDiagnostics",
    "classify_threshold",
    "compare_baselines",
    "dem_mass_split",
    "graph_size",
    "observable_flip_rates",
    "report_to_dict",
]


@dataclass(frozen=True, slots=True)
class CountDistribution:
    """Summary of a per-shot count over one frozen set.

    Parameters
    ----------
    mean : float
        Mean over shots.
    p99 : float
        99th percentile, linearly interpolated.
    max : int
        Largest value in any shot.
    """

    mean: float
    p99: float
    max: int


@dataclass(frozen=True, slots=True)
class GraphSize:
    """Size of the fired-detector graphs one frozen set builds.

    Parameters
    ----------
    num_detectors : int
        Detectors in the circuit.
    num_shots : int
        Shots summarised.
    fired_detectors : CountDistribution
        Fired detectors per shot, which is the node count of its graph.
    edges : CountDistribution
        Directed edges per graph.
    """

    num_detectors: int
    num_shots: int
    fired_detectors: CountDistribution
    edges: CountDistribution


def _distribution(counts: NDArray[np.int64]) -> CountDistribution:
    """Summarise one count per shot."""
    return CountDistribution(
        mean=float(counts.mean()),
        p99=float(np.percentile(counts, 99)),
        max=int(counts.max()),
    )


def graph_size(syndromes: NDArray[np.uint8]) -> GraphSize:
    """Summarise the graphs a set of shots builds, without building them.

    Parameters
    ----------
    syndromes : ndarray, shape ``(N, D)``, uint8
        Binary syndrome vectors, ``N >= 1``.

    Returns
    -------
    GraphSize
        Exact over all ``N`` shots: the edge count of each shot follows from
        its fired-detector count through the law the graph record validates.
    """
    fired = np.count_nonzero(syndromes, axis=1).astype(np.int64, copy=False)
    return GraphSize(
        num_detectors=syndromes.shape[1],
        num_shots=syndromes.shape[0],
        fired_detectors=_distribution(fired),
        edges=_distribution(complete_graph_edge_count(fired)),
    )


@dataclass(frozen=True, slots=True)
class ObservableFlipRates:
    """How often the logical observables flip over one frozen set.

    A flip is the circuit's own outcome for a shot, before any decoder, so
    these rates are the label balance every decoder is scored against.

    Parameters
    ----------
    per_observable : dict[str, WilsonInterval]
        Flip rate of each observable, keyed by name in the profile's order.
    any_observable : WilsonInterval
        Rate of shots in which at least one observable flips.
    """

    per_observable: dict[str, WilsonInterval]
    any_observable: WilsonInterval

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "per_observable": [
                {"name": name, **_flip_rate_to_dict(rate)}
                for name, rate in self.per_observable.items()
            ],
            "any_observable": _flip_rate_to_dict(self.any_observable),
        }


def _flip_rate_to_dict(rate: WilsonInterval) -> dict[str, Any]:
    """Serialize one flip rate with its counts and Wilson interval."""
    return {
        "n_shots": rate.n_total,
        "n_flips": rate.n_errors,
        "rate": rate.point,
        "rate_ci_95": [rate.lower, rate.upper],
    }


def observable_flip_rates(
    observables: NDArray[np.uint8], observable_names: Sequence[str]
) -> ObservableFlipRates:
    """Count the observable flips of a set of shots.

    Parameters
    ----------
    observables : ndarray, shape ``(N, O)``, uint8
        True observable flips, ``N >= 1``.
    observable_names : sequence of str
        Name of each of the ``O`` columns, in column order.

    Returns
    -------
    ObservableFlipRates
        Exact over all ``N`` shots, each rate with its Wilson 95% interval.

    Raises
    ------
    ValueError
        If the number of names differs from the number of columns.
    """
    num_shots, num_observables = observables.shape
    if num_observables != len(observable_names):
        raise ValueError(
            f"Shots carry {num_observables} observable column(s) but "
            f"{len(observable_names)} name(s) were given: {list(observable_names)}"
        )
    flipped = observables.astype(np.bool_, copy=False)
    flips = np.count_nonzero(flipped, axis=0)
    return ObservableFlipRates(
        per_observable={
            name: wilson_interval(int(count), num_shots)
            for name, count in zip(observable_names, flips, strict=True)
        },
        any_observable=wilson_interval(
            int(np.count_nonzero(flipped.any(axis=1))), num_shots
        ),
    )


@dataclass(frozen=True, slots=True)
class MechanismClassShare:
    """Share of a DEM's probability mass held by one mechanism class.

    Parameters
    ----------
    num_mechanisms : int
        Mechanisms in the class.
    mass_share : float
        The class's probability mass over the total.
    dem_weight_mass_share : float
        The part of ``mass_share`` that reaches the ``dem_weight`` edge
        feature, also as a share of the total.
    """

    num_mechanisms: int
    mass_share: float
    dem_weight_mass_share: float


@dataclass(frozen=True, slots=True)
class DemMassSplit:
    """Probability mass of a decomposed DEM, split by mechanism class.

    Parameters
    ----------
    num_mechanisms : int
        Error mechanisms in the model.
    probability_mass : float
        Sum of the mechanisms' probabilities: the expected number of
        mechanisms occurring per shot.
    classes : dict[MechanismClass, MechanismClassShare]
        One entry per class.  The ``mass_share`` values sum to 1.
    dem_weight_mass_share : float
        Share of the total mass carried by mechanisms the ``dem_weight`` edge
        feature is built from.
    """

    num_mechanisms: int
    probability_mass: float
    classes: dict[MechanismClass, MechanismClassShare]
    dem_weight_mass_share: float


def dem_mass_split(mechanisms: DemMechanisms) -> DemMassSplit:
    """Split a DEM's probability mass by mechanism class.

    Parameters
    ----------
    mechanisms : DemMechanisms
        Mechanisms of a decomposed detector error model.

    Returns
    -------
    DemMassSplit

    Raises
    ------
    ValueError
        If the model carries no probability mass, or holds a mechanism
        outside the three classes.
    """
    probability = mechanisms.probability
    total = float(probability.sum())
    if total <= 0.0:
        raise ValueError(
            f"DEM carries no probability mass over {probability.size} mechanism(s); "
            f"a noiseless circuit has nothing to split"
        )
    in_feature = dem_weight_mechanisms(mechanisms)
    classes = {
        mechanism_class: MechanismClassShare(
            num_mechanisms=int(mask.sum()),
            mass_share=float(probability[mask].sum()) / total,
            dem_weight_mass_share=float(probability[mask & in_feature].sum()) / total,
        )
        for mechanism_class, mask in mechanisms.class_masks().items()
    }
    return DemMassSplit(
        num_mechanisms=int(probability.size),
        probability_mass=total,
        classes=classes,
        dem_weight_mass_share=float(probability[in_feature].sum()) / total,
    )


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Two classical baselines scored on every shot of one frozen set.

    Parameters
    ----------
    reference : DecoderPointResult
        Rates of the reference decoder.
    compared : DecoderPointResult
        Rates of the decoder compared against it, on the same shots.
    test : McNemarResult
        Paired test on the shots the two decoders disagree on.  Its
        ``gnn_wins`` counts shots only the reference decoded correctly and
        its ``baseline_wins`` shots only the compared decoder did.
    outcome : EvalOutcome
        Verdict of the final analysis.
    include_per_round : bool
        Whether the operation's metric policy admits per-round LER.
    space_time_convention : str
        Space-time accounting convention the operation's rates are stated in.
    """

    reference: DecoderPointResult
    compared: DecoderPointResult
    test: McNemarResult
    outcome: EvalOutcome
    include_per_round: bool
    space_time_convention: str

    @property
    def better(self) -> str | None:
        """Name of the decoder with fewer errors, if the test resolved one."""
        if self.outcome is not EvalOutcome.RESOLVED_DIFFERENT:
            return None
        ahead = (
            self.reference
            if self.test.gnn_wins > self.test.baseline_wins
            else self.compared
        )
        return ahead.decoder_name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "reference": self.reference.decoder_name,
            "compared": self.compared.decoder_name,
            "space_time_convention": self.space_time_convention,
            "decoders": {
                rates.decoder_name: decoder_result_to_dict(
                    rates, include_per_round=self.include_per_round
                )
                for rates in (self.reference, self.compared)
            },
            "mcnemar": {
                "statistic": self.test.statistic,
                "p_value": self.test.p_value,
                "n_discordant": self.test.n_discordant,
                "reference_only_correct": self.test.gnn_wins,
                "compared_only_correct": self.test.baseline_wins,
            },
            "outcome": self.outcome.value,
            "better": self.better,
        }


def compare_baselines(
    eval_set: EvalSet,
    *,
    reference: Decoder,
    compared: Decoder,
    profile: OperationProfile,
) -> BaselineComparison:
    """Score two baselines on every frozen shot and run the final analysis once.

    The harness's check interval is set to the whole set, so its stopping
    sequence has a single step: McNemar at ``FINAL_ALPHA`` behind the
    ``MIN_ERRORS_PER_DECODER`` floor.  No shot is left undecoded, and the
    rates are not conditioned on an interim look.

    Parameters
    ----------
    eval_set : EvalSet
        Frozen evaluation set, at least one shot.
    reference : Decoder
        Decoder the comparison is oriented around.
    compared : Decoder
        Decoder compared against the reference.
    profile : OperationProfile
        Profile of the set's operation: observable names, metric policy and
        space-time convention.

    Returns
    -------
    BaselineComparison
    """
    scored = evaluate_point(
        eval_set,
        {reference.name: reference, compared.name: compared},
        reference_decoder=reference.name,
        check_interval=eval_set.num_shots,
        metric_policy=profile.metric_policy,
        observable_names=profile.observable_names,
    )
    return BaselineComparison(
        reference=scored.decoder_results[reference.name],
        compared=scored.decoder_results[compared.name],
        test=scored.mcnemar_results[compared.name],
        outcome=scored.outcome,
        include_per_round=profile.metric_policy.include_per_round_ler,
        space_time_convention=profile.space_time_convention,
    )


@dataclass(frozen=True, slots=True)
class PointDiagnostics:
    """Diagnostics of one experiment point.

    Parameters
    ----------
    key : ExperimentKey
        Identity of the point.
    circuit_file : str
        Circuit the frozen shots were sampled from.
    circuit_sha256 : str
        Hash of that circuit, as verified against the set's manifest.
    shots_sha256 : str
        Hash of the frozen syndromes and observables that were scored.
    graph : GraphSize
        Graph sizes over the frozen shots.
    observable_flips : ObservableFlipRates
        Observable flip rates over the frozen shots.
    dem : DemMassSplit
        Probability-mass split of the circuit's detector error model.
    baselines : BaselineComparison
        The two baselines on the frozen shots.
    """

    key: ExperimentKey
    circuit_file: str
    circuit_sha256: str
    shots_sha256: str
    graph: GraphSize
    observable_flips: ObservableFlipRates
    dem: DemMassSplit
    baselines: BaselineComparison

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            **asdict(self.key),
            "circuit_file": self.circuit_file,
            "circuit_sha256": self.circuit_sha256,
            "shots_sha256": self.shots_sha256,
            "graph": asdict(self.graph),
            "observable_flips": self.observable_flips.to_dict(),
            "dem": asdict(self.dem),
            "baselines": self.baselines.to_dict(),
        }


def classify_threshold(
    points: Sequence[PointDiagnostics],
) -> list[ThresholdClassification]:
    """Apply the sub-threshold rule to every operation, decoder and error rate.

    Parameters
    ----------
    points : sequence of PointDiagnostics
        Diagnosed points.  Those sharing an operation and an error rate form
        one distance ladder per decoder.

    Returns
    -------
    list of ThresholdClassification
        Sorted by ``(operation, decoder, error_prob)``.

    Raises
    ------
    ValueError
        If two points of one operation share an error rate and a distance.
    """
    ladders: defaultdict[tuple[str, str, float], list[DistanceRate]] = defaultdict(list)
    for point in sorted(points, key=lambda point: point.key.distance):
        for rates in (point.baselines.reference, point.baselines.compared):
            ladder = (point.key.operation, rates.decoder_name, point.key.error_prob)
            ladders[ladder].append(
                DistanceRate(
                    distance=point.key.distance,
                    n_errors=rates.n_errors,
                    n_shots=rates.n_shots,
                )
            )
    return [
        ThresholdClassification(
            operation=operation,
            decoder=decoder,
            error_prob=error_prob,
            rates=tuple(rates),
        )
        for (operation, decoder, error_prob), rates in sorted(ladders.items())
    ]


def _rules() -> dict[str, Any]:
    """State the rules the artifact's numbers were produced under."""
    return {
        "graph": (
            "Counts are exact over every frozen shot. A graph is complete over "
            "the fired detectors, so a shot with N fired detectors has N*(N-1) "
            "directed edges under every registered representation."
        ),
        "observable_flips": (
            "Counts are exact over every frozen shot. An observable flips in a "
            "shot when the sampled circuit flips it, independent of any decoder; "
            "any_observable counts shots with at least one flip and equals the "
            "set manifest's positive_count. Every rate carries its Wilson 95% "
            "interval and decides nothing."
        ),
        "dem": (
            "Mass is the sum of mechanism probabilities in the decomposed DEM. "
            "two_detector and boundary are undecomposed mechanisms flipping two "
            "detectors and one; decomposed is every mechanism listed as two or "
            "more components. dem_weight sees a mechanism when it flips exactly "
            "two detectors in total."
        ),
        "baselines": {
            "test": "McNemar on the per-shot disagreement matrix, two-sided",
            "analysis": "one final analysis on every frozen shot, no interim looks",
            "alpha": FINAL_ALPHA,
            "min_errors_per_decoder": MIN_ERRORS_PER_DECODER,
            "outcomes": [outcome.value for outcome in EvalOutcome],
        },
        "threshold": rule_to_dict(),
    }


def _verdict_summary(
    thresholds: Sequence[ThresholdClassification],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """List the error rates under each verdict, per operation and decoder."""
    summary: dict[str, dict[str, dict[str, list[float]]]] = {}
    for classification in thresholds:
        by_verdict = summary.setdefault(classification.operation, {}).setdefault(
            classification.decoder, {verdict.value: [] for verdict in ThresholdVerdict}
        )
        by_verdict[classification.verdict.value].append(classification.error_prob)
    return summary


def report_to_dict(
    points: Sequence[PointDiagnostics], *, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Assemble the scaling-diagnostics artifact.

    Parameters
    ----------
    points : sequence of PointDiagnostics
        Every diagnosed point.
    provenance : dict
        How the artifact was produced: command and library versions.

    Returns
    -------
    dict
        JSON-compatible.  ``points`` are ordered by experiment key;
        ``threshold`` holds one row per operation, decoder and error rate,
        and ``threshold_summary`` the error rates under each verdict.
    """
    ordered = sorted(
        points,
        key=lambda point: (
            point.key.operation,
            point.key.distance,
            point.key.rounds,
            point.key.error_prob,
        ),
    )
    thresholds = classify_threshold(ordered)
    return {
        "provenance": provenance,
        "rules": _rules(),
        "representations": [
            {
                "version": descriptor.version,
                "node_dim": descriptor.node_dim,
                "edge_dim": descriptor.edge_dim,
            }
            for descriptor in registered_representations()
        ],
        "points": [point.to_dict() for point in ordered],
        "threshold": [classification.to_dict() for classification in thresholds],
        "threshold_summary": _verdict_summary(thresholds),
    }
