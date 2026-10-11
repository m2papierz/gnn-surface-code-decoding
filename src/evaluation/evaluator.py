"""
Evaluation harness: multi-decoder comparison on frozen eval sets.

Orchestrates paired evaluation of decoders (GNN, MWPM, BP+OSD) on identical
pre-sampled shots with adaptive early stopping via the Haybittle-Peto boundary.

The harness loads frozen eval sets, runs all decoders on the same syndromes in
check-interval increments, and stops per point once McNemar resolves or the
shot budget is exhausted.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray

from artifact_file import write_json
from decoders import Decoder, GNNCheckpoint
from evaluation.config import EvalConfig
from evaluation.stats import (
    CHECK_INTERVAL,
    EvalOutcome,
    McNemarResult,
    StoppingDecision,
    WilsonInterval,
    adaptive_stop,
    mcnemar_test,
    per_round_ler,
    wilson_interval,
)
from sampling.experiment import ExperimentKey, eval_set_key, read_manifest
from sampling.graph import CircuitMetadata
from sampling.profile import MetricPolicy, resolve_profile


logger = logging.getLogger(__name__)


_MANIFEST_NAME: Final[str] = "manifest.json"


@dataclass(frozen=True, slots=True)
class EvalSet:
    """Frozen evaluation shots, bound to the committed circuit they came from.

    Parameters
    ----------
    key : ExperimentKey
        Identity of the point: the key of the set's circuit.
    syndromes : ndarray, shape ``(N, D)``, uint8
        Detection events of each shot, ``N >= 1``.
    observables : ndarray, shape ``(N, K)``, uint8
        True logical observable flips of each shot.
    observable_names : tuple of str, length ``K``
        The operation's ordered observable names.
    circuit_file : Path
        Circuit the shots were sampled from.
    circuit_sha256 : str
        Digest of the circuit's bytes, verified at load.
    circuit_metadata : CircuitMetadata
        Metadata the operation's extractor read from the circuit; every
        consumer builds graphs from it.
    manifest : mapping
        The set's manifest as written by its generator.

    Raises
    ------
    ValueError
        If the shot arrays are not 2-D uint8 over the same ``N >= 1`` shots,
        or disagree with the metadata's detector count, the observable names
        or the key's distance and rounds.
    """

    key: ExperimentKey
    syndromes: NDArray[np.uint8]
    observables: NDArray[np.uint8]
    observable_names: tuple[str, ...]
    circuit_file: Path
    circuit_sha256: str
    circuit_metadata: CircuitMetadata
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name, shots in (
            ("syndromes", self.syndromes),
            ("observables", self.observables),
        ):
            if shots.ndim != 2 or shots.dtype != np.uint8:
                raise ValueError(
                    f"{name} must be a 2-D uint8 array, got {shots.ndim}-D "
                    f"{shots.dtype}"
                )
        num_shots, num_detectors = self.syndromes.shape
        if num_shots == 0 or self.observables.shape[0] != num_shots:
            raise ValueError(
                f"syndromes hold {num_shots} shot(s) and observables "
                f"{self.observables.shape[0]}; a frozen set holds the same N >= 1"
            )
        if num_detectors != self.circuit_metadata.num_detectors:
            raise ValueError(
                f"shots carry {num_detectors} detectors, but the circuit has "
                f"{self.circuit_metadata.num_detectors}"
            )
        if self.observables.shape[1] != len(self.observable_names):
            raise ValueError(
                f"shots carry {self.observables.shape[1]} observable(s), but "
                f"{self.key.operation} names {list(self.observable_names)}"
            )
        metadata_dr = (self.circuit_metadata.distance, self.circuit_metadata.rounds)
        if metadata_dr != (self.key.distance, self.key.rounds):
            raise ValueError(
                f"circuit metadata is at (d, r) = {metadata_dr}, but the set is "
                f"{self.key}"
            )

    @property
    def operation(self) -> str:
        """Logical operation the set's circuit realizes."""
        return self.key.operation

    @property
    def distance(self) -> int:
        """Code distance."""
        return self.key.distance

    @property
    def rounds(self) -> int:
        """Syndrome measurement rounds."""
        return self.key.rounds

    @property
    def error_prob(self) -> float:
        """Physical error probability."""
        return self.key.error_prob

    @property
    def num_shots(self) -> int:
        """Shots the set holds."""
        return self.syndromes.shape[0]


@dataclass(frozen=True, slots=True)
class ObservableResult:
    """Per-observable error statistics for one decoder at one point.

    Parameters
    ----------
    name : str
        Observable name (from the profile's ordered list).
    n_shots : int
        Number of shots scored.
    n_errors : int
        Number of shots where this observable was decoded incorrectly.
    ler : float
        Per-shot error rate for this observable.
    ler_interval : WilsonInterval
        Wilson 95% CI for this observable's error rate.
    """

    name: str
    n_shots: int
    n_errors: int
    ler: float
    ler_interval: WilsonInterval


@dataclass(frozen=True, slots=True)
class DecoderPointResult:
    """Per-decoder results for one (d, p) evaluation point.

    Parameters
    ----------
    decoder_name : str
        Decoder identifier.
    n_shots : int
        Number of shots processed.
    n_errors : int
        Number of logical errors (shot-level: any observable wrong).
    ler : float
        Per-shot logical error rate.
    ler_interval : WilsonInterval
        Wilson 95% CI for per-shot LER.
    per_round_ler : float
        Per-round logical error rate (epsilon).
    per_round_interval : WilsonInterval
        Wilson 95% CI for per-round LER.
    correct : NDArray[np.bool_]
        Per-shot correctness vector (True = correct decode).
    per_observable : tuple[ObservableResult, ...]
        Per-observable breakdown.
    """

    decoder_name: str
    n_shots: int
    n_errors: int
    ler: float
    ler_interval: WilsonInterval
    per_round_ler: float
    per_round_interval: WilsonInterval
    correct: NDArray[np.bool_]
    per_observable: tuple[ObservableResult, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalPointResult:
    """Complete evaluation result for one experiment point.

    Parameters
    ----------
    key : ExperimentKey
        Identity of the point - ``(operation, distance, rounds, error_prob)``.
        Carried so a serialized row cannot be read as belonging to another
        experiment.
    n_shots_used : int
        Shots actually processed (may be less than available due to early stop).
    decoder_results : dict[str, DecoderPointResult]
        Per-decoder results keyed by decoder name.
    mcnemar_results : dict[str, McNemarResult]
        McNemar test results for GNN vs each baseline, keyed by baseline name.
    stopping : StoppingDecision
        Final stopping decision.
    outcome : EvalOutcome
        Final evaluation outcome for this point.
    observable_names : tuple of str
        Ordered observable names of the point's operation.
    metric_policy : MetricPolicy or None
        Metric policy from the operation's profile.  Controls which metrics
        the serializer emits.  ``None`` for backward compatibility with
        callers that predate the profile registry.
    space_time_convention : str or None
        Label for the space-time accounting convention.  Required on every
        result artifact so a table cannot exist without stating which
        convention produced it.
    """

    key: ExperimentKey
    n_shots_used: int
    decoder_results: dict[str, DecoderPointResult]
    mcnemar_results: dict[str, McNemarResult]
    stopping: StoppingDecision
    outcome: EvalOutcome
    observable_names: tuple[str, ...]
    metric_policy: MetricPolicy | None = None
    space_time_convention: str | None = None

    @property
    def operation(self) -> str:
        """Logical operation this point scores."""
        return self.key.operation

    @property
    def distance(self) -> int:
        """Code distance."""
        return self.key.distance

    @property
    def rounds(self) -> int:
        """Syndrome measurement rounds."""
        return self.key.rounds

    @property
    def error_prob(self) -> float:
        """Physical error probability."""
        return self.key.error_prob


class EvalMode(StrEnum):
    """How a report's numbers were produced."""

    HARNESS = "harness"
    SANITY = "sanity"
    DRY_RUN = "dry-run"


def report_metadata(
    mode: EvalMode,
    cfg: EvalConfig,
    *,
    source: Path,
    checkpoint: GNNCheckpoint | None,
    device: torch.device,
) -> dict[str, Any]:
    """What produced a report: mode, shots, resolved config, checkpoint, device.

    Parameters
    ----------
    mode : EvalMode
        The evaluation mode.
    cfg : EvalConfig
        The resolved config, every CLI override applied.
    source : Path
        Where the scored shots came from: the frozen eval sets' root, the
        circuit root fresh shots were sampled from, or the CI shard.
    checkpoint : GNNCheckpoint or None
        The scored checkpoint; ``None`` for a dry run's untrained model.
    device : torch.device
        Device the GNN ran on.

    Returns
    -------
    dict
        JSON-ready metadata; the checkpoint is named by path and by the
        sha256 of the bytes that were loaded.
    """
    return {
        "mode": mode.value,
        "source": str(source),
        "config": cfg.to_dict(),
        "checkpoint": None if checkpoint is None else checkpoint.provenance(),
        "device": str(device),
    }


@dataclass(slots=True)
class EvalReport:
    """Full evaluation report across all (d, p) points.

    Parameters
    ----------
    results : list[EvalPointResult]
        Per-point results.
    metadata : dict
        What produced the report (:func:`report_metadata`).
    """

    results: list[EvalPointResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "metadata": self.metadata,
            "points": [_point_to_dict(r) for r in self.results],
        }

    def save(self, path: Path) -> None:
        """Write report as JSON, atomically."""
        write_json(path, self.to_dict())


def load_eval_set(eval_dir: Path) -> EvalSet:
    """Load a frozen evaluation set, bound to the circuit it was sampled from.

    The directory holds ``manifest.json`` and the shots: ``syndromes`` and
    ``observables`` in ``data.npz``, or as ``syndromes.npy`` and
    ``observables.npy``.  Everything else comes from the set's circuit, which
    must hash to the digest the manifest recorded: the key it declares, and
    the metadata its operation's extractor reads from it.  Any other arrays a
    set stored at generation are not read.

    Parameters
    ----------
    eval_dir : Path
        Directory containing the eval set.

    Returns
    -------
    EvalSet

    Raises
    ------
    FileNotFoundError
        If the manifest, the shots, the circuit or the circuit's manifest is
        missing.
    ValueError
        If the set is not bound to its circuit (see
        ``sampling.experiment.eval_set_key``), its operation is not
        registered, its circuit is refused by the operation's extractor, or
        its shots disagree with the circuit; the message names the set.
    """
    manifest_path = eval_dir / _MANIFEST_NAME
    try:
        manifest = read_manifest(manifest_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Eval set {eval_dir} has no {_MANIFEST_NAME}; a frozen set names "
            f"the circuit it was sampled from there"
        ) from exc
    key = eval_set_key(manifest, manifest_path)
    circuit_file = Path(manifest["circuit_file"])
    profile = resolve_profile(key.operation)
    syndromes, observables = _read_shots(eval_dir)
    metadata = profile.metadata_extractor(
        circuit_file, num_observables=profile.data_contract.num_observables
    )
    try:
        eval_set = EvalSet(
            key=key,
            syndromes=syndromes,
            observables=observables,
            observable_names=profile.observable_names,
            circuit_file=circuit_file,
            circuit_sha256=manifest["circuit_sha256"],
            circuit_metadata=metadata,
            manifest=manifest,
        )
    except ValueError as exc:
        raise ValueError(
            f"Eval set {eval_dir}, sampled from {circuit_file}: {exc}"
        ) from exc
    logger.debug("Loaded eval set %s from %s", key, eval_dir)
    return eval_set


def _read_shots(eval_dir: Path) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Read ``(syndromes, observables)`` from ``data.npz`` or the two ``.npy``."""
    npz_path = eval_dir / "data.npz"
    if npz_path.is_file():
        with np.load(npz_path) as archive:
            return archive["syndromes"], archive["observables"]
    try:
        return (
            np.load(eval_dir / "syndromes.npy"),
            np.load(eval_dir / "observables.npy"),
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Eval set {eval_dir} holds neither data.npz nor syndromes.npy and "
            f"observables.npy"
        ) from exc


def evaluate_point(
    eval_set: EvalSet,
    decoders: dict[str, Decoder],
    reference_decoder: str,
    *,
    stopping_baseline: str | None = None,
    check_interval: int = CHECK_INTERVAL,
    metric_policy: MetricPolicy | None = None,
    space_time_convention: str | None = None,
) -> EvalPointResult:
    """Evaluate all decoders on a single experiment point with adaptive stopping.

    Processes shots in increments of ``check_interval``. At each checkpoint,
    evaluates the Haybittle-Peto stopping criterion on the reference decoder
    vs ``stopping_baseline`` disagreement matrix only. McNemar results are
    computed for all comparison pairs at the end regardless of which pair
    drove the stopping decision.

    Parameters
    ----------
    eval_set : EvalSet
        Frozen evaluation set to decode.
    decoders : dict[str, Decoder]
        Decoders to evaluate, keyed by name.
    reference_decoder : str
        Name of the reference decoder (GNN in typical usage).
        Must be a key in ``decoders``.
    stopping_baseline : str or None
        Comparison decoder whose McNemar test drives adaptive stopping.
        If None, defaults to the first non-reference decoder. Only this
        pair is checked for the Haybittle-Peto boundary; other decoders
        are evaluated on the same shots but do not influence stopping.
    check_interval : int
        Shots between adaptive stopping checks.
    metric_policy : MetricPolicy or None
        Metric policy from the operation's resolved profile.  Carried on
        the result so the serializer can gate which metrics to emit.
        ``None`` preserves backward compatibility with callers that
        predate the profile registry.
    space_time_convention : str or None
        Space-time accounting convention label from the profile.

    Returns
    -------
    EvalPointResult

    Raises
    ------
    ValueError
        If reference_decoder not in decoders, fewer than 2 decoders, or
        stopping_baseline not in decoders.
    """
    if reference_decoder not in decoders:
        raise ValueError(
            f"reference_decoder '{reference_decoder}' not in decoders: "
            f"{list(decoders.keys())}"
        )
    if len(decoders) < 2:
        raise ValueError("Need at least 2 decoders for paired comparison")

    # Resolve which comparison decoder drives stopping
    comparison_decoders = [k for k in decoders if k != reference_decoder]
    if stopping_baseline is None:
        stopping_baseline = comparison_decoders[0]
    elif stopping_baseline not in decoders:
        raise ValueError(
            f"stopping_baseline '{stopping_baseline}' not in decoders: "
            f"{list(decoders.keys())}"
        )
    elif stopping_baseline == reference_decoder:
        raise ValueError(
            f"stopping_baseline must differ from reference_decoder "
            f"(both are '{reference_decoder}')"
        )

    n_total = eval_set.num_shots
    n_obs = eval_set.observables.shape[1]

    # Preallocate correctness arrays
    correct_arrays: dict[str, NDArray[np.bool_]] = {
        name: np.empty(n_total, dtype=np.bool_) for name in decoders
    }
    # Per-observable correctness for breakdown
    obs_correct_arrays: dict[str, NDArray[np.bool_]] = {
        name: np.empty((n_total, n_obs), dtype=np.bool_) for name in decoders
    }

    # Determine check points (ceiling division to include tail shots)
    n_checks = max(1, -(-n_total // check_interval))
    shots_processed = 0
    stopping: StoppingDecision | None = None

    for check_idx in range(n_checks):
        start = shots_processed
        end = min(start + check_interval, n_total)
        chunk_syndromes = eval_set.syndromes[start:end]
        chunk_observables = eval_set.observables[start:end]

        for name, decoder in decoders.items():
            predictions = decoder.decode_batch(chunk_syndromes)
            if predictions.ndim == 1:
                predictions = predictions[:, np.newaxis]
            predictions = predictions[:, :n_obs]

            per_obs_correct = predictions == chunk_observables
            obs_correct_arrays[name][start:end] = per_obs_correct
            shot_correct = np.all(per_obs_correct, axis=1)
            correct_arrays[name][start:end] = shot_correct

        shots_processed = end
        is_final = end >= n_total

        # Adaptive stopping checks only the primary comparison pair
        stopping = adaptive_stop(
            correct_arrays[reference_decoder][:shots_processed],
            correct_arrays[stopping_baseline][:shots_processed],
            is_final=is_final,
        )
        if stopping.action == "stop":
            break

    # If we ran all shots without a stopping decision
    if stopping is None or (
        stopping.action == "continue" and shots_processed >= n_total
    ):
        stopping = adaptive_stop(
            correct_arrays[reference_decoder][:shots_processed],
            correct_arrays[stopping_baseline][:shots_processed],
            is_final=True,
        )

    # Build per-decoder results
    decoder_results: dict[str, DecoderPointResult] = {}
    for name in decoders:
        correct_slice = correct_arrays[name][:shots_processed]
        n_errors = int(np.sum(~correct_slice))
        ler = n_errors / shots_processed if shots_processed > 0 else 0.0
        ler_ci = wilson_interval(n_errors, shots_processed)

        eps = per_round_ler(ler, eval_set.rounds)
        eps_lower = per_round_ler(ler_ci.lower, eval_set.rounds)
        eps_upper = per_round_ler(ler_ci.upper, eval_set.rounds)
        eps_interval = WilsonInterval(
            lower=eps_lower,
            upper=eps_upper,
            point=eps,
            n_errors=n_errors,
            n_total=shots_processed,
            alpha=0.05,
        )

        # Per-observable breakdown
        obs_slice = obs_correct_arrays[name][:shots_processed]
        per_obs: list[ObservableResult] = []
        for obs_idx, obs_name in enumerate(eval_set.observable_names):
            obs_col = obs_slice[:, obs_idx]
            obs_n_err = int(np.sum(~obs_col))
            obs_ler_val = obs_n_err / shots_processed if shots_processed > 0 else 0.0
            obs_ci = wilson_interval(obs_n_err, shots_processed)
            per_obs.append(
                ObservableResult(
                    name=obs_name,
                    n_shots=shots_processed,
                    n_errors=obs_n_err,
                    ler=obs_ler_val,
                    ler_interval=obs_ci,
                )
            )

        decoder_results[name] = DecoderPointResult(
            decoder_name=name,
            n_shots=shots_processed,
            n_errors=n_errors,
            ler=ler,
            ler_interval=ler_ci,
            per_round_ler=eps,
            per_round_interval=eps_interval,
            correct=correct_slice,
            per_observable=tuple(per_obs),
        )

    # McNemar results for reference decoder vs each other decoder
    mcnemar_results: dict[str, McNemarResult] = {}
    ref_correct = correct_arrays[reference_decoder][:shots_processed]
    for comp_name in comparison_decoders:
        comp_correct = correct_arrays[comp_name][:shots_processed]
        mcnemar_results[comp_name] = mcnemar_test(ref_correct, comp_correct)

    outcome = (
        stopping.outcome if stopping.outcome is not None else EvalOutcome.UNRESOLVED
    )

    return EvalPointResult(
        key=eval_set.key,
        n_shots_used=shots_processed,
        decoder_results=decoder_results,
        mcnemar_results=mcnemar_results,
        stopping=stopping,
        outcome=outcome,
        metric_policy=metric_policy,
        observable_names=eval_set.observable_names,
        space_time_convention=space_time_convention,
    )


def discover_eval_sets(
    eval_dir: Path,
    *,
    distances: list[int] | None = None,
    error_probs: list[float] | None = None,
) -> list[Path]:
    """Discover the frozen sets under one operation's evaluation root.

    ``eval_dir`` is one operation's evaluation root.  Discovery does not
    descend into another operation's root, so a sweep cannot span two
    experiments because a filter was forgotten at the call site.  Each set is
    filtered on its key as ``sampling.experiment.eval_set_key`` resolves it,
    the same rule the loader applies.

    Parameters
    ----------
    eval_dir : Path
        One operation's eval root, holding one directory per set.
    distances : list[int] or None
        Filter to these distances. None = all.
    error_probs : list[float] or None
        Filter to these error probabilities. None = all.

    Returns
    -------
    list[Path]
        Sorted list of eval set directories.

    Raises
    ------
    FileNotFoundError, ValueError
        If a set under the root is not bound to its circuit.
    """
    if not eval_dir.is_dir():
        return []
    dirs: list[Path] = []
    for manifest_path in sorted(eval_dir.glob(f"*/{_MANIFEST_NAME}")):
        key = eval_set_key(read_manifest(manifest_path), manifest_path)
        if distances is not None and key.distance not in distances:
            continue
        if error_probs is not None and key.error_prob not in error_probs:
            continue
        dirs.append(manifest_path.parent)
    return dirs


def decoder_result_to_dict(
    rates: DecoderPointResult, *, include_per_round: bool
) -> dict:
    """Serialize one decoder's rates at one point to a JSON-compatible dict.

    Every rate is written with its Wilson 95% interval and the counts behind
    it.

    Parameters
    ----------
    rates : DecoderPointResult
        The decoder's results at one evaluation point.
    include_per_round : bool
        Whether the operation's metric policy admits per-round LER.

    Returns
    -------
    dict
    """
    entry: dict = {
        "n_shots": rates.n_shots,
        "n_errors": rates.n_errors,
        "ler": rates.ler,
        "ler_ci_95": [rates.ler_interval.lower, rates.ler_interval.upper],
    }
    if include_per_round:
        entry["per_round_ler"] = rates.per_round_ler
        entry["per_round_ler_ci_95"] = [
            rates.per_round_interval.lower,
            rates.per_round_interval.upper,
        ]
    if rates.per_observable:
        entry["per_observable"] = [
            {
                "name": obs.name,
                "n_shots": obs.n_shots,
                "n_errors": obs.n_errors,
                "ler": obs.ler,
                "ler_ci_95": [obs.ler_interval.lower, obs.ler_interval.upper],
            }
            for obs in rates.per_observable
        ]
    return entry


def _point_to_dict(result: EvalPointResult) -> dict:
    """Serialize an EvalPointResult to a JSON-compatible dict.

    The serializer emits the metrics the metric policy declares and nothing
    else: per-round LER is included only when the policy's
    ``include_per_round_ler`` is ``True`` (or when no policy is set, for
    backward compatibility).
    """
    include_per_round = (
        result.metric_policy is None or result.metric_policy.include_per_round_ler
    )

    decoders = {
        name: decoder_result_to_dict(dr, include_per_round=include_per_round)
        for name, dr in result.decoder_results.items()
    }

    mcnemar = {}
    for name, mr in result.mcnemar_results.items():
        mcnemar[name] = {
            "statistic": mr.statistic,
            "p_value": mr.p_value,
            "n_discordant": mr.n_discordant,
            "gnn_wins": mr.gnn_wins,
            "baseline_wins": mr.baseline_wins,
        }

    out: dict = {
        "operation": result.operation,
        "distance": result.distance,
        "rounds": result.rounds,
        "error_prob": result.error_prob,
        "n_shots_used": result.n_shots_used,
        "outcome": result.outcome.value,
        "stopping_reason": result.stopping.reason,
        "decoders": decoders,
        "mcnemar": mcnemar,
    }
    if result.space_time_convention is not None:
        out["space_time_convention"] = result.space_time_convention
    out["observable_names"] = list(result.observable_names)

    return out
