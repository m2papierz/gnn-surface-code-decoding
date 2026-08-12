"""
Evaluation harness: multi-decoder comparison on frozen eval sets.

Orchestrates paired evaluation of decoders (GNN, MWPM, BP+OSD) on identical
pre-sampled shots with adaptive early stopping via the Haybittle-Peto boundary.

The harness loads frozen eval sets, runs all decoders on the same syndromes in
check-interval increments, and stops per point once McNemar resolves or the
shot budget is exhausted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from decoders import Decoder
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
from sampling.experiment import ExperimentKey, eval_set_operation
from sampling.graph import CircuitMetadata
from sampling.profile import MetricPolicy


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalSet:
    """Frozen evaluation set loaded from disk.

    Parameters
    ----------
    syndromes : ndarray, shape ``(N, D)``, uint8
        Binary syndrome vectors.
    observables : ndarray, shape ``(N, O)``, uint8
        True logical observable flips.
    detector_coords : ndarray, shape ``(D, 3)``, float64
        Detector (x, y, t) coordinates.
    distance : int
        Code distance.
    rounds : int
        Syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    num_shots : int
        Total available shots.
    circuit_file : str
        Relative path to source circuit file.
    manifest : dict
        Full manifest contents.
    phase_ids : ndarray or None, shape ``(D,)``, intp
        Per-detector phase-window index.  ``None`` for spatial-only sets.
    phase_names : tuple of str or None
        Ordered phase-window names.
    patch_ids : ndarray or None, shape ``(D,)``, intp
        Per-detector patch (code-block) index.
    seam_mask : ndarray or None, shape ``(D,)``, bool
        ``True`` for detectors on the merge boundary.
    operation : str
        Operation this set belongs to.  Resolved at construction from the
        manifest when it declares one, otherwise from the circuit the shots
        were sampled from.
    circuit_metadata : CircuitMetadata
        Pre-built metadata for graph construction.  Consumers use this
        instead of extracting their own from the circuit.
    """

    syndromes: NDArray[np.uint8]
    observables: NDArray[np.uint8]
    detector_coords: np.ndarray
    distance: int
    rounds: int
    error_prob: float
    num_shots: int
    circuit_file: str
    manifest: dict
    phase_ids: np.ndarray | None = None
    phase_names: tuple[str, ...] | None = None
    patch_ids: np.ndarray | None = None
    seam_mask: np.ndarray | None = None
    observable_names: tuple[str, ...] | None = None
    dem_edge_weights: np.ndarray | None = None
    operation: str = field(init=False)
    circuit_metadata: CircuitMetadata = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "operation", eval_set_operation(self.manifest, self.circuit_file)
        )
        n_det = self.syndromes.shape[1]
        dem_w = (
            self.dem_edge_weights
            if self.dem_edge_weights is not None
            else np.zeros((n_det, n_det), dtype=np.float64)
        )
        object.__setattr__(
            self,
            "circuit_metadata",
            CircuitMetadata(
                detector_coords=self.detector_coords,
                distance=self.distance,
                rounds=self.rounds,
                num_detectors=n_det,
                dem_edge_weights=dem_w,
                phase_ids=self.phase_ids,
                phase_names=self.phase_names,
                patch_ids=self.patch_ids,
                seam_mask=self.seam_mask,
            ),
        )


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
    metric_policy : MetricPolicy or None
        Metric policy from the operation's profile.  Controls which metrics
        the serializer emits.  ``None`` for backward compatibility with
        callers that predate the profile registry.
    observable_names : tuple of str or None
        Ordered observable names from the profile.  ``None`` when not
        available (legacy eval sets without profile metadata).
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
    metric_policy: MetricPolicy | None = None
    observable_names: tuple[str, ...] | None = None
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


@dataclass(slots=True)
class EvalReport:
    """Full evaluation report across all (d, p) points.

    Parameters
    ----------
    results : list[EvalPointResult]
        Per-point results.
    metadata : dict
        Report-level metadata (decoder names, config, etc).
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
        """Write report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_eval_set(eval_dir: Path) -> EvalSet:
    """Load a frozen evaluation set from a directory.

    Supports two formats:
    - Compressed npz (``data.npz`` with keys: syndromes, observables,
      detector_coords, and optionally phase_ids/patch_ids/seam_mask for
      phased representations) + ``manifest.json``
    - Individual npy files (``syndromes.npy``, ``observables.npy``,
      ``detector_coords.npy``) + ``manifest.json``

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
        If required files are missing, or if the circuit the set names cannot
        be reached to resolve its operation.
    ValueError
        If manifest is missing required fields, shapes mismatch, the set
        declares an operation its circuit does not belong to, or the manifest
        declares a representation whose metadata is not present.
    """
    manifest_path = eval_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in eval set directory: {eval_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    phase_ids: np.ndarray | None = None
    phase_names: tuple[str, ...] | None = None
    patch_ids: np.ndarray | None = None
    seam_mask: np.ndarray | None = None

    npz_path = eval_dir / "data.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        syndromes = data["syndromes"].astype(np.uint8, copy=False)
        observables = data["observables"].astype(np.uint8, copy=False)
        detector_coords = data["detector_coords"].astype(np.float64, copy=False)

        if "phase_ids" in data:
            phase_ids = data["phase_ids"].astype(np.intp, copy=False)
            patch_ids = data["patch_ids"].astype(np.intp, copy=False)
            seam_mask = data["seam_mask"].astype(np.bool_, copy=False)
            raw_names = manifest.get("phase_names")
            if raw_names is None:
                raise ValueError(
                    f"Eval set {eval_dir} has phase_ids in data but manifest "
                    f"lacks phase_names"
                )
            phase_names = tuple(raw_names)
    else:
        syn_path = eval_dir / "syndromes.npy"
        obs_path = eval_dir / "observables.npy"
        coords_path = eval_dir / "detector_coords.npy"

        if not syn_path.exists():
            raise FileNotFoundError(f"No syndromes file in {eval_dir}")

        syndromes = np.load(syn_path).astype(np.uint8, copy=False)
        observables = np.load(obs_path).astype(np.uint8, copy=False)
        detector_coords = np.load(coords_path).astype(np.float64, copy=False)

    if observables.ndim == 1:
        observables = observables[:, np.newaxis]

    _required_fields = ["distance", "rounds", "error_prob", "circuit_file"]
    for f in _required_fields:
        if f not in manifest:
            raise ValueError(f"Manifest missing required field '{f}': {manifest_path}")

    # Validate: if the manifest declares a representation that needs phased
    # metadata, the stored data must provide it.
    rep_version = manifest.get("representation_version", "spatial")
    if rep_version != "spatial" and phase_ids is None:
        raise ValueError(
            f"Eval set {eval_dir} declares representation {rep_version!r} but "
            f"lacks the phased metadata arrays (phase_ids, patch_ids, seam_mask) "
            f"that representation requires"
        )

    raw_obs_names = manifest.get("observable_names")
    obs_names: tuple[str, ...] | None = (
        tuple(raw_obs_names) if raw_obs_names is not None else None
    )

    import stim

    from sampling.graph import extract_circuit_metadata

    dem_weights: np.ndarray | None = None
    circuit_path = Path(manifest["circuit_file"])
    if circuit_path.exists():
        circuit = stim.Circuit.from_file(str(circuit_path))
        if circuit.num_detectors == syndromes.shape[1]:
            base_meta = extract_circuit_metadata(
                circuit, manifest["distance"], manifest["rounds"]
            )
            dem_weights = base_meta.dem_edge_weights

    eval_set = EvalSet(
        syndromes=syndromes,
        observables=observables,
        detector_coords=detector_coords,
        distance=manifest["distance"],
        rounds=manifest["rounds"],
        error_prob=manifest["error_prob"],
        num_shots=syndromes.shape[0],
        circuit_file=manifest["circuit_file"],
        manifest=manifest,
        phase_ids=phase_ids,
        phase_names=phase_names,
        patch_ids=patch_ids,
        seam_mask=seam_mask,
        observable_names=obs_names,
        dem_edge_weights=dem_weights,
    )
    logger.debug(
        "Loaded eval set for operation %s from %s", eval_set.operation, eval_dir
    )
    return eval_set


def evaluate_point(
    eval_set: EvalSet,
    decoders: dict[str, Decoder],
    reference_decoder: str,
    *,
    stopping_baseline: str | None = None,
    check_interval: int = CHECK_INTERVAL,
    metric_policy: MetricPolicy | None = None,
    observable_names: tuple[str, ...] | None = None,
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
    observable_names : tuple of str or None
        Ordered observable names for per-observable breakdown labels.
        Falls back to ``eval_set.observable_names`` when not given
        explicitly.
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

    resolved_obs_names = observable_names or eval_set.observable_names

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
        for obs_idx in range(n_obs):
            obs_col = obs_slice[:, obs_idx]
            obs_n_err = int(np.sum(~obs_col))
            obs_ler_val = obs_n_err / shots_processed if shots_processed > 0 else 0.0
            obs_ci = wilson_interval(obs_n_err, shots_processed)
            obs_name = (
                resolved_obs_names[obs_idx]
                if resolved_obs_names is not None and obs_idx < len(resolved_obs_names)
                else f"observable_{obs_idx}"
            )
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
        key=ExperimentKey(
            operation=eval_set.operation,
            distance=eval_set.distance,
            rounds=eval_set.rounds,
            error_prob=eval_set.error_prob,
        ),
        n_shots_used=shots_processed,
        decoder_results=decoder_results,
        mcnemar_results=mcnemar_results,
        stopping=stopping,
        outcome=outcome,
        metric_policy=metric_policy,
        observable_names=resolved_obs_names,
        space_time_convention=space_time_convention,
    )


def discover_eval_sets(
    eval_dir: Path,
    *,
    distances: list[int] | None = None,
    error_probs: list[float] | None = None,
) -> list[Path]:
    """Discover eval set directories matching the (d, p) naming convention.

    ``eval_dir`` is one operation's evaluation root.  Discovery does not
    descend into another operation's root, so a sweep cannot span two
    experiments because a filter was forgotten at the call site.

    Parameters
    ----------
    eval_dir : Path
        One operation's eval root, containing ``d{d}_p{p_str}/`` subdirs.
    distances : list[int] or None
        Filter to these distances. None = all.
    error_probs : list[float] or None
        Filter to these error probabilities. None = all.

    Returns
    -------
    list[Path]
        Sorted list of eval set directories.
    """
    dirs: list[Path] = []
    if not eval_dir.is_dir():
        return dirs

    for sub in sorted(eval_dir.iterdir()):
        if not sub.is_dir():
            continue
        manifest_path = sub / "manifest.json"
        if not manifest_path.exists():
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        d = manifest.get("distance")
        p = manifest.get("error_prob")

        if distances is not None and d not in distances:
            continue
        if error_probs is not None and p not in error_probs:
            continue

        dirs.append(sub)

    return dirs


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

    decoders = {}
    for name, dr in result.decoder_results.items():
        entry: dict = {
            "n_shots": dr.n_shots,
            "n_errors": dr.n_errors,
            "ler": dr.ler,
            "ler_ci_95": [dr.ler_interval.lower, dr.ler_interval.upper],
        }
        if include_per_round:
            entry["per_round_ler"] = dr.per_round_ler
            entry["per_round_ler_ci_95"] = [
                dr.per_round_interval.lower,
                dr.per_round_interval.upper,
            ]
        if dr.per_observable:
            entry["per_observable"] = [
                {
                    "name": obs.name,
                    "n_shots": obs.n_shots,
                    "n_errors": obs.n_errors,
                    "ler": obs.ler,
                    "ler_ci_95": [obs.ler_interval.lower, obs.ler_interval.upper],
                }
                for obs in dr.per_observable
            ]
        decoders[name] = entry

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
    if result.observable_names is not None:
        out["observable_names"] = list(result.observable_names)

    return out
