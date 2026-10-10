"""Streaming sampling for QEC decoding.

Provides ``WorkerSampler`` for per-worker streaming syndrome generation and
helpers for circuit construction and setting discovery from committed circuit
files.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

from sampling.experiment import ExperimentKey, circuit_key, validate_operation_name
from sampling.graph import CircuitMetadata, MetadataExtractor
from sampling.seeding import stable_seed


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CircuitSetting:
    """A single (d, r, p) setting with its committed circuit file.

    Parameters
    ----------
    circuit_path : Path
        Path to the ``.stim`` circuit file.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    """

    circuit_path: Path
    distance: int
    rounds: int
    error_prob: float

    def __post_init__(self) -> None:
        if self.distance < 1:
            raise ValueError(f"distance must be >= 1, got {self.distance}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if not (0 < self.error_prob < 1):
            raise ValueError(f"error_prob must be in (0, 1), got {self.error_prob}")


@dataclass(frozen=True, slots=True)
class ExperimentPoint(CircuitSetting):
    """A committed circuit resolved to its full experiment identity.

    ``CircuitSetting`` is the operation-agnostic view the sampler consumes: it
    needs a circuit and its physical parameters and nothing else.  This record
    adds the axis the sampler must stay ignorant of - which experiment the
    circuit belongs to - so that discovery can hand the same object to a
    sampler and to a consumer that does care.

    Parameters
    ----------
    circuit_path : Path
        Path to the ``.stim`` circuit file.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    operation : str
        Logical operation the circuit realizes.
    """

    operation: str

    def __post_init__(self) -> None:
        # Named explicitly rather than via ``super()``: ``slots=True`` rebuilds
        # the class, so the zero-argument form resolves against the discarded
        # original and raises.
        CircuitSetting.__post_init__(self)
        validate_operation_name(self.operation)

    @property
    def key(self) -> ExperimentKey:
        """Identity of this point on the experiment axis."""
        return ExperimentKey(
            operation=self.operation,
            distance=self.distance,
            rounds=self.rounds,
            error_prob=self.error_prob,
        )


def settings_from_circuit_dir(
    circuit_dir: Path,
    *,
    operation: str,
    distances: Sequence[int] | None = None,
    error_probs: Sequence[float] | None = None,
) -> list[ExperimentPoint]:
    """Discover one operation's experiment points from committed ``.stim`` files.

    Every ``.stim`` file directly in ``circuit_dir`` is resolved to its
    ``(operation, d, r, p)`` identity by the manifest beside it; a filename
    carries none.  Discovery does not descend into subdirectories, and every
    circuit in the root must belong to ``operation`` - before any filter, so
    a root of another operation cannot reach a sampler under this one's
    contract.

    Parameters
    ----------
    circuit_dir : Path
        Directory containing circuit files.
    operation : str
        Operation every circuit in ``circuit_dir`` must belong to.
    distances : sequence of int, optional
        Include only these code distances.  ``None`` means all.
    error_probs : sequence of float, optional
        Include only these error probabilities.  ``None`` means all.

    Returns
    -------
    list[ExperimentPoint]
        Sorted by ``(distance, rounds, error_prob)``.

    Raises
    ------
    NotADirectoryError
        If ``circuit_dir`` does not exist or is not a directory.
    FileNotFoundError
        If a circuit in the directory carries no manifest.
    ValueError
        If a circuit in the directory belongs to another operation (naming
        each such circuit and its operation), if two circuits resolve to the
        same experiment key, or if no circuit matches the filters.
    """
    circuit_dir = Path(circuit_dir)
    if not circuit_dir.is_dir():
        raise NotADirectoryError(f"Circuit directory does not exist: {circuit_dir}")

    keys = _operation_keys(circuit_dir, operation)

    dist_set = set(distances) if distances is not None else None
    prob_set = {round(p, 12) for p in error_probs} if error_probs is not None else None
    points = [
        ExperimentPoint(
            circuit_path=path,
            distance=key.distance,
            rounds=key.rounds,
            error_prob=key.error_prob,
            operation=key.operation,
        )
        for path, key in keys.items()
        if (dist_set is None or key.distance in dist_set)
        and (prob_set is None or round(key.error_prob, 12) in prob_set)
    ]
    if not points:
        raise ValueError(
            f"No matching circuit files in {circuit_dir} "
            f"(distances={distances}, error_probs={error_probs})"
        )

    points.sort(key=lambda pt: (pt.distance, pt.rounds, pt.error_prob))
    return points


def _operation_keys(circuit_dir: Path, operation: str) -> dict[Path, ExperimentKey]:
    """Key of every circuit in ``circuit_dir``, each one of ``operation``'s."""
    keys = {path: circuit_key(path) for path in sorted(circuit_dir.glob("*.stim"))}

    foreign = [
        f"{path} declares {key.operation!r}"
        for path, key in keys.items()
        if key.operation != operation
    ]
    if foreign:
        raise ValueError(
            f"Circuit root {circuit_dir} is discovered for operation "
            f"{operation!r}, but {len(foreign)} circuit(s) belong to another: "
            f"{'; '.join(foreign)}"
        )

    # Two circuits under one identity would be sampled as two settings and
    # silently double the weight of that point in the training mixture.
    claimed: dict[ExperimentKey, Path] = {}
    for path, key in keys.items():
        if key in claimed:
            raise ValueError(
                f"{path} and {claimed[key]} both resolve to the same experiment "
                f"point ({key}); an identity names exactly one circuit"
            )
        claimed[key] = path
    return keys


class WorkerSampler:
    """Per-worker streaming sampler owning Stim CompiledDetectorSamplers.

    Each DataLoader worker creates one ``WorkerSampler`` with a
    deterministic seed.  The sampler holds compiled samplers for every
    circuit setting and a PCG64 RNG for setting selection.

    Parameters
    ----------
    settings : sequence of CircuitSetting
        Circuit settings to sample from.
    worker_seed : int
        Deterministic per-worker seed (derived from master seed + worker id
        via BLAKE2b ``stable_seed``).
    metadata_extractor : MetadataExtractor
        The operation's extractor, building the metadata its representation's
        builder reads.
    num_observables : int
        Observable count of the contract the shots are labelled under; every
        circuit is extracted against it.
    """

    def __init__(
        self,
        settings: Sequence[CircuitSetting],
        worker_seed: int,
        *,
        metadata_extractor: MetadataExtractor,
        num_observables: int,
    ) -> None:
        if not settings:
            raise ValueError("At least one CircuitSetting required")

        self._rng = np.random.Generator(np.random.PCG64(worker_seed))
        self._n_settings = len(settings)
        self._error_probs: list[float] = []
        self._samplers: list[stim.CompiledDetectorSampler] = []
        self._metadata: list[CircuitMetadata] = []

        for i, s in enumerate(settings):
            circuit = stim.Circuit.from_file(str(s.circuit_path))
            sampler_seed = stable_seed("sampler", f"idx={i}", base=worker_seed)
            compiled = circuit.compile_detector_sampler(seed=sampler_seed)
            meta = metadata_extractor(s.circuit_path, num_observables=num_observables)

            self._samplers.append(compiled)
            self._metadata.append(meta)
            self._error_probs.append(s.error_prob)

    def sample(self) -> tuple[np.ndarray, np.ndarray, CircuitMetadata, float]:
        """Sample one shot from a uniformly chosen setting.

        Returns
        -------
        syndrome : ndarray, shape ``(D,)``, uint8
            Detector syndrome bit-vector.
        observables : ndarray, shape ``(num_obs,)``, uint8
            Observable flip vector.
        metadata : CircuitMetadata
            Circuit metadata for the sampled setting.
        error_prob : float
            Physical error probability of the sampled setting.
        """
        idx = int(self._rng.integers(self._n_settings))
        dets, obs = self._samplers[idx].sample(
            shots=1, separate_observables=True, bit_packed=False
        )
        return (
            dets[0].astype(np.uint8, copy=False),
            obs[0].astype(np.uint8, copy=False),
            self._metadata[idx],
            self._error_probs[idx],
        )
