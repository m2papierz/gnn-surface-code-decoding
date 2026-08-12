"""Streaming sampling for QEC decoding.

Provides ``WorkerSampler`` for per-worker streaming syndrome generation and
helpers for circuit construction and setting discovery from committed circuit
files.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

from sampling.experiment import ExperimentKey, circuit_key, validate_operation_name
from sampling.graph import CircuitMetadata, extract_circuit_metadata
from sampling.seeding import stable_seed


logger = logging.getLogger(__name__)


_CIRCUIT_FILENAME_RE = re.compile(r"d(\d+)_r(\d+)_p(.+)\.stim$")


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
    distances: Sequence[int] | None = None,
    error_probs: Sequence[float] | None = None,
) -> list[ExperimentPoint]:
    """Discover experiment points from committed ``.stim`` files.

    Every ``.stim`` file directly in ``circuit_dir`` is resolved to its
    ``(operation, d, r, p)`` identity.  The manifest beside a circuit is
    authoritative; the ``d{d}_r{r}_p{p}.stim`` filename is read only for a
    circuit that carries none.  Discovery does not descend into
    subdirectories, so a root holding one operation's circuits is discovered
    by pointing this function at that root.

    Parameters
    ----------
    circuit_dir : Path
        Directory containing circuit files.
    distances : sequence of int, optional
        Include only these code distances.  ``None`` means all.
    error_probs : sequence of float, optional
        Include only these error probabilities.  ``None`` means all.

    Returns
    -------
    list[ExperimentPoint]
        Sorted by ``(operation, distance, rounds, error_prob)``.

    Raises
    ------
    NotADirectoryError
        If ``circuit_dir`` does not exist or is not a directory.
    ValueError
        If no matching circuit files are found, or if two circuits in the
        directory resolve to the same experiment key.
    """
    circuit_dir = Path(circuit_dir)
    if not circuit_dir.is_dir():
        raise NotADirectoryError(f"Circuit directory does not exist: {circuit_dir}")

    dist_set = set(distances) if distances is not None else None
    prob_set = {round(p, 12) for p in error_probs} if error_probs is not None else None

    points: list[ExperimentPoint] = []
    claimed: dict[ExperimentKey, Path] = {}
    for f in sorted(circuit_dir.iterdir()):
        m = _CIRCUIT_FILENAME_RE.match(f.name)
        if m is None:
            continue

        key = circuit_key(
            f,
            distance=int(m.group(1)),
            rounds=int(m.group(2)),
            error_prob=float(m.group(3).replace("_", ".")),
        )

        # Two circuits under one identity would be sampled as two settings and
        # silently double the weight of that point in the training mixture.
        if key in claimed:
            raise ValueError(
                f"{f} and {claimed[key]} both resolve to the same experiment "
                f"point ({key}); an identity names exactly one circuit"
            )
        claimed[key] = f

        if dist_set is not None and key.distance not in dist_set:
            continue
        if prob_set is not None and round(key.error_prob, 12) not in prob_set:
            continue

        points.append(
            ExperimentPoint(
                circuit_path=f,
                distance=key.distance,
                rounds=key.rounds,
                error_prob=key.error_prob,
                operation=key.operation,
            )
        )

    if not points:
        raise ValueError(
            f"No matching circuit files in {circuit_dir} "
            f"(distances={distances}, error_probs={error_probs})"
        )

    points.sort(key=lambda pt: (pt.operation, pt.distance, pt.rounds, pt.error_prob))
    return points


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
    weights : ndarray, shape ``(len(settings),)``, optional
        Per-setting sampling probabilities.  If ``None``, settings are
        sampled uniformly.  Weights are normalized internally.
    """

    def __init__(
        self,
        settings: Sequence[CircuitSetting],
        worker_seed: int,
        weights: np.ndarray | None = None,
        metadata_extractor: Callable[[stim.Circuit, int, int], CircuitMetadata]
        | None = None,
    ) -> None:
        if not settings:
            raise ValueError("At least one CircuitSetting required")

        self._rng = np.random.Generator(np.random.PCG64(worker_seed))
        self._n_settings = len(settings)
        self._error_probs: list[float] = []
        self._samplers: list[stim.CompiledDetectorSampler] = []
        self._metadata: list[CircuitMetadata] = []

        if weights is not None:
            if len(weights) != len(settings):
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"settings length ({len(settings)})"
                )
            total = weights.sum()
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            self._weights: np.ndarray | None = weights / total
        else:
            self._weights = None

        _extractor = (
            metadata_extractor
            if metadata_extractor is not None
            else extract_circuit_metadata
        )
        for i, s in enumerate(settings):
            circuit = stim.Circuit.from_file(str(s.circuit_path))
            sampler_seed = stable_seed("sampler", f"idx={i}", base=worker_seed)
            compiled = circuit.compile_detector_sampler(seed=sampler_seed)
            meta = _extractor(circuit, s.distance, s.rounds)

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
        if self._weights is not None:
            idx = int(self._rng.choice(self._n_settings, p=self._weights))
        else:
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
