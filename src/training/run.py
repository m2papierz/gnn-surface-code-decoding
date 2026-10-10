"""A training run: what names it, where it lives, and where it stands.

``RunSpec`` resolves a config against the circuits discovery returned.  The
sha256 of its JSON names the run directory, so two runs that differ in any
setting, circuit or representation never share one, and a run that would
repeat an existing one is refused rather than overwriting it.  The directory
is created whole, holding ``config.json`` (the spec) and ``last.pt`` (the
training state at sample 0).  ``last.pt`` is replaced at every validation and
is all a resume reads; ``best.pt``, written once when training ends, marks the
run complete.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from artifact_file import file_sha256, sync_directory, write_json
from sampling.profile import resolve_profile
from sampling.representation import DataContract
from sampling.sampler import CircuitSetting
from sampling.seeding import stable_seed
from training.config import TrainConfig
from training.primitives import (
    TrainingState,
    checkpoint_payload,
    load_checkpoint,
    write_payload,
)


__all__ = [
    "BEST_CHECKPOINT",
    "CONFIG_FILE",
    "HISTORY_FILE",
    "LAST_CHECKPOINT",
    "BestState",
    "CircuitDigest",
    "RunDirectoryError",
    "RunProgress",
    "RunSpec",
    "create_run_dir",
    "open_run_dir",
    "read_last_checkpoint",
    "require_new_run_dir",
    "resolve_run_dir",
    "segment_seed",
    "write_history",
    "write_last_checkpoint",
]


CONFIG_FILE: Final[str] = "config.json"
LAST_CHECKPOINT: Final[str] = "last.pt"
BEST_CHECKPOINT: Final[str] = "best.pt"
HISTORY_FILE: Final[str] = "history.json"

# Hex digits of the spec's sha256 that name its directory: 48 bits, far beyond
# any number of runs one (operation, distances) will ever hold.
_RUN_ID_HEX_DIGITS: Final[int] = 12


class RunDirectoryError(Exception):
    """A run directory is not in the state the requested step needs."""


@dataclass(frozen=True, slots=True)
class CircuitDigest:
    """A circuit a run samples, by path and the sha256 of its bytes.

    Parameters
    ----------
    path : Path
        The ``.stim`` file, as discovery found it.
    sha256 : str
        Hex digest of its bytes.
    """

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything that determines what a run trains.

    Parameters
    ----------
    config : TrainConfig
        The resolved training config.
    contract : DataContract
        The operation's data contract.
    distances : tuple of int
        Code distances of the circuits the run samples, ascending.
    circuits : tuple of CircuitDigest
        Every circuit the run samples, in discovery order.
    """

    config: TrainConfig
    contract: DataContract
    distances: tuple[int, ...]
    circuits: tuple[CircuitDigest, ...]

    def __post_init__(self) -> None:
        if not self.circuits:
            raise ValueError(
                f"A run samples at least one circuit; the spec for operation "
                f"{self.config.operation!r} lists none"
            )

    @classmethod
    def resolve(
        cls, config: TrainConfig, settings: Sequence[CircuitSetting]
    ) -> RunSpec:
        """Resolve *config* against the settings discovery returned for it.

        Parameters
        ----------
        config : TrainConfig
            The resolved training config.
        settings : sequence of CircuitSetting
            The circuits the run samples.

        Returns
        -------
        RunSpec
        """
        return cls(
            config=config,
            contract=resolve_profile(config.operation).data_contract,
            distances=tuple(sorted({s.distance for s in settings})),
            circuits=tuple(
                CircuitDigest(path=s.circuit_path, sha256=file_sha256(s.circuit_path))
                for s in settings
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """The spec as JSON values.

        ``output_dir`` is left out: it is where the run lives, not what it
        is, so moving the outputs root renames no run.
        """
        config = self.config.to_dict()
        del config["output_dir"]
        return {
            "config": config,
            "contract": self.contract.to_dict(),
            "distances": list(self.distances),
            "circuits": [
                {"path": str(c.path), "sha256": c.sha256} for c in self.circuits
            ],
        }

    @property
    def run_id(self) -> str:
        """Leading hex digits of the sha256 of the spec's canonical JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[:_RUN_ID_HEX_DIGITS]

    def record(self) -> dict[str, Any]:
        """What ``config.json`` holds: the run id and the spec."""
        return {"run_id": self.run_id, **self.to_dict()}


@dataclass(frozen=True, slots=True)
class BestState:
    """The training state at the best validation so far.

    Parameters
    ----------
    samples_consumed : int
        Training samples consumed at that validation.
    metric : float
        Its validation metric.
    state : dict
        A :func:`training.primitives.snapshot_state` copy of the state.
    """

    samples_consumed: int
    metric: float
    state: dict[str, Any]


@dataclass(slots=True)
class RunProgress:
    """Where a run stands beside its training state; ``last.pt`` records it.

    Parameters
    ----------
    samples_consumed : int
        Training samples consumed.
    checks_without_improvement : int
        Consecutive validations without a new best, for early stopping.
    segment_starts : list of int
        Samples consumed when each training segment began: ``[0]`` for a run
        that was never resumed.
    history : list of dict
        One entry per validation.
    best : BestState or None
        The state at the best validation; ``None`` before the first one.
    """

    samples_consumed: int
    checks_without_improvement: int
    segment_starts: list[int]
    history: list[dict[str, Any]]
    best: BestState | None

    @classmethod
    def start(cls) -> RunProgress:
        """The progress of a run that has not consumed a sample."""
        return cls(
            samples_consumed=0,
            checks_without_improvement=0,
            segment_starts=[0],
            history=[],
            best=None,
        )

    @property
    def best_metric(self) -> float:
        """The best validation metric so far; infinite before the first."""
        return math.inf if self.best is None else self.best.metric

    def is_finished(self, cfg: TrainConfig) -> bool:
        """Whether the budget is spent or early stopping has triggered."""
        if self.samples_consumed >= cfg.sample_budget:
            return True
        return cfg.patience > 0 and self.checks_without_improvement >= cfg.patience

    def to_dict(self) -> dict[str, Any]:
        """The progress as stored in ``last.pt``."""
        best = None
        if self.best is not None:
            best = {
                "samples_consumed": self.best.samples_consumed,
                "metric": self.best.metric,
                "state": self.best.state,
            }
        return {
            "samples_consumed": self.samples_consumed,
            "checks_without_improvement": self.checks_without_improvement,
            "segment_starts": self.segment_starts,
            "history": self.history,
            "best": best,
        }

    @classmethod
    def from_dict(cls, saved: dict[str, Any]) -> RunProgress:
        """Rebuild the progress :meth:`to_dict` stored."""
        best = saved["best"]
        return cls(
            samples_consumed=saved["samples_consumed"],
            checks_without_improvement=saved["checks_without_improvement"],
            segment_starts=saved["segment_starts"],
            history=saved["history"],
            best=None
            if best is None
            else BestState(
                samples_consumed=best["samples_consumed"],
                metric=best["metric"],
                state=best["state"],
            ),
        )


def resolve_run_dir(spec: RunSpec) -> Path:
    """The directory of the run *spec* names.

    Parameters
    ----------
    spec : RunSpec
        The run's resolved spec.

    Returns
    -------
    Path
        ``<output_dir>/<operation>/d<distances>/<run_id>``, the distances
        joined by ``_`` (``d3``, ``d3_5_7``).
    """
    config = spec.config
    operation_segment = resolve_profile(config.operation).run_dir_segment
    distance_segment = "d" + "_".join(str(d) for d in spec.distances)
    return config.output_dir / operation_segment / distance_segment / spec.run_id


def segment_seed(seed: int, samples_consumed: int) -> int:
    """Seed of the training segment that starts after *samples_consumed* samples.

    Stim's compiled samplers cannot save their state, so a resumed run cannot
    continue its stream.  Each segment draws a fresh stream from this seed:
    no shot is reused, and the shots a run sees are determined by where each
    earlier segment stopped.

    Parameters
    ----------
    seed : int
        The run's master seed.
    samples_consumed : int
        Samples consumed before the segment, 0 for the first.

    Returns
    -------
    int
    """
    return stable_seed("train", f"from_sample={samples_consumed}", base=seed)


def require_new_run_dir(run_dir: Path) -> None:
    """Refuse a run directory that already holds a run.

    Raises
    ------
    RunDirectoryError
        If *run_dir* exists and is not empty.
    """
    if not (run_dir.exists() and any(run_dir.iterdir())):
        return
    if (run_dir / BEST_CHECKPOINT).exists():
        raise RunDirectoryError(
            f"{run_dir} holds a finished run of this spec, and a run is never "
            f"overwritten: remove the directory to train it again"
        )
    raise RunDirectoryError(
        f"{run_dir} holds an unfinished run of this spec, and a run is never "
        f"overwritten: continue it with --resume, or remove the directory to "
        f"train it again"
    )


def create_run_dir(
    spec: RunSpec, write_initial_checkpoint: Callable[[Path], None]
) -> Path:
    """Create the run directory whole, with ``config.json`` and ``last.pt``.

    Both are written into a hidden sibling, which is then renamed into place,
    so a run directory either does not exist or can be resumed.

    Parameters
    ----------
    spec : RunSpec
        The run's resolved spec, recorded as ``config.json``.
    write_initial_checkpoint : callable
        Writes the training state at sample 0 to the path it is given.

    Returns
    -------
    Path
        The run directory.

    Raises
    ------
    RunDirectoryError
        If the directory already holds a run.
    """
    run_dir = resolve_run_dir(spec)
    require_new_run_dir(run_dir)
    staging = run_dir.with_name(f".{run_dir.name}.{secrets.token_hex(8)}")
    staging.mkdir(parents=True)
    try:
        write_json(staging / CONFIG_FILE, spec.record())
        write_initial_checkpoint(staging / LAST_CHECKPOINT)
        try:
            staging.rename(run_dir)
        except OSError as exc:
            raise RunDirectoryError(
                f"{run_dir} was created by another process while this run was "
                f"being set up"
            ) from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    sync_directory(run_dir.parent)
    return run_dir


def open_run_dir(spec: RunSpec) -> Path:
    """Return the directory of the unfinished run *spec* names, to resume it.

    Parameters
    ----------
    spec : RunSpec
        The run's resolved spec.

    Returns
    -------
    Path
        The run directory.

    Raises
    ------
    RunDirectoryError
        If no run of this spec exists, the run is finished, its
        ``config.json`` records another spec, or it has no ``last.pt``.
    """
    run_dir = resolve_run_dir(spec)
    if not run_dir.is_dir():
        raise RunDirectoryError(
            f"No run to resume at {run_dir}: no run of this spec was started. "
            f"The directory is named by the resolved config, its circuits and "
            f"its contract, so resume with the config the run was started with"
        )
    if (run_dir / BEST_CHECKPOINT).exists():
        raise RunDirectoryError(
            f"{run_dir} holds a finished run ({BEST_CHECKPOINT} exists); a "
            f"finished run is never resumed"
        )
    config_path = run_dir / CONFIG_FILE
    if json.loads(config_path.read_text(encoding="utf-8")) != spec.record():
        raise RunDirectoryError(
            f"{config_path} records another spec than the one resolved now"
        )
    if not (run_dir / LAST_CHECKPOINT).is_file():
        raise RunDirectoryError(f"{run_dir} has no {LAST_CHECKPOINT} to resume from")
    return run_dir


def write_last_checkpoint(
    path: Path,
    state: TrainingState,
    progress: RunProgress,
    *,
    contract: DataContract,
    cfg: TrainConfig,
) -> None:
    """Write the resume checkpoint: the training state and the run's progress.

    Parameters
    ----------
    path : Path
        Destination, the run's ``last.pt``.
    state : TrainingState
        Current training state.
    progress : RunProgress
        Where the run stands, best state included.
    contract : DataContract
        The run's data contract.
    cfg : TrainConfig
        The run's config.
    """
    payload = checkpoint_payload(
        state,
        samples_consumed=progress.samples_consumed,
        best_metric=progress.best_metric,
        # Training validates at the default threshold; calibration happens
        # once, on the best state, when training ends.
        decision_threshold=0.0,
        contract=contract,
        cfg=cfg,
    )
    payload["progress"] = progress.to_dict()
    write_payload(path, payload)


def read_last_checkpoint(
    path: Path,
    state: TrainingState,
    cfg: TrainConfig,
    *,
    contract: DataContract,
) -> RunProgress:
    """Restore the training state from ``last.pt`` and return the run's progress.

    Parameters
    ----------
    path : Path
        The run's ``last.pt``.
    state : TrainingState
        Training state to restore into.
    cfg : TrainConfig
        The run's config (checked for architecture compatibility).
    contract : DataContract
        The run's data contract; the checkpoint's must equal it.

    Returns
    -------
    RunProgress

    Raises
    ------
    KeyError
        If the checkpoint lacks a key the trainer always writes, naming it.
    """
    ckpt = load_checkpoint(path, state, cfg, contract=contract)
    try:
        return RunProgress.from_dict(ckpt["progress"])
    except KeyError as exc:
        raise KeyError(f"Checkpoint {path} has no {exc.args[0]!r}") from exc


def write_history(path: Path, progress: RunProgress) -> None:
    """Write the run's segment starts and validation history as JSON."""
    write_json(
        path,
        {"segment_starts": progress.segment_starts, "validations": progress.history},
    )
