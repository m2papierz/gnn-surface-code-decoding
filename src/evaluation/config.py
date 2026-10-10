"""Evaluation configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from config_file import config_fields, read_config_fields
from sampling.profile import require_operation, resolve_profile


# YAML keys read into EvalConfig fields.
_EVAL_KEYS: Final[tuple[str, ...]] = (
    "operation",
    "checkpoint",
    "eval_root",
    "circuit_dir",
    "distances",
    "error_probs",
    "batch_size",
    "include_belief_matching",
    "include_tesseract",
    "shots",
    "seed",
)

# Fields holding a path; YAML and the CLI deliver them as strings.
_EVAL_PATH_FIELDS: Final[tuple[str, ...]] = ("checkpoint", "eval_root", "circuit_dir")


@dataclass(kw_only=True)
class EvalConfig:
    """Configuration for GNN evaluation against classical baselines.

    Parameters
    ----------
    operation : str
        Operation name, resolved to an ``OperationProfile`` at
        construction.  Determines circuit root, representation, and
        eval-set discovery path.
    checkpoint : Path or None
        Path to the trained GNN checkpoint (``best.pt``).
    eval_root : Path
        Root of the frozen eval-set tree.  Eval sets are discovered
        under ``<eval_root>/<operation>/``.
    circuit_dir : Path
        Circuit directory for sanity evaluation (fresh-sample mode).
        :meth:`from_fields` fills it from the operation's profile when the
        fields leave it out.
    distances : list of int or None
        Filter to these code distances.  ``None`` means all.
    error_probs : list of float or None
        Filter to these error probabilities.  ``None`` means all.
    batch_size : int
        Batch size for GNN inference.
    include_belief_matching : bool
        Include the Belief-Matching decoder in the harness.
    include_tesseract : bool
        Include the Tesseract near-MLE decoder (slow).
    shots : int
        Shots per setting for sanity evaluation.
    seed : int
        Stim sampler seed for sanity evaluation.
    """

    operation: str
    checkpoint: Path | None = None
    eval_root: Path = Path("data/eval")
    circuit_dir: Path
    distances: list[int] | None = None
    error_probs: list[float] | None = None
    batch_size: int = 256
    include_belief_matching: bool = True
    include_tesseract: bool = False
    shots: int = 100_000
    seed: int = 99

    def __post_init__(self) -> None:
        resolve_profile(self.operation)  # rejects an unregistered operation

    def to_dict(self) -> dict[str, Any]:
        """The field values as JSON values, paths as text."""
        return config_fields(self)

    @staticmethod
    def read_yaml_fields(path: str | Path) -> dict[str, Any]:
        """Read a YAML config into field values, resolving nothing.

        Parameters
        ----------
        path : str or Path
            Path to YAML config file.

        Returns
        -------
        dict
            Field name to raw value, for the fields the file sets.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file sets a key no field reads, or is not a mapping.
        """
        return read_config_fields(Path(path), _EVAL_KEYS)

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> EvalConfig:
        """Build a config from field values, resolving what they leave out.

        ``operation`` is required.  ``circuit_dir`` falls back to the circuit
        root of the operation the fields name, so callers apply every override
        before calling this.

        Parameters
        ----------
        fields : Mapping
            Field name to value; path fields may be strings.

        Returns
        -------
        EvalConfig

        Raises
        ------
        ValueError
            If the fields name no registered operation.
        """
        values = dict(fields)
        for key in _EVAL_PATH_FIELDS:
            if values.get(key) is not None:
                values[key] = Path(values[key])
        profile = require_operation(values)
        if values.get("circuit_dir") is None:
            values["circuit_dir"] = profile.circuit_root
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML config file.

        Returns
        -------
        EvalConfig
        """
        return cls.from_fields(cls.read_yaml_fields(path))
