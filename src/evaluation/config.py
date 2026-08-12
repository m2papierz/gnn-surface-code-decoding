"""Evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
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
    circuit_dir : Path or None
        Circuit directory for sanity evaluation (fresh-sample mode).
        If ``None`` (default), populated from the resolved profile's
        ``circuit_root``.
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

    operation: str = "memory"
    checkpoint: Path | None = None
    eval_root: Path = Path("data/eval")
    circuit_dir: Path | None = None
    distances: list[int] | None = None
    error_probs: list[float] | None = None
    batch_size: int = 256
    include_belief_matching: bool = True
    include_tesseract: bool = False
    shots: int = 100_000
    seed: int = 99

    def __post_init__(self) -> None:
        from sampling.profile import resolve_profile

        profile = resolve_profile(self.operation)

        if self.circuit_dir is None:
            self.circuit_dir = profile.circuit_root

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
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        flat: dict[str, Any] = {}

        for key in (
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
        ):
            if key in raw:
                flat[key] = raw[key]

        for key in ("checkpoint", "eval_root", "circuit_dir"):
            if key in flat and flat[key] is not None:
                flat[key] = Path(flat[key])

        return cls(**flat)
