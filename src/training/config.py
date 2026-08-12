"""Training configuration and seeding utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import yaml


# Backends that can carry a training run. ``model.ops`` also exposes ``cuda``;
# it is deliberately absent here. Those kernels are forward-only and register
# no autograd backward, so selecting one for training detaches the graph.
_TRAINING_BACKENDS: Final[frozenset[str]] = frozenset({"pytorch", "compiled"})

# Autocast dtypes, resolved here rather than by getattr(torch, name) so that a
# typo cannot silently resolve to a default, and a name that happens to exist on
# the torch module (``nn``, ``float64``) cannot reach autocast as a dtype.
_AMP_DTYPES: Final[dict[str, torch.dtype]] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class TrainConfig:
    """Training hyperparameters for sample-budget training.

    Parameters
    ----------
    operation : str
        Operation name, resolved to an ``OperationProfile`` at
        construction.  Determines the circuit root, representation, and
        run-directory segment.
    circuit_dir : Path or None
        Directory containing committed ``.stim`` circuit files.  If
        ``None`` (default), populated from the resolved profile's
        ``circuit_root``.
    output_dir : Path
        Directory for checkpoints and logs.
    hidden_dim : int
        Encoder hidden dimensionality.
    num_layers : int
        Number of message-passing layers.
    dropout : float
        Dropout probability.
    lr : float
        Peak learning rate (after warmup).
    weight_decay : float
        AdamW weight decay.
    sample_budget : int
        Total training samples to consume before halting.
    batch_size : int
        Graphs per training batch.
    num_workers : int
        DataLoader worker processes.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    patience : int
        Early stopping patience (validation checks without improvement).
        Set to 0 to disable early stopping.
    val_interval_samples : int
        Run validation every this many training samples consumed.
    val_size : int
        Number of samples in the frozen validation set, pre-sampled
        once at training start with a deterministic seed.
    seed : int
        Master random seed.
    resume : Path or None
        Path to checkpoint to resume from.
    backend : str
        Compute backend: ``"pytorch"`` (default) or ``"compiled"``
        (recommended on GPU).  Rejected in ``__post_init__`` if it is
        anything else; in particular the ``"cuda"`` backend is inference-only
        and cannot train, since its kernels register no autograd backward.
    compile_mode : str
        ``torch.compile`` mode (only used when backend is ``"compiled"``).
        Use ``"default"`` for training - GNN batches have dynamic shapes
        (variable N/E) which cause ``"reduce-overhead"`` to record
        excessive CUDA graphs and degrade performance over time.
    amp_dtype : str
        Autocast dtype for mixed precision: ``"bfloat16"`` (default,
        recommended on Ampere+ GPUs) or ``"float16"``.  Any other value is
        rejected in ``__post_init__``; use :attr:`amp_torch_dtype` to get the
        resolved ``torch.dtype``.  Only used when training on CUDA.  Model
        weights, optimizer state, and loss remain in float32 regardless.
    focal_alpha : float
        Focal loss balancing factor for the positive class.
    focal_gamma : float
        Focal loss focusing exponent.
    warmup_fraction : float
        Fraction of sample budget for LR warmup (linear ramp).
    include_p_feature : bool
        Attach physical error probability as a graph-level feature.
    """

    operation: str = "memory"
    circuit_dir: Path | None = None
    output_dir: Path = Path("outputs/runs")
    hidden_dim: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    sample_budget: int = 1_000_000
    batch_size: int = 64
    num_workers: int = 4
    max_grad_norm: float = 1.0
    patience: int = 10
    val_interval_samples: int = 50_000
    val_size: int = 10_000
    seed: int = 42
    resume: Path | None = None
    backend: str = "pytorch"
    compile_mode: str = "default"
    amp_dtype: str = "bfloat16"
    focal_alpha: float = 0.75
    focal_gamma: float = 1.0
    warmup_fraction: float = 0.05
    include_p_feature: bool = False
    distances: list[int] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load configuration from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        flat: dict[str, Any] = {}

        for key in (
            "operation",
            "circuit_dir",
            "output_dir",
            "num_workers",
            "max_grad_norm",
            "patience",
            "seed",
            "backend",
            "compile_mode",
            "amp_dtype",
            "focal_alpha",
            "focal_gamma",
            "sample_budget",
            "val_interval_samples",
            "val_size",
            "warmup_fraction",
            "include_p_feature",
            "distances",
        ):
            if key in raw:
                flat[key] = raw[key]

        for key in ("hidden_dim", "num_layers", "dropout"):
            if key in raw.get("model", {}):
                flat[key] = raw["model"][key]

        for key in ("lr", "weight_decay", "batch_size"):
            if key in raw.get("optimisation", {}):
                flat[key] = raw["optimisation"][key]

        for key in ("circuit_dir", "output_dir"):
            if key in flat:
                flat[key] = Path(flat[key])

        return cls(**flat)

    def __post_init__(self) -> None:
        from sampling.profile import resolve_profile

        profile = resolve_profile(self.operation)

        if self.circuit_dir is None:
            self.circuit_dir = profile.circuit_root

        if self.sample_budget < 1:
            raise ValueError(f"sample_budget must be >= 1, got {self.sample_budget}")
        if self.val_interval_samples < 1:
            raise ValueError(
                f"val_interval_samples must be >= 1, got {self.val_interval_samples}"
            )
        if self.val_size < 1:
            raise ValueError(f"val_size must be >= 1, got {self.val_size}")
        if not (0.0 < self.warmup_fraction < 1.0):
            raise ValueError(
                f"warmup_fraction must be in (0, 1), got {self.warmup_fraction}"
            )
        if self.backend == "cuda":
            raise ValueError(
                "backend='cuda' is inference-only and cannot train: the custom "
                "kernels are forward-only and register no autograd backward, so "
                "the graph detaches and no gradient reaches the encoder. Use "
                "'pytorch', or 'compiled' on GPU."
            )
        if self.backend not in _TRAINING_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(_TRAINING_BACKENDS)}, "
                f"got {self.backend!r}"
            )
        if self.amp_dtype not in _AMP_DTYPES:
            raise ValueError(
                f"amp_dtype must be one of {sorted(_AMP_DTYPES)}, "
                f"got {self.amp_dtype!r}"
            )

    @property
    def amp_torch_dtype(self) -> torch.dtype:
        """The autocast dtype named by :attr:`amp_dtype`.

        Validated in ``__post_init__``, so this cannot raise.
        """
        return _AMP_DTYPES[self.amp_dtype]


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One stage of the distance curriculum.

    Parameters
    ----------
    distances : tuple of int
        Active code distances in this stage.
    budget : int
        Training samples to consume in this stage.
    """

    distances: tuple[int, ...]
    budget: int

    def __post_init__(self) -> None:
        if not self.distances:
            raise ValueError("Stage must have at least one distance")
        if self.budget < 1:
            raise ValueError(f"budget must be >= 1, got {self.budget}")


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Configuration for mixed-distance curriculum training.

    Parameters
    ----------
    stages : tuple of CurriculumStage
        Curriculum stages in order.  Each stage extends the active
        distance set - once a distance enters, it never leaves.
    mwpm_ler : dict mapping distance to float
        Baseline MWPM LER per distance (average across p values on the
        eval sets).  Used to compute the LER gap that drives adaptive
        sampling weights.
    gap_eval_interval : int
        Re-measure per-distance LER gap every this many training
        samples and update sampling weights.
    min_distance_weight : float
        Floor on per-distance sampling weight to prevent starvation.
    val_size_per_distance : int
        Frozen validation samples per distance for gap measurement.
    """

    stages: tuple[CurriculumStage, ...]
    mwpm_ler: dict[int, float]
    gap_eval_interval: int = 500_000
    min_distance_weight: float = 0.1
    val_size_per_distance: int = 2_000

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("Must have at least one curriculum stage")
        if self.gap_eval_interval < 1:
            raise ValueError(
                f"gap_eval_interval must be >= 1, got {self.gap_eval_interval}"
            )
        if not (0.0 < self.min_distance_weight < 1.0):
            raise ValueError(
                f"min_distance_weight must be in (0, 1), got {self.min_distance_weight}"
            )
        if self.val_size_per_distance < 1:
            raise ValueError(
                f"val_size_per_distance must be >= 1, got {self.val_size_per_distance}"
            )
        prev: set[int] = set()
        for stage in self.stages:
            current = set(stage.distances)
            dropped = prev - current
            if dropped:
                raise ValueError(
                    f"Curriculum violates replay: distances {dropped} "
                    f"were in a previous stage but dropped from "
                    f"{stage.distances}"
                )
            prev = current

    @property
    def total_budget(self) -> int:
        """Total sample budget across all stages."""
        return sum(s.budget for s in self.stages)

    @classmethod
    def from_yaml(cls, raw: dict[str, Any]) -> CurriculumConfig:
        """Parse curriculum config from a YAML dict.

        Parameters
        ----------
        raw : dict
            The ``curriculum`` sub-dict from the YAML config file.
        """
        stages = tuple(
            CurriculumStage(
                distances=tuple(s["distances"]),
                budget=s["budget"],
            )
            for s in raw["stages"]
        )
        mwpm_ler = {int(k): float(v) for k, v in raw["mwpm_ler"].items()}
        kwargs: dict[str, Any] = {"stages": stages, "mwpm_ler": mwpm_ler}
        for key in (
            "gap_eval_interval",
            "min_distance_weight",
            "val_size_per_distance",
        ):
            if key in raw:
                kwargs[key] = raw[key]
        return cls(**kwargs)


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
