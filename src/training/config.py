"""Training configuration and seeding utilities."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from config_file import read_config_fields
from sampling.profile import require_operation, resolve_profile


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


# Fields holding a path; YAML and the CLI deliver them as strings.
_TRAIN_PATH_FIELDS: Final[tuple[str, ...]] = ("circuit_dir", "output_dir", "resume")

# YAML keys read into TrainConfig fields: top-level, then per nested section.
_TOP_LEVEL_KEYS: Final[tuple[str, ...]] = (
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
)
_SECTION_KEYS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("model", ("hidden_dim", "num_layers", "dropout")),
    ("optimisation", ("lr", "weight_decay", "batch_size")),
)


@dataclass(kw_only=True)
class TrainConfig:
    """Training hyperparameters for sample-budget training.

    Parameters
    ----------
    operation : str
        Operation name, resolved to an ``OperationProfile`` at
        construction.  Determines the circuit root, representation, and
        run-directory segment.
    circuit_dir : Path
        Directory containing committed ``.stim`` circuit files.
        :meth:`from_fields` fills it from the operation's profile when the
        fields leave it out.
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

    operation: str
    circuit_dir: Path
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

    @staticmethod
    def read_yaml_fields(path: str | Path) -> dict[str, Any]:
        """Read a YAML config into field values, resolving nothing.

        Parameters
        ----------
        path : str or Path
            YAML file; its ``model`` and ``optimisation`` sections are
            flattened into the fields they hold.

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
        return read_config_fields(Path(path), _TOP_LEVEL_KEYS, _SECTION_KEYS)

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> TrainConfig:
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
        TrainConfig

        Raises
        ------
        ValueError
            If the fields name no registered operation.
        """
        values = dict(fields)
        for key in _TRAIN_PATH_FIELDS:
            if values.get(key) is not None:
                values[key] = Path(values[key])
        profile = require_operation(values)
        if values.get("circuit_dir") is None:
            values["circuit_dir"] = profile.circuit_root
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load configuration from a YAML file."""
        return cls.from_fields(cls.read_yaml_fields(path))

    def __post_init__(self) -> None:
        resolve_profile(self.operation)  # rejects an unregistered operation
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


def seed_everything(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
