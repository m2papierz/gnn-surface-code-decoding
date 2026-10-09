"""Graph representation versioning and data contract.

Owns the frozen descriptor that names a representation version, its feature
dimensions, and ordered feature names, plus a label spec (observable count
and ordered names).  Together they answer "what shape of data was this
trained on" and travel as one record from dataset through model, checkpoint,
and inference path.

Builder resolution is a lookup keyed by representation version so no
consumer names a graph builder directly.

Representation versions are named by what they capture, not by ordinal:

- ``"spatial"`` - fired detectors with static spatial features
  (position, boundary distance, measurement basis).
- ``"phased"`` - spatial core plus phase-window one-hot, patch identity,
  and dynamic boundary distances at the seam.

A label spec belongs to an operation, not to a representation, so this module
holds no contract: an operation's contract is its profile's ``data_contract``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from sampling.graph import (
    CircuitMetadata,
    FiredDetectorGraph,
    build_fired_detector_graph,
    build_phased_detector_graph,
)


type GraphBuilder = Callable[[np.ndarray, CircuitMetadata], FiredDetectorGraph]


@dataclass(frozen=True, slots=True)
class RepresentationDescriptor:
    """Names a graph representation version and its feature layout.

    Parameters
    ----------
    version : str
        Short version tag (e.g. ``"spatial"``).
    node_dim : int
        Node feature dimensionality.
    edge_dim : int
        Edge feature dimensionality.
    node_features : tuple of str
        Ordered node feature names, length must equal *node_dim*.
    edge_features : tuple of str
        Ordered edge feature names, length must equal *edge_dim*.
    """

    version: str
    node_dim: int
    edge_dim: int
    node_features: tuple[str, ...]
    edge_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("version must be a non-empty string")
        if self.node_dim < 1:
            raise ValueError(f"node_dim must be >= 1, got {self.node_dim}")
        if self.edge_dim < 1:
            raise ValueError(f"edge_dim must be >= 1, got {self.edge_dim}")
        if len(self.node_features) != self.node_dim:
            raise ValueError(
                f"node_features length {len(self.node_features)} != "
                f"node_dim {self.node_dim}"
            )
        if len(self.edge_features) != self.edge_dim:
            raise ValueError(
                f"edge_features length {len(self.edge_features)} != "
                f"edge_dim {self.edge_dim}"
            )


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """Label shape for a training or inference task.

    Parameters
    ----------
    num_observables : int
        Number of logical observables predicted per shot.
    observable_names : tuple of str
        Ordered observable names, length must equal *num_observables*.
    """

    num_observables: int
    observable_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.num_observables < 1:
            raise ValueError(
                f"num_observables must be >= 1, got {self.num_observables}"
            )
        if len(self.observable_names) != self.num_observables:
            raise ValueError(
                f"observable_names length {len(self.observable_names)} != "
                f"num_observables {self.num_observables}"
            )


@dataclass(frozen=True, slots=True)
class DataContract:
    """Complete data contract: representation descriptor + label spec.

    Replaces today's separate ``node_dim`` / ``edge_dim`` parameters so the
    signature does not grow a parameter per axis.

    Parameters
    ----------
    representation : RepresentationDescriptor
        Which graph representation this data uses.
    labels : LabelSpec
        What the model predicts (observable count and names).
    """

    representation: RepresentationDescriptor
    labels: LabelSpec

    @property
    def node_dim(self) -> int:
        """Node feature dimensionality."""
        return self.representation.node_dim

    @property
    def edge_dim(self) -> int:
        """Edge feature dimensionality."""
        return self.representation.edge_dim

    @property
    def num_observables(self) -> int:
        """Number of logical observables."""
        return self.labels.num_observables

    @property
    def version(self) -> str:
        """Representation version tag."""
        return self.representation.version

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a checkpoint-safe dict."""
        return {
            "version": self.representation.version,
            "node_dim": self.representation.node_dim,
            "edge_dim": self.representation.edge_dim,
            "node_features": list(self.representation.node_features),
            "edge_features": list(self.representation.edge_features),
            "num_observables": self.labels.num_observables,
            "observable_names": list(self.labels.observable_names),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataContract:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            representation=RepresentationDescriptor(
                version=d["version"],
                node_dim=d["node_dim"],
                edge_dim=d["edge_dim"],
                node_features=tuple(d["node_features"]),
                edge_features=tuple(d["edge_features"]),
            ),
            labels=LabelSpec(
                num_observables=d["num_observables"],
                observable_names=tuple(d["observable_names"]),
            ),
        )

    @classmethod
    def from_checkpoint_config(cls, cfg: dict[str, Any]) -> DataContract:
        """Reconstruct from a checkpoint config dict.

        Raises
        ------
        KeyError
            If the checkpoint lacks a ``contract`` key.
        """
        return cls.from_dict(cfg["contract"])


SPATIAL = RepresentationDescriptor(
    version="spatial",
    node_dim=6,
    edge_dim=6,
    node_features=("x_norm", "y_norm", "t_norm", "d_x", "d_y", "basis"),
    edge_features=("dx", "dy", "dt", "euclidean", "chebyshev", "dem_weight"),
)

PHASED = RepresentationDescriptor(
    version="phased",
    node_dim=11,
    edge_dim=6,
    node_features=(
        "x_norm",
        "y_norm",
        "t_norm",
        "d_x",
        "d_y",
        "basis",
        "phase_memory",
        "phase_merge",
        "phase_split",
        "patch_id",
        "is_seam",
    ),
    edge_features=("dx", "dy", "dt", "euclidean", "chebyshev", "dem_weight"),
)


_BUILDER_REGISTRY: Final[dict[str, GraphBuilder]] = {
    "spatial": build_fired_detector_graph,
    "phased": build_phased_detector_graph,
}

_DESCRIPTOR_REGISTRY: dict[str, RepresentationDescriptor] = {
    "spatial": SPATIAL,
    "phased": PHASED,
}

SUPPORTED_FAST_PATH_VERSIONS: frozenset[str] = frozenset({"spatial"})

# Versions whose builder reads the circuit's logical-operation layout.
_LAYOUT_VERSIONS: Final[frozenset[str]] = frozenset({"phased"})


def registered_representations() -> tuple[RepresentationDescriptor, ...]:
    """Return every registered representation descriptor, ordered by version."""
    return tuple(
        _DESCRIPTOR_REGISTRY[version] for version in sorted(_DESCRIPTOR_REGISTRY)
    )


def resolve_descriptor(version: str) -> RepresentationDescriptor:
    """Look up the representation descriptor for a version tag.

    Parameters
    ----------
    version : str
        Representation version tag (e.g. ``"spatial"``).

    Returns
    -------
    RepresentationDescriptor

    Raises
    ------
    ValueError
        If *version* is not registered.
    """
    descriptor = _DESCRIPTOR_REGISTRY.get(version)
    if descriptor is None:
        raise ValueError(
            f"Unknown representation version {version!r}, "
            f"registered: {sorted(_DESCRIPTOR_REGISTRY)}"
        )
    return descriptor


def resolve_builder(version: str) -> GraphBuilder:
    """Look up the graph builder for a representation version.

    Parameters
    ----------
    version : str
        Representation version tag (e.g. ``"spatial"``).

    Returns
    -------
    GraphBuilder
        The builder function: ``(syndrome, metadata) -> FiredDetectorGraph``.

    Raises
    ------
    ValueError
        If *version* is not registered.
    """
    builder = _BUILDER_REGISTRY.get(version)
    if builder is None:
        raise ValueError(
            f"Unknown representation version {version!r}, "
            f"registered: {sorted(_BUILDER_REGISTRY)}"
        )
    return builder


def requires_layout(version: str) -> bool:
    """Whether the builder of a representation reads a logical-operation layout.

    Parameters
    ----------
    version : str
        Representation version tag (e.g. ``"phased"``).

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If *version* is not registered.
    """
    return resolve_descriptor(version).version in _LAYOUT_VERSIONS


# What a contract fixes, each axis named the way a mismatch on it is reported.
_CONTRACT_AXES: Final[tuple[tuple[str, Callable[[DataContract], object]], ...]] = (
    ("Representation", lambda contract: contract.version),
    ("Node feature", lambda contract: contract.representation.node_features),
    ("Edge feature", lambda contract: contract.representation.edge_features),
    ("Observable count", lambda contract: contract.num_observables),
    ("Observable name", lambda contract: contract.labels.observable_names),
)


def require_contract(
    expected: DataContract, found: DataContract, *, source: str | Path
) -> None:
    """Reject a contract read from an artifact unless it is the expected one.

    Parameters
    ----------
    expected : DataContract
        Contract the consumer runs under.
    found : DataContract
        Contract the artifact declares.
    source : str or Path
        The artifact *found* was read from, for the error message.

    Raises
    ------
    ValueError
        Naming every axis on which the two contracts differ.
    """
    mismatches = [
        f"{axis} mismatch: {source} declares {read(found)!r}, "
        f"expected {read(expected)!r}"
        for axis, read in _CONTRACT_AXES
        if read(found) != read(expected)
    ]
    if mismatches:
        raise ValueError("; ".join(mismatches))


__all__ = [
    "DataContract",
    "LabelSpec",
    "PHASED",
    "RepresentationDescriptor",
    "SUPPORTED_FAST_PATH_VERSIONS",
    "SPATIAL",
    "registered_representations",
    "require_contract",
    "requires_layout",
    "resolve_builder",
    "resolve_descriptor",
]
