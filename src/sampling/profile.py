"""Operation profile registry: everything that varies along the experiment axis.

Each operation is registered once with a frozen record declaring its circuit
root, metadata extractor, representation version, ordered observable names,
metric policy, and run-directory segment.  Resolution is by operation name;
unknown names raise, listing the registered ones.  There is no default entry
and no argument anywhere that falls back to ``"memory"``.

The registry is a module-level mapping of frozen records, resolved by name,
with every entry written in-tree.  Not entry points, not dynamic import, not
a config-supplied class path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as _np

from sampling.experiment import validate_operation_name
from sampling.graph import (
    CircuitMetadata,
    LogicalOperationLayout,
    MetadataExtractor,
    extract_circuit_metadata,
)
from sampling.representation import (
    DataContract,
    LabelSpec,
    resolve_builder,
    resolve_descriptor,
)


if TYPE_CHECKING:
    import stim


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    """Declares which metrics may appear in result artifacts for an operation.

    Parameters
    ----------
    include_per_round_ler : bool
        Whether per-round logical error rate may be reported.  ``True``
        for memory experiments where rounds are the repeating unit,
        ``False`` for logical operations where the "round" count is the
        full schedule depth and per-round LER is not meaningful.
    """

    include_per_round_ler: bool


@dataclass(frozen=True, slots=True)
class OperationProfile:
    """Frozen capability declaration for one operation on the experiment axis.

    Parameters
    ----------
    operation : str
        Lowercase identifier matching the experiment key convention.
    circuit_root : Path
        Relative path to the operation's circuit directory.
    metadata_extractor : MetadataExtractor
        Builds the circuit metadata the representation's builder reads.
    representation_version : str
        Representation version tag (e.g. ``"spatial"``).
    observable_names : tuple of str
        Ordered observable names.
    metric_policy : MetricPolicy
        Which metrics may appear in result artifacts.
    run_dir_segment : str
        Operation segment under ``outputs/runs/{operation}/{distance}/{strategy}``.
    """

    operation: str
    circuit_root: Path
    metadata_extractor: MetadataExtractor
    representation_version: str
    observable_names: tuple[str, ...]
    metric_policy: MetricPolicy
    run_dir_segment: str
    space_time_convention: str = "per_shot"

    def __post_init__(self) -> None:
        validate_operation_name(self.operation)
        if not self.observable_names:
            raise ValueError(
                f"observable_names must be non-empty for operation {self.operation!r}"
            )
        if not self.run_dir_segment:
            raise ValueError(
                f"run_dir_segment must be non-empty for operation {self.operation!r}"
            )
        if not callable(self.metadata_extractor):
            raise ValueError(
                f"metadata_extractor must be callable for operation {self.operation!r}"
            )
        resolve_descriptor(self.representation_version)
        resolve_builder(self.representation_version)

    @property
    def data_contract(self) -> DataContract:
        """Build the data contract this operation trains and infers under."""
        return DataContract(
            representation=resolve_descriptor(self.representation_version),
            labels=LabelSpec(
                num_observables=len(self.observable_names),
                observable_names=self.observable_names,
            ),
        )


_REGISTRY: dict[str, OperationProfile] = {}


def resolve_profile(operation: str) -> OperationProfile:
    """Look up the profile for a registered operation.

    Parameters
    ----------
    operation : str
        Operation name (e.g. ``"memory"``).

    Returns
    -------
    OperationProfile

    Raises
    ------
    ValueError
        If *operation* is not registered, listing the known names.
    """
    profile = _REGISTRY.get(operation)
    if profile is None:
        raise ValueError(
            f"Unknown operation {operation!r}, registered: {sorted(_REGISTRY)}"
        )
    return profile


def registered_operations() -> frozenset[str]:
    """Return the set of registered operation names."""
    return frozenset(_REGISTRY)


def require_operation(fields: Mapping[str, Any]) -> OperationProfile:
    """Resolve the profile named by the ``operation`` field of a config.

    A config states its operation; there is none to fall back to.

    Parameters
    ----------
    fields : Mapping
        Config field name to value.

    Returns
    -------
    OperationProfile

    Raises
    ------
    ValueError
        If the fields name no operation, or an unregistered one; both list the
        registered operations.
    """
    if "operation" not in fields:
        raise ValueError(f"Config sets no operation; set one of {sorted(_REGISTRY)}")
    return resolve_profile(fields["operation"])


def _register(profile: OperationProfile) -> None:
    if profile.operation in _REGISTRY:
        raise ValueError(f"Operation {profile.operation!r} is registered twice")
    _REGISTRY[profile.operation] = profile


_register(
    OperationProfile(
        operation="memory",
        circuit_root=Path("data/circuits/memory"),
        metadata_extractor=extract_circuit_metadata,
        representation_version="spatial",
        observable_names=("logical_observable",),
        metric_policy=MetricPolicy(include_per_round_ler=True),
        run_dir_segment="memory",
        space_time_convention="per_round_cycle",
    )
)


def _make_logical_op_extractor(
    circuit_root: Path,
    observable_names: tuple[str, ...],
) -> MetadataExtractor:
    """Build a metadata extractor for a logical operation.

    The returned function reads the circuit's manifest to obtain
    phase/seam/patch metadata.  tqec is never imported: the manifest
    carries the pre-validated structural data.

    Parameters
    ----------
    circuit_root : Path
        Root directory of the operation's committed circuits.
    observable_names : tuple of str
        Ordered observable names from the profile.
    """

    def _extractor(
        circuit: stim.Circuit, distance: int, rounds: int
    ) -> CircuitMetadata:
        base = extract_circuit_metadata(circuit, distance, rounds)

        prefix = f"d{distance}_r{rounds}_"
        manifest: dict | None = None
        for f in sorted(circuit_root.iterdir()):
            if f.name.startswith(prefix) and f.name.endswith(".manifest.json"):
                manifest = json.loads(f.read_text(encoding="utf-8"))
                break
        if manifest is None:
            raise FileNotFoundError(
                f"No manifest for d={distance}, r={rounds} in {circuit_root}"
            )

        n_obs = manifest.get("num_observables")
        if n_obs is not None and int(n_obs) != len(observable_names):
            raise ValueError(
                f"Manifest declares {n_obs} observables but profile provides "
                f"{len(observable_names)} names: {observable_names}"
            )

        raw_windows = manifest["phase_windows"]
        coords = base.detector_coords
        n_det = base.num_detectors

        phase_ids = _np.empty(n_det, dtype=_np.intp)
        for det_idx in range(n_det):
            t = coords[det_idx, 2]
            for pw_idx, pw in enumerate(raw_windows):
                if pw["t_start"] <= t < pw["t_end"]:
                    phase_ids[det_idx] = pw_idx
                    break

        seam_indices = _np.array(manifest["seam_detector_indices"], dtype=_np.intp)
        seam_mask = _np.zeros(n_det, dtype=_np.bool_)
        seam_mask[seam_indices] = True

        return CircuitMetadata(
            detector_coords=coords,
            distance=base.distance,
            rounds=base.rounds,
            num_detectors=n_det,
            dem_edge_weights=base.dem_edge_weights,
            layout=LogicalOperationLayout(
                phase_ids=phase_ids,
                phase_names=tuple(pw["name"] for pw in raw_windows),
                patch_ids=_np.array(manifest["patch_ids"], dtype=_np.intp),
                seam_mask=seam_mask,
            ),
        )

    return _extractor


_register(
    OperationProfile(
        operation="zz_merge_split",
        circuit_root=Path("data/circuits/zz_merge_split"),
        metadata_extractor=_make_logical_op_extractor(
            circuit_root=Path("data/circuits/zz_merge_split"),
            observable_names=("Z_0", "Z_1"),
        ),
        representation_version="phased",
        observable_names=("Z_0", "Z_1"),
        metric_policy=MetricPolicy(include_per_round_ler=False),
        run_dir_segment="zz_merge_split",
        space_time_convention="per_shot",
    )
)


__all__ = [
    "MetricPolicy",
    "OperationProfile",
    "registered_operations",
    "require_operation",
    "resolve_profile",
]
