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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as _np

from sampling.experiment import validate_operation_name
from sampling.representation import (
    DataContract,
    LabelSpec,
    resolve_builder,
    resolve_descriptor,
)


if TYPE_CHECKING:
    import stim

    from sampling.graph import CircuitMetadata


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
    metadata_extractor : callable
        ``(stim.Circuit, distance, rounds) -> CircuitMetadata``.
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
    metadata_extractor: Callable[["stim.Circuit", int, int], "CircuitMetadata"]
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


def _register_memory() -> OperationProfile:
    from sampling.graph import extract_circuit_metadata

    profile = OperationProfile(
        operation="memory",
        circuit_root=Path("data/circuits/memory"),
        metadata_extractor=extract_circuit_metadata,
        representation_version="spatial",
        observable_names=("logical_observable",),
        metric_policy=MetricPolicy(include_per_round_ler=True),
        space_time_convention="per_round_cycle",
        run_dir_segment="memory",
    )
    _REGISTRY[profile.operation] = profile
    return profile


MEMORY_PROFILE: Final[OperationProfile] = _register_memory()


def _make_logical_op_extractor(
    circuit_root: Path,
    observable_names: tuple[str, ...],
) -> Callable[["stim.Circuit", int, int], "CircuitMetadata"]:
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
    import json

    def _extractor(
        circuit: "stim.Circuit", distance: int, rounds: int
    ) -> "CircuitMetadata":
        from sampling.graph import CircuitMetadata, extract_circuit_metadata

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

        phase_names = tuple(pw["name"] for pw in raw_windows)
        patch_ids = _np.array(manifest["patch_ids"], dtype=_np.intp)
        seam_indices = _np.array(manifest["seam_detector_indices"], dtype=_np.intp)
        seam_mask = _np.zeros(n_det, dtype=_np.bool_)
        seam_mask[seam_indices] = True

        return CircuitMetadata(
            detector_coords=coords,
            distance=base.distance,
            rounds=base.rounds,
            num_detectors=n_det,
            dem_edge_weights=base.dem_edge_weights,
            phase_ids=phase_ids,
            phase_names=phase_names,
            patch_ids=patch_ids,
            seam_mask=seam_mask,
        )

    return _extractor


def _register_zz_merge_split() -> OperationProfile:
    profile = OperationProfile(
        operation="zz_merge_split",
        circuit_root=Path("data/circuits/zz_merge_split"),
        metadata_extractor=_make_logical_op_extractor(
            circuit_root=Path("data/circuits/zz_merge_split"),
            observable_names=("Z_0", "Z_1"),
        ),
        representation_version="phased",
        observable_names=("Z_0", "Z_1"),
        metric_policy=MetricPolicy(include_per_round_ler=False),
        space_time_convention="per_shot",
        run_dir_segment="zz_merge_split",
    )
    _REGISTRY[profile.operation] = profile
    return profile


_register_zz_merge_split()


__all__ = [
    "MEMORY_PROFILE",
    "MetricPolicy",
    "OperationProfile",
    "registered_operations",
    "resolve_profile",
]
