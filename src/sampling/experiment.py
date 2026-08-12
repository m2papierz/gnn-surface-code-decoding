"""Experiment identity for training and evaluation points.

A training or evaluation point is identified by ``(operation, distance,
rounds, error_prob)``.  This module owns that record and the rules that
resolve it from a committed artifact: the manifest beside a circuit, an
evaluation set's own manifest, or the circuit an evaluation set was sampled
from.

The identity answers *which experiment* a point is on.  It says nothing about
how the circuit is built, how the shots are turned into graphs, or which
metrics may be reported for it - consumers read the resolved key rather than
comparing an operation name to a literal.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


__all__ = [
    "ExperimentKey",
    "circuit_key",
    "circuit_manifest_path",
    "eval_set_operation",
    "read_circuit_manifest",
    "validate_operation_name",
    "write_circuit_manifest",
]


_OPERATION_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")

_MANIFEST_SUFFIX: Final[str] = ".manifest.json"

# Fields every circuit manifest must declare: the experiment the circuit
# belongs to.  Provenance fields are added by whichever generator wrote it.
_KEY_FIELDS: Final[tuple[str, ...]] = ("operation", "distance", "rounds", "error_prob")

# Circuits committed before the experiment axis existed carry no manifest, and
# neither do the throwaway circuit directories built during testing.  Every
# such circuit is a static memory experiment.  A circuit written by any
# generator declares its operation in a manifest and is read from it, so this
# is a rule about unlabelled artifacts - not a value a consumer may default to.
_UNMANIFESTED_CIRCUIT_OPERATION: Final[str] = "memory"


def validate_operation_name(operation: str) -> None:
    """Reject an operation name that is not a lowercase identifier.

    Parameters
    ----------
    operation : str
        Candidate operation name.

    Raises
    ------
    ValueError
        If the name does not match ``[a-z][a-z0-9_]*``.
    """
    if not _OPERATION_NAME_RE.match(operation):
        raise ValueError(
            f"operation must match {_OPERATION_NAME_RE.pattern}, got {operation!r}"
        )


@dataclass(frozen=True, slots=True)
class ExperimentKey:
    """Identity of one training or evaluation point.

    Parameters
    ----------
    operation : str
        Logical operation the circuit realizes.  Static memory is the
        operation named ``"memory"``.
    distance : int
        Code distance.
    rounds : int
        Number of syndrome measurement rounds.
    error_prob : float
        Physical error probability.
    """

    operation: str
    distance: int
    rounds: int
    error_prob: float

    def __post_init__(self) -> None:
        validate_operation_name(self.operation)
        if self.distance < 1:
            raise ValueError(f"distance must be >= 1, got {self.distance}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if not (0 < self.error_prob < 1):
            raise ValueError(f"error_prob must be in (0, 1), got {self.error_prob}")

    def __str__(self) -> str:
        return (
            f"{self.operation} d={self.distance} r={self.rounds} p={self.error_prob:g}"
        )


def circuit_manifest_path(circuit_path: Path) -> Path:
    """Path of the manifest that sits beside a circuit file.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit file.

    Returns
    -------
    Path
        ``<circuit stem>.manifest.json`` in the circuit's directory.
    """
    return circuit_path.with_name(circuit_path.stem + _MANIFEST_SUFFIX)


def read_circuit_manifest(circuit_path: Path) -> dict[str, Any] | None:
    """Read the manifest beside a circuit.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit file.

    Returns
    -------
    dict or None
        Manifest contents, or ``None`` if the circuit carries no manifest.
    """
    manifest_path = circuit_manifest_path(circuit_path)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Circuit manifest {manifest_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Circuit manifest {manifest_path} must be a JSON object, got "
            f"{type(manifest).__name__}"
        )
    return manifest


def _key_from_manifest(manifest: Mapping[str, Any], source: Path) -> ExperimentKey:
    """Read the experiment key a manifest declares.

    Parameters
    ----------
    manifest : mapping
        Manifest contents.
    source : Path
        Path the manifest was read from, for error messages.

    Returns
    -------
    ExperimentKey

    Raises
    ------
    ValueError
        If any key field is absent, or if the declared values are invalid.
    """
    missing = [field for field in _KEY_FIELDS if field not in manifest]
    if missing:
        raise ValueError(
            f"Manifest {source} is missing required field(s) {missing}: a manifest "
            f"declares the experiment its artifact belongs to"
        )
    try:
        return ExperimentKey(
            operation=manifest["operation"],
            distance=int(manifest["distance"]),
            rounds=int(manifest["rounds"]),
            error_prob=float(manifest["error_prob"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Manifest {source} declares an invalid key: {exc}") from exc


def circuit_key(
    circuit_path: Path,
    *,
    distance: int,
    rounds: int,
    error_prob: float,
) -> ExperimentKey:
    """Resolve the experiment key of a committed circuit.

    The manifest beside the circuit is authoritative for the whole key, so a
    filename can never contradict it.  ``distance``, ``rounds`` and
    ``error_prob`` are the caller's reading of the filename and are used only
    for a circuit that carries no manifest.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit file.
    distance, rounds : int
        Code parameters read from the filename.
    error_prob : float
        Physical error probability read from the filename.

    Returns
    -------
    ExperimentKey
    """
    manifest = read_circuit_manifest(circuit_path)
    if manifest is None:
        return ExperimentKey(
            operation=_UNMANIFESTED_CIRCUIT_OPERATION,
            distance=distance,
            rounds=rounds,
            error_prob=error_prob,
        )
    return _key_from_manifest(manifest, circuit_manifest_path(circuit_path))


def _circuit_operation(circuit_path: Path) -> str:
    """Resolve the operation a committed circuit belongs to.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit file.

    Returns
    -------
    str

    Raises
    ------
    FileNotFoundError
        If neither the circuit nor a manifest beside it exists, so there is
        nothing to resolve the operation from.
    """
    manifest = read_circuit_manifest(circuit_path)
    if manifest is not None:
        return _key_from_manifest(
            manifest, circuit_manifest_path(circuit_path)
        ).operation
    if not circuit_path.exists():
        raise FileNotFoundError(
            f"Cannot resolve the operation of {circuit_path}: neither the circuit "
            f"nor a manifest beside it exists"
        )
    return _UNMANIFESTED_CIRCUIT_OPERATION


def eval_set_operation(manifest: Mapping[str, Any], circuit_file: str | Path) -> str:
    """Resolve the operation a frozen evaluation set belongs to.

    A set that declares its own operation is taken at its word, and the circuit
    it names must agree.  A set that does not declare one - every set frozen
    before the experiment axis existed - inherits the operation of that
    circuit, which is where the shots came from.

    Parameters
    ----------
    manifest : mapping
        Evaluation set manifest contents.
    circuit_file : str or Path
        Path to the circuit the shots were sampled from, as recorded in the
        manifest.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If the set declares an operation the circuit does not belong to.
    """
    circuit_path = Path(circuit_file)
    from_circuit = _circuit_operation(circuit_path)

    declared = manifest.get("operation")
    if declared is None:
        return from_circuit

    validate_operation_name(declared)
    if declared != from_circuit:
        raise ValueError(
            f"Eval set declares operation {declared!r} but its circuit "
            f"{circuit_path} belongs to operation {from_circuit!r}"
        )
    return declared


def write_circuit_manifest(
    circuit_path: Path,
    key: ExperimentKey,
    *,
    provenance: Mapping[str, Any],
) -> Path:
    """Write the manifest that declares which experiment a circuit belongs to.

    Parameters
    ----------
    circuit_path : Path
        Path to the ``.stim`` circuit file the manifest describes.
    key : ExperimentKey
        Identity to declare.
    provenance : mapping
        Generator-specific fields (versions, hashes, counts, command).  May
        not contain any key field - the key has one source.

    Returns
    -------
    Path
        Path the manifest was written to.

    Raises
    ------
    ValueError
        If ``provenance`` would overwrite a key field.
    """
    clashing = sorted(set(provenance) & set(_KEY_FIELDS))
    if clashing:
        raise ValueError(
            f"provenance may not redeclare key field(s) {clashing} for {circuit_path}"
        )

    manifest = {
        "operation": key.operation,
        "distance": key.distance,
        "rounds": key.rounds,
        "error_prob": key.error_prob,
        **provenance,
    }
    manifest_path = circuit_manifest_path(circuit_path)

    # Written through a temporary file in the same directory and renamed, so an
    # interrupted generator leaves either the previous manifest or the new one.
    # A half-written manifest would be worse than none: a truncated JSON object
    # can still parse, and would then declare an identity nobody chose.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, manifest_path)
    return manifest_path
