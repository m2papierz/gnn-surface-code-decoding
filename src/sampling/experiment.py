"""Experiment identity for training and evaluation points.

A training or evaluation point is identified by ``(operation, distance,
rounds, error_prob)``.  This module owns that record and the rules that
resolve it from a committed artifact: a circuit declares its key in the
manifest beside it, and a frozen evaluation set inherits the key of the
circuit it is bound to by that circuit's bytes.

The identity answers *which experiment* a point is on.  It says nothing about
how the circuit is built, how the shots are turned into graphs, or which
metrics may be reported for it - consumers read the resolved key rather than
comparing an operation name to a literal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from artifact_file import file_sha256, write_json


__all__ = [
    "ExperimentKey",
    "circuit_key",
    "circuit_manifest_path",
    "eval_set_key",
    "manifest_key",
    "read_circuit_manifest",
    "read_manifest",
    "validate_operation_name",
    "write_circuit_manifest",
]


_OPERATION_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")

_MANIFEST_SUFFIX: Final[str] = ".manifest.json"

# Fields every circuit manifest must declare: the experiment the circuit
# belongs to.  Provenance fields are added by whichever generator wrote it.
_KEY_FIELDS: Final[tuple[str, ...]] = ("operation", "distance", "rounds", "error_prob")

# Fields that bind a frozen evaluation set to the bytes of its circuit.
_EVAL_SET_BINDING_FIELDS: Final[tuple[str, ...]] = ("circuit_file", "circuit_sha256")


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


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read a manifest: a JSON object.

    Parameters
    ----------
    manifest_path : Path
        Path to the manifest file.

    Returns
    -------
    dict
        Manifest contents.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON or holds anything but an object.
    """
    text = manifest_path.read_text(encoding="utf-8")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest {manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Manifest {manifest_path} must be a JSON object, got "
            f"{type(manifest).__name__}"
        )
    return manifest


def read_circuit_manifest(circuit_path: Path) -> dict[str, Any]:
    """Read the manifest beside a circuit.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit file.

    Returns
    -------
    dict
        Manifest contents.

    Raises
    ------
    FileNotFoundError
        If the circuit carries no manifest: nothing else may declare which
        experiment it belongs to.
    ValueError
        If the manifest is not a JSON object.
    """
    manifest_path = circuit_manifest_path(circuit_path)
    try:
        return read_manifest(manifest_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Circuit {circuit_path} has no manifest at {manifest_path}; a circuit "
            f"declares its experiment through write_circuit_manifest"
        ) from exc


def manifest_key(manifest: Mapping[str, Any], source: Path) -> ExperimentKey:
    """Read the experiment key a manifest declares.

    For a caller that needs the manifest's other fields too, so that it is
    read once; ``circuit_key`` is the path-only form.

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


def circuit_key(circuit_path: Path) -> ExperimentKey:
    """Resolve the experiment key of a committed circuit from its manifest.

    Parameters
    ----------
    circuit_path : Path
        Path to a ``.stim`` circuit file.

    Returns
    -------
    ExperimentKey

    Raises
    ------
    FileNotFoundError
        If the circuit carries no manifest.
    ValueError
        If the manifest is unreadable or declares an invalid key.
    """
    manifest = read_circuit_manifest(circuit_path)
    return manifest_key(manifest, circuit_manifest_path(circuit_path))


def eval_set_key(manifest: Mapping[str, Any], source: Path) -> ExperimentKey:
    """Resolve the experiment key of a frozen evaluation set.

    A set is bound to the circuit its shots were sampled from by that
    circuit's bytes: its manifest names the circuit (``circuit_file``) and
    records its ``circuit_sha256``, which the circuit must still hash to.  The
    set inherits the circuit's key; every key field its manifest also records
    must agree with it.

    Parameters
    ----------
    manifest : mapping
        Evaluation set manifest contents.
    source : Path
        Path the manifest was read from, for error messages.

    Returns
    -------
    ExperimentKey
        The key of the set's circuit.

    Raises
    ------
    FileNotFoundError
        If the circuit the set names, or that circuit's manifest, does not
        exist.
    ValueError
        If the manifest lacks ``circuit_file`` or ``circuit_sha256``, if the
        circuit's bytes do not hash to the recorded digest, or if a key field
        the set records disagrees with its circuit's.
    """
    missing = [field for field in _EVAL_SET_BINDING_FIELDS if field not in manifest]
    if missing:
        raise ValueError(
            f"Manifest {source} is missing required field(s) {missing}: a frozen "
            f"set names the circuit it was sampled from and that circuit's sha256"
        )
    circuit_path = Path(manifest["circuit_file"])
    _verify_circuit_sha256(circuit_path, manifest["circuit_sha256"], source)
    key = circuit_key(circuit_path)
    disagreeing = {
        field: manifest[field]
        for field in _KEY_FIELDS
        if field in manifest and manifest[field] != getattr(key, field)
    }
    if disagreeing:
        raise ValueError(
            f"Manifest {source} records {disagreeing}, but the circuit its shots "
            f"were sampled from, {circuit_path}, is {key}"
        )
    return key


def _verify_circuit_sha256(circuit_path: Path, recorded: str, source: Path) -> None:
    """Require the circuit's bytes to hash to the digest the set recorded."""
    try:
        actual = file_sha256(circuit_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Manifest {source} names circuit {circuit_path}, which does not exist"
        ) from exc
    if actual != recorded:
        raise ValueError(
            f"Circuit {circuit_path} hashes to {actual}, but the set of {source} "
            f"was sampled from bytes hashing to {recorded}"
        )


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

    # Atomic, so an interrupted generator leaves the previous manifest or the
    # new one. A half-written manifest would be worse than none: a truncated
    # JSON object can still parse, and would then declare an identity nobody
    # chose.
    write_json(manifest_path, manifest)
    return manifest_path
