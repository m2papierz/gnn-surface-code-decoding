"""An artifact is replaced whole or not at all."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from artifact_file import atomic_write, file_sha256, write_json


class _Interrupted(Exception):
    pass


def test_the_destination_holds_the_new_file_and_no_temporary_remains(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("old", encoding="utf-8")

    with atomic_write(destination) as temp:
        temp.write_text("new", encoding="utf-8")

    assert destination.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.iterdir()) == [destination]


def test_an_interrupted_write_leaves_the_previous_file_and_no_temporary(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("old", encoding="utf-8")

    with pytest.raises(_Interrupted), atomic_write(destination) as temp:
        temp.write_text("half", encoding="utf-8")
        raise _Interrupted

    assert destination.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [destination]


def test_the_temporary_keeps_the_suffix_writers_infer_formats_from(
    tmp_path: Path,
) -> None:
    """np.save appends ``.npy`` to a path without it."""
    destination = tmp_path / "syndromes.npy"
    array = np.arange(6, dtype=np.uint8)

    with atomic_write(destination) as temp:
        np.save(temp, array)

    np.testing.assert_array_equal(np.load(destination), array)
    assert list(tmp_path.iterdir()) == [destination]


def test_missing_parent_directories_are_created(tmp_path: Path) -> None:
    destination = tmp_path / "a" / "b" / "report.json"
    write_json(destination, {"n_shots": 3})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"n_shots": 3}


def test_a_nan_is_refused_and_the_previous_file_kept(tmp_path: Path) -> None:
    """JSON has no NaN; writing one would produce a file no strict reader parses."""
    destination = tmp_path / "report.json"
    write_json(destination, {"ler": 0.5})

    with pytest.raises(ValueError):
        write_json(destination, {"ler": float("nan")})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"ler": 0.5}


def test_file_sha256_is_the_digest_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "circuit.stim"
    path.write_bytes(b"DETECTOR(0, 0, 0) rec[-1]\n")
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
