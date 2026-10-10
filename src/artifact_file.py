"""Writing and hashing artifact files.

An artifact is written beside its destination and renamed into place, so a
reader sees either the previous file or the whole new one: a killed process
never leaves a truncated checkpoint, or a truncated JSON object that still
parses, for a later step to trust.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


__all__ = ["atomic_write", "file_sha256", "sync_directory", "write_json"]


@contextmanager
def atomic_write(path: Path) -> Iterator[Path]:
    """Yield a temporary path beside *path*, renamed onto it when the block succeeds.

    The temporary file is a hidden sibling with *path*'s suffix, so writers
    that infer a format from the suffix (``Figure.savefig``) or append one
    (``np.save``) behave as they would on *path*.  The block creates it; on
    success it is flushed to disk, renamed over *path* and the directory
    synced.  On any exception it is removed and *path* is left untouched.

    Parameters
    ----------
    path : Path
        Destination file; its parent directory is created if missing.

    Yields
    ------
    Path
        Where the block writes the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{secrets.token_hex(8)}{path.suffix}")
    try:
        yield temp
        _fsync(temp, os.O_RDONLY)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    sync_directory(path.parent)


def write_json(path: Path, payload: Any) -> None:
    """Write *payload* to *path* as indented JSON, atomically.

    Parameters
    ----------
    path : Path
        Destination file.
    payload : Any
        JSON-serialisable value.

    Raises
    ------
    ValueError
        If *payload* holds a NaN or an infinity, which JSON cannot represent.
    """
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    with atomic_write(path) as temp:
        temp.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Return the hex sha256 digest of the file at *path*."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sync_directory(path: Path) -> None:
    """Flush the entries of directory *path* to disk, making a rename in it durable."""
    _fsync(path, os.O_RDONLY | os.O_DIRECTORY)


def _fsync(path: Path, flags: int) -> None:
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
