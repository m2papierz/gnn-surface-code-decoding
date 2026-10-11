"""The compiled kernel extension: what it is built from, and loading it.

``kernels._C`` is a build artifact outside version control, so nothing ties a
given binary to the sources beside it.  The build therefore stamps the
extension with :func:`build_sha256` - a digest of the compile flags and every
file under ``cpp/`` - and :func:`load` refuses a binary whose stamp differs
from the digest of the sources on disk.  A kernel edited without a rebuild
fails to load instead of letting the equivalence tests pass against the code
it replaced.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Final

# `_C` links against libc10 and libtorch, which only importing torch loads;
# without it the import fails with `ImportError: libc10.so` on a sound build.
import torch  # noqa: F401

from artifact_file import file_sha256


__all__ = [
    "BUILD_COMMAND",
    "CXX_FLAGS",
    "NVCC_FLAGS",
    "SOURCE_DIR",
    "build_sha256",
    "load",
    "translation_units",
]

BUILD_COMMAND: Final = "make kernels"

SOURCE_DIR: Final = Path(__file__).resolve().parent / "cpp"

CXX_FLAGS: Final = ("-O3", "-std=c++17")

# No --use_fast_math: it lets nvcc contract and approximate float arithmetic,
# and the graph-build kernels are bit-identical to the numpy builder only under
# IEEE round-to-nearest.
NVCC_FLAGS: Final = (
    "-O3",
    "-std=c++17",
    "-lineinfo",
    "--ptxas-options=-v",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_80,code=sm_80",
)

_TRANSLATION_UNIT_SUFFIXES: Final = frozenset({".cu", ".cpp"})
_SOURCE_SUFFIXES: Final = _TRANSLATION_UNIT_SUFFIXES | {".h"}


def translation_units() -> list[Path]:
    """Return every ``.cu`` and ``.cpp`` file under ``cpp/``, in name order."""
    return _source_files(_TRANSLATION_UNIT_SUFFIXES)


def build_sha256() -> str:
    """Return the digest identifying an extension built from the current tree.

    Covers both flag sets and every translation unit and header under
    ``cpp/``, each by name and content, so a change to any of them - or a file
    added or removed - yields a different digest.

    Returns
    -------
    str
        Hex sha256 digest.
    """
    flags = json.dumps({"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS})
    digest = hashlib.sha256(flags.encode())
    for path in _source_files(_SOURCE_SUFFIXES):
        digest.update(f"\0{path.name}\0{file_sha256(path)}".encode())
    return digest.hexdigest()


def _source_files(suffixes: frozenset[str]) -> list[Path]:
    """Return the files under ``cpp/`` with one of *suffixes*, in name order."""
    return [path for path in sorted(SOURCE_DIR.iterdir()) if path.suffix in suffixes]


@functools.cache
def load() -> ModuleType:
    """Import ``kernels._C``, refusing a build of anything but the current tree.

    Returns
    -------
    ModuleType
        The extension module.

    Raises
    ------
    ImportError
        If the extension is not built, does not load (an undefined symbol, a
        missing shared library), or was built from other sources or flags.
    """
    try:
        # Imported by name: the module exists only after a build, and a
        # static import would make type-checking depend on one.
        extension = importlib.import_module("kernels._C")
    except ImportError as exc:
        raise ImportError(
            f"the CUDA kernel extension does not load ({exc}); "
            f"build it with `{BUILD_COMMAND}`"
        ) from exc

    expected = build_sha256()
    built = getattr(extension, "build_sha256", None)
    if built != expected:
        raise ImportError(
            f"the CUDA kernel extension at {extension.__file__} was built from "
            f"other sources or flags (build {built}, tree {expected}); "
            f"rebuild it with `{BUILD_COMMAND}`"
        )
    return extension
