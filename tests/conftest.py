"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

from kernels import extension
from sampling.experiment import circuit_key, write_circuit_manifest


REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_report_header() -> str:
    """Name the device the CUDA tests ran on, or that none did."""
    if not torch.cuda.is_available():
        return "cuda: no device - kernel tests skip"
    major, minor = torch.cuda.get_device_capability()
    return (
        f"cuda: {torch.cuda.get_device_name()} (sm_{major}{minor}), "
        f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}"
    )


@pytest.fixture(autouse=True)
def isolate_float32_matmul_precision() -> Iterator[None]:
    """Keep one test's float32 matmul precision from reaching the next.

    ``build_training_state`` enables TF32 process-wide when it sees a CUDA
    device.  Without this fixture that setting leaks out of the trainer tests
    and silently degrades every later float32 comparison - TF32 matmul carries
    roughly 1e-3 relative error, so backend-equivalence assertions would be
    measuring the precision mode rather than the backends.
    """
    previous = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.fixture(scope="session")
def cuda_extension() -> ModuleType:
    """The compiled kernel extension, for tests that run the CUDA kernels.

    Skips only when the machine has no CUDA device.  The load itself is not
    guarded: where a device exists, an extension that is missing, stale or
    fails to load errors every test that requests it with the ImportError,
    so a broken build cannot pass as a run of skipped tests.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return extension.load()


@pytest.fixture
def two_observable_memory_root(tmp_path: Path) -> Path:
    """A circuit root whose one circuit declares memory but has two observables.

    A committed merge-split circuit under a memory manifest: discovery for
    memory admits it, so only extraction can refuse it.
    """
    source = min((REPO_ROOT / "data/circuits/zz_merge_split").glob("*.stim"))
    root = tmp_path / "memory"
    root.mkdir()
    circuit_path = root / source.name
    shutil.copyfile(source, circuit_path)
    write_circuit_manifest(
        circuit_path, replace(circuit_key(source), operation="memory"), provenance={}
    )
    return root
