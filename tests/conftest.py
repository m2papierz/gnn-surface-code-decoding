"""Shared pytest fixtures."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sampling.experiment import circuit_key, write_circuit_manifest


REPO_ROOT = Path(__file__).resolve().parent.parent


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
