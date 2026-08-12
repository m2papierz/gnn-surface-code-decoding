"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch


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
