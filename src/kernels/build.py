"""JIT build configuration for CUDA extensions.

Usage::

    python -c "from kernels.build import build; build()"

Or via setup.py for ahead-of-time compilation.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import load


_CPP = Path(__file__).parent / "cpp"

_SOURCES = [
    str(_CPP / "fused_edge_update.cu"),
    str(_CPP / "fused_norm_residual.cu"),
    str(_CPP / "graph_build.cu"),
    str(_CPP / "bindings.cpp"),
]

_CUDA_ARCH_FLAGS = [
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_80,code=sm_80",
]

_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-lineinfo",
    "--ptxas-options=-v",
] + _CUDA_ARCH_FLAGS

_CXX_FLAGS = ["-O3", "-std=c++17"]


def build(verbose: bool = True) -> object:
    """JIT-compile the CUDA extension and return the loaded module."""
    return load(
        name="_C",
        sources=_SOURCES,
        extra_include_paths=[str(_CPP)],
        extra_cflags=_CXX_FLAGS,
        extra_cuda_cflags=_NVCC_FLAGS,
        verbose=verbose,
    )
