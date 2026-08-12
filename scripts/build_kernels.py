"""Build CUDA extensions for GNN-QEC decoding.

Usage (from project root):
    python scripts/build_kernels.py build_ext --inplace
"""

import os
from pathlib import Path


# Ensure we run from project root regardless of cwd.  setuptools and the torch
# extension builder resolve paths at import time, so the chdir has to precede
# them - hence the deliberate non-top-level imports below.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

from setuptools import setup  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402


CUDA_ARCH_FLAGS = [
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_80,code=sm_80",
]

cuda_src = PROJECT_ROOT / "src" / "kernels" / "cpp"

setup(
    name="qec-cuda-ops",
    ext_modules=[
        CUDAExtension(
            name="kernels._C",
            sources=[
                str(cuda_src / "fused_edge_update.cu"),
                str(cuda_src / "fused_norm_residual.cu"),
                str(cuda_src / "graph_build.cu"),
                str(cuda_src / "bindings.cpp"),
            ],
            include_dirs=[
                str(cuda_src),
            ],
            # No --use_fast_math - see the note in src/kernels/build.py.
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",
                    "--ptxas-options=-v",
                ]
                + CUDA_ARCH_FLAGS,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
