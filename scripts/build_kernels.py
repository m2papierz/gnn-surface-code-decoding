"""Build the CUDA kernel extension in place, as ``src/kernels/_C``.

Run through ``make kernels``.  What is built - translation units, flags and
the digest the extension is stamped with - is defined by
:mod:`kernels.extension`; this script hands it to setuptools.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from kernels import extension


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    # setuptools reads pyproject.toml from the working directory, which is
    # what places the in-place build under src/kernels/.
    os.chdir(PROJECT_ROOT)
    setup(
        name="qec-cuda-ops",
        ext_modules=[
            CUDAExtension(
                name="kernels._C",
                sources=[str(path) for path in extension.translation_units()],
                include_dirs=[str(extension.SOURCE_DIR)],
                extra_compile_args={
                    # Only bindings.cpp reads the stamp; keeping it off the
                    # nvcc line leaves unchanged .cu objects to ninja's cache.
                    "cxx": [
                        *extension.CXX_FLAGS,
                        f"-DQEC_BUILD_SHA256={extension.build_sha256()}",
                    ],
                    "nvcc": list(extension.NVCC_FLAGS),
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        script_args=["build_ext", "--inplace"],
    )


if __name__ == "__main__":
    main()
