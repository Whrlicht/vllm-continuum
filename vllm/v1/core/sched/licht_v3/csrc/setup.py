# SPDX-License-Identifier: Apache-2.0
# Build the fused round-kv scatter CUDA op as an installable extension.
#
#   cd vllm/v1/core/sched/licht_v3/csrc
#   export CUDA_HOME=/usr/local/cuda-12.2     # see note below
#   pip install .            # or: python setup.py install
#
# NOTE on CUDA_HOME: torch derives it from `which nvcc`; if nvcc is symlinked
# into /bin (as here: /bin/nvcc -> /usr/local/cuda-12.2/bin/nvcc) torch derives
# "/" and picks up wrong headers.  Set CUDA_HOME to the real toolkit root.
# We also auto-fix it below if it's unset.
import os
import shutil

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

if not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
    nvcc = shutil.which("nvcc")
    if nvcc:
        os.environ["CUDA_HOME"] = os.path.dirname(
            os.path.dirname(os.path.realpath(nvcc)))

setup(
    name="licht_fused_scatter",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="licht_fused_scatter",
            sources=["fused_scatter.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
