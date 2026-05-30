# SPDX-License-Identifier: Apache-2.0
"""Build for licht_arena_atomic C++ extension.

独立扩展, 与 ../csrc/fused_scatter 完全解耦.

Usage:
    cd vllm/v1/core/sched/licht_v3/csrc_arena
    pip install .

构建说明:
    - 文件后缀用 .cu 但不含 CUDA kernel, 只是为了统一 torch 扩展构建流程
    - xxhash 用 vendored single-header (third_party/xxhash.h, Stage 6 才用)
    - 不需要全局 CUDA_HOME, setup.py 内部找
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 自动定位 CUDA, 优先用 nvcc 所在路径
_cuda_home = os.environ.get("CUDA_HOME")
if not _cuda_home:
    for candidate in ("/usr/local/cuda-12.2", "/usr/local/cuda"):
        if os.path.isdir(candidate):
            _cuda_home = candidate
            break
if _cuda_home:
    os.environ["CUDA_HOME"] = _cuda_home


setup(
    name="licht_arena_atomic",
    version="0.1.0",
    description=(
        "LICHT Round-KV Arena cross-process atomic primitives "
        "(mutex, pin/gen CAS, atomic load/store)"
    ),
    ext_modules=[
        CUDAExtension(
            name="licht_arena_atomic",
            sources=["arena_atomic.cu"],
            include_dirs=["./third_party"],  # 含 xxhash.h
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                    "-pthread",
                    "-Wall",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                ],
            },
            extra_link_args=["-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
