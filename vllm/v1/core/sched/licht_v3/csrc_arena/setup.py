# SPDX-License-Identifier: Apache-2.0
"""Build for licht_arena_atomic C++ extension.

独立扩展, 与 ../csrc/fused_scatter 完全解耦.

Usage:
    cd vllm/v1/core/sched/licht_v3/csrc_arena
    pip install --no-build-isolation .

构建说明:
    - 本扩展不含 CUDA kernel, 用 CppExtension (纯 host C++)
    - xxhash 用 vendored single-header (third_party/xxhash.h, Stage 6 才用)
    - 必须 --no-build-isolation, 因为依赖外部已装的 torch
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# 绝对路径: ninja 编译时 cwd 在 build 临时目录, 相对 "./third_party" 解析不到
_HERE = os.path.dirname(os.path.abspath(__file__))
_THIRD_PARTY = os.path.join(_HERE, "third_party")


setup(
    name="licht_arena_atomic",
    version="0.1.0",
    description=(
        "LICHT Round-KV Arena cross-process atomic primitives "
        "(mutex, pin/gen CAS, atomic load/store)"
    ),
    ext_modules=[
        CppExtension(
            name="licht_arena_atomic",
            sources=[os.path.join(_HERE, "arena_atomic.cpp")],
            include_dirs=[_THIRD_PARTY],  # 含 xxhash.h (绝对路径)
            extra_compile_args=[
                "-O3",
                "-std=c++17",
                "-pthread",
                "-Wall",
            ],
            extra_link_args=["-pthread"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
