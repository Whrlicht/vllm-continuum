# SPDX-License-Identifier: Apache-2.0
import os

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension
    import torch_npu.utils.cpp_extension as npu_cpp_extension
except Exception as exc:  # pragma: no cover - only available on Ascend hosts
    raise RuntimeError(
        "Building licht_fused_scatter_npu requires torch-npu. Run this on the "
        "Ascend environment after sourcing CANN set_env.sh.") from exc

NpuExtension = getattr(npu_cpp_extension, "NpuExtension", None)
if NpuExtension is None:  # pragma: no cover - depends on torch-npu version
    exported = ", ".join(sorted(k for k in dir(npu_cpp_extension)
                               if "Extension" in k))
    raise RuntimeError(
        "torch_npu.utils.cpp_extension does not export NpuExtension. "
        f"Available extension helpers: {exported or '<none>'}")

ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
    "ASCEND_TOOLKIT_HOME") or "/usr/local/Ascend/ascend-toolkit/latest"

include_dirs = [
    os.path.join(ASCEND_HOME, "include"),
    os.path.join(ASCEND_HOME, "include", "aclnn"),
]

include_dirs = [p for p in dict.fromkeys(include_dirs) if os.path.isdir(p)]
print("licht_fused_scatter_npu include_dirs:")
for include_dir in include_dirs:
    print("  ", include_dir)

setup(
    name="licht_fused_scatter_npu",
    version="0.1.0",
    ext_modules=[
        NpuExtension(
            name="licht_fused_scatter_npu",
            sources=[
                "fused_scatter_npu.cpp",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
