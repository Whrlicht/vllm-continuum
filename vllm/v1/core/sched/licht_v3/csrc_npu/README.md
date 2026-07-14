# LICHT NPU fused scatter

This extension provides the Ascend/NPU staged fast path for C3 round-kv loads:

```text
CPU arena / SSD -> NPU staging tensor -> paged KV cache
```

CUDA's direct arena path depends on `cudaHostRegister` and device kernels that
dereference the registered host pointer.  Ascend/CANN does not expose the same
UVA host-pointer contract here, so the NPU path keeps the arena-to-device copy
as a CANN-managed transfer and fuses only the NPU staging-to-paged-KV scatter.

Build on the Ascend machine:

```bash
cd vllm/v1/core/sched/licht_v3/csrc_npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python setup.py build_ext --inplace
```

The built module is imported as `licht_fused_scatter_npu`.
