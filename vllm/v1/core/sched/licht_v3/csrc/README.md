# licht_fused_scatter — fused multi-layer block scatter CUDA op

Replaces the round-kv arena load's per-chunk `nL` (=32) `index_put`s with **one
CUDA launch** per chunk. In the busy serving process the many small
CPU-dispatched launches starve the GPU (it idles between launches under GIL
contention) → ~1.5 GB/s even though each kernel is fast. One fused launch (which
also releases the GIL) keeps the GPU fed.

## Files
- `fused_scatter.cu` — the kernel + pybind binding (module `licht_fused_scatter`).
- `setup.py` — torch `CUDAExtension` build.
- `test_fused_scatter.py` — correctness (vs the Python per-layer scatter) + speed.

## Build (you run this)
```bash
cd vllm/v1/core/sched/licht_v3/csrc
export CUDA_HOME=/usr/local/cuda-12.2     # see gotcha #1
pip install -e .                           # or: pip install .
python test_fused_scatter.py               # expect: FA/MLA PASS + KERNEL >> python
```
Then enable in serving: `--round-kv-fused` (or `LICHT_ROUND_KV_FUSED=1`).
`round_kv_store.py` imports `licht_fused_scatter`; if it's not installed it logs
a warning and falls back to the Python per-layer scatter (no crash).

## Validated
Kernel logic was verified with a JIT build (`load_inline`) on this box:
FA & MLA scatter **0/32 element mismatch** vs the Python path, and the scatter
ran **24.8 GB/s with 40 launches vs the Python 7.1 GB/s with 1280** for a
10 000-block wave (~32× fewer launches — the whole point).

## Build gotchas seen on this box (Ubuntu 20.04, conda, gcc 9.4, CUDA 12.2)
1. **CUDA_HOME**: `nvcc` is symlinked into `/bin` (`/bin/nvcc ->
   /usr/local/cuda-12.2/bin/nvcc`), so torch derives `CUDA_HOME=/` and grabs
   wrong headers (`cuda_bf16.h` missing, old `host_config.h`). **Export
   `CUDA_HOME=/usr/local/cuda-12.2`** before building (setup.py also tries to
   auto-fix this).
2. **`cannot dynamically load position-independent executable`** on import:
   Ubuntu gcc defaults to `-pie` and can stamp a `PT_INTERP` into the `.so` so
   Python refuses to `dlopen` it. If you hit this, rebuild forcing a plain
   shared object, e.g. add to the `CUDAExtension`:
   `extra_link_args=["-Wl,--no-dynamic-linker"]`  (strips the INTERP), **or**
   `LDFLAGS="-shared"` / use a non-conda gcc. Confirm with
   `file licht_fused_scatter*.so` → should say *shared object*, not *pie
   executable*. After building, `python test_fused_scatter.py` must PASS before
   enabling `--round-kv-fused` in serving.

## Op signature
```
licht_scatter(staging, idx, layer_ptrs, nb, nL, dim, NBLK, P)
  staging    : [nb, nL, 2, *rest] contiguous, 2-byte dtype (fp16/bf16)
  idx        : [nb] int64  destination block ids
  layer_ptrs : [nL] int64  each = a paged KV layer tensor's data_ptr()
  dim        : 1 FlashAttention (layer [2,NBLK,*rest]) / 0 MLA (layer [NBLK,2,*rest])
  NBLK       : blocks per layer ;  P = prod(rest)  (must be a multiple of 8)
```
