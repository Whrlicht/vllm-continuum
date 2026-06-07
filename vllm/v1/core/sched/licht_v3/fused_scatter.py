# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused multi-layer block-scatter CUDA op for the round-kv arena load.

The arena load's scatter writes a chunk's blocks into the paged KV cache.  The
naive path issues nL (=32) `index_put`s PER CHUNK; in the busy serving process
those many small CPU-dispatched launches starve the GPU (it idles waiting for
the next launch under GIL contention) -> ~1.5 GB/s despite the kernels being
fast.  This op does the WHOLE chunk's nL layers in ONE launch (and releases the
GIL during execution), cutting per-chunk dispatches nL->1.

Compiled lazily via cpp_extension.load_inline (needs nvcc; the build env has
it).  On any failure, returns None and the caller falls back to the Python
per-layer scatter.

Layout (matches round_kv_store block-major staging):
  staging : [nb, nL, 2, *rest] contiguous (fp16/bf16, 2-byte)
  idx     : [nb] int64  destination block ids
  layer_ptrs : [nL] int64  each = a paged KV layer tensor's data_ptr()
  FA  (dim==1, layer [2, NBLK, *rest]):  dst off = (kv*NBLK + blk)*P + r
  MLA (dim==0, layer [NBLK, 2, *rest]):  dst off = (blk*2  + kv)*P + r
  P = prod(rest); requires P % 8 == 0 (int4-vectorized) and 2-byte dtype.
"""
from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# The kernel is an ahead-of-time CUDA extension (see csrc/fused_scatter.cu).
# Build + install it once:
#     cd vllm/v1/core/sched/licht_v3/csrc
#     export CUDA_HOME=/usr/local/cuda-12.2   # real toolkit (nvcc is /bin-symlinked)
#     pip install .
_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")

_fn = None
_tried = False


def get_scatter():
    """Return the prebuilt `licht_scatter` callable, or None if the extension
    is not installed (caller then falls back to the Python per-layer scatter).
    Looks up cached result after the first call."""
    global _fn, _tried
    if _tried:
        return _fn
    _tried = True
    try:
        import licht_fused_scatter as _ext  # built from csrc/ via setup.py
        _fn = _ext.licht_scatter
        logger.info("round-kv FUSED scatter: using prebuilt "
                    "licht_fused_scatter extension")
    except Exception as e:
        logger.warning(
            "round-kv FUSED scatter: extension not available (%s). Build it: "
            "cd %s && export CUDA_HOME=/usr/local/cuda-12.2 && pip install . "
            "-- falling back to per-layer index_put.", e, _CSRC_DIR)
        _fn = None
    return _fn


_fn_arena = None
_tried_arena = False


def get_arena_scatter():
    """Return the prebuilt `licht_scatter_from_arena` callable, or None if the
    extension is not installed / too old (caller then falls back to the
    staging-based or per-layer path).

    This is the DIRECT load path: source is the cudaHostRegister'd shared
    arena (host pinned), scattered straight into the paged buffer in one
    launch — NO GPU staging buffer, NO extra HBM round-trip.
    Signature: fn(arena_host_ptr, src_slots, dst_idx, layer_ptrs,
                  nb, nL, dim, NBLK, P)
    """
    global _fn_arena, _tried_arena
    if _tried_arena:
        return _fn_arena
    _tried_arena = True
    try:
        import licht_fused_scatter as _ext
        _fn_arena = _ext.licht_scatter_from_arena
        logger.info("round-kv DIRECT arena scatter: using prebuilt "
                    "licht_fused_scatter.licht_scatter_from_arena")
    except (ImportError, AttributeError) as e:
        # AttributeError: old .so without the new symbol -> needs rebuild.
        logger.warning(
            "round-kv DIRECT arena scatter unavailable (%s); rebuild csrc/ "
            "to get licht_scatter_from_arena. Falling back.", e)
        _fn_arena = None
    return _fn_arena


_fn_arena_layer = None
_tried_arena_layer = False


def get_arena_scatter_layer():
    """Return the prebuilt `licht_scatter_from_arena_layer` callable, or None.

    Per-layer 流水线版: 只 scatter 单层 (固定 layer_idx) -> 单层 paged ptr.
    用于逐层加载与 forward 重叠. 旧 .so 没这符号 -> None, 调用方回退批量直读.
    Signature: fn(arena_host_ptr, src_slots, dst_idx, layer_ptr,
                  nb, nL, layer_idx, dim, NBLK, P)
    """
    global _fn_arena_layer, _tried_arena_layer
    if _tried_arena_layer:
        return _fn_arena_layer
    _tried_arena_layer = True
    try:
        import licht_fused_scatter as _ext
        _fn_arena_layer = _ext.licht_scatter_from_arena_layer
        logger.info("round-kv PIPELINE per-layer arena scatter: using prebuilt "
                    "licht_fused_scatter.licht_scatter_from_arena_layer")
    except (ImportError, AttributeError) as e:
        logger.warning(
            "round-kv per-layer arena scatter unavailable (%s); rebuild csrc/. "
            "Pipeline falls back to batched direct.", e)
        _fn_arena_layer = None
    return _fn_arena_layer


_fn_gather_layer = None
_tried_gather_layer = False


def get_arena_gather_layer():
    """Return the prebuilt `licht_gather_to_arena_layer` callable, or None.

    Per-layer 直写 (store): 单层 paged 块 -> arena host pinned slot. GPU kernel
    经 PCIe 直写, 省掉 D2H gather + CPU memcpy. 旧 .so 没此符号 -> None, 调用方
    回退旧 gather+memcpy 存路径.
    Signature: fn(arena_host_ptr, dst_slots, src_idx, layer_ptr,
                  nb, nL, layer_idx, dim, NBLK, P)
    """
    global _fn_gather_layer, _tried_gather_layer
    if _tried_gather_layer:
        return _fn_gather_layer
    _tried_gather_layer = True
    try:
        import licht_fused_scatter as _ext
        _fn_gather_layer = _ext.licht_gather_to_arena_layer
        logger.info("round-kv STORE-DIRECT per-layer arena write: using prebuilt "
                    "licht_fused_scatter.licht_gather_to_arena_layer")
    except (ImportError, AttributeError) as e:
        logger.warning(
            "round-kv per-layer arena write unavailable (%s); rebuild csrc/. "
            "Store falls back to gather+memcpy.", e)
        _fn_gather_layer = None
    return _fn_gather_layer
