# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-aware fused scatter loaders for round-kv arena load/store.

CUDA has an existing extension in csrc/ that includes a direct host-pinned
arena path.  Ascend/NPU uses a separate extension name and intentionally does
not claim CUDA's host-pointer direct path: the NPU fast path is
CPU arena -> NPU staging -> Ascend C fused scatter into paged KV.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_CUDA_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
_NPU_CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc_npu")

_cache: dict[tuple[str, str], Any | None] = {}
_tried: set[tuple[str, str]] = set()


def _normalize_backend(device_type: str | None) -> str:
    if device_type in ("npu", "ascend"):
        return "npu"
    return "cuda"


def _get_symbol(backend: str, symbol: str, *, direct: bool = False):
    key = (backend, symbol)
    if key in _tried:
        return _cache.get(key)
    _tried.add(key)
    try:
        if backend == "npu":
            if _NPU_CSRC_DIR not in sys.path:
                sys.path.insert(0, _NPU_CSRC_DIR)
            import licht_fused_scatter_npu as _ext
            if direct:
                logger.warning(
                    "round-kv NPU direct arena symbol %s requested, but "
                    "Ascend does not use CUDA host-pointer direct access; "
                    "falling back to staged NPU scatter.", symbol)
                _cache[key] = None
                return None
        else:
            if _CUDA_CSRC_DIR not in sys.path:
                sys.path.insert(0, _CUDA_CSRC_DIR)
            import licht_fused_scatter as _ext
        fn = getattr(_ext, symbol)
        _cache[key] = fn
        logger.info("round-kv %s fused scatter: using %s.%s",
                    backend.upper(), _ext.__name__, symbol)
        return fn
    except Exception as e:
        if backend == "npu":
            logger.warning(
                "round-kv NPU fused scatter symbol %s unavailable (%s). "
                "Build it in the Ascend environment: cd %s && python "
                "setup.py build_ext --inplace. Falling back to Python "
                "per-layer scatter.", symbol, e, _NPU_CSRC_DIR)
        else:
            logger.warning(
                "round-kv CUDA fused scatter symbol %s unavailable (%s). "
                "Build it: cd %s && export CUDA_HOME=/usr/local/cuda-12.2 "
                "&& pip install . Falling back to Python per-layer scatter.",
                symbol, e, _CUDA_CSRC_DIR)
        _cache[key] = None
        return None


def get_scatter(device_type: str | None = None):
    """Return staged fused scatter for the requested backend, or None."""
    backend = _normalize_backend(device_type)
    return _get_symbol(backend, "licht_scatter")


def get_arena_scatter(device_type: str | None = None):
    """Return CUDA direct host-pinned arena scatter, or None.

    This direct symbol is CUDA-only.  NPU must use the staged NPU scatter after
    host/CPU arena data is copied to NPU memory.
    """
    backend = _normalize_backend(device_type)
    return _get_symbol(backend, "licht_scatter_from_arena", direct=True)


def get_arena_scatter_layer(device_type: str | None = None):
    """Return CUDA per-layer direct arena scatter, or None."""
    backend = _normalize_backend(device_type)
    return _get_symbol(backend, "licht_scatter_from_arena_layer",
                       direct=True)


def get_arena_gather_layer(device_type: str | None = None):
    """Return CUDA per-layer direct paged-KV -> arena writer, or None."""
    backend = _normalize_backend(device_type)
    return _get_symbol(backend, "licht_gather_to_arena_layer", direct=True)
