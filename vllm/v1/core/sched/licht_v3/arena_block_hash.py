# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena 内容寻址: 前缀链式 block hash (Stage 6).

链式定义:
    h[-1] = SEED0
    h[i]  = block_hash(h[i-1], pack(token_ids[i*bs : (i+1)*bs]))

性质:
    - h[i] 由整个 [0, i] 前缀 token 序列唯一决定 (seed 携带前缀, 碰撞 ~2^-64)
    - 两个请求只有前缀完全相同时, 对应 block 的 h 才会逐块相等, 在首个分叉块 diverge
    - 跨进程确定性: token 打包成 uint32 LE, block_size 由 chunk 长度隐式编码

约定:
    - 只对"完整 block"算 hash (n_blocks = len // block_size); 尾部不足一 block 的
      token 忽略 (与 arena 的 block 粒度一致)
    - token_id 非负且 < 2^32 (词表远小于此), 用 uint32 LE 打包
"""
from __future__ import annotations

import struct
from typing import List, Sequence

import licht_arena_atomic as _atomic

# 链式起点. 固定常量, 跨进程必须一致. 改它会让所有已存 hash 失效.
SEED0 = 0x9E3779B97F4A7C15  # 任取的 magic (golden ratio), 非 0 防退化


def _pack(chunk: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(chunk)}I", *chunk)


def block_hashes(token_ids: Sequence[int],
                 block_size: int,
                 n_blocks: int | None = None) -> List[int]:
    """链式算前 n_blocks 个完整 block 的 hash.

    返回 list[int] 长度 = n_blocks; out[i] = block i 的 hash.
    out[i] 同时是"算 block i+1 时的 prev", 故 out[-1] 可用于续链.
    """
    if block_size <= 0:
        return []
    if n_blocks is None:
        n_blocks = len(token_ids) // block_size
    n_blocks = min(n_blocks, len(token_ids) // block_size)
    out: List[int] = []
    prev = SEED0
    for i in range(n_blocks):
        chunk = token_ids[i * block_size:(i + 1) * block_size]
        prev = _atomic.block_hash(prev, _pack(chunk))
        out.append(prev)
    return out


def prefix_hash(token_ids: Sequence[int],
                block_size: int,
                upto_blocks: int) -> int:
    """算到第 upto_blocks 个 block 末尾的链式 prev (用于跨 job 续链 lookup).

    upto_blocks==0 返回 SEED0. 否则返回 block (upto_blocks-1) 的 hash.
    """
    if upto_blocks <= 0:
        return SEED0
    hs = block_hashes(token_ids, block_size, upto_blocks)
    return hs[-1] if hs else SEED0
