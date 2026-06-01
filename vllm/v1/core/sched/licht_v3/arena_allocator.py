# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena slot-paged allocator.

负责:
  - alloc_n_slots(n): first-fit 扫 bitmap 找 n 个 free slot, 标记 used 返回 slot_id 列表
  - free_n_slots(slot_ids): 标记一组 slot 为 free
  - free_count: 缓存的空闲 slot 数 (alloc/free 时维护)

设计约束:
  - 所有 alloc/free 操作必须由调用者持 alloc_mutex
  - 本类不实现 evict 逻辑 (在 arena_lru_store 里)
  - bitmap 用 atomic fetch_or/fetch_and 即使 mutex 外调用也安全

性能:
  - alloc_n_slots O(num_slots / 64) 扫 bitmap
  - 12000 slot 实测 ~30us (Python 循环)
  - 如成瓶颈, Stage 2.x 可在 C++ 侧用 ctz/popcount intrinsics 加速
"""
from __future__ import annotations

from typing import List

import licht_arena_atomic as _atomic

from vllm.v1.core.sched.licht_v3.arena_hdr import ArenaHdr


class ArenaAllocator:
    """slot-paged bitmap allocator.

    使用模式 (在 alloc_mutex 内):
        slot_ids = allocator.alloc_n(n)
        if slot_ids is None:
            # 空间不够, 调用者触发 evict 后重试
            ...
        # 写数据 + publish gen
        allocator.free_n(slot_ids)  # 释放时
    """
    def __init__(self, hdr: ArenaHdr):
        self._hdr = hdr
        self._layout = hdr.layout
        self._n_words = (self._layout.num_slots + 63) // 64
        # 缓存的空闲 slot 数, 在 alloc/free 时维护
        # 启动时由 init_all_free() 或 sync_from_bitmap() 设置
        self._free_count = 0

    # ============================================================
    # Init / sync
    # ============================================================
    def init_all_free(self) -> None:
        """初始化: 所有 slot 标 free, 缓存计数到 num_slots."""
        self._hdr.bitmap_init_all_free()
        self._free_count = self._layout.num_slots

    def sync_from_bitmap(self) -> None:
        """从 bitmap 重新算 free_count. 仅在状态不确定时调用."""
        self._free_count = self._hdr.count_free()

    @property
    def free_count(self) -> int:
        """本进程视角的空闲 slot 近似数 (日志/监控用).

        警告: 跨进程双 writer (Phase 2) 下不准 — 只累计本进程的 alloc/free,
        看不到对方进程的操作. alloc_n 已不依赖它做决策 (直接扫共享 bitmap).
        需要准确值用 count_free_accurate().
        """
        return self._free_count

    def count_free_accurate(self) -> int:
        """实时扫共享 bitmap 得到准确的空闲 slot 数 (跨进程真相).
        O(num_slots/64) popcount; 需持 alloc_mutex 保证扫描期间 bitmap 不变."""
        return self._hdr.count_free()

    @property
    def num_slots(self) -> int:
        return self._layout.num_slots

    # ============================================================
    # Alloc / Free (必须持 alloc_mutex)
    # ============================================================
    def alloc_n(self, n: int) -> List[int] | None:
        """First-fit 扫共享 bitmap 找 n 个 free slot.

        成功: 把这些 slot 标 used, 返回 slot_id 列表
        失败 (扫完 bitmap 不够 n 个 free): 返回 None, 不做任何修改

        正确性 (跨进程双 writer, 如 Phase 2 prefill+decode 同写 arena):
          唯一真相源是共享 bitmap (受 alloc_mutex 保护). 本方法直接扫 bitmap
          判断够不够 + 收集 slot_id, 不依赖进程本地的 _free_count 缓存 (该缓存
          只反映本进程的 alloc/free, 看不到对方进程的操作, 双进程下必然失真).
          扫描"找 slot 位置"本来就省不掉 (free_count 只知总数不知哪些 free),
          所以把"够不够"判断合并进这次扫描零额外开销.

        注意: 不需要"连续"的 n 个 slot, 任意 n 个 free 都可以.
        """
        if n <= 0:
            return []

        out: List[int] = []
        # 扫共享 bitmap, 逐 word 处理; 扫够 n 个即停 (不信本地缓存)
        for word_idx in range(self._n_words):
            if len(out) == n:
                break
            word_addr = self._hdr.bitmap_word_addr(word_idx)
            word = _atomic.atomic_load_u64(word_addr)
            if word == 0:
                continue  # 全 used, 跳过
            base_slot = word_idx * 64
            n_bits = min(64, self._layout.num_slots - base_slot)
            for bit_idx in range(n_bits):
                if (word >> bit_idx) & 1:
                    out.append(base_slot + bit_idx)
                    if len(out) == n:
                        break

        if len(out) < n:
            # bitmap 里 free slot 真的不够 n 个 -> 上层应 evict 后重试.
            # (此时 out 里的 slot 还没 mark used, 无需回滚.)
            return None

        # 扫够了, 统一 mark used (失败已在上面 return, 不会半标记)
        for slot_id in out:
            self._hdr.bitmap_set_used(slot_id)
        # _free_count 降级为本进程视角的近似计数 (日志/监控用, alloc 决策已不依赖它)
        self._free_count -= n
        return out

    def free_n(self, slot_ids: List[int]) -> None:
        """把一组 slot 标 free, 同步 free_count.

        调用者已经在 alloc_mutex 内, 且已经 evict_slot 把 gen 加好.
        """
        for slot_id in slot_ids:
            self._hdr.bitmap_set_free(slot_id)
        self._free_count += len(slot_ids)

    def is_free(self, slot_id: int) -> bool:
        """诊断用: 查询某 slot 是否 free."""
        return self._hdr.bitmap_is_free(slot_id)
