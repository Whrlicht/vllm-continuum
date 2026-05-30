# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena 共享 hdr 布局.

管理 /dev/shm/_arena.hdr 的内存布局, 初始化, 跨进程 mmap.

hdr 布局 (按 num_slots 参数化):

    Field         Offset                 Size              说明
    -----         ------                 ----              ---
    alloc_mutex   0                      MUTEX_PAD (64)    pthread_mutex_t PROCESS_SHARED + ROBUST
    free_bitmap   MUTEX_PAD              bitmap_bytes      1=free, 0=used (每 slot 一 bit)
    slot_state    bitmap_off + bytes     num_slots * 8     pin(16) | gen(48), atomic CAS
    slot_refcnt   ★ Stage 6 预留          num_slots * 2     atomic uint16 refcnt
    hash_table    ★ Stage 6 预留          HASH_CAP * 16     content-addressing index
    <padding to 1MB>

Stage 1: 只使用 alloc_mutex, free_bitmap, slot_state 区域;
         slot_refcnt 和 hash_table 区域 ftruncate 后自然 zero, 不动.
Stage 6: 启动时打开 use_content_addressing=True, 复用同一 hdr.

设计约束:
  - hdr 必须 mmap MAP_SHARED 在 /dev/shm/ 下
  - 不持久化: 进程启动时清理 hdr 文件 (ftruncate 重置)
  - 多进程: 创建者 init 一次, 其他进程仅 mmap 不 init
"""
from __future__ import annotations

import ctypes
import mmap
import os
from dataclasses import dataclass
from typing import Optional

import licht_arena_atomic as _atomic


# ============================================================
# 布局常量
# ============================================================
MUTEX_PAD_BYTES = 64        # mutex 区域 (含 padding), 缓存行对齐
PAGE_ALIGN_BYTES = 4096     # hdr 末尾 padding 到页对齐
HDR_TARGET_SIZE = 1 << 20   # 1 MB, 确保 Stage 6 预留区充足


def _round_up(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


def _next_prime_at_least(n: int) -> int:
    """找 >= n 的下一个质数 (用于 hash_table 容量, Stage 6)."""
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        if all(n % p != 0 for p in range(3, int(n ** 0.5) + 1, 2)):
            return n
        n += 2


# ============================================================
# Layout 计算
# ============================================================

@dataclass(frozen=True)
class ArenaHdrLayout:
    """计算并暴露 hdr 内各字段的偏移和大小.

    所有 offset 都是相对 hdr 起点的字节偏移.
    """
    num_slots: int

    # 计算字段 (post-init)
    bitmap_offset: int = 0
    bitmap_bytes: int = 0
    slot_state_offset: int = 0
    slot_state_bytes: int = 0
    slot_refcnt_offset: int = 0       # ★ Stage 6
    slot_refcnt_bytes: int = 0         # ★ Stage 6
    hash_table_capacity: int = 0       # ★ Stage 6
    hash_table_offset: int = 0         # ★ Stage 6
    hash_table_bytes: int = 0          # ★ Stage 6
    total_used_bytes: int = 0
    total_size: int = 0                # ftruncate 目标 (页对齐)

    @classmethod
    def compute(cls, num_slots: int) -> "ArenaHdrLayout":
        assert num_slots > 0
        # alloc_mutex 占 [0, MUTEX_PAD_BYTES)
        bitmap_off = MUTEX_PAD_BYTES
        # 每 64 个 slot 一个 uint64 word (1 bit / slot)
        n_words = (num_slots + 63) // 64
        bitmap_bytes = n_words * 8
        # 对齐到 8 字节 (实际已经是)
        slot_state_off = bitmap_off + bitmap_bytes
        slot_state_bytes = num_slots * 8

        # ★ Stage 6 预留
        slot_refcnt_off = slot_state_off + slot_state_bytes
        slot_refcnt_bytes = num_slots * 2
        # 对齐到 8 字节
        slot_refcnt_bytes_aligned = _round_up(slot_refcnt_bytes, 8)

        hash_cap = _next_prime_at_least(2 * num_slots)
        # 每 entry 16 字节: hash(8) + slot_id(4) + epoch(4)
        hash_table_off = slot_refcnt_off + slot_refcnt_bytes_aligned
        hash_table_bytes = hash_cap * 16

        total_used = hash_table_off + hash_table_bytes
        # 取 max(总用量, 1 MB), 然后页对齐
        total = _round_up(max(total_used, HDR_TARGET_SIZE), PAGE_ALIGN_BYTES)

        return cls(
            num_slots=num_slots,
            bitmap_offset=bitmap_off,
            bitmap_bytes=bitmap_bytes,
            slot_state_offset=slot_state_off,
            slot_state_bytes=slot_state_bytes,
            slot_refcnt_offset=slot_refcnt_off,
            slot_refcnt_bytes=slot_refcnt_bytes,
            hash_table_capacity=hash_cap,
            hash_table_offset=hash_table_off,
            hash_table_bytes=hash_table_bytes,
            total_used_bytes=total_used,
            total_size=total,
        )

    def slot_state_addr(self, base_addr: int, slot_id: int) -> int:
        """slot_state[slot_id] 的字节地址."""
        assert 0 <= slot_id < self.num_slots
        return base_addr + self.slot_state_offset + slot_id * 8

    def bitmap_word_addr(self, base_addr: int, word_idx: int) -> int:
        """bitmap[word_idx] (uint64) 的字节地址."""
        assert 0 <= word_idx * 64 < self.num_slots
        return base_addr + self.bitmap_offset + word_idx * 8

    def mutex_addr(self, base_addr: int) -> int:
        return base_addr  # offset 0


# ============================================================
# Hdr 创建 / 打开
# ============================================================

class ArenaHdr:
    """封装 hdr mmap + 字段访问.

    使用模式:
        hdr = ArenaHdr.create(path, num_slots=12000)   # 首次创建
        # or
        hdr = ArenaHdr.open(path, num_slots=12000)     # 重 mmap
        ...
        hdr.close()

    创建语义:
      - create(): ftruncate 重置文件大小 + 清零 + init mutex
      - open(): 仅 mmap, 不修改内容
    """
    def __init__(self,
                 mmap_obj: mmap.mmap,
                 layout: ArenaHdrLayout,
                 base_addr: int,
                 path: str):
        self._mmap = mmap_obj
        self._layout = layout
        self._base_addr = base_addr
        self._path = path
        self._closed = False

    @classmethod
    def create(cls, path: str, num_slots: int) -> "ArenaHdr":
        """首次创建 hdr: ftruncate + 清零 + init mutex."""
        layout = ArenaHdrLayout.compute(num_slots)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            # ftruncate 总是清零新增加的页, 即使文件已存在我们也强制 resize 一次
            os.ftruncate(fd, 0)
            os.ftruncate(fd, layout.total_size)
            mm = mmap.mmap(fd, layout.total_size,
                           mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        rc = _atomic.mutex_init(layout.mutex_addr(base))
        if rc != 0:
            mm.close()
            raise OSError(rc, f"arena_mutex_init failed: rc={rc}")
        return cls(mm, layout, base, path)

    @classmethod
    def open(cls, path: str, num_slots: int) -> "ArenaHdr":
        """打开已存在 hdr (其他进程已 create), 仅 mmap."""
        layout = ArenaHdrLayout.compute(num_slots)
        fd = os.open(path, os.O_RDWR)
        try:
            st = os.fstat(fd)
            if st.st_size < layout.total_size:
                raise ValueError(
                    f"hdr too small: {st.st_size} < {layout.total_size}")
            mm = mmap.mmap(fd, layout.total_size,
                           mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        return cls(mm, layout, base, path)

    @property
    def layout(self) -> ArenaHdrLayout:
        return self._layout

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def mutex_addr(self) -> int:
        return self._layout.mutex_addr(self._base_addr)

    def slot_state_addr(self, slot_id: int) -> int:
        return self._layout.slot_state_addr(self._base_addr, slot_id)

    def bitmap_word_addr(self, word_idx: int) -> int:
        return self._layout.bitmap_word_addr(self._base_addr, word_idx)

    # ---- bitmap 操作 (注意: 必须在 alloc_mutex 内调用, 不需要原子) ----
    def bitmap_set_free(self, slot_id: int) -> None:
        assert 0 <= slot_id < self._layout.num_slots
        word_idx = slot_id // 64
        bit_idx = slot_id % 64
        addr = self.bitmap_word_addr(word_idx)
        cur = _atomic.atomic_load_u64(addr)
        _atomic.atomic_store_u64(addr, cur | (1 << bit_idx))

    def bitmap_set_used(self, slot_id: int) -> None:
        assert 0 <= slot_id < self._layout.num_slots
        word_idx = slot_id // 64
        bit_idx = slot_id % 64
        addr = self.bitmap_word_addr(word_idx)
        cur = _atomic.atomic_load_u64(addr)
        _atomic.atomic_store_u64(addr, cur & ~(1 << bit_idx))

    def bitmap_is_free(self, slot_id: int) -> bool:
        assert 0 <= slot_id < self._layout.num_slots
        word_idx = slot_id // 64
        bit_idx = slot_id % 64
        addr = self.bitmap_word_addr(word_idx)
        return bool((_atomic.atomic_load_u64(addr) >> bit_idx) & 1)

    def bitmap_init_all_free(self) -> None:
        """启动时把所有 slot 标 free (1)."""
        n_words = (self._layout.num_slots + 63) // 64
        # 完整 word 全置 1
        for w in range(n_words - 1):
            _atomic.atomic_store_u64(self.bitmap_word_addr(w),
                                     0xFFFFFFFFFFFFFFFF)
        # 最后一个 word 可能不满 64 bit, 只置低 N bit
        rem_bits = self._layout.num_slots - (n_words - 1) * 64
        last_mask = (1 << rem_bits) - 1
        _atomic.atomic_store_u64(self.bitmap_word_addr(n_words - 1), last_mask)

    def count_free(self) -> int:
        """O(num_slots / 64), 仅在 alloc_mutex 内调用."""
        n_words = (self._layout.num_slots + 63) // 64
        total = 0
        for w in range(n_words):
            total += bin(_atomic.atomic_load_u64(
                self.bitmap_word_addr(w))).count("1")
        return total

    # ---- Lifecycle ----
    def destroy_mutex(self) -> None:
        """销毁 mutex. 通常仅在 owner 进程退出时调用一次."""
        _atomic.mutex_destroy(self.mutex_addr)

    def close(self) -> None:
        if not self._closed:
            self._mmap.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
