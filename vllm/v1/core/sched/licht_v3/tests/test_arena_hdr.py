# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena 共享 hdr 布局测试.

测试矩阵:

Group A - Layout 计算:
  - test_layout_basic
  - test_layout_alignment
  - test_layout_stage6_reserved_present
  - test_layout_total_size_at_least_1MB

Group B - Hdr 创建 / 打开:
  - test_create_and_close
  - test_create_then_open_same_layout
  - test_mutex_works_after_create

Group C - Bitmap:
  - test_bitmap_init_all_free
  - test_bitmap_set_used_set_free
  - test_bitmap_partial_last_word
  - test_count_free

Group D - Slot state 地址:
  - test_slot_state_addr_distinct_per_slot
  - test_slot_state_through_atomic_ops
"""
import os
import tempfile

import pytest

try:
    import licht_arena_atomic as A
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.arena_hdr import (
    ArenaHdr,
    ArenaHdrLayout,
    HDR_TARGET_SIZE,
    MUTEX_PAD_BYTES,
    PAGE_ALIGN_BYTES,
)


@pytest.fixture
def tmp_hdr_path(tmp_path):
    """每个测试一个独立的临时 hdr 文件."""
    return str(tmp_path / "test_arena.hdr")


# ============================================================
# Group A - Layout 计算
# ============================================================

class TestGroupALayout:
    def test_layout_basic(self):
        layout = ArenaHdrLayout.compute(num_slots=12000)
        assert layout.num_slots == 12000
        # bitmap: 12000 bit -> 188 word (12032 bit), 1504 byte
        assert layout.bitmap_bytes == 188 * 8
        # slot_state: 12000 * 8
        assert layout.slot_state_bytes == 12000 * 8

    def test_layout_alignment(self):
        layout = ArenaHdrLayout.compute(num_slots=12000)
        assert layout.bitmap_offset == MUTEX_PAD_BYTES
        # slot_state 应对齐到 8
        assert layout.slot_state_offset % 8 == 0
        # 总大小 4KB 页对齐
        assert layout.total_size % PAGE_ALIGN_BYTES == 0

    def test_layout_stage6_reserved_present(self):
        """Stage 6 字段必须有预留空间, 即使我们 Stage 1 不用."""
        layout = ArenaHdrLayout.compute(num_slots=12000)
        # refcnt: num_slots * 2 byte
        assert layout.slot_refcnt_bytes == 24000
        # hash_table 容量必须 >= 2 * num_slots 且是质数
        assert layout.hash_table_capacity >= 2 * 12000
        # hash_table_bytes = cap * 24 (Stage 6c: entry 带 gen)
        assert layout.hash_table_bytes == layout.hash_table_capacity * 24

    def test_layout_total_size_at_least_1MB(self):
        layout = ArenaHdrLayout.compute(num_slots=12000)
        assert layout.total_size >= HDR_TARGET_SIZE

    def test_layout_offsets_no_overlap(self):
        layout = ArenaHdrLayout.compute(num_slots=12000)
        # 顺序检查
        assert MUTEX_PAD_BYTES <= layout.bitmap_offset
        assert (layout.bitmap_offset + layout.bitmap_bytes
                <= layout.slot_state_offset)
        assert (layout.slot_state_offset + layout.slot_state_bytes
                <= layout.slot_refcnt_offset)
        assert (layout.slot_refcnt_offset + layout.slot_refcnt_bytes
                <= layout.hash_table_offset)
        assert (layout.hash_table_offset + layout.hash_table_bytes
                <= layout.total_size)

    def test_layout_small_num_slots(self):
        """num_slots=1 边界."""
        layout = ArenaHdrLayout.compute(num_slots=1)
        assert layout.num_slots == 1
        assert layout.bitmap_bytes == 8  # 1 word
        assert layout.slot_state_bytes == 8


# ============================================================
# Group B - Hdr 创建 / 打开
# ============================================================

class TestGroupBLifecycle:
    def test_create_and_close(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            assert os.path.exists(tmp_hdr_path)
            assert os.path.getsize(tmp_hdr_path) >= HDR_TARGET_SIZE
            assert hdr.base_addr > 0
        finally:
            hdr.close()

    def test_create_then_open_same_layout(self, tmp_hdr_path):
        hdr1 = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            # 通过 mutex 写一个 marker
            hdr2 = ArenaHdr.open(tmp_hdr_path, num_slots=100)
            try:
                assert hdr2.layout.total_size == hdr1.layout.total_size
                assert hdr2.layout.slot_state_offset == hdr1.layout.slot_state_offset
                # 两个 mmap 应映射同一文件
                # mutex 应已被 create 时 init 过
                rc = A.mutex_lock(hdr2.mutex_addr)
                assert rc == 0
                A.mutex_unlock(hdr2.mutex_addr)
            finally:
                hdr2.close()
        finally:
            hdr1.close()

    def test_mutex_works_after_create(self, tmp_hdr_path):
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=100)
        try:
            assert A.mutex_lock(hdr.mutex_addr) == 0
            assert A.mutex_unlock(hdr.mutex_addr) == 0
        finally:
            hdr.destroy_mutex()
            hdr.close()


# ============================================================
# Group C - Bitmap
# ============================================================

class TestGroupCBitmap:
    def test_bitmap_init_all_free(self, tmp_hdr_path):
        N = 200
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            hdr.bitmap_init_all_free()
            # 所有 slot 都 free
            for i in range(N):
                assert hdr.bitmap_is_free(i), f"slot {i} not free"
            assert hdr.count_free() == N
        finally:
            hdr.close()

    def test_bitmap_set_used_set_free(self, tmp_hdr_path):
        N = 100
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            hdr.bitmap_init_all_free()
            hdr.bitmap_set_used(50)
            assert not hdr.bitmap_is_free(50)
            # 邻居必须不受影响
            assert hdr.bitmap_is_free(49)
            assert hdr.bitmap_is_free(51)

            hdr.bitmap_set_free(50)
            assert hdr.bitmap_is_free(50)
        finally:
            hdr.close()

    def test_bitmap_partial_last_word(self, tmp_hdr_path):
        """num_slots 非 64 倍数时, 末尾 word 处理."""
        N = 100  # 不是 64 倍数, 末尾 word 只用 100-64=36 bit
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            hdr.bitmap_init_all_free()
            # 边界 slot
            assert hdr.bitmap_is_free(N - 1)
            hdr.bitmap_set_used(N - 1)
            assert not hdr.bitmap_is_free(N - 1)
            # 其他 slot 不受影响
            assert hdr.bitmap_is_free(0)
            assert hdr.count_free() == N - 1
        finally:
            hdr.close()

    def test_count_free(self, tmp_hdr_path):
        N = 1000
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            hdr.bitmap_init_all_free()
            assert hdr.count_free() == N
            for i in range(0, N, 3):
                hdr.bitmap_set_used(i)
            expected_free = N - len(range(0, N, 3))
            assert hdr.count_free() == expected_free
        finally:
            hdr.close()


# ============================================================
# Group D - Slot state 地址
# ============================================================

class TestGroupDSlotState:
    def test_slot_state_addr_distinct_per_slot(self, tmp_hdr_path):
        N = 100
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            addrs = [hdr.slot_state_addr(i) for i in range(N)]
            # 全部不同
            assert len(set(addrs)) == N
            # 相邻差 8
            for i in range(N - 1):
                assert addrs[i + 1] - addrs[i] == 8
        finally:
            hdr.close()

    def test_slot_state_through_atomic_ops(self, tmp_hdr_path):
        """通过 atomic API 写/读 slot_state, 跨 slot 不串扰."""
        N = 100
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            # 给每个 slot 设不同 gen
            for i in range(N):
                A.publish_slot(hdr.slot_state_addr(i), i + 1)
            # 验证 gen 读回
            for i in range(N):
                assert A.get_gen(hdr.slot_state_addr(i)) == i + 1
                assert A.get_pin(hdr.slot_state_addr(i)) == 0
            # pin 几个 slot
            for i in [3, 7, 42]:
                assert A.try_pin(hdr.slot_state_addr(i), i + 1) == 1
            # 其他 slot pin 不受影响
            for i in range(N):
                expected_pin = 1 if i in [3, 7, 42] else 0
                assert A.get_pin(hdr.slot_state_addr(i)) == expected_pin
        finally:
            hdr.close()

    def test_initial_state_is_zero(self, tmp_hdr_path):
        """ftruncate 后所有 slot_state 都是 0 (gen=0, pin=0)."""
        N = 50
        hdr = ArenaHdr.create(tmp_hdr_path, num_slots=N)
        try:
            for i in range(N):
                addr = hdr.slot_state_addr(i)
                assert A.get_gen(addr) == 0
                assert A.get_pin(addr) == 0
        finally:
            hdr.close()
