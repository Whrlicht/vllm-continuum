# SPDX-License-Identifier: Apache-2.0
"""ArenaAllocator 单元测试."""
import pytest

try:
    import licht_arena_atomic  # noqa: F401
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.arena_hdr import ArenaHdr
from vllm.v1.core.sched.licht_v3.arena_allocator import ArenaAllocator


@pytest.fixture
def small_hdr(tmp_path):
    """100 slot 的小 hdr."""
    path = str(tmp_path / "small.hdr")
    hdr = ArenaHdr.create(path, num_slots=100)
    yield hdr
    hdr.close()


@pytest.fixture
def alloc100(small_hdr):
    a = ArenaAllocator(small_hdr)
    a.init_all_free()
    yield a


# ============================================================
# Group A - 基本 alloc/free
# ============================================================

class TestGroupABasicAlloc:
    def test_init_all_free(self, alloc100):
        assert alloc100.free_count == 100
        assert alloc100.num_slots == 100

    def test_alloc_zero(self, alloc100):
        result = alloc100.alloc_n(0)
        assert result == []
        assert alloc100.free_count == 100

    def test_alloc_one_returns_lowest_slot(self, alloc100):
        slots = alloc100.alloc_n(1)
        assert slots == [0]
        assert alloc100.free_count == 99
        assert not alloc100.is_free(0)
        for i in range(1, 100):
            assert alloc100.is_free(i)

    def test_alloc_multiple_sequential(self, alloc100):
        slots = alloc100.alloc_n(5)
        assert slots == [0, 1, 2, 3, 4]
        assert alloc100.free_count == 95

    def test_alloc_all(self, alloc100):
        slots = alloc100.alloc_n(100)
        assert slots == list(range(100))
        assert alloc100.free_count == 0

    def test_alloc_too_many_returns_none(self, alloc100):
        result = alloc100.alloc_n(101)
        assert result is None
        assert alloc100.free_count == 100

    def test_alloc_after_free(self, alloc100):
        slots = alloc100.alloc_n(10)
        alloc100.free_n([3, 5, 7])
        assert alloc100.free_count == 93
        # 下次 alloc 应填回 [3, 5, 7]
        new_slots = alloc100.alloc_n(3)
        assert new_slots == [3, 5, 7]

    def test_alloc_after_full_free(self, alloc100):
        slots = alloc100.alloc_n(100)
        alloc100.free_n(slots)
        assert alloc100.free_count == 100
        new_slots = alloc100.alloc_n(50)
        assert new_slots == list(range(50))


# ============================================================
# Group B - 边界情况
# ============================================================

class TestGroupBEdgeCases:
    def test_num_slots_not_multiple_of_64(self, tmp_path):
        path = str(tmp_path / "x.hdr")
        hdr = ArenaHdr.create(path, num_slots=100)
        try:
            a = ArenaAllocator(hdr)
            a.init_all_free()
            slots = a.alloc_n(100)
            assert slots == list(range(100))
            a.free_n([99, 98])
            assert a.alloc_n(2) == [98, 99]
        finally:
            hdr.close()

    def test_alloc_at_word_boundary(self, tmp_path):
        path = str(tmp_path / "x.hdr")
        hdr = ArenaHdr.create(path, num_slots=200)
        try:
            a = ArenaAllocator(hdr)
            a.init_all_free()
            slots = a.alloc_n(64)
            assert slots == list(range(64))
            slots2 = a.alloc_n(1)
            assert slots2 == [64]
        finally:
            hdr.close()

    def test_alloc_sparse_pattern(self, tmp_path):
        path = str(tmp_path / "x.hdr")
        hdr = ArenaHdr.create(path, num_slots=200)
        try:
            a = ArenaAllocator(hdr)
            a.init_all_free()
            full = a.alloc_n(200)
            assert a.free_count == 0
            sparse = [10, 50, 100, 150, 199]
            a.free_n(sparse)
            assert a.free_count == 5
            new_slots = a.alloc_n(5)
            assert new_slots == sparse
        finally:
            hdr.close()

    def test_alloc_n_too_large_does_not_modify(self, alloc100):
        alloc100.alloc_n(80)
        assert alloc100.free_count == 20
        result = alloc100.alloc_n(30)
        assert result is None
        assert alloc100.free_count == 20
        result = alloc100.alloc_n(20)
        assert result is not None
        assert len(result) == 20

    def test_repeated_alloc_free_cycle_does_not_leak(self, tmp_path):
        path = str(tmp_path / "x.hdr")
        hdr = ArenaHdr.create(path, num_slots=100)
        try:
            a = ArenaAllocator(hdr)
            a.init_all_free()
            for _ in range(1000):
                slots = a.alloc_n(50)
                assert slots is not None
                assert a.free_count == 50
                a.free_n(slots)
                assert a.free_count == 100
        finally:
            hdr.close()


# ============================================================
# Group C - sync_from_bitmap
# ============================================================

class TestGroupCSync:
    def test_sync_from_bitmap_after_external_modify(self, alloc100, small_hdr):
        for i in range(0, 100, 2):
            small_hdr.bitmap_set_used(i)
        assert alloc100.free_count == 100  # stale
        alloc100.sync_from_bitmap()
        assert alloc100.free_count == 50


# ============================================================
# Group D - 大规模
# ============================================================

class TestGroupDLarge:
    def test_large_num_slots(self, tmp_path):
        path = str(tmp_path / "big.hdr")
        hdr = ArenaHdr.create(path, num_slots=12000)
        try:
            a = ArenaAllocator(hdr)
            a.init_all_free()
            assert a.free_count == 12000
            allocated_per_inc = []
            for _ in range(100):
                s = a.alloc_n(30)
                assert s is not None
                allocated_per_inc.append(s)
            assert a.free_count == 12000 - 3000
            for inc_slots in allocated_per_inc[::2]:
                a.free_n(inc_slots)
            assert a.free_count == 12000 - 1500
        finally:
            hdr.close()
