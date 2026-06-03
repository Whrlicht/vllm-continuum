# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena Stage 6b/6e: 内容寻址 dedup store + refcnt evict 测试.

验证:
  - 共享前缀: 第二个 job store 时逐块 HIT (refcnt++), 不重复分配 slot / 不重复搬数据
  - arena 占用 = unique block 数 (而非 sum of job blocks)
  - .slot 写成 v2 (content_addr on) / v1 (off)
  - lookup/load own-path 在 v2 下正常
  - evict: 共享块 refcnt-- 不到 0 不销毁 (数据留给别的 job); 最后引用者淘到 0 才真释放
"""
import struct

import pytest

try:
    import licht_arena_atomic as A
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.arena_block_hash import block_hashes
from vllm.v1.core.sched.licht_v3.arena_lru_store import LruArenaStore
from vllm.v1.core.sched.licht_v3.arena_slot_file import read_slot_file_version

BS = 16


def _make_store(tmp_path, monkeypatch, num_slots, content=True):
    monkeypatch.setenv("LICHT_ARENA_CONTENT_ADDR", "1" if content else "0")
    store = LruArenaStore.create(str(tmp_path / "arena"),
                                 num_slots=num_slots, block_size=BS)
    arena = {}          # slot_id -> value
    writes = []         # (slot_id, block_idx) 每次 data_writer 调用
    def writer(slot_id, i, src):
        arena[slot_id] = src[i]
        writes.append((slot_id, i))
    store.bind_data_writer(writer)
    return store, arena, writes


def _slot_of(store, token_ids, block_idx):
    """通过链式 hash + ht_probe 拿到某 block 对应的物理 slot."""
    hs = block_hashes(token_ids, BS, block_idx + 1)
    return A.ht_probe(store._ht_base, store._ht_cap, hs[block_idx])[0]


def _refcnt(store, slot):
    return A.refcnt_get(store._hdr.slot_refcnt_addr(slot))


def _evict_job(store, job_id, max_need=10**9):
    rc = A.mutex_lock(store._hdr.mutex_addr)
    try:
        return store._evict_job_tail_first(str(job_id), max_need)
    finally:
        A.mutex_unlock(store._hdr.mutex_addr)


# ============================================================
# dedup store 行为
# ============================================================

class TestDedupStore:
    def test_shared_prefix_dedup(self, tmp_path, monkeypatch):
        store, arena, writes = _make_store(tmp_path, monkeypatch, num_slots=64)
        # job A: 3 block 前缀
        toks_a = list(range(48))
        assert store.write_inc("A", 0, 3, toks_a, [b"a0", b"a1", b"a2"])
        assert len(arena) == 3                  # 3 个新 slot
        assert len(writes) == 3                 # 3 次搬数据
        for i in range(3):
            assert _refcnt(store, _slot_of(store, toks_a, i)) == 1

        # job B: 共享 A 的 3 block 前缀 + 1 个独有 block
        toks_b = list(range(48)) + list(range(100, 116))
        writes.clear()
        assert store.write_inc("B", 0, 4, toks_b,
                               [b"b0", b"b1", b"b2", b"b3"])
        # 共享前 3 块 HIT: 不新增 slot, 只为第 4 块新分配 1 个
        assert len(arena) == 4                  # 总 unique = 4 (而非 3+4=7)
        # B 只为 MISS 块 (i=3) 搬了数据
        assert writes == [(_slot_of(store, toks_b, 3), 3)]
        # 共享块 refcnt 升到 2, 独有块 refcnt 1
        for i in range(3):
            assert _refcnt(store, _slot_of(store, toks_b, i)) == 2
        assert _refcnt(store, _slot_of(store, toks_b, 3)) == 1
        # 共享块 A/B 指向同一物理 slot
        for i in range(3):
            assert _slot_of(store, toks_a, i) == _slot_of(store, toks_b, i)

    def test_slot_file_is_v2(self, tmp_path, monkeypatch):
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=32)
        store.write_inc("J", 0, 2, list(range(32)), [b"0", b"1"])
        path = store._slot_path("J", 0, 2)
        assert read_slot_file_version(path) == 2

    def test_plain_mode_is_v1_no_dedup(self, tmp_path, monkeypatch):
        store, arena, _ = _make_store(tmp_path, monkeypatch, num_slots=32,
                                      content=False)
        store.write_inc("A", 0, 2, list(range(32)), [b"0", b"1"])
        store.write_inc("B", 0, 2, list(range(32)), [b"0", b"1"])
        # plain 模式无 dedup: 两 job 各占 2 slot
        assert len(arena) == 4
        assert read_slot_file_version(store._slot_path("A", 0, 2)) == 1

    def test_idempotent_incremental(self, tmp_path, monkeypatch):
        """同 job 分两次 store [0,2) 再 [2,4): 不重复 refcnt 前缀."""
        store, arena, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        toks = list(range(64))
        store.write_inc("J", 0, 2, toks, [b"0", b"1"])
        store.write_inc("J", 2, 4, toks, [b"0", b"1", b"2", b"3"])
        # block 0,1 只在第一次 store, refcnt 仍为 1 (没被第二次 inc 重复 ++)
        for i in range(4):
            assert _refcnt(store, _slot_of(store, toks, i)) == 1
        assert len(arena) == 4


# ============================================================
# lookup / load own-path (v2)
# ============================================================

class TestLookupLoadV2:
    def test_lookup_own(self, tmp_path, monkeypatch):
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        toks = list(range(48))
        store.write_inc("J", 0, 3, toks, [b"0", b"1", b"2"])
        res = store.lookup("J", toks)
        assert res == (3 * BS, 3)

    def test_load_request_pins(self, tmp_path, monkeypatch):
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        toks = list(range(48))
        store.write_inc("J", 0, 3, toks, [b"0", b"1", b"2"])
        handle = store.load_request("J", dst_block_ids=[10, 11, 12],
                                    src_block_offset=0)
        assert handle is not None
        assert len(handle.slot_ids) == 3
        assert handle.dst_block_ids == [10, 11, 12]
        # 加载期间这些 slot 被 pin
        for addr in handle.slot_state_addrs:
            assert A.get_pin(addr) == 1
        assert handle.post_load_validate()
        handle.release()
        for addr in handle.slot_state_addrs:
            assert A.get_pin(addr) == 0


# ============================================================
# evict refcnt
# ============================================================

class TestEvictRefcnt:
    def test_shared_block_survives_one_evict(self, tmp_path, monkeypatch):
        store, arena, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        toks_a = list(range(48))
        toks_b = list(range(48)) + list(range(100, 116))
        store.write_inc("A", 0, 3, toks_a, [b"a0", b"a1", b"a2"])
        store.write_inc("B", 0, 4, toks_b, [b"b0", b"b1", b"b2", b"b3"])
        shared = [_slot_of(store, toks_b, i) for i in range(3)]
        uniq_b = _slot_of(store, toks_b, 3)

        # 淘 A: 共享块 refcnt 2->1, 没有任何块到 0 -> freed=0, 数据保留
        freed_a = _evict_job(store, "A")
        assert freed_a == 0
        for s in shared:
            assert _refcnt(store, s) == 1
            assert not store._allocator.is_free(s)
        # B 仍能 lookup/load 全部 4 块
        assert store.lookup("B", toks_b) == (4 * BS, 4)

        # 淘 B: 共享块 refcnt 1->0 真销毁 + 独有块也销毁 -> freed=4
        freed_b = _evict_job(store, "B")
        assert freed_b == 4
        for s in shared + [uniq_b]:
            assert store._allocator.is_free(s)
        # hash 表里也已删除 (probe miss)
        for i in range(4):
            hs = block_hashes(toks_b, BS, i + 1)
            assert A.ht_probe(store._ht_base, store._ht_cap, hs[i])[0] == -1

    def test_evict_triggers_on_pressure(self, tmp_path, monkeypatch):
        """arena 满时 store 自动 evict 老 job 腾空间."""
        store, arena, _ = _make_store(tmp_path, monkeypatch, num_slots=4)
        # A 占 3 个独有 slot
        store.write_inc("A", 0, 3, list(range(48)), [b"a0", b"a1", b"a2"])
        assert store._allocator.count_free_accurate() == 1
        # C 要 3 个全新块 (不同内容) -> 只剩 1 free -> 触发 evict A
        toks_c = list(range(200, 248))
        ok = store.write_inc("C", 0, 3, toks_c, [b"c0", b"c1", b"c2"])
        assert ok
        # C 的 3 块都在
        assert store.lookup("C", toks_c) == (3 * BS, 3)
