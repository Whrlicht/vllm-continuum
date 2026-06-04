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

    def test_crossjob_lookup_then_evict_invalidates(self, tmp_path, monkeypatch):
        """淘汰掉源 job 后, 之前 lookup_resolve 拿到的 slot 在 load 时失效."""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        toks_a = list(range(48))
        store.write_inc("A", 0, 3, toks_a, [b"a0", b"a1", b"a2"])
        res = store.lookup_resolve(toks_a)
        assert res is not None
        _, nb, sg = res
        assert nb == 3
        # 淘掉 A (refcnt 1->0, gen bump)
        _evict_job(store, "A")
        # 用淘汰前解析的 slot/gen 去 load -> try_pin 应失配 -> None
        handle = store.load_pin_explicit(sg, dst_block_ids=[1, 2, 3])
        assert handle is None


# ============================================================
# 跨 job lookup_resolve + load_pin_explicit (6c)
# ============================================================

class TestCrossJobLookup:
    def test_brand_new_job_hits_other_job_prefix(self, tmp_path, monkeypatch):
        """全新 job (从没 store 过) 的 prompt 共享 A 的前缀 → 命中 A 的 slot."""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        toks_a = list(range(48))                       # A: 3 block
        store.write_inc("A", 0, 3, toks_a, [b"a0", b"a1", b"a2"])
        a_slots = [_slot_of(store, toks_a, i) for i in range(3)]

        # 全新 prompt: 共享 A 前 3 块, 第 4 块独有 (没人存过)
        toks_new = list(range(48)) + list(range(900, 916))
        res = store.lookup_resolve(toks_new)
        assert res is not None
        matched_tokens, nb, sg = res
        assert nb == 3                                  # 只命中共享前缀 (第4块 miss)
        assert matched_tokens == 3 * BS
        # 解析出的 slot 正是 A 的物理 slot, gen 是已发布 gen (>0)
        assert [s for (s, g) in sg] == a_slots
        assert all(g > 0 for (s, g) in sg)

        # 用这些 slot load (pin → 校验 → release)
        handle = store.load_pin_explicit(sg, dst_block_ids=[20, 21, 22])
        assert handle is not None
        assert handle.slot_ids == a_slots
        assert handle.dst_block_ids == [20, 21, 22]
        for addr in handle.slot_state_addrs:
            assert A.get_pin(addr) == 1
        assert handle.post_load_validate()
        handle.release()

    def test_load_batch_pin_explicit_4tuple(self, tmp_path, monkeypatch):
        """connector 实际路径: load_batch_pin 收 4 元组 (含 slot_gen) → 走显式 pin."""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        toks_a = list(range(48))
        store.write_inc("A", 0, 3, toks_a, [b"a0", b"a1", b"a2"])
        _, _, sg = store.lookup_resolve(list(range(48)) + list(range(900, 916)))
        # 全新 job B 用解析出的 slot 走 4 元组 batch load
        bh = store.load_batch_pin([("B", [30, 31, 32], 0, sg)])
        assert bh.per_item_ok == [True]
        assert bh.slot_ids == [_slot_of(store, toks_a, i) for i in range(3)]
        assert bh.dst_block_ids == [30, 31, 32]
        assert bh.post_load_validate()
        bh.release()
        # 3 元组 (own-job) 仍正常 (向后兼容)
        bh2 = store.load_batch_pin([("A", [40, 41, 42], 0)])
        assert bh2.per_item_ok == [True]
        bh2.release()

    def test_c_lookup_matches_python_reference(self, tmp_path, monkeypatch):
        """C lookup_resolve 与 Python 逐块参考逻辑结果完全一致."""
        from vllm.v1.core.sched.licht_v3.arena_block_hash import block_hashes
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=128)
        store.write_inc("A", 0, 4, list(range(64)),
                        [b"a0", b"a1", b"a2", b"a3"])
        prompt = list(range(64)) + list(range(500, 516))  # 4 共享 + 1 未存
        res_c = store.lookup_resolve(prompt)
        # Python 参考 (复刻 fallback 逻辑)
        bs = store.block_size
        n_full = len(prompt) // bs
        ref = []
        for h in block_hashes(prompt, bs, n_full):
            slot, egen = A.ht_probe(store._ht_base, store._ht_cap, h)
            if slot < 0 or egen == 0:
                break
            if A.get_gen(store._hdr.slot_state_addr(slot)) != egen:
                break
            if A.refcnt_get(store._hdr.slot_refcnt_addr(slot)) == 0:
                break
            ref.append((slot, egen))
        if not ref:
            assert res_c is None
        else:
            assert res_c == (len(ref) * bs, len(ref), ref)
            assert res_c[1] == 4   # 命中 4 个共享块

    def test_lookup_resolve_no_prefix_returns_none(self, tmp_path, monkeypatch):
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        store.write_inc("A", 0, 2, list(range(32)), [b"0", b"1"])
        # 完全不同前缀 → 第 0 块就 miss
        assert store.lookup_resolve(list(range(500, 532))) is None

    def test_lookup_resolve_partial_prefix(self, tmp_path, monkeypatch):
        """共享 2 块、第 3 块分叉 → 只解析出 2 块."""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=64)
        store.write_inc("A", 0, 3, list(range(48)), [b"0", b"1", b"2"])
        toks = list(range(32)) + list(range(700, 716))  # 前2块同, 第3块异
        res = store.lookup_resolve(toks)
        assert res is not None
        _, nb, _sg = res
        assert nb == 2

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
