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
    # 关后台 evictor: 单测要确定性 (后台线程会异步淘汰干扰断言). 单独测它.
    monkeypatch.setenv("LICHT_ARENA_BG_EVICTOR", "0")
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

    def test_gpu_write_fn_hook(self, tmp_path, monkeypatch):
        """store-direct: gpu_write_fn 钩子替代逐块 CPU 搬运. 只对 MISS 块调钩子
        (传 arena slot + inc 内位置), HIT 不调; dedup/refcnt/gen/lookup 全正常."""
        store, arena, writes = _make_store(tmp_path, monkeypatch, num_slots=64)
        # 用钩子模拟 GPU 直写: (miss_slots, miss_pos) -> 从 src[pos] 写 slot
        src_a = [b"a0", b"a1", b"a2"]
        cap = []
        def gw_a(miss_slots, miss_pos):
            cap.append((list(miss_slots), list(miss_pos)))
            for s, p in zip(miss_slots, miss_pos):
                arena[s] = src_a[p]
        toks_a = list(range(48))
        assert store.write_inc("A", 0, 3, toks_a, source_obj=None,
                               gpu_write_fn=gw_a)
        # 全 MISS: 钩子拿到 3 块, 位置 0/1/2; CPU data_writer 没被调
        assert cap[-1][1] == [0, 1, 2]
        assert writes == []
        assert len(arena) == 3
        # gen 已 publish -> lookup own 命中
        res = store.lookup("A", toks_a)
        assert res is not None and res[0] == 3 * BS

        # job B 共享 A 前 3 块 -> 只第 4 块 MISS 经钩子, 前 3 块 HIT refcnt++
        src_b = [b"b0", b"b1", b"b2", b"b3"]
        cap.clear()
        def gw_b(miss_slots, miss_pos):
            cap.append((list(miss_slots), list(miss_pos)))
            for s, p in zip(miss_slots, miss_pos):
                arena[s] = src_b[p]
        toks_b = list(range(48)) + list(range(100, 116))
        assert store.write_inc("B", 0, 4, toks_b, source_obj=None,
                               gpu_write_fn=gw_b)
        assert cap[-1][1] == [3]          # 仅 MISS 第 4 块
        assert len(arena) == 4            # 共享 3 + 新 1, 不是 7
        for i in range(3):               # 共享块 refcnt 升到 2
            assert _refcnt(store, _slot_of(store, toks_b, i)) == 2

    def test_evict_deferred_self_heal(self, tmp_path, monkeypatch):
        """write_inc 内部淘汰走 deferred self-heal (锁外批量重写 manifest):
        淘汰 victim 尾 inc 后, 其 manifest total_blocks 正确回退, 新 job 存入."""
        # 关预淘汰余量, 测精确尾淘 (余量是独立优化, 否则小 arena 会多淘)
        monkeypatch.setenv("LICHT_ARENA_PREEVICT_MARGIN", "0")
        store, arena, _ = _make_store(tmp_path, monkeypatch, num_slots=4)
        toks_a = list(range(64))                      # A: 4 block, 两个 inc
        assert store.write_inc("A", 0, 2, toks_a, [b"a0", b"a1"])
        assert store.write_inc("A", 2, 4, toks_a, [b"a2", b"a3"])
        assert store._read_manifest("A")["total_blocks"] == 4   # arena 满 (4/4)
        # B 全新 1 block → alloc 失败 → 内部淘汰 A 尾 inc [2,4) (free 2 >= need 1)
        toks_b = list(range(200, 216))
        assert store.write_inc("B", 0, 1, toks_b, [b"b0"])
        rb = store.lookup("B", toks_b)
        assert rb is not None and rb[0] == 1 * BS        # B 存入命中
        # A 尾 inc 被淘, [0,2) 存活 → deferred self-heal 把 total_blocks 4→2
        ma = store._read_manifest("A")
        assert ma is not None and ma["total_blocks"] == 2

    def test_evict_lockfree(self, tmp_path, monkeypatch):
        """锁外两阶段淘汰 _evict_lockfree: 释放 slot + 尾 inc self-heal + 存活块可查."""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toks_a = list(range(64))                       # A: 4 block, 两个 inc
        assert store.write_inc("A", 0, 2, toks_a, [b"a0", b"a1"])
        assert store.write_inc("A", 2, 4, toks_a, [b"a2", b"a3"])
        free_before = store._allocator.free_count      # 8-4=4
        freed = store._evict_lockfree(1)               # 淘 A 尾 inc [2,4) → free 2
        assert freed == 2
        assert store._allocator.free_count == free_before + 2
        # 尾 inc 淘掉, [0,2) 存活 → inline self-heal manifest 回退到 2
        assert store._read_manifest("A")["total_blocks"] == 2
        r = store.lookup("A", toks_a)                  # 前 2 块仍可查
        assert r is not None and r[0] == 2 * BS

    def test_slot_index_populated(self, tmp_path, monkeypatch):
        """方案C: store 后进程内 _job_slot_index 有正确 (s,e,records) 条目。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toks = list(range(64))
        assert store.write_inc("A", 0, 2, toks, [b"a0", b"a1"])
        assert store.write_inc("A", 2, 4, toks, [b"a2", b"a3"])
        idx = store._job_slot_index["A"]
        assert len(idx) == 2
        assert (idx[0][0], idx[0][1]) == (0, 2)
        assert (idx[1][0], idx[1][1]) == (2, 4)
        slot, gen, h = idx[0][2][0]                  # records = (slot,gen,hash)
        assert slot == _slot_of(store, toks, 0)

    def test_evict_via_index_updates_index(self, tmp_path, monkeypatch):
        """方案C: 走索引淘汰尾 inc 后, 索引剔除该 inc (不读 .slot 文件)。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toks = list(range(64))
        store.write_inc("A", 0, 2, toks, [b"a0", b"a1"])
        store.write_inc("A", 2, 4, toks, [b"a2", b"a3"])
        assert len(store._job_slot_index["A"]) == 2
        freed = store._evict_lockfree(1)             # 淘尾 inc [2,4)
        assert freed == 2
        assert len(store._job_slot_index["A"]) == 1  # 尾 inc 从索引剔除
        assert store._job_slot_index["A"][0][1] == 2

    def test_evict_fallback_no_index(self, tmp_path, monkeypatch):
        """方案C: 内存索引缺失 (模拟跨重启/跨进程) → 回退读 .slot 文件仍能淘。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toks = list(range(64))
        store.write_inc("A", 0, 2, toks, [b"a0", b"a1"])
        store.write_inc("A", 2, 4, toks, [b"a2", b"a3"])
        store._job_slot_index.clear()                # 内存索引没了
        free_before = store._allocator.free_count
        freed = store._evict_lockfree(1)             # 回退文件路径
        assert freed == 2
        assert store._allocator.free_count == free_before + 2
        assert store._read_manifest("A")["total_blocks"] == 2

    def test_evict_recheck_aborts_wasted(self, tmp_path, monkeypatch):
        """Phase 0.1: recheck_fn 返回 0 (并发 store 已满足需求) → 不做无效淘汰;
        返回正数则正常淘。直接对应 miss=0 的 116s '淘了个寂寞'。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toks = list(range(64))
        store.write_inc("A", 0, 2, toks, [b"a0", b"a1"])
        store.write_inc("A", 2, 4, toks, [b"a2", b"a3"])
        free_before = store._allocator.free_count
        # recheck 说"已不缺槽" → 立即停, A 一个块都不淘
        freed = store._evict_lockfree(4, recheck_fn=lambda: 0)
        assert freed == 0
        assert store._allocator.free_count == free_before
        # recheck 返回正数 → 正常淘
        freed2 = store._evict_lockfree(1, recheck_fn=lambda: 4)
        assert freed2 == 2

    def test_evict_index_only_skips_file_victim(self, tmp_path, monkeypatch):
        """Phase 1a: index_only=True 跳过非本进程索引(文件回退)的 victim →
        后台 evictor 不会淘对方进程的 job (杜绝跨进程 double-decref)。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toks = list(range(64))
        store.write_inc("A", 0, 2, toks, [b"a0", b"a1"])
        store.write_inc("A", 2, 4, toks, [b"a2", b"a3"])
        store._job_slot_index.pop("A", None)        # 模拟 A 是"对方进程"的 job
        free_before = store._allocator.free_count
        freed = store._evict_lockfree(4, index_only=True)   # 跳过 A
        assert freed == 0
        assert store._allocator.free_count == free_before
        freed2 = store._evict_lockfree(1, index_only=False)  # 回退文件淘 A
        assert freed2 == 2

    def test_protected_not_evicted(self, tmp_path, monkeypatch):
        """修2: protected store 的 slot 被 pin 住 → 淘汰跳过(can_evict pin==0);
        release_protected 后回到可淘。①②(ARENA_SINK/preempt)用它防被淘→重算。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toksA = list(range(64))
        toksB = list(range(1000, 1032))
        # A = 在途(protected), B = 普通可淘
        assert store.write_inc("A", 0, 2, toksA, [b"a0", b"a1"],
                               protected=True, protect_key="reqA")
        assert store.write_inc("B", 0, 2, toksB, [b"b0", b"b1"])
        slotA0 = _slot_of(store, toksA, 0)
        assert store._protect_pinned_n == 2
        store._evict_lockfree(4)                 # 想腾4: A 被 pin 跳过, 只淘 B
        assert not store._allocator.is_free(slotA0)   # A 被保护, 没淘
        store.release_protected("reqA")          # 释放保护
        assert store._protect_pinned_n == 0
        store._evict_lockfree(2)                 # A 现在可淘
        assert store._allocator.is_free(slotA0)

    def test_job_claim_flock_exclusive(self, tmp_path, monkeypatch):
        """Phase 1b: job claim 用独立 fd 的 flock → 互斥 (同进程不同 fd 也互斥,
        即跨进程/跨线程保证'同一 job 同时只一个淘汰者' → 防 double refcnt--)。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        store.write_inc("A", 0, 2, list(range(2 * BS)), [b"a0", b"a1"])
        fd1 = store._claim_job("A")
        assert fd1 is not None
        fd2 = store._claim_job("A")          # 另一独立 fd → 抢不到
        assert fd2 is None
        store._release_claim(fd1)
        fd3 = store._claim_job("A")          # 释放后可再抢
        assert fd3 is not None
        store._release_claim(fd3)
        assert store._claim_job("NOPE") is None   # 不存在的 job → None

    def test_evict_lock_timeout(self, tmp_path, monkeypatch):
        """Phase 0.2: _evict_lock 被占时, 设了 budget_ms 的淘汰超时返回 0 (不排队,
        干掉 lw=86s 集体等锁); 锁释放后正常淘。budget_ms=None(后台) 仍阻塞。"""
        import time as _t
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=8)
        toks = list(range(64))
        store.write_inc("A", 0, 2, toks, [b"a0", b"a1"])
        store.write_inc("A", 2, 4, toks, [b"a2", b"a3"])
        store._evict_lock.acquire()                  # 占住锁
        try:
            t0 = _t.time()
            freed = store._evict_lockfree(2, budget_ms=20)   # 拿不到 → 超时
            assert freed == 0
            assert _t.time() - t0 < 0.5
        finally:
            store._evict_lock.release()
        freed2 = store._evict_lockfree(1, budget_ms=50)      # 锁空了正常淘
        assert freed2 == 2

    def test_bg_evictor_frees(self, tmp_path, monkeypatch):
        """Phase 1a: 后台 evictor 启动后, free<low 时自动把池淘到 high 附近。"""
        import time as _t
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=16)
        store._bg_evictor_on = True                  # fixture 默认关, 这里开
        store._bg_low = 2
        store._bg_high = 4
        store._bg_interval = 0.01
        for j in range(8):                           # 8 job × 2 块 = 填满 16 槽
            base = j * 1000                          # 每 job 唯一 token → 唯一内容
            store.write_inc(f"J{j}", 0, 2,
                            list(range(base, base + 2 * BS)),
                            [f"{j}a".encode(), f"{j}b".encode()])
        store._ensure_bg_evictor()
        store._signal_bg_evictor()
        ok = False
        for _ in range(300):                         # 最多 ~3s 等后台
            if store._stat_bg_freed > 0:
                ok = True
                break
            _t.sleep(0.01)
        store._bg_stop = True
        assert ok and store._stat_bg_freed > 0

    def test_evict_gen_revalidation(self, tmp_path, monkeypatch):
        """两阶段 apply 的 gen 复核: 记录 gen 与 slot 当前 gen 不符 (模拟锁外读后被
        别进程 free/复用), 则跳过 — 不误减引用、不 free。gen 对则正常淘。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=16)
        toks = list(range(48))
        assert store.write_inc("A", 0, 3, toks, [b"a0", b"a1", b"a2"])
        slot0 = _slot_of(store, toks, 0)
        h0 = block_hashes(toks, BS, 1)[0]
        cur_gen = A.get_gen(store._hdr.slot_state_addr(slot0))
        rc0 = _refcnt(store, slot0)
        # 错 gen → 复核跳过
        freed, _ = store._evict_inc_apply_locked([(slot0, cur_gen + 999, h0)])
        assert freed == 0
        assert _refcnt(store, slot0) == rc0
        assert not store._allocator.is_free(slot0)
        # 对 gen → 正常 free
        freed2, _ = store._evict_inc_apply_locked([(slot0, cur_gen, h0)])
        assert freed2 == 1
        assert store._allocator.is_free(slot0)

    def test_content_addr_manifest_no_tokens(self, tmp_path, monkeypatch):
        """content_addr 下 manifest 不存全量 token_ids (瘦身, 省 O(token) 写),
        lookup 仍命中 (走哈希表 lookup_resolve), own + cross-job 都覆盖。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=16)
        toks = list(range(48))                       # 3 block
        assert store.write_inc("A", 0, 3, toks, [b"a0", b"a1", b"a2"])
        m = store._read_manifest("A")
        assert m["total_blocks"] == 3
        assert m["token_ids"] == []                  # 不再存全量 token
        # own-job lookup 仍命中 (内部走哈希表)
        res = store.lookup("A", toks)
        assert res is not None and res[0] == 3 * BS
        # cross-job: 新 job 同前缀也命中
        res2 = store.lookup_resolve(toks)
        assert res2 is not None and res2[0] == 3 * BS

    def test_content_addr_off_keeps_tokens(self, tmp_path, monkeypatch):
        """content_addr 关时仍存 token_ids (own-job LCP lookup 依赖它)。"""
        store, _, _ = _make_store(tmp_path, monkeypatch, num_slots=16,
                                  content=False)
        toks = list(range(32))
        assert store.write_inc("A", 0, 2, toks, [b"a0", b"a1"])
        m = store._read_manifest("A")
        assert len(m["token_ids"]) == 32             # 仍存全量
        res = store.lookup("A", toks)
        assert res is not None and res[0] == 2 * BS

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
