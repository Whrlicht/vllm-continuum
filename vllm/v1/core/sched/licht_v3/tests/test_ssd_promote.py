# SPDX-License-Identifier: Apache-2.0
"""P2 升级路径 (SSD -> CPU promote) 测试, CPU-only.

测试矩阵:

Group A - resolve_range / probe_slots (账本探测原语)
Group B - promote_inc 数据面: 字节往返 / dedup 幂等 / pin 失败放弃
Group C - 两层闭环: store -> demote -> CPU 驱逐 -> tiered 拼接 ->
          promote -> 再探全命中 + 字节比对
Group D - RoundKVStore.lookup_resolve_tiered (scheduler 侧模拟)
"""
import json
import mmap
import os
import time

import pytest

try:
    import licht_arena_atomic as A
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.arena_block_hash import block_hashes
from vllm.v1.core.sched.licht_v3.arena_lru_store import LruArenaStore
from vllm.v1.core.sched.licht_v3.ssd_tier import SsdTier

BS = 16
SLOT_BYTES = 4096


def _tokens(n_blocks, seed=0):
    return [seed * 100000 + i for i in range(n_blocks * BS)]


def _cpu_env(monkeypatch):
    monkeypatch.setenv("LICHT_ARENA_CONTENT_ADDR", "1")
    monkeypatch.setenv("LICHT_ARENA_BG_EVICTOR", "0")
    import tempfile
    monkeypatch.setenv("LICHT_SSD_RING_DIR", tempfile.mkdtemp(prefix="ssdring_"))


def _fake_cpu_arena(n_slots):
    buf = mmap.mmap(-1, n_slots * SLOT_BYTES)

    def writer(slot_id, i, src):
        data = (src[i] * (SLOT_BYTES // len(src[i]) + 1))[:SLOT_BYTES]
        buf[slot_id * SLOT_BYTES:(slot_id + 1) * SLOT_BYTES] = data

    return buf, writer


def _slot_bytes_of(buf, slot):
    return bytes(buf[slot * SLOT_BYTES:(slot + 1) * SLOT_BYTES])


@pytest.fixture
def duo(tmp_path, monkeypatch):
    """CPU store (fake mmap arena) + SsdTier, 数据面全绑好."""
    _cpu_env(monkeypatch)
    cpu = LruArenaStore.create(str(tmp_path / "cpu"), num_slots=64,
                               block_size=BS)
    cpu_buf, cpu_writer = _fake_cpu_arena(64)
    cpu.bind_data_writer(cpu_writer)
    tier = SsdTier.open_or_create(
        meta_path=str(tmp_path / "ssd_meta"),
        data_path=str(tmp_path / "ssd_data"),
        ssd_gb=SLOT_BYTES * 64 / (1024 ** 3),
        slot_bytes=SLOT_BYTES, block_size=BS)
    tier.bind_cpu_source(cpu_buf, SLOT_BYTES)
    cpu.bind_demote_fn(tier.capture_inc)   # capture-at-eviction (零 pin)
    yield cpu, cpu_buf, tier
    tier.close()
    cpu.close()


def _store_and_demote(cpu, cpu_buf, tier, job, n_blocks, seed):
    """写 job -> CPU 驱逐 (capture 释放前 memmove 进 SHM 环, 零 pin) ->
    drain 环 (模拟写进程) 刷进 SSD. 返回 {hash: 原字节}."""
    blocks = [f"{job}-b{i}".encode() for i in range(n_blocks)]
    assert cpu.write_inc(job, 0, n_blocks, _tokens(n_blocks, seed), blocks)
    recs = cpu._job_slot_index[job][0][2]
    orig = {r[2]: _slot_bytes_of(cpu_buf, r[0]) for r in recs}
    # 纯 LRU 驱逐: 释放前 _capture_fn(=tier.capture_inc) memmove 进环.
    assert cpu._evict_lockfree(999) == n_blocks
    tier.drain_ring(tier._ring)        # 模拟独立写进程 drain -> SSD
    assert tier._stat_demote_blocks >= n_blocks
    return orig


# ============================================================
# Group A - 探测原语
# ============================================================
class TestResolvePrimitives:

    def test_resolve_range_mid_start(self, tmp_path, monkeypatch):
        _cpu_env(monkeypatch)
        s = LruArenaStore.create(str(tmp_path / "a"), num_slots=32,
                                 block_size=BS)
        s.bind_data_writer(lambda *a: None)
        toks = _tokens(8, seed=3)
        assert s.write_inc("j", 0, 8, toks, [b"x"] * 8)
        # 从 block 3 起续探 -> 应命中 [3,8)
        rr = s.resolve_range(toks, 3)
        assert rr is not None
        end, recs = rr
        assert end == 8 and len(recs) == 5
        # 每条 (slot, gen, hash) 与账本一致
        hs = block_hashes(toks, BS, 8)
        assert [r[2] for r in recs] == hs[3:8]
        # cap 生效
        end2, recs2 = s.resolve_range(toks, 3, max_blocks=2)
        assert end2 == 5 and len(recs2) == 2
        # 起点越界 / 无命中
        assert s.resolve_range(toks, 8) is None
        assert s.resolve_range(_tokens(8, seed=99), 0) is None
        s.close()

    def test_probe_slots_all_or_none(self, tmp_path, monkeypatch):
        _cpu_env(monkeypatch)
        s = LruArenaStore.create(str(tmp_path / "a"), num_slots=32,
                                 block_size=BS)
        s.bind_data_writer(lambda *a: None)
        toks = _tokens(4, seed=5)
        assert s.write_inc("j", 0, 4, toks, [b"x"] * 4)
        hs = block_hashes(toks, BS, 4)
        sg = s.probe_slots(hs)
        assert sg is not None and len(sg) == 4
        # 混入一个不存在的 hash -> 整体 None
        assert s.probe_slots(hs + [123456789]) is None
        s.close()


# ============================================================
# Group B - promote 数据面
# ============================================================
class TestPromoteDataPlane:

    def test_promote_bytes_roundtrip(self, duo):
        cpu, cpu_buf, tier = duo
        n = 4
        toks = _tokens(n, seed=7)
        orig = _store_and_demote(cpu, cpu_buf, tier, "jobA", n, seed=7)
        # CPU 已空; 从 SSD 账本解析该段
        rr = tier.store.resolve_range(toks, 0)
        assert rr is not None and rr[0] == n
        sg = tier.promote_inc(cpu, "jobA", 0, n, rr[1])
        assert sg is not None and len(sg) == n
        # 字节回到 CPU fake arena, 逐块比对 (按 hash 对应)
        for (slot, _gen), rec in zip(sg, rr[1]):
            assert _slot_bytes_of(cpu_buf, slot) == orig[rec[2]]
        # promote 后 CPU 表可直接探到全段
        hs = block_hashes(toks, BS, n)
        assert cpu.probe_slots(hs) is not None

    def test_promote_idempotent_dedup(self, duo):
        cpu, cpu_buf, tier = duo
        n = 3
        toks = _tokens(n, seed=8)
        _store_and_demote(cpu, cpu_buf, tier, "jobA", n, seed=8)
        rr = tier.store.resolve_range(toks, 0)
        sg1 = tier.promote_inc(cpu, "jobA", 0, n, rr[1])
        free_after = cpu.free_count()
        # 二次 promote (如另一请求同前缀): 全 HIT, 不再占新槽
        sg2 = tier.promote_inc(cpu, "jobB", 0, n, rr[1])
        assert sg2 is not None
        assert [s for (s, _g) in sg2] == [s for (s, _g) in sg1]
        assert cpu.free_count() == free_after

    def test_promote_pin_fail_aborts(self, duo):
        cpu, cpu_buf, tier = duo
        n = 3
        toks = _tokens(n, seed=9)
        _store_and_demote(cpu, cpu_buf, tier, "jobA", n, seed=9)
        rr = tier.store.resolve_range(toks, 0)
        recs = list(rr[1])
        # 模拟 claim->load 窗口内 SSD 槽被淘: 直接淘掉 SSD 侧该 job
        assert tier.store._evict_lockfree(999) == n
        sg = tier.promote_inc(cpu, "jobA", 0, n, recs)
        assert sg is None            # 整段放弃, 不半截


# ============================================================
# Group C - 两层闭环
# ============================================================
class TestTwoTierLoop:

    def test_full_cycle(self, duo):
        cpu, cpu_buf, tier = duo
        n = 6
        toks = _tokens(n, seed=11)
        orig = _store_and_demote(cpu, cpu_buf, tier, "jobA", n, seed=11)
        # CPU 全空: lookup_resolve 应 miss, SSD resolve_range 应全命中
        assert cpu.lookup_resolve(toks) is None
        rr = tier.store.resolve_range(toks, 0)
        assert rr is not None and rr[0] == n
        # promote 回 CPU -> CPU 恢复全命中, 字节一致
        sg = tier.promote_inc(cpu, "jobA", 0, n, rr[1])
        assert sg is not None
        res = cpu.lookup_resolve(toks)
        assert res is not None and res[1] == n
        for (slot, _gen), rec in zip(sg, rr[1]):
            assert _slot_bytes_of(cpu_buf, slot) == orig[rec[2]]


# ============================================================
# Group E - enqueue_promote (Phase2 defer 期后台修复, worker 侧模拟)
# ============================================================
class TestBackgroundPromote:

    def test_enqueue_promote_repairs_cpu(self, duo):
        """★ 2026-07-05 定案: 后台修复线程把 SSD 段搬回 CPU, CPU 重新
        可探到 (Phase2 之后按纯 CPU 判定 admit)."""
        cpu, cpu_buf, tier = duo
        n = 4
        toks = _tokens(n, seed=31)
        orig = _store_and_demote(cpu, cpu_buf, tier, "jobA", n, seed=31)
        assert cpu.lookup_resolve(toks) is None      # CPU 空 (被淘)
        rr = tier.store.resolve_range(toks, 0)
        assert rr is not None
        # 模拟 worker: RoundKVStore 壳 + 注入两层 (不需要 GPU)
        import tempfile
        from vllm.v1.core.sched.licht_v3.round_kv_store import RoundKVStore
        rks = RoundKVStore(tempfile.mkdtemp(prefix="rk_bgp_"), BS)
        rks._lru_store = cpu
        rks._ssd_tier = tier
        try:
            assert rks.enqueue_promote("jobA", 0, rr[1])
            t0 = time.time()
            while time.time() - t0 < 5:
                if cpu.lookup_resolve(toks) is not None:
                    break
                time.sleep(0.02)
            res = cpu.lookup_resolve(toks)
            assert res is not None and res[1] == n   # CPU 齐了
            # 字节正确
            for (slot, _g), rec in zip(cpu.probe_slots(
                    block_hashes(toks, BS, n)), rr[1]):
                assert _slot_bytes_of(cpu_buf, slot) == orig[rec[2]]
        finally:
            rks._lru_store = None   # 防 shutdown 关掉共享 store
            rks._ssd_tier = None
            rks.shutdown()


# ============================================================
# Group D - RoundKVStore.lookup_resolve_tiered (scheduler 侧模拟)
# ============================================================
class TestTieredResolve:

    def test_tiered_stitches_cpu_and_ssd(self, tmp_path, monkeypatch):
        _cpu_env(monkeypatch)
        cpu_dir = str(tmp_path / "cpu")
        ssd_meta = str(tmp_path / "ssd_meta")
        ssd_data = str(tmp_path / "ssd_data")
        # 1) 布账本: CPU 存 [0,4), SSD 存 [4,8) (同一条 token 链)
        toks = _tokens(8, seed=13)
        hs = block_hashes(toks, BS, 8)
        cpu = LruArenaStore.create(cpu_dir, num_slots=64, block_size=BS)
        cpu.bind_data_writer(lambda *a: None)
        assert cpu.write_inc("j", 0, 4, toks[:4 * BS], [b"x"] * 4)
        tier = SsdTier.open_or_create(
            meta_path=ssd_meta, data_path=ssd_data,
            ssd_gb=SLOT_BYTES * 64 / (1024 ** 3),
            slot_bytes=SLOT_BYTES, block_size=BS)
        tier.store.bind_data_writer(lambda *a: None)
        assert tier.store.write_inc("j", 4, 8, [], [b"y"] * 4,
                                    inc_hashes=hs[4:8])
        # 2) scheduler 侧 RoundKVStore: 只读表模式 (meta json 引导)
        with open(os.path.join(cpu_dir, "_arena_meta.json"), "w") as f:
            json.dump({"num_slots": 64, "block_size": BS}, f)
        monkeypatch.setenv("LICHT_ROUND_KV_LRU", "1")
        monkeypatch.setenv("LICHT_SSD_TIER", "1")
        monkeypatch.setenv("LICHT_SSD_PATH", ssd_data)
        monkeypatch.setenv("LICHT_SSD_META_PATH", ssd_meta)
        from vllm.v1.core.sched.licht_v3.round_kv_store import RoundKVStore
        rks = RoundKVStore(cpu_dir, BS)
        try:
            res = rks.lookup_resolve_tiered("j", toks)
            assert res is not None
            mt, mb, sg, ssd_seg = res
            assert mb == 8 and mt == 8 * BS       # claim 拼到 8 块
            assert sg is not None and len(sg) == 4  # CPU 段 [0,4)
            assert ssd_seg is not None
            a, recs = ssd_seg
            assert a == 4 and len(recs) == 4      # SSD 段 [4,8)
            assert [r[2] for r in recs] == hs[4:8]
            # inflight 代理可用 (mark + clear 不炸, 文件落在 SSD meta 目录)
            rks.ssd_mark_inflight("j")
            assert tier.store.is_inflight("j")
            rks.ssd_clear_inflight("j")
            assert not tier.store.is_inflight("j")
        finally:
            rks.shutdown()
            tier.close()
            cpu.close()

    def _mk_two_tier(self, tmp_path, monkeypatch, **extra_env):
        """CPU 账本 [0,4) + SSD 账本 [4,8) + scheduler 侧 RoundKVStore."""
        _cpu_env(monkeypatch)
        cpu_dir = str(tmp_path / "cpu")
        ssd_meta = str(tmp_path / "ssd_meta")
        toks = _tokens(8, seed=21)
        hs = block_hashes(toks, BS, 8)
        cpu = LruArenaStore.create(cpu_dir, num_slots=64, block_size=BS)
        cpu.bind_data_writer(lambda *a: None)
        assert cpu.write_inc("j", 0, 4, toks[:4 * BS], [b"x"] * 4)
        tier = SsdTier.open_or_create(
            meta_path=ssd_meta, data_path=str(tmp_path / "ssd_data"),
            ssd_gb=SLOT_BYTES * 64 / (1024 ** 3),
            slot_bytes=SLOT_BYTES, block_size=BS)
        tier.store.bind_data_writer(lambda *a: None)
        assert tier.store.write_inc("j", 4, 8, [], [b"y"] * 4,
                                    inc_hashes=hs[4:8])
        with open(os.path.join(cpu_dir, "_arena_meta.json"), "w") as f:
            json.dump({"num_slots": 64, "block_size": BS}, f)
        monkeypatch.setenv("LICHT_ROUND_KV_LRU", "1")
        monkeypatch.setenv("LICHT_SSD_TIER", "1")
        monkeypatch.setenv("LICHT_SSD_PATH", str(tmp_path / "ssd_data"))
        monkeypatch.setenv("LICHT_SSD_META_PATH", ssd_meta)
        for k, v in extra_env.items():
            monkeypatch.setenv(k, v)
        from vllm.v1.core.sched.licht_v3.round_kv_store import RoundKVStore
        rks = RoundKVStore(cpu_dir, BS)
        return rks, cpu, tier, toks, hs

    def test_fresh_probe_reflects_eviction(self, tmp_path, monkeypatch):
        """★ Fix1 核心: 探测永远反映当刻状态 —— hash 缓存 + 现探,
        SSD 内容被淘后, 同一批 hash 再探必须返回 None (不再有陈旧地图)."""
        rks, cpu, tier, toks, hs = self._mk_two_tier(tmp_path, monkeypatch)
        try:
            tail = rks.ssd_tail_hashes(toks, 4)
            assert tail == hs[4:8]
            seg1 = rks.ssd_probe_fresh("j", 4, tail)
            assert seg1 is not None and len(seg1[1]) == 4
            # 模拟排队期间 SSD 淘汰: 先清 inflight (probe 已挂) 再驱逐
            tier.store.clear_inflight("j")
            assert tier.store._evict_lockfree(999) == 4
            # 同一批缓存 hash, 当步现探 -> 立刻看到"没了"
            assert rks.ssd_probe_fresh("j", 4, tail) is None
        finally:
            rks.shutdown()
            tier.close()
            cpu.close()

    def test_probe_marks_inflight_immediately(self, tmp_path, monkeypatch):
        """★ Fix1: claim 意向 (探中) 即挂 inflight, 不等 admit."""
        rks, cpu, tier, toks, hs = self._mk_two_tier(tmp_path, monkeypatch)
        try:
            assert not tier.store.is_inflight("j")
            seg = rks.ssd_probe_fresh("j", 4, hs[4:8])
            assert seg is not None
            assert tier.store.is_inflight("j")   # 探中瞬间已受保护
            # inflight 生效: 驱逐绕行, 淘不动
            assert tier.store._evict_lockfree(999) == 0
        finally:
            rks.ssd_clear_inflight("j")
            rks.shutdown()
            tier.close()
            cpu.close()

    def test_probe_no_mark_when_disabled(self, tmp_path, monkeypatch):
        """★ 2026-07-05 (prefill inflight 久占修): mark_inflight=False 时探中
        也【不】挂 inflight —— producer claim 期不锁槽, 等待期让别的 KV 复用.
        对照 test_probe_marks_inflight_immediately (默认 True 仍挂)."""
        rks, cpu, tier, toks, hs = self._mk_two_tier(tmp_path, monkeypatch)
        try:
            assert not tier.store.is_inflight("j")
            seg = rks.ssd_probe_fresh("j", 4, hs[4:8], mark_inflight=False)
            assert seg is not None                     # 段照样探中/可搬
            assert not tier.store.is_inflight("j")     # 但没挂 inflight
            # 没挂 -> 驱逐器可以淘走 (等待期腾容量)
            tier.store.clear_inflight("j")             # 幂等
            assert tier.store._evict_lockfree(999) == 4
        finally:
            rks.shutdown()
            tier.close()
            cpu.close()

    def test_benefit_gate_closes_and_forces(self, tmp_path, monkeypatch):
        """★ Fix2: 搬 > 算 -> 不 claim (慢盘常闭 = 正确经济学);
        FACTOR=99 强制放行 (功能验证用).
        ★ 2026-07-05: 收益闸门默认关, 本测试显式开 (GATE=1) 才验证经济学."""
        rks, cpu, tier, toks, hs = self._mk_two_tier(
            tmp_path, monkeypatch, LICHT_SSD_PROMOTE_GATE="1",
            LICHT_SSD_BW_MBPS="1")   # 极慢盘
        try:
            assert rks.ssd_probe_fresh("j", 4, hs[4:8]) is None
            assert rks._stat_promote_gated == 1
        finally:
            rks.shutdown()

        rks2, cpu2, tier2, toks2, hs2 = self._mk_two_tier(
            tmp_path / "b", monkeypatch, LICHT_SSD_PROMOTE_GATE="1",
            LICHT_SSD_BW_MBPS="1", LICHT_SSD_PROMOTE_FACTOR="99")
        try:
            assert rks2.ssd_probe_fresh("j", 4, hs2[4:8]) is not None
        finally:
            rks2.ssd_clear_inflight("j")
            rks2.shutdown()
            tier2.close()
            cpu2.close()
            tier.close()
            cpu.close()

    def test_gate_off_by_default_promotes_on_slow_disk(self, tmp_path,
                                                        monkeypatch):
        """★ 2026-07-05 用户决策: 收益闸门默认关. 极慢盘 (BW=1) 且不显式开门
        时, producer (apply_gate 默认 True) 也应无条件搬回 —— 先保证 SSD 端到端
        复用能跑通, 再谈经济学."""
        rks, cpu, tier, toks, hs = self._mk_two_tier(
            tmp_path, monkeypatch, LICHT_SSD_BW_MBPS="1")   # 极慢盘, 但门关
        try:
            seg = rks.ssd_probe_fresh("j", 4, hs[4:8])   # apply_gate 默认 True
            assert seg is not None and len(seg[1]) == 4  # 门关 -> 照搬
            assert rks._stat_promote_gated == 0          # 从未被闸门拒绝
        finally:
            rks.ssd_clear_inflight("j")
            rks.shutdown()
            tier.close()
            cpu.close()

    def test_gate_bypass_for_consumer(self, tmp_path, monkeypatch):
        """★ Phase2 修: apply_gate=False (decode admission) 无视收益闸门 —
        那边的替代方案是死等超时, 不是重算, 搬永远划算."""
        rks, cpu, tier, toks, hs = self._mk_two_tier(
            tmp_path, monkeypatch, LICHT_SSD_PROMOTE_GATE="1",
            LICHT_SSD_BW_MBPS="1")   # 闸门必关的慢盘
        try:
            assert rks.ssd_probe_fresh("j", 4, hs[4:8]) is None     # 闸门关
            seg = rks.ssd_probe_fresh("j", 4, hs[4:8], apply_gate=False)
            assert seg is not None and len(seg[1]) == 4             # 旁路开
            assert tier.store.is_inflight("j")   # 旁路路径同样挂保护
        finally:
            rks.ssd_clear_inflight("j")
            rks.shutdown()
            tier.close()
            cpu.close()

    def test_tiered_falls_back_when_ssd_off(self, tmp_path, monkeypatch):
        _cpu_env(monkeypatch)
        cpu_dir = str(tmp_path / "cpu")
        toks = _tokens(4, seed=14)
        cpu = LruArenaStore.create(cpu_dir, num_slots=32, block_size=BS)
        cpu.bind_data_writer(lambda *a: None)
        assert cpu.write_inc("j", 0, 4, toks, [b"x"] * 4)
        with open(os.path.join(cpu_dir, "_arena_meta.json"), "w") as f:
            json.dump({"num_slots": 32, "block_size": BS}, f)
        monkeypatch.setenv("LICHT_ROUND_KV_LRU", "1")
        monkeypatch.delenv("LICHT_SSD_TIER", raising=False)
        from vllm.v1.core.sched.licht_v3.round_kv_store import RoundKVStore
        rks = RoundKVStore(cpu_dir, BS)
        try:
            res = rks.lookup_resolve_tiered("j", toks)
            assert res is not None
            mt, mb, sg, ssd_seg = res
            assert mb == 4 and ssd_seg is None    # 纯 CPU, 4 元组语义不变
        finally:
            rks.shutdown()
            cpu.close()
