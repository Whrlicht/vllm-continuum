# SPDX-License-Identifier: Apache-2.0
"""P1 demote-ahead (跨层降级) 测试, CPU-only.

测试矩阵:

Group A - LruArenaStore 降级基建 (假 demote_fn):
  扫描/入队/pin 配对/写线程标干净/inflight-finished 跳过/队列有界
Group B - 驱逐闸门:
  脏 inc 跳过 / 干净 inc 放行 + 标记作废 / 紧急旁路 (clean_gate=False)
Group C - SsdTier.demote_inc 数据面 (真文件):
  逐字节比对 / dedup 第二次零写 / 布局不符拒绝
Group D - 端到端: CPU store -> 扫描 -> SSD 文件, 字节一致
"""
import mmap
import os
import threading
import time

import pytest

try:
    import licht_arena_atomic as A
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.arena_block_hash import block_hashes
from vllm.v1.core.sched.licht_v3.arena_lru_store import LruArenaStore
from vllm.v1.core.sched.licht_v3.ssd_tier import SsdTier

BS = 16            # tokens per block
SLOT_BYTES = 4096  # 测试用小 slot


def _wait(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _make_store(tmp_path, monkeypatch, num_slots=64, qdepth=8):
    monkeypatch.setenv("LICHT_ARENA_CONTENT_ADDR", "1")
    monkeypatch.setenv("LICHT_ARENA_BG_EVICTOR", "0")  # 手动调扫描, 确定性
    monkeypatch.setenv("LICHT_SSD_QUEUE", str(qdepth))
    monkeypatch.setenv("LICHT_SSD_WRITERS", "1")
    store = LruArenaStore.create(str(tmp_path / "cpu_arena"),
                                 num_slots=num_slots, block_size=BS)
    store.bind_data_writer(lambda slot, i, src: None)  # Group A/B 不看数据
    return store


def _tokens(n_blocks, seed=0):
    return [seed * 100000 + i for i in range(n_blocks * BS)]


def _write_job(store, job, n_blocks, seed=0, start=0):
    """写一个 inc, 数据用可校验的 bytes."""
    blocks = [f"{job}-blk{start + i}".encode().ljust(16, b".")
              for i in range(n_blocks - start)]
    ok = store.write_inc(job, start, n_blocks, _tokens(n_blocks, seed),
                         blocks)
    assert ok
    return blocks


def _get_pin(store, slot):
    return A.get_pin(store._hdr.slot_state_addr(slot))


def _setup_capture(tmp_path, monkeypatch, num_slots=64, **env):
    """CPU arena + SsdTier + 假 CPU buffer, 绑 capture-at-eviction.

    CPU store 的 data_writer 往假 mmap 写真字节; SsdTier 从同一 buffer 读.
    """
    monkeypatch.setenv("LICHT_ARENA_CONTENT_ADDR", "1")
    monkeypatch.setenv("LICHT_ARENA_BG_EVICTOR", "0")   # 手动驱逐, 确定性
    # 每测试独立 ring 目录 (环名带 pid, 单进程多测试会撞名)
    _rd = str(tmp_path / "ring")
    __import__("os").makedirs(_rd, exist_ok=True)
    monkeypatch.setenv("LICHT_SSD_RING_DIR", _rd)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    store = LruArenaStore.create(str(tmp_path / "cpu_arena"),
                                 num_slots=num_slots, block_size=BS)
    cpu, _ = _fake_cpu_arena(num_slots)

    def cpu_writer(slot_id, i, src):
        data = (src[i] * (SLOT_BYTES // len(src[i]) + 1))[:SLOT_BYTES]
        cpu[slot_id * SLOT_BYTES:(slot_id + 1) * SLOT_BYTES] = data

    store.bind_data_writer(cpu_writer)
    tier = SsdTier.open_or_create(
        meta_path=str(tmp_path / "ssd_meta"),
        data_path=str(tmp_path / "ssd_data"),
        ssd_gb=SLOT_BYTES * num_slots / (1024 ** 3),
        slot_bytes=SLOT_BYTES, block_size=BS)
    tier.bind_cpu_source(cpu, SLOT_BYTES)
    store.bind_demote_fn(tier.capture_inc)   # capture-at-eviction (零 pin)
    return store, cpu, tier


# ============================================================
# Group A - capture-at-eviction (2026-07-06, 取代 demote-ahead+pin)
# ============================================================
class TestCapture:
    """驱逐【释放块前】capture memmove 进 SHM 环 (零 pin, 纯 LRU 驱逐);
    独立写进程 (测试里用 tier.drain_ring 同步模拟) 从环 drain -> SSD.
    Q2: 写路径搬出 decode 进程 (2026-07-06)."""

    def test_capture_ring_to_ssd_no_pin(self, tmp_path, monkeypatch):
        store, cpu, tier = _setup_capture(tmp_path, monkeypatch)
        n = 4
        _write_job(store, "jobA", n, seed=7)
        recs = store._job_slot_index["jobA"][0][2]
        orig = {r[2]: bytes(cpu[r[0] * SLOT_BYTES:(r[0] + 1) * SLOT_BYTES])
                for r in recs}
        # 纯 LRU 驱逐: 全部释放 (capture 不 pin -> 无 left_pinned -> gained==n)
        assert store._evict_lockfree(999) == n
        assert tier._stat_capture_ok >= 1              # 已推进环
        # 模拟写进程: drain 环 -> 写 SSD
        assert tier.drain_ring(tier._ring) >= 1
        assert tier._stat_demote_blocks >= n
        # 逐字节: SSD 上按内容 hash 找回, 与原 CPU 字节一致
        with open(tier.data_file, "rb") as f:
            for h, want in orig.items():
                s_slot = A.ht_probe(tier.store._ht_base,
                                    tier.store._ht_cap, h)[0]
                assert s_slot >= 0, "内容 hash 应能在 SSD 账本中找到"
                f.seek(tier.slot_offset(s_slot))
                assert f.read(SLOT_BYTES) == want
        tier.close()
        store.close()

    def test_capture_chunks_large_inc(self, tmp_path, monkeypatch):
        """★ 大 inc 切块存, 不整段丢 (修 96% drop 根因): chunk=2, 写 5 块的
        inc -> 切成 3 段 (2+2+1) 推环, drain 后全部进 SSD, 逐字节一致."""
        store, cpu, tier = _setup_capture(
            tmp_path, monkeypatch, LICHT_SSD_STAGE_CHUNK_BLK="2")
        assert tier._stage_chunk == 2
        n = 5
        _write_job(store, "jobBig", n, seed=3)
        recs = store._job_slot_index["jobBig"][0][2]
        orig = {r[2]: bytes(cpu[r[0] * SLOT_BYTES:(r[0] + 1) * SLOT_BYTES])
                for r in recs}
        assert store._evict_lockfree(999) == n        # 纯 LRU, 全释放
        assert tier._stat_capture_ok == 3             # 2+2+1 三段
        assert tier._stat_capture_blocks == n
        assert tier._stat_capture_drop == 0           # 一块没丢
        assert tier.drain_ring(tier._ring) == 3       # drain 3 段
        assert tier._stat_demote_blocks >= n
        with open(tier.data_file, "rb") as f:         # 5 块逐字节都在 SSD
            for h, want in orig.items():
                s_slot = A.ht_probe(tier.store._ht_base,
                                    tier.store._ht_cap, h)[0]
                assert s_slot >= 0
                f.seek(tier.slot_offset(s_slot))
                assert f.read(SLOT_BYTES) == want
        tier.close()
        store.close()

    def test_capture_ring_full_drops_but_evict_unaffected(self, tmp_path,
                                                          monkeypatch):
        """环满 -> capture 丢; 但驱逐照常全部释放 (LRU 不受 capture 影响).
        不 drain (环槽永不释放) -> 环 (2 槽) 推满后必丢."""
        store, cpu, tier = _setup_capture(
            tmp_path, monkeypatch,
            LICHT_SSD_STAGE_CHUNK_BLK="1", LICHT_SSD_STAGE_MB="0")
        assert tier._ring_n == 2                   # 预算 0 -> 环压到最小 2 槽
        # 环只有 2 槽; 驱逐 4 个 job × 2 块 = 8 段, 推满 2 个后全丢
        for i in range(4):
            _write_job(store, f"job{i}", 2, seed=i)
        assert store._evict_lockfree(999) == 8    # 驱逐全部释放, 不被拖
        assert tier._stat_capture_ok == 2          # 环 2 槽, 只进 2
        assert tier._stat_capture_drop >= 1        # 其余丢
        tier.close()
        store.close()

    def test_capture_skips_finished(self, tmp_path, monkeypatch):
        """已结束 job 不 capture (不会回来, 不值得写盘); 驱逐照常释放."""
        store, cpu, tier = _setup_capture(tmp_path, monkeypatch)
        _write_job(store, "jobFin", 2, seed=1)
        store._finished_jobs.add("jobFin")
        assert store._evict_lockfree(999) == 2
        assert tier._stat_capture_ok == 0
        store._finished_jobs.discard("jobFin")
        tier.close()
        store.close()


# ============================================================
# Group B - 驱逐闸门
# ============================================================
class TestEvictGate:
    # NB: test_gate_skips_dirty_frees_clean 已删除 (2026-07-06): 驱逐闸门依赖
    # demote-ahead 填充的 clean-set, 该机制已被 capture-at-eviction 取代;
    # 闸门默认关 (LICHT_SSD_EVICT_GATE), _clean_incs 不再被填充. 保留下面两
    # 个直接测 _evict_lockfree(clean_gate=) 参数语义的用例 (与 demote 无关).

    def test_gate_off_frees_dirty(self, tmp_path, monkeypatch):
        """clean_gate=False (默认): 纯 LRU, 全部释放 (今天的语义)."""
        store = _make_store(tmp_path, monkeypatch)
        _write_job(store, "jobA", 4, seed=1)
        gained = store._evict_lockfree(999, clean_gate=False)
        assert gained == 4
        store.close()

    def test_gate_lets_finished_pass(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        store.bind_demote_fn(lambda *a: True)
        _write_job(store, "jobA", 4, seed=1)
        store._finished_jobs.add("jobA")        # 已结束: 脏也放行
        gained = store._evict_lockfree(999, clean_gate=True)
        assert gained == 4
        store._finished_jobs.discard("jobA")
        store.close()


# ============================================================
# Group C - SsdTier 数据面
# ============================================================
@pytest.fixture
def ssd(tmp_path, monkeypatch):
    monkeypatch.setenv("LICHT_ARENA_CONTENT_ADDR", "1")
    monkeypatch.setenv("LICHT_ARENA_BG_EVICTOR", "0")
    tier = SsdTier.open_or_create(
        meta_path=str(tmp_path / "ssd_meta"),
        data_path=str(tmp_path / "ssd_data"),
        ssd_gb=SLOT_BYTES * 64 / (1024 ** 3),   # 64 slots
        slot_bytes=SLOT_BYTES, block_size=BS)
    yield tier
    tier.close()


def _fake_cpu_arena(n_slots):
    """匿名 mmap 当 CPU arena; 返回 (buf, fill(slot, pattern))."""
    buf = mmap.mmap(-1, n_slots * SLOT_BYTES)

    def fill(slot, pattern: bytes):
        data = (pattern * (SLOT_BYTES // len(pattern) + 1))[:SLOT_BYTES]
        buf[slot * SLOT_BYTES:(slot + 1) * SLOT_BYTES] = data
        return data

    return buf, fill


class TestSsdDataPlane:

    def test_demote_bytes_roundtrip(self, ssd):
        cpu, fill = _fake_cpu_arena(8)
        ssd.bind_cpu_source(cpu, SLOT_BYTES)
        want = [fill(i, f"D{i}".encode()) for i in range(3)]
        records = [(i, 1, 1000 + i) for i in range(3)]  # (cpu_slot,gen,hash)
        assert ssd.demote_inc("jobA", 0, 3, records)
        # 逐字节校验: 按 hash 从 SSD 账本找到 slot, pread 文件比对
        with open(ssd.data_file, "rb") as f:
            for i in range(3):
                s_slot = A.ht_probe(ssd.store._ht_base, ssd.store._ht_cap,
                                    1000 + i)[0]
                assert s_slot >= 0
                f.seek(ssd.slot_offset(s_slot))
                assert f.read(SLOT_BYTES) == want[i]
        assert ssd.stats()["demote_blocks"] == 3

    def test_demote_dedup_second_time_zero_io(self, ssd):
        cpu, fill = _fake_cpu_arena(8)
        ssd.bind_cpu_source(cpu, SLOT_BYTES)
        for i in range(3):
            fill(i, f"D{i}".encode())
        records = [(i, 1, 2000 + i) for i in range(3)]
        assert ssd.demote_inc("jobA", 0, 3, records)
        assert ssd.stats()["demote_blocks"] == 3
        # 另一个 job 降级同内容 (同 hash): 全 HIT, 零字节写盘
        assert ssd.demote_inc("jobB", 0, 3, records)
        st = ssd.stats()
        assert st["demote_blocks"] == 3          # 没涨
        assert st["demote_hit_blocks"] == 3
        # 账本只占 3 个 slot (refcnt=2), 不是 6 个
        assert st["free"] == ssd.num_slots - 3

    def test_bind_rejects_mismatched_slot_bytes(self, ssd):
        cpu, _ = _fake_cpu_arena(2)
        with pytest.raises(ValueError):
            ssd.bind_cpu_source(cpu, SLOT_BYTES * 2)


# ============================================================
# Group D - 端到端: CPU store -> 扫描/写线程 -> SSD 文件
# ============================================================
class TestEndToEnd:

    def test_cpu_evict_pressure_demotes_to_ssd(self, tmp_path, monkeypatch):
        _rd = str(tmp_path / "ring")
        __import__("os").makedirs(_rd, exist_ok=True)
        monkeypatch.setenv("LICHT_SSD_RING_DIR", _rd)
        store = _make_store(tmp_path, monkeypatch, num_slots=64)
        tier = SsdTier.open_or_create(
            meta_path=str(tmp_path / "ssd_meta"),
            data_path=str(tmp_path / "ssd_data"),
            ssd_gb=SLOT_BYTES * 64 / (1024 ** 3),
            slot_bytes=SLOT_BYTES, block_size=BS)
        # 假 CPU arena: CPU store 的 data_writer 往 mmap 的 slot 偏移写真字节
        cpu, _ = _fake_cpu_arena(64)

        def cpu_writer(slot_id, i, src):
            data = (src[i] * (SLOT_BYTES // len(src[i]) + 1))[:SLOT_BYTES]
            cpu[slot_id * SLOT_BYTES:(slot_id + 1) * SLOT_BYTES] = data

        store.bind_data_writer(cpu_writer)
        tier.bind_cpu_source(cpu, SLOT_BYTES)
        store.bind_demote_fn(tier.capture_inc)   # capture-at-eviction

        n_blocks = 4
        _write_job(store, "jobA", n_blocks, seed=7)
        # 记录原始 CPU 字节 (按索引里的 slot)
        recs = store._job_slot_index["jobA"][0][2]
        orig = {r[2]: bytes(cpu[r[0] * SLOT_BYTES:(r[0] + 1) * SLOT_BYTES])
                for r in recs}

        # 纯 LRU 驱逐: 释放前 capture memmove 进环; drain (模拟写进程) -> SSD
        gained = store._evict_lockfree(999)
        assert gained == n_blocks
        assert tier.drain_ring(tier._ring) >= 1
        assert tier._stat_demote_blocks >= n_blocks
        with open(tier.data_file, "rb") as f:
            for h, want in orig.items():
                s_slot = A.ht_probe(tier.store._ht_base, tier.store._ht_cap,
                                    h)[0]
                assert s_slot >= 0, "内容 hash 应能在 SSD 账本中找到"
                f.seek(tier.slot_offset(s_slot))
                assert f.read(SLOT_BYTES) == want
        tier.close()
        store.close()
