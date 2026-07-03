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


# ============================================================
# Group A - 降级基建
# ============================================================
class TestDemoteInfra:

    def test_scan_demote_marks_clean_and_unpins(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        calls = []

        def fake_demote(job, s, e, records):
            # 调用时 inc 的每个 slot 必须处于 pin 状态 (数据受保护)
            assert all(_get_pin(store, r[0]) >= 1 for r in records)
            calls.append((job, s, e, [r[2] for r in records]))
            return True

        store.bind_demote_fn(fake_demote)
        _write_job(store, "jobA", 4, seed=1)
        store._demote_scan()
        assert _wait(lambda: store._stat_demote_ok == 1)
        assert len(calls) == 1 and calls[0][0] == "jobA"
        # 洗完: 干净集合有键, 全部 unpin
        assert len(store._clean_incs) == 1
        idx = store._job_slot_index["jobA"][0][2]
        assert all(_get_pin(store, r[0]) == 0 for r in idx)
        # 重复扫描: 已干净, 不再入队
        store._demote_scan()
        time.sleep(0.1)
        assert store._stat_demote_ok == 1
        store.close()

    def test_demote_fail_not_marked_clean(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        store.bind_demote_fn(lambda *a: False)
        _write_job(store, "jobA", 2, seed=1)
        store._demote_scan()
        assert _wait(lambda: store._stat_demote_fail == 1)
        assert len(store._clean_incs) == 0
        # 失败后 inflight 已清 -> 可重试
        store._demote_scan()
        assert _wait(lambda: store._stat_demote_fail == 2)
        store.close()

    def test_scan_skips_inflight_and_finished(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        seen = []
        store.bind_demote_fn(
            lambda job, s, e, r: (seen.append(job), True)[1])
        _write_job(store, "jobIn", 2, seed=1)
        _write_job(store, "jobFin", 2, seed=2)
        _write_job(store, "jobOk", 2, seed=3)
        store.mark_inflight("jobIn")
        store._finished_jobs.add("jobFin")   # 模拟 mark->delete 窗口
        store._demote_scan()
        assert _wait(lambda: store._stat_demote_ok == 1)
        assert seen == ["jobOk"]
        store.clear_inflight("jobIn")
        store.close()

    def test_queue_bounded(self, tmp_path, monkeypatch):
        """队列有限深: 写线程被堵住时, 扫描不无界入队."""
        store = _make_store(tmp_path, monkeypatch, qdepth=1)
        gate = threading.Event()
        store.bind_demote_fn(lambda *a: gate.wait(10) or True)
        for i in range(4):
            _write_job(store, f"job{i}", 2, seed=i)
        store._demote_scan()   # 1 个进写线程堵住, 1 个占队列, 其余留脏
        time.sleep(0.2)
        with store._clean_lock:
            n_inflight = len(store._demote_inflight)
        assert n_inflight <= 2
        gate.set()
        # 反复扫描最终全部洗净
        assert _wait(lambda: (store._demote_scan() or
                              store._stat_demote_ok == 4), timeout=8)
        store.close()


# ============================================================
# Group B - 驱逐闸门
# ============================================================
class TestEvictGate:

    def test_gate_skips_dirty_frees_clean(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        store.bind_demote_fn(lambda *a: True)
        _write_job(store, "jobA", 4, seed=1)   # 将洗净
        _write_job(store, "jobB", 4, seed=2)   # 保持脏
        # 只把 jobA 洗净 (手工构造: 扫描全部, 但 fn 只对 A 成功)
        store._demote_fn = lambda job, s, e, r: job == "jobA"
        store._demote_scan()
        assert _wait(lambda: store._stat_demote_ok + store._stat_demote_fail
                     == 2)
        free0 = store.free_count()
        gained = store._evict_lockfree(999, clean_gate=True)
        # 只有 jobA 的 4 块被释放; jobB 脏 -> gate 跳过
        assert gained == 4
        assert store.free_count() == free0 + 4
        assert store._stat_gate_skip >= 1
        # jobA 的干净标记已随释放作废
        assert all(k[0] != "jobA" for k in store._clean_incs)
        store.close()

    def test_gate_off_frees_dirty(self, tmp_path, monkeypatch):
        """紧急旁路 (clean_gate=False): 脏 inc 也放行 = 今天的丢弃语义."""
        store = _make_store(tmp_path, monkeypatch)
        store.bind_demote_fn(lambda *a: True)   # 有降级基建但没洗过
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
        store.bind_demote_fn(tier.demote_inc)

        n_blocks = 4
        _write_job(store, "jobA", n_blocks, seed=7)
        # 记录原始 CPU 字节 (按索引里的 slot)
        recs = store._job_slot_index["jobA"][0][2]
        orig = {r[2]: bytes(cpu[r[0] * SLOT_BYTES:(r[0] + 1) * SLOT_BYTES])
                for r in recs}

        store._demote_scan()
        assert _wait(lambda: store._stat_demote_ok == 1)

        # 洗净后闸门放行 -> CPU 释放; SSD 上按 hash 找回并逐字节比对
        gained = store._evict_lockfree(999, clean_gate=True)
        assert gained == n_blocks
        with open(tier.data_file, "rb") as f:
            for h, want in orig.items():
                s_slot = A.ht_probe(tier.store._ht_base, tier.store._ht_cap,
                                    h)[0]
                assert s_slot >= 0, "内容 hash 应能在 SSD 账本中找到"
                f.seek(tier.slot_offset(s_slot))
                assert f.read(SLOT_BYTES) == want
        tier.close()
        store.close()
