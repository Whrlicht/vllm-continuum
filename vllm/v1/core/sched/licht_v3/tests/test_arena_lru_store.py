# SPDX-License-Identifier: Apache-2.0
"""LruArenaStore 集成测试.

测试矩阵 (CPU only, 不依赖 GPU):

Group A - 基本 write/lookup/load 回路
Group B - 多 inc 累积写入
Group C - LRU 淘汰 (tail-first)
Group D - Self-heal (_last_stored 回退 + manifest 截断)
Group E - Pin 保护: 正在 load 的 slot 不被 evict
Group F - 多 job 共存
"""
import os
import time

import pytest

try:
    import licht_arena_atomic  # noqa: F401
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)

from vllm.v1.core.sched.licht_v3.arena_lru_store import (
    LruArenaStore,
    align_blocks,
)


# ============================================================
# 测试用的 "假数据存储": 模拟 arena_view, 一个 dict[slot_id] = data
# ============================================================
class _FakeArena:
    """假 arena_view: slot_id → 数据 (任何对象)."""
    def __init__(self):
        self.slots: dict[int, object] = {}

    def writer(self, slot_id: int, block_idx_in_src: int, src):
        """传给 LruArenaStore.bind_data_writer.

        src 是 list, src[block_idx] 是该 block 的数据 (任意 Python 对象).
        """
        self.slots[slot_id] = src[block_idx_in_src]

    def read(self, slot_id: int) -> object:
        return self.slots.get(slot_id)


@pytest.fixture
def store_and_arena(tmp_path):
    """LruArenaStore + FakeArena, num_slots=100, block_size=16."""
    arena = _FakeArena()
    store = LruArenaStore.create(str(tmp_path / "arena"),
                                 num_slots=100, block_size=16)
    store.bind_data_writer(arena.writer)
    yield store, arena
    store.close()


def _make_tokens(n: int, seed: int = 0) -> list[int]:
    """生成 n 个 token id, seed 不同则内容不同."""
    return [(seed * 1000 + i) for i in range(n)]


def _block_data(job_id: str, start: int, end: int) -> list[str]:
    """生成 inc 内每 block 的内容 (字符串方便比较)."""
    return [f"{job_id}_blk{i}" for i in range(start, end)]


# ============================================================
# Group A - 基本回路
# ============================================================

class TestGroupABasic:
    def test_write_then_lookup_hits(self, store_and_arena):
        store, arena = store_and_arena
        tokens = _make_tokens(100)  # 100 token = 6.25 block, 整 6 个
        # 写 [0, 6)
        ok = store.write_inc(
            job_id="job1", start_block=0, end_block=6,
            token_ids=tokens,
            source_obj=_block_data("job1", 0, 6))
        assert ok
        # lookup 同样的 token 应命中 6 个 block (96 token)
        result = store.lookup("job1", tokens)
        assert result is not None
        matched_tokens, matched_blocks = result
        assert matched_blocks == 6
        assert matched_tokens == 96

    def test_lookup_partial_prompt_match(self, store_and_arena):
        store, arena = store_and_arena
        tokens = _make_tokens(100, seed=1)
        store.write_inc("job1", 0, 6, tokens, _block_data("job1", 0, 6))
        # 用前 80 token 查, 应只命中 5 个 block (80//16=5)
        result = store.lookup("job1", tokens[:80])
        assert result is not None
        _, matched_blocks = result
        assert matched_blocks == 5

    def test_lookup_different_job_no_match(self, store_and_arena):
        store, arena = store_and_arena
        tokens = _make_tokens(100, seed=1)
        store.write_inc("job1", 0, 6, tokens, _block_data("job1", 0, 6))
        assert store.lookup("job2", tokens) is None

    def test_lookup_different_prompt_no_match(self, store_and_arena):
        store, arena = store_and_arena
        tokens1 = _make_tokens(100, seed=1)
        store.write_inc("job1", 0, 6, tokens1, _block_data("job1", 0, 6))
        # 完全不同的 token (第 0 个就不同)
        tokens2 = _make_tokens(100, seed=2)
        assert store.lookup("job1", tokens2) is None

    def test_load_request_returns_correct_slots(self, store_and_arena):
        store, arena = store_and_arena
        tokens = _make_tokens(100, seed=1)
        store.write_inc("job1", 0, 6, tokens, _block_data("job1", 0, 6))
        # 加载 [0, 4) 到 GPU dst block [100, 101, 102, 103]
        dst_blocks = [100, 101, 102, 103]
        handle = store.load_request("job1",
                                    dst_block_ids=dst_blocks,
                                    src_block_offset=0)
        assert handle is not None
        assert len(handle.slot_ids) == 4
        assert handle.dst_block_ids == dst_blocks
        # 验证 arena 数据
        expected = _block_data("job1", 0, 6)[:4]
        actual = [arena.read(sid) for sid in handle.slot_ids]
        assert actual == expected
        # post-load 校验
        assert handle.post_load_validate() is True
        handle.release()

    def test_load_request_miss_returns_none(self, store_and_arena):
        store, _ = store_and_arena
        # 没写任何东西就 load
        handle = store.load_request("nonexistent",
                                    dst_block_ids=[0], src_block_offset=0)
        assert handle is None


# ============================================================
# Group B - 多 inc 累积
# ============================================================

class TestGroupBMultiInc:
    def test_two_incs_cover_continuous_range(self, store_and_arena):
        store, arena = store_and_arena
        tokens = _make_tokens(160)  # 10 block
        # 第 1 inc: [0, 5)
        store.write_inc("job1", 0, 5, tokens, _block_data("job1", 0, 5))
        # 第 2 inc: [5, 10)
        store.write_inc("job1", 5, 10, tokens, _block_data("job1", 5, 10))
        # lookup 应命中全部 10 block
        result = store.lookup("job1", tokens)
        assert result is not None
        _, matched_blocks = result
        assert matched_blocks == 10

    def test_load_across_two_incs(self, store_and_arena):
        store, arena = store_and_arena
        tokens = _make_tokens(160)
        store.write_inc("job1", 0, 5, tokens, _block_data("job1", 0, 5))
        store.write_inc("job1", 5, 10, tokens, _block_data("job1", 5, 10))
        # load [3, 8) 跨两 inc
        dst = [200 + i for i in range(5)]
        handle = store.load_request("job1", dst, src_block_offset=3)
        assert handle is not None
        assert len(handle.slot_ids) == 5
        expected = _block_data("job1", 0, 10)[3:8]
        actual = [arena.read(sid) for sid in handle.slot_ids]
        assert actual == expected
        handle.release()


# ============================================================
# Group C - LRU 淘汰 (tail-first)
# ============================================================

class TestGroupCEviction:
    def test_evict_lru_job_when_full(self, tmp_path):
        """arena 满, 第三个 job 写入会触发 LRU 淘汰最老 job."""
        arena = _FakeArena()
        # arena 容量 6 slot, 每 job 一个 3-block inc
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=6, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            tokens = _make_tokens(48)
            assert store.write_inc("jobA", 0, 3, tokens,
                                   _block_data("jobA", 0, 3))
            time.sleep(0.01)  # 让 mtime 有差
            assert store.write_inc("jobB", 0, 3, tokens,
                                   _block_data("jobB", 0, 3))
            # 此时 arena 满 (6/6)
            assert store.free_count() == 0
            # 第 3 个 job, 应触发 LRU 淘汰 jobA (最老)
            time.sleep(0.01)
            assert store.write_inc("jobC", 0, 3, tokens,
                                   _block_data("jobC", 0, 3))
            # jobA 的 lookup 应失败 (manifest 已被 self-heal, total_blocks=0)
            res_a = store.lookup("jobA", tokens)
            assert res_a is None
            # jobB 和 jobC 仍然 ok
            assert store.lookup("jobB", tokens) is not None
            assert store.lookup("jobC", tokens) is not None
        finally:
            store.close()

    def test_tail_first_within_job(self, tmp_path):
        """单个 job 多 inc, 适量空间需求时只淘 tail inc, 保留 head incs."""
        arena = _FakeArena()
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=10, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            tokens = _make_tokens(160)
            # 写 3 inc: [0,3), [3,6), [6,9), 共 9 slot, 还剩 1
            store.write_inc("jobA", 0, 3, tokens, _block_data("jobA", 0, 3))
            time.sleep(0.005)
            store.write_inc("jobA", 3, 6, tokens, _block_data("jobA", 3, 6))
            time.sleep(0.005)
            store.write_inc("jobA", 6, 9, tokens, _block_data("jobA", 6, 9))
            assert store.free_count() == 1
            # jobB 写 2 block: need_more = 2-1 = 1, 只需淘 tail inc
            time.sleep(0.005)
            store.write_inc("jobB", 0, 2, tokens, _block_data("jobB", 0, 2))
            # jobA 的 tail inc [6,9) 被淘, lookup 应命中前 6 block
            result = store.lookup("jobA", tokens)
            assert result is not None
            _, matched_blocks = result
            assert matched_blocks == 6, f"expected 6 got {matched_blocks}"
            # 同时 jobB 也命中 2 block
            result_b = store.lookup("jobB", tokens)
            assert result_b is not None
            _, matched_b = result_b
            assert matched_b == 2
        finally:
            store.close()

    def test_evicts_multiple_incs_when_needed(self, tmp_path):
        """空间需求大时一次淘多个 inc, _last_stored 回退到最早被淘的 inc.start."""
        arena = _FakeArena()
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=10, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            tokens = _make_tokens(160)
            store.write_inc("jobA", 0, 3, tokens, _block_data("jobA", 0, 3))
            time.sleep(0.005)
            store.write_inc("jobA", 3, 6, tokens, _block_data("jobA", 3, 6))
            time.sleep(0.005)
            store.write_inc("jobA", 6, 9, tokens, _block_data("jobA", 6, 9))
            # need 5 -> 淘 2 个 inc (3+3=6 释放, > 4 need_more)
            time.sleep(0.005)
            store.write_inc("jobB", 0, 5, tokens, _block_data("jobB", 0, 5))
            # _last_stored 回退到最早被淘 inc.start = 3
            assert store._last_stored["jobA"] == 3
            # jobA 只剩 [0,3) 可复用
            result = store.lookup("jobA", tokens)
            assert result is not None
            _, matched_blocks = result
            assert matched_blocks == 3
        finally:
            store.close()


# ============================================================
# Group D - Self-heal
# ============================================================

class TestGroupDSelfHeal:
    def test_last_stored_rolls_back_on_evict(self, tmp_path):
        """tail inc 被淘后, _last_stored 同步回退."""
        arena = _FakeArena()
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=10, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            tokens = _make_tokens(320)
            # jobA: 3 inc 各 3 block, _last_stored=9
            store.write_inc("jobA", 0, 3, tokens, _block_data("jobA", 0, 3))
            time.sleep(0.005)
            store.write_inc("jobA", 3, 6, tokens, _block_data("jobA", 3, 6))
            time.sleep(0.005)
            store.write_inc("jobA", 6, 9, tokens, _block_data("jobA", 6, 9))
            # jobB 来抢 2 slot, 只淘 jobA tail inc (6-9)
            time.sleep(0.005)
            store.write_inc("jobB", 0, 2, tokens, _block_data("jobB", 0, 2))
            # 验证 jobA 的 _last_stored 回退到 6 (tail inc.start)
            assert store._last_stored["jobA"] == 6
            # 验证 manifest 的 total_blocks 也回退
            manifest_a = store._read_manifest("jobA")
            assert manifest_a["total_blocks"] == 6
        finally:
            store.close()

    def test_self_heal_allows_seamless_next_inc(self, tmp_path):
        """self-heal 后下一轮 store 续接, lookup 应命中全部."""
        arena = _FakeArena()
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=12, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            tokens = _make_tokens(320)
            # 4 inc 各 3 block, 共 12 slot, 全满
            for k in range(4):
                store.write_inc("jobA", k * 3, (k + 1) * 3,
                                tokens, _block_data("jobA", k * 3, (k + 1) * 3))
                time.sleep(0.005)
            assert store.free_count() == 0
            assert store._last_stored["jobA"] == 12
            # 另一 job 来抢, 淘 jobA tail
            store.write_inc("jobB", 0, 3, tokens, _block_data("jobB", 0, 3))
            # jobA 的 _last_stored 应回退 (具体值看淘了几 inc)
            new_last = store._last_stored["jobA"]
            assert new_last < 12
            # jobA 再来一轮 store, 从 new_last 续 → 应继续覆盖到更高 block
            store.write_inc("jobA", new_last, 12,
                            tokens, _block_data("jobA", new_last, 12))
            # 现在 jobA 的 lookup 应命中全部 12 block
            result = store.lookup("jobA", tokens)
            assert result is not None
            _, matched_blocks = result
            assert matched_blocks == 12
        finally:
            store.close()


# ============================================================
# Group E - Pin 保护
# ============================================================

class TestGroupEPin:
    def test_load_handle_pins_prevent_evict(self, tmp_path):
        """正在 load 的 slot, 即使该 job 是 LRU 受害者, 也不被 evict."""
        arena = _FakeArena()
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=6, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            tokens = _make_tokens(48)
            store.write_inc("jobA", 0, 3, tokens, _block_data("jobA", 0, 3))
            time.sleep(0.005)
            store.write_inc("jobB", 0, 3, tokens, _block_data("jobB", 0, 3))
            # jobA load 中 (pin)
            handle = store.load_request("jobA", [100, 101, 102], 0)
            assert handle is not None
            # 此时 jobC 想写, 触发 evict
            time.sleep(0.005)
            ok = store.write_inc("jobC", 0, 3, tokens,
                                 _block_data("jobC", 0, 3))
            # 应该成功 — 但 evict 不会动 jobA pin 住的 slot, 改去淘 jobB
            assert ok
            # jobA 仍然 valid (pin 保护)
            res_a = store.lookup("jobA", tokens)
            assert res_a is not None
            # 验证 jobA 的 post-load 校验仍通过
            assert handle.post_load_validate() is True
            handle.release()
        finally:
            store.close()


# ============================================================
# Group F - 多 job 共存
# ============================================================

class TestGroupFMultiJob:
    def test_many_jobs_in_one_arena(self, tmp_path):
        """100 个 job, 每个 1 inc 各 2 block, num_slots=300 (一半空闲)."""
        arena = _FakeArena()
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=300, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            for j in range(100):
                tokens = _make_tokens(32, seed=j)
                ok = store.write_inc(f"job{j}", 0, 2, tokens,
                                     _block_data(f"job{j}", 0, 2))
                assert ok
            # 每个 job lookup 应命中
            for j in range(100):
                tokens = _make_tokens(32, seed=j)
                result = store.lookup(f"job{j}", tokens)
                assert result is not None, f"job{j} miss"
                _, matched = result
                assert matched == 2
            assert store.free_count() == 100  # 300 - 200 used
            assert store.num_jobs() == 100
        finally:
            store.close()

    def test_delete_job_releases_slots(self, tmp_path):
        arena = _FakeArena()
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=100, block_size=16)
        store.bind_data_writer(arena.writer)
        try:
            tokens = _make_tokens(80)
            store.write_inc("jobA", 0, 5, tokens,
                            _block_data("jobA", 0, 5))
            store.write_inc("jobB", 0, 5, tokens,
                            _block_data("jobB", 0, 5))
            assert store.free_count() == 90
            store.delete_job("jobA")
            assert store.free_count() == 95
            # jobA lookup 失败
            assert store.lookup("jobA", tokens) is None
            # jobB 不受影响
            assert store.lookup("jobB", tokens) is not None
        finally:
            store.close()


# ============================================================
# Group G - 跨进程 race (open_or_create)
# ============================================================

class TestGroupGOpenOrCreate:
    def test_open_or_create_single_process(self, tmp_path):
        """单进程: open_or_create 应等同于 create."""
        store = LruArenaStore.open_or_create(
            str(tmp_path / "arena"),
            num_slots=100, block_size=16)
        try:
            assert store.free_count() == 100  # 全 free
            assert store.num_slots == 100
        finally:
            store.close()

    def test_open_or_create_second_caller_sees_initialized(self, tmp_path):
        """同一进程内两次 open_or_create: 第二次看到已 init 的 bitmap, 不重置."""
        arena_dir = str(tmp_path / "arena")
        store1 = LruArenaStore.open_or_create(
            arena_dir, num_slots=100, block_size=16)
        arena1 = _FakeArena()
        store1.bind_data_writer(arena1.writer)
        try:
            tokens = _make_tokens(48)
            ok = store1.write_inc("jobA", 0, 3, tokens,
                                  _block_data("jobA", 0, 3))
            assert ok
            assert store1.free_count() == 97  # 3 slot 已用
            # 第二个 store 打开同一 dir, 不应重置 bitmap
            store2 = LruArenaStore.open_or_create(
                arena_dir, num_slots=100, block_size=16)
            try:
                assert store2.free_count() == 97  # 看到第一个 store 的状态
                # 第二个 store 也能 lookup
                result = store2.lookup("jobA", tokens)
                assert result is not None
                _, matched = result
                assert matched == 3
            finally:
                store2.close()
        finally:
            store1.close()

    def test_open_or_create_cross_process(self, tmp_path):
        """跨进程并发 open_or_create: 两端都 work, 共享同一 arena."""
        import ctypes
        import multiprocessing as mp

        arena_dir = str(tmp_path / "arena")
        ctx = mp.get_context("fork")

        # 用 Value 在父子间通信子进程看到的 free_count
        child_fc = ctx.Value(ctypes.c_int, -1)

        def child(arena_dir, child_fc):
            from vllm.v1.core.sched.licht_v3.arena_lru_store import (
                LruArenaStore)
            child_store = LruArenaStore.open_or_create(
                arena_dir, num_slots=50, block_size=16)
            child_fc.value = int(child_store.free_count())
            child_store.close()

        # 父进程先创建
        parent = LruArenaStore.open_or_create(
            arena_dir, num_slots=50, block_size=16)
        try:
            # 父进程写一段
            parent.bind_data_writer(_FakeArena().writer)
            tokens = _make_tokens(48)
            parent.write_inc("jobA", 0, 3, tokens,
                              _block_data("jobA", 0, 3))
            assert parent.free_count() == 47

            # 子进程 open
            proc = ctx.Process(target=child, args=(arena_dir, child_fc))
            proc.start()
            proc.join(timeout=30)
            assert proc.exitcode == 0, f"child exited with {proc.exitcode}"
            # 子进程应看到 47 (父写了 3 个)
            assert child_fc.value == 47, (
                f"child saw free_count={child_fc.value}, expected 47")
        finally:
            parent.close()


# ============================================================
# Group H - 并发 write_inc (复现 hotfix #3 的死锁场景)
# ============================================================

class TestGroupHConcurrentWrite:
    def test_concurrent_writers_no_deadlock(self, tmp_path):
        """多个线程同时 write_inc 不同 job, 必须不死锁 + 数据正确.

        复现 hotfix #3 死锁: 缩小临界区前, 多个 store 线程在 mutex 临界区内
        做 data_writer + 文件 IO, 4 线程争一把锁串行甚至死锁. 缩小后临界区只
        做 alloc/publish, 应该并发无阻塞.

        data_writer 故意 sleep 模拟 GPU memcpy 慢操作 — 缩小临界区前这会让
        其他线程在锁上等很久; 缩小后并发执行.
        """
        import threading

        arena = _FakeArena()
        slot_lock = threading.Lock()

        def slow_writer(slot_id, block_idx, src):
            # 模拟慢 GPU memcpy
            time.sleep(0.002)
            with slot_lock:  # _FakeArena dict 非线程安全, 测试里加锁
                arena.slots[slot_id] = src[block_idx]

        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=2000, block_size=16)
        store.bind_data_writer(slow_writer)
        try:
            n_threads = 8
            n_jobs_per_thread = 10
            errors = []

            def worker(tid):
                try:
                    for j in range(n_jobs_per_thread):
                        job = f"job_{tid}_{j}"
                        tokens = _make_tokens(48, seed=tid * 100 + j)
                        ok = store.write_inc(job, 0, 3, tokens,
                                             _block_data(job, 0, 3))
                        if not ok:
                            errors.append(f"{job} write failed")
                except Exception as e:
                    errors.append(f"thread {tid}: {e}")

            threads = [threading.Thread(target=worker, args=(i,))
                       for i in range(n_threads)]
            t0 = time.time()
            for t in threads:
                t.start()
            # 关键: join 带 timeout, 死锁则超时失败而非永久挂起
            for t in threads:
                t.join(timeout=30)
            elapsed = time.time() - t0

            # 没有线程还活着 (死锁检测)
            alive = [t for t in threads if t.is_alive()]
            assert not alive, f"DEADLOCK: {len(alive)} threads still alive after 30s"
            assert not errors, f"errors: {errors[:5]}"

            # 所有 job 都能 lookup 命中
            total = n_threads * n_jobs_per_thread
            hit = 0
            for tid in range(n_threads):
                for j in range(n_jobs_per_thread):
                    job = f"job_{tid}_{j}"
                    tokens = _make_tokens(48, seed=tid * 100 + j)
                    if store.lookup(job, tokens) is not None:
                        hit += 1
            assert hit == total, f"only {hit}/{total} jobs hit after concurrent write"
            # 并发应该比串行快很多 (8 线程 * 10 job * 3 block * 2ms ≈ 0.48s 串行,
            # 并发应 < 0.2s); 不强求但 elapsed 不应接近串行上界
            print(f"\n  concurrent write elapsed={elapsed:.2f}s "
                  f"(serial upper bound ~{n_threads * n_jobs_per_thread * 3 * 0.002:.2f}s)")
        finally:
            store.close()

    def test_concurrent_writers_same_arena_fill_and_evict(self, tmp_path):
        """并发 writer 把 arena 写满触发 evict, 验证不死锁 + 无 slot 泄漏."""
        import threading

        arena = _FakeArena()
        slot_lock = threading.Lock()

        def w(slot_id, block_idx, src):
            with slot_lock:
                arena.slots[slot_id] = src[block_idx]

        # 小 arena 强制 evict
        store = LruArenaStore.create(str(tmp_path / "arena"),
                                     num_slots=300, block_size=16)
        store.bind_data_writer(w)
        try:
            errors = []

            def worker(tid):
                try:
                    for j in range(20):
                        job = f"j_{tid}_{j}"
                        tokens = _make_tokens(48, seed=tid * 1000 + j)
                        store.write_inc(job, 0, 3, tokens,
                                        _block_data(job, 0, 3))
                        time.sleep(0.001)
                except Exception as e:
                    errors.append(f"thread {tid}: {e}")

            threads = [threading.Thread(target=worker, args=(i,))
                       for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            alive = [t for t in threads if t.is_alive()]
            assert not alive, f"DEADLOCK: {len(alive)} threads alive"
            assert not errors, f"errors: {errors[:5]}"
            # free_count 应该 >= 0 且 <= num_slots (无泄漏/无负数)
            fc = store.free_count()
            assert 0 <= fc <= 300, f"free_count corrupted: {fc}"
        finally:
            store.close()
