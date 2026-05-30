# SPDX-License-Identifier: Apache-2.0
"""LICHT Round-KV Arena 原子原语单元测试.

测试矩阵:

Group A - 单线程基本:
  - test_constants
  - test_mutex_lock_unlock
  - test_try_pin_basic
  - test_try_pin_gen_mismatch
  - test_unpin
  - test_pin_saturation
  - test_evict_slot
  - test_publish_slot
  - test_atomic_load_store
  - test_atomic_fetch_add

Group B - 多线程并发:
  - test_concurrent_pin_unpin_no_corrupt
  - test_concurrent_atomic_inc_no_lost_update

Group C - 跨进程 mutex:
  - test_cross_process_mutex_basic
  - test_cross_process_pin_visible

Group D - 异常恢复:
  - test_robust_mutex_recover_after_crash
"""
import ctypes
import mmap
import multiprocessing
import os
import signal
import time
import threading

import numpy as np
import pytest

try:
    import licht_arena_atomic as A
except ImportError:
    pytest.skip("licht_arena_atomic not built", allow_module_level=True)


# ============================================================
# Test helpers
# ============================================================

def _aligned_buffer(size_bytes: int, align: int = 8):
    """分配 align 对齐的 buffer, 返回 (np_array, address)."""
    raw = np.zeros(size_bytes + align, dtype=np.uint8)
    addr = raw.ctypes.data
    pad = (align - (addr % align)) % align
    aligned = raw[pad : pad + size_bytes]
    return aligned, aligned.ctypes.data


def _alloc_slot_state(initial_gen: int = 1):
    """分配一个 8 字节 slot_state, 用 initial_gen 初始化 (pin=0)."""
    buf, addr = _aligned_buffer(8, align=8)
    A.atomic_store_u64(addr, initial_gen)
    return buf, addr


def _alloc_mutex():
    """分配 pthread_mutex_t 并初始化."""
    size = A.pthread_mutex_size()
    align = A.pthread_mutex_alignment()
    buf, addr = _aligned_buffer(size, align=align)
    rc = A.mutex_init(addr)
    assert rc == 0, f"mutex_init failed rc={rc}"
    return buf, addr


def _shared_mmap(size_bytes: int) -> mmap.mmap:
    """创建匿名共享 mmap (跨 fork 子进程可见)."""
    return mmap.mmap(-1, size_bytes,
                     flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                     prot=mmap.PROT_READ | mmap.PROT_WRITE)


# ============================================================
# Group A - 单线程基本
# ============================================================

class TestGroupASingleThread:
    def test_constants(self):
        assert A.arena_gen_mask() == (1 << 48) - 1
        assert A.arena_pin_shift() == 48
        assert A.arena_pin_max() == (1 << 16) - 1
        assert A.pthread_mutex_size() > 0
        assert A.errno_eownerdead() == 130  # Linux EOWNERDEAD

    def test_mutex_lock_unlock(self):
        buf, addr = _alloc_mutex()
        try:
            assert A.mutex_lock(addr) == 0
            assert A.mutex_unlock(addr) == 0
            # 再 lock 一次确保可重入用
            assert A.mutex_lock(addr) == 0
            assert A.mutex_unlock(addr) == 0
        finally:
            A.mutex_destroy(addr)

    def test_try_pin_basic(self):
        buf, addr = _alloc_slot_state(initial_gen=42)
        # 第一次 pin 成功
        assert A.try_pin(addr, 42) == 1
        assert A.get_pin(addr) == 1
        assert A.get_gen(addr) == 42
        # 第二次 pin 同 gen 也成功 (累计 +1)
        assert A.try_pin(addr, 42) == 1
        assert A.get_pin(addr) == 2

    def test_try_pin_gen_mismatch(self):
        buf, addr = _alloc_slot_state(initial_gen=42)
        # 期望 gen=99, 实际 42 -> 拒绝
        assert A.try_pin(addr, 99) == 0
        assert A.get_pin(addr) == 0  # pin 没动

    def test_unpin(self):
        buf, addr = _alloc_slot_state(initial_gen=42)
        A.try_pin(addr, 42)
        A.try_pin(addr, 42)
        assert A.get_pin(addr) == 2
        A.unpin(addr)
        assert A.get_pin(addr) == 1
        A.unpin(addr)
        assert A.get_pin(addr) == 0
        # gen 必须保持不变
        assert A.get_gen(addr) == 42

    def test_pin_saturation(self):
        """pin 计数到 MAX 时 try_pin 应拒绝."""
        buf, addr = _alloc_slot_state(initial_gen=1)
        # 人为构造 pin == MAX 的 slot_state
        max_pin = A.arena_pin_max()
        state = (max_pin << A.arena_pin_shift()) | 1
        A.atomic_store_u64(addr, state)
        assert A.get_pin(addr) == max_pin
        assert A.try_pin(addr, 1) == 0  # 拒绝 (饱和)

    def test_evict_slot(self):
        """evict 必须递增 gen, 保持 pin = 0."""
        buf, addr = _alloc_slot_state(initial_gen=42)
        assert A.can_evict(addr) == 1
        new_gen = A.evict_slot(addr)
        assert new_gen == 43
        assert A.get_gen(addr) == 43
        assert A.get_pin(addr) == 0
        # 再 evict 一次
        new_gen = A.evict_slot(addr)
        assert new_gen == 44

    def test_can_evict_with_pin(self):
        """pin > 0 时 can_evict 必须返回 0."""
        buf, addr = _alloc_slot_state(initial_gen=42)
        A.try_pin(addr, 42)
        assert A.can_evict(addr) == 0
        A.unpin(addr)
        assert A.can_evict(addr) == 1

    def test_publish_slot(self):
        """publish 写新 gen, pin 必须保持 0 (调用者保证)."""
        buf, addr = _alloc_slot_state(initial_gen=0)
        A.publish_slot(addr, 100)
        assert A.get_gen(addr) == 100
        assert A.get_pin(addr) == 0

    def test_atomic_load_store(self):
        buf, addr = _aligned_buffer(8, align=8)
        A.atomic_store_u64(addr, 0xDEADBEEF12345678)
        assert A.atomic_load_u64(addr) == 0xDEADBEEF12345678

    def test_atomic_fetch_add(self):
        buf, addr = _aligned_buffer(8, align=8)
        A.atomic_store_u64(addr, 100)
        prev = A.atomic_fetch_add_u64(addr, 50)
        assert prev == 100
        assert A.atomic_load_u64(addr) == 150


# ============================================================
# Group B - 多线程并发
# ============================================================

class TestGroupBConcurrent:
    def test_concurrent_pin_unpin_no_corrupt(self):
        """100 线程同时对一个 slot pin/unpin 各 1000 次, 最终 pin 必须为 0,
        gen 必须不变."""
        buf, addr = _alloc_slot_state(initial_gen=42)
        N_THREADS = 100
        N_ITERS = 1000

        def worker():
            for _ in range(N_ITERS):
                while not A.try_pin(addr, 42):
                    pass  # pin 满了会拒绝, 极少发生; 重试
                A.unpin(addr)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert A.get_pin(addr) == 0
        assert A.get_gen(addr) == 42

    def test_concurrent_atomic_inc_no_lost_update(self):
        """100 线程同时 fetch_add 1, 各 1000 次, 最终值必须精确."""
        buf, addr = _aligned_buffer(8, align=8)
        A.atomic_store_u64(addr, 0)
        N_THREADS = 100
        N_ITERS = 1000

        def worker():
            for _ in range(N_ITERS):
                A.atomic_fetch_add_u64(addr, 1)

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert A.atomic_load_u64(addr) == N_THREADS * N_ITERS

    def test_concurrent_pin_evict_cas_correctness(self):
        """混合 reader (try_pin/unpin) 和 writer (evict).

        核心验证: writer 在 pin==0 时 evict, reader 用对应 gen pin 时不丢失更新.
        测试 CAS 在并发下不产生 corrupt 状态:
          - pin counter 不会因为 evict 漏减
          - gen 单调递增
          - reader pin 上的 gen 在 unpin 前不会被改 (因为 evict 仅在 pin==0 跑)
        """
        buf, addr = _alloc_slot_state(initial_gen=1)
        N_READERS = 8           # 减少 reader, 给 writer 留 pin==0 窗口
        N_WRITER_ITERS = 200
        evict_count = [0]
        stop = [False]
        max_gen_seen = [1]
        reader_iter_count = [0]

        def reader(rid: int):
            local_count = 0
            while not stop[0]:
                cur_gen = A.get_gen(addr)
                if A.try_pin(addr, cur_gen):
                    # pin 成功后 gen 必须依然 >= 我们 pin 的 gen
                    # (writer 只在 pin==0 时 evict, 不会撞)
                    assert A.get_gen(addr) >= cur_gen
                    local_count += 1
                    A.unpin(addr)
                # reader 主动 yield 一下, 给 writer 留 pin==0 窗口
                time.sleep(0.0005)
            reader_iter_count[0] += local_count

        def writer():
            for _ in range(N_WRITER_ITERS):
                # 模拟 alloc_mutex 内 evict: 必须先 can_evict (pin==0)
                if A.can_evict(addr):
                    new_gen = A.evict_slot(addr)
                    evict_count[0] += 1
                    if new_gen > max_gen_seen[0]:
                        max_gen_seen[0] = new_gen
                time.sleep(0.001)

        readers = [threading.Thread(target=reader, args=(i,))
                   for i in range(N_READERS)]
        w = threading.Thread(target=writer)
        for t in readers: t.start()
        w.start()
        w.join()
        stop[0] = True
        for t in readers: t.join()

        # 终态一致性
        assert A.get_pin(addr) == 0, "pin leak detected"
        # gen 必须严格 >= 初始
        assert A.get_gen(addr) >= 1
        # writer 至少成功 evict 一些次数 (1 次都没有说明 reader 完全把窗口挡死了)
        assert evict_count[0] > 0, "writer never got pin==0 window"
        # gen 单调递增到接近 evict_count
        assert A.get_gen(addr) == 1 + evict_count[0], (
            f"gen mismatch: gen={A.get_gen(addr)}, evicts={evict_count[0]}")
        # reader 也得有 progress (不应全部死锁)
        assert reader_iter_count[0] > 0, "no reader progress"


# ============================================================
# Group C - 跨进程 mutex
# ============================================================

def _child_lock_unlock(addr, n_iters, counter_addr):
    """子进程: 加锁 -> 读 counter -> +1 -> 写回 -> 解锁."""
    for _ in range(n_iters):
        rc = A.mutex_lock(addr)
        assert rc == 0, f"child mutex_lock rc={rc}"
        cur = A.atomic_load_u64(counter_addr)
        # 模拟临界区工作
        time.sleep(0.0001)
        A.atomic_store_u64(counter_addr, cur + 1)
        A.mutex_unlock(addr)


class TestGroupCCrossProcess:
    def test_cross_process_mutex_basic(self):
        """父子进程通过共享 mmap 上的 mutex 协调 counter inc.
        两端各 inc N 次, 最终 counter 必须精确."""
        # 共享 mmap: mutex + counter (8 字节)
        mutex_size = A.pthread_mutex_size()
        total_size = mutex_size + 64  # padding + counter
        shm = _shared_mmap(total_size)
        shm_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm))
        mutex_addr = shm_addr
        counter_addr = shm_addr + 64  # 64 对齐 padding

        # 父进程初始化 mutex 和 counter
        rc = A.mutex_init(mutex_addr)
        assert rc == 0
        A.atomic_store_u64(counter_addr, 0)

        N_ITERS = 100
        # 启动子进程
        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_lock_unlock,
                           args=(mutex_addr, N_ITERS, counter_addr))
        proc.start()

        # 父进程同时跑
        _child_lock_unlock(mutex_addr, N_ITERS, counter_addr)
        proc.join(timeout=30)
        assert proc.exitcode == 0, f"child exited with {proc.exitcode}"

        final = A.atomic_load_u64(counter_addr)
        assert final == 2 * N_ITERS, f"expected {2*N_ITERS} got {final}"

        A.mutex_destroy(mutex_addr)
        shm.close()


# ============================================================
# Group D - Robust 异常恢复
# ============================================================

def _child_crash_with_lock(mutex_addr, ready_evt_addr):
    """子进程: 加锁后置 ready=1, 然后死循环等被 kill (不解锁)."""
    A.mutex_lock(mutex_addr)
    A.atomic_store_u64(ready_evt_addr, 1)
    while True:
        time.sleep(1.0)


class TestGroupDRobust:
    def test_robust_mutex_recover_after_crash(self):
        """子进程持锁被 SIGKILL, 父进程必须能 recover + 继续用."""
        mutex_size = A.pthread_mutex_size()
        shm = _shared_mmap(mutex_size + 64)
        shm_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm))
        mutex_addr = shm_addr
        ready_addr = shm_addr + 64

        rc = A.mutex_init(mutex_addr)
        assert rc == 0
        A.atomic_store_u64(ready_addr, 0)

        ctx = multiprocessing.get_context("fork")
        proc = ctx.Process(target=_child_crash_with_lock,
                           args=(mutex_addr, ready_addr))
        proc.start()

        # 等子进程拿到锁
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if A.atomic_load_u64(ready_addr) == 1:
                break
            time.sleep(0.01)
        assert A.atomic_load_u64(ready_addr) == 1, "child never acquired lock"

        # SIGKILL 子进程 (不让 atexit 跑)
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=5)
        assert proc.exitcode == -signal.SIGKILL

        # 父进程尝试加锁: 应得 EOWNERDEAD
        rc = A.mutex_lock(mutex_addr)
        assert rc == A.errno_eownerdead(), (
            f"expected EOWNERDEAD ({A.errno_eownerdead()}), got {rc}")

        # recover 后可继续用
        rc = A.mutex_recover(mutex_addr)
        assert rc == 0, f"mutex_recover rc={rc}"

        # 解锁
        rc = A.mutex_unlock(mutex_addr)
        assert rc == 0

        # 再 lock/unlock 一轮验证 mutex 已可用
        assert A.mutex_lock(mutex_addr) == 0
        assert A.mutex_unlock(mutex_addr) == 0

        A.mutex_destroy(mutex_addr)
        shm.close()
