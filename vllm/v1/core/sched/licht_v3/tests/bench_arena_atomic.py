# SPDX-License-Identifier: Apache-2.0
"""LICHT Arena 原子原语性能基准.

不属于 pytest 测试 (跑得慢, 噪声大), 单独 invoke:
    python vllm/v1/core/sched/licht_v3/tests/bench_arena_atomic.py

测试场景:
  1. 单线程 try_pin/unpin 吞吐
  2. 单线程 atomic_load_u64/store_u64 吞吐
  3. 单线程 atomic_fetch_add 吞吐
  4. 单线程 mutex_lock/unlock 吞吐
  5. 单线程 evict_slot 吞吐
  6. 多线程 try_pin/unpin 在不同线程数下的吞吐扩展性
  7. 多线程 atomic_fetch_add 争用扩展性
  8. 跨进程 mutex 吞吐 (单对父子)
  9. bitmap 操作 (set_used/set_free 通过 hdr) 吞吐

输出对比: 操作/秒 + 单操作 ns
"""
import ctypes
import multiprocessing
import os
import statistics
import sys
import time
import threading

import numpy as np

# 直接 import arena_hdr 模块, 绕过 vllm/__init__.py
import importlib.util
_arena_hdr_path = os.path.join(os.path.dirname(__file__), "..", "arena_hdr.py")
_spec = importlib.util.spec_from_file_location("arena_hdr", _arena_hdr_path)
_arena_hdr = importlib.util.module_from_spec(_spec)
sys.modules["arena_hdr"] = _arena_hdr  # dataclass 需要 sys.modules 里有 module
_spec.loader.exec_module(_arena_hdr)
ArenaHdr = _arena_hdr.ArenaHdr

import licht_arena_atomic as A


# ============================================================
# Helpers
# ============================================================
def _bench(name, n_iters, fn):
    """跑 fn() n_iters 次, 报 ops/sec + ns/op."""
    # 预热
    for _ in range(min(1000, n_iters // 10 or 1)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    elapsed = time.perf_counter() - t0
    ops_per_sec = n_iters / elapsed
    ns_per_op = elapsed * 1e9 / n_iters
    print(f"  {name:<40s}  {ops_per_sec:>12,.0f} ops/sec  ({ns_per_op:>8.1f} ns/op)")
    return ops_per_sec, ns_per_op


def _aligned_buffer(size, align=8):
    raw = np.zeros(size + align, dtype=np.uint8)
    addr = raw.ctypes.data
    pad = (align - (addr % align)) % align
    return raw, raw.ctypes.data + pad


def _alloc_slot():
    buf, addr = _aligned_buffer(8)
    A.atomic_store_u64(addr, 1)  # gen=1
    return buf, addr


def _alloc_mutex():
    size = A.pthread_mutex_size()
    align = A.pthread_mutex_alignment()
    buf, addr = _aligned_buffer(size, align)
    rc = A.mutex_init(addr)
    assert rc == 0
    return buf, addr


# ============================================================
# Single-thread benchmarks
# ============================================================
def bench_single_thread():
    print("=" * 70)
    print("Single-thread throughput")
    print("=" * 70)

    N = 1_000_000

    # try_pin + unpin pair (CAS + sub)
    _, slot = _alloc_slot()
    def do():
        A.try_pin(slot, 1)
        A.unpin(slot)
    _bench("try_pin + unpin (1 pair)", N, do)

    # raw atomic load
    _, addr = _aligned_buffer(8)
    A.atomic_store_u64(addr, 42)
    _bench("atomic_load_u64", N, lambda: A.atomic_load_u64(addr))

    # raw atomic store
    _bench("atomic_store_u64", N, lambda: A.atomic_store_u64(addr, 42))

    # fetch_add
    A.atomic_store_u64(addr, 0)
    _bench("atomic_fetch_add_u64", N, lambda: A.atomic_fetch_add_u64(addr, 1))

    # fetch_or
    A.atomic_store_u64(addr, 0)
    _bench("atomic_fetch_or_u64", N, lambda: A.atomic_fetch_or_u64(addr, 1))

    # fetch_and
    A.atomic_store_u64(addr, 0xFFFFFFFFFFFFFFFF)
    _bench("atomic_fetch_and_u64", N, lambda: A.atomic_fetch_and_u64(addr, ~1 & 0xFFFFFFFFFFFFFFFF))

    # mutex lock + unlock
    _, mutex_addr = _alloc_mutex()
    def do_mutex():
        A.mutex_lock(mutex_addr)
        A.mutex_unlock(mutex_addr)
    _bench("mutex_lock + unlock (uncontended)", N // 5, do_mutex)
    A.mutex_destroy(mutex_addr)

    # evict_slot
    _, slot = _alloc_slot()
    _bench("evict_slot (gen++)", N, lambda: A.evict_slot(slot))

    # get_gen (atomic_load + mask)
    _, slot = _alloc_slot()
    _bench("get_gen", N, lambda: A.get_gen(slot))


# ============================================================
# Multi-thread benchmarks
# ============================================================
def bench_multi_thread_pin():
    print("=" * 70)
    print("Multi-thread try_pin/unpin (1 slot, contended)")
    print("=" * 70)

    for n_threads in [1, 2, 4, 8, 16, 32]:
        _, slot = _alloc_slot()
        n_iters = 200_000

        def worker():
            for _ in range(n_iters):
                while not A.try_pin(slot, 1):
                    pass
                A.unpin(slot)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = time.perf_counter() - t0
        total_ops = n_threads * n_iters
        ops_per_sec = total_ops / elapsed
        ns_per_op = elapsed * 1e9 / total_ops
        print(f"  threads={n_threads:>2}  total ops={total_ops:>9,}  "
              f"{ops_per_sec:>11,.0f} ops/sec  ({ns_per_op:>7.1f} ns/op)")


def bench_multi_thread_atomic_inc():
    print("=" * 70)
    print("Multi-thread atomic_fetch_add (1 counter, max contention)")
    print("=" * 70)

    for n_threads in [1, 2, 4, 8, 16]:
        _, addr = _aligned_buffer(8)
        A.atomic_store_u64(addr, 0)
        n_iters = 500_000

        def worker():
            for _ in range(n_iters):
                A.atomic_fetch_add_u64(addr, 1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        t0 = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = time.perf_counter() - t0
        assert A.atomic_load_u64(addr) == n_threads * n_iters
        total_ops = n_threads * n_iters
        ops_per_sec = total_ops / elapsed
        ns_per_op = elapsed * 1e9 / total_ops
        print(f"  threads={n_threads:>2}  total ops={total_ops:>9,}  "
              f"{ops_per_sec:>11,.0f} ops/sec  ({ns_per_op:>7.1f} ns/op)")


# ============================================================
# Cross-process mutex
# ============================================================
def _child_mutex_pingpong(mutex_addr, n_iters, counter_addr):
    for _ in range(n_iters):
        A.mutex_lock(mutex_addr)
        A.atomic_fetch_add_u64(counter_addr, 1)
        A.mutex_unlock(mutex_addr)


def bench_cross_process_mutex():
    print("=" * 70)
    print("Cross-process mutex lock/unlock (父+1子, 共享 counter)")
    print("=" * 70)
    import mmap as mm
    mutex_size = A.pthread_mutex_size()
    total_size = 4096
    shm = mm.mmap(-1, total_size,
                  flags=mm.MAP_SHARED | mm.MAP_ANONYMOUS,
                  prot=mm.PROT_READ | mm.PROT_WRITE)
    shm_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm))
    mutex_addr = shm_addr
    counter_addr = shm_addr + 64
    A.mutex_init(mutex_addr)
    A.atomic_store_u64(counter_addr, 0)

    n_iters = 500_000
    ctx = multiprocessing.get_context("fork")
    t0 = time.perf_counter()
    proc = ctx.Process(target=_child_mutex_pingpong,
                       args=(mutex_addr, n_iters, counter_addr))
    proc.start()
    _child_mutex_pingpong(mutex_addr, n_iters, counter_addr)
    proc.join()
    elapsed = time.perf_counter() - t0

    final = A.atomic_load_u64(counter_addr)
    assert final == 2 * n_iters, f"counter {final} != {2*n_iters}"
    total_ops = 2 * n_iters
    ops_per_sec = total_ops / elapsed
    ns_per_op = elapsed * 1e9 / total_ops
    print(f"  total ops (2 procs)={total_ops:>9,}  "
          f"{ops_per_sec:>11,.0f} ops/sec  ({ns_per_op:>7.1f} ns/op)")
    A.mutex_destroy(mutex_addr)
    shm.close()


# ============================================================
# Hdr bitmap
# ============================================================
def bench_bitmap():
    print("=" * 70)
    print("Bitmap operations via ArenaHdr")
    print("=" * 70)
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    try:
        N = 12000   # 真实 num_slots
        hdr = ArenaHdr.create(path, num_slots=N)
        try:
            hdr.bitmap_init_all_free()

            # set_used 循环
            n_iters = 100_000
            t0 = time.perf_counter()
            for i in range(n_iters):
                hdr.bitmap_set_used(i % N)
                hdr.bitmap_set_free(i % N)
            elapsed = time.perf_counter() - t0
            total_ops = 2 * n_iters  # set_used + set_free
            ops_per_sec = total_ops / elapsed
            print(f"  bitmap_set_used + set_free pair       "
                  f"{ops_per_sec:>11,.0f} ops/sec  ({elapsed*1e9/total_ops:>7.1f} ns/op)")

            # is_free 循环
            n_iters = 200_000
            t0 = time.perf_counter()
            for i in range(n_iters):
                hdr.bitmap_is_free(i % N)
            elapsed = time.perf_counter() - t0
            ops_per_sec = n_iters / elapsed
            print(f"  bitmap_is_free                        "
                  f"{ops_per_sec:>11,.0f} ops/sec  ({elapsed*1e9/n_iters:>7.1f} ns/op)")

            # count_free (扫全 bitmap)
            n_iters = 1000
            t0 = time.perf_counter()
            for _ in range(n_iters):
                hdr.count_free()
            elapsed = time.perf_counter() - t0
            ops_per_sec = n_iters / elapsed
            print(f"  count_free (扫 12000 slots)            "
                  f"{ops_per_sec:>11,.0f} ops/sec  ({elapsed*1e6/n_iters:>7.1f} us/op)")
        finally:
            hdr.close()
    finally:
        os.unlink(path)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print()
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}, "
          f"licht_arena_atomic loaded")
    print()
    bench_single_thread()
    print()
    bench_multi_thread_pin()
    print()
    bench_multi_thread_atomic_inc()
    print()
    bench_cross_process_mutex()
    print()
    bench_bitmap()
    print()
