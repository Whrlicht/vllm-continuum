// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// LICHT Round-KV Arena 跨进程原子原语实现 (Stage 0 骨架)
//
// 注意:
//   - 所有 atomic 用 __atomic_* GCC builtin (C++11 std::atomic 不能跨进程)
//   - mutex 用 PTHREAD_PROCESS_SHARED + ROBUST
//   - 文件后缀 .cu 仅为统一构建流程, 本文件不含 CUDA kernel
//
// Stage 1 落地: 把 TODO 替换为真实实现, 加 PYBIND11_MODULE

#include "arena_atomic.h"
#include <errno.h>

// ============================================================
// Mutex
// ============================================================
int arena_mutex_init(pthread_mutex_t* m) {
    pthread_mutexattr_t attr;
    int rc;
    if ((rc = pthread_mutexattr_init(&attr)) != 0) return rc;
    if ((rc = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED)) != 0) {
        pthread_mutexattr_destroy(&attr);
        return rc;
    }
    if ((rc = pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST)) != 0) {
        pthread_mutexattr_destroy(&attr);
        return rc;
    }
    rc = pthread_mutex_init(m, &attr);
    pthread_mutexattr_destroy(&attr);
    return rc;
}

int arena_mutex_destroy(pthread_mutex_t* m) {
    return pthread_mutex_destroy(m);
}

int arena_mutex_lock(pthread_mutex_t* m) {
    return pthread_mutex_lock(m);
}

int arena_mutex_unlock(pthread_mutex_t* m) {
    return pthread_mutex_unlock(m);
}

int arena_mutex_recover(pthread_mutex_t* m) {
    return pthread_mutex_consistent(m);
}

// ============================================================
// Slot state CAS (TODO Stage 1 落地)
// ============================================================
int arena_try_pin(uint64_t* slot_state, uint64_t expected_gen) {
    // Stage 1 实现:
    //   uint64_t old = __atomic_load_n(slot_state, __ATOMIC_ACQUIRE);
    //   for (;;) {
    //     uint64_t cur_gen = old & ARENA_GEN_MASK;
    //     if (cur_gen != expected_gen) return 0;
    //     uint64_t cur_pin = old >> ARENA_PIN_SHIFT;
    //     if (cur_pin == ARENA_PIN_MAX) return 0;
    //     uint64_t neu = ((cur_pin + 1) << ARENA_PIN_SHIFT) | cur_gen;
    //     if (__atomic_compare_exchange_n(slot_state, &old, neu, 0,
    //                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    //         return 1;
    //   }
    return 0;
}

void arena_unpin(uint64_t* slot_state) {
    // Stage 1: __atomic_sub_fetch(slot_state, 1ULL << ARENA_PIN_SHIFT, __ATOMIC_RELEASE);
}

uint64_t arena_get_gen(uint64_t* slot_state) {
    // Stage 1: return __atomic_load_n(slot_state, __ATOMIC_ACQUIRE) & ARENA_GEN_MASK;
    return 0;
}

uint64_t arena_get_pin(uint64_t* slot_state) {
    // Stage 1: return __atomic_load_n(slot_state, __ATOMIC_ACQUIRE) >> ARENA_PIN_SHIFT;
    return 0;
}

int arena_can_evict(uint64_t* slot_state) {
    // Stage 1: return (arena_get_pin(slot_state) == 0) ? 1 : 0;
    return 0;
}

void arena_evict_slot(uint64_t* slot_state, uint64_t* new_gen_out) {
    // Stage 1:
    //   uint64_t old = __atomic_load_n(slot_state, __ATOMIC_RELAXED);
    //   uint64_t new_gen = (old & ARENA_GEN_MASK) + 1;
    //   __atomic_store_n(slot_state, new_gen, __ATOMIC_RELEASE);  // pin 仍为 0
    //   if (new_gen_out) *new_gen_out = new_gen;
}

void arena_publish_slot(uint64_t* slot_state, uint64_t new_gen) {
    // Stage 1:
    //   __atomic_store_n(slot_state, new_gen, __ATOMIC_RELEASE);
}

// ============================================================
// 通用原子原语
// ============================================================
uint64_t arena_atomic_load_u64(uint64_t* addr) {
    return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
}

void arena_atomic_store_u64(uint64_t* addr, uint64_t val) {
    __atomic_store_n(addr, val, __ATOMIC_RELEASE);
}

uint64_t arena_atomic_fetch_add_u64(uint64_t* addr, uint64_t delta) {
    return __atomic_fetch_add(addr, delta, __ATOMIC_ACQ_REL);
}

// ============================================================
// Python binding (TODO Stage 1)
// ============================================================
// Stage 1 落地时加 PYBIND11_MODULE, 把上述 API 暴露给 Python
//
// 关键 binding 要点:
//   - mutex/slot_state 地址用 uint64 (int) 在 Python 端传入
//   - Python 端通过 ctypes 拿 mmap 的 buffer.ctypes.data 得到地址
//   - 不要在 binding 里做 mmap 操作, Python 端做更灵活
