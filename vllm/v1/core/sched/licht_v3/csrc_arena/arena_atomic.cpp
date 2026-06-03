// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// LICHT Round-KV Arena 跨进程原子原语实现
//
// 注意:
//   - 所有 atomic 用 __atomic_* GCC builtin (C++11 std::atomic 不能跨进程)
//   - mutex 用 PTHREAD_PROCESS_SHARED + ROBUST
//   - 文件后缀 .cu 仅为统一构建流程, 本文件不含 CUDA kernel
//
// Memory ordering 约定:
//   - load: ACQUIRE   (读者保证后续 access 不被重排到 load 之前)
//   - store: RELEASE  (写者保证之前 access 不被重排到 store 之后)
//   - CAS / RMW: ACQ_REL

#include "arena_atomic.h"
#include <errno.h>
#include <string>
#include <torch/extension.h>

// xxhash single-header: 在本 TU 内联全部实现 (Stage 6 内容寻址 block hash 用)
#define XXH_INLINE_ALL
#include "xxhash.h"

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
// Slot state CAS
// ============================================================
//
// slot_state[i] 编码: (pin << 48) | gen
//   pin: 高 16 位, reader pin 计数
//   gen: 低 48 位, slot 内容版本号
//
// 不变量:
//   - 任何时刻, 只要 pin > 0, 这个 slot 的 gen 不能被 writer 改变
//   - writer 改 gen 必须在 alloc_mutex 临界区内, 且 pin == 0
//   - reader 加 pin 是 lock-free CAS, 必须同时校验 gen 未变

int arena_try_pin(uint64_t* slot_state, uint64_t expected_gen) {
    uint64_t old = __atomic_load_n(slot_state, __ATOMIC_ACQUIRE);
    for (;;) {
        uint64_t cur_gen = old & ARENA_GEN_MASK;
        // gen 不匹配 -> slot 内容已变, 拒绝 pin
        if (cur_gen != expected_gen) return 0;
        uint64_t cur_pin = old >> ARENA_PIN_SHIFT;
        // pin 饱和 (理论上不应发生, 因为 16 位足够大)
        if (cur_pin == ARENA_PIN_MAX) return 0;
        uint64_t neu = ((cur_pin + 1) << ARENA_PIN_SHIFT) | cur_gen;
        // CAS: 期望 slot_state 仍为 old, 改为 neu
        // 失败时 __atomic_compare_exchange_n 自动把新值写回 old, 重试循环
        if (__atomic_compare_exchange_n(slot_state, &old, neu, 0 /* strong */,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            return 1;
        }
        // CAS 失败 -> 期间有别人改了 slot_state, 用更新后的 old 重试
    }
}

void arena_unpin(uint64_t* slot_state) {
    // pin -= 1, gen 不变
    // 用 sub 而不是 CAS: 因为只动高 16 位, 不会破坏低 48 位 gen
    __atomic_sub_fetch(slot_state, 1ULL << ARENA_PIN_SHIFT, __ATOMIC_RELEASE);
}

uint64_t arena_get_gen(uint64_t* slot_state) {
    return __atomic_load_n(slot_state, __ATOMIC_ACQUIRE) & ARENA_GEN_MASK;
}

uint64_t arena_get_pin(uint64_t* slot_state) {
    return __atomic_load_n(slot_state, __ATOMIC_ACQUIRE) >> ARENA_PIN_SHIFT;
}

int arena_can_evict(uint64_t* slot_state) {
    return (arena_get_pin(slot_state) == 0) ? 1 : 0;
}

void arena_evict_slot(uint64_t* slot_state, uint64_t* new_gen_out) {
    // 仅在 alloc_mutex 内 + pin == 0 时调用
    // 因此读 old 可用 RELAXED (没有并发改写)
    uint64_t old = __atomic_load_n(slot_state, __ATOMIC_RELAXED);
    // mask 防御 gen 溢出 (理论 2^48 evict 才会发生, 实际几千年不到):
    // 不 mask -> 写入 bit 48 会把 pin 位污染
    uint64_t new_gen = ((old & ARENA_GEN_MASK) + 1) & ARENA_GEN_MASK;
    // gen 更新, pin 保持 0
    __atomic_store_n(slot_state, new_gen, __ATOMIC_RELEASE);
    if (new_gen_out) *new_gen_out = new_gen;
}

void arena_publish_slot(uint64_t* slot_state, uint64_t new_gen) {
    // 仅在 alloc_mutex 内、memcpy 完成后调用
    // pin 必须为 0 (alloc 完后 reader 还看不到)
    // RELEASE 保证 memcpy 的所有 store 在 publish 之前对 reader 可见
    // mask 防御调用方传入的 new_gen 超过 48 位
    __atomic_store_n(slot_state, new_gen & ARENA_GEN_MASK, __ATOMIC_RELEASE);
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

uint64_t arena_atomic_fetch_or_u64(uint64_t* addr, uint64_t mask) {
    return __atomic_fetch_or(addr, mask, __ATOMIC_ACQ_REL);
}

uint64_t arena_atomic_fetch_and_u64(uint64_t* addr, uint64_t mask) {
    return __atomic_fetch_and(addr, mask, __ATOMIC_ACQ_REL);
}

// ============================================================
// Stage 6: slot refcnt (atomic uint16)
// ============================================================
uint32_t arena_refcnt_inc(uint16_t* p) {
    return (uint32_t)__atomic_add_fetch(p, (uint16_t)1, __ATOMIC_ACQ_REL);
}
uint32_t arena_refcnt_dec(uint16_t* p) {
    return (uint32_t)__atomic_sub_fetch(p, (uint16_t)1, __ATOMIC_ACQ_REL);
}
uint32_t arena_refcnt_get(uint16_t* p) {
    return (uint32_t)__atomic_load_n(p, __ATOMIC_ACQUIRE);
}
void arena_refcnt_set(uint16_t* p, uint16_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}

// ============================================================
// Stage 6: 内容寻址 hash 表 (open-addressing, 24B entry, 无 seqlock)
// ============================================================
// entry 24 字节: [hash:u64 @0][slot_id:i64 @8][gen:u64 @16]
// 无 seqlock: 每字段 8B 原子读写 (写 hash 用 RELEASE 作 commit, 读用 ACQUIRE);
// 撕裂读最坏读到 slot/gen 错配, 下游 try_pin(slot, gen) 必失配 fail-closed.
static inline uint64_t* ht_hash_ptr(char* base, uint64_t idx) {
    return reinterpret_cast<uint64_t*>(base + idx * ARENA_HT_ENTRY_BYTES + 0);
}
static inline int64_t* ht_slot_ptr(char* base, uint64_t idx) {
    return reinterpret_cast<int64_t*>(base + idx * ARENA_HT_ENTRY_BYTES + 8);
}
static inline uint64_t* ht_gen_ptr(char* base, uint64_t idx) {
    return reinterpret_cast<uint64_t*>(base + idx * ARENA_HT_ENTRY_BYTES + 16);
}

// 仅在 alloc_mutex 内调用. 写 gen/slot 后, 最后 RELEASE 写 hash 作 commit:
// reader 读到新 hash (ACQUIRE) 即保证看到已写好的 slot/gen.
static inline void ht_put(char* base, uint64_t idx,
                          uint64_t hash, int64_t slot_id, uint64_t gen) {
    __atomic_store_n(ht_gen_ptr(base, idx), gen, __ATOMIC_RELAXED);
    __atomic_store_n(ht_slot_ptr(base, idx), slot_id, __ATOMIC_RELAXED);
    __atomic_store_n(ht_hash_ptr(base, idx), hash, __ATOMIC_RELEASE);
}

void arena_ht_clear(void* table_base, uint64_t cap) {
    char* base = reinterpret_cast<char*>(table_base);
    for (uint64_t i = 0; i < cap; i++) {
        __atomic_store_n(ht_hash_ptr(base, i), (uint64_t)0, __ATOMIC_RELAXED);
        __atomic_store_n(ht_slot_ptr(base, i),
                         (int64_t)ARENA_HT_EMPTY, __ATOMIC_RELAXED);
        __atomic_store_n(ht_gen_ptr(base, i),
                         (uint64_t)ARENA_HT_GEN_UNPUB, __ATOMIC_RELAXED);
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

int arena_ht_probe(void* table_base, uint64_t cap, uint64_t hash,
                   int64_t* out_slot, uint64_t* out_gen) {
    if (cap == 0) return 0;
    char* base = reinterpret_cast<char*>(table_base);
    uint64_t start = hash % cap;
    for (uint64_t step = 0; step < cap; step++) {
        uint64_t idx = start + step;
        if (idx >= cap) idx -= cap;
        int64_t es = __atomic_load_n(ht_slot_ptr(base, idx), __ATOMIC_ACQUIRE);
        if (es == ARENA_HT_EMPTY) return 0;        // 探测链尽头 -> miss
        if (es == ARENA_HT_TOMBSTONE) continue;    // 墓碑, 继续
        uint64_t eh = __atomic_load_n(ht_hash_ptr(base, idx), __ATOMIC_ACQUIRE);
        if (eh == hash) {                          // 命中
            if (out_slot) *out_slot = es;
            if (out_gen) {
                *out_gen = __atomic_load_n(ht_gen_ptr(base, idx),
                                           __ATOMIC_ACQUIRE);
            }
            return 1;
        }
        // hash 不同 -> 线性继续
    }
    return 0;
}

int arena_ht_insert(void* table_base, uint64_t cap,
                    uint64_t hash, int64_t slot_id) {
    if (cap == 0) return 0;
    char* base = reinterpret_cast<char*>(table_base);
    uint64_t start = hash % cap;
    int64_t first_tomb = -1;
    for (uint64_t step = 0; step < cap; step++) {
        uint64_t idx = start + step;
        if (idx >= cap) idx -= cap;
        int64_t es = __atomic_load_n(ht_slot_ptr(base, idx), __ATOMIC_ACQUIRE);
        if (es == ARENA_HT_EMPTY) {
            uint64_t put = (first_tomb >= 0) ? (uint64_t)first_tomb : idx;
            ht_put(base, put, hash, slot_id, ARENA_HT_GEN_UNPUB);  // 插入未发布
            return 1;
        }
        if (es == ARENA_HT_TOMBSTONE) {
            if (first_tomb < 0) first_tomb = (int64_t)idx;
            continue;
        }
        uint64_t eh = __atomic_load_n(ht_hash_ptr(base, idx), __ATOMIC_RELAXED);
        if (eh == hash) {                          // 已存在 -> 重置 slot+未发布
            ht_put(base, idx, hash, slot_id, ARENA_HT_GEN_UNPUB);
            return 1;
        }
    }
    if (first_tomb >= 0) {
        ht_put(base, (uint64_t)first_tomb, hash, slot_id, ARENA_HT_GEN_UNPUB);
        return 1;
    }
    return 0;  // 表满
}

int arena_ht_set_gen(void* table_base, uint64_t cap,
                     uint64_t hash, uint64_t gen) {
    if (cap == 0) return 0;
    char* base = reinterpret_cast<char*>(table_base);
    uint64_t start = hash % cap;
    for (uint64_t step = 0; step < cap; step++) {
        uint64_t idx = start + step;
        if (idx >= cap) idx -= cap;
        int64_t es = __atomic_load_n(ht_slot_ptr(base, idx), __ATOMIC_ACQUIRE);
        if (es == ARENA_HT_EMPTY) return 0;
        if (es == ARENA_HT_TOMBSTONE) continue;
        uint64_t eh = __atomic_load_n(ht_hash_ptr(base, idx), __ATOMIC_RELAXED);
        if (eh == hash) {
            __atomic_store_n(ht_gen_ptr(base, idx), gen, __ATOMIC_RELEASE);
            return 1;
        }
    }
    return 0;
}

int arena_ht_remove(void* table_base, uint64_t cap, uint64_t hash) {
    if (cap == 0) return 0;
    char* base = reinterpret_cast<char*>(table_base);
    uint64_t start = hash % cap;
    for (uint64_t step = 0; step < cap; step++) {
        uint64_t idx = start + step;
        if (idx >= cap) idx -= cap;
        int64_t es = __atomic_load_n(ht_slot_ptr(base, idx), __ATOMIC_ACQUIRE);
        if (es == ARENA_HT_EMPTY) return 0;        // 探测链尽头, 未找到
        if (es == ARENA_HT_TOMBSTONE) continue;
        uint64_t eh = __atomic_load_n(ht_hash_ptr(base, idx), __ATOMIC_RELAXED);
        if (eh == hash) {
            __atomic_store_n(ht_slot_ptr(base, idx),
                             (int64_t)ARENA_HT_TOMBSTONE, __ATOMIC_RELEASE);
            return 1;
        }
    }
    return 0;
}

// ============================================================
// Stage 6: 链式 block hash (XXH3, seed=prev)
// ============================================================
uint64_t arena_block_hash(uint64_t prev, const void* data, uint64_t len) {
    return (uint64_t)XXH3_64bits_withSeed(data, (size_t)len, prev);
}

// ============================================================
// Python binding
// ============================================================
//
// Binding 约定:
//   - 所有指针参数在 Python 端表达为 uint64 (int) 地址
//   - Python 端通过 numpy.ndarray.ctypes.data 或 mmap+ctypes 拿地址
//   - C++ 端 cast 回对应指针类型
//
// 不在 binding 里做的事:
//   - 不做 mmap (Python 端用 mmap 模块)
//   - 不做 hdr layout 计算 (Python 端做)
//   - 不做错误重试 (Python 端按返回值决定)

namespace py = pybind11;

// ---- Mutex (地址传 int, 内部 cast 成 pthread_mutex_t*) ----
static int py_mutex_init(uint64_t addr) {
    return arena_mutex_init(reinterpret_cast<pthread_mutex_t*>(addr));
}
static int py_mutex_destroy(uint64_t addr) {
    return arena_mutex_destroy(reinterpret_cast<pthread_mutex_t*>(addr));
}
static int py_mutex_lock(uint64_t addr) {
    // 必须 release GIL: pthread_mutex_lock 会阻塞 (等其他线程/进程释放锁).
    // 若持 GIL 阻塞, 持锁线程拿不到 GIL 跑不完临界区, 死锁 (整个解释器冻结).
    // 这是单进程多 store 线程下的核心正确性要求.
    pthread_mutex_t* m = reinterpret_cast<pthread_mutex_t*>(addr);
    int rc;
    {
        py::gil_scoped_release release;
        rc = arena_mutex_lock(m);
    }
    return rc;
}
static int py_mutex_unlock(uint64_t addr) {
    return arena_mutex_unlock(reinterpret_cast<pthread_mutex_t*>(addr));
}
static int py_mutex_recover(uint64_t addr) {
    return arena_mutex_recover(reinterpret_cast<pthread_mutex_t*>(addr));
}

// ---- Slot state CAS ----
static int py_try_pin(uint64_t addr, uint64_t expected_gen) {
    return arena_try_pin(reinterpret_cast<uint64_t*>(addr), expected_gen);
}
static void py_unpin(uint64_t addr) {
    arena_unpin(reinterpret_cast<uint64_t*>(addr));
}
static uint64_t py_get_gen(uint64_t addr) {
    return arena_get_gen(reinterpret_cast<uint64_t*>(addr));
}
static uint64_t py_get_pin(uint64_t addr) {
    return arena_get_pin(reinterpret_cast<uint64_t*>(addr));
}
static int py_can_evict(uint64_t addr) {
    return arena_can_evict(reinterpret_cast<uint64_t*>(addr));
}
// 返回值: new_gen (避免 Python 端处理输出参数)
static uint64_t py_evict_slot(uint64_t addr) {
    uint64_t new_gen = 0;
    arena_evict_slot(reinterpret_cast<uint64_t*>(addr), &new_gen);
    return new_gen;
}
static void py_publish_slot(uint64_t addr, uint64_t new_gen) {
    arena_publish_slot(reinterpret_cast<uint64_t*>(addr), new_gen);
}

// ---- 通用原子 ----
static uint64_t py_atomic_load_u64(uint64_t addr) {
    return arena_atomic_load_u64(reinterpret_cast<uint64_t*>(addr));
}
static void py_atomic_store_u64(uint64_t addr, uint64_t val) {
    arena_atomic_store_u64(reinterpret_cast<uint64_t*>(addr), val);
}
static uint64_t py_atomic_fetch_add_u64(uint64_t addr, uint64_t delta) {
    return arena_atomic_fetch_add_u64(reinterpret_cast<uint64_t*>(addr), delta);
}
static uint64_t py_atomic_fetch_or_u64(uint64_t addr, uint64_t mask) {
    return arena_atomic_fetch_or_u64(reinterpret_cast<uint64_t*>(addr), mask);
}
static uint64_t py_atomic_fetch_and_u64(uint64_t addr, uint64_t mask) {
    return arena_atomic_fetch_and_u64(reinterpret_cast<uint64_t*>(addr), mask);
}

// ---- Stage 6: refcnt ----
static uint64_t py_refcnt_inc(uint64_t addr) {
    return arena_refcnt_inc(reinterpret_cast<uint16_t*>(addr));
}
static uint64_t py_refcnt_dec(uint64_t addr) {
    return arena_refcnt_dec(reinterpret_cast<uint16_t*>(addr));
}
static uint64_t py_refcnt_get(uint64_t addr) {
    return arena_refcnt_get(reinterpret_cast<uint16_t*>(addr));
}
static void py_refcnt_set(uint64_t addr, uint64_t v) {
    arena_refcnt_set(reinterpret_cast<uint16_t*>(addr), (uint16_t)v);
}

// ---- Stage 6: hash 表 ----
// probe 返回 (slot_id, gen); miss 时 (-1, 0). gen=0 表示已占位未发布.
static py::tuple py_ht_probe(uint64_t base, uint64_t cap, uint64_t hash) {
    int64_t slot = -1;
    uint64_t gen = 0;
    int found = arena_ht_probe(reinterpret_cast<void*>(base), cap, hash,
                               &slot, &gen);
    if (!found) { slot = -1; gen = 0; }
    return py::make_tuple(slot, gen);
}
static int py_ht_insert(uint64_t base, uint64_t cap,
                        uint64_t hash, int64_t slot_id) {
    return arena_ht_insert(reinterpret_cast<void*>(base), cap, hash, slot_id);
}
static int py_ht_set_gen(uint64_t base, uint64_t cap,
                         uint64_t hash, uint64_t gen) {
    return arena_ht_set_gen(reinterpret_cast<void*>(base), cap, hash, gen);
}
static int py_ht_remove(uint64_t base, uint64_t cap, uint64_t hash) {
    return arena_ht_remove(reinterpret_cast<void*>(base), cap, hash);
}
static void py_ht_clear(uint64_t base, uint64_t cap) {
    arena_ht_clear(reinterpret_cast<void*>(base), cap);
}

// ---- Stage 6: 链式 block hash ----
// data 是 token block 的字节序列 (Python 端打包成 int32 LE bytes).
static uint64_t py_block_hash(uint64_t prev, py::bytes data) {
    std::string s = data;  // 小 (block_size 个 token), 拷贝可接受
    return arena_block_hash(prev, s.data(), (uint64_t)s.size());
}

// ---- 常量查询 (Python 端避免 hardcode) ----
static uint64_t py_arena_ht_entry_bytes() {
    return (uint64_t)ARENA_HT_ENTRY_BYTES;
}
static uint64_t py_pthread_mutex_size() {
    return sizeof(pthread_mutex_t);
}
static uint64_t py_pthread_mutex_alignment() {
    return alignof(pthread_mutex_t);
}
static uint64_t py_arena_gen_mask() {
    return ARENA_GEN_MASK;
}
static uint64_t py_arena_pin_shift() {
    return ARENA_PIN_SHIFT;
}
static uint64_t py_arena_pin_max() {
    return ARENA_PIN_MAX;
}
// 把 EOWNERDEAD 这种 errno 暴露给 Python 校验
static int py_errno_eownerdead() {
    return EOWNERDEAD;
}

PYBIND11_MODULE(licht_arena_atomic, m) {
    m.doc() = "LICHT Round-KV Arena cross-process atomic primitives";

    // Mutex
    m.def("mutex_init",    &py_mutex_init,    py::arg("addr"));
    m.def("mutex_destroy", &py_mutex_destroy, py::arg("addr"));
    m.def("mutex_lock",    &py_mutex_lock,    py::arg("addr"));
    m.def("mutex_unlock",  &py_mutex_unlock,  py::arg("addr"));
    m.def("mutex_recover", &py_mutex_recover, py::arg("addr"));

    // Slot state CAS
    m.def("try_pin",       &py_try_pin,       py::arg("addr"), py::arg("expected_gen"));
    m.def("unpin",         &py_unpin,         py::arg("addr"));
    m.def("get_gen",       &py_get_gen,       py::arg("addr"));
    m.def("get_pin",       &py_get_pin,       py::arg("addr"));
    m.def("can_evict",     &py_can_evict,     py::arg("addr"));
    m.def("evict_slot",    &py_evict_slot,    py::arg("addr"));
    m.def("publish_slot",  &py_publish_slot,  py::arg("addr"), py::arg("new_gen"));

    // 通用原子
    m.def("atomic_load_u64",      &py_atomic_load_u64,      py::arg("addr"));
    m.def("atomic_store_u64",     &py_atomic_store_u64,     py::arg("addr"), py::arg("val"));
    m.def("atomic_fetch_add_u64", &py_atomic_fetch_add_u64, py::arg("addr"), py::arg("delta"));
    m.def("atomic_fetch_or_u64",  &py_atomic_fetch_or_u64,  py::arg("addr"), py::arg("mask"));
    m.def("atomic_fetch_and_u64", &py_atomic_fetch_and_u64, py::arg("addr"), py::arg("mask"));

    // Stage 6: refcnt
    m.def("refcnt_inc", &py_refcnt_inc, py::arg("addr"));
    m.def("refcnt_dec", &py_refcnt_dec, py::arg("addr"));
    m.def("refcnt_get", &py_refcnt_get, py::arg("addr"));
    m.def("refcnt_set", &py_refcnt_set, py::arg("addr"), py::arg("v"));

    // Stage 6: hash 表 (probe 返回 (slot, gen) 元组)
    m.def("ht_probe",   &py_ht_probe,   py::arg("base"), py::arg("cap"), py::arg("hash"));
    m.def("ht_insert",  &py_ht_insert,  py::arg("base"), py::arg("cap"),
          py::arg("hash"), py::arg("slot_id"));
    m.def("ht_set_gen", &py_ht_set_gen, py::arg("base"), py::arg("cap"),
          py::arg("hash"), py::arg("gen"));
    m.def("ht_remove",  &py_ht_remove,  py::arg("base"), py::arg("cap"), py::arg("hash"));
    m.def("ht_clear",   &py_ht_clear,   py::arg("base"), py::arg("cap"));

    // Stage 6: 链式 block hash
    m.def("block_hash", &py_block_hash, py::arg("prev"), py::arg("data"));

    // 常量查询
    m.def("pthread_mutex_size",      &py_pthread_mutex_size);
    m.def("pthread_mutex_alignment", &py_pthread_mutex_alignment);
    m.def("arena_gen_mask",          &py_arena_gen_mask);
    m.def("arena_pin_shift",         &py_arena_pin_shift);
    m.def("arena_pin_max",           &py_arena_pin_max);
    m.def("errno_eownerdead",        &py_errno_eownerdead);
    m.def("arena_ht_entry_bytes",    &py_arena_ht_entry_bytes);
}
