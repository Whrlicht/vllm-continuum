// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// LICHT Round-KV Arena 跨进程原子原语
// ====================================
//
// 设计目标:
//   1. 跨进程 mutex (pthread_mutex_t PROCESS_SHARED + ROBUST)
//      包住 alloc + evict + memcpy + commit 临界区
//   2. Slot 级 reader pin (atomic CAS), lock-free 读保护
//   3. Slot 级 gen 计数器 (uint48), 跨进程 slot 失效判断
//
// 共享 hdr 布局 (mmap /dev/shm/_arena.hdr):
//   按页对齐, 总大小 ftruncate 到 1 MB (含 Stage 6 预留空间)
//
//   Field                Offset       Size            说明
//   ---                  ---          ---             ---
//   alloc_mutex          0            sizeof(mutex)   pthread_mutex_t PROCESS_SHARED + ROBUST
//   <padding to 64B>
//   free_bitmap          64           num_slots/8     1=free, 0=used
//   slot_state           64+bitmap    num_slots*8     pin(16) | gen(48), atomic CAS
//   slot_refcnt          [STAGE 6]    num_slots*2     atomic uint16 refcnt (预留, Stage 1-4 不动)
//   hash_table           [STAGE 6]    HASH_CAP*16     content-addressing index (预留)
//   <padding to 1 MB>
//
// 设计约束:
//   - 必须用 __atomic_* GCC builtin (std::atomic 不能跨进程)
//   - mutex 设为 ROBUST: 持锁进程崩溃时, 另一进程可 recover
//   - 所有 atomic 操作必须使用正确的 memory order
//     - load: ACQUIRE
//     - store: RELEASE
//     - CAS / fetch_add: ACQ_REL

#ifndef LICHT_ARENA_ATOMIC_H
#define LICHT_ARENA_ATOMIC_H

#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Slot state 编码常量
// ============================================================
// slot_state[i] = (pin << 48) | gen
//   pin: 16 bits, 容量 65535 个并发 reader (绝对超够)
//   gen: 48 bits, 每秒 1000 次 bump 也要 8000 年才用完
#define ARENA_GEN_BITS      48ULL
#define ARENA_PIN_BITS      16ULL
#define ARENA_GEN_MASK      ((1ULL << ARENA_GEN_BITS) - 1ULL)
#define ARENA_PIN_SHIFT     ARENA_GEN_BITS
#define ARENA_PIN_MAX       ((1ULL << ARENA_PIN_BITS) - 1ULL)

// ============================================================
// Stage 6: 内容寻址 hash 表 entry 编码常量
// ============================================================
// hash_table[i] 占 16 字节:
//   offset 0  : uint64 hash       (内容指纹, 链式)
//   offset 8  : int32  slot_id    (EMPTY=-1 / TOMBSTONE=-2 / >=0 物理 slot)
//   offset 12 : uint32 epoch      (seqlock 序号: 偶=稳定, 奇=写入中)
// 设计:
//   - insert/remove 在 alloc_mutex 内 (单写者), 但仍维护 epoch,
//     供 Stage 6c 跨进程无锁 probe (多读者) 检测撕裂读.
//   - open-addressing 线性探测; remove 用 tombstone 不回填, 保探测链完整.
//   - EMPTY != 0 (零页是 slot_id=0 有效 slot), 故 create 时必须 ht_clear.
#define ARENA_HT_ENTRY_BYTES   16
#define ARENA_HT_EMPTY         (-1)
#define ARENA_HT_TOMBSTONE     (-2)

// ============================================================
// 跨进程 mutex 操作
// ============================================================

// 初始化共享 mutex (在 hdr 偏移 0 处)
// 必须以 PTHREAD_PROCESS_SHARED + PTHREAD_MUTEX_ROBUST 属性初始化
// 返回 0 成功, 非 0 为 errno
int arena_mutex_init(pthread_mutex_t* m);

// 销毁 mutex
int arena_mutex_destroy(pthread_mutex_t* m);

// 加锁 (阻塞)
// 返回:
//   0          成功
//   EOWNERDEAD 前一进程持锁时崩溃, 需要调用 arena_mutex_recover 后才能再用
//   其他       errno
int arena_mutex_lock(pthread_mutex_t* m);

// 解锁
int arena_mutex_unlock(pthread_mutex_t* m);

// 前一进程崩溃时 (lock 返回 EOWNERDEAD) 调用
// 让 mutex 可重新使用
int arena_mutex_recover(pthread_mutex_t* m);

// ============================================================
// Slot pin / gen 操作 (无锁, CAS 风格)
// ============================================================

// 尝试给 slot 加 reader pin
//   - 仅当 slot 的 gen == expected_gen 且 pin < MAX 时成功
// 返回:
//   1 成功 (pin += 1)
//   0 失败 (gen 不匹配 / pin 饱和)
int arena_try_pin(uint64_t* slot_state, uint64_t expected_gen);

// 减 pin (与 try_pin 配对)
// 不检查溢出 (由调用者保证配对正确)
void arena_unpin(uint64_t* slot_state);

// 读取当前 gen / pin (atomic load)
uint64_t arena_get_gen(uint64_t* slot_state);
uint64_t arena_get_pin(uint64_t* slot_state);

// 检查 slot 是否可被淘汰 (pin == 0)
// 仅在 alloc_mutex 内调用 (writer 路径)
int arena_can_evict(uint64_t* slot_state);

// 淘汰 slot: 增加 gen, 宣告内容失效
// 仅在 alloc_mutex 内、且 can_evict 返回 1 之后调用
// 新 gen 通过 *new_gen_out 返回 (可传 NULL 忽略)
void arena_evict_slot(uint64_t* slot_state, uint64_t* new_gen_out);

// 为新分配的 slot 发布 gen
//   - 用于 store 路径: alloc -> memcpy -> publish
//   - pin 必须为 0
//   - new_gen 是当前 gen 已 +1 的值 (调用者保证)
// 仅在 alloc_mutex 内调用
void arena_publish_slot(uint64_t* slot_state, uint64_t new_gen);

// ============================================================
// 通用原子原语
// ============================================================
uint64_t arena_atomic_load_u64(uint64_t* addr);
void     arena_atomic_store_u64(uint64_t* addr, uint64_t val);
uint64_t arena_atomic_fetch_add_u64(uint64_t* addr, uint64_t delta);
uint64_t arena_atomic_fetch_or_u64(uint64_t* addr, uint64_t mask);
uint64_t arena_atomic_fetch_and_u64(uint64_t* addr, uint64_t mask);

// ============================================================
// Stage 6: slot refcnt (atomic uint16)
// ============================================================
// 每 slot 一个 uint16, 记"有几个 job manifest 引用这块". store 命中 +1,
// evict 减 1; 减到 0 才真 evict (bump gen + free + ht_remove).
// 仅在 alloc_mutex 内调用 (与 alloc/evict 同临界区).
uint32_t arena_refcnt_inc(uint16_t* p);   // 返回自增后的值
uint32_t arena_refcnt_dec(uint16_t* p);   // 返回自减后的值
uint32_t arena_refcnt_get(uint16_t* p);
void     arena_refcnt_set(uint16_t* p, uint16_t v);

// ============================================================
// Stage 6: 内容寻址 hash 表 (open-addressing + seqlock)
// ============================================================
// table_base: hash_table 区起始地址; cap: HASH_CAP (entry 数).
//
// probe: 无锁多读者安全 (seqlock epoch 检测撕裂). 返回 slot_id (>=0) 或 -1 (miss).
//        遇 EMPTY 即停 (探测链尽头); 遇 TOMBSTONE 继续.
// insert/remove: 仅在 alloc_mutex 内 (单写者); 仍维护 epoch 供 probe.
int64_t arena_ht_probe(void* table_base, uint64_t cap, uint64_t hash);
int     arena_ht_insert(void* table_base, uint64_t cap,
                        uint64_t hash, int32_t slot_id);  // 1 成功 / 0 表满
int     arena_ht_remove(void* table_base, uint64_t cap, uint64_t hash);  // 1 删除 / 0 未找到
void    arena_ht_clear(void* table_base, uint64_t cap);   // 全置 EMPTY (create 时调)

// ============================================================
// Stage 6: 链式 block hash (xxhash XXH3, seed=prev)
// ============================================================
// h[i] = arena_block_hash(h[i-1], token_bytes_i, len). seed 携带前缀,
// 保证 "block i 命中 <=> 整个 [0,i] 前缀 token 完全相同". 跨进程确定性.
uint64_t arena_block_hash(uint64_t prev, const void* data, uint64_t len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LICHT_ARENA_ATOMIC_H
