# Pinned Arena LRU 改造 + 直读 CUDA Kernel 完整设计

> 本文档是 2026-05-28 ~ 05-29 期间多轮讨论的完整定稿。包括：
> 1. 改造背景与现状问题
> 2. 全面设计（数据结构、算法、跨进程协议、kernel 接口）
> 3. 实施路线图
> 4. 验证清单与风险点
> 5. 不在本次改造范围的相关项
>
> 用途：防止上下文遗失，作为后续动手依据。

---

## Part 0 · 改造背景

### 0.1 现状

`vllm/v1/core/sched/licht_v3/round_kv_store.py` 实现的 `RoundKVStore` 提供 PD-disaggregated 多轮 agent serving 下的跨进程 KV 复用 arena。当前架构：

- 一块 `/dev/shm/_arena.bin`（默认 24 GB）整片 mmap + `cudaHostRegister`，prefill 与 decode 共享
- **FIFO 环形 bump 分配器**：单调递增 `next_slot`，物理 slot 号 = `next_slot % num_slots`
- 环满后自动覆盖最早数据
- 唯一保护：active job 的 `inc_0` 不被自己后续轮次覆盖
- 每个 STORE 增量是 arena 上一段**物理连续 slot run**
- `.slot` 文件存 `(bump_base)`，prefill 校验有效性靠 `base ≥ next_slot - num_slots`

### 0.2 FIFO 的根本问题

1. **冷热不分**：常用 prefix 与一次性对话同等淘汰
2. **多轮尾段无谓占用**：N 轮对话的最后一轮 inc 在 N+1 轮 store 后再无价值，FIFO 不区分
3. **多 tenant 竞争下热门 prefix 被冲走**：长跑会议被新 tenant 频繁淘汰

### 0.3 Load 路径的另一问题

当前 load 走 **两步**：
1. cudaMemcpyAsync 把 arena 连续 run 搬进 GPU staging buffer（512MB 复用）
2. 按层 `index_put_` 把 staging 散写进 paged KV

对比 LMCache 的 `multi_layer_kv_transfer` kernel：**一次 CUDA kernel 直接从 cudaHostRegister'd host pinned 读 + 散写 paged**，无中间 GPU staging。我们多出一次 HBM 中转 + nL 次 dispatch 开销。

---

## Part 1 · 整体架构

替换 FIFO 为 **基于文件系统真相 + 共享 mutex + 原子 pin 的 LRU 分配器**；并独立优化 load kernel 干掉 GPU staging。两个改造**完全解耦**，分两个独立项目做。

**条件性增量**：在基础 LRU arena 上叠加 **跨 job 内容寻址 + refcnt 共享前缀**，把 arena 从 per-job LRU 升级为 cross-job content-addressable pool（详见 Part 12）。**Month 1 workload characterization 数据决定是否做**：跨请求 block 重复率 > 30% 才启动阶段 6。

### 1.1 核心模型

- **写者**（decode 多轮 store / decode preempt-save / prefill ARENA_SINK）：持跨进程 mutex → alloc/evict/memcpy/commit 临界区 → 释放
- **读者**（多轮 prefill lookup+load / phase1 decode self-recover / phase2 decode admission）：lock-free
  - gen-check → atomic pin → gen-double-check → load → post-load gen-check → pin release
- **LRU 真相**：`{arena_root}/{job_id}/manifest.json` 的 mtime（文件系统天然一致）
- **物理布局**：每 block 一个独立 slot，无连续段约束（slot-paged）

### 1.2 淘汰策略

**两层**：
- 上层 per-job LRU：按 manifest mtime 升序，最老 job 当受害者
- 下层 per-job 尾巴优先：从最大 inc_id 开始释放

**不做**：
- inc_0 active 保护（任何 inc 都可淘）
- `mark_finished` 特殊处理（让它自然沉到 LRU 底）
- shadow scheduler 喂预测进 LRU
- grace-period 时间窗（reader pin 已经足够）

### 1.3 读写路径全表

| 路径 | 触发 | 写者 | 读者 |
|---|---|---|---|
| 多轮复用 | 默认开 | decode（每轮 enqueue_store） | prefill（下轮 lookup+load） |
| Phase 1: save-on-preempt | `LICHT_PHASE1_SAVE_ON_PREEMPT=1` | decode（preempt 时 save_preempted_sync） | decode（同进程，重新 admit 时 consumer load_batch） |
| Phase 2: admission gate | `LICHT_PHASE2_ADMISSION_GATE=1`, decode 占用>80% | prefill（接 ARENA_SINK RPC，enqueue_store） | decode（admit 时 lookup+consumer load） |

新 LRU arena 必须**同时**支持这三条路径。

---

## Part 2 · 数据结构与文件格式

### 2.1 共享 mmap `/dev/shm/_arena.bin`（数据区）

```
shape: [num_slots, nL, 2, *rest]
typical: 24 GB / 2.097 MB-per-slot = ~12000 slots
register: 整片 cudaHostRegister
layout: block-major（保持现状）
```

### 2.2 共享 mmap `/dev/shm/_arena.hdr`（元数据区）

```c
struct ArenaHdr {
    pthread_mutex_t   alloc_mutex;                   // PROCESS_SHARED + ROBUST
    uint64_t          free_bitmap[num_slots / 64];   // 1=free, 0=used
    uint64_t          slot_state[num_slots];         // pin(16) | gen(48)
};
```

- 12000 slots → bitmap 1.5 KB + state 96 KB ≈ **100 KB hdr**
- ftruncate 到 128 KB（对齐 32 页）
- 两边进程 mmap 同样 size、同样 layout
- `free_bitmap` 修改受 alloc_mutex 保护，不需原子
- `slot_state` 读写 reader/writer 互见，需原子（CAS）

`slot_state[s]` 编码：
- 高 16 位 = pin count（容量 65535 reader 并发，远远超够）
- 低 48 位 = gen（1000 次/秒下要 8000 年才用完）
- 单个 uint64 既能 atomic load/store 又能 atomic CAS

### 2.3 文件系统结构

```
{arena_root}/
  _arena.bin                    # 数据区
  _arena.hdr                    # 元数据区
  {job_id}/                     # 每 job 一目录
    manifest.json               # total_blocks + token_ids；mtime = last_store_ts
    inc_000000000_000000050.slot   # inc_0
    inc_000000050_000000080.slot   # inc_1
    ...
```

### 2.4 `.slot` 文件格式

```
header (8 字节):     n = num_blocks_in_inc (uint64)
records (16n 字节):  对每 block i ∈ [0, n):
                       slot_id : int64    # 物理 slot 号
                       gen     : int64    # store 时的 gen
```

典型 inc 10-30 block → 文件 168-488 字节。写入用 mkstemp + atomic rename。

### 2.5 per-process in-memory 缓存

```python
# decode + prefill 各自维护，懒加载、按需失效
self._inc_cache       : dict[job_id, (mtime, list[(s,e,path)])]    # 1s TTL
self._slot_cache      : dict[path, (slot_ids, gens)]               # write-once 永久
self._job_mtime_cache : dict[job_id, mtime]                        # 1s TTL，LRU 排序用
self._last_stored     : dict[job_id, int]                          # 本进程视角
```

---

## Part 3 · 关键操作算法

### 3.1 读操作（lookup + load）

```python
def lookup(job_id, prompt_tokens):
    """读 manifest 算 LCP，校验每 inc 的有效性，返回 matched_blocks"""
    manifest = read_manifest(job_id)
    if not manifest: return None
    stored = manifest['token_ids']
    lcp = longest_common_prefix(stored, prompt_tokens)
    matched_blocks = lcp // block_size
    if matched_blocks <= 0: return None

    # 按 inc 顺序扫，遇 gen mismatch 即截断
    inc_list = list_incs(job_id)                # 1s TTL cache
    valid_blocks = 0
    for (s, e, slot_path) in inc_list:
        if s != valid_blocks: break              # gap in coverage
        slot_ids, gens = read_slot_file_cached(slot_path)
        for (sid, g0) in zip(slot_ids, gens):
            state = atomic_load_u64(&slot_state[sid])
            if (state & 0x0000FFFFFFFFFFFF) != g0:
                return (valid_blocks * block_size, valid_blocks)
        valid_blocks = e
        if valid_blocks >= matched_blocks: break

    return (min(valid_blocks, matched_blocks) * block_size,
            min(valid_blocks, matched_blocks))


def load_batch(items):
    """整波 wave 一起 load，每 slot 用 try_pin 防止 mid-load 被覆盖"""
    plan = []
    for (job_id, dst_block_ids, src_offset) in items:
        slot_pairs = resolve_slots(job_id, src_offset, len(dst_block_ids))
        if slot_pairs is None:
            continue                              # miss
        pinned = []
        success = True
        for (sid, g0) in slot_pairs:
            if not try_pin(sid, g0):              # CAS pin+1 且 gen 不变
                success = False
                break
            pinned.append((sid, g0))
        if not success:
            for (sid, _) in pinned: unpin(sid)
            continue                              # treat as miss
        plan.append((dst_block_ids, pinned))

    # Launch H2D + scatter（旧 staging 路径 或 直读 kernel）
    run_load_kernel(plan)

    # Post-load 二次校验 + unpin
    for (dst, pinned) in plan:
        for (sid, g0) in pinned:
            state = atomic_load_u64(&slot_state[sid])
            if (state & 0x0000FFFFFFFFFFFF) != g0:
                mark_request_load_invalid(...)    # 走 fallback recompute
            unpin(sid)
```

### 3.2 try_pin / unpin（C++ atomic）

```cpp
// uint64 slot_state[s] = (pin << 48) | gen
// 高 16 位 = pin，低 48 位 = gen

bool try_pin(uint64_t* state_ptr, uint64_t expected_gen) {
    uint64_t old = __atomic_load_n(state_ptr, __ATOMIC_ACQUIRE);
    while (true) {
        uint64_t cur_gen = old & 0x0000FFFFFFFFFFFFULL;
        if (cur_gen != expected_gen) return false;
        uint64_t cur_pin = old >> 48;
        if (cur_pin == 0xFFFF) return false;          // pin 饱和
        uint64_t neu = ((cur_pin + 1) << 48) | cur_gen;
        if (__atomic_compare_exchange_n(state_ptr, &old, neu, false,
                                        __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return true;
        // CAS 失败：old 已被刷成最新值，重试循环
    }
}

void unpin(uint64_t* state_ptr) {
    // pin -= 1; gen 不变
    __atomic_sub_fetch(state_ptr, 1ULL << 48, __ATOMIC_RELEASE);
}

// 写者 evict 时检查 pin（在 mutex 内调用，无需 CAS）
bool can_evict(uint64_t* state_ptr) {
    return (__atomic_load_n(state_ptr, __ATOMIC_ACQUIRE) >> 48) == 0;
}

// 写者 evict（在 mutex 内，无并发 evict）
void evict_slot(uint64_t* state_ptr) {
    uint64_t old = __atomic_load_n(state_ptr, __ATOMIC_RELAXED);
    uint64_t new_gen = (old & 0x0000FFFFFFFFFFFFULL) + 1;
    __atomic_store_n(state_ptr, new_gen, __ATOMIC_RELEASE);  // pin 仍为 0
}
```

### 3.3 写操作（store）

```python
def store(job_id, inc_id, block_ids, token_ids):
    """整段在跨进程 mutex 内：alloc + evict + memcpy + commit"""
    with arena_mutex:                                  # pthread_mutex_lock
        n_needed = len(block_ids)
        slot_ids = alloc_n_slots(n_needed)
        if slot_ids is None:
            # 空间不够，触发 evict
            need = n_needed - count_free()
            evict_until_free(need)
            slot_ids = alloc_n_slots(n_needed)
            assert slot_ids is not None

        # 拿到 slot_ids，gather + memcpy 进 arena
        gather_to_arena(block_ids, slot_ids)           # GPU→CPU→shm

        # 发布新 gen
        gens = []
        for sid in slot_ids:
            old_state = atomic_load_u64(&slot_state[sid])
            assert (old_state >> 48) == 0              # 已在 evict 时确认
            new_gen = (old_state & 0x0000FFFFFFFFFFFFULL) + 1
            atomic_store_u64(&slot_state[sid], new_gen)
            gens.append(new_gen)

        # 写 .slot 文件（mkstemp + atomic rename）
        write_slot_file(job_id, inc_id, slot_ids, gens)

        # 重写 manifest.json（atomic rename → mtime 自动刷新 → LRU 自动提权）
        rewrite_manifest(job_id, total_blocks=inc_end_block,
                         token_ids=token_ids)
    # 释放 mutex


def alloc_n_slots(n):
    """从 free_bitmap 找 n 个 free slot；不需连续"""
    free = []
    for i in range(num_slots):
        if bitmap_get(i) == 1:
            free.append(i)
            if len(free) == n: break
    if len(free) < n: return None
    for sid in free: bitmap_clear(sid)
    return free


def evict_until_free(need):
    """LRU 上层 + tail-first 下层"""
    while need > 0:
        victim_job = pick_lru_victim()                 # 扫 mtime
        if victim_job is None:
            raise NoSpace("arena 完全锁死 / 全部 pin")
        incs = list_incs(victim_job)                   # 升序
        progress = False
        for inc in reversed(incs):                     # 尾巴优先
            slot_ids, _ = read_slot_file(inc.path)
            freeable = [s for s in slot_ids
                        if can_evict(&slot_state[s])]
            if not freeable:
                continue                                # 整 inc 锁着，换下一个 inc
            for sid in freeable:
                evict_slot(&slot_state[sid])            # bump gen
                bitmap_set(sid)                         # 标 free
            need -= len(freeable)
            progress = True

            # ★ self-heal：同步回退 _last_stored
            update_last_stored(victim_job, inc.start_block)

            if need <= 0: break
        if not progress:
            # 这个 victim 所有 inc 都锁着，跳到下一个 victim
            mark_victim_skipped(victim_job)


def pick_lru_victim():
    """扫 {arena_root}/{job_id}/manifest.json 的 mtime，取最老"""
    candidates = []
    for entry in os.listdir(arena_root):
        if entry.startswith('_'): continue
        try:
            mtime = os.stat(f"{arena_root}/{entry}/manifest.json").st_mtime
        except FileNotFoundError:
            continue
        candidates.append((mtime, entry))
    if not candidates: return None
    candidates.sort()
    return candidates[0][1]
```

### 3.4 Self-Heal: evict 后回退 `_last_stored`

```python
def update_last_stored(job_id, evict_start_block):
    """evict 把 [evict_start_block, ...] 段淘了 → 回退 _last_stored
    下一轮 store 时增量自然从 evict_start_block 续起，arena heal 回连续"""
    cur = self._last_stored.get(job_id, 0)
    if evict_start_block < cur:
        self._last_stored[job_id] = evict_start_block
        # 也要重写 manifest 反映这个变化（下次 lookup 不会被误导）
        rewrite_manifest(job_id, total_blocks=evict_start_block)
        # 注：另一进程读 manifest 时自动同步（atomic rename）
```

**Self-heal 工作原理**：

```
初始：inc_0 [0,50), inc_1 [50,80), inc_2 [80,120)  → 全 arena
evict tail：inc_2 被淘

  _last_stored 回退 80（原 120 → 80）
  manifest.total_blocks 回退 80

round 4 来：
  prefill lookup → 走 inc_0, inc_1 → valid_end = 80
  prefill chunk-fill [80, current_end)
  decode 跑完，要 store

  end = align_blocks(tokens_this_round) = 150
  _last_stored = 80
  delta = block_ids[80:150]   # GPU paged buffer 里就有这段 KV
  → 新 inc 覆盖 [80, 150)

arena 状态：inc_0, inc_1, new_inc [80, 150)  → 全部连续 ★
round 5 lookup: valid_end = 150, 全部 reusable
```

---

## Part 4 · 代码组织

### 4.1 文件改动清单

| 文件 | 改动 | 内容 |
|---|---|---|
| `vllm/v1/core/sched/licht_v3/round_kv_store.py` | 大改 ~50% 重写 | `_arena_init` / `_arena_alloc` / `_write_inc_arena` / `_load_request_arena` / `lookup` / `_arena_valid_prefix_blocks` 全部按新模型重写 |
| `vllm/v1/core/sched/licht_v3/csrc/arena_atomic.cu`（新） | 全新 ~200 行 | 跨进程 mutex 初始化/lock/unlock；`try_pin` / `unpin` / `can_evict` / `evict_slot`；`atomic_load_u64` / `atomic_store_u64` |
| `vllm/v1/core/sched/licht_v3/csrc/fused_scatter.cu` | 加 ~150 行（阶段 5）| 新增 `arena_scatter_direct` kernel：源 = host pinned arena 指针 + slot_id 数组 |
| `vllm/v1/core/sched/licht_v3/csrc/setup.py` | 小改 | 导出 atomic helper 的 Python binding |
| `vllm/v1/core/sched/licht_v3/fused_scatter.py` | 加 ~30 行 | `get_atomic_helpers()` / `get_scatter_from_arena()` getter |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` | 小改 ~50 行 | `start_load_kv` 内 load 路径调用新接口；增加 post-load 校验失败的 fallback；Phase 2 写路径接入新 alloc API |
| `vllm/v1/core/sched/scheduler.py` | 小改 ~30 行 | post-load fallback 时重置 request 的 `num_external_computed_tokens=0` |

### 4.2 新增辅助文件

- `vllm/v1/core/sched/licht_v3/csrc/arena_atomic.h`：C++ atomic 操作 header
- `vllm/v1/core/sched/licht_v3/arena_layout.md`：文档（hdr 格式、`.slot` 格式、协议）
- `vllm/v1/core/sched/licht_v3/test_arena_lru.py`：单元测试

---

## Part 5 · 实施路线图（5 阶段，6 天）

### 阶段 0 · 准备（0.5 天）

- [ ] 写 `arena_layout.md`
- [ ] git tag 标记改造起点
- [ ] 备份当前可工作的 RoundKVStore

### 阶段 1 · 原子原语 + hdr 升级（1.5 天）

- [ ] 在 `csrc/` 加 `arena_atomic.cu`：
  - 跨进程 `pthread_mutex_t` + `PTHREAD_PROCESS_SHARED` + `PTHREAD_MUTEX_ROBUST` 初始化
  - `try_pin` / `unpin` / `can_evict` / `evict_slot`
  - `atomic_load_u64` / `atomic_store_u64`
- [ ] 更新 `setup.py` 暴露所有 helper
- [ ] 写 `test_arena_atomic.py`：
  - 单线程基本功能
  - 多线程 pin/unpin 压力（同一 slot 1000 个 reader 并发 pin/unpin 不串位）
  - 跨进程 mutex（fork + 两个进程交替 lock）
  - robust mutex 测试（一个进程持锁被 kill，另一进程能恢复）
- [ ] 改 `_arena_init`：hdr 改成 128 KB；初始化 mutex；初始化 slot_state 全 0、free_bitmap 全 1

**验收**：所有单元测试过

### 阶段 2 · slot-paged 分配器（2 天）

- [ ] 重写 `_arena_alloc(n)` 为 free_bitmap 扫描
- [ ] 实现 `_pick_lru_victim`（scan mtime + 1s TTL cache）
- [ ] 实现 `_evict_until_free`（尾巴优先 + skip pinned）
- [ ] 实现 `update_last_stored` self-heal
- [ ] 重写 `_write_inc_arena`：alloc → memcpy → bump gen → 写新 .slot
- [ ] 改写 `.slot` 文件读写（新格式）
- [ ] 改写 `lookup`：gen 校验代替 ring 边界
- [ ] 改写 `_load_request_arena`：try_pin + load + post-load + unpin
- [ ] **保留旧 GPU staging kernel**（先不动 kernel）

**验收**：
- 多轮 trace_replay 1 小时跑，token-level diff 与改造前**完全一致**
- 故意小容量 + 多 job，观察 LRU 行为
- 监控 evict 频率、pin 冲突次数日志

### 阶段 3 · post-load fallback + scheduler 接入（1 天）

- [ ] connector.start_load_kv 检测 post-load gen mismatch → 标记 request `load_invalid`
- [ ] 通过 `get_finished` 或新机制告诉 scheduler：这个请求 reuse 失效
- [ ] scheduler 把这个 request 的 `num_external_computed_tokens` 当 0 处理，下 step 走 chunked prefill
- [ ] 加 metric/日志：fallback 触发次数

**验收**：
- 人为注入 race（delay load + 触发 evict），fallback 正确触发
- fallback 后 token 输出与无 race 一致

### 阶段 4 · Phase 1/2 接入 + stale-pin 清理（1.5 天）

- [ ] Phase 1 `save_preempted_sync` 走新 alloc 接口
- [ ] Phase 2 prefill ARENA_SINK handler 走新 alloc 接口
- [ ] 实现 stale-pin 清理：定时扫 `pin > 0` 且 `last_pin_ts > 30s` 的 slot，强制清零（需要给 pin 加时间戳，hdr 中 slot_state 旁加一个 last_pin_ts 数组，或者用 robust 机制）
- [ ] 测试 LICHT_PHASE1=1 + LICHT_PHASE2=1 同时开

**验收**：
- preempt-recovery 链路正确
- admission gate 路径正确
- 三条路径并发跑无死锁、无错配

### 阶段 5 · 直读 kernel（独立项目，1 天，可选）

> 本阶段独立于 LRU 改造，不在本次范围内，待 LRU 稳定后单独启动。
> 详见 Part 6。

### 阶段 6 · 内容寻址扩展（条件性，约 7.5 天）

> 本阶段是基础 LRU arena（阶段 0-4）之上的增量。Month 1 workload characterization 数据支持时启动。
> 完整设计见 Part 12。

主要工作：
- Content hash 计算（chain-based per block）
- 共享 hash 表（hdr 内，open-addressing + epoch）
- Per-slot refcnt（atomic uint16）
- Store 路径改造：hash lookup → hit 则 refcnt++ 不分配新 slot
- Lookup 路径改造：cross-job 路径补全跨 job 共享前缀
- Eviction 路径改造：refcnt-- 后才看是否真销毁
- `.slot` 格式扩展：每条记录加 `content_hash` 字段
- Self-heal 在 refcnt 下的语义验证

**验收**：
- 实测真实 workload dedup 率 > 30%（前提：workload 有共享前缀）
- 跨 job 共享路径在 trace 上 token-level 正确
- refcnt 在并发 store/evict 下不泄漏、不双减
- arena 等效容量提升的可测量指标

- [ ] `fused_scatter.cu` 加 `arena_scatter_direct` kernel：
  - 签名：`(uint64 arena_host_ptr, int64* src_slots, int64* dst_idx, int64* layer_ptrs, nb, nL, dim, NBLK, P)`
  - kernel 内：grid `(nb, nL, 2)`，每 thread block 处理一个 (block, layer, kv)
  - 源地址：`arena_host_ptr + ((src_slots[j] * nL + li) * 2 + kv) * P`（PCIe 直读）
  - 目标地址：`layer_ptrs[li] + ((dim==1) ? (kv*NBLK+blk)*P : (blk*2+kv)*P)`
  - vectorized int4（要求 P % 8 == 0、dtype == 2 byte）
- [ ] Python 端 `LICHT_ROUND_KV_DIRECT_KERNEL=1` 开关
- [ ] 灰度对比：旧 staging vs 直读 engine_block_ms

**验收**：直读 engine_block_ms 下降 ≥ 30%

---

## Part 6 · 直读 CUDA Kernel 详细设计（阶段 5）

### 6.1 当前 fused_scatter.cu 结构

```cpp
// 当前：源是 GPU staging
__global__ void licht_scatter_kernel(
    const uint16_t* staging,           // 源：GPU buffer
    const int64_t* idx,                // dst block ids
    const int64_t* layer_ptrs,
    int nb, int nL, int dim, long NBLK, long P)
{
    long total_runs = (long)nb * nL * 2;
    long P8 = P >> 3;
    for (long run = blockIdx.x; run < total_runs; run += gridDim.x) {
        int kv = run & 1;
        long m  = run >> 1;
        int li  = m % nL;
        long j  = m / nL;
        long blk = idx[j];
        const uint16_t* src = staging + (((j*nL + li)*2 + kv) * P);
        uint16_t* dstbase = (uint16_t*)layer_ptrs[li];
        long dstoff = (dim==1) ? ((kv*NBLK + blk)*P) : ((blk*2 + kv)*P);
        for (long r = threadIdx.x; r < P8; r += blockDim.x) {
            ((int4*)(dstbase+dstoff))[r] = ((const int4*)src)[r];
        }
    }
}
```

### 6.2 直读改造

```cpp
// 新：源是 host pinned arena（cudaHostRegister'd）
__global__ void arena_scatter_direct_kernel(
    const uint16_t* arena_host_ptr,    // ★ 改为 host 指针
    const int64_t* src_slots,          // ★ 新增：每 block 在 arena 的物理 slot 号
    const int64_t* dst_idx,            // dst block ids
    const int64_t* layer_ptrs,
    int nb, int nL, int dim, long NBLK, long P)
{
    long total_runs = (long)nb * nL * 2;
    long P8 = P >> 3;
    for (long run = blockIdx.x; run < total_runs; run += gridDim.x) {
        int kv = run & 1;
        long m  = run >> 1;
        int li  = m % nL;
        long j  = m / nL;
        long blk     = dst_idx[j];
        long slot_id = src_slots[j];   // ★ 新增 indirection
        // ★ 源地址改算 arena 物理 slot
        const uint16_t* src = arena_host_ptr
                            + ((slot_id*nL + li)*2 + kv) * P;
        uint16_t* dstbase = (uint16_t*)layer_ptrs[li];
        long dstoff = (dim==1) ? ((kv*NBLK + blk)*P) : ((blk*2 + kv)*P);
        // ★ int4 vectorized 通过 PCIe 直读 host pinned
        for (long r = threadIdx.x; r < P8; r += blockDim.x) {
            ((int4*)(dstbase+dstoff))[r] = ((const int4*)src)[r];
        }
    }
}
```

### 6.3 C++ wrapper 签名

```cpp
void arena_scatter_direct(
    int64_t arena_host_ptr,        // 不是 tensor，直接传 host 指针整数
    torch::Tensor src_slots,       // GPU int64 [nb]
    torch::Tensor dst_idx,         // GPU int64 [nb]
    torch::Tensor layer_ptrs,      // GPU int64 [nL]
    int64_t nb, int64_t nL, int64_t dim, int64_t NBLK, int64_t P);
```

**注意**：`arena_host_ptr` 必须用 `int64_t` 整数传，不能包成 CUDA tensor（会报"非 device pointer"）。

### 6.4 Python 端 `_load_batch_arena` 重写（简化版）

```python
def _load_batch_arena(self, items):
    results = [False] * len(items)
    all_src = []
    all_dst = []
    for k, (job_id, dst_blocks, src_off) in enumerate(items):
        runs = self._resolve_arena_runs(job_id, dst_blocks, src_off)
        if runs is None:
            continue
        results[k] = True
        for (slot_a, cnt, dst_sub) in runs:
            for i in range(cnt):
                all_src.append(slot_a + i)
                all_dst.append(dst_sub[i])
    if not all_src:
        return results

    src_slots = torch.as_tensor(all_src, device=dev, dtype=torch.long)
    dst_idx   = torch.as_tensor(all_dst, device=dev, dtype=torch.long)

    lstream = self._get_load_stream()
    with torch.cuda.stream(lstream):
        self._fn_arena(
            self._arena_host_ptr_int,        # host 指针
            src_slots, dst_idx, self._layer_ptrs,
            len(all_src), self._arena_nL, self._arena_dim,
            self._fused_NBLK, self._fused_P,
        )
    ev = torch.cuda.Event()
    ev.record(lstream)
    torch.cuda.current_stream().wait_event(ev)
    return results
```

**砍掉的代码**：
- `_get_stage_gpu` / `_stage_gpu`（512 MB GPU staging）
- `_next_stage_pin` / `_stage_pins[2]`
- chunk 切分循环（cap_blocks）
- 每 chunk 一次 `as_tensor(idx_all)` 的 idx H2D
- `staging[pos:pos+span].copy_(arena_view[...])`
- `_load_stream_scatter` 配置

**显存峰值净省**：512 MB（staging buffer）

**GPU HBM 带宽净省**：每 byte 走两次 HBM（写 staging + 读出散写）→ 每 byte 只走一次（PCIe 直接写 paged）

### 6.5 fallback 路径

- .so 旧版本（无 `arena_scatter_direct` 符号）→ 回落 staging 路径
- dtype 非 fp16/bf16 → 回落
- P % 8 != 0 → 回落
- arena 未成功 `cudaHostRegister` → 回落 raw/safetensors 路径
- 单请求 miss → fallback recompute（不影响其他请求）

### 6.6 灰度开关

`LICHT_ROUND_KV_DIRECT_KERNEL=1`（默认 0）。先线上灰度验证 PCIe 带宽确实接近 24 GB/s、KV 内容字节级正确，再设为默认。

---

## Part 7 · 验证清单

### 7.1 正确性

- [ ] trace_replay 全程 token-by-token diff，改造前后零差异
- [ ] 长跑 4 小时无 hang / 无 deadlock
- [ ] Phase 1/2 单独开 trace 跑通
- [ ] 手动注入 race 触发 fallback，输出仍正确
- [ ] kill -9 一个进程持有 mutex → 另一进程能恢复（robust mutex）
- [ ] kill -9 reader 持有 pin → 30s 后清零

### 7.2 性能

- [ ] 多轮 lookup → load 延迟分布无回退
- [ ] arena 满载下 evict 频率合理（< 10/sec）
- [ ] pin CAS 冲突频率 < 0.1%（日志计数）
- [ ] post-load fallback 频率 < 0.01%
- [ ] hdr 共享内存峰值 ~128 KB
- [ ] 阶段 5 完成后：直读 engine_block_ms 下降 ≥ 30%

### 7.3 鲁棒性

- [ ] arena 满 + 所有 inc 都 pinned → 写者得到 NoSpace 错误，不死锁
- [ ] 跨进程并发 alloc 压力测试（两进程同时大量 store）
- [ ] 大规模 LRU evict 压力（小容量 arena + 大量 job）

---

## Part 8 · 风险点

| 风险 | 严重度 | 缓解 |
|---|---|---|
| C++ atomic 实现 bug → 正确性问题 | 高 | 阶段 1 充分单元测试 + 阶段 2 token diff |
| robust mutex 在 Python ctypes 下行为意外 | 中 | 用 native C++ helper 而非纯 ctypes |
| pin 泄漏（reader 崩了不减） | 中 | 阶段 4 加 stale-pin 清理（last_pin_ts + 30s 阈值） |
| LRU 扫 mtime 在 job 数爆炸时慢 | 低 | 1s TTL cache；超过 5000 job 时考虑 hdr-list 备用 |
| Phase 2 临界区跨进程 mutex 排队 | 低 | 先观测；必要时分 shard |
| evict 时 `update_last_stored` 回退错误 | 中 | 阶段 2 单元测试覆盖；trace_replay 验证 |
| 直读 kernel PCIe 带宽不达预期 | 中 | 阶段 5 灰度对比；不达预期可继续用 staging |
| 跨进程 hdr 格式两边不一致 | 高 | 启动时清根目录 + 版本号校验 |

---

## Part 9 · 不在本次改造范围

明确不做：
- NUMA-aware allocation（LMCache 有，PD 单机场景不强需要）
- Hugepage backing arena
- 自动 defrag（slot-paged 不会真碎片）
- 跨主机 arena 共享（PD 单机外）
- 旧版本 hdr/`.slot` 格式兼容（启动清根目录）
- pin/gen 元数据跨重启持久化
- shadow scheduler 喂预测进 LRU
- inc_0 active 保护
- finished job 特殊路径

**条件性纳入**（取决于 Month 1 workload characterization）：
- **跨 job 内容寻址 + refcnt 共享前缀**（Part 12，作为独立增量阶段 6 实施）

---

## Part 10 · 决策记录

按 commit 时间排，记录所有拍板：

| 日期 | 决策项 | 内容 |
|---|---|---|
| 2026-05-29 | 淘汰策略 | per-job LRU + per-job 尾巴优先 |
| 2026-05-29 | inc_0 active 保护 | 不做 |
| 2026-05-29 | finished job 特殊处理 | 不做 |
| 2026-05-29 | gen with job_hash | 不要，gen-only |
| 2026-05-29 | memory barrier | x86 TSO 自动保证，不需显式 barrier |
| 2026-05-29 | 物理布局 | slot-paged，每 block 独立 |
| 2026-05-29 | inc 概念 | 保留作为逻辑分组 |
| 2026-05-29 | `.slot` 格式 | per-block (slot_id, gen) 列表 |
| 2026-05-29 | evict 后 store 衔接 | 同步回退 `_last_stored` self-heal |
| 2026-05-29 | 走向 | B 路：全支持多轮 + Phase 1 + Phase 2 |
| 2026-05-29 | 跨进程同步 | 单跨进程 mutex 包临界区 + reader pin lock-free |
| 2026-05-29 | 架构 | 对称共享（filesystem 真相），非 single-owner |
| 2026-05-29 | LRU 真相来源 | manifest.json mtime |
| 2026-05-29 | gen 位宽 | 48 位 |
| 2026-05-29 | pin 位宽 | 16 位 |
| 2026-05-29 | grace-period | 不做（reader pin 已足够） |
| 2026-05-29 | post-load 二次校验 | 做（10 行兜底） |
| 2026-05-29 | 同步原语 | 编进 fused_scatter.so |
| 2026-05-29 | 直读 kernel | 独立项目（阶段 5），与 LRU 改造解耦 |
| 2026-05-29 | startup | 全清 arena 根目录 + hdr |
| 2026-05-30 | 内容寻址 + refcnt 扩展 | 作为阶段 6 独立增量；Month 1 workload characterization 决定是否启动 |
| 2026-05-30 | refcnt 数据结构 | 独立 `slot_refcnt[num_slots]` atomic uint16 数组（与 slot_state 分离） |
| 2026-05-30 | hash 表设计 | open-addressing + epoch tombstone，HASH_CAP ≈ 2×num_slots |
| 2026-05-30 | content hash 方案 | chain-based per block（仿 vLLM PagedAttention prefix cache） |
| 2026-05-30 | `.slot` 文件扩展 | 每记录从 16 字节 (slot_id, gen) 扩到 24 字节 (slot_id, gen, hash) |
| 2026-05-30 | refcnt 与 pin 关系 | 独立维护：pin 短生命周期（一次 load），refcnt 长生命周期（直到 manifest 被 evict） |
| 2026-05-30 | 跨 job 共享 evict 语义 | refcnt-- 才检查 == 0；为 0 时才 bump gen + free + hash 表删；self-heal 仅 logical 回退 _last_stored |

---

## Part 11 · 时间估算

| 阶段 | 工时 |
|---|---|
| 阶段 0 | 0.5 天 |
| 阶段 1 | 1.5 天 |
| 阶段 2 | 2 天 |
| 阶段 3 | 1 天 |
| 阶段 4 | 1.5 天 |
| **LRU 改造总计** | **6.5 天** |
| 阶段 5（独立）| 1 天 |
| **阶段 6（条件性增量）**| **7.5 天** |
| **含内容寻址的总计** | **14 天** |

按单人专注估，含测试调试。阶段 1-2 是主要风险点，可能 +1-2 天。
阶段 6 是阶段 2-4 完成后的独立增量，不必和前面交错。

---

## Part 12 · 内容寻址扩展（阶段 6，条件性）

> 本部分是基础 LRU arena（Part 0-5）的增量扩展。
> 启动条件：Month 1 workload characterization 实测真实 workload 的 block-level content 跨请求重复率 > 30%。
> 不启动条件：重复率 < 10%（增量收益不抵工程量）。

### 12.0 动机

基础 LRU arena 的 KV 复用粒度是 **per-job**：每个 job 自己存自己的 inc 链，跨 job 即使内容相同也各存一份。

**实际 workload 中跨请求共享的 prefix 普遍存在**：
- 多用户共用 system prompt（agent 场景常见 5K-50K token）
- RAG 系统共享检索文档（命中分布广）
- Few-shot examples 在 agent tool library 中复用

vLLM PagedAttention 已经在 **GPU paged buffer 层**做了内容寻址 dedup（automatic prefix caching）——但被挤出 GPU 的 prefix 落到 arena 时，**现状是 per-job 再各存一份**，dedup 失效。

本扩展把内容寻址延伸到 **host pinned arena 层**：跨 job 共享物理 slot，arena 等效容量倍增。

### 12.1 概念变化

| 维度 | 基础 LRU | + 内容寻址 |
|---|---|---|
| Slot 归属 | 一个 inc 独占 | 多个 inc 共享（refcnt） |
| Slot 销毁 | inc 被 evict 时 bump gen | 仅当 refcnt 减到 0 时才 bump gen |
| Store 寻路 | 总是 alloc 新 slot | 先查 hash，hit 则 refcnt++ |
| Lookup 寻路 | 自己 manifest 链 | 自己 manifest 链 + 跨 job hash 路径 |
| 容量利用 | sum(inc_sizes) | sum(unique_block_contents) |

### 12.2 共享 hdr 新增字段

在阶段 1 已扩展的 `ArenaHdr` 基础上增加：

```
struct ArenaHdr {
    pthread_mutex_t   alloc_mutex;
    uint64_t          free_bitmap[num_slots / 64];
    uint64_t          slot_state[num_slots];        // pin(16) + gen(48)
    uint16_t          slot_refcnt[num_slots];       // ★ 新增：atomic refcnt
    HashEntry         hash_table[HASH_CAP];         // ★ 新增：内容索引
};

struct HashEntry {
    uint64_t hash;           // 8 字节内容指纹
    int32_t  slot_id;        // 4 字节（-1 = empty / tombstone）
    uint32_t epoch;          // 4 字节（lock-free 读校验用）
};
```

容量估算（num_slots = 12000）：
- `slot_refcnt`：12000 × 2 = **24 KB**
- `hash_table`（HASH_CAP = 24593，下一质数）：24593 × 16 = **384 KB**
- hdr 总大小从 128 KB 升至 **~550 KB**（ftruncate 对齐到 1 MB）

### 12.3 Content hash 方案

**链式 hash**（chain-based）：每个 block 的 hash 同时编码 token 内容 + 前缀位置，确保 "block i 的 hash 命中 ⇔ 整个 [0, i] 前缀相同"。

```
def block_hash(block_idx, token_ids, prev_block_hash):
    # block_idx 用于防御 block_size 不一致
    h = sha256()
    h.update(struct.pack("<Q", prev_block_hash))   # 8 字节 prev
    h.update(struct.pack("<Q", block_idx))         # 8 字节 idx
    h.update(token_ids_bytes)                       # block_size × 4 字节
    return int.from_bytes(h.digest()[:8], 'little') # 截 64 位
```

初始 `prev_block_hash = 0`（或固定 magic）。

**性质**：
- block i 的 hash 唯一决定 [0, i] 整段 token 序列（碰撞概率 2^-64）
- 跨请求只有 prefix 完全相同时才会 hash 命中
- block size、tokenizer 不变即可保证两端进程算出同样 hash

### 12.4 `.slot` 文件格式扩展

每条记录从 16 字节 → 24 字节：

```
header (8 字节):     n = num_blocks_in_inc (uint64)
records (24n 字节):  对每 block i:
                       slot_id      : int64
                       gen          : int64
                       content_hash : int64
```

典型 inc 30 block → 720 字节，仍很小。

### 12.5 算法核心改动

#### Store 时 hash-then-alloc

```
def store_inc(job_id, inc_id, block_ids, token_ids):
    with alloc_mutex:
        # 计算每 block 的 chain hash
        hashes = []
        prev = 0
        for i in range(num_blocks):
            tok = token_ids[i*BS : (i+1)*BS]
            h = block_hash(global_block_index_i, tok, prev)
            hashes.append(h)
            prev = h

        # 对每 block 决定 alloc 还是 refcnt++
        records = []
        for i, h in enumerate(hashes):
            existing_slot = hash_table_lookup(h)
            if existing_slot is not None:
                # 跨 job 命中：refcnt++ 复用
                atomic_inc(slot_refcnt[existing_slot])
                gen_now = slot_state[existing_slot] & GEN_MASK
                records.append((existing_slot, gen_now, h))
            else:
                # miss：新 alloc + memcpy + 注册 hash
                slot_id = alloc_one_slot()  # 不够触发 evict_until_free
                memcpy(arena[slot_id], block_i_data_on_gpu)  # gather + H2D-style
                slot_refcnt[slot_id] = 1
                new_gen = bump_gen(slot_id)
                hash_table_insert(h, slot_id)
                records.append((slot_id, new_gen, h))

        write_slot_file(job_id, inc_id, records)
        rewrite_manifest(job_id, total_blocks=inc_end_block, token_ids=...)
```

**关键不变**：整段在 `alloc_mutex` 内，refcnt++ 和 alloc 互不竞争。

#### Lookup 时 own + cross-job 双路径

```
def lookup(job_id, prompt_tokens):
    # Path 1: 自己 manifest（快路径，复用基础 LRU 的逻辑）
    own_match_blocks = self_manifest_lookup(job_id, prompt_tokens)
    # 走 .slot + gen 校验，返回 valid_end_block

    # Path 2: 跨 job hash 命中（补全自己 manifest 之后的部分）
    extra_pairs = []  # (slot_id, gen, hash) 列表
    # 重算到 own_match 末尾时的 prev_hash
    prev_h = compute_prefix_hash_up_to(prompt_tokens, own_match_blocks)
    for i in range(own_match_blocks, total_prompt_blocks):
        tok = prompt_tokens[i*BS : (i+1)*BS]
        h = block_hash(i, tok, prev_h)
        slot_id = hash_table_lookup_lockfree(h)
        if slot_id is None: break  # 不再共享
        # 校验 slot 仍有效（refcnt > 0、未被 evict 后又 reuse）
        state = atomic_load(slot_state[slot_id])
        gen_now = state & GEN_MASK
        rc = atomic_load(slot_refcnt[slot_id])
        if rc == 0: break  # 已被全部释放
        # 校验 hash 表里这条记录还指向我们以为的 slot
        if not hash_table_verify(h, slot_id): break
        extra_pairs.append((slot_id, gen_now, h))
        prev_h = h

    return own_match_blocks + len(extra_pairs), extra_pairs
```

**注意**：Path 2 的 extra_pairs **不修改 refcnt**——lookup 只是"查询是否能用"，引用关系是 manifest 写入时建立的。要真正"拿到"这些 slot 的引用，必须走下一步 store 触发 hash-then-alloc。

#### Eviction 时 refcnt-- 才物理释放

```
def evict_inc(job_id, inc):
    # 读这个 inc 的 .slot 文件
    records = read_slot_file(inc.path)
    for (slot_id, gen, h) in records:
        new_rc = atomic_dec(slot_refcnt[slot_id])
        if new_rc == 0:
            # 真销毁
            bump_gen(slot_id)
            bitmap_set_free(slot_id)
            hash_table_remove(h, slot_id)
        # else: 仅减引用，slot 数据保留给其他 manifest

    delete inc.slot_file
    update_last_stored(job_id, inc.start_block)  # self-heal 仍生效
    rewrite_manifest(job_id, total_blocks=inc.start_block)
```

### 12.6 Self-heal 在 refcnt 下的语义

之前 self-heal 是"evict 后下一轮 store 续接"。**有 refcnt 后**：

- **逻辑层面**：`_last_stored` 回退到被 evict 的 inc 起点，下一轮从这里续 store
- **物理层面**：
  - 如果 evict 的 inc 里 slot 都是独占（refcnt=1 → 0）→ 物理释放
  - 如果有共享（refcnt > 1 → 仍 > 0）→ 数据保留
- **下一轮 store 时**：
  - 算 hash 链
  - 即使被淘的那段 inc，如果 hash 表里还有它的 slot 注册（被别的 job 用着），**lookup-then-alloc 会命中 refcnt++**
  - **等于免费复用回来**

也就是说：refcnt + content addressing **让 self-heal 在共享场景下更强**——不仅是逻辑续接，物理上也可能直接复用。

### 12.7 跨进程 hash 表的并发

#### 写（insert / remove）

**总是发生在 `alloc_mutex` 临界区内**（store 和 evict 都持锁）。
单写者模型，简单 atomic + open-addressing 即可。

#### 读（lookup）

Lookup 在 mutex 外（性能关键路径）。**Lock-free 读**通过 epoch 实现：

```
def hash_table_lookup_lockfree(target_hash):
    idx = target_hash % HASH_CAP
    for probe in range(HASH_CAP):
        entry = hash_table[(idx + probe) % HASH_CAP]
        e0 = atomic_load(entry.epoch)
        h = atomic_load(entry.hash)
        s = atomic_load(entry.slot_id)
        e1 = atomic_load(entry.epoch)
        if e0 != e1: continue  # 读到撕裂，重试
        if s == -1 and not entry.is_tombstone(e0): return None  # 真空
        if h == target_hash and s >= 0: return s
        # 否则继续探测
    return None

def hash_table_verify(target_hash, expected_slot):
    """二次校验：lookup 后到使用之间没被改"""
    idx = target_hash % HASH_CAP
    for probe in range(HASH_CAP):
        entry = hash_table[(idx + probe) % HASH_CAP]
        e0 = atomic_load(entry.epoch)
        if entry.slot_id == expected_slot and entry.hash == target_hash:
            # 再读一遍 epoch
            e1 = atomic_load(entry.epoch)
            return e0 == e1
        if entry.is_empty(e0): return False
    return False
```

writer 每次修改 entry 都先 `epoch++` 再改字段再 `epoch++`，使奇数 epoch = 修改中。

### 12.8 Refcnt 与 pin 的关系（必须区分）

| | pin | refcnt |
|---|---|---|
| 含义 | "我正在读这个 slot" | "有 N 个 manifest 指向这个 slot" |
| 持有方 | reader（lookup→load 期间） | 各 job 的 .slot 记录 |
| 生命周期 | 一次 load，~10-50 ms | 直到 inc 被 evict，可能小时级 |
| 数据结构 | `slot_state` 高 16 位 | 独立 `slot_refcnt[]` |
| 阻止 evict | 是（writer 检查 `pin == 0`）| 是（refcnt > 0 不真销毁，但可减 refcnt） |
| 谁修改 | reader（CAS）| writer（在 mutex 内 atomic_inc/dec）|

**注意 evict 的语义**：
- `pin > 0` → 整个 slot 不能动（连 refcnt-- 都不行，因为 reader 假设 slot 内容稳定）
- `pin == 0 && refcnt > 1` → 可以 refcnt--（仅 evict 当前 job 的引用），但不真销毁
- `pin == 0 && refcnt == 1 && evict` → refcnt-- 到 0 → 真销毁

所以 evict 候选筛选要看**两个条件**：`pin == 0` AND（让它进入候选）；然后做 refcnt--。

### 12.9 Hash 碰撞处理

64-bit hash 与 12000 slot 的碰撞概率：~2^-52，**接近不可能**。但仍做防御：
- `.slot` 文件除了 hash，也存 `slot_id` 和 `gen`
- Lookup 命中 hash 后 → 跟着 `slot_state[slot_id]` 校验 gen，gen mismatch 直接 miss
- gen 比对作为兜底——内容真的不同时 gen 也会不同（因为 evict 后新内容会 bump gen）

实际上 **gen 校验提供了 hash 碰撞的隐式保护**：即使 64-bit hash 撞了，gen 也不会撞。

### 12.10 阶段 6 实施细分

| 子任务 | 工时 |
|---|---|
| Content hash 函数实现 + 测试 | 0.5 天 |
| 共享 hash 表 C++（open-addressing + epoch）| 1 天 |
| `slot_refcnt` atomic 数组初始化 + helper | 0.5 天 |
| Store 路径改造（hash → lookup → refcnt++ or alloc）| 1 天 |
| Lookup 路径改造（cross-job hash 补全）| 1 天 |
| Eviction 路径改造（refcnt-- → 条件销毁）| 0.5 天 |
| Self-heal + manifest 在 refcnt 下的语义测试 | 0.5 天 |
| `.slot` 文件 24 字节格式 + atomic rename | 0.5 天 |
| 单元测试（refcnt 正确性、跨 job 共享、并发）| 1 天 |
| 集成测试（trace 上跑通 + 性能数据）| 1 天 |

**总：~7.5 天**

### 12.11 验证清单（阶段 6 专属）

正确性：
- [ ] 单 job 多轮 store + 同内容前缀：第二轮起 refcnt > 1
- [ ] 两个 job 同 prompt 前缀：第二个 job store 时 hash hit → refcnt++
- [ ] evict 其中一个 job 的共享 inc：refcnt-- 到 1，slot 数据对另一 job 仍有效
- [ ] 两个 job 都 evict 共享 inc：refcnt -- 到 0，slot 真销毁
- [ ] 并发 store（多 writer 线程）下 refcnt 不泄漏、不双减
- [ ] hash 碰撞兜底：人工注入碰撞，gen 校验拦下

性能：
- [ ] 真实 workload 实测 dedup 率（共享 block / 总 block）
- [ ] arena 等效容量提升（unique blocks / total stored blocks）
- [ ] Lookup latency 不显著退化（hash 路径加在 own 路径之后，命中前几 block 即可）
- [ ] Store latency 加 hash 计算后 < 原 store 的 5%

### 12.12 阶段 6 风险

| 风险 | 严重度 | 缓解 |
|---|---|---|
| 真实 workload 重复率不到 30% → 收益不抵工程 | 高 | Month 1 数据驱动决策；不达标则跳过本阶段 |
| Hash 表 lock-free 读实现 bug → lookup 返回错 slot | 高 | epoch 双读 + gen 二次校验兜底；充分单元测试 |
| Refcnt 泄漏（评估时 hot reload / 异常路径漏减）| 中 | 周期性扫描 refcnt > 0 但所有 .slot 文件已删的 slot，告警 |
| Refcnt 双减 / 下溢 | 高 | atomic 操作 + assert(rc > 0)；崩溃比静默错好 |
| Hash 表满（HASH_CAP 不够）| 低 | 启动时 HASH_CAP 设 2× num_slots；如不够动态 alarm |
| 跨进程 epoch 撕裂读 | 低 | epoch 是 single-writer，读用 acquire/release 配对 |
| `.slot` 文件大小翻倍 | 低 | 24 字节 × 30 ≈ 720 字节，仍微小 |

### 12.13 论文意义

加入内容寻址扩展后，arena 工作从"per-job LRU 缓存"升级为：

> "Cross-process, cross-round, **cross-job content-addressable** KV pool with refcnt-based sharing"

跟现有工作对比：

| 系统 | 跨进程 | 跨 round | 跨 job 内容寻址 | refcnt 共享 |
|---|---|---|---|---|
| vLLM PagedAttention prefix cache | ✗（GPU 内） | ✗ | ✓ | ✓ |
| LMCache | 部分 | ✓ | ✗（per-chunk LRU）| ✗ |
| AttentionStore | ✗ | ✓ | ✗ | ✗ |
| Mooncake Store | ✓ | ✓ | ✓（hash key dedup）| ✗（多副本反向）|
| **本工作（含阶段 6）** | ✓ | ✓ | ✓ | ✓ |

**论文价值**：这套组合是 mainstream 没有的——尤其 "跨进程 PD-disagg + refcnt 共享" 是真实空白点。

但**仍不足以让 arena 成为 paper 的 C1**：它是"有 substance 的 infrastructure"，可在论文 § implementation 多写一些细节、用 dedup rate 实验数据撑场，但不应当主 contribution 卖。

---

## 12.14 实施记录 (as-built, 2026-06-03)

第一部分 (6a 基础设施 + 6b store dedup + 6e refcnt evict) 已落地, `LICHT_ARENA_CONTENT_ADDR=1`
门控, 关时旧路径字节级不变. commit: 6a `534df92`, 6b+6e `25f47c4`.

### 相对前文设计的精化 (讨论后定稿)

1. **HashEntry 不存 gen, 维持 16B `{hash(8), slot_id(4), epoch(4)}`**.
   原 §12.2 纠结过把 gen 放进 entry. 定论: gen 48 位留在 `slot_state` 不动;
   命中表拿到 slot_id 后, 期望 gen 从 **.slot (v2)** 读、当前 gen 从 slot_state 读
   比对. epoch 仅用于无锁 probe 的 seqlock 防撕裂, 与 gen 无关.

2. **保留 .slot, 不删** (一度考虑全部走表查、每 job 只剩 manifest). 定论保留并升
   v2 (24B 含 hash): ① own-path 走 per-进程缓存的 .slot, 不碰共享表无跨进程 cache
   争用; ② evict 直接读 .slot 拿 hash, 不必在锁内重算链式 hash. "收成 manifest-only"
   挂为 6f 实测 own-vs-crossjob 比例后再议.

3. **store 三段式 + insert 在 CS1** (§12.5 旧伪码 memcpy 在锁内、insert 时机模糊).
   实际: CS1 probe+HIT(refcnt++)/MISS(alloc+**CS1 内 insert**+refcnt=1) → 锁外只对
   MISS 搬数据 → CS2 只 publish MISS. insert 必须在 CS1: 跨进程同 mutex 串行, 并发
   同内容写者出 CS1 即见 entry → HIT, 不会重复分配 slot.

4. **HIT refcnt++ 必须排在 evict 之前**. 否则 evict 可能把某 HIT slot (若属受害 job)
   refcnt 减到 0 释放, 之后 inc 到已释放 slot. 先 ++ 保证 evict 时它 refcnt>=2 不被淘.

5. **写窗口 gen 安全 (补 §12.9 完整证明)**: A 锁外写 MISS 数据期间 slot_state.gen 停在
   "淘汰后、未发布"的裸奔值; 旧引用期望值 < 它、新引用 (A 的 entry) 期望值 > 它, 且该值
   从无 entry 认领 → 任何 reader try_pin 必失配 fail-closed, 绝不读半写数据. 写-写则
   CS1 串行后者必 HIT 不重复写. 故无需额外锁/屏障.

6. **pinned 块淘汰**: 只淘 `pin==0`; pinned 跳过留下轮 (不做延迟释放, 对齐 §12.8).
   evict 另加 `ht_probe(hash)==slot` 纵深防御, 防 slot 被淘后复用给别 hash 时误减 refcnt.

7. **崩溃恢复**: 接受 EOWNERDEAD 后罕见 refcnt 泄漏 (arena 非持久化, 重启自愈),
   v1 不做 refcnt 重建 (原 §12.12 风险表里的"周期扫描"降级为按需 6h).

### 部署不变量 (重要)

**`LICHT_ARENA_CONTENT_ADDR` 必须在 prefill 和 decode 两端取值一致.** 否则一端写 v1 .slot
(无 hash、无 refcnt/表), 另一端 (=1) 的 evict 用 v2 读会判为损坏直接 unlink, 破坏数据.
启动脚本须对两个角色统一设置. 默认 0 (关), 安全.

### SSD 分层挂载点 (本期不实现)

CPU `lookup` 全 miss 返回 0/None 即为将来 SSD 层入口. CPU 与 SSD 分开: CPU 命中走
arena load; CPU miss 才下探 SSD. 本期只做 CPU.

### 待办 (第二部分 + 收尾)

- 6c/6d 跨 job lookup (全新 job 直接命中别 job 前缀, 省 prefill 重算): 需改 lookup
  Path2 (无锁 ht_probe 续链) + load 接表解析的跨 job slot 对. 测完第一部分 dedup 率再启.
- 6f 收尾: 实跑 trace 验证 token-level 一致 + 跨进程双 writer refcnt 不泄漏 + 埋点看
  dedup 命中率 / own-vs-crossjob 比例, 据此定是否收 manifest-only.

## 12.15 lookup/load 性能优化 + idle 根因排查 (2026-06-04, 实跑驱动)

6c/6d 上线后实跑发现 `idle_ms ≫ load_ms`。逐层埋点定位 + 三连优化 (均不改语义):

1. **lookup 缓存** (commit 232d83a): `get_num_new_matched_tokens` 与
   `update_state_after_alloc` 对同一请求各查一次 (每步 2×), 未准入还跨步反复 probe.
   按 request_id 缓存 `lookup_resolve` 结果 (`_rk_lookup_cached`), build_connector_meta
   每步丢弃未再 probe 的条目. load 末尾 try_pin fail-closed 兜底, 缓存安全.
2. **lookup_resolve 下沉 C** (commit 875bc5e): 整条 per-block 循环 (链式 hash +
   ht_probe + gen/refcnt 校验) 放进 `arena_lookup_resolve` 一个 C 调用, ~48× (微基准
   800 块 1.35ms→0.028ms). Python 端 `struct.pack` 整条 prompt 一次传入.
3. **★ 根因** (commit 954aae4): `bind_kv_caches` 只在 worker 侧建 `_lru_store`, 但
   lookup 跑在 **scheduler 侧** connector 实例 → 一直落到 `self.lookup` 读 .slot 文件
   (own-job, 文件 IO ~32ms/次), content-addr 表查询 + C-loop **从未在 scheduler 侧跑过**.
   修复: worker 写 `_arena_meta.json {num_slots, block_size}`; scheduler 侧
   `_ensure_lookup_store()` 据 meta lazy `open_or_create` 一个只读表 LruArenaStore
   (共享同一 shm hdr, 不绑 GPU) → lookup 走 C 表; 顺带 load 改 `load_pin_explicit` 不读文件.
   实测 (reqs≈22): lookup_ms 1565→~50, admit_loop 2500→~700, load_ms 3000→~100, idle 大降.

**idle 根因结论 (排查到底)**: idle 大头是**模型 prefill forward 本身**. 用 CUDA event
量 `gpu_fwd` (forward GPU 真实时长 3-11s) ≈ `fwd_ms`(CPU model() 发射, 被 GPU 反压拖住)
+ `bookkeep_sync`(`_bookkeeping_sync` D2H 采样 token 时同步等 forward 算完). GPU 全程
满载, compute-bound, 只能靠减 token (复用) 缩短. 剩 <1s/步 GPU util=0 = 同步引擎循环里
CPU 调度 (vLLM 块分配 + LICHT timeline + detokenize + handoff) 不与 forward 重叠,
round-kv 只占 ~120ms. 消这 <1s 需引擎 async/重叠调度 (LICHT 侧, 非 round-kv).
**round-kv 优化线到此为止.**

---

## 附录 A · 相关已有技术对比

### vs LMCache
- 共同：pinned arena、LRU、按 chunk/slot 组织 KV
- 差异：LMCache LRU 是 per-chunk（独立对象），我们是 per-job 两层（LRU + 尾巴优先）
- 差异：LMCache 支持多 backend（NUMA/hugepage/shm），我们只 shm
- 差异：LMCache 单进程内为主，我们 cross-process PD-shared

### vs Mooncake
- 不同 setting：Mooncake 是分布式 RDMA，我们是单机 shm
- 相似：都用 paged block + hash key + 多副本（Mooncake）vs 单副本（我们）

### vs vLLM swap_in
- vLLM swap：CPU mirror GPU paged，per-block cudaMemcpyAsync
- 我们：独立 layout（block-major slot），全功能 LRU

### vs AttentionStore (CachedAttention)
- 共同：多轮 KV 复用，hierarchical 存储
- 差异：AttentionStore 是 host memory + disk 两级，我们单级 shm
- 差异：AttentionStore scheduler-aware eviction，我们 mtime LRU + tail-first

---

## 附录 B · 相关代码入口（改造时索引）

| 入口 | 位置 |
|---|---|
| `RoundKVStore` 类 | `vllm/v1/core/sched/licht_v3/round_kv_store.py:74` |
| `_arena_init` | `:359` |
| `_arena_alloc` | `:479` |
| `_write_inc_arena` | `:919` |
| `_load_batch_arena` | `:1898` |
| `_load_request_arena` | `:1769` |
| `lookup` | `:1122` |
| `_arena_valid_prefix_blocks` | `:1156` |
| connector 调用 | `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:343-476` |
| scheduler 接入 | `vllm/v1/core/sched/scheduler.py:1615-1771` |
| fused kernel | `vllm/v1/core/sched/licht_v3/csrc/fused_scatter.cu` |

---

**文档版本**：v1.0（2026-05-29 初稿）
**状态**：设计定稿，待动手实施

---

## 待查问题（暂放，arena 改造完成后处理）

### idle_ms 偏高 — 调度/桥接层，非 arena
- 现象: 请求已 arriving（日志可见），但 Running=0 Waiting=0，引擎空转 ~4.5s 才开始 load
  - 20:48:47 大批请求 arriving
  - 20:48:48 Running=0 reqs, Waiting=0 reqs  (到了但没跑没排队)
  - 20:48:55 STEP idle_ms=4581
- 判断: 不是"没请求"(请求明明到了), 也不是 arena load 慢 (load_ms=1626 正常, GPU 24.6 GB/s)
- 怀疑(按可能性):
  1. BRIDGE/decode 反压: prefill 等 decode 收走上一波 KV (NCCL 桥接阻塞)
  2. scheduler admit 节流: LICHT timeline gate 挡着, 请求 arriving 后没进 waiting
  3. proxy -> engine 之间排队
- 与 arena LRU / load 路径无关, 独立问题
- 优先级: 低于 Stage 3/4, 等 arena 改造收尾后查

---

## Stage 4 完成记录 (2026-06-01)

### Phase 2 (admission gate) - GPU 端到端验证通过
两次生产跑 (LICHT_PHASE2_GATE_THRESHOLD=0.30, --round-kv-lru):
- 写侧 (prefill 当 arena writer, 新场景): 146 sink = 146 enqueued = 143 D2H done,
  135 次 prefill LRU write_inc. 计数自洽.
- 读侧 (decode 从 arena 读回): 146 admit-from-arena, 93 consumer load ok=1
  (差额为正常调度时序).
- 跨进程双 writer (prefill 135 + decode 多轮): 0 冲突.
- 0 race, 0 crash, 0 alloc-fail, 0 deadlock.

### 发现并修复的真 bug
1. alloc_n 跨进程 free_count 缓存失真 (commit 2c850b1): 改为直接扫共享 bitmap.
2. consumer-load gate bug (commit 95243ce): decode 读回 arena 的 load 之前只在
   phase1 开关下触发, Phase-2-only 部署会导致 sink 的 KV 永远读不回. 改为
   (phase1 OR phase2).

### Phase 1 (save-on-preempt) - 代码完成, 未实跑触发 (默认成功)
- 代码审查: save_preempted_sync 时序在 LRU 下安全 (gather 完才 mark_done).
- 未触发: 抢占在 Phase 2 准入门槛下本来就少, decode KV 峰值 ~63% 没满到抢占.
- 用户决定: 先默认成功, 全量跑遇到再验证. 风险低 (路径与 Phase 2 写/读共用).

### stale-pin 清理 - 评估后降级不做
- 进程崩=重启=清空 arena 无累积; load 异常有 finally release 兜底.
- 真正泄漏只剩代码 bug 靠测试抓. 优先级低暂不做.
