# LICHT 抢占策略与 Fake Delay-Free 修复日志

本文档记录一次对 prefill-only worker 在 LICHT 模式下出现的大规模请求超时 + engine 长期空转问题所做的修复。改动覆盖 4 个相关 bug，分 3 个文件，共约 130 行代码。最关键的一条修复是连接器层对 abort 请求的 delay-free 误判（Bug 4）——它是整个雪崩链的唯一入口。

---

## 1. 背景与症状

### 1.1 实验设置

- **模型**: Llama-3.1-8B-Instruct (fp16, enforce-eager)
- **GPU**: prefill worker 1 GPU (利用率 0.95)、decode worker 1 GPU
- **调度器**: LICHT 模式 (`--licht`)，prefill 端使用 `licht_prefill_sched_enabled`
- **KV transfer**: P2P NCCL, BLOCK_MIGRATE 模式
- **`long_prefill_token_threshold`**: 5242
- **`max_num_batched_tokens`**: 265944
- **`request_completion_timeout_s`**: 600
- **客户端 HTTP timeout**: 300 秒 (`multiturn_trace_client.py:122`)
- **Workload**: SWE-bench 多轮 agent 交互 trace，77 分钟

### 1.2 观察到的宏观现象

77 分钟实验里：

- **55 条请求生命周期 ≈ 300 秒被客户端砍掉**（`delay_free_start_ts - arrival_time ≥ 298s`）
- 其中 **46 条** `release_timeout=True`，`total_delay_free_duration = 600.0s`（等满了 `request_completion_timeout_s` 才被强制释放）
- **prefill engine 出现 11 次长空转**，`iteration_stats` 上相邻 active step 的间隔 > 30s，累计停工约 26 分钟
- 最长一次 stall 持续 **529.5 秒**（08:16:14 → 08:25:04）
- 雪崩呈 cluster 式发生：5 个簇聚集在 08:09、08:42、09:11、09:17、09:27 附近

### 1.3 典型受害者画像

`chatcmpl-28f51d60...` (`joke2k__faker...2ofipk7n`, `agent_round=4`, `prompt=15516`, 首条点火请求)：

```
T=0s       arrival (08:09:57.713)
T=227s     首次 waiting_to_running（LICHT 饥饿补偿爬到 ~6.3 分）
T=241s     Request_evicted_from_running_queue  ← 跑了 13.5s 被踢
T=289s     evicted_to_running                   ← 等 49s 重新被选
T=305s     Request_departure, num_generation_tokens=0  ← 客户端 300s 超时砍掉
T=905s     delay_free 强制释放（600s 超时）
```

真实 GPU 时间约 30 秒，**其中大半是重算被清零的进度**。

### 1.4 异常数据特征

broken 请求的 `monitoring_timestamps.delay_free_stats` 条目只有：

```json
{
  "delay_free_start_ts": 1776989703.32,
  "deferred_cleanup_ts": 1776990303.35,      // 正好 +600s
  "delay_free_end_ts": 1776990303.35,
  "total_delay_free_duration": 600.03,
  "finished_sending_ts": 1776990303.35,
  "release_timeout": true
}
```

**关键：缺少所有正常 delay-free 应该有的字段**（`bridge_staged_ts` / `bridge_popped_ts` / `migration_*` / `release_received_ts`）。说明这些请求从未 stage 过 bridge 到 decode 侧，decode 也从未见过它们——它们的 delay-free 状态是伪造出来的。

---

## 2. 根因定位：4 个相互串联的 Bug

### 2.1 Bug 2: LICHT 选进 ↔ FCFS 踢出 错配

**现象**: 高 round / 高 LICHT 分请求刚被 LICHT 从 waiting 选进 running，下一 step 就被从 tail 踢出。

**根因**:
- `_peek_waiting_request` (scheduler.py:432-442) 用 `max(waiting, key=licht_score)` 挑分最高的，然后 `self.running.append(request)` (scheduler.py:894) → **高分请求永远在 running tail**
- FCFS preempt 分支 (scheduler.py:621-623) `preempted_req = self.running.pop()` → **永远从 tail pop**

两个动作方向完全相反：前门 LICHT 挑赢家，后门 FCFS 踢赢家。

**实测**: `arrow-py round=7` 133 秒熬出饥饿分数进来后被踢 2 次；`joke2k round=4` 225 秒熬进来后被踢 1 次。

### 2.2 Bug 3-A: Preempt 进度清零导致 thrashing

**现象**: 被 preempt 的 chunked prefill 请求 `num_computed_tokens` 归零，已算过的所有 tokens 全部作废，resume 只能靠 prefix cache 保住 ~960 tokens。

**根因** (scheduler.py:631-632):

```python
preempted_req.status = RequestStatus.PREEMPTED
preempted_req.num_computed_tokens = 0
```

vLLM v1 的 preempt 语义是"释放所有 KV block + 计算进度清零"。这个设计在不开 chunked prefill 时不是问题——完整 prefill 一 step 原子完成，根本不会在中间被 preempt。但 chunked prefill 场景下每个 step 之间都是 preempt 风险点。

**实测**: arrow-py 三次 running (18.3s + 13.5s + 16s = 47.8s) 里前两段的 ~16K tokens 全部白算。

### 2.3 Bug 3-B: Delay-free 准入控制一刀切

**现象**: `_num_delay_free_blocks > 0`（哪怕 1 个 block）+ `allocate_slots` 失败时，scheduler 直接 `break`，不 preempt、不调度。

**根因** (scheduler.py:586-592):

```python
if new_blocks is None:
    if self._num_delay_free_blocks > 0:
        can_schedule = False
        break
```

设计本意合理：如果有 delay-free 马上会 RELEASE，等 1-2 秒比抢占强。但检查只看 `> 0` 不看 delay-free 有多老。

**在 Bug 4 触发下的灾难**: Bug 4 产生的 fake delay-free 永远等不到 RELEASE，要 600 秒超时才释放。这 600 秒里准入控制一直 break → engine 彻底空转。

实测 08:16:14 起的 529.5 秒 stall 就是这个路径。

### 2.4 Bug 4: Connector 把 abort 当作 delay-free (点火器和放大器)

**现象**: 客户端 300 秒超时砍掉一条未完成 prefill 的请求后，它的 KV block 不会立刻释放，而是被错误标记 delay-free，锁 600 秒。

**根因** (`p2p_nccl_connector.py:666-667`):

```python
if self.is_producer and self.direct_block_mode:
    return len(block_ids) > 0, None   # ← 只要分过 block 就返回 True
```

`request_finished` 有两个进入路径：

| 路径 | 请求状态 | bridge 是否 staged | 正确处理 |
|---|---|---|---|
| `update_from_output` → trace_replay_end | prefill 完成 + sampled 1 token | 已 staged | `delay_free=True` |
| `finish_requests` (外部 abort) | prefill 可能中途被砍 | 未 staged | **应该 `delay_free=False`** |

原代码不区分两条路径，一律 `True`。

**后果链**:

1. 中途被砍的请求被标记 delay-free，`_num_delay_free_blocks` 错误增加
2. Decode 从未收到这条请求的 bridge metadata → 永远不发 RELEASE
3. 锁 600 秒直到 `pending_release_deadlines` 超时强制 free
4. 这 600 秒里触发 Bug 3-B 的空转，engine 停工
5. 停工期间 running 里的大 prompt 也陆续被客户端 300s 砍掉
6. 每砍一条又走 Bug 4，产生新的 fake delay-free → 滚雪球

**实测**: 46 条 broken 请求全部是这个路径。

### 2.5 四个 Bug 的因果链

```
[点火] round=4/7 低 round 请求靠饥饿补偿进入 running
  ↓
[Bug 2] 进入即被 FCFS pop tail 踢出
  ↓
[Bug 3-A] 被踢 = 进度归零 → 重算浪费
  ↓ 反复 2-3 次
[客户端 300s 超时] abort
  ↓
[Bug 4] abort 误标 delay-free，KV block 锁 600s
  ↓
[Bug 3-B] _num_delay_free_blocks > 0 → scheduler 准入控制死等
  ↓
[engine 空转数百秒] 期间 running 里的请求也被砍，每一条又走 Bug 4
  ↓ (雪崩放大)
[600s 超时强制 free] → engine 脱困，下一波 burst 再次重演
```

---

## 3. 修复方案

### 3.1 修复优先级

| Bug | 优先级 | 是否必修 | 改动量 |
|---|---|---|---|
| Bug 4 | **P0** | 必修（唯一的点火器入口）| ~30 行 |
| Bug 2 | **P1** | 必修（消除主要 thrashing）| ~70 行 |
| 方案 B (wait_start 不重置) | **P1** | 必修（LICHT 分数回血）| ~15 行 |
| Bug 3-A | 选修 | 方案 B + Bug 2 + Bug 3 新 preempt 策略已间接缓解；原方案的 min-run quota 暂不上 | 0 行 |
| Bug 3-B | 选修 | Bug 4 修复后不再触发；不改 | 0 行 |

**本次实际修复**: Bug 4 + Bug 2 + 方案 B + 新的三维度 preempt 策略。

### 3.2 方案 B: wait_start 永远用 arrival_time

**问题**: 原 `_reset_licht_waiting_state` 把 `licht_waiting_round_start_ts[req_id] = time.monotonic()`。请求被 preempt 回 waiting 时 wait_start 被重置为"now"，T_wait 从 0 开始，LICHT 饥饿补偿瞬间清零。k=4 的请求熬 200 秒挣到的 1.44 分补偿，被踢后立刻跌回 4.83，要再等 200 秒才能回到同分。

**修法** (`scheduler.py:388-402`):

```python
def _reset_licht_waiting_state(
    self,
    request: Request,
    now_monotonic: Optional[float] = None,  # kept for back-compat; ignored
) -> None:
    # Plan B: wait_start is always the request's arrival_time; it is
    # never reset on preempt.
    if not self.licht_enabled:
        return
    self.licht_waiting_round_start_ts[request.request_id] = (
        request.arrival_time)
```

`_ensure_licht_waiting_start_timestamps` 同步改成 `setdefault(req_id, req.arrival_time)`。

**时钟统一**: `arrival_time` 是 wall clock (`time.time()`)，原 LICHT 打分用 `time.monotonic()`——两个时钟坐标系不一致，直接相减会错。统一改成 wall clock：
- `_compute_licht_prefill_score(request, now)` 里 `now` 必须是 `time.time()`
- `_peek_waiting_request` 里 `now = time.time()`
- `_pick_preempt_victim_licht` 里 `now = time.time()`

**语义**: T_wait = `now - arrival_time`，从请求到达起算，running 期间继续累加，preempt 后不归零。

### 3.3 Bug 2 修复: LICHT-aware preempt 选择器

**目标**: preempt 选人策略和 admit 选人策略对称——都按 LICHT 打分走。

**新方法** (`scheduler.py:471-537`):

```python
def _pick_preempt_victim_licht(
    self,
    scheduler_request: Request,
) -> Optional[Request]:
    """三维度 rank 归一化加权:
        EvictScore = 0.5 * rank_credit
                   + 0.2 * rank_preempt_count
                   + 0.3 * rank_real_computed
    选 EvictScore 最低的踢 (低 = 最该被踢)。
    """
```

**三个维度**:

1. **rank_credit** (权 0.5): LICHT 打分升序 —— 低分优先被踢，符合 LICHT 语义
2. **rank_preempt_count** (权 0.2): 被踢次数升序 —— 曾被反复踢的 受害者豁免
3. **rank_real_computed** (权 0.3): `num_computed_tokens - num_cached_tokens` 升序 —— 已投入的 GPU 算力越少越该被踢（prefix cache 部分不计，那是白送的）

**归一化**: rank 归一化到 `[0, 1]`。比 min-max 稳定——不会被单一维度的极端值压扁其他维度。

**tie-breaking**: `(EvictScore, -arrival_time, request_id)` —— 分数相同时偏向踢新到的；再不行用 request_id 保证确定性。

**新字段**: `Request.preempt_count`（`request.py:117-123`），每次 preempt 时 +1。字段无条件存在（零开销），只有 LICHT 路径读写。

### 3.4 Preempt 分支改造

原代码 (`scheduler.py:697-701`) 只有 PRIORITY / CONTINUUM / FCFS (`else: self.running.pop()`) 三个分支。在 CONTINUUM 之后、原 `else` 之前插入 LICHT 分支：

```python
elif self.licht_prefill_sched_enabled:
    preempted_req = self._pick_preempt_victim_licht(request)
    if preempted_req is None:
        preempted_req = request   # 只剩自己，走 self-preempt 路径
    self.running.remove(preempted_req)
    if preempted_req in scheduled_running_reqs:
        scheduled_running_reqs.remove(preempted_req)
    if preempted_req.request_id in num_scheduled_tokens:
        del num_scheduled_tokens[preempted_req.request_id]
    if preempted_req.request_id in req_to_new_blocks:
        del req_to_new_blocks[preempted_req.request_id]
    self.continuum_recorder.request_evicted_from_running_queue(preempted_req)
```

完整清理 `scheduled_running_reqs` / `num_scheduled_tokens` / `req_to_new_blocks`——与 CONTINUUM 分支对齐。原 FCFS pop tail 天然不需要这个清理（tail 是刚 append 的、这一轮从未被 schedule 过），但 LICHT 可能从 running 中间某个位置踢出，必须完整清理。

**非 LICHT 路径零影响**: PRIORITY / CONTINUUM / 纯 FCFS 都走原分支。

### 3.5 Preempt 计数

在 `preempted_req.status = RequestStatus.PREEMPTED` 之后统一 `+= 1`：

```python
preempted_req.status = RequestStatus.PREEMPTED
preempted_req.num_computed_tokens = 0
# Accumulate victim count for the LICHT preempt selector.
preempted_req.preempt_count += 1
```

位置在 `if is_unpin: pass else:` 的 else 分支——只有真正"回 waiting"时才 +1，CONTINUUM 的 unpin 路径不 +1。不加 LICHT 条件（非 LICHT 路径不读这个字段）。

### 3.6 Bug 4 修复: request_finished 检查 bridge_staged

**新 helper** (`p2p_nccl_engine.py:295-309`):

```python
def was_bridge_staged(self, request_id: str) -> bool:
    """Return True iff stage_bridge_request has ever been called."""
    ts = self._delay_free_ts.get(request_id)
    return ts is not None and "bridge_staged_ts" in ts
```

读 `_delay_free_ts[request_id]["bridge_staged_ts"]` 判断 bridge 是否被发布过。不加 state_lock——写方是 worker forward 线程的 `stage_bridge_request`，读方是 scheduler 线程。极少数并发 race 下读到"尚未 set"的后果只是该请求降级成 abort 处理（立刻 free block），比原来的 600 秒锁好得多。

**`request_finished` 分支改造** (`p2p_nccl_connector.py:666-689`):

```python
if self.is_producer and self.direct_block_mode:
    bridge_staged = (
        self.p2p_nccl_engine is not None
        and self.p2p_nccl_engine.was_bridge_staged(
            request.request_id))
    if not bridge_staged:
        logger.debug(
            "[REQUEST_FINISHED] req=%s aborted before bridge "
            "staged; freeing blocks immediately", request.request_id)
    return (bridge_staged and len(block_ids) > 0), None
```

| 调用路径 | bridge_staged | 返回值 | 行为 |
|---|---|---|---|
| 正常完成 (trace_replay_end) | True | `(True, None)` | 进 delay-free 等 RELEASE |
| 外部 abort (client 300s timeout) | False | `(False, None)` | 立刻 free block |

---

## 4. 改动文件清单

| 文件 | 行数变化 | 内容 |
|---|---|---|
| `vllm/v1/request.py` | +5 | 加 `preempt_count: int = 0` 字段 |
| `vllm/v1/core/sched/scheduler.py` | +100 | 方案 B + Bug 2: `_reset/_ensure_licht` 语义改、`_pick_preempt_victim_licht` 新增、preempt 分支加 LICHT 分支、preempt_count 累加、时钟统一 wall-clock |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py` | +17 | 加 `was_bridge_staged` helper |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` | +21 | `request_finished` producer + direct_block_mode 分支加 bridge_staged 检查 |
| **合计** | **~143 行** | |

### 4.1 被保留但语义调整的既有代码

- **`_reset_licht_waiting_state`** 保留了签名和调用点（向后兼容），但内部逻辑改成永远设 `arrival_time`。调用方 `add_request`、preempt 路径都不动，新语义下这些调用幂等——T_wait 从不归零。
- **`_compute_licht_prefill_score`** 的参数名从 `now_monotonic` 改成 `now`，语义从 monotonic 时钟变成 wall clock。所有调用点同步改 `time.time()`。

### 4.2 未动的代码

- Bug 3-A 的 preempt 进度清零（`num_computed_tokens = 0`）保留——改动代价太大（涉及 kv_cache_manager），通过方案 B + Bug 2 新 preempt 策略间接缓解
- Bug 3-B 的准入控制（`if self._num_delay_free_blocks > 0: break`）保留——修完 Bug 4 后它回归合理的"等 RELEASE"设计
- 非 LICHT 路径（PRIORITY / CONTINUUM / 纯 FCFS）完全不变

---

## 5. 预期效果

### 5.1 Bug 4 修复直接消除的问题

1. **46 条 `release_timeout=True` 的 fake delay-free 应降到 0**（或极个位数边界 case）
2. **11 次 engine stall 中的 10 次消失**——只有真正 decode 侧 RELEASE 延迟时才会触发准入控制，典型持续 1-2 秒
3. **5 波雪崩不再发生**——因为没有 fake delay-free 累积，单条请求被砍不再拖垮 engine

### 5.2 Bug 2 + 方案 B 联合修复的问题

1. **高 round 请求刚进就被踢的现象消失**——preempt 选的是 LICHT 分最低的，而不是 tail
2. **低 round 熬饥饿进来的请求被保护**——方案 B 让它们 preempt 后 LICHT 分不归零，下一轮依然有机会被选回
3. **arrow-py / joke2k 类请求** 预计能在 300 秒内完成 prefill，不再撞客户端超时

### 5.3 三维度打分的预期分布

用 `0.5 / 0.2 / 0.3` 权重的初始调参下，典型被踢画像是：

- LICHT 分数不算最高（rank_credit 偏低）
- 没被反复踢过（rank_preempt_count 偏低）
- 已算 tokens 较少（rank_real_computed 偏低）

综合起来被踢的是"当前池里最容易牺牲"的请求。和之前 FCFS 无脑踢 tail（永远踢最新进来的）完全不同。

### 5.4 未解决但预期不致命的残余

- **Bug 3-A 仍在**: 被踢的人依然丢进度。但 Bug 2 改变了"谁被踢"的分布，挑的是 GPU 投入最少的人，单次损失最小
- **LICHT 打分死亡谷 (k=3~15)**: 第一项 `3·log(1+k)` 在这段增长慢、第二项 `exp(-k)·wait_term` 衰减快，两头都不占——这个需要调整 LICHT 打分函数形状才能彻底解决。当前修复通过方案 B 保证死亡谷请求的分数不再被 preempt 清零，**足够跑通大多数场景**，但极限压力下可能还是有少量 k=3~15 请求超时

### 5.5 验证方法

重跑原 SWE-bench trace，对比以下指标：

1. `monitoring_timestamps.delay_free_stats` 里 `release_timeout=True` 的条数（原 46 → 预期 0-5）
2. `monitoring_timestamps.delay_free_stats` 里 `total_delay_free_duration > 500s` 的条数（原 46 → 预期 0）
3. `iteration_stats` 里相邻 active iter 间隔 > 30s 的次数（原 11 → 预期 0-1）
4. `arrival_time - delay_free_start_ts >= 298s` 的请求数（原 55 → 预期 <10）
5. 目标请求 (arrow-py round=7 / joke2k round=4 / 各 cluster 首条) 的 `Request_evicted_from_running_queue_time` 次数（原 2-3 次 → 预期 0-1）

---

## 6. 风险与回滚

### 6.1 风险点

1. **时钟坐标系改动**: 如果有其他代码依赖 `licht_waiting_round_start_ts` 是 monotonic 时钟，会错。已 grep 确认无外部引用
2. **`was_bridge_staged` race**: 理论上存在读到"尚未写入"状态的极短窗口。后果是降级为"立刻 free block"，若此时 decode 刚要来 migrate 会拿不到 bridge → 走 BRIDGE_POP 超时（60 秒）失败一条请求。概率极低，远优于原 600 秒卡死
3. **三维度权重 (0.5/0.2/0.3) 未经实验调参**: 可能有更优组合。建议上线后观察 preempt 日志的受害者分布再调
4. **`Request.preempt_count` 字段**: 非 LICHT 路径也会被 `+= 1`，但没人读。如未来有新代码依赖它做非 LICHT 决策需注意语义

### 6.2 回滚方案

- 方案 B / Bug 2 的改动全部在 `licht_enabled / licht_prefill_sched_enabled` 门禁内。把这个开关关掉即可恢复原行为
- Bug 4 的改动在 `is_producer and direct_block_mode` 分支内。要紧急回滚只需注释掉 `bridge_staged` 检查、恢复 `return len(block_ids) > 0, None` 一行

### 6.3 环境变量 / 配置

本次修复没有引入新的配置项。所有阈值（0.5/0.2/0.3 权重、A/B/Tmax LICHT 参数、`request_completion_timeout_s`）沿用现有。

---

## 7. 后续工作（未实施）

按重要性排序：

1. **LICHT 打分死亡谷** (k=3~15): 考虑把 `exp(-k)` 换成 `1/(1+k)` 让低 k 的饥饿补偿衰减更慢。需要重新跑 ablation 验证
2. **Bug 3-A 彻底修复**: preempt 时保留 `num_computed_tokens` 和 KV block（"软 preempt"）。需要改 `kv_cache_manager` 支持 "preempted but retained" 状态，涉及 300-500 行
3. **Preempt watchdog 日志**: 每次 preempt 发生时记一行 `[PREEMPT] victim=X (round=Y, computed=Z, preempt_count=N) for req=R`，便于下次实验观察受害者分布是否合理
4. **Stall watchdog 日志**: 每 5 秒强制打一行 stall 状态，覆盖 `empty_iteration_stall_fix` 之后 stall 期间完全静默的问题
5. **Admission control 保险丝 (Bug 3-B)**: 对 delay-free 条目加"老化"判断，超过 30 秒未 RELEASE 的允许 preempt 绕过。作为异常 decode 场景（RELEASE 丢包、decode hang）的纵深防御
6. **`request_completion_timeout_s` 默认下调**: 当前 600 秒 > 客户端 300 秒 timeout，这个 gap 本身就是风险源。建议降到 120-180 秒

---

## 附录 A: 完整因果链时序图

```
 08:09:57  arrival (joke2k round=4)
   ↓ 225s waiting (LICHT 饥饿补偿慢慢积累)
 08:13:44  waiting_to_running (第一次被选进)
   ↓ 13.5s running (chunked prefill 做了 2 个 chunk)
 08:13:58  Request_evicted (Bug 2: FCFS pop tail 踢出)
           - num_computed_tokens = 0 (Bug 3-A: 进度清零)
           - wait_start 重置为 now (原 bug，方案 B 修复)
   ↓ 49s preempted (饥饿补偿从 0 重新累积)
 08:14:47  evicted_to_running (第二次被选进)
   ↓ 16s running (又做了 1 个 chunk)
 08:15:03  Request_departure (num_generation_tokens=0)
           - 客户端 300s 超时砍掉
           - connector.request_finished 返回 (True, None)  ← Bug 4
           - _num_delay_free_blocks += 485 blocks
   ↓
 08:16:14  其他正常完成的 delay-free 进来凑够 3000+ blocks
           - _num_delay_free_blocks > 0 触发 Bug 3-B
           - scheduler 空转开始
   ↓ 529s engine stall (期间 running 里其他请求也被砍，每一条又走 Bug 4)
 08:25:03  joke2k 的 fake delay-free 到 600s 超时
           - pending_release_deadlines 强制 free 485 blocks
 08:25:04  engine 脱困，新一轮调度开始
```

## 附录 B: 代码位置速查

| 改动 | 文件:行 |
|---|---|
| `preempt_count` 字段 | `vllm/v1/request.py:123` |
| `_reset_licht_waiting_state` (方案 B) | `vllm/v1/core/sched/scheduler.py:388-402` |
| `_ensure_licht_waiting_start_timestamps` (方案 B) | `vllm/v1/core/sched/scheduler.py:412-420` |
| `_compute_licht_prefill_score` (时钟) | `vllm/v1/core/sched/scheduler.py:422-445` |
| `_peek_waiting_request` (时钟) | `vllm/v1/core/sched/scheduler.py:447-455` |
| `_pick_preempt_victim_licht` (新增) | `vllm/v1/core/sched/scheduler.py:471-537` |
| Preempt 分支 LICHT 新增 | `vllm/v1/core/sched/scheduler.py:699-721` |
| `preempt_count += 1` | `vllm/v1/core/sched/scheduler.py:737` |
| `was_bridge_staged` helper | `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:296-309` |
| `request_finished` Bug 4 fix | `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:666-689` |
