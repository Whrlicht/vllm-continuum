# 空转 Stats 堵塞 + BRIDGE_PUSH 死代码清理

本文档记录为修复 disagg serving 在 delay-free stall 期间出现的**多秒级 HTTP 响应延迟**所做的一系列改动，涉及三个独立但相互关联的变更：

1. **scheduler 空转迭代节流 stats**（根治事件循环拥堵）
2. **BRIDGE_PUSH 优化完全移除**（回到纯 decode-pull 模式）
3. **monitoring 字段清理**（删掉误导性与冗余指标）

## 1. 背景与症状

### 1.1 观察到的现象

在 `BLOCK_MIGRATE` 模式跑 SWE-bench 多轮流量时发现：

- **请求 `total_delay_free_duration` 由正常的 ~1s 激增到 15s 乃至 300s**
- Prefill 日志出现多达 2.5 分钟无任何 "request finishing" 事件的空白段
- `GPU KV cache usage` 长期保持 99%+ 不回落
- `monitoring_timestamps` 文件膨胀到 4 GB（单次 ~3.5 小时实验）

### 1.2 问题定位数据

事后 dump 出的 `monitoring_timestamps`：

- `scheduler_stats` 数组包含 **14,677,924 条记录**（~1165 条/秒）
- 同一个"空转 stall"区间内，连续上万条 stats 的 `num_running` / `kv_cache_usage` 完全相同，只有 `timestamp` 在前进

例如某个 stall 期的连续 entry：
```json
{"timestamp": 1776890650.9914606, "num_running": 0, "num_waiting": 2, "kv_cache_usage": 0.9883693, ...}
{"timestamp": 1776890650.9914727, "num_running": 0, "num_waiting": 2, "kv_cache_usage": 0.9883693, ...}
```
相邻时间戳差 ~12 μs——engine core 确实在以 ≥1000 Hz 的频率空转产生无信息量 snapshot。

## 2. 根因链

### 2.1 代码层面的因果

```
KV usage 99% + _num_delay_free_blocks > 0
  → scheduler.schedule() 的 running loop 命中
    "allocate_slots is None and delay_free > 0" 分支 (scheduler.py:589, 856)
    → break，total_num_scheduled_tokens = 0
  → execute_model 走 kv_connector_no_forward 分支
    (gpu_model_runner.py:2083-2088)
  → update_from_output 处理空 model_output
    → outputs dict 为空
    → engine_core_outputs dict 为空
    → 但 log_stats=True, make_stats() 仍产出 SchedulerStats
    → 代码 (原 scheduler.py:1419-1425) 强制把 stats 塞进
      engine_core_outputs[0] = EngineCoreOutputs(scheduler_stats=...)
    → output_queue.put_nowait 一条消息
  → engine core 主循环立刻进下一轮 iter（没有 sleep）
    → 每 ms 级产出一条 stats EngineCoreOutputs
```

### 2.2 API server 侧的放大

API server 进程跑一个单 asyncio event loop，里面并存：

- **coroutine A** `process_outputs_socket` (core_client.py:808-835)：zmq PULL → put 进 asyncio.Queue
- **coroutine B** `output_handler` (async_llm.py:442-494)：从 queue 拿消息同步处理
- **coroutine C_x** 每个 HTTP 请求一个 handler，挂在 `req_state.queue.get()` 上

**关键发现**：`output_handler` 处理一条 stats-only 消息时：

```python
# async_llm.py:455-471 原代码
if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:  # num_outputs=0 → True
    slices = (outputs.outputs, )                    # 只有 1 片
for i, outputs_slice in enumerate(slices):          # 迭代 1 次
    processed_outputs = output_processor.process_outputs(...)
    if i + 1 < len(slices):                         # 1 < 1 → False
        await asyncio.sleep(0)                      # 永远不触发
```

**循环中唯一可能 yield 的点是顶部的 `await engine_core.get_output_async()`**，而它底层是 `asyncio.Queue.get()`——**非空时同步返回，不 yield event loop**。

所以当 engine 产出 stats 的速率 ≈ output_handler 处理速率（二者都是 Python 同步代码、几十 μs 级），**outputs_queue 长期非空**，`output_handler` 变成"永不让位"的同步循环，`HTTP handler coroutine (C_x)` 被饿死。

### 2.3 HTTP 链路级联 stall

```
engine finish request pr_5612 (产 token, 标 delay_free)
  → pr_5612 的 EngineCoreOutput 进 output_queue，目标 API server
  → API server output_handler 正忙处理 stats，event loop 轮不到 HTTP writer
  → pr_5612 的 HTTP response 卡在 API server 里
  → proxy 等不到 prefill 响应, 不给 decode 发请求
  → decode 闲着，不做 migration，不发 RELEASE
  → prefill 这个 req 的 delay_free block 持续占用
  → KV 使用率降不下来
  → scheduler 继续 "delay_free > 0 → break"
  → 继续空转产 stats
  → cascading stall 持续到其他 delay_free 请求偶然先完成、释放出空间
```

`wait_bridge_pop`（`delay_free_start_ts → bridge_popped_ts`）从 ~1s 涨到 15-30s，**这段时间里 decode 其实根本没收到 HTTP 请求**，不是 decode 懒，是 proxy 等不到 prefill 的 response。

## 3. 改动详情

### 3.1 改动 A：空转不产 stats

**文件**：`vllm/v1/core/sched/scheduler.py`

**定位**：`update_from_output` 末尾（原 line 1419-1425）

**原逻辑**：
```python
if (stats := self.make_stats(spec_decoding_stats)) is not None:
    if (eco := next(iter(engine_core_outputs.values()), None)) is None:
        engine_core_outputs[0] = eco = EngineCoreOutputs()  # ← 空壳
    eco.scheduler_stats = stats
```

注释里自己写着 "We must return the stats even if there are no request outputs this step"——**这句话就是 bug 的源头**。

**新逻辑**：
```python
# 只在有真 activity 时 emit stats
if self.log_stats and engine_core_outputs:
    stats = self.make_stats(spec_decoding_stats)
    if stats is not None:
        eco = next(iter(engine_core_outputs.values()))
        eco.scheduler_stats = stats
```

#### 为什么 `bool(engine_core_outputs)` 是 "real activity" 的正确判据

经过追溯，节流点之前的 `engine_core_outputs` 只有两条路径能变成非空：

**路径 1**（scheduler.py:1401-1404）：遍历 `outputs` dict，而 `outputs` 只在 line 1386 被 append，条件 `new_token_ids or pooler_output is not None or kv_transfer_params`：
  - `new_token_ids`：真的生成 token
  - `pooler_output`：pooling 模型输出（LLM 不走这条）
  - `kv_transfer_params`：`_free_request()` 返回 params，只在请求真的 stopped=finished 时非空

**路径 2**（scheduler.py:1406-1417）：`finished_req_ids_dict` 非空时添加 "只含 finished_requests 的壳"——但这个 dict 只在 multi-client（`include_finished_set=True`）场景下非 None，单 API server 模式下是死代码。

其他可能进 `output_queue` 的消息类型（`utility_output`、`start_wave`、`wave_complete`）都**不经过** `update_from_output`，直接在 core.py 里 `put_nowait`，不会影响这里的判据。

**结论**：在单 API server + LLM 场景下，`bool(engine_core_outputs)` 完全等价于"至少一个请求真的产 token 或真的结束了"。

#### 一个刻意接受的 trade-off

**chunked prefill 中间步也被算作"空转"**：

- 大 prompt（如 88K tokens）会被 `long_prefill_token_threshold=5242` 切成 ~17 块
- 前 16 块 forward 完不产 token（请求还没 prefill 完），`outputs` 不 append
- `engine_core_outputs` 为空 → **不 emit stats**

代价：长 chunked prefill 期间 `monitoring_timestamps.scheduler_stats` 会有几秒到十几秒的真空，看不到 KV 增长过程。

但补偿是：最后一块完成（产 token）那一步会 emit 一次 stats，反映最新状态。而且 stall 本身基本消失（不再有 event loop 拥堵），loggers 的秒级摘要日志仍然可用（就是可能一段时间内重复打印同一个 snapshot 的内容）。

### 3.2 改动 B：彻底移除 BRIDGE_PUSH（Change 2 revert）

#### 为什么要删

在调查根因过程中发现，`docs/design/delay_free_optimization.md` 里的 "Change 2: P→D Bridge 主动推送" **从未真正生效过**：

- `wait_for_save` 里的 `push_bridge_to_decode` 调用前，`self.parse_request_id(request_id, is_prefill=True)` 用的是老 proxy 格式的正则 `___decode_addr_(.*):(\d+)`
- 新 proxy `disagg_proxy_p2p_nccl_xpyd_prod.py` 不往 request_id 里塞 decode 地址，改用 `kv_transfer_params["decode_zmq_address"]`
- `parse_request_id` 对 `chatcmpl-xxx` 格式永远抛 `ValueError`，被外层 `except Exception: pass` 静默吞掉
- **每个请求的 BRIDGE_PUSH 都失败**，decode 每次都走 `pop_bridge_request`（BRIDGE_POP RPC 轮询）fallback 路径

硬证据：`monitoring_timestamps` 里**每一条** delay_free 记录都有 `bridge_popped_ts` 字段——此字段**只在** `pop_bridge_request` (`p2p_nccl_engine.py:319, 334`) 写，`pop_prefetched_bridge` 不写。

既然是一直没生效的死代码，且用户明确想回到纯 decode-pull 模式，直接删。

#### 删除清单

**文件 1**：`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

- `wait_for_save`（原 line 385-404）：删掉 `try/except push` 块，只保留 `stage_bridge_request` 调用
- `start_load_kv`（原 line 176-192）：删掉 `pop_prefetched_bridge` 查询，直接走 `pop_bridge_request(..., timeout_s=0.0)`

**文件 2**：`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

- 删除 `__init__` 里 `self._prefetched_bridges: dict[str, list[int]] = {}` 字段（原 line 214）
- 删除 `pop_prefetched_bridge` 和 `push_bridge_to_decode` 两个方法（原 line 661-686）
- 删除 `listen_for_requests` 里 `elif data["cmd"] == "BRIDGE_PUSH"` 整个分支（原 line 904-914）

#### 行为影响

**零影响**——本来就在跑纯 pull 模式。删除前后对比：

| 指标 | 删除前 | 删除后 |
|---|---|---|
| `wait_bridge_pop` | ~1s | ~1s |
| `schedule_wait` | ~1ms | ~1ms |
| `total_delay_free_duration` | ~1s | ~1s |
| 代码行数 | +45 行死代码 | - |

### 3.3 改动 C：monitoring 字段瘦身

**文件**：`vllm/v1/metrics/monitoring.py`

**删掉的字段**（4 个）：

| 字段 | 删除原因 |
|---|---|
| `bridge_wait_duration` = `popped - staged` | **误导性极强**。`bridge_staged_ts` 在 `wait_for_save` 里设置（CPU 层，forward kernel 刚 launch 还没 GPU sync）；`bridge_popped_ts` 在 decode 端设置（远端 CPU 时刻）。二者差值**包含 GPU forward 剩余计算时间**。eager 模式下大 batch forward 3-7 秒，这个字段会报出 7-10 秒的"bridge 等待"，但其中绝大部分是 GPU 计算。真正的 bridge 等待看 `wait_bridge_pop`（起点换成 `delay_free_start_ts`，它在 `update_from_output` 里设置，已 GPU sync 过）。 |
| `cuda_memcpy_raw` = `complete - launch` | 和 `cuda_memcpy_duration`（`complete - popped`）差一个 `ipc_setup_duration`，完全冗余。`ipc_setup_duration` 已单独记录。 |
| `poll_to_release_duration` = `rel_sent - complete` | 有 Change 1 的后台 poll 线程之后，这个时间永远在 μs 级，没信息量。 |
| `release_rpc_duration` = `rel_recv - rel_sent` | 和 `release_rpc_total`（`rel_recv - complete`）差一个 `poll_to_release_duration`（≈0），冗余。`release_rpc_total` 就够用。 |

**保留的字段**：

| 字段 | 定义 |
|---|---|
| `total_delay_free_duration` | `delay_free_end_ts - delay_free_start_ts` |
| `wait_migration` | `release_received_ts - delay_free_start_ts`（seg1） |
| `wait_bridge_pop` | `bridge_popped_ts - delay_free_start_ts` |
| `cuda_memcpy_duration` | `migration_complete_ts - bridge_popped_ts` |
| `release_rpc_total` | `release_received_ts - migration_complete_ts` |
| `schedule_wait` | `finished_sending_ts - release_received_ts`（seg2） |
| `deferred_cleanup_lag` | `deferred_cleanup_ts - block_freed_ts` |
| `ipc_setup_duration` | `migration_launch_ts - bridge_popped_ts`（保留作 kernel-launch debug 用） |

注释也同步更新：去掉 "BRIDGE_PUSH or BRIDGE_POP"、明确标注"pure decode-pull"。同时加了一段明确的提示说明为什么 `staged → popped` 不能用作 delay-free 指标——避免后续维护者重复踩这个坑。

## 4. 预期效果

### 4.1 文件体积

`monitoring_timestamps` 单次实验体积：
- `scheduler_stats` 数组：14.7M 条 → 几千条（与请求数同量级）
- 每条 `request_stats` 瘦身 4 个浮点字段
- **4 GB → 10-50 MB 量级**

### 4.2 延迟

- **stall 期间的 HTTP 响应延迟**（pr_5612 类）：15-300s → 预期 <100ms（event loop 不再被 stats 霸占）
- **正常态 delay-free 耗时**：无变化（~1s）
- **engine core CPU 占用**：空转期间显著下降（不再每 ms 构造 SchedulerStats + msgpack + ZMQ send）

### 4.3 可观测性代价

- `loggers.py:123` 的 "Avg prompt throughput, KV cache usage" 秒级摘要日志：**stall 期间会重复显示 stall 进入前最后一次 snapshot 的值**（因为 `last_scheduler_stats` 不更新）
- 长 chunked prefill 期间 monitoring 会有真空段（chunk 中间步不 emit）

接受这个 trade-off 的理由：
- stall 期间 scheduler state 几乎不变，重复 snapshot 本来就没有新信息
- delay-free block 机制本身工作正常（由改动 1/3/4/5 保证），stall 会自然消退
- 真要调试 stall，用 `monitoring_timestamps.request_stats` 的逐请求分析比 stats snapshot 更精确

## 5. 验证方法

### 5.1 单变量对照实验

先什么都不改，跑一次 baseline，把 `monitoring_timestamps` 备份。然后上这三个改动，再跑一次同样负载。对比：

```bash
# 1. 文件体积
ls -lh continuum_exp/prefill_20003/monitoring_timestamps

# 2. scheduler_stats 条数
python3 -c "
import json
d = json.load(open('continuum_exp/prefill_20003/monitoring_timestamps'))
ss = d.get('scheduler_stats', [])
print(f'stats count: {len(ss)}')
if len(ss) >= 2:
    intervals = [ss[i+1]['timestamp'] - ss[i]['timestamp']
                 for i in range(min(100, len(ss)-1))]
    print(f'avg interval: {sum(intervals)/len(intervals)*1000:.2f} ms')
"

# 3. wait_bridge_pop 的 P50 / P99
python3 -c "
import json
d = json.load(open('continuum_exp/prefill_20003/monitoring_timestamps'))
rs = d.get('request_stats', {})
vals = sorted([r.get('wait_bridge_pop', 0)
               for r in rs.values()
               if r.get('wait_bridge_pop') is not None])
n = len(vals)
print(f'wait_bridge_pop: P50={vals[n//2]:.3f}s, P99={vals[int(n*0.99)]:.3f}s, max={vals[-1]:.3f}s')
"
```

**预期改动后**：
- 文件体积下降 ~100 倍
- stats 间隔从 μs 级变成几十 ms 到几秒不等
- `wait_bridge_pop` P99 从 15-30s 下降到 1-2s

### 5.2 逐请求轨迹

对比 baseline 和 fix 后同一个 `chatcmpl-*` 请求（用 `job_id + agent_round` 定位）的 `total_delay_free_duration`、`wait_bridge_pop` 是否一致或更快。

## 6. 回滚指引

### 6.1 完全回到改动前

```bash
git revert <commit_hash>
```

### 6.2 仅恢复某一部分

**恢复 stats 节流**（想临时看完整 stats）：
- 编辑 `vllm/v1/core/sched/scheduler.py` 的 `update_from_output` 末尾
- 将 `if self.log_stats and engine_core_outputs:` 改回 `if (stats := self.make_stats(...)) is not None:`
- 并加回创建空 EngineCoreOutputs 的逻辑

**恢复 BRIDGE_PUSH**：
- 需要修好 `parse_request_id` fallback，从 `request.kv_transfer_params["decode_zmq_address"]` 取地址
- 参考本仓库 commit 4717a39 的原始实现
- 注意：即便修好 BRIDGE_PUSH，也只能把 `wait_bridge_pop` 从 ~1s 压到 ~0.1s，对 stall 场景没有额外帮助

**恢复 monitoring 字段**：
- 照搬删除前的 `monitoring.py:153-163` legacy 段

## 7. 相关提交

| 提交 | 内容 |
|---|---|
| 4717a39 | 原 delay_free 优化 bundle（引入 Change 1-5，其中 Change 2 未真正生效） |
| 0278275 | 存在卡死 bug（stats flood 问题未解决的快照） |
| (本次) | 空转 stats 节流 + BRIDGE_PUSH 删除 + monitoring 瘦身 |

## 8. 待办

- [ ] `docs/design/delay_free_optimization.md` 在 Change 2 部分标注"已 revert"
- [ ] 跑一次完整对照实验，验证预期效果（见 §5.1）
- [ ] 观察 `loggers.py:123` 日志在 stall 期间的表现，如果 stale snapshot 确实造成困惑，可考虑改 `LoggingStatLogger.log()` 在 snapshot 超过 N 秒未更新时显示 "(stale)" 标记
