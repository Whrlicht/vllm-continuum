# LICHT 调度设计与实现说明

## 1. 背景与目标

本次改造目标是在不破坏现有调度逻辑的前提下，引入一个可开关的 LICHT 模式。

核心要求：

1. 默认不带开关时，系统行为保持原样。
2. 开启 LICHT 后：
   - prefill 实例采用动态优先级调度。
   - decode 实例保持 FCFS。
3. KV cache 传输策略阶段性保持默认实现，不做算法层改动。
4. 动态优先级分数使用：

$$
score = a \cdot \log(1 + k_i) + b \cdot \min\left(\frac{t_{wait}}{T_{max}}, 1\right)
$$

其中：

- $a = 3$
- $b = 1$
- $T_{max} = 2s$

## 2. 设计原则

### 2.1 默认路径零侵入

LICHT 必须是一个显式开关，不允许“默认改变调度语义”。

### 2.2 最小侵入

尽量复用现有的请求队列、调度主循环、CLI 配置与 OpenAI extra_args 透传能力。

### 2.3 可解释与可回滚

实现上尽量保证每个行为都能通过日志和配置定位，出现问题时只需关闭 `--licht` 即可回退。

### 2.4 语义正确优先于技巧优化

先保证 $k_i$、$t_{wait}$、抢占公平性语义正确，再考虑复杂度优化（例如从 O(N) 到 O(logN)）。

## 3. 整体架构改造点

本次修改覆盖 5 条链路。

1. 配置层：`SchedulerConfig` 增加 `licht` 字段。
2. CLI 层：`EngineArgs` 增加 `--licht` 参数并透传到 `SchedulerConfig`。
3. 请求元数据层：`Request` 增加 `agent_round`，从 `extra_args` 解析真实轮次。
4. 调度层：V1 scheduler 增加 LICHT 分流逻辑（prefill 动态优先级、decode FCFS）。
5. 启动脚本层：生产脚本新增 `--licht`，可同时透传到 prefill/decode worker。

## 4. 配置与开关接入

### 4.1 调度配置

在 `vllm/config/scheduler.py` 中新增：

- `licht: bool = False`

并明确语义：

- prefill 用动态优先级
- decode 保持 FCFS
- KV transfer 先沿用默认实现

### 4.2 Engine 参数与 CLI

在 `vllm/engine/arg_utils.py` 中：

1. `EngineArgs` 新增 `licht` 字段。
2. 注册命令行参数 `--licht`。
3. 在创建 `SchedulerConfig` 时传递 `licht=self.licht`。

这样，调度开关可由单一 CLI 参数控制。

### 4.3 生产脚本

在 `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/run_disagg_p2p_nccl_xpyd_prod.sh` 中：

1. 新增 `LICHT=false` 默认值。
2. 新增 `--licht` 参数解析。
3. 配置输出中打印 `LICHT=...`。
4. 当 `LICHT=true` 时，给 prefill 和 decode 的 `vllm serve` 都追加 `--licht`。

注意：虽然 prefill/decode 都收到了开关，但具体行为由 scheduler 内部通过实例角色自动分流。

## 5. 请求轮次数据路径（k_i 的来源）

## 5.1 设计问题

最初实现里，$k_i$ 使用了 scheduler 的轮询 tick 计数。这在高并发下会快速膨胀，语义错误。

$k_i$ 的正确语义应是“请求所属 agent 对话真实轮次”，而不是“调度循环次数”。

## 5.2 透传机制

OpenAI 协议层已支持 `vllm_xargs`，并在 `to_sampling_params` 中写入 `extra_args`。

请求可通过 `vllm_xargs` 传：

- `agent_round`
- `assistant_round`
- `trace_replay_assistant_round`

## 5.3 Request 侧解析

在 `vllm/v1/request.py` 中新增：

1. `_parse_non_negative_int`：安全解析非负整数。
2. `_extract_agent_round`：按兼容优先级读取轮次键。
3. `self.agent_round` 字段：默认 0，解析成功则保存解析值。

这样，scheduler 可直接读取 request 级真实轮次，不依赖外部临时状态。

## 6. 调度算法实现细节

实现文件：`vllm/v1/core/sched/scheduler.py`

### 6.1 角色识别与开关分流

基于 `kv_transfer_config.kv_role` 推导实例角色：

- prefill（kv_producer）
- decode（kv_consumer）
- 其他/单机

内部状态：

- `licht_enabled`
- `licht_prefill_sched_enabled = licht_enabled and role != decode`
- `licht_decode_fcfs_enabled = licht_enabled and role == decode`

语义：

- prefill 走 LICHT 动态优先级
- decode 显式固定 FCFS

### 6.2 动态分数计算

固定参数：

- `LICHT_PREFILL_SCORE_A = 3.0`
- `LICHT_PREFILL_SCORE_B = 1.0`
- `LICHT_PREFILL_SCORE_TMAX_S = 2.0`

分数计算函数核心逻辑：

1. `k_i = max(request.agent_round, 0)`
2. `t_wait = now - waiting_start_ts`
3. `wait_term = min(t_wait / Tmax, 1)`
4. 返回 `a*log1p(k_i) + b*wait_term`

### 6.3 等待时间状态管理

新增等待时间状态字典：

- `licht_waiting_round_start_ts: dict[request_id, monotonic_time]`

辅助函数：

1. `_reset_licht_waiting_state(request)`
   - 设置/重置请求等待起点时间。
2. `_drop_licht_waiting_state(request_id)`
   - 请求出队或结束时删除状态。
3. `_ensure_licht_waiting_start_timestamps()`
   - 每轮调度仅补齐缺失时间戳，不再修改轮次。

关键点：scheduler 不再“自增轮次”，只负责 `t_wait` 的计时。

### 6.4 waiting 队列选择与出队

统一入口：

- `_peek_waiting_request()`
- `_pop_waiting_request(request)`

行为：

1. prefill + LICHT：遍历 waiting，按最大 score 选取。
2. decode + LICHT：按 `(arrival_time, request_id)` 做 FCFS。
3. 非 LICHT：回退到原 policy（fcfs/priority/continuum）。

为什么要拆出 `_pop_waiting_request(request)`：

- LICHT 选中项不一定在队头，必须支持“按对象移除”。

## 7. 抢占、公平性与生命周期处理

### 7.1 抢占后公平性

当 RUNNING 请求被抢占并回到 waiting：

- 保留 `request.agent_round` 不变。
- 仅重置等待起始时间戳。

这保证了“对话轮次公平性”不会因为抢占被清零。

### 7.2 请求状态切换时机

1. `add_request` 时初始化等待起始时间。
2. waiting -> running 成功后删除等待状态。
3. 请求结束 `_free_request` 时删除等待状态。

这样可避免状态泄露和时间戳复用错误。

## 8. 与默认策略和 KV 传输的关系

### 8.1 默认调度策略保持不变

当不带 `--licht` 时：

- `fcfs/priority/continuum` 原行为保持不变。
- LICHT 代码路径不生效。

### 8.2 KV cache 传输策略保持默认

本阶段 LICHT 只改变“waiting 请求选择逻辑”。

- 不改变 connector 协议
- 不改变 block 迁移策略
- 不改 producer/consumer 数据通路

## 9. 复杂度与性能权衡

当前 LICHT prefill 选择为：

- 选择：遍历 waiting 求最大 score，复杂度 O(N)
- 移除：队列按对象移除，deque 场景通常 O(N)

原因：

1. 先优先保证语义正确与最小侵入。
2. 对中小 waiting 队列，性能通常可接受。
3. 未来可升级到堆结构（lazy deletion）降低热点开销。

## 10. 参数调优位置

目前参数硬编码在 scheduler 常量：

- `LICHT_PREFILL_SCORE_A`
- `LICHT_PREFILL_SCORE_B`
- `LICHT_PREFILL_SCORE_TMAX_S`

若要调优：

1. 直接修改常量值。
2. 重启服务即可生效。

未来可以扩展为 CLI/配置文件暴露，但本次按需求保持内部参数。

## 11. 兼容性与边界条件

1. `agent_round` 解析失败时默认 0，不影响请求可运行性。
2. 支持 int/float(整数值)/数字字符串的轮次输入。
3. 对 bool 显式拒绝，避免 `True/False` 被误解析成 1/0。
4. waiting 时间使用 `time.monotonic()`，避免系统时钟跳变影响。
5. 通过 `request_id` 作为状态键，保证每请求隔离。

## 12. 验证与结果

本次实现后已完成针对关键文件的静态错误检查，相关文件无语法与类型报错。

重点验证点：

1. 不带 `--licht`：行为与改造前一致。
2. 带 `--licht` + prefill：按动态分数调度。
3. 带 `--licht` + decode：保持 FCFS。
4. 抢占后请求：轮次不丢失，仅等待时间重置。

## 13. 典型请求示例（轮次透传）

客户端可在请求体中通过 `vllm_xargs` 传递轮次，例如：

```json
{
  "model": "your-model",
  "messages": [{"role": "user", "content": "..."}],
  "vllm_xargs": {
    "agent_round": 3,
    "job_id": "your_job"
  }
}
```

或复用 trace replay 客户端常见字段：

```json
{
  "vllm_xargs": {
    "trace_replay_assistant_round": 3
  }
}
```

## 14. 后续优化建议

1. 将 LICHT waiting 选择从 O(N) 优化到 O(logN)。
2. 增加调度观测指标：score 分布、等待项数量、round 维度延迟。
3. 将 a/b/Tmax 暴露为受控配置，支持实验自动调参。
4. 与 KV 迁移策略联动，形成端到端 disagg 优先级闭环。

---

本文档对应当前仓库实现状态，可直接作为 LICHT 功能的设计与实现记录。