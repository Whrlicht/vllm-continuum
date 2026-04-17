# Trace Replay 功能设计与实现报告

## 1. 目标

实现一个可开关的 trace_replay 接口：

- 开启时：模型仍然正常 forward 计算 logits，但在“输出 token 拼接”阶段强制使用 trace 数据集中的 token 序列，保证输出严格按轨迹复现。
- 关闭时：走 vLLM 原始采样逻辑，行为与默认一致。

## 2. 设计原则

1. 不破坏默认路径：trace_replay 默认关闭，不影响线上常规请求。
2. 最小侵入：只在“token 产生后、写入状态前”做覆盖。
3. 双层兜底：在 worker 内部状态和 scheduler 输出路径都执行强制覆盖，防止状态分叉。
4. 显式接口：将 trace_replay/traj_id 作为 OpenAI 请求模型字段，而不是隐式 getattr。

## 3. 修改清单

### 3.1 第一步（已完成，4 个文件）

1. vllm/trace_replay/store.py
- 从占位实现改为可加载 trace 文件。
- 支持读取环境变量 VLLM_TRACE_REPLAY_PATH，默认回退 trace_data/swe_bench_sample_100_with_timings.json。
- 支持按 traj_id/instance_id 建索引。
- 支持从常见字段抽取 token ids（trace_token_ids / forced_token_ids / output_token_ids / token_ids / tokens）。

2. vllm/v1/request.py
- 引入 trace replay 请求状态：trace_replay_enabled、traj_id、trace_pos、trace_finished、trace_token_ids。
- 新增 pop_next_trace_token()，用于按位置读取强制 token。
- 在 from_engine_core_request 中支持 extra_args.trace_replay_token_ids 透传（可选）。

3. vllm/v1/core/sched/scheduler.py
- 在 _update_request_with_output() 中将“新 token 写回”改为：若开启 trace_replay，则使用 request.pop_next_trace_token() 覆盖模型采样 token。
- trace token 用尽时，设置 FINISHED_STOPPED 和 stop_reason=trace_replay_end。

4. vllm/v1/worker/gpu_model_runner.py
- 在 _bookkeeping_sync() 中，CPU 同步前先调用 _apply_trace_replay_to_sampled_ids()。
- 按请求当前 output_token_ids 长度定位轨迹位置，覆盖 sampled_token_ids，保证下一步 forward 的上下文也严格跟随 trace。

### 3.2 第二步（本次新增）

5. vllm/entrypoints/openai/protocol.py
- 将 trace_replay/traj_id 正式加入以下请求模型：
  - ResponsesRequest
  - ChatCompletionRequest
  - CompletionRequest
- 在 to_sampling_params() 中统一透传到 SamplingParams：
  - sampling_params.trace_replay = bool(self.trace_replay)
  - sampling_params.traj_id = self.traj_id
- 新增参数校验：
  - trace_replay=true 必须提供 traj_id
  - trace_replay 仅支持 n=1
  - trace_replay 不支持 beam search
- 移除调试输出 print，避免污染服务日志。

6. vllm/entrypoints/openai/serving_chat.py
- 删除原来基于 getattr 的临时 trace_replay 注入逻辑。
- 改为依赖 protocol.py 的正式字段和 to_sampling_params 透传。

7. vllm/entrypoints/openai/serving_{chat,completion,responses}.py
- 当 trace_replay=true 且 trace 文件里没有 token ids 时，使用当前服务模型的 tokenizer 对文本轨迹做实时 tokenization。
- 将得到的 token ids 通过 sampling_params.extra_args.trace_replay_token_ids 下发到引擎侧，避免引擎层重复依赖 tokenizer。
- 新增多轮前缀对齐：若 full_trace_token_ids 以当前请求 prompt_token_ids 为前缀，则仅回放“前缀之后”的 token；否则回退到 full trace 并打印告警。

8. vllm/trace_replay/store.py（增强）
- 新增 materialize_trace_token_ids(traj_id, tokenizer) 能力：
  - 优先读取已存在 token ids；
  - 否则从 messages 中按顺序提取多轮对话文本（包含 user/assistant/tool 等角色及其 content/thought/action/tool_calls/tool_call_ids 等字段）并使用 tokenizer 编码；
  - 结果缓存到 by_traj_id，后续复用。

## 4. 接口定义

### Chat Completions
- 请求字段：
  - trace_replay: bool（默认 false）
  - traj_id: str（trace_replay=true 时必填）

### Completions
- 请求字段：
  - trace_replay: bool（默认 false）
  - traj_id: str（trace_replay=true 时必填）

### Responses
- 请求字段：
  - trace_replay: bool（默认 false）
  - traj_id: str（trace_replay=true 时必填）

## 5. 数据流向（端到端）

1. API 入参
- 客户端在 OpenAI 兼容接口传入 trace_replay/traj_id。
- protocol.py 在请求模型中完成解析与校验。

2. 采样参数构建
- to_sampling_params() 把 trace_replay/traj_id 写入 SamplingParams。

2.5. 文本轨迹转 token（仅当缺 token ids）
- serving 层调用 TraceStore.materialize_trace_token_ids(traj_id, tokenizer)。
- 若 trace 本身无 token ids，则从 messages 中按消息顺序提取“全角色多轮内容”并用当前模型 tokenizer 编码。
- 编码结果写入 sampling_params.extra_args.trace_replay_token_ids。

2.6. 多轮 prompt 前缀对齐
- serving 层拿到 full_trace_token_ids 后，与当前请求的 prompt_token_ids 做前缀匹配。
- 匹配成功：仅将 full_trace_token_ids[prompt_len:] 作为本轮强制输出 token。
- 匹配失败：回退到 full_trace_token_ids（并记录 warning，便于排查模板/tokenizer 差异）。

3. 请求进入 V1 引擎
- v1/request.py 读取 SamplingParams：
  - 若 extra_args 里有 trace_replay_token_ids，直接使用（来自 serving 层 tokenizer 转换结果）；
  - 否则按 traj_id 从 TraceStore 取 token 序列。
  - 初始化 trace_pos=0。

4. 模型执行与采样
- 模型仍正常 forward 计算 logits 与采样。
- gpu_model_runner 在 sampled_token_ids 产出后、写入内部 batch 状态前，按 trace 覆盖 sampled_token_ids。

5. 调度器写回输出
- scheduler._update_request_with_output() 再次按 trace 覆盖写回 token，确保对外输出与内部状态一致。

6. 输出返回
- 返回给客户端的 new_token_ids 严格来自 trace 序列。

## 6. 开关行为

- trace_replay=false：
  - 不做任何覆盖，完整沿用原始采样流程。
- trace_replay=true：
  - 强制覆盖输出 token；若轨迹耗尽则请求结束。

## 7. 使用示例

### Chat Completions

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [{"role": "user", "content": "hello"}],
  "trace_replay": true,
  "traj_id": "django-money__django-money.835c1ab8.func_pm_ctrl_shuffle__viqnyl9u.8qa84d2e",
  "n": 1
}
```

### 关闭 trace replay（默认路径）

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [{"role": "user", "content": "hello"}],
  "trace_replay": false
}
```

## 8. 已知约束与注意事项

1. 严格复现依赖 token ids
- trace 数据必须包含每条轨迹的 token id 序列。
- 若仅有文本消息而无 token ids，无法做到严格 token-level 复现。

1.1 文本轨迹自动转换（新增）
- 现在支持当 trace 仅有文本时，使用当前服务模型 tokenizer 在线转换为 token ids。
- 注意：这会受到 tokenizer 版本、聊天模板、以及文本抽取序列化策略影响；若与线上 prompt 构造不一致，可能出现前缀不匹配并触发回退。

2. 当前不支持 beam search
- trace_replay 与 beam_search 语义冲突，已显式拒绝。

3. 当前要求 n=1
- 多样本并行采样（n>1）未定义轨迹对齐规则，已显式拒绝。

## 9. 文件索引

- vllm/trace_replay/store.py
- vllm/v1/request.py
- vllm/v1/core/sched/scheduler.py
- vllm/v1/worker/gpu_model_runner.py
- vllm/entrypoints/openai/protocol.py
- vllm/entrypoints/openai/serving_chat.py

## 10. 后续可扩展方向

1. 支持多候选轨迹（n>1）
- 为每个候选定义独立轨迹或共享策略。

2. 支持文本轨迹自动重编码
- 若只有文本轨迹，可在服务端按 tokenizer 预编码为 token ids 后复用。

3. 增加可观测性
- 打点统计覆盖率（forced token count / total token count）和轨迹命中失败率。
