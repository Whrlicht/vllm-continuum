# Disagg Block Migration 落地说明（DistServe 语义对齐）

## 1. 目标与背景

本次改造目标是把 Disaggregated Serving 的 KV 迁移流程从“按 layer/tensor 的中转传输”升级为 DistServe 风格的“按 block_id 直接迁移”，核心语义如下：

1. Prefill 侧不再做 sendstore / kv buffer / mem pool staging。
2. Decode 侧主动拉取 bridge 元数据（request_id + context_block_ids）。
3. Decode 侧基于本地已分配的 decode block_ids，直接从 Prefill block pool 迁移。
4. 迁移使用异步 CUDA copy（D2D），迁移完成后再回调 Prefill 释放。
5. Prefill 仅在收到 release callback 后延迟释放块，保证生命周期顺序正确。

---

## 2. 改造前后的核心差异

### 2.1 改造前（旧版）

主要路径是 PUT / PUT_ASYNC / GET：

1. 以 tensor_id（request_id#layer_name）为传输单位。
2. Prefill 侧通过 send_store / send_queue 暂存数据，Decode 侧按 layer 拉取。
3. 可选依赖 kv_buffer_size、mem_pool_size_gb 等中转参数。
4. 地址路由历史上可能通过 request_id 编码解析。

这条路径仍被保留为兼容回退路径（非 BLOCK_MIGRATE 模式）。

### 2.2 改造后（BLOCK_MIGRATE）

主路径变为 block 级直迁：

1. Prefill forward 完成后只发布 bridge 元数据（request_id -> context_block_ids）。
2. Decode 侧 pop bridge 元数据后，触发 block 级迁移。
3. 通过 CUDA IPC 获取远端 KV cache 指针视图，按 block_id 对齐迁移。
4. 使用 cudaMemcpy2DAsync 在当前 stream 提交异步 D2D 拷贝。
5. Decode 事件完成后发送 RELEASE 回调；Prefill 收到回调才 finished_sending。

---

## 3. 代码落地清单（按层）

## 3.1 数据面：C++/CUDA 自定义算子

1. 新增“从远端指针创建本地 CUDA 视图”能力。
2. 新增“按 block_id 批量迁移 KV block”能力。

关键位置：

- csrc/cuda_view.cu
  - get_cuda_view_from_ptr_like
  - migrate_kv_cache_blocks
  - 内部使用 cudaMemcpy2DAsync 实现 block 迁移
- csrc/ops.h
  - 新增 get_cuda_view_from_ptr_like 声明
- csrc/cache.h
  - 新增 migrate_kv_cache_blocks 声明
- csrc/torch_bindings.cpp
  - 注册 _C::get_cuda_view_from_ptr_like
  - 注册 _C_cache_ops::migrate_kv_cache_blocks
- vllm/_custom_ops.py
  - 增加 Python wrapper：get_cuda_view_from_ptr_like / migrate_kv_cache_blocks

## 3.2 IPC 封装层

- vllm/distributed/device_communicators/cuda_wrapper.py
  - 补全 cudaIpcCloseMemHandle
  - 与已有 cudaIpcGetMemHandle / cudaIpcOpenMemHandle 形成完整生命周期

## 3.3 P2P 引擎层（控制面 + 执行面）

- vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py

新增/重构要点：

1. direct block mode 开关（BLOCK_MIGRATE/BLOCK_DIRECT/DISTSERVE）。
2. register_kv_caches：引擎持有本地 KV cache 映射。
3. stage_bridge_request / BRIDGE_POP：桥接队列发布与消费。
4. GET_IPC_METADATA：按 layer 返回 IPC handle。
5. _ensure_remote_kv_views：打开 IPC handle 并创建远端 tensor view。
6. launch_block_migration：按 src/dst block_ids 触发迁移。
7. _poll_completed_migrations：轮询 event 完成。
8. RELEASE 回调：Decode 完成后通知 Prefill。
9. _get_finished_block_mode：仅在正确阶段返回 finished_sending/finished_recving。
10. close：补全 IPC handle 关闭。

## 3.4 Connector 层（调度/执行对接）

- vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py

改动要点：

1. ReqMeta 扩展 remote_prefill_address / remote_decode_address。
2. start_load_kv：
   - Producer：仅缓存待发布 bridge 请求，wait_for_save 时发布。
   - Consumer：pop bridge + launch block migration。
3. save_kv_layer：direct mode 下 no-op（不做 layer staging）。
4. get_num_new_matched_tokens：direct mode consumer 返回 async load。
5. build_connector_meta：支持从 kv_transfer_params 显式取路由地址。
6. request_finished：producer 在 direct mode 下 delayed free。

## 3.5 Proxy 路由层

- examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd_prod.py

改动要点：

1. request_id 改为纯 uuid，不再承载路由编码。
2. 显式下发 kv_transfer_params：
   - kv_transfer_mode=BLOCK_MIGRATE
   - prefill_zmq_address
   - decode_zmq_address

## 3.6 启动脚本层

- examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/run_disagg_p2p_nccl_xpyd_prod.sh

改动要点：

1. 默认 KV_SEND_TYPE=BLOCK_MIGRATE。
2. 移除 mem_pool / kv_buffer 的用户配置入口（避免误导为 staging 方案）。
3. 保留 request_completion_timeout / get_retry_* 作为 bridge 与控制面容错参数。

---

## 4. 关键流程（请求生命周期）

1. Proxy 把 prefill_zmq_address / decode_zmq_address 写入 kv_transfer_params。
2. Prefill 执行完 prefill 后发布 bridge 元数据（request_id -> context_block_ids）。
3. Decode 拿到调度分配后的 decode block_ids，向 Prefill BRIDGE_POP 获取 context_block_ids。
4. Decode 通过 GET_IPC_METADATA 建立远端 KV 视图并触发 block 迁移。
5. 迁移在 CUDA stream 异步提交，event 完成后 Decode 发送 RELEASE。
6. Prefill 收到 RELEASE 后 finished_sending，scheduler 才最终 free 对应 block。

---

## 5. 与旧版本的差异总结（行为层）

1. 传输粒度：
   - 旧：layer/tensor 级（request_id#layer_name）。
   - 新：block_id 级（context -> decode 一一映射，按 min(len(src), len(dst)) 迁移）。

2. 控制关系：
   - 旧：producer 发送主导，consumer 被动接收。
   - 新：consumer 迁移主导，producer 仅发布元数据并等待 release。

3. 生命周期管理：
   - 旧：依赖 send queue 空/GET 完成等时机。
   - 新：严格由 decode 迁移完成回调驱动 producer 释放。

4. 配置模型：
   - 旧：存在 mem_pool_size_gb / kv_buffer_size 等 staging 参数语义。
   - 新：主路径不依赖 staging 参数，核心是 bridge + IPC + block migration。

5. 路由携带：
   - 旧：可通过 request_id 编码路由。
   - 新：统一通过 kv_transfer_params 显式传路由。

---

## 6. 本轮补充修复（避免角色误判）

在 direct block 模式下，完成态处理新增了角色约束：

1. 只有 Producer 侧等待 RELEASE 回调并上报 finished_sending。
2. Consumer 侧不进入 release-timeout -> finished_sending 分支。

目的：避免 Consumer 误报 finished_sending，导致上层 scheduler 对已清理请求重复 free。

---

## 7. 日志中的 Mismatched block_ids 提示说明

当前实现对 src/dst block 列表长度不一致采用“取最小长度”迁移，并打印告警。

这类日志形态通常是：

- src: N, dst: N-1, copy: N-1

其语义是保护性降级而非崩溃路径，主要用于揭示两侧 block 视图在某些边界轮次的对齐差异。若需要进一步收敛该告警频率，可继续在 bridge 元数据产出阶段增加更严格的一致性策略。

---

## 8. 验收建议

建议按以下顺序验收：

1. 编译验收：确认新增 C++/CUDA op 可成功编译并被 Python 层加载。
2. 功能验收：观察完整链路日志是否出现
   - bridge publish/pop
   - launch block migration
   - migration event complete
   - RELEASE ack
   - producer delayed free
3. 稳定性验收：在 fixed 并发与 trace replay 下跑中长时任务，检查是否出现超时释放、重复释放、IPC 句柄泄漏。

---

## 9. 当前状态

代码层已完成 direct block migration 主路径落地，并补齐了关键角色一致性修复。静态检查已通过。

待继续完成：

1. 全量 C++ 扩展编译与运行时加载验证。
2. 端到端长压回放下的性能与稳定性量化报告。
