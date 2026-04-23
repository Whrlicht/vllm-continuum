# Delay-Free KV Cache Release 优化设计文档

## 1. 背景：Disaggregated Serving 中的 Delay-Free 机制

在 Prefill/Decode 分离架构中：
- **Prefill GPU** 计算 KV cache，生成完后标记请求为 finished
- **Decode GPU** 通过 CUDA IPC (`cudaMemcpy2DAsync`) 迁移 KV cache blocks
- Prefill 端不能立即释放 blocks，因为 Decode 端可能还在读取
- 所以 Prefill 端将这些 blocks 置为 **delay-free** 状态，等待 Decode 通过 ZMQ `RELEASE` RPC 确认迁移完成后，才真正释放

### 1.1 原始流程（无优化）

```
P-worker: prefill forward 完成
    ↓
P-worker: wait_for_save() → stage_bridge_request()  [bridge_staged_ts]
    ↓
P-scheduler: update_from_output() → _free_request()
    ↓ 发现 delay_free=True → 标记 delay-free，不释放blocks [delay_free_start_ts]
    ↓
P-scheduler: schedule() → build_connector_meta() → ...
    ↓
P-worker: execute_model() (下一个 forward step)
    ↓
    ↓  ← 与此同时，D 端开始迁移 ↓
    ↓
    ↓  D-worker: start_load_kv()
    ↓      ↓
    ↓  D-worker: pop_bridge_request() → ZMQ BRIDGE_POP RPC 到 P 端  [bridge_popped_ts]
    ↓      ↓
    ↓  D-worker: launch_block_migration() → cudaMemcpy2DAsync          [migration_launch_ts]
    ↓      ↓
    ↓  D-worker: get_finished() 轮询 cuda event (每个 forward step 末尾)
    ↓      ↓ event.query() == True                                     [migration_complete_ts]
    ↓      ↓
    ↓  D-worker: _send_release_callback() → ZMQ RELEASE RPC            [release_callback_sent_ts]
    ↓      ↓
    ↓  P-listener: 收到 RELEASE → completed_release_req_ids.add()      [release_received_ts]
    ↓
P-worker: get_finished() (下一个 forward step 末尾)
    ↓ 检查 completed_release_req_ids → finished_sending                [finished_sending_ts]
    ↓
P-scheduler: update_from_output() → _update_from_kv_xfer_finished()
    ↓ → _free_blocks()                                                 [delay_free_end_ts]
```

### 1.2 原始流程中的瓶颈

对 584 个请求的生产数据分析：

```
bridge_wait_duration   (staged → popped):        26.3%  等 D 端来取 bridge 数据
ipc_setup_duration     (popped → launch):          0.0%  IPC handle 解析
cuda_memcpy_duration   (launch → complete):        7.1%  实际 GPU 拷贝 ← 唯一有用的工作
poll_to_release        (complete → sent):          0.0%  等 D 端 forward step 结束才发 RELEASE
release_rpc_duration   (sent → received):          0.0%  ZMQ 网络传输
engine_poll_lag        (received → finished):     31.3%  等 P 端下一个 forward step 的 get_finished()
scheduler_free_lag     (finished → freed):        35.5%  等 P-scheduler 的 update_from_output()
```

**93% 是调度和通信开销，只有 7% 是 GPU 工作。**

核心瓶颈：
1. **D 端 cuda event 轮询被 forward step 卡住**：`_poll_completed_migrations()` 只在 `get_finished()` 中调用，而 `get_finished()` 在每次 forward step 末尾才执行
2. **D 端取 bridge 数据需要同步 RPC**：Decode 发 `BRIDGE_POP` 到 Prefill，等待回复
3. **P 端 RELEASE 信号被 forward step 卡住**：listener 线程收到 RELEASE 放入 `completed_release_req_ids`，但要等 worker 下一次 `get_finished()` 才能传递给 scheduler
4. **`_update_from_kv_xfer_finished()` 在 `update_from_output()` 末尾调用**：释放 blocks 的时机太晚

---

## 2. 优化方案概览

```
改动1: D 端后台 poll 线程        → 消除 poll_to_release (D端forward卡住)
改动2: P→D bridge 主动推送       → 消除 bridge_wait_duration
改动3: P 端 fast-release 侧通道  → 消除 engine_poll_lag (P端forward卡住)
改动4: 重排 update_from_output   → 减少 scheduler_free_lag
改动5: 后台 block-free 线程      → 消除 schedule_wait (engine core 主循环间隔)
```

---

## 3. 改动详情

### 3.1 改动1: D 端后台 Migration Poll 线程

**问题**：Decode 端的 `_poll_completed_migrations()` 只在 `get_finished()` 中被调用。`get_finished()` 在每个 forward step 末尾执行。如果 D 端正在跑一个长 forward step（2秒+），cuda event 早就 ready 了，但 RELEASE RPC 要等 forward 结束才发出。

**方案**：专用后台线程 `_migration_poll_loop()` 以 1ms 间隔持续轮询 cuda event。一旦检测到完成，立即通过自己的 ZMQ DEALER socket 发送 RELEASE RPC。

**修改文件**：`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py`

**新增方法**：
```python
def _migration_poll_loop(self) -> None:
    """后台线程：轮询 cuda event，立即发 RELEASE。"""
    poll_socks: dict[str, zmq.Socket] = {}   # 线程私有 sockets
    poll_ctx = zmq.Context()                  # 线程私有 context

    while not self._migration_poll_stop.is_set():
        done = []
        with self.state_lock:
            for req_id, (event, addr) in list(self.pending_migrations.items()):
                if event.query():
                    done.append((req_id, addr))
        if not done:
            self._migration_poll_stop.wait(0.001)
            continue
        migration_complete_ts = time.time()
        for req_id, addr in done:
            decode_ts = self._delay_free_ts.pop(req_id, {})
            decode_ts["migration_complete_ts"] = migration_complete_ts
            decode_ts["release_callback_sent_ts"] = time.time()
            self._poll_thread_send_release(poll_ctx, poll_socks, req_id, addr, decode_ts)
        with self.state_lock:
            for req_id, _ in done:
                self.pending_migrations.pop(req_id, None)
                self.completed_recving_req_ids.add(req_id)

def _poll_thread_send_release(self, ctx, socks, request_id, remote_address, extra_ts):
    """通过 poll 线程私有的 ZMQ socket 发送 RELEASE。"""
    if remote_address not in socks:
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, f"{self.zmq_address}#poll")
        sock.connect(f"tcp://{remote_address}")
        socks[remote_address] = sock
    rpc_payload = {"cmd": "RELEASE", "request_id": request_id}
    if extra_ts:
        rpc_payload["timestamps"] = extra_ts
    socks[remote_address].send(msgpack.dumps(rpc_payload))
    resp = msgpack.loads(socks[remote_address].recv())
```

**线程安全**：
- ZMQ sockets 不是线程安全的 → 每个线程创建自己的 context 和 sockets
- `pending_migrations`、`completed_recving_req_ids` → 通过 `state_lock` 保护
- 当 poll 线程活跃时，原 `_poll_completed_migrations()` 变为 no-op

**初始化** (`__init__`):
```python
self._migration_poll_stop = threading.Event()
self._migration_poll_thread = None
if self.direct_block_mode:
    self._migration_poll_thread = threading.Thread(
        target=self._migration_poll_loop, daemon=True)
    self._migration_poll_thread.start()
```

---

### 3.2 改动2: P→D Bridge 主动推送

**问题**：Decode 端获取 bridge 数据需要发送同步 `BRIDGE_POP` RPC 到 Prefill，等待 Prefill listener 线程从 `bridge_queue` 中查找并返回。这增加了一个完整的 ZMQ 往返延迟。

**方案**：Prefill 在 `wait_for_save()` 中 `stage_bridge_request()` 之后，主动通过 `BRIDGE_PUSH` RPC 将 bridge 数据推送到 Decode。Decode 存入 `_prefetched_bridges`，在 `start_load_kv()` 中优先检查本地缓存。

**修改文件**：
- `p2p_nccl_engine.py`：新增 `push_bridge_to_decode()`、`pop_prefetched_bridge()`、`BRIDGE_PUSH` listener handler
- `p2p_nccl_connector.py`：`wait_for_save()` 在 stage 后调用 push；`start_load_kv()` 先查 prefetched

**Engine 新��方法**：
```python
def push_bridge_to_decode(self, request_id, context_block_ids, decode_address):
    """P→D 主动推送 bridge 数据。"""
    self._rpc(decode_address, {
        "cmd": "BRIDGE_PUSH",
        "request_id": request_id,
        "context_block_ids": context_block_ids,
    })

def pop_prefetched_bridge(self, request_id) -> Optional[list[int]]:
    """D 端取出预取的 bridge 数据。"""
    with self.state_lock:
        return self._prefetched_bridges.pop(request_id, None)
```

**Listener handler**：
```python
elif data["cmd"] == "BRIDGE_PUSH":
    request_id = data["request_id"]
    context_block_ids = data.get("context_block_ids")
    with self.state_lock:
        self._prefetched_bridges[request_id] = context_block_ids
    self.router_socket.send_multipart([remote_address, msgpack.dumps({"ret": 0})])
```

**Connector `wait_for_save()` 修改**：
```python
def wait_for_save(self):
    if self.is_producer and self.direct_block_mode:
        for request_id, context_block_ids in self._pending_bridge_reqs:
            self.p2p_nccl_engine.stage_bridge_request(request_id, context_block_ids)
            # 新增：主动推送到 decode
            ip, port = self.parse_request_id(request_id, is_prefill=True)
            decode_address = f"{ip}:{port + self._rank}"
            self.p2p_nccl_engine.push_bridge_to_decode(
                request_id, context_block_ids, decode_address)
        self._pending_bridge_reqs.clear()
```

**Connector `start_load_kv()` ���改**：
```python
# 优先检查 prefetched，再 fallback 到 BRIDGE_POP
context_block_ids = self.p2p_nccl_engine.pop_prefetched_bridge(req_meta.request_id)
if context_block_ids is None:
    context_block_ids = self.p2p_nccl_engine.pop_bridge_request(...)
```

---

### 3.3 改动3: P 端 Fast-Release 侧通道

**问题**：P 端 listener 线程收到 RELEASE 后放入 `completed_release_req_ids`。但这个信号要经过：listener → worker `get_finished()` → model_runner_output → scheduler `update_from_output()` → `_free_blocks()` 才能释放。中间被 P 端的 forward step 卡住。

**方案**：在 P 端进程内（UniProcExecutor, TP=1 同进程），用 module-level `SimpleQueue` 实现 listener 线程到 scheduler 的直连通道。

**修改文件**：
- `p2p_nccl_engine.py`：module-level `_fast_release_queue`；RELEASE handler 推送到 queue（不再写 `completed_release_req_ids`）
- `p2p_nccl_connector.py`：新增 `poll_fast_releases()` 方法
- `scheduler.py`：`schedule()` 开头调用 `_poll_fast_releases()`

**Engine module-level 设施**：
```python
_fast_release_queue: Optional[queue_mod.SimpleQueue] = None

def get_fast_release_queue():
    return _fast_release_queue
```

**RELEASE handler 修改**（互斥逻辑）：
```python
elif data["cmd"] == "RELEASE":
    ...
    # 当 fast queue 存在时，只走快路径，不写 completed_release_req_ids
    if _fast_release_queue is not None:
        _fast_release_queue.put_nowait((request_id, ts_entry.copy()))
    else:
        with self.state_lock:
            self.completed_release_req_ids.add(request_id)
```

**关键设计**：`_fast_release_queue` 存在时不写 `completed_release_req_ids`，避免两条路径竞争同一个请求。

---

### 3.4 改动4: 重排 `_update_from_kv_xfer_finished()`

**问题**：原来 `_update_from_kv_xfer_finished()` 在 `update_from_output()` 的末尾调用，释放 blocks 在所有 token output 处理之后。

**方案**：移到 `update_from_output()` ��最开头。

```python
def update_from_output(self, scheduler_output, model_runner_output):
    # 最先处理 KV transfer 完成，尽早释放 blocks
    if model_runner_output.kv_connector_output:
        self._update_from_kv_xfer_finished(model_runner_output.kv_connector_output)
    ...
```

---

### 3.5 改动5: 后台 Block-Free 线程

**问题**：即使有了改动3的 fast-release queue，RELEASE 信号到达 queue 后仍然要等 engine core 主循环下一次 `schedule()` 才会被排空。Engine core 主循环：

```
schedule()            ← _poll_fast_releases() 在此排空 queue
    ↓
execute_model()       ← 2秒长 prefill，RELEASE 在此期间到达 queue
    ↓
update_from_output()
    ↓
[loop]  → schedule()  ← 下一次排空，已经过了2秒
```

**数据印证**（改动3之后 648 个请求）：

```
wait_migration          (start → release_recv):   1.10s  33.7%
schedule_wait           (release_recv → sched):   2.17s  66.2%  ← 最大瓶颈
scheduler_free_lag      (sched → freed):          0.003s  0.1%
```

`schedule_wait` 就是等 engine core 主循环走完 `execute_model() + update_from_output()` 回到 `schedule()` 的时间。

**方案**：在 scheduler 中启动一个后台线程，持续排空 `_fast_release_queue`，在 `execute_model()` 运行期间就释放 KV blocks。

**架构**：

```
         fast_release_queue
              │
              ▼
    ┌─────────────────────┐
    │  bg_free_loop 线程   │  持续 drain queue
    │                     │  with _kv_free_lock:
    │                     │      kv_cache_manager.free(request)
    │                     │  → deferred_frees.put(req_id, ts)
    └──��──────────────────┘
              │
              ▼
         deferred_frees queue
              │
              ▼
    ┌─────────────────────┐
    │  主线程               │  在 schedule() / update_from_output() 开头
    │  _drain_deferred()   │  del self.requests[req_id]
    │                     │  pin/unpin 逻辑
    │                     │  monitoring_recorder
    └───────���─────────────┘
```

**为什么分两步**：

`_free_blocks()` 做了4件事：
1. `kv_cache_manager.free(request)` — 释放 block → **可以后台做**（只需 lock）
2. `del self.requests[request_id]` — 修改 scheduler dict → 必须主线程
3. pin/unpin 逻辑 — 修改 `pinned_requests` list → 必须主线程
4. monitoring 记录 → 必须主线程

所以拆成：后台线程做步骤1（最关键，block 立即可被新请求复用），主��程做步骤2-4（清理工作，不紧急）。

**GIL 分析**：

Python GIL 在这里不是瓶颈。`execute_model()` 的 2 秒中，绝大部分是 GPU kernel 执行时间。PyTorch 在执行每个 CUDA op 时通过 `pybind11::gil_scoped_release` 释放 GIL。所以 CPU 大部分时间是空闲的，后台线程可以在 CUDA kernel 执行间隙获得 GIL，运行 `kv_cache_manager.free()`（微秒级纯 Python 操作）。

**锁竞争分析**：

```
时间轴：
  schedule()           → 主线程持 _kv_free_lock（allocate_slots）
  execute_model()      → 后台线程持 _kv_free_lock（free）← 不重叠
  update_from_output() → 主线程可能持 _kv_free_lock（_free_blocks）
```

`allocate_slots` 只在 `schedule()` 中调用，后台线程的 `free` 主要发生在 `execute_model()` 期间。两者不在同一个 engine core 阶段，**几乎零竞争**。

**修改文件**：`vllm/v1/core/sched/scheduler.py`

**`__init__` 新增**：
```python
self._kv_free_lock = threading.Lock()
self._deferred_frees: queue_mod.SimpleQueue = queue_mod.SimpleQueue()
self._bg_free_thread: Optional[threading.Thread] = None

# 在 kv_cache_manager 创建后启动
if connector is not None and connector.is_producer:
    self._bg_free_thread = threading.Thread(target=self._bg_free_loop, daemon=True)
    self._bg_free_thread.start()
```

**`_bg_free_loop()` 后台线程**：
```python
def _bg_free_loop(self):
    while True:
        released = connector.poll_fast_releases()
        if not released:
            time.sleep(0.001)
            continue
        block_free_ts = time.time()
        for req_id, ts in released:
            ts["block_freed_ts"] = block_free_ts
            request = self.requests.get(req_id)
            if request is None or not request.is_finished():
                continue
            with self._kv_free_lock:
                self.kv_cache_manager.free(request)
            self._deferred_frees.put_nowait((req_id, ts))
```

**`_drain_deferred_frees()` 主线程清理**：
```python
def _drain_deferred_frees(self):
    while True:
        try:
            req_id, ts = self._deferred_frees.get_nowait()
        except:
            break
        request = self.requests.get(req_id)
        if request is None:
            continue
        ts.setdefault("finished_sending_ts", ts.get("block_freed_ts", now))
        monitoring_recorder.record_delay_free_end(req_id, ts)
        # pin/unpin 逻辑...
        del self.requests[req_id]
        self._fast_released_req_ids.add(req_id)
```

**所有 `kv_cache_manager` 操作加锁**：
```python
# allocate_slots (schedule 中)
with self._kv_free_lock:
    new_blocks = self.kv_cache_manager.allocate_slots(...)

# free (preemption, unpin, _free_blocks 中)
with self._kv_free_lock:
    self.kv_cache_manager.free(request)
```

---

## 4. 优化后的完整时间线

```
P-worker: prefill forward 完成
    ↓
P-worker: wait_for_save()
    ├─ stage_bridge_request()                          [bridge_staged_ts]
    └─ push_bridge_to_decode() → BRIDGE_PUSH RPC       (改动2：主动推送)
    ↓
P-scheduler: _free_request() → delay_free             [delay_free_start_ts]
    ↓
    ↓  ← D 端收到 BRIDGE_PUSH ↓
    ↓
    ↓  D-listener: 收到 BRIDGE_PUSH → _prefetched_bridges  (改动2)
    ↓  D-worker: start_load_kv()
    ↓      ↓ pop_prefetched_bridge() → 本地命中           [bridge_popped_ts]
    ↓      ↓ launch_block_migration()                     [migration_launch_ts]
    ↓      ↓ cudaMemcpy2DAsync
    ↓
    ↓  D-poll-thread: event.query() == True               [migration_complete_ts]  (改动1)
    ↓      ↓ 立即发 RELEASE RPC                            [release_callback_sent_ts]
    ↓
    ↓  P-listener: 收到 RELEASE
    ↓      ├─ _fast_release_queue.put()                   [release_received_ts]  (改动3)
    ↓      └─ 不写 completed_release_req_ids
    ↓
    ↓  P-bg-free-thread: poll queue                       [block_freed_ts]  (改动5)
    ↓      ├─ with _kv_free_lock: kv_cache_manager.free()  ← blocks 立即可复用！
    ���      └─ deferred_frees.put()
    ↓
P-scheduler: schedule() 或 update_from_output()
    ↓ _drain_deferred_frees()                             [delay_free_end_ts]
    ↓   ├─ monitoring_recorder.record_delay_free_end()
    ↓   ├─ pin/unpin 逻辑
    ↓   └─ del self.requests[req_id]
```

---

## 5. 监控指标

### 5.1 主要段（加和 = total_delay_free_duration）

```
total_delay_free_duration = wait_migration + schedule_wait + scheduler_free_lag

├─ wait_migration        (delay_free_start → release_received)
│   ├─ wait_bridge_pop   (delay_free_start → bridge_popped)
│   ├─ cuda_memcpy_duration (bridge_popped → migration_complete)
│   └─ release_rpc_total (migration_complete → release_received)
│
├─ schedule_wait         (release_received → finished_sending)
│   改动5之后：= block_freed_ts - release_received_ts
│   即 bg-free-thread 排空 queue 的延迟（应接近 1ms）
│
└─ scheduler_free_lag    (finished_sending → delay_free_end)
    主线程 drain deferred 的延迟
```

### 5.2 新增时间戳

| 时间戳 | 位置 | 含义 |
|--------|------|------|
| `block_freed_ts` | bg-free thread | KV blocks 实际释放时刻 |

### 5.3 保留的 legacy 指标（worker-to-worker 视角）

| 指标 | 公式 | 说明 |
|------|------|------|
| `bridge_wait_duration` | popped - staged | 跨进程 bridge 等待（起点在 delay_free_start 之前） |
| `ipc_setup_duration` | launch - popped | IPC handle 解析 |
| `cuda_memcpy_raw` | complete - launch | 纯 GPU memcpy |
| `poll_to_release_duration` | sent - complete | poll thread → 发 RELEASE |
| `release_rpc_duration` | received - sent | ZMQ 网络延迟 |

---

## 6. 文件修改清单

| 文件 | 改动 |
|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py` | 改动1: poll thread; 改动2: bridge push/prefetch; ��动3: fast-release queue |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` | 改动2: wait_for_save push, start_load_kv prefetch; 改动3: poll_fast_releases() |
| `vllm/v1/core/sched/scheduler.py` | 改���3: _poll_fast_releases; 改动4: reorder; 改动5: bg free thread + lock |
| `vllm/v1/metrics/monitoring.py` | ��控指标更新 |

---

## 7. 预期效果

| 瓶颈 | 优化前 | ��化后 | 改动 |
|------|--------|--------|------|
| bridge_wait (D端取bridge) | 26.3% / 1.0s | ~0 (主动推送) | 2 |
| cuda_memcpy (GPU拷贝) | 7.1% / 0.08s | 不变 | - |
| D端 forward 卡住 poll | 隐藏在 bridge_wait 中 | ~0 (后台 poll) | 1 |
| engine_poll_lag (P端 forward 卡住) | 31.3% | ~0 (fast queue) | 3 |
| scheduler_free_lag | 35.5% | ~0 (reorder) | 4 |
| schedule_wait (engine core 循环) | 66.2% (改动3后) | ~1ms (bg thread) | 5 |

改动5之后，`total_delay_free_duration` 应该约等于 `wait_migration`（~1秒），其中大部分是 `wait_bridge_pop`（等 D 端下一次 `start_load_kv`）和 `cuda_memcpy`（0.08秒）。

---

## 8. Bug 修复：RELEASE 提前到达导致 blocks 泄漏

### 8.1 问题描述

改动5上线后发现：**驱逐（eviction）数量反而增加了**。

### 8.2 根本原因：RELEASE 在 request 标记 finished 之前到达

Engine core 主循环的时序：

```
schedule()
    ↓
execute_model()
    ├─ forward pass
    ├─ wait_for_save() → bridge push 到 D
    ├─ D 收到 bridge → 迁移 → 发 RELEASE
    ├─ P-listener 收到 RELEASE → _fast_release_queue.put()
    └─ bg 线程从 queue 消费 → request.is_finished() == False ← !!
    ↓
update_from_output()
    └─ 此时才标记 request finished + delay_free
```

**关键时序问题**：当同机器迁移速度很快时，RELEASE 可以在同一个 engine-core loop 中到达，即在 `execute_model()` 期间。此时 `update_from_output()` 还没有把请求标记为 finished。

原来的 `_bg_free_loop` 代码：
```python
request = self.requests.get(req_id)
if request is None or not request.is_finished():
    continue  # ← 丢弃！RELEASE 已从 queue 消费，永远不会重试
```

由于 fast-release 路径是互斥的（RELEASE handler 不写 `completed_release_req_ids`），这些 RELEASE 没有任何 fallback 路径。**blocks 永远不会被释放** → KV cache 逐渐耗尽 → 更多驱逐。

### 8.3 为什么 prefix cache 不能掩盖问题

即使 blocks 被 `kv_cache_manager.free()` 释放，它们的 hash 仍然保留在 `block_pool` 中，后续请求应该能 prefix cache 命中。

但这里的问题不是"blocks 被错误释放"，而是"blocks 永远不被释放"——**泄漏**。泄漏的 blocks 持续占用 KV cache 空间（`ref_cnt > 0`），`allocate_slots` 在 free queue 中找不到足够的 blocks，只能驱逐其他请求的 blocks。

### 8.4 修复方案

**核心思路**：bg 线程遇到未 finished 的请求时，不要丢弃 RELEASE，而是保存到本地 `pending` buffer，下次 poll 循环时重试。

```python
def _bg_free_loop(self) -> None:
    poll_fn = ...
    is_continuum = (self.policy == SchedulingPolicy.CONTINUUM)
    pending: list[tuple[str, dict]] = []  # 未 finished 的 RELEASE，等待重试

    while True:
        released = poll_fn()
        if not released and not pending:
            time.sleep(0.001)
            continue

        # 合并新到达�� RELEASE 和之前 pending 的
        to_process = pending + (released or [])
        pending = []

        block_free_ts = time.time()
        for req_id, ts in to_process:
            ts.setdefault("block_freed_ts", block_free_ts)
            request = self.requests.get(req_id)

            if request is None or not request.is_finished():
                # RELEASE 先于 update_from_output() 到达，保留重试
                pending.append((req_id, ts))
                continue

            # Continuum pin 检查
            might_pin = (is_continuum
                         and not getattr(request, "is_last_step", True))
            if might_pin:
                self._deferred_frees.put_nowait((req_id, ts, False))
                continue

            with self._kv_free_lock:
                self.kv_cache_manager.free(request)
            self._deferred_frees.put_nowait((req_id, ts, True))
```

### 8.5 `_drain_deferred_frees()` 3-tuple 设计

`_deferred_frees` 中的每个 item 是 `(req_id, ts, blocks_freed)`：

- `blocks_freed=True`：bg 线程已经调用了 `kv_cache_manager.free()`，主线程只做 cleanup（del requests、unpin、monitoring）
- `blocks_freed=False`：bg 线程跳过了 free（Continuum pin），主线程调用完整 `_free_blocks()`

### 8.6 监控指标变化

修复后，`delay_free_end_ts` 的含义取决于 `blocks_freed`：

- `blocks_freed=True`：`delay_free_end_ts = block_freed_ts`（后台线程实际释放时刻）
- `blocks_freed=False`：`delay_free_end_ts = deferred_cleanup_ts`（主线程 drain 时刻）

`deferred_cleanup_lag = deferred_cleanup_ts - block_freed_ts` 反映主线程处理延迟（但此期间 blocks 要么已被释放，要么本就不应释放）。
