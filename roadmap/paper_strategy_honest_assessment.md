# 论文工作严格评估与故事重构

> 本文档是 2026-05-29 ~ 05-30 期间论文方向讨论的诚实记录。
> 不为了维护信心而粉饰任何判断。
> 用途：作为后续 3.5 个月工作的指导，以及论文写作的对齐基准。

---

## Part 1 · 目前已完成的工作

### 1.1 Foresight Backfill Scheduler

**机制**：
- 把请求按自制优先级排队
- 创建一个 "timeline" 列表，每格表示该 step 结束后还剩多少 free KV block
- 把请求按顺序放入 timeline，每 step 减去消耗
- 贪心策略 + 类 SLURM backfill 思路决定 admit

**声称的结果**：
- 0 preemption
- KV 利用率保持高水准

**实现状态**：代码完成，跑通真实 workload

### 1.2 Shadow Simulator

**机制**：
- 零训练
- 利用真实 scheduler 的决策规则做 deterministic replay
- 预测"下一 step 这个请求能不能被 admit"

**实际能力（重要修正）**：
- **预测目标**：binary 单步 "next-step admit yes/no"
- **准确率**：98%（agent workload）
- **扩展尝试**：曾尝试预测 K-step admit，准确率显著下降
- **适用范围**：间隔短的请求（agent tool 调用）。**长间隔请求（chat 用户回归）shadow 不能预测**——只能等请求真实进入 prefill 队列后才有依据
- **真正在用的下游**：prefetch（KV 提前搬运）

**实现状态**：代码完成，准确率验证完成

### 1.3 KV Reuse Arena Infrastructure（部分完成）

**当前状态**：
- 已有共享 pinned arena + cudaHostRegister 机制
- 当前使用 FIFO 环形 bump 分配器（已识别问题）
- 已有跨多轮 KV 复用（lookup + load）链路跑通
- Phase 1 (save-on-preempt) 和 Phase 2 (admission gate) 已实现但默认关闭

**待做**：
- FIFO → LRU 改造（已有完整设计文档 `pinned_arena_lru_redesign.md`）
- 直读 CUDA kernel（可选阶段 5）

---

## Part 2 · 现有工作的诚实评分

### 2.1 强项

#### Backfill Scheduler

- 真创新候选 ★★★★
- 普适性强（不限 workload type）
- 0-preempt 是干净的卖点
- 跟 Sarathi-Serve / vLLM / SGLang 默认调度的差异可以 articulate

#### Shadow Simulator

- 真创新候选 ★★★★（mechanism 层面）
- 零训练 + 98% 准确率是反直觉的好结果
- 概念"deterministic replay = prediction"有学术 insight 潜质

#### KV Reuse Infrastructure

- 工程扎实
- 跨进程、多写多读的设计是真实工程贡献
- 但**作为论文 contribution 不够分量**——属于 infrastructure

### 2.2 弱项

#### Backfill Scheduler 的弱点

- **跟 Sarathi-Serve 的差异化需要拿放大镜找**——他们也做 chunked-prefill aware 调度
- 跟 Llumnix / FastServe 等也要区分清楚
- "0 preempt" 在某些 baseline 上也能做到 → 必须找出 baseline 撞墙的 regime 证明优势

#### Shadow Simulator 的弱点

- **能力比想象的窄**：binary 单步、agent-only
- 无法直接驱动 SLO 估计（要 latency 数字，binary 给不了）
- 无法直接驱动 anti-starvation（要多步预测）
- 无法直接驱动 cross-instance LB（要全实例时间维预测）
- 在 long-interval workload（chat）上**根本不能用**

#### 整体下游应用的弱点

- shadow 的下游目前只有 prefetch
- prefetch 节省的就是"一个 layer 的搬运时间" ≈ 几十 ms
- 在 end-to-end 上是 5-15% throughput up 量级
- **不构成 ASPLOS 级别的 sell point**

#### Arena 工作的定位

- 是 infrastructure，不是 contribution
- 写 paper 时只能放 implementation 章节
- 投入的工程量不会转化为论文档次提升

---

## Part 3 · 旧故事线及其问题

### 3.1 旧故事链

```
多轮 agent 高并发下 KV 被 LRU 冲走
   ↓
PD-disagg + 全局缓存池让 decode 算完的 KV 不丢
   ↓
prompt 长度方差大 → prefill HOL 阻塞
   ↓
chunked prefill 救 HOL（已有技术）
   ↓
chunked 多 → KV 紧张 → preempt
   ↓
backfill 调度器 → 0 preempt + 高利用 (C1)
   ↓
shadow 模拟器 98% 准确 (C2)
   ↓
shadow 驱动 prefetch overlap (C3)
```

### 3.2 旧故事链的断裂点

**断裂点 A：HOL motivation 在 KV 复用下被削弱（部分救得回）**

观察：当 KV 复用率高时，所有请求"实际计算的 token 数" 都很短（多轮 agent 场景几乎都是一个 chunk），原本 chunked-prefill 解决的 HOL 不再是主要矛盾。

但这条**不完全死**——多场景共存下 HOL 仍存在：
- eviction-induced recompute：被 LRU evict 的 inc 重新算时变长请求
- 混合 workload：agent（间隔短，命中高）+ chat（间隔长，命中低）+ RAG（首轮长 context）
- cold start：新 job 第一轮全量 prefill
- system prompt cache miss：个性化 RAG context 不命中
- 突发流量：arena 冷启动

**条件**：救活 motivation 必须做 workload characterization 实验，量化"在真实/混合 workload 下，effective prefill length 分布 heavy-tail"。

**断裂点 B：prefetch overlap 收益过小（救不回）**

层级 prefetch 节省的就是 "下一波 admit 的第一层搬运时间" ≈ 几十 ms。在 end-to-end 上比例小，**不是机制问题，是物理问题**。换 workload 也救不回。

**断裂点 C：shadow 的能力被高估**

之前讨论时假设 shadow 是"general 未来状态预测器"，实际是 binary 单步 agent-only。这导致很多设想的下游应用（SLO 估计、anti-starvation、跨实例 LB、hierarchy staging）**实际跑不起来**。

### 3.3 严格审稿模拟（旧故事）

假设按旧 framing 投 ASPLOS：

> Weak reject (3/6)：
> - Motivation 不自洽（KV reuse 后 HOL 论证模糊）
> - Prefetch overlap 收益小，难以支撑 end-to-end 数字
> - Backfill 跟 Sarathi-Serve 差异化弱
> - shadow 98% 是好数字但下游应用单一
> - 整体读起来像系统 grab-bag

---

## Part 4 · 改进后的故事线

### 4.1 新故事链

```
PD-disaggregated 多轮 / 混合 workload LLM serving
   ↓
现实 workload 分析: agent + chat + RAG 共存下
   - effective prefill length 仍 heavy-tail
   - eviction-induced recompute + cold start 周期性出现
   - HOL 仍是真实问题（数据支撑）
   ↓
已有调度都是 myopic（FCFS / priority / vLLM 默认）
   - 不考虑未来 KV 占用
   - 不考虑结构化交互的可预测性
   ↓
我们提出 predictive resource management 框架
   ├── Shadow Simulator: 零训练 deterministic replay
   │      - binary 单步 98%（高置信度即时决策）
   │      - K-step 多步 ~80% (中等置信度中短期决策)
   │      - 明确承认 K>5 不可信，长间隔 workload 不适用
   │
   ├── Foresight Backfill Scheduler: 用 shadow 做未来感知 admission
   │      - timeline-based + backfill
   │      - 0 preempt + 高 util
   │
   ├── Predictive Resource Decisions (新增 architecture flavor):
   │      - Bandwidth-aware scheduling（PCIe + NUMA）
   │      - Speculative prefill warmup（架构经典思想引进）
   │
   └── KV Hierarchy Infrastructure: arena LRU + tail-first + self-heal
          - 作为 implementation 章节，不当 contribution
   ↓
End-to-end 评估：混合 workload 下 throughput + SLO + tail latency 改善
```

### 4.2 各 contribution 在新故事中的角色

| Contribution | 角色 | 备注 |
|---|---|---|
| Workload characterization | **Motivation 弹药** | 决定整篇 paper 能不能立 |
| Shadow Simulator | **核心 mechanism** | 但要诚实声明 scope（agent / 间隔可预测） |
| Foresight Backfill | **核心调度算法** | 通用，跨 workload type |
| Bandwidth-aware scheduling | **架构 flavor** | PCIe + NUMA + HBM 协同 |
| Speculative prefill warmup | **架构 flavor** | speculative execution paradigm |
| KV Hierarchy Infrastructure | **Implementation** | Arena 工作；不当 contribution |

### 4.3 工作重心

按重要性排序：

1. **Workload characterization 实验**（最关键，月 1 前两周必须出）
2. **Shadow K-step 扩展**（实验 + 准确率分析）
3. **Bandwidth-aware scheduling**（架构 flavor）
4. **Speculative prefill warmup**（架构 flavor + speculative paradigm）
5. **Arena LRU 改造**（infrastructure，工程量大但论文价值有限）
6. **Baseline 跑通 + 端到端实验**

工作重心**绝对不应该**放在：
- 层级 prefetch 优化（机制收益小）
- 直读 CUDA kernel（性能优化，不影响 paper 故事）
- 多机改造（3.5 个月做不完）
- 新做文件系统（跟 Mooncake/LMCache 撞）

---

## Part 5 · 主线 thesis

**一句话主线**：

> 在 PD-disaggregated 多轮 + 混合 workload LLM serving 中，调度和资源管理决策本质需要未来信息；我们用零训练的 deterministic shadow 模拟提供这些信息，并基于此设计了跨 OS-Architecture 层的预测式资源管理（调度 + 带宽感知 + speculative execution），在真实混合 workload 下证明显著的 throughput + SLO 提升。

**关键 message**：
- **Predictive** 是 paradigm shift 关键词（过去十年 reactive scheduling，我们 predictive）
- **Cross-layer** 强调 OS + Architecture 交叉
- **Zero-training** 强调 shadow 的反直觉创新点

**绝对不要 sell 的 message**：
- "我们做了通用 LLM serving 的所有事"——scope 太宽撑不住
- "我们 prefetch 隐藏了 KV 搬运时间"——收益小，证不出
- "我们的 arena 是新设计"——是工程不是创新

---

## Part 6 · 如何满足 ASPLOS Architecture + OS 交叉

### 6.1 ASPLOS scope 要求

ASPLOS 要求至少两个维度交叉：**Architecture / OS / PL**。

你的工作**自然挂 OS**（调度、资源管理、跨进程同步）。
**架构 flavor 需要主动注入**——PL 不挂，别强凑。

### 6.2 注入架构 flavor 的轻量路径

#### 路径 A：Bandwidth-aware predictive scheduling

把 **PCIe 带宽 / NUMA 拓扑 / GPU memory hierarchy** 当 first-class resource，scheduler 决策时考虑：

- PCIe 拥塞：多个请求同时 H2D 会撞 PCIe Gen5 64 GB/s 上限
- NUMA 亲和性：arena pinned 内存所在 NUMA 节点 vs GPU 所在 NUMA 节点
- GPU HBM 占用 vs Host pinned arena 占用 vs SSD（如果有第三层）

shadow 预测下来这些资源的未来占用，**调度决策考虑它们**。

**架构挂钩**：硬件资源建模 + 跨层资源管理 = architecture
**OS 挂钩**：scheduling、资源分配 = OS

**工程量**：中等。基本上是在现有 scheduler 内加几个 resource model + 决策时考虑。

#### 路径 B：Speculative prefill warmup

shadow 高置信度预测下一 step 会 admit 某个请求 → **在 scheduler 真正决策前**就启动这个请求的 prefill kernel：
- 如果预测正确：admission-to-first-token 延迟降低
- 如果预测错误：abort kernel，浪费少量 SM 周期

**架构挂钩**：speculative execution 是经典 architecture paradigm（branch prediction、speculative load 等），引进到 ML serving = architecture-flavor 创新
**OS 挂钩**：scheduling + speculative resource allocation = OS

**工程量**：中等。需要在 worker 端加 speculative kernel 启动 + abort 机制。

#### 路径 C：Memory hierarchy migration（可选）

如果加 NVMe SSD 第三层：
- shadow 预测 K-step 后命中 → 提前从 SSD 升 host pinned
- shadow 预测短期不命中 → 主动降到 SSD

**架构挂钩**：跨层数据移动 + 容量层次管理 = architecture
**OS 挂钩**：cache 管理、background migration = OS

**工程量**：大。需要做 tiered storage 完整工程。

**判断**：跟 AttentionStore 撞较多，**不推荐做**。

### 6.3 推荐组合

**核心架构 flavor = 路径 A + 路径 B**（一定要做）
**路径 C 不做**（撞 AttentionStore，工程量大）

这样的组合：
- OS 维度：backfill scheduler + shadow simulator + admission policy
- Architecture 维度：bandwidth-aware decisions + speculative execution
- 交叉点：shadow predictions 同时驱动 OS 决策和 architecture 资源管理

**这是合法的 ASPLOS OS+Architecture 交叉**。

---

## Part 7 · 不应该追求的扩张

### 7.1 不做：加文件系统

**理由**：
- KV 存进分布式 FS 跟 Mooncake Store / AttentionStore 撞车
- 新做 KV-specific FS 工作量 = 单独一篇论文
- 3.5 个月内做不完且做不深
- 没有清晰的 differentiator

### 7.2 不做：改多机分布式

**理由**：
- 需要重做几乎所有现有 contribution（backfill、shadow、arena 都要 redesign）
- 必须正面对比 Mooncake（FAST'25 标杆），打不过
- 工程量估算 20 周以上（远超 14 周可用时间）
- 即便做完也大概率撞 Mooncake 的设计空间

### 7.3 不做：层级 prefetch overlap 深度优化

**理由**：
- 收益在物理层面就有上限（几十 ms / layer）
- 不是 paper 故事的核心
- 调到极致也撑不起 contribution

### 7.4 不做：直读 CUDA kernel 干掉 staging（可推迟）

**理由**：
- 是性能优化，不影响 paper 故事
- 工程量中等，但不会让 paper 升档
- 推迟到论文接收后再做

### 7.5 不做：理论 competitive bound 正经证明

**理由**：
- 没有理论 mentor 指导下从零起步 2-3 月可能证不出来
- ASPLOS 是系统会议，bound 不是必须
- 失败成本高（占用 1-2 月写不出可发表的证明）

**替代**：可写一个**轻量分析章节**——"在我们 cost model 下 tail-first 期望 loss 比 random eviction 减少 X%"，简单概率算即可。

---

## Part 8 · 严格审稿模拟（新故事）

假设按新故事 framing + 完成 workload characterization + 完成 K-step shadow + 完成 bandwidth-aware + speculative：

> Reviewer #2: weak accept (4.5/6)
>
> Strengths:
> - 故事 coherent，predictive resource management 是有 thesis 的框架
> - Workload characterization 提供 motivation 的实证基础（前提：数据真实）
> - Shadow simulator with tiered K-step prediction 是 mechanism 创新
> - Backfill + Bandwidth-aware + Speculative 三层应用展示 shadow 的 generality
> - ASPLOS scope 上挂 OS + Architecture 自洽
>
> Weaknesses:
> - Foresight Backfill 跟 Sarathi-Serve 差异化需要 sharper
> - Shadow K-step 多步准确率掉下去后实际 usability 需要展示
> - Speculative prefill warmup 需要展示真实架构 insight，不能只是 "use shadow + start kernel early"
> - Arena 实现细节可不必出现在正文（移附录）
> - 缺少 scaling experiments（即便单节点，也要在多 GPU 上验证）

**预期评分**：4.0-4.5（borderline accept 到 weak accept 区间）

**Best Paper 距离**：很远。需要再加：
- 理论深度
- 大规模评估（多节点）
- 颠覆性 insight
- 工业级 reproducibility

3.5 个月做不到 Best Paper，但**solid accept 是可达的**。

---

## Part 9 · 3.5 个月时间分配

### Month 1（最关键，决定生死的 4 周）

**Week 1-2**：Workload characterization 实验
- 跑 multiturn_trace_client.json + LMSYS-Chat-1M（或 ShareGPT）+ 真实 agent trace
- 量化 effective prefill length 分布
- 量化 eviction-induced recompute 频率
- 量化混合 workload 下 HOL 出现率
- 产出 2-3 张关键 motivation figure

**Week 3-4**：定 C4/C5 设计（bandwidth-aware + speculative prefill）
- 不写代码，先做算法 sketch
- 设计实验验证点
- 跟 mentor / collaborator 反复打磨

**Month 1 末关键判断点**：
- Workload characterization 数据是否支撑 HOL 在混合 workload 下显著存在
- 如果数据不支撑 → motivation 塌房 → 必须重新找 framing 或转会议

### Month 2（实现期）

**Week 5-6**：Arena LRU 改造（按 pinned_arena_lru_redesign.md 阶段 0-3 落地）

**Week 7-8**：Bandwidth-aware scheduling 实现 + speculative prefill warmup 实现

### Month 3（实验期）

**Week 9**：Shadow K-step 扩展 + 准确率分布实验

**Week 10-11**：端到端实验 + baseline 跑通
- vLLM、SGLang、Sarathi-Serve、LMCache adapted 全部跑通
- 多 workload + 多 metric
- microbenchmark + macrobenchmark
- 多 GPU scaling（至少 1 → 2 → 4 GPU）

**Week 12**：补实验 + 异常分析（reviewer 一定会问 failure case）

### Month 3.5（写作期）

**Week 13-14**：写作 + 反复打磨

### 投稿后 buffer

补充实验 + rebuttal 准备

---

## Part 10 · 主要风险点

| 风险 | 严重度 | 应对 |
|---|---|---|
| Workload characterization 出来发现 HOL 在真实 workload 下不显著 | 致命 | Month 1 前两周必须验证；不成立则换会议或换 framing |
| Shadow K-step 准确率掉到 60% 以下 | 高 | 设计 fallback：binary 仍然用、K-step 用置信度过滤；论文 honest 报告 |
| Bandwidth-aware 在 H100 PCIe Gen5 下不构成瓶颈 | 高 | 多 GPU 共享 PCIe lane 时构成；用 scaling experiment 暴露 |
| Speculative prefill 错预测浪费 SM > 节省时间 | 中 | 加置信度门限；只在高置信度预测时启动 |
| Sarathi-Serve / vLLM 默认调度在大多数场景下也 ~0 preempt | 中 | 实验设计要找出 baseline 撞墙的高负载 regime |
| Arena 改造工程量超预期 | 中 | 阶段 0-3 必做，阶段 4（Phase 1/2 接入）可推迟 |
| ASPLOS 8 月 due，时间紧 | 中 | Buffer 用月 3.5；不成立则转 MLSys 2026（10-11 月 due） |
| Best Paper 期望落空 | 低 | 不应作为目标设计，专注 solid accept |

---

## Part 11 · 关于会议选择

### 11.1 ASPLOS 2026 现实评估

**支持因素**：
- OS + Architecture 交叉成立（路径 A + B）
- 多轮 / agent serving 是 hot topic
- 已有 ML serving 论文先例（Sarathi-Serve、Splitwise、DistServe）

**风险因素**：
- 8 月底 due，时间极紧（如果 Month 1 数据出问题就来不及）
- ASPLOS 接收率 ~20%，竞争激烈
- 论文需要 polish 程度高

**建议**：
- **主投 ASPLOS**：按上面 plan 推进
- **备选 MLSys 2026**（约 11 月 due）：如果 ASPLOS 时间不够或数据未到位

### 11.2 不推荐的会议

- **FAST**：你不是存储论文，强行写存储角度会被嫌"调度伪装存储"
- **OSDI**（如果有当年的）：你的工作没到 OSDI 期待的 paradigm-shifting 高度

### 11.3 Best Paper 期望

**不应作为目标设计**。Best Paper 是接收论文 5% 顶端，**3.5 个月内难以达到**。

按 solid accept 为目标设计，Best Paper 是抽奖结果，不是规划目标。

---

## Part 12 · 立即可执行的下一步

按时间紧急程度排：

1. **Week 1 第一件事**：开始 workload characterization 实验设计
   - 确定要用哪些 trace（已有 multiturn_trace_client + 至少一个公开 trace）
   - 确定要测量的 metric（effective prefill length / KV hit rate / eviction rate / etc.）
   - 设计实验脚本

2. **Week 1 第二件事**：写 ASPLOS submission abstract 草稿
   - 把新故事主线落到 200 词
   - 用这个 abstract 反复检验后续工作是不是 on track

3. **Week 1 第三件事**：扫一遍最新 baseline 实现
   - vLLM 最新版（v0.6+）
   - SGLang
   - Sarathi-Serve（如果开源）
   - LMCache + Mooncake 是否需要 baseline

4. **Week 2-4**：按 Part 9 的 plan 推进

---

## Part 13 · 总结判断

### 13.1 工作总体定位

你做的工作**有真实贡献**（backfill + shadow），但**目前的 paper framing 不能直接 sell**。

诚实评估：
- 在 ASPLOS 期望档次上：现状是 weak reject (3/6)
- 经过改进（workload characterization + 架构 flavor + 故事重构）：可达 weak accept (4.5/6)
- Best Paper：不在 3.5 个月内可达范围

### 13.2 核心修复

1. **接受 shadow 的真实能力边界**（binary + 单步 + agent-only / K-step 中等准确率）
2. **接受 prefetch 是弱应用**，重心移到 speculative + bandwidth-aware
3. **接受 arena 是 infrastructure**，不当 contribution
4. **接受单机是合理 scope**，不强扩多机
5. **接受 Best Paper 不是目标**，专注 solid accept

### 13.3 主线 thesis 一句话再说一遍

> 在 PD-disaggregated 多轮 / 混合 workload LLM serving 中，调度和资源管理决策本质需要未来信息；我们用零训练的 deterministic shadow 模拟提供这些信息，并基于此设计了跨 OS-Architecture 层的预测式资源管理（backfill 调度 + 带宽感知 + speculative execution），在真实混合 workload 下证明显著的 throughput + SLO 提升。

---

## 附录 · 我作为对话者的诚实声明

讨论过程中：

- 第一次评估 L6/L7/L8（tail-first / self-heal / unified pool）时**过度包装为创新**，实际它们都很朴素
- 第一次评估 shadow 时**假设它是 general 时间预测器**，实际是 binary 单步 agent-only
- 第一次推荐 SLO 预测 / anti-starvation / hierarchy staging 等下游应用，**未考虑 shadow 的真实能力边界**
- 多次给"信心补足型"建议，**没有诚实指出 best paper 距离**

这些都是讨论过程中的偏差，本文档已修正。

**未来讨论建议**：你随时质疑我的乐观判断，我会进一步严格化。如果某个判断我说不准，我会明确说"这一条我没把握"。

---

**文档版本**：v1.0（2026-05-30 初稿）
**状态**：诚实评估，待你确认是否接受这套 framing
