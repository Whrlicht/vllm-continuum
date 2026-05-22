# Tool Call Time Predictor

预测 agent 工具调用执行时间 (T1)。设计为通用框架, 不依赖任何数据集专属
ID (instance_id / repo / model name 都不进特征)。

## 数据流

```
trace JSON
  -> dataset.iter_tool_calls()      # 按消息顺序遍历
  -> bucket.classify()              # v3 桶分类 (纯规则)
  -> features.TrajectoryState       # A+B 静态 + E 轨迹历史观测
  -> train.train_full_bundle()      # 三个 LightGBM head
                                     #   p_timeout (binary)
                                     #   p50, p95   (quantile on log(1+t))
  -> Predictor.predict_df()         # bash 走 ML, submit/editor 查表
  -> evaluate.*                     # log-MAE / MAPE / Spearman / P95 cov
```

## 文件清单

| 文件 | 职责 |
|---|---|
| `bucket.py` | v3 桶分类器 (规则, 跨数据集稳定) |
| `features.py` | A (命令结构) + B (子参数语义) + E (轨迹观测) 特征 |
| `dataset.py` | trace 加载 + 因果滚动构造样本 DataFrame |
| `train.py` | LightGBM 训练 + 保存 + Predictor 推理类 |
| `evaluate.py` | 桶级 + 总体指标 |
| `main.py` | CLI 入口 (baseline / train / evaluate) |
| `runs/` | 训好的模型 + bundle.json 保存这里 |

## 用法

```bash
# 1. baseline: 仅 train 桶 median 查表, 看预测下界
python3 main.py baseline

# 2. 完整训练 + 评估
python3 main.py train

# 3. 已训完, 仅在 val 上评估
python3 main.py evaluate
```

默认输入:
- train = `/data/whr/vllm-continuum/trace_data/swe_bench_sample_2902_tool_clean_with_timings.json`
- val   = `/data/whr/vllm-continuum/trace_data/swe_bench_sample_300_tool_clean_with_timings.json`

## 设计决策记录

1. **去掉 instance_id**, 用 E 类轨迹观测特征恢复 repo 复杂度信号。换数据集
   时桶分类器和特征体系完全保持工作。
2. **submit / editor 查表**: 这两个 family 真值方差 < 4ms, ML 不带额外收益。
3. **bash 统一一个模型** (而不是每桶一个): 桶 ID 进 categorical, GBDT 自己
   学桶间迁移; 维护简单, 跨桶共享统计强度。
4. **三个 head**: P_timeout (二分类) + 给定不 timeout 的 P50 + P95。下游
   调度器拿这三件套自己决定保守度。
5. **训练时严格 causal**: `TrajectoryState` 按消息顺序滚动, 第 j 条 call 的
   特征只来自 0..j-1 的真值。训练-推理特征构造对称, 无 distribution shift。

## 当前 Phase

- ✅ Phase 0: 桶分类器 + 查表 baseline + 评估管线
- ✅ Phase 1: A+B+E 特征 + LightGBM 三头训练
- ✅ Phase 2: P_timeout 二分类 + P95 训完后在 val 上做 quantile-shift calibration
  (bundle.json 里的 `p95_log_shift`, 推理时自动应用)
- ✅ Phase 3: C 类工件特征 (`features.FileCache` 维护 trajectory 内
  create/view/find 的文件内容 + testbed 结构, 给 python script /
  pip install -e / pytest 三类调用注入额外特征)
- ✅ Phase 4: leave-one-repo-out (`cross_repo.py`) 跨 repo K-fold 评估
- ✅ 加速: pandas categorical → 手动 factorize 整数编码 (训练应再快 5-10 倍)

## 跑 LOO 评估

```bash
python3 cross_repo.py -k 5
```

5 折交叉, 每折按 owner__repo 切分。报每折 + 平均的
log_MAE / Spearman / P95 cov, 用以判定模型 repo-agnostic 程度。
