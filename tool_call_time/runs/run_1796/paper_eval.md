# Tool Call Time Predictor — Paper Evaluation

数据集: SWE-smith Trajectories tool split, 199 train / 300 val

模型: LightGBM 4 head (P50 quantile / P95 quantile / P_long_run binary / 2 specialists)

特征: A 命令结构 + B 子参数语义 + C 工件 + E 轨迹观测 + E5 OBSERVATION 状态信号 (共 63 列, 0 instance_id 依赖)

## Method 1 — editor/submit 用 train 95th-percentile 替代 median 充 P95

- 全集 P95 coverage **before**: `0.6105`  (median 充 P95 → 50% 下界)
- 全集 P95 coverage **after**:  `0.9350` ✅

改动: editor/submit 真值近似常数, train 经验分布的 95 分位点比 median 更合理代表"上界"。

## 总体指标 (after Method 1)

| 切片 | n | log_MAE | Spearman | Acc@(5ms,5%) | Acc@(10ms,10%) | Acc@(50ms,20%) | P95 cov |
|---|---|---|---|---|---|---|---|
| all (含 editor/submit) | 7907 | 0.0537 | 0.8803 | 0.7187 | 0.7918 | 0.8914 | 0.9350 |
| editor + submit | 4804 | 0.0001 | 0.4944 | 1.0000 | 1.0000 | 1.0000 | 0.9255 |
| bash only | 3103 | 0.1366 | 0.8946 | 0.2833 | 0.4695 | 0.7232 | 0.9497 |

## Method 3 — 按 Tier 分层

| Tier | n | actual_p50 | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|---|
| constant_submit | 510 | 0.0000s | 0.0000s | 0.0000 | nan | 1.0000 | 1.0000 |
| constant_editor | 4294 | 0.0002s | 0.0002s | 0.0001 | 0.3162 | 1.0000 | 0.9166 |
| light_bash | 1036 | 0.1906s | 0.1988s | 0.0276 | 0.3126 | 0.3726 | 0.9817 |
| normal_bash | 392 | 0.4239s | 0.7261s | 0.1084 | 0.7993 | 0.1990 | 0.9184 |
| heavy_bash | 1675 | 0.4880s | 4.9471s | 0.2107 | 0.8415 | 0.2478 | 0.9373 |

## 桶级 (n ≥ 5, 按 n 降序)

| bucket | n | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|
| `bash::python::script_repro` | 1470 | 2.530s | 0.1506 | 0.8196 | 0.2673 | 0.9544 |
| `editor::view::with_range` | 1228 | 0.000s | 0.0002 | nan | 1.0000 | 0.9357 |
| `editor::view::no_range` | 805 | 0.000s | 0.0001 | nan | 1.0000 | 0.9516 |
| `editor::str_replace_lt5000` | 660 | 0.000s | 0.0001 | nan | 1.0000 | 0.9379 |
| `editor::str_replace_lt1000` | 521 | 0.000s | 0.0001 | nan | 1.0000 | 0.9578 |
| `submit` | 510 | 0.000s | 0.0000 | nan | 1.0000 | 1.0000 |
| `editor::create_lt1KB` | 407 | 0.000s | 0.0001 | nan | 1.0000 | 0.7862 |
| `bash::light::grep` | 379 | 0.190s | 0.0136 | 0.3703 | 0.4354 | 0.9868 |
| `editor::create_lt10KB` | 365 | 0.000s | 0.0000 | nan | 1.0000 | 0.7726 |
| `bash::find::with_name_pipe` | 286 | 0.202s | 0.0159 | 0.1291 | 0.4196 | 0.9895 |
| `editor::str_replace_lt200` | 276 | 0.000s | 0.0001 | nan | 1.0000 | 0.9891 |
| `bash::light::rm` | 218 | 0.186s | 0.0162 | 0.0418 | 0.3303 | 0.9725 |
| `bash::python::script_other` | 183 | 0.688s | 0.1010 | 0.7851 | 0.2186 | 0.9180 |
| `bash::python::pytest_path` | 52 | 0.743s | 0.0919 | 0.6458 | 0.2500 | 0.9231 |
| `bash::light::grep::chained` | 47 | 0.208s | 0.0348 | 0.2679 | 0.2340 | 0.9362 |
| `bash::pip::pypi::single` | 42 | 8.177s | 0.8415 | 0.5402 | 0.1429 | 0.7143 |
| `bash::env_prefix` | 31 | 0.440s | 0.1084 | 0.5403 | 0.1613 | 0.9355 |
| `bash::pip::editable_local::default` | 29 | 79.115s | 0.6296 | 0.5210 | 0.1379 | 0.7931 |
| `editor::str_replace_ge5000` | 29 | 0.000s | 0.0001 | nan | 1.0000 | 0.8966 |
| `bash::python::module_profiler` | 28 | 0.282s | 0.0311 | 0.1275 | 0.0714 | 1.0000 |
| `bash::find::exec_grep` | 27 | 1.660s | 0.2345 | 0.4737 | 0.1481 | 1.0000 |
| `bash::python::pytest_single_test` | 25 | 0.883s | 0.1177 | 0.7523 | 0.2000 | 0.8400 |
| `bash::python::module_mypy` | 23 | 3.393s | 0.4878 | 0.5059 | 0.0000 | 0.8696 |
| `bash::find::with_name_basic` | 22 | 0.201s | 0.0277 | 0.1753 | 0.2273 | 1.0000 |
| `bash::python::unittest_specific` | 20 | 0.287s | 0.0337 | 0.1669 | 0.2000 | 0.9000 |
| `bash::light::rm::chained` | 19 | 0.230s | 0.0372 | 0.2509 | 0.2105 | 1.0000 |
| `bash::python::pytest_full_discovery` | 19 | 6.919s | 0.3330 | 0.7333 | 0.2632 | 0.9474 |
| `bash::pip::pypi::multi_pkg` | 16 | 11.919s | 0.7474 | -0.0088 | 0.1250 | 0.8750 |
| `bash::python::interactive_short` | 16 | 0.482s | 0.1302 | 0.5294 | 0.0625 | 1.0000 |
| `bash::light::ls` | 14 | 0.187s | 0.0156 | nan | 0.3571 | 0.9286 |
| `bash::python::interactive_long` | 11 | 1.268s | 0.1380 | 0.0455 | 0.2727 | 1.0000 |
| `bash::python::unittest_discover` | 10 | 3.045s | 0.3898 | 0.5030 | 0.0000 | 0.8000 |
| `bash::python::setup_dev_build` | 9 | 1.323s | 0.0709 | 0.0000 | 0.3333 | 0.8889 |
| `bash::light::mkdir` | 7 | 0.196s | 0.0216 | 0.2143 | 0.0000 | 0.8571 |
| `bash::light::source::chained` | 6 | 0.594s | 0.1699 | nan | 0.1667 | 1.0000 |
| `bash::python::interactive_medium` | 6 | 0.323s | 0.0798 | -0.8857 | 0.1667 | 1.0000 |
| `bash::python::module_other` | 6 | 0.335s | 0.1191 | 0.4286 | 0.0000 | 1.0000 |
| `bash::bg_server` | 5 | 4.219s | 0.8695 | 0.4472 | 0.0000 | 0.6000 |
| `bash::light::unzip::chained` | 5 | 0.175s | 0.5579 | 0.9000 | 0.0000 | 1.0000 |
| `bash::pip::pypi::with_uninstall` | 5 | 20.029s | 0.8005 | -0.4000 | 0.0000 | 0.8000 |

## Method 2 — Accuracy within tolerance 解释

`Acc@(absolute_tol, relative_tol)` 含义: |pred - actual| ≤ max(absolute_tol, relative_tol × actual) 的样本占比。

- `Acc@(5ms, 5%)`: 误差 ≤ 5ms 或 ≤ 5% (取宽松). 严指标。
- `Acc@(10ms, 10%)`: 较宽松, 容忍 10ms or 10%。
- `Acc@(50ms, 20%)`: 宽松，容忍 50ms or 20%。下游调度器关注的实际"够用"度。

## 论文推荐展示 (headline)

> 在 SWE-smith val (n=7907 tool calls) 上, 模型达到
> **log_MAE = 0.054**, **Spearman ρ = 0.880**, **Accuracy@(50ms, 20%) = 89.1%**, **P95 coverage = 93.5%**.

