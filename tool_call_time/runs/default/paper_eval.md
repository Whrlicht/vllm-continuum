# Tool Call Time Predictor — Paper Evaluation

数据集: SWE-smith Trajectories tool split, 199 train / 300 val

模型: LightGBM 4 head (P50 quantile / P95 quantile / P_long_run binary / 2 specialists)

特征: A 命令结构 + B 子参数语义 + C 工件 + E 轨迹观测 + E5 OBSERVATION 状态信号 (共 63 列, 0 instance_id 依赖)

## Method 1 — editor/submit 用 train 95th-percentile 替代 median 充 P95

- 全集 P95 coverage **before**: `0.6909`  (median 充 P95 → 50% 下界)
- 全集 P95 coverage **after**:  `0.9361` ✅

改动: editor/submit 真值近似常数, train 经验分布的 95 分位点比 median 更合理代表"上界"。

## 总体指标 (after Method 1)

| 切片 | n | log_MAE | Spearman | Acc@(5ms,5%) | Acc@(10ms,10%) | Acc@(50ms,20%) | P95 cov |
|---|---|---|---|---|---|---|---|
| all (含 editor/submit) | 7907 | 0.0607 | 0.8929 | 0.6951 | 0.7698 | 0.8635 | 0.9361 |
| editor + submit | 4804 | 0.0001 | 0.5558 | 1.0000 | 1.0000 | 1.0000 | 0.9257 |
| bash only | 3103 | 0.1546 | 0.8797 | 0.2230 | 0.4135 | 0.6523 | 0.9523 |

## Method 3 — 按 Tier 分层

| Tier | n | actual_p50 | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|---|
| constant_submit | 510 | 0.0000s | 0.0000s | 0.0000 | nan | 1.0000 | 1.0000 |
| constant_editor | 4294 | 0.0002s | 0.0002s | 0.0001 | 0.3958 | 1.0000 | 0.9169 |
| light_bash | 1036 | 0.1906s | 0.1988s | 0.0333 | 0.2851 | 0.3272 | 0.9990 |
| normal_bash | 392 | 0.4239s | 0.7261s | 0.1209 | 0.7541 | 0.1556 | 0.9694 |
| heavy_bash | 1675 | 0.4880s | 4.9471s | 0.2376 | 0.8191 | 0.1743 | 0.9194 |

## 桶级 (n ≥ 5, 按 n 降序)

| bucket | n | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|
| `bash::python::script_repro` | 1470 | 2.530s | 0.1701 | 0.8056 | 0.1898 | 0.9374 |
| `editor::view::with_range` | 1228 | 0.000s | 0.0002 | nan | 1.0000 | 0.9357 |
| `editor::view::no_range` | 805 | 0.000s | 0.0001 | nan | 1.0000 | 0.9516 |
| `editor::str_replace_lt5000` | 660 | 0.000s | 0.0001 | nan | 1.0000 | 0.9379 |
| `editor::str_replace_lt1000` | 521 | 0.000s | 0.0001 | nan | 1.0000 | 0.9578 |
| `submit` | 510 | 0.000s | 0.0000 | nan | 1.0000 | 1.0000 |
| `editor::create_lt1KB` | 407 | 0.000s | 0.0000 | nan | 1.0000 | 0.7862 |
| `bash::light::grep` | 379 | 0.190s | 0.0139 | 0.3705 | 0.4195 | 1.0000 |
| `editor::create_lt10KB` | 365 | 0.000s | 0.0000 | nan | 1.0000 | 0.7726 |
| `bash::find::with_name_pipe` | 286 | 0.202s | 0.0160 | 0.1591 | 0.3566 | 0.9965 |
| `editor::str_replace_lt200` | 276 | 0.000s | 0.0001 | nan | 1.0000 | 0.9891 |
| `bash::light::rm` | 218 | 0.186s | 0.0201 | 0.0295 | 0.2752 | 1.0000 |
| `bash::python::script_other` | 183 | 0.688s | 0.1024 | 0.7609 | 0.1967 | 0.9672 |
| `bash::python::pytest_path` | 52 | 0.743s | 0.0946 | 0.5945 | 0.2308 | 0.9808 |
| `bash::light::grep::chained` | 47 | 0.208s | 0.0483 | 0.2846 | 0.1702 | 1.0000 |
| `bash::pip::pypi::single` | 42 | 8.177s | 0.7406 | 0.5420 | 0.0476 | 0.8810 |
| `bash::env_prefix` | 31 | 0.440s | 0.1649 | 0.4452 | 0.0645 | 1.0000 |
| `bash::pip::editable_local::default` | 29 | 79.115s | 0.6068 | 0.6337 | 0.1034 | 1.0000 |
| `editor::str_replace_ge5000` | 29 | 0.000s | 0.0001 | nan | 1.0000 | 0.8966 |
| `bash::python::module_profiler` | 28 | 0.282s | 0.0398 | 0.2693 | 0.1429 | 1.0000 |
| `bash::find::exec_grep` | 27 | 1.660s | 0.3459 | 0.2354 | 0.0741 | 0.7778 |
| `bash::python::pytest_single_test` | 25 | 0.883s | 0.1434 | 0.5500 | 0.0400 | 0.9600 |
| `bash::python::module_mypy` | 23 | 3.393s | 0.7368 | nan | 0.0000 | 0.6522 |
| `bash::find::with_name_basic` | 22 | 0.201s | 0.1385 | 0.3221 | 0.0455 | 1.0000 |
| `bash::python::unittest_specific` | 20 | 0.287s | 0.0359 | 0.0722 | 0.1500 | 1.0000 |
| `bash::light::rm::chained` | 19 | 0.230s | 0.0712 | 0.2895 | 0.2105 | 1.0000 |
| `bash::python::pytest_full_discovery` | 19 | 6.919s | 0.4847 | 0.4274 | 0.1579 | 0.9474 |
| `bash::pip::pypi::multi_pkg` | 16 | 11.919s | 0.8391 | -0.0618 | 0.1250 | 0.8125 |
| `bash::python::interactive_short` | 16 | 0.482s | 0.1011 | 0.4706 | 0.0625 | 1.0000 |
| `bash::light::ls` | 14 | 0.187s | 0.0627 | 0.3890 | 0.0714 | 1.0000 |
| `bash::python::interactive_long` | 11 | 1.268s | 0.1743 | -0.4273 | 0.1818 | 1.0000 |
| `bash::python::unittest_discover` | 10 | 3.045s | 0.4569 | 0.5636 | 0.0000 | 0.7000 |
| `bash::python::setup_dev_build` | 9 | 1.323s | 0.0930 | 0.1500 | 0.0000 | 1.0000 |
| `bash::light::mkdir` | 7 | 0.196s | 0.0758 | 0.0357 | 0.0000 | 1.0000 |
| `bash::light::source::chained` | 6 | 0.594s | 0.7315 | 0.0286 | 0.0000 | 1.0000 |
| `bash::python::interactive_medium` | 6 | 0.323s | 0.0733 | 0.1429 | 0.0000 | 1.0000 |
| `bash::python::module_other` | 6 | 0.335s | 0.1437 | -0.0857 | 0.1667 | 1.0000 |
| `bash::bg_server` | 5 | 4.219s | 1.1669 | 0.2236 | 0.0000 | 0.6000 |
| `bash::light::unzip::chained` | 5 | 0.175s | 0.2138 | 0.6000 | 0.0000 | 1.0000 |
| `bash::pip::pypi::with_uninstall` | 5 | 20.029s | 0.6759 | 0.3000 | 0.0000 | 0.8000 |

## Method 2 — Accuracy within tolerance 解释

`Acc@(absolute_tol, relative_tol)` 含义: |pred - actual| ≤ max(absolute_tol, relative_tol × actual) 的样本占比。

- `Acc@(5ms, 5%)`: 误差 ≤ 5ms 或 ≤ 5% (取宽松). 严指标。
- `Acc@(10ms, 10%)`: 较宽松, 容忍 10ms or 10%。
- `Acc@(50ms, 20%)`: 宽松，容忍 50ms or 20%。下游调度器关注的实际"够用"度。

## 论文推荐展示 (headline)

> 在 SWE-smith val (n=7907 tool calls) 上, 模型达到
> **log_MAE = 0.061**, **Spearman ρ = 0.893**, **Accuracy@(50ms, 20%) = 86.4%**, **P95 coverage = 93.6%**.

