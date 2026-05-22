# Tool Call Time Predictor — Paper Evaluation

数据集: SWE-smith Trajectories tool split, 199 train / 300 val

模型: LightGBM 4 head (P50 quantile / P95 quantile / P_long_run binary / 2 specialists)

特征: A 命令结构 + B 子参数语义 + C 工件 + E 轨迹观测 + E5 OBSERVATION 状态信号 (共 63 列, 0 instance_id 依赖)

## Method 1 — editor/submit 用 train 95th-percentile 替代 median 充 P95

- 全集 P95 coverage **before**: `0.6416`  (median 充 P95 → 50% 下界)
- 全集 P95 coverage **after**:  `0.9444` ✅

改动: editor/submit 真值近似常数, train 经验分布的 95 分位点比 median 更合理代表"上界"。

## 总体指标 (after Method 1)

| 切片 | n | log_MAE | Spearman | Acc@(5ms,5%) | Acc@(10ms,10%) | Acc@(50ms,20%) | P95 cov |
|---|---|---|---|---|---|---|---|
| all (含 editor/submit) | 7907 | 0.0490 | 0.8828 | 0.7251 | 0.8041 | 0.8993 | 0.9444 |
| editor + submit | 4804 | 0.0001 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.9405 |
| bash only | 3103 | 0.1247 | 0.9136 | 0.2994 | 0.5008 | 0.7435 | 0.9504 |

## Method 3 — 按 Tier 分层

| Tier | n | actual_p50 | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|---|
| constant_submit | 510 | 0.0000s | 0.0000s | 0.0000 | nan | 1.0000 | 1.0000 |
| constant_editor | 4294 | 0.0002s | 0.0002s | 0.0001 | 0.3238 | 1.0000 | 0.9334 |
| light_bash | 1036 | 0.1906s | 0.1988s | 0.0202 | 0.3912 | 0.3996 | 0.9846 |
| normal_bash | 392 | 0.4239s | 0.7261s | 0.0999 | 0.8368 | 0.2423 | 0.8954 |
| heavy_bash | 1675 | 0.4880s | 4.9471s | 0.1952 | 0.8548 | 0.2507 | 0.9421 |

## 桶级 (n ≥ 5, 按 n 降序)

| bucket | n | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|
| `bash::python::script_repro` | 1470 | 2.530s | 0.1436 | 0.8359 | 0.2707 | 0.9497 |
| `editor::view::with_range` | 1228 | 0.000s | 0.0002 | nan | 1.0000 | 0.9300 |
| `editor::view::no_range` | 805 | 0.000s | 0.0001 | nan | 1.0000 | 0.9528 |
| `editor::str_replace_lt5000` | 660 | 0.000s | 0.0001 | nan | 1.0000 | 0.9470 |
| `editor::str_replace_lt1000` | 521 | 0.000s | 0.0001 | nan | 1.0000 | 0.9597 |
| `submit` | 510 | 0.000s | 0.0000 | nan | 1.0000 | 1.0000 |
| `editor::create_lt1KB` | 407 | 0.000s | 0.0000 | nan | 1.0000 | 0.8747 |
| `bash::light::grep` | 379 | 0.190s | 0.0128 | 0.4677 | 0.4749 | 0.9842 |
| `editor::create_lt10KB` | 365 | 0.000s | 0.0000 | nan | 1.0000 | 0.8658 |
| `bash::find::with_name_pipe` | 286 | 0.202s | 0.0149 | 0.1813 | 0.3951 | 0.9895 |
| `editor::str_replace_lt200` | 276 | 0.000s | 0.0001 | nan | 1.0000 | 0.9891 |
| `bash::light::rm` | 218 | 0.186s | 0.0141 | 0.1559 | 0.3624 | 0.9908 |
| `bash::python::script_other` | 183 | 0.688s | 0.0963 | 0.8174 | 0.2842 | 0.8907 |
| `bash::python::pytest_path` | 52 | 0.743s | 0.0863 | 0.6899 | 0.1923 | 0.9231 |
| `bash::light::grep::chained` | 47 | 0.208s | 0.0245 | 0.5031 | 0.3404 | 0.9362 |
| `bash::pip::pypi::single` | 42 | 8.177s | 0.7675 | 0.6444 | 0.1667 | 0.8571 |
| `bash::env_prefix` | 31 | 0.440s | 0.0618 | 0.5653 | 0.2258 | 0.9032 |
| `bash::pip::editable_local::default` | 29 | 79.115s | 0.3890 | 0.7762 | 0.1379 | 0.9655 |
| `editor::str_replace_ge5000` | 29 | 0.000s | 0.0001 | nan | 1.0000 | 0.8966 |
| `bash::python::module_profiler` | 28 | 0.282s | 0.0280 | 0.4191 | 0.2143 | 0.8571 |
| `bash::find::exec_grep` | 27 | 1.660s | 0.2529 | 0.3371 | 0.0741 | 1.0000 |
| `bash::python::pytest_single_test` | 25 | 0.883s | 0.1206 | 0.6569 | 0.1600 | 0.7600 |
| `bash::python::module_mypy` | 23 | 3.393s | 0.6025 | 0.4348 | 0.0870 | 0.8696 |
| `bash::find::with_name_basic` | 22 | 0.201s | 0.0281 | 0.3289 | 0.3182 | 1.0000 |
| `bash::python::unittest_specific` | 20 | 0.287s | 0.0289 | 0.3940 | 0.3000 | 0.9500 |
| `bash::light::rm::chained` | 19 | 0.230s | 0.0265 | 0.3684 | 0.4211 | 1.0000 |
| `bash::python::pytest_full_discovery` | 19 | 6.919s | 0.2666 | 0.8860 | 0.1053 | 0.9474 |
| `bash::pip::pypi::multi_pkg` | 16 | 11.919s | 0.8211 | -0.1441 | 0.0625 | 0.8750 |
| `bash::python::interactive_short` | 16 | 0.482s | 0.0829 | 0.6794 | 0.3125 | 1.0000 |
| `bash::light::ls` | 14 | 0.187s | 0.0158 | 0.1033 | 0.3571 | 0.9286 |
| `bash::python::interactive_long` | 11 | 1.268s | 0.1434 | -0.0182 | 0.0909 | 1.0000 |
| `bash::python::unittest_discover` | 10 | 3.045s | 0.3515 | 0.6018 | 0.0000 | 0.9000 |
| `bash::python::setup_dev_build` | 9 | 1.323s | 0.0513 | -0.2667 | 0.2222 | 1.0000 |
| `bash::light::mkdir` | 7 | 0.196s | 0.0253 | 0.0000 | 0.0000 | 0.8571 |
| `bash::light::source::chained` | 6 | 0.594s | 0.1437 | nan | 0.1667 | 1.0000 |
| `bash::python::interactive_medium` | 6 | 0.323s | 0.0862 | -0.0286 | 0.3333 | 1.0000 |
| `bash::python::module_other` | 6 | 0.335s | 0.1060 | 0.6000 | 0.1667 | 1.0000 |
| `bash::bg_server` | 5 | 4.219s | 0.9362 | 0.3354 | 0.0000 | 0.8000 |
| `bash::light::unzip::chained` | 5 | 0.175s | 0.3718 | 0.6000 | 0.0000 | 1.0000 |
| `bash::pip::pypi::with_uninstall` | 5 | 20.029s | 0.7213 | nan | 0.2000 | 0.8000 |

## Method 2 — Accuracy within tolerance 解释

`Acc@(absolute_tol, relative_tol)` 含义: |pred - actual| ≤ max(absolute_tol, relative_tol × actual) 的样本占比。

- `Acc@(5ms, 5%)`: 误差 ≤ 5ms 或 ≤ 5% (取宽松). 严指标。
- `Acc@(10ms, 10%)`: 较宽松, 容忍 10ms or 10%。
- `Acc@(50ms, 20%)`: 宽松，容忍 50ms or 20%。下游调度器关注的实际"够用"度。

## 论文推荐展示 (headline)

> 在 SWE-smith val (n=7907 tool calls) 上, 模型达到
> **log_MAE = 0.049**, **Spearman ρ = 0.883**, **Accuracy@(50ms, 20%) = 89.9%**, **P95 coverage = 94.4%**.

