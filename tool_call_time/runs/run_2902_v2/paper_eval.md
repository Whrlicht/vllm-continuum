# Tool Call Time Predictor — Paper Evaluation

数据集: SWE-smith Trajectories tool split, 199 train / 300 val

模型: LightGBM 4 head (P50 quantile / P95 quantile / P_long_run binary / 2 specialists)

特征: A 命令结构 + B 子参数语义 + C 工件 + E 轨迹观测 + E5 OBSERVATION 状态信号 (共 63 列, 0 instance_id 依赖)

## Method 1 — editor/submit 用 train 95th-percentile 替代 median 充 P95

- 全集 P95 coverage **before**: `0.6458`  (median 充 P95 → 50% 下界)
- 全集 P95 coverage **after**:  `0.9485` ✅

改动: editor/submit 真值近似常数, train 经验分布的 95 分位点比 median 更合理代表"上界"。

## 总体指标 (after Method 1)

| 切片 | n | log_MAE | Spearman | Acc@(5ms,5%) | Acc@(10ms,10%) | Acc@(50ms,20%) | P95 cov |
|---|---|---|---|---|---|---|---|
| all (含 editor/submit) | 7907 | 0.0485 | 0.8828 | 0.7230 | 0.8023 | 0.8949 | 0.9485 |
| editor + submit | 4804 | 0.0001 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.9405 |
| bash only | 3103 | 0.1236 | 0.9105 | 0.2942 | 0.4963 | 0.7322 | 0.9610 |

## Method 3 — 按 Tier 分层

| Tier | n | actual_p50 | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|---|
| constant_submit | 510 | 0.0000s | 0.0000s | 0.0000 | nan | 1.0000 | 1.0000 |
| constant_editor | 4294 | 0.0002s | 0.0002s | 0.0001 | 0.3238 | 1.0000 | 0.9334 |
| light_bash | 1036 | 0.1906s | 0.1988s | 0.0214 | 0.3675 | 0.3793 | 0.9903 |
| normal_bash | 392 | 0.4239s | 0.7261s | 0.1013 | 0.8283 | 0.2398 | 0.9337 |
| heavy_bash | 1675 | 0.4880s | 4.9471s | 0.1920 | 0.8557 | 0.2543 | 0.9493 |

## 桶级 (n ≥ 5, 按 n 降序)

| bucket | n | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |
|---|---|---|---|---|---|---|
| `bash::python::script_repro` | 1470 | 2.530s | 0.1431 | 0.8355 | 0.2748 | 0.9599 |
| `editor::view::with_range` | 1228 | 0.000s | 0.0002 | nan | 1.0000 | 0.9300 |
| `editor::view::no_range` | 805 | 0.000s | 0.0001 | nan | 1.0000 | 0.9528 |
| `editor::str_replace_lt5000` | 660 | 0.000s | 0.0001 | nan | 1.0000 | 0.9470 |
| `editor::str_replace_lt1000` | 521 | 0.000s | 0.0001 | nan | 1.0000 | 0.9597 |
| `submit` | 510 | 0.000s | 0.0000 | nan | 1.0000 | 1.0000 |
| `editor::create_lt1KB` | 407 | 0.000s | 0.0000 | nan | 1.0000 | 0.8747 |
| `bash::light::grep` | 379 | 0.190s | 0.0136 | 0.4616 | 0.4644 | 0.9921 |
| `editor::create_lt10KB` | 365 | 0.000s | 0.0000 | nan | 1.0000 | 0.8658 |
| `bash::find::with_name_pipe` | 286 | 0.202s | 0.0155 | 0.1146 | 0.3846 | 0.9895 |
| `editor::str_replace_lt200` | 276 | 0.000s | 0.0001 | nan | 1.0000 | 0.9891 |
| `bash::light::rm` | 218 | 0.186s | 0.0160 | 0.1612 | 0.3394 | 0.9908 |
| `bash::python::script_other` | 183 | 0.688s | 0.0970 | 0.8011 | 0.2623 | 0.9180 |
| `bash::python::pytest_path` | 52 | 0.743s | 0.0846 | 0.6663 | 0.2308 | 0.9423 |
| `bash::light::grep::chained` | 47 | 0.208s | 0.0270 | 0.4554 | 0.3191 | 0.9574 |
| `bash::pip::pypi::single` | 42 | 8.177s | 0.6917 | 0.6986 | 0.0952 | 0.7857 |
| `bash::env_prefix` | 31 | 0.440s | 0.0769 | 0.5319 | 0.1935 | 1.0000 |
| `bash::pip::editable_local::default` | 29 | 79.115s | 0.3694 | 0.7880 | 0.2069 | 0.9310 |
| `editor::str_replace_ge5000` | 29 | 0.000s | 0.0001 | nan | 1.0000 | 0.8966 |
| `bash::python::module_profiler` | 28 | 0.282s | 0.0364 | 0.1511 | 0.1429 | 1.0000 |
| `bash::find::exec_grep` | 27 | 1.660s | 0.2391 | 0.4805 | 0.0741 | 1.0000 |
| `bash::python::pytest_single_test` | 25 | 0.883s | 0.1169 | 0.7354 | 0.2400 | 0.8800 |
| `bash::python::module_mypy` | 23 | 3.393s | 0.5339 | 0.4585 | 0.1304 | 0.8696 |
| `bash::find::with_name_basic` | 22 | 0.201s | 0.0304 | 0.4069 | 0.1364 | 1.0000 |
| `bash::python::unittest_specific` | 20 | 0.287s | 0.0321 | 0.2767 | 0.2000 | 0.9500 |
| `bash::light::rm::chained` | 19 | 0.230s | 0.0265 | 0.1965 | 0.2632 | 1.0000 |
| `bash::python::pytest_full_discovery` | 19 | 6.919s | 0.2832 | 0.8228 | 0.1053 | 1.0000 |
| `bash::pip::pypi::multi_pkg` | 16 | 11.919s | 0.8357 | -0.0265 | 0.0000 | 0.8750 |
| `bash::python::interactive_short` | 16 | 0.482s | 0.0850 | 0.6412 | 0.3750 | 1.0000 |
| `bash::light::ls` | 14 | 0.187s | 0.0230 | 0.3859 | 0.0714 | 1.0000 |
| `bash::python::interactive_long` | 11 | 1.268s | 0.1075 | 0.2818 | 0.0909 | 1.0000 |
| `bash::python::unittest_discover` | 10 | 3.045s | 0.3328 | 0.6000 | 0.3000 | 0.9000 |
| `bash::python::setup_dev_build` | 9 | 1.323s | 0.0519 | 0.1667 | 0.4444 | 0.8889 |
| `bash::light::mkdir` | 7 | 0.196s | 0.0144 | 0.2143 | 0.4286 | 1.0000 |
| `bash::light::source::chained` | 6 | 0.594s | 0.1400 | 0.2732 | 0.0000 | 1.0000 |
| `bash::python::interactive_medium` | 6 | 0.323s | 0.0675 | -0.0857 | 0.3333 | 1.0000 |
| `bash::python::module_other` | 6 | 0.335s | 0.1323 | 0.4286 | 0.0000 | 1.0000 |
| `bash::bg_server` | 5 | 4.219s | 0.7858 | 0.2236 | 0.0000 | 0.6000 |
| `bash::light::unzip::chained` | 5 | 0.175s | 0.3256 | 0.7000 | 0.0000 | 1.0000 |
| `bash::pip::pypi::with_uninstall` | 5 | 20.029s | 0.7498 | -0.3000 | 0.0000 | 0.8000 |

## Method 2 — Accuracy within tolerance 解释

`Acc@(absolute_tol, relative_tol)` 含义: |pred - actual| ≤ max(absolute_tol, relative_tol × actual) 的样本占比。

- `Acc@(5ms, 5%)`: 误差 ≤ 5ms 或 ≤ 5% (取宽松). 严指标。
- `Acc@(10ms, 10%)`: 较宽松, 容忍 10ms or 10%。
- `Acc@(50ms, 20%)`: 宽松，容忍 50ms or 20%。下游调度器关注的实际"够用"度。

## 论文推荐展示 (headline)

> 在 SWE-smith val (n=7907 tool calls) 上, 模型达到
> **log_MAE = 0.049**, **Spearman ρ = 0.883**, **Accuracy@(50ms, 20%) = 89.5%**, **P95 coverage = 94.9%**.

