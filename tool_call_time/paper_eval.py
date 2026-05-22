"""论文用的验证脚本: 在已训好的 default 模型上, 应用 Method 1/2/3 三种修正,
并把所有指标写入 markdown 报告。

Method 1: editor/submit 桶 P95 用训练集真 P95 (替代 median)。
          这样 P95 cov 不再被定义性的 50% 拉低。
Method 2: 加 accuracy@tolerance 指标 (绝对+相对容忍度)。
          论文里给读者一个直观的"X% 命中"数字。
Method 3: 按 tier 分层 (constant 类 / variable 类), 不再硬算全集加权平均。

输出文件: runs/default/paper_eval.md (markdown 表格)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bucket import classify
from dataset import (
    build_samples,
    iter_tool_calls,
    load_trajectories,
)
from train import Predictor

DEFAULT_TRAIN = ('/data/whr/vllm-continuum/trace_data/'
                 'swe_bench_sample_2902_tool_clean_with_timings.json')
DEFAULT_VAL = ('/data/whr/vllm-continuum/trace_data/'
               'swe_bench_sample_300_tool_clean_with_timings.json')
DEFAULT_RUN = Path(__file__).resolve().parent / 'runs' / 'default'

# ---------------------------------------------------------------------------
# Method 1: train P95 by bucket
# ---------------------------------------------------------------------------

def compute_bucket_p95_from_train(trajectories: list[dict],
                                   q: float = 0.95) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for tj in trajectories:
        for _, tc, t, _ in iter_tool_calls(tj):
            grouped.setdefault(classify(tc), []).append(t)
    out: dict[str, float] = {}
    for b, ts in grouped.items():
        if len(ts) >= 3:
            out[b] = float(np.quantile(ts, q))
        elif ts:
            out[b] = float(max(ts))
    return out


def apply_method1_override(df: pd.DataFrame,
                           p95_by_bucket: dict[str, float],
                           override_families=('editor', 'submit')
                           ) -> pd.DataFrame:
    df = df.copy()
    mask = df['_family'].isin(override_families)
    df.loc[mask, 'pred_p95'] = (
        df.loc[mask, 'bucket'].map(p95_by_bucket)
        .fillna(df.loc[mask, 'pred_p95']))
    # P95 >= P50 不变量
    df['pred_p95'] = np.maximum(df['pred_p95'], df['pred_p50'])
    return df


# ---------------------------------------------------------------------------
# Method 2: accuracy@tolerance
# ---------------------------------------------------------------------------

def accuracy_within(df: pd.DataFrame, abs_tol_s: float, rel_tol: float,
                    point_col: str = 'pred_p50') -> float:
    actual = df['_actual_t'].to_numpy()
    pred = df[point_col].to_numpy()
    err = np.abs(pred - actual)
    tol = np.maximum(abs_tol_s, rel_tol * actual)
    return float((err <= tol).mean())


# ---------------------------------------------------------------------------
# Method 3: tier 分层
# ---------------------------------------------------------------------------

_HEAVY_BASH_PREFIXES = (
    'bash::pip::', 'bash::conda::', 'bash::apt::', 'bash::bg_server',
)
_HEAVY_BASH_BUCKETS = {
    'bash::python::script_repro',
    'bash::python::module_mypy',
    'bash::python::pytest_full_discovery',
    'bash::python::unittest_discover',
    'bash::python::module_other',
    'bash::find::exec_grep',
}


def tier_of(bucket: str) -> str:
    if bucket == 'submit':
        return 'constant_submit'
    if bucket.startswith('editor::'):
        return 'constant_editor'
    if bucket.startswith('bash::light::') or bucket == 'bash::cd_only':
        return 'light_bash'
    if bucket.startswith('bash::find::') and 'exec_grep' not in bucket:
        return 'light_bash'
    if bucket.startswith('bash::git::'):
        return 'light_bash'
    if bucket in _HEAVY_BASH_BUCKETS or bucket.startswith(_HEAVY_BASH_PREFIXES):
        return 'heavy_bash'
    return 'normal_bash'


# ---------------------------------------------------------------------------
# 总体指标
# ---------------------------------------------------------------------------

def evaluate_subset(df: pd.DataFrame, label: str) -> dict | None:
    if df.empty:
        return None
    actual = df['_actual_t'].to_numpy()
    pred = df['pred_p50'].to_numpy()
    log_a = np.log1p(actual)
    log_p = np.log1p(np.clip(pred, 0.0, None))
    log_mae = float(np.abs(log_a - log_p).mean())

    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(actual, pred)
        spearman = float(rho) if rho == rho else float('nan')
    except Exception:
        spearman = float('nan')

    return {
        'label': label,
        'n': len(df),
        'p50_actual': float(np.median(actual)),
        'mean_actual': float(np.mean(actual)),
        'log_mae': log_mae,
        'spearman': spearman,
        'acc_5ms_5pct': accuracy_within(df, 0.005, 0.05),
        'acc_10ms_10pct': accuracy_within(df, 0.010, 0.10),
        'acc_50ms_20pct': accuracy_within(df, 0.050, 0.20),
        'p95_cov': float((df['_actual_t'] <= df['pred_p95']).mean()),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print(f'加载模型: {DEFAULT_RUN}', flush=True)
    pred = Predictor.load(DEFAULT_RUN)

    print(f'加载 train: {DEFAULT_TRAIN}', flush=True)
    train_traj = load_trajectories(DEFAULT_TRAIN)
    print(f'加载 val:   {DEFAULT_VAL}', flush=True)
    val_traj = load_trajectories(DEFAULT_VAL)

    log_means = pred.bundle['log_mean_by_bucket']
    val_df = build_samples(val_traj, log_means, 60.0)
    print(f'val samples: {len(val_df)}', flush=True)

    print('predict...', flush=True)
    v_raw = pred.predict_df(val_df)

    # Method 1: 用 train P95 替代 editor/submit 的 P95
    print('Method 1: 用 train P95 by bucket 替代 editor/submit median...',
          flush=True)
    p95_by_bucket = compute_bucket_p95_from_train(train_traj, q=0.95)
    print(f'  按桶算出 train P95: {len(p95_by_bucket)} 个桶', flush=True)
    v_fixed = apply_method1_override(v_raw, p95_by_bucket)

    # 三种切片
    cut_all = v_fixed
    cut_editor_submit = v_fixed[v_fixed['_family'].isin(['editor', 'submit'])]
    cut_bash = v_fixed[v_fixed['_family'] == 'bash']

    overall = [
        evaluate_subset(cut_all, 'all (含 editor/submit)'),
        evaluate_subset(cut_editor_submit, 'editor + submit'),
        evaluate_subset(cut_bash, 'bash only'),
    ]
    overall = [r for r in overall if r is not None]

    # Method 1 前后 P95 cov 对比
    cov_before = float((v_raw['_actual_t'] <= v_raw['pred_p95']).mean())
    cov_after = float((v_fixed['_actual_t'] <= v_fixed['pred_p95']).mean())

    # Method 3: tier 分层
    v_fixed = v_fixed.copy()
    v_fixed['_tier'] = v_fixed['bucket'].apply(tier_of)
    tiers = []
    for tier_name in ('constant_submit', 'constant_editor', 'light_bash',
                       'normal_bash', 'heavy_bash'):
        sub = v_fixed[v_fixed['_tier'] == tier_name]
        r = evaluate_subset(sub, tier_name)
        if r:
            tiers.append(r)

    # 桶级 (n >= 5)
    bucket_rows = []
    for bucket, sub in v_fixed.groupby('bucket'):
        if len(sub) >= 5:
            r = evaluate_subset(sub, bucket)
            if r:
                bucket_rows.append(r)
    bucket_rows.sort(key=lambda x: -x['n'])

    # 写报告
    out_path = Path(DEFAULT_RUN) / 'paper_eval.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('# Tool Call Time Predictor — Paper Evaluation\n\n')
        f.write('数据集: SWE-smith Trajectories tool split, 199 train / 300 val\n\n')
        f.write('模型: LightGBM 4 head (P50 quantile / P95 quantile / P_long_run binary / 2 specialists)\n\n')
        f.write('特征: A 命令结构 + B 子参数语义 + C 工件 + E 轨迹观测 + E5 OBSERVATION 状态信号 (共 63 列, 0 instance_id 依赖)\n\n')

        f.write('## Method 1 — editor/submit 用 train 95th-percentile 替代 median 充 P95\n\n')
        f.write(f'- 全集 P95 coverage **before**: `{cov_before:.4f}`  (median 充 P95 → 50% 下界)\n')
        f.write(f'- 全集 P95 coverage **after**:  `{cov_after:.4f}` ✅\n\n')
        f.write('改动: editor/submit 真值近似常数, train 经验分布的 95 分位点比 median 更合理代表"上界"。\n\n')

        f.write('## 总体指标 (after Method 1)\n\n')
        f.write('| 切片 | n | log_MAE | Spearman | Acc@(5ms,5%) | Acc@(10ms,10%) | Acc@(50ms,20%) | P95 cov |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for r in overall:
            f.write(f"| {r['label']} | {r['n']} | {r['log_mae']:.4f} | "
                    f"{r['spearman']:.4f} | {r['acc_5ms_5pct']:.4f} | "
                    f"{r['acc_10ms_10pct']:.4f} | {r['acc_50ms_20pct']:.4f} | "
                    f"{r['p95_cov']:.4f} |\n")
        f.write('\n')

        f.write('## Method 3 — 按 Tier 分层\n\n')
        f.write('| Tier | n | actual_p50 | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |\n')
        f.write('|---|---|---|---|---|---|---|---|\n')
        for r in tiers:
            f.write(f"| {r['label']} | {r['n']} | {r['p50_actual']:.4f}s | "
                    f"{r['mean_actual']:.4f}s | {r['log_mae']:.4f} | "
                    f"{r['spearman']:.4f} | {r['acc_5ms_5pct']:.4f} | "
                    f"{r['p95_cov']:.4f} |\n")
        f.write('\n')

        f.write('## 桶级 (n ≥ 5, 按 n 降序)\n\n')
        f.write('| bucket | n | actual_mean | log_MAE | Spearman | Acc@(5ms,5%) | P95 cov |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for r in bucket_rows:
            f.write(f"| `{r['label']}` | {r['n']} | {r['mean_actual']:.3f}s | "
                    f"{r['log_mae']:.4f} | {r['spearman']:.4f} | "
                    f"{r['acc_5ms_5pct']:.4f} | {r['p95_cov']:.4f} |\n")
        f.write('\n')

        # Method 2 解释
        f.write('## Method 2 — Accuracy within tolerance 解释\n\n')
        f.write('`Acc@(absolute_tol, relative_tol)` 含义: |pred - actual| ≤ max(absolute_tol, relative_tol × actual) 的样本占比。\n\n')
        f.write('- `Acc@(5ms, 5%)`: 误差 ≤ 5ms 或 ≤ 5% (取宽松). 严指标。\n')
        f.write('- `Acc@(10ms, 10%)`: 较宽松, 容忍 10ms or 10%。\n')
        f.write('- `Acc@(50ms, 20%)`: 宽松，容忍 50ms or 20%。下游调度器关注的实际"够用"度。\n\n')

        f.write('## 论文推荐展示 (headline)\n\n')
        f.write('> 在 SWE-smith val (n=7907 tool calls) 上, 模型达到\n')
        f.write(f'> **log_MAE = {overall[0]["log_mae"]:.3f}**, '
                f'**Spearman ρ = {overall[0]["spearman"]:.3f}**, '
                f'**Accuracy@(50ms, 20%) = {overall[0]["acc_50ms_20pct"]:.1%}**, '
                f'**P95 coverage = {overall[0]["p95_cov"]:.1%}**.\n\n')

    print(f'\n报告写入: {out_path}', flush=True)
    print(f'  全集 P95 cov: {cov_before:.4f} -> {cov_after:.4f}', flush=True)
    print(f'  bash log_MAE: {[r for r in overall if r["label"]=="bash only"][0]["log_mae"]:.4f}', flush=True)
    print(f'  全集 Acc@(50ms,20%): {overall[0]["acc_50ms_20pct"]:.4f}', flush=True)
    print(f'  全集 Acc@(5ms,5%):   {overall[0]["acc_5ms_5pct"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
