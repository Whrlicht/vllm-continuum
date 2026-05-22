"""跑 simulator 预测 T2, 与 ground truth 对比.

输出
----
1. 全集指标: log_MAE, MAPE, Spearman, 误差分布, 系统性 over/under prediction 程度
2. 分层指标: 按 T2_actual 分桶 (短尾 vs 长尾)、按 K、按 n_waiting_init
3. 误差 CSV: 每个 round 一行 (含 T2_sim, T2_actual, error 等)
4. M 扫描: 试 M ∈ {10, 15, 20, 25, 30} 找最佳并发槽数
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulator import LichtV2Simulator


def metrics(y_actual: np.ndarray, y_pred: np.ndarray) -> dict:
    """log_MAE, MAPE, Spearman, 系统性 over/under 程度."""
    mask = np.isfinite(y_pred) & np.isfinite(y_actual)
    a = np.asarray(y_actual)[mask]
    p = np.asarray(y_pred)[mask]
    if len(a) == 0:
        return {}
    a_log = np.log1p(a)
    p_log = np.log1p(np.clip(p, 0, None))
    log_err = p_log - a_log  # 正 = over predict
    log_mae = float(np.abs(log_err).mean())
    log_bias = float(log_err.mean())  # >0 over, <0 under
    abs_err = p - a
    mape = float(np.mean(np.abs(abs_err) / np.maximum(a, 0.01)))
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, p)
        spearman = float(rho)
    except ImportError:
        spearman = float('nan')
    p_under = float((p < a * 0.5).mean())  # 严重低估比例
    p_over = float((p > a * 2.0).mean())   # 严重高估比例
    p_within_2x = float(((p >= a * 0.5) & (p <= a * 2.0)).mean())
    return {
        'n': int(mask.sum()),
        'log_MAE': log_mae,
        'log_bias': log_bias,
        'MAPE': mape,
        'Spearman': spearman,
        'p_under_2x': p_under,
        'p_over_2x': p_over,
        'p_within_2x': p_within_2x,
    }


def print_metrics(name: str, m: dict) -> None:
    print(f'\n=== [{name}]  n={m.get("n","?")} ===')
    print(f'  log_MAE        = {m.get("log_MAE", float("nan")):.4f}')
    print(f'  log_bias       = {m.get("log_bias", float("nan")):+.4f}  '
          f'(>0=over predict, <0=under predict)')
    print(f'  MAPE           = {m.get("MAPE", float("nan")):.3f}')
    print(f'  Spearman ρ     = {m.get("Spearman", float("nan")):.4f}')
    print(f'  P(within 2x)   = {m.get("p_within_2x", float("nan")):.3f}')
    print(f'  P(under by 2x) = {m.get("p_under_2x", float("nan")):.3f}')
    print(f'  P(over  by 2x) = {m.get("p_over_2x", float("nan")):.3f}')


def slice_eval(df: pd.DataFrame, sim_col: str = 'T2_sim',
                actual_col: str = 'T2_actual') -> None:
    """分层评估."""
    # 按 actual 大小分桶
    print('\n=== 按真值 T2 分层 ===')
    buckets = [
        ('T2<0.5s (backfill 友好)', df[df[actual_col] < 0.5]),
        ('0.5≤T2<5s (中段)', df[(df[actual_col] >= 0.5) & (df[actual_col] < 5)]),
        ('5≤T2<30s (队尾)', df[(df[actual_col] >= 5) & (df[actual_col] < 30)]),
        ('T2≥30s (重尾)', df[df[actual_col] >= 30]),
    ]
    for name, sub in buckets:
        if len(sub) > 0:
            m = metrics(sub[actual_col].to_numpy(),
                        sub[sim_col].to_numpy())
            m['n'] = len(sub)
            print(f'  {name:30s}  n={len(sub):>4d}  '
                  f'logMAE={m["log_MAE"]:.4f}  bias={m["log_bias"]:+.4f}  '
                  f'Spearman={m["Spearman"]:.3f}')

    # 按 K 分层
    print('\n=== 按 K 分层 ===')
    for K_range in [(1, 1), (2, 3), (4, 7), (8, 15), (16, 999)]:
        sub = df[(df['round_idx'] >= K_range[0]) &
                  (df['round_idx'] <= K_range[1])]
        if len(sub) >= 30:
            m = metrics(sub[actual_col].to_numpy(),
                        sub[sim_col].to_numpy())
            print(f'  K∈[{K_range[0]:>2d},{K_range[1]:>3d}]   n={len(sub):>4d}  '
                  f'logMAE={m["log_MAE"]:.4f}  bias={m["log_bias"]:+.4f}  '
                  f'p50_actual={sub[actual_col].median():.2f}s  '
                  f'p50_sim={sub[sim_col].median():.2f}s')

    # 按 n_waiting_init 分层
    print('\n=== 按 n_waiting_init 分层 (模拟器看到的等待队列长度) ===')
    if 'n_waiting_init' in df.columns:
        for nw_range in [(0, 2), (3, 5), (6, 10), (11, 15), (16, 30)]:
            sub = df[(df['n_waiting_init'] >= nw_range[0]) &
                      (df['n_waiting_init'] <= nw_range[1])]
            if len(sub) >= 30:
                m = metrics(sub[actual_col].to_numpy(),
                            sub[sim_col].to_numpy())
                print(f'  n_wait∈[{nw_range[0]:>2d},{nw_range[1]:>3d}]   '
                      f'n={len(sub):>4d}  logMAE={m["log_MAE"]:.4f}  '
                      f'bias={m["log_bias"]:+.4f}  '
                      f'p50_actual={sub[actual_col].median():.2f}s  '
                      f'p50_sim={sub[sim_col].median():.2f}s')


def run_sweep(df: pd.DataFrame, Ms: list[int]) -> pd.DataFrame:
    """扫 M 找最佳并发槽数."""
    rows = []
    for M in Ms:
        sim = LichtV2Simulator(df, M=M)
        t0 = time.time()
        pred = sim.predict_batch(np.arange(len(df)))
        elapsed = time.time() - t0
        m = metrics(df['T2_s'].to_numpy(), pred)
        m['M'] = M
        m['elapsed_s'] = elapsed
        rows.append(m)
        print(f'M={M:>3d}: log_MAE={m["log_MAE"]:.4f}  bias={m["log_bias"]:+.4f}'
              f'  MAPE={m["MAPE"]:.3f}  Spearman={m["Spearman"]:.3f}'
              f'  P(within 2x)={m["p_within_2x"]:.3f}  '
              f'P(under 2x)={m["p_under_2x"]:.3f}  '
              f'P(over 2x)={m["p_over_2x"]:.3f}  ({elapsed:.1f}s)')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground-truth', default=(
        '/data/whr/vllm-continuum/queue_time/t2/t2_ground_truth.parquet'))
    ap.add_argument('--only-round-ge', type=int, default=1,
                    help='跳过 round_idx 小于此值的 (默认 1, 因为 round 0 是首次 prefill)')
    ap.add_argument('--sweep-M', nargs='+', type=int,
                    default=[10, 15, 20, 25, 30])
    ap.add_argument('--final-M', type=int, default=None,
                    help='跑完 sweep 后用此 M 做完整分层评估; None=自动选 log_MAE 最小')
    ap.add_argument('--out-pred', default=(
        '/data/whr/vllm-continuum/queue_time/sim_predictions.csv'))
    args = ap.parse_args()

    print(f'读取真值: {args.ground_truth}')
    df_all = pd.read_parquet(args.ground_truth)
    print(f'  全集 rows: {len(df_all)}')

    # 模拟器用全集 (round 0 也作为先到达请求影响后续, 不能丢)
    # 评估时只看 round >= only_round_ge
    eval_mask = df_all['round_idx'] >= args.only_round_ge
    print(f'  评估子集 (round >= {args.only_round_ge}): {eval_mask.sum()}')

    # ----- M 扫描 -----
    print(f'\n========== M sweep ==========')
    df_sweep = run_sweep(df_all.loc[eval_mask].reset_index(drop=True)
                          if False else df_all, args.sweep_M)
    # 注意: 模拟器要用全集 (round 0 是先到达上下文); 但 metrics 算的是包括 round 0 的
    # —— 我们再单独算 round >= 1 的指标
    print(f'\n========== 重算 M sweep (仅 round >= {args.only_round_ge}) ==========')
    sims = {}
    for M in args.sweep_M:
        sim = LichtV2Simulator(df_all, M=M)
        pred = sim.predict_batch(np.arange(len(df_all)))
        sims[M] = pred
        sub_pred = pred[eval_mask.to_numpy()]
        sub_actual = df_all.loc[eval_mask, 'T2_s'].to_numpy()
        m = metrics(sub_actual, sub_pred)
        print(f'M={M:>3d}: log_MAE={m["log_MAE"]:.4f}  bias={m["log_bias"]:+.4f}'
              f'  MAPE={m["MAPE"]:.3f}  Spearman={m["Spearman"]:.3f}'
              f'  P(within 2x)={m["p_within_2x"]:.3f}')

    # 选最佳 M
    if args.final_M is None:
        best_M = min(args.sweep_M,
                     key=lambda M: metrics(
                         df_all.loc[eval_mask, 'T2_s'].to_numpy(),
                         sims[M][eval_mask.to_numpy()])['log_MAE'])
    else:
        best_M = args.final_M
    print(f'\n========== 选定 M={best_M} 做完整分层评估 ==========')

    final_pred = sims[best_M]
    out_df = df_all.copy()
    out_df['T2_sim'] = final_pred
    out_df['T2_actual'] = out_df['T2_s']
    out_df['log_err'] = (np.log1p(np.clip(final_pred, 0, None))
                          - np.log1p(out_df['T2_actual']))
    # 加上 init state 信息 (来自 simulator 诊断, 这里粗略用 qs_n_waiting/running)
    out_df['n_waiting_init'] = out_df.get('qs_n_waiting', 0)
    out_df['n_running_init'] = out_df.get('qs_n_running', 0)

    eval_df = out_df.loc[eval_mask].reset_index(drop=True)
    m = metrics(eval_df['T2_actual'].to_numpy(),
                eval_df['T2_sim'].to_numpy())
    print_metrics(f'全集 round >= {args.only_round_ge} (M={best_M})', m)

    slice_eval(eval_df, sim_col='T2_sim', actual_col='T2_actual')

    # 保存
    eval_df[[
        'traj_id', 'round_idx', 'T2_actual', 'T2_sim', 'log_err',
        'prompt_length', 'hit_rate', 'n_waiting_init', 'n_running_init',
        'pf_arrival', 'pf_wtr',
    ]].to_csv(args.out_pred, index=False)
    print(f'\n误差表写入: {args.out_pred}')


if __name__ == '__main__':
    sys.exit(main())
