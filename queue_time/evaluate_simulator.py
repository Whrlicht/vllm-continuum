"""Evaluate LichtV2SimulatorV2 (复刻服务端 backfill) 与真值的对比."""
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
    mask = np.isfinite(y_pred) & np.isfinite(y_actual)
    a = np.asarray(y_actual)[mask]
    p = np.asarray(y_pred)[mask]
    if len(a) == 0:
        return {'n': 0}
    a_log = np.log1p(a)
    p_log = np.log1p(np.clip(p, 0, None))
    log_err = p_log - a_log
    log_mae = float(np.abs(log_err).mean())
    log_bias = float(log_err.mean())
    abs_err = p - a
    mape = float(np.mean(np.abs(abs_err) / np.maximum(a, 0.01)))
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, p)
        spearman = float(rho) if not np.isnan(rho) else 0.0
    except ImportError:
        spearman = float('nan')
    p_under = float((p < a * 0.5).mean())
    p_over = float((p > a * 2.0).mean())
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
        'n_inf': int(np.isinf(y_pred).sum()),
    }


def print_metrics(name: str, m: dict) -> None:
    print(f'\n=== [{name}]  n={m.get("n", "?")} ===')
    print(f'  log_MAE        = {m.get("log_MAE", float("nan")):.4f}')
    print(f'  log_bias       = {m.get("log_bias", float("nan")):+.4f}  '
          f'(>0=over, <0=under)')
    print(f'  MAPE           = {m.get("MAPE", float("nan")):.3f}')
    print(f'  Spearman ρ     = {m.get("Spearman", float("nan")):.4f}')
    print(f'  P(within 2x)   = {m.get("p_within_2x", float("nan")):.3f}')
    print(f'  P(under by 2x) = {m.get("p_under_2x", float("nan")):.3f}')
    print(f'  P(over  by 2x) = {m.get("p_over_2x", float("nan")):.3f}')
    if m.get('n_inf', 0):
        print(f'  ⚠️ inf preds   = {m["n_inf"]} (永远 admit 不上)')


def slice_eval(df: pd.DataFrame) -> None:
    print('\n=== 按真值 T2 分层 ===')
    for name, sub in [
        ('T2<0.5s', df[df['T2_actual'] < 0.5]),
        ('0.5≤T2<5s', df[(df['T2_actual'] >= 0.5) & (df['T2_actual'] < 5)]),
        ('5≤T2<30s', df[(df['T2_actual'] >= 5) & (df['T2_actual'] < 30)]),
        ('T2≥30s', df[df['T2_actual'] >= 30]),
    ]:
        if len(sub) > 0:
            m = metrics(sub['T2_actual'].to_numpy(), sub['T2_sim'].to_numpy())
            print(f'  {name:18s}  n={len(sub):>4d}  '
                  f'p50_actual={sub["T2_actual"].median():.3f}s  '
                  f'p50_sim={sub["T2_sim"].median():.3f}s  '
                  f'logMAE={m["log_MAE"]:.4f}  bias={m["log_bias"]:+.4f}  '
                  f'Spearman={m["Spearman"]:.3f}')

    print('\n=== 按 K 分层 ===')
    for K_range in [(1, 1), (2, 3), (4, 7), (8, 15), (16, 999)]:
        sub = df[(df['round_idx'] >= K_range[0]) &
                  (df['round_idx'] <= K_range[1])]
        if len(sub) >= 30:
            m = metrics(sub['T2_actual'].to_numpy(), sub['T2_sim'].to_numpy())
            print(f'  K∈[{K_range[0]:>2d},{K_range[1]:>3d}]   n={len(sub):>4d}  '
                  f'p50_actual={sub["T2_actual"].median():.3f}s  '
                  f'p50_sim={sub["T2_sim"].median():.3f}s  '
                  f'logMAE={m["log_MAE"]:.4f}  bias={m["log_bias"]:+.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ground-truth', default=(
        '/data/whr/vllm-continuum/queue_time/t2/t2_ground_truth.parquet'))
    ap.add_argument('--only-round-ge', type=int, default=1)
    ap.add_argument('--out-pred', default=(
        '/data/whr/vllm-continuum/queue_time/sim_v2_predictions.csv'))
    args = ap.parse_args()

    print(f'读取真值: {args.ground_truth}')
    df_all = pd.read_parquet(args.ground_truth)
    print(f'  全集 rows: {len(df_all)}')
    eval_mask = (df_all['round_idx'] >= args.only_round_ge).to_numpy()
    print(f'  评估子集 (round >= {args.only_round_ge}): {eval_mask.sum()}')

    print(f'\n初始化 simulator (严格复刻 LICHTV2 future_free timeline)...')
    sim = LichtV2Simulator(df_all)
    print(f'  实例参数: NUM_GPU_BLOCKS={16853}  block_size={16}  '
          f'chunk_size={5242}  MAX_NUM_SEQS={256}')

    t0 = time.time()
    print(f'\n批量预测 {len(df_all)} 个 round...')
    pred = sim.predict_batch(np.arange(len(df_all)))
    elapsed = time.time() - t0
    print(f'  耗时 {elapsed:.1f}s ({elapsed/len(df_all)*1000:.2f} ms/round)')

    out_df = df_all.copy()
    out_df['T2_sim'] = pred
    out_df['T2_actual'] = out_df['T2_s']
    out_df['log_err'] = (np.log1p(np.clip(pred, 0, None))
                          - np.log1p(out_df['T2_actual']))

    eval_df = out_df.loc[eval_mask].reset_index(drop=True)
    m = metrics(eval_df['T2_actual'].to_numpy(),
                eval_df['T2_sim'].to_numpy())
    print_metrics('全集 round >= 1', m)
    slice_eval(eval_df)

    eval_df[[
        'traj_id', 'round_idx', 'T2_actual', 'T2_sim', 'log_err',
        'prompt_length', 'hit_length', 'hit_rate',
        'qs_n_waiting', 'qs_n_running', 'qs_sum_prompt_ahead',
    ]].to_csv(args.out_pred, index=False)
    print(f'\n误差表写入: {args.out_pred}')


if __name__ == '__main__':
    sys.exit(main())
