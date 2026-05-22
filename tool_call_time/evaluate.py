"""评估指标: log-MAE / MAPE / Spearman ρ / P95 calibration / timeout AUC。

按桶报, 不要看总平均 (会被 light 桶虚高)。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def _safe_log1p(x):
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


def per_sample_metrics(df: pd.DataFrame, point_col: str = 'pred_p50'
                       ) -> pd.DataFrame:
    """每行算 log AE / abs error / pct error。"""
    out = df.copy()
    actual = out['_actual_t'].to_numpy()
    pred = out[point_col].to_numpy()
    out['_log_ae'] = np.abs(_safe_log1p(pred) - _safe_log1p(actual))
    out['_abs_err'] = np.abs(pred - actual)
    safe_actual = np.maximum(actual, 0.05)  # 避免除 0
    out['_pct_err'] = out['_abs_err'] / safe_actual
    return out


def overall_metrics(df: pd.DataFrame, point_col: str = 'pred_p50') -> dict:
    df = per_sample_metrics(df, point_col)
    out = {
        'n': len(df),
        'log_mae': float(df['_log_ae'].mean()),
        'log_mae_p50': float(df['_log_ae'].median()),
        'log_mae_p95': float(df['_log_ae'].quantile(0.95)),
        'mape': float(df['_pct_err'].mean()),
    }
    # Spearman
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(df['_actual_t'], df[point_col])
        out['spearman_rho'] = float(rho) if rho == rho else float('nan')
    except Exception:
        out['spearman_rho'] = float('nan')
    # P95 calibration: 实际 t 落在预测 P95 之下的比例
    if 'pred_p95' in df.columns:
        out['p95_coverage'] = float((df['_actual_t']
                                      <= df['pred_p95']).mean())
    # timeout 指标: 区分 raw model / effective (含 D3 P95 兜底)
    if df['_label_is_timeout'].sum() > 0:
        for prob_col, suffix in [
            ('pred_p_timeout_raw', '_raw'),
            ('pred_p_timeout', '_eff'),
        ]:
            if prob_col not in df.columns:
                continue
            try:
                from sklearn.metrics import roc_auc_score
                out[f'timeout_auc{suffix}'] = float(roc_auc_score(
                    df['_label_is_timeout'], df[prob_col]))
            except Exception:
                pass
            for thresh, label in [(0.5, '0p5')]:
                pred_to = (df[prob_col] >= thresh).astype(int)
                tp = int(((pred_to == 1)
                          & (df['_label_is_timeout'] == 1)).sum())
                fp = int(((pred_to == 1)
                          & (df['_label_is_timeout'] == 0)).sum())
                fn = int(((pred_to == 0)
                          & (df['_label_is_timeout'] == 1)).sum())
                rec = tp / max(1, tp + fn)
                prec = tp / max(1, tp + fp)
                out[f'timeout_recall_at_{label}{suffix}'] = float(rec)
                out[f'timeout_precision_at_{label}{suffix}'] = float(prec)
                out[f'timeout_fp_at_{label}{suffix}'] = fp
    return out


def per_bucket_metrics(df: pd.DataFrame, point_col: str = 'pred_p50',
                       min_n: int = 3) -> pd.DataFrame:
    df = per_sample_metrics(df, point_col)
    rows = []
    for bucket, sub in df.groupby('bucket'):
        if len(sub) < min_n:
            continue
        rows.append({
            'bucket': bucket,
            'n': len(sub),
            'p50_actual': float(sub['_actual_t'].median()),
            'mean_actual': float(sub['_actual_t'].mean()),
            'log_mae': float(sub['_log_ae'].mean()),
            'mape': float(sub['_pct_err'].mean()),
            'p95_coverage': (float((sub['_actual_t'] <= sub['pred_p95']).mean())
                             if 'pred_p95' in sub.columns else float('nan')),
        })
    return pd.DataFrame(rows).sort_values('n', ascending=False)


def print_overall(metrics: dict, label: str = '') -> None:
    print(f'\n=== overall metrics{(" [" + label + "]") if label else ""} ===')
    print(f'  n           = {metrics["n"]}')
    print(f'  log_MAE     = {metrics["log_mae"]:.4f}  '
          f'(p50={metrics["log_mae_p50"]:.4f}, '
          f'p95={metrics["log_mae_p95"]:.4f})')
    print(f'  MAPE        = {metrics["mape"]:.4f}')
    print(f'  Spearman ρ  = {metrics.get("spearman_rho", float("nan")):.4f}')
    if 'p95_coverage' in metrics:
        print(f'  P95 coverage = {metrics["p95_coverage"]:.4f}  '
              f'(target ≥ 0.95)')
    for tag in ('_raw', '_eff'):
        auc_key = f'timeout_auc{tag}'
        if auc_key in metrics:
            label = 'raw model' if tag == '_raw' else 'eff (含 P95 兜底)'
            print(f'  timeout [{label}]: '
                  f'AUC={metrics[auc_key]:.4f}, '
                  f'Recall@0.5='
                  f'{metrics.get(f"timeout_recall_at_0p5{tag}", float("nan")):.4f}, '
                  f'Precision@0.5='
                  f'{metrics.get(f"timeout_precision_at_0p5{tag}", float("nan")):.4f}, '
                  f'FP={metrics.get(f"timeout_fp_at_0p5{tag}", 0)}')


def print_per_bucket(table: pd.DataFrame, label: str = '',
                     top: int = 30) -> None:
    print(f'\n=== per-bucket metrics{(" [" + label + "]") if label else ""} '
          f'(top {top} by n) ===')
    print(f'  {"bucket":52s} {"n":>5s} {"p50_t":>8s} {"mean_t":>8s} '
          f'{"logMAE":>7s} {"MAPE":>7s} {"P95cov":>7s}')
    print('-' * 100)
    for _, row in table.head(top).iterrows():
        print(f'  {row["bucket"]:52s} {int(row["n"]):>5d} '
              f'{row["p50_actual"]:>8.4f} {row["mean_actual"]:>8.4f} '
              f'{row["log_mae"]:>7.4f} {row["mape"]:>7.4f} '
              f'{row["p95_coverage"]:>7.4f}')
