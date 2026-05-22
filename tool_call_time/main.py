"""端到端: 加载 199 train / 300 val -> 训练 -> 评估 -> 报告。

用法:
    python3 main.py train     # 训练 + 保存 + 评估
    python3 main.py evaluate  # 仅用已有 bundle 在 val 上评估
    python3 main.py baseline  # T0 查表 baseline (无 ML)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 同目录导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (
    build_samples, compute_global_bucket_log_means,
    compute_global_bucket_medians, load_trajectories,
)
from evaluate import (
    overall_metrics, per_bucket_metrics, print_overall, print_per_bucket,
)
from train import Predictor, TrainConfig, train_full_bundle

DEFAULT_TRAIN = ('/data/whr/vllm-continuum/trace_data/'
                 'swe_bench_sample_2902_tool_clean_with_timings.json')
DEFAULT_VAL = ('/data/whr/vllm-continuum/trace_data/'
               'swe_bench_sample_300_tool_clean_with_timings.json')
DEFAULT_RUN_DIR = Path(__file__).resolve().parent / 'runs' / 'default'


def _load_and_build(train_path: str, val_path: str, threshold_s: float
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = time.time()
    print(f'loading train: {train_path}', flush=True)
    train_traj = load_trajectories(train_path)
    print(f'  loaded {len(train_traj)} train trajs in {time.time()-t0:.1f}s',
          flush=True)

    t1 = time.time()
    print(f'loading val:   {val_path}', flush=True)
    val_traj = load_trajectories(val_path)
    print(f'  loaded {len(val_traj)} val trajs in {time.time()-t1:.1f}s',
          flush=True)

    t1 = time.time()
    log_means = compute_global_bucket_log_means(train_traj)
    print(f'  global log-mean buckets in train: {len(log_means)} '
          f'({time.time()-t1:.1f}s)', flush=True)

    t1 = time.time()
    print('building train samples (causal rolling)...', flush=True)
    train_df = build_samples(train_traj, log_means, threshold_s)
    print(f'  train samples = {len(train_df)} ({time.time()-t1:.1f}s)',
          flush=True)

    t1 = time.time()
    print('building val samples (causal rolling)...', flush=True)
    val_df = build_samples(val_traj, log_means, threshold_s)
    print(f'  val samples = {len(val_df)} ({time.time()-t1:.1f}s)', flush=True)

    by_family = train_df['_family'].value_counts().to_dict()
    print(f'  train by family: {by_family}', flush=True)
    print(f'  data prep total: {time.time()-t0:.1f}s', flush=True)

    return train_df, val_df, log_means


def _baseline_predict(df: pd.DataFrame,
                      bucket_medians: dict[str, float],
                      global_median: float) -> pd.DataFrame:
    df = df.copy()
    df['pred_p50'] = df['bucket'].map(bucket_medians).fillna(global_median)
    df['pred_p95'] = df['pred_p50']  # baseline 不预测 P95
    df['pred_p_timeout'] = 0.0
    df['pred_expected'] = df['pred_p50']
    return df


def cmd_baseline(args):
    train_df, val_df, _ = _load_and_build(args.train, args.val, args.threshold)
    train_traj = load_trajectories(args.train)
    medians = compute_global_bucket_medians(train_traj)
    global_median = float(train_df['_actual_t'].median())

    val_pred = _baseline_predict(val_df, medians, global_median)
    print_overall(overall_metrics(val_pred), 'baseline (val)')
    print_overall(overall_metrics(val_pred[val_pred['_family'] == 'bash']),
                  'baseline (val, bash only)')
    table = per_bucket_metrics(val_pred)
    print_per_bucket(table, 'baseline (val)')


def cmd_train(args):
    t0 = time.time()
    train_df, val_df, _ = _load_and_build(args.train, args.val, args.threshold)

    out_dir = Path(args.run_dir)
    print(f'\ntraining LightGBM bundle -> {out_dir} ...', flush=True)
    cfg = TrainConfig(
        timeout_threshold_s=args.threshold,
        num_iter=args.num_iter,
        num_leaves=args.num_leaves,
        learning_rate=args.lr,
        min_child_samples=args.min_child,
        log_period=args.log_period,
        early_stopping=args.early_stopping,
    )
    t1 = time.time()
    bundle = train_full_bundle(train_df, out_dir, cfg, val_df=val_df)
    print(f'\nbundle saved. feature_cols={len(bundle["feature_cols"])}, '
          f'p_timeout={bundle["p_timeout_trained"]}, '
          f'training time={time.time()-t1:.1f}s', flush=True)

    pred = Predictor.load(out_dir)

    print('\nevaluating on TRAIN (sanity, expected to be optimistic)...',
          flush=True)
    train_pred = pred.predict_df(train_df)
    print_overall(overall_metrics(train_pred), 'ML (train)')

    print('\nevaluating on VAL ...', flush=True)
    val_pred = pred.predict_df(val_df)
    print_overall(overall_metrics(val_pred), 'ML (val, all)')
    print_overall(overall_metrics(val_pred[val_pred['_family'] == 'bash']),
                  'ML (val, bash only)')

    print_per_bucket(per_bucket_metrics(val_pred), 'ML (val)')

    # 同时把 baseline 跑一遍, 方便对照
    print('\n--- baseline 对照 ---', flush=True)
    train_traj = load_trajectories(args.train)
    medians = compute_global_bucket_medians(train_traj)
    base_pred = _baseline_predict(val_df, medians,
                                  float(train_df['_actual_t'].median()))
    print_overall(overall_metrics(base_pred[base_pred['_family'] == 'bash']),
                  'baseline (val, bash only)')

    print(f'\n=== TOTAL WALL TIME: {time.time()-t0:.1f}s ===', flush=True)


def cmd_evaluate(args):
    train_df, val_df, _ = _load_and_build(args.train, args.val, args.threshold)
    pred = Predictor.load(args.run_dir)
    val_pred = pred.predict_df(val_df)
    print_overall(overall_metrics(val_pred), 'ML (val, all)')
    print_overall(overall_metrics(val_pred[val_pred['_family'] == 'bash']),
                  'ML (val, bash only)')
    print_per_bucket(per_bucket_metrics(val_pred), 'ML (val)')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--train', default=DEFAULT_TRAIN)
    common.add_argument('--val', default=DEFAULT_VAL)
    common.add_argument('--threshold', type=float, default=60.0,
                        help='binary head 二分类阈值 (秒). 60s = "长跑预警"; '
                             '290s = "硬 timeout" (训练样本极少, 不推荐)')

    sp = sub.add_parser('baseline', parents=[common])
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser('train', parents=[common])
    sp.add_argument('--run-dir', default=str(DEFAULT_RUN_DIR))
    sp.add_argument('--num-iter', type=int, default=300,
                    help='boosting 最大轮数 (默认 300, 配合早停一般跑不满)')
    sp.add_argument('--num-leaves', type=int, default=63)
    sp.add_argument('--lr', type=float, default=0.05)
    sp.add_argument('--min-child', type=int, default=5)
    sp.add_argument('--log-period', type=int, default=25,
                    help='每 N 轮打印 train/val loss')
    sp.add_argument('--early-stopping', type=int, default=30,
                    help='val loss 连续 N 轮不降则早停; 0 表示关闭')
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser('evaluate', parents=[common])
    sp.add_argument('--run-dir', default=str(DEFAULT_RUN_DIR))
    sp.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
