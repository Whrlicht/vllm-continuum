"""Phase 4: leave-one-repo-out 跨 repo 泛化评估。

把 199+300 trajectory 合在一起按 owner__repo 分组, 做 K-fold:
每折 hold-out 一组 repo 作 val, 其余训。报每折 + 平均 log-MAE。

如果跨 repo log-MAE 接近 in-domain val log-MAE, 说明模型 repo-agnostic;
否则提示模型可能藏了 repo-specific 钩子。
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (
    build_samples, compute_global_bucket_log_means, load_trajectories,
)
from evaluate import overall_metrics, per_bucket_metrics
from train import Predictor, TrainConfig, train_full_bundle


def repo_of(trajectory: dict) -> str:
    inst = trajectory.get('instance_id') or ''
    head = inst.split('.', 1)[0]
    return head or 'unknown'


def group_by_repo(trajectories: list[dict]) -> dict[str, list[dict]]:
    g = defaultdict(list)
    for t in trajectories:
        g[repo_of(t)].append(t)
    return dict(g)


def make_folds(repo_keys: list[str], k: int, seed: int = 1024
               ) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    perm = list(repo_keys)
    rng.shuffle(perm)
    folds: list[list[str]] = [[] for _ in range(k)]
    for i, key in enumerate(perm):
        folds[i % k].append(key)
    return folds


def run_loo(train_paths: list[str], k: int, threshold_s: float,
            run_root: Path, train_cfg: TrainConfig) -> None:
    print(f'\nloading all trajectories from {len(train_paths)} files...')
    all_traj: list[dict] = []
    for p in train_paths:
        all_traj.extend(load_trajectories(p))
    # 按 traj_id 去重 (同 traj_id 在不同文件可能重复)
    seen = set()
    deduped = []
    for t in all_traj:
        tid = t.get('traj_id')
        if tid in seen:
            continue
        seen.add(tid)
        deduped.append(t)
    all_traj = deduped
    print(f'  total trajectories: {len(all_traj)}')

    repo_groups = group_by_repo(all_traj)
    repo_keys = sorted(repo_groups.keys())
    print(f'  unique repos: {len(repo_keys)}')

    folds = make_folds(repo_keys, k)
    fold_results: list[dict] = []

    for fold_i, val_repos in enumerate(folds):
        print(f'\n=== fold {fold_i+1}/{k}: hold out {len(val_repos)} repos '
              f'({sum(len(repo_groups[r]) for r in val_repos)} trajs) ===',
              flush=True)
        val_set = set(val_repos)
        train_traj = [t for t in all_traj if repo_of(t) not in val_set]
        val_traj = [t for t in all_traj if repo_of(t) in val_set]
        print(f'  train trajs={len(train_traj)}, val trajs={len(val_traj)}',
              flush=True)

        log_means = compute_global_bucket_log_means(train_traj)

        t0 = time.time()
        train_df = build_samples(train_traj, log_means, threshold_s)
        val_df = build_samples(val_traj, log_means, threshold_s)
        print(f'  data prep: {time.time()-t0:.1f}s, '
              f'train_n={len(train_df)}, val_n={len(val_df)}', flush=True)

        out_dir = run_root / f'fold_{fold_i+1}'
        t0 = time.time()
        train_full_bundle(train_df, out_dir, train_cfg, val_df=val_df)
        print(f'  train: {time.time()-t0:.1f}s', flush=True)

        pred = Predictor.load(out_dir)
        val_pred = pred.predict_df(val_df)
        m_all = overall_metrics(val_pred)
        m_bash = overall_metrics(val_pred[val_pred['_family'] == 'bash'])
        print(f'  fold {fold_i+1} val(all):  log_MAE={m_all["log_mae"]:.4f}, '
              f'Spearman={m_all.get("spearman_rho", float("nan")):.4f}',
              flush=True)
        print(f'  fold {fold_i+1} val(bash): log_MAE={m_bash["log_mae"]:.4f}, '
              f'Spearman={m_bash.get("spearman_rho", float("nan")):.4f}, '
              f'P95cov={m_bash.get("p95_coverage", float("nan")):.4f}',
              flush=True)
        fold_results.append({
            'fold': fold_i + 1,
            'n_val_repos': len(val_repos),
            'n_val_trajs': len(val_traj),
            'log_mae_all': m_all['log_mae'],
            'log_mae_bash': m_bash['log_mae'],
            'spearman_bash': m_bash.get('spearman_rho', float('nan')),
            'p95_cov_bash': m_bash.get('p95_coverage', float('nan')),
        })

    print('\n=== Leave-One-Repo-Out 汇总 ===')
    df = pd.DataFrame(fold_results)
    print(df.to_string(index=False))
    print(f'\n  bash log_MAE  mean = {df["log_mae_bash"].mean():.4f}, '
          f'std = {df["log_mae_bash"].std():.4f}')
    print(f'  bash Spearman mean = {df["spearman_bash"].mean():.4f}, '
          f'std = {df["spearman_bash"].std():.4f}')
    print(f'  bash P95 cov  mean = {df["p95_cov_bash"].mean():.4f}, '
          f'std = {df["p95_cov_bash"].std():.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--paths', nargs='+', default=[
        '/data/whr/vllm-continuum/trace_data/'
        'swe_bench_sample_2902_tool_clean_with_timings.json',
        '/data/whr/vllm-continuum/trace_data/'
        'swe_bench_sample_300_tool_clean_with_timings.json',
    ])
    p.add_argument('-k', '--folds', type=int, default=5)
    p.add_argument('--threshold', type=float, default=60.0)
    p.add_argument('--run-root', type=Path,
                   default=Path(__file__).resolve().parent / 'runs' / 'loo')
    p.add_argument('--num-iter', type=int, default=300)
    p.add_argument('--num-leaves', type=int, default=63)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--min-child', type=int, default=5)
    p.add_argument('--log-period', type=int, default=50)
    p.add_argument('--early-stopping', type=int, default=30)
    args = p.parse_args()

    cfg = TrainConfig(
        timeout_threshold_s=args.threshold,
        num_iter=args.num_iter,
        num_leaves=args.num_leaves,
        learning_rate=args.lr,
        min_child_samples=args.min_child,
        log_period=args.log_period,
        early_stopping=args.early_stopping,
    )
    run_loo(args.paths, args.folds, args.threshold, args.run_root, cfg)


if __name__ == '__main__':
    main()
