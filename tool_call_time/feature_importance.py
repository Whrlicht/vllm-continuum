"""特征重要性诊断: 哪些特征真有用、哪些是噪音。

逐 head 报 LightGBM 的 gain importance, 排序输出 top-N。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import Predictor


def importance_of(booster, feature_names: list[str], top: int = 30):
    """Returns sorted [(feature, gain, split_count)]."""
    gains = booster.feature_importance(importance_type='gain')
    splits = booster.feature_importance(importance_type='split')
    rows = list(zip(feature_names, gains, splits))
    rows.sort(key=lambda x: -x[1])
    return rows[:top]


def main():
    pred = Predictor.load('runs/default')
    feat_cols = pred.bundle['feature_cols']

    for head_name, booster in [
        ('p50', pred.p50),
        ('p95', pred.p95),
        ('p_timeout', pred.p_timeout) if pred.p_timeout else (None, None),
    ]:
        if booster is None:
            continue
        print(f'\n=== {head_name} 特征重要性 (top 30 by gain) ===')
        rows = importance_of(booster, feat_cols)
        print(f'{"feature":40s} {"gain":>14s} {"splits":>8s}')
        print('-' * 70)
        for name, gain, splits in rows:
            print(f'{name:40s} {gain:>14.0f} {splits:>8d}')

    # specialists
    if pred.specialists:
        for bucket, models in pred.specialists.items():
            for head_name in ('p50', 'p95'):
                booster = models[head_name]
                print(f'\n=== specialist[{bucket}]/{head_name} top 15 ===')
                rows = importance_of(booster, feat_cols, top=15)
                for name, gain, splits in rows:
                    if gain > 0:
                        print(f'  {name:40s} {gain:>14.0f} {splits:>6d}')

    # 完全没用 (gain=0) 的特征
    print(f'\n=== 完全没被用的特征 (所有 head gain=0) ===')
    used = set()
    for booster in [pred.p50, pred.p95, pred.p_timeout]:
        if booster is None:
            continue
        gains = booster.feature_importance(importance_type='gain')
        for name, gain in zip(feat_cols, gains):
            if gain > 0:
                used.add(name)
    unused = [f for f in feat_cols if f not in used]
    print(f'共 {len(unused)}/{len(feat_cols)} 个特征 0 gain:')
    for f in unused:
        print(f'  {f}')


if __name__ == '__main__':
    main()
