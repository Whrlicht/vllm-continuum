"""训练 LightGBM 模型: P_timeout (二分类) + P50/P95 (quantile regression)。

  - submit / editor 两个 family: 用查表 (train median), 不训 ML。
  - bash family: 训三个 head, 共用相同特征集。

模型 + bundle 保存到 runs/<name>/。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

# 不参与 ML 的列 (内部记账 / 标签)
_META_COLS = {
    '_traj_id', '_assist_idx', '_actual_t', '_label_log_t',
    '_label_is_timeout', '_family',
}

# 类别特征 (LightGBM categorical, 模型自己学 one-hot)
_CATEGORICAL_COLS = ['bucket', 'tool_name', 'editor_cmd']

# 高方差桶 → 用独立 GBDT 专门训, 替代 unified 模型对这些桶的预测。
# 仅在 train n >= 20 才生效 (实测 n=15 的桶在 val 上反而过拟合)。
_SPECIALIST_BUCKETS = (
    'bash::pip::editable_local::default',
    'bash::python::pytest_full_discovery',
)
_SPECIALIST_MIN_TRAIN_N = 20


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META_COLS]


def _prepare_xy(df: pd.DataFrame,
                cat_level_maps: Optional[dict[str, list[str]]] = None
                ) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """把字符串 categorical 列预转成 int32 codes, 绕开 pandas 3 + LightGBM
    在 categorical 上的性能 regression。

    训练: cat_level_maps=None, 函数自己 factorize, 返回 maps 供保存。
    推理: 传入训练时保存的 maps, 应用同一份 level→code, 未知值→-1。
    """
    feat_cols = _select_feature_cols(df)
    X = df[feat_cols].copy()
    out_maps: dict[str, list[str]] = {} if cat_level_maps is None else dict(
        cat_level_maps)

    for col in _CATEGORICAL_COLS:
        if col not in X.columns:
            continue
        s = X[col].astype('object').where(X[col].notna(), '__nan__')
        s = s.astype(str)
        if cat_level_maps is None:
            cat = pd.Categorical(s)
            out_maps[col] = list(cat.categories)
            X[col] = cat.codes.astype('int32')
        else:
            level_to_code = {lv: i for i, lv in enumerate(out_maps[col])}
            # 未知值 -> -1, LightGBM 当 missing 处理
            X[col] = s.map(level_to_code).fillna(-1).astype('int32')

    # 数值列的 NaN 保持 NaN (LightGBM 原生支持)
    for col in feat_cols:
        if col in _CATEGORICAL_COLS:
            continue
        # 强制 float64, 避免 pandas 3 NA dtype 漏
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
        elif str(X[col].dtype).startswith(('Int', 'Float', 'boolean')):
            X[col] = X[col].astype('float64')
    return X, feat_cols, out_maps


def _train_one(X: pd.DataFrame, y: np.ndarray, *,
               objective: str, alpha: Optional[float] = None,
               num_leaves: int = 63,
               learning_rate: float = 0.05,
               num_iter: int = 800,
               min_child_samples: int = 5,
               weight: Optional[np.ndarray] = None,
               scale_pos_weight: Optional[float] = None,
               X_val: Optional[pd.DataFrame] = None,
               y_val: Optional[np.ndarray] = None,
               log_period: int = 50,
               early_stopping: int = 50,
               head_name: str = '',
               seed: int = 42,
               feature_fraction: float = 1.0,
               bagging_fraction: float = 1.0) -> lgb.Booster:
    params = {
        'objective': objective,
        'learning_rate': learning_rate,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_child_samples,
        'verbose': -1,
        'feature_pre_filter': False,
        # 多线程在这台机器上有问题, 强制单线程 (实测 63 leaves x 10 iters
        # 单线程 0.09s, 多线程 30s+ 不动)
        'num_threads': 1,
        # ensemble diversity: 不同 seed + feature/row subsampling
        'seed': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,
        'deterministic': False,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': 1 if bagging_fraction < 1.0 else 0,
    }
    if alpha is not None:
        params['alpha'] = alpha
    if objective == 'binary':
        params['metric'] = 'binary_logloss'
        # D1: 类不平衡修复, 把正样本梯度放大, 让模型不会因为
        # timeout 占比 1.5% 就保守预测全 0
        if scale_pos_weight is not None and scale_pos_weight > 1.0:
            params['scale_pos_weight'] = float(scale_pos_weight)
    elif objective == 'quantile':
        params['metric'] = 'quantile'

    cat_cols = [c for c in _CATEGORICAL_COLS if c in X.columns]
    # X 已经是 int32 codes, 在此明确告诉 LightGBM 这些列是 categorical
    train_set = lgb.Dataset(X.values, label=y, weight=weight,
                            feature_name=list(X.columns),
                            categorical_feature=cat_cols,
                            free_raw_data=False)
    valid_sets = [train_set]
    valid_names = ['train']
    if X_val is not None and y_val is not None and len(X_val) > 0:
        val_set = lgb.Dataset(X_val.values, label=y_val,
                              feature_name=list(X_val.columns),
                              categorical_feature=cat_cols,
                              reference=train_set, free_raw_data=False)
        valid_sets.append(val_set)
        valid_names.append('val')

    callbacks = [lgb.log_evaluation(period=log_period)]
    if X_val is not None and early_stopping > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping,
                                            verbose=True))

    if head_name:
        print(f'\n--- training head [{head_name}] '
              f'(n_train={len(X)}, n_val={len(X_val) if X_val is not None else 0}) ---')
    booster = lgb.train(params, train_set, num_boost_round=num_iter,
                        valid_sets=valid_sets, valid_names=valid_names,
                        callbacks=callbacks)
    return booster


def _detect_unreliable_buckets_cv(bash_df: pd.DataFrame,
                                    cat_maps: dict[str, list[str]],
                                    k: int = 3,
                                    exclude: Optional[set] = None,
                                    spearman_thresh: float = 0.1
                                    ) -> list[str]:
    """3-fold CV on bash data. 每桶 OOF Spearman <= spearman_thresh (含负数)
    或 n<5 -> 标记为不可靠。
    """
    try:
        from sklearn.model_selection import KFold
        from scipy.stats import spearmanr
    except ImportError:
        return []

    exclude = exclude or set()
    bash_df = bash_df.reset_index(drop=True)
    n = len(bash_df)
    if n < 30:
        return []

    oof = np.full(n, np.nan)
    y = bash_df['_label_log_t'].to_numpy()
    kf = KFold(n_splits=k, shuffle=True, random_state=1024)

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(bash_df), 1):
        tr = bash_df.iloc[tr_idx]
        va = bash_df.iloc[va_idx]
        X_tr, _, _ = _prepare_xy(tr, cat_level_maps=cat_maps)
        X_va, _, _ = _prepare_xy(va, cat_level_maps=cat_maps)
        # 列对齐(cat_maps 已固定级别, 应该一致)
        X_va = X_va[X_tr.columns]
        cat_cols = [c for c in _CATEGORICAL_COLS if c in X_tr.columns]
        params = {
            'objective': 'quantile', 'alpha': 0.5,
            'learning_rate': 0.05, 'num_leaves': 31,
            'min_data_in_leaf': 5, 'verbose': -1,
            'feature_pre_filter': False, 'num_threads': 1,
            'metric': 'quantile',
        }
        ds_tr = lgb.Dataset(X_tr.values, label=y[tr_idx],
                            feature_name=list(X_tr.columns),
                            categorical_feature=cat_cols)
        booster = lgb.train(params, ds_tr, num_boost_round=200)
        oof[va_idx] = booster.predict(X_va.values)

    bash_df = bash_df.copy()
    bash_df['_oof_log'] = oof

    unreliable: list[str] = []
    for bucket, sub in bash_df.groupby('bucket'):
        if bucket in exclude:
            continue
        n_b = len(sub)
        if n_b < 5:
            unreliable.append(bucket)
            continue
        if sub['_oof_log'].isna().any():
            continue
        try:
            rho, _ = spearmanr(sub['_label_log_t'], sub['_oof_log'])
            if rho != rho:  # NaN (constant column)
                continue
            if rho <= spearman_thresh:
                unreliable.append(bucket)
        except Exception:
            continue
    return unreliable


# ---------------------------------------------------------------------------
# v2 helpers: 5-seed ensemble + conformal prediction + isotonic calibration
# + multi-quantile heads. 这些都是与数据集无关的通用方法。
# ---------------------------------------------------------------------------

# 多分位回归头: 比单一 P50/P95 提供更细的不确定度信息
_QUANTILES: tuple = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
_DEFAULT_N_SEEDS: int = 5  # ensemble 大小
_DEFAULT_CAL_FRACTION: float = 0.1  # 训练集中留 10% 做 conformal/isotonic 校准


def _split_train_cal_by_traj(
    df: pd.DataFrame, frac: float = 0.1, seed: int = 1024
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 traj_id 切分: 同 traj 不跨 fit/cal, 避免 conformal 假设被破坏。"""
    if '_traj_id' not in df.columns:
        raise RuntimeError("train_df 缺 _traj_id 列, 无法做 traj-aware cal split")
    traj_ids = df['_traj_id'].drop_duplicates().to_numpy()
    rng = np.random.RandomState(seed)
    rng.shuffle(traj_ids)
    n_cal = max(1, int(round(len(traj_ids) * frac)))
    cal_set = set(traj_ids[:n_cal].tolist())
    cal_mask = df['_traj_id'].isin(cal_set)
    return df[~cal_mask].reset_index(drop=True), df[cal_mask].reset_index(drop=True)


def _train_ensemble(X: pd.DataFrame, y: np.ndarray, *,
                    n_seeds: int = _DEFAULT_N_SEEDS,
                    head_base_name: str = '',
                    **kwargs) -> list[lgb.Booster]:
    """训练 n_seeds 个独立模型, 不同 seed + bagging 提供多样性."""
    boosters = []
    for s in range(n_seeds):
        b = _train_one(
            X, y,
            seed=42 + s * 17,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            head_name=f'{head_base_name} [seed {s}]' if head_base_name else '',
            **kwargs,
        )
        boosters.append(b)
    return boosters


def _predict_ensemble(boosters: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    """K 个 booster 的预测均值."""
    return np.mean(np.stack([b.predict(X) for b in boosters], axis=0), axis=0)


def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """有限样本 split-conformal 分位.

    给定 cal 残差 r_i = y_actual - y_pred_quantile_α, 返回 q 满足:
      P(y_actual <= y_pred + q) >= alpha  (在 i.i.d. 假设下严格)
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    # 有限样本修正: 取 ceil((n+1)*alpha) / n-th order statistic
    k = int(np.ceil((n + 1) * alpha)) - 1
    k = max(0, min(k, n - 1))
    return float(np.sort(residuals)[k])


def _fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> dict:
    """拟合 isotonic regression 把 raw prob 映射到经验校准 prob.
    返回可序列化的 (X_thresholds, y_thresholds) 对."""
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return {'X_thresholds': [], 'y_thresholds': [], 'enabled': False}
    n_pos = int(labels.sum())
    if n_pos < 3 or len(labels) - n_pos < 3:
        return {'X_thresholds': [], 'y_thresholds': [], 'enabled': False}
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0,
                              increasing=True)
    iso.fit(probs, labels.astype(float))
    return {
        'X_thresholds': iso.X_thresholds_.tolist(),
        'y_thresholds': iso.y_thresholds_.tolist(),
        'enabled': True,
    }


def _apply_isotonic(mapping: dict, probs: np.ndarray) -> np.ndarray:
    """把 _fit_isotonic 保存的映射应用到新 probs (numpy.interp 是 monotonic 的)."""
    if not mapping.get('enabled'):
        return probs
    xs = np.asarray(mapping['X_thresholds'], dtype=float)
    ys = np.asarray(mapping['y_thresholds'], dtype=float)
    if len(xs) == 0:
        return probs
    p = np.asarray(probs, dtype=float)
    return np.clip(np.interp(p, xs, ys), 0.0, 1.0)


@dataclass
class TrainConfig:
    # binary head 阈值: t >= 这个值算"长跑". 290s 表示硬 timeout, train 里只有
    # 2 个样本几乎学不到; 60s 让模型见到 ~20+ 个长跑样本, 真能学。
    timeout_threshold_s: float = 60.0
    num_iter: int = 300
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_child_samples: int = 5
    log_period: int = 25
    early_stopping: int = 30
    # v2 通用优化参数
    n_seeds: int = _DEFAULT_N_SEEDS         # ensemble 大小
    cal_fraction: float = _DEFAULT_CAL_FRACTION  # 留多少 train 做 calibration
    quantiles: tuple = _QUANTILES           # 训哪些分位


def train_full_bundle(train_df: pd.DataFrame,
                      out_dir: str | Path,
                      cfg: Optional[TrainConfig] = None,
                      val_df: Optional[pd.DataFrame] = None) -> dict:
    """训练完整 bundle (v2) 并保存。返回 bundle 元信息。

    v2 新增:
      - 5-seed ensemble (主头 + specialists)
      - 多分位预测 (P10/P25/P50/P75/P90/P95/P99)
      - Split conformal calibration (用 train 10% 留出做 distribution-free 覆盖率保证)
      - Isotonic probability calibration (timeout 头)
    """
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 全样本统计 (不分 fit/cal, 给查表 fallback 用) ----
    median_t_by_bucket = (
        train_df.groupby('bucket')['_actual_t'].median().to_dict())
    mean_t_by_bucket = (
        train_df.groupby('bucket')['_actual_t'].mean().to_dict())
    log_mean_by_bucket = (
        train_df.assign(_log_t=np.log1p(train_df['_actual_t']))
                .groupby('bucket')['_log_t'].mean().to_dict())

    # 只在 bash 上训 ML
    bash_df = train_df[train_df['_family'] == 'bash'].copy()
    if bash_df.empty:
        raise RuntimeError('训练集没有 bash 样本')

    # ---- 按 traj_id 切 fit / cal ----
    fit_bash, cal_bash = _split_train_cal_by_traj(
        bash_df, frac=cfg.cal_fraction, seed=1024)
    print(f'\n[v2] traj-aware split: '
          f'fit={len(fit_bash)} samples, cal={len(cal_bash)} samples '
          f'({cfg.cal_fraction*100:.0f}% by traj_id)')

    X_fit, feat_cols, cat_maps = _prepare_xy(fit_bash)
    X_cal_all, _, _ = _prepare_xy(cal_bash, cat_level_maps=cat_maps)
    for col in X_fit.columns:
        if col not in X_cal_all.columns:
            X_cal_all[col] = float('nan')
    X_cal = X_cal_all[X_fit.columns]

    # ---- 准备 val (如果提供) ----
    X_val_bash = y_val_to = y_val_log_all = None
    val_bash_df = None
    if val_df is not None:
        val_bash_df = val_df[val_df['_family'] == 'bash'].copy()
        if not val_bash_df.empty:
            X_val_all, _, _ = _prepare_xy(val_bash_df, cat_level_maps=cat_maps)
            for col in X_fit.columns:
                if col not in X_val_all.columns:
                    X_val_all[col] = float('nan')
            X_val_bash = X_val_all[X_fit.columns]
            y_val_to = val_bash_df['_label_is_timeout'].astype(int).to_numpy()
            y_val_log_all = val_bash_df['_label_log_t'].to_numpy()

    # ---- 训 timeout 二分类 ensemble ----
    y_to_fit = fit_bash['_label_is_timeout'].astype(int).to_numpy()
    y_to_cal = cal_bash['_label_is_timeout'].astype(int).to_numpy()
    n_pos = int(y_to_fit.sum())
    n_neg = len(y_to_fit) - n_pos
    spw = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
    p_timeout_models: list[lgb.Booster] = []
    if n_pos >= 1:
        p_timeout_models = _train_ensemble(
            X_fit, y_to_fit,
            n_seeds=cfg.n_seeds,
            objective='binary',
            num_leaves=min(cfg.num_leaves, max(7, 4 * n_pos)),
            learning_rate=cfg.learning_rate,
            num_iter=cfg.num_iter,
            min_child_samples=max(1, min(cfg.min_child_samples, n_pos)),
            scale_pos_weight=spw,
            X_val=X_val_bash, y_val=y_val_to,
            log_period=cfg.log_period * 4,
            early_stopping=cfg.early_stopping,
            head_base_name=f'p_timeout (n_pos={n_pos})',
        )

    # ---- 训多分位 ensemble (P10..P99) ----
    y_log_fit = fit_bash['_label_log_t'].to_numpy()
    quantile_models: dict[float, list[lgb.Booster]] = {}
    for q in cfg.quantiles:
        boosters = _train_ensemble(
            X_fit, y_log_fit,
            n_seeds=cfg.n_seeds,
            objective='quantile', alpha=float(q),
            num_leaves=cfg.num_leaves,
            learning_rate=cfg.learning_rate,
            num_iter=cfg.num_iter,
            min_child_samples=cfg.min_child_samples,
            X_val=X_val_bash, y_val=y_val_log_all,
            log_period=cfg.log_period * 4,
            early_stopping=cfg.early_stopping,
            head_base_name=f'p{int(q*100):02d}',
        )
        quantile_models[q] = boosters

    # 保存模型文件
    for q, boosters in quantile_models.items():
        for s_idx, b in enumerate(boosters):
            b.save_model(str(out_dir / f'p{int(q*100):02d}_seed{s_idx}.lgb'))
    for s_idx, b in enumerate(p_timeout_models):
        b.save_model(str(out_dir / f'p_timeout_seed{s_idx}.lgb'))

    # ---- specialists (高方差桶, 单 booster, 不 ensemble 因 n 太小) ----
    specialists: dict[str, dict] = {}
    for bucket in _SPECIALIST_BUCKETS:
        sub = fit_bash[fit_bash['bucket'] == bucket]
        if len(sub) < _SPECIALIST_MIN_TRAIN_N:
            continue
        sub_X, _, _ = _prepare_xy(sub, cat_level_maps=cat_maps)
        sub_X = sub_X[X_fit.columns]
        sub_y = sub['_label_log_t'].to_numpy()
        if val_bash_df is not None:
            sub_val = val_bash_df[val_bash_df['bucket'] == bucket]
            sub_val_X = (X_val_bash.loc[sub_val.index]
                         if X_val_bash is not None and len(sub_val) > 0
                         else None)
            sub_val_y = (sub_val['_label_log_t'].to_numpy()
                         if len(sub_val) > 0 else None)
        else:
            sub_val_X = sub_val_y = None
        leaves = max(7, min(31, len(sub) // 2))
        sp_p50 = _train_one(
            sub_X, sub_y, objective='quantile', alpha=0.5,
            num_leaves=leaves, learning_rate=0.05, num_iter=200,
            min_child_samples=2,
            X_val=sub_val_X, y_val=sub_val_y,
            log_period=cfg.log_period * 8, early_stopping=20,
            head_name=f'specialist[{bucket}]/p50 (n={len(sub)})',
        )
        sp_p95 = _train_one(
            sub_X, sub_y, objective='quantile', alpha=0.95,
            num_leaves=leaves, learning_rate=0.05, num_iter=200,
            min_child_samples=2,
            X_val=sub_val_X, y_val=sub_val_y,
            log_period=cfg.log_period * 8, early_stopping=20,
            head_name=f'specialist[{bucket}]/p95 (n={len(sub)})',
        )
        sp_p50.save_model(
            str(out_dir / f'specialist_{bucket.replace("::","_")}_p50.lgb'))
        sp_p95.save_model(
            str(out_dir / f'specialist_{bucket.replace("::","_")}_p95.lgb'))
        specialists[bucket] = {
            'p50_file': f'specialist_{bucket.replace("::","_")}_p50.lgb',
            'p95_file': f'specialist_{bucket.replace("::","_")}_p95.lgb',
            'n_train': len(sub),
        }

    # ---- Conformal: 在 cal 集上算每个 quantile 的 log-shift ----
    # 目标: y_actual <= y_pred + q, q = finite-sample-corrected quantile of residuals
    # conformal q 直接覆盖 P50/95 等所有 quantile, 提供 distribution-free 保证
    print(f'\n[v2 conformal] 在 {len(X_cal)} 个 cal 样本上算 conformal shift...')
    cal_X_arr = X_cal.values
    y_log_cal = cal_bash['_label_log_t'].to_numpy()
    conformal_shift: dict[str, float] = {}
    for q in cfg.quantiles:
        # 主头 ensemble 预测
        cal_pred = _predict_ensemble(quantile_models[q], cal_X_arr)
        # specialists 覆盖
        for sp_bucket, sp_paths in specialists.items():
            mask = (cal_bash['bucket'] == sp_bucket).values
            if not mask.any():
                continue
            sp_path = out_dir / (sp_paths['p95_file'] if q >= 0.9 else sp_paths['p50_file'])
            if sp_path.exists():
                sp_b = lgb.Booster(model_file=str(sp_path))
                cal_pred[mask] = sp_b.predict(cal_X_arr[mask])
        residuals = y_log_cal - cal_pred
        # conformal 单边: 让 P(actual <= pred + shift) >= q
        shift = _conformal_quantile(residuals, q)
        # 下界 quantile 允许负 shift (让区间真的代表分位), 上界保护非负
        if q < 0.5:
            conformal_shift[f'{q:.2f}'] = float(shift)  # 可为负
        else:
            conformal_shift[f'{q:.2f}'] = float(max(shift, 0.0))
        print(f'  q={q:.2f}: shift={conformal_shift[f"{q:.2f}"]:+.4f}')

    # ---- Isotonic: 在 cal 上校准 timeout prob ----
    isotonic_mapping = {'enabled': False, 'X_thresholds': [], 'y_thresholds': []}
    if p_timeout_models and y_to_cal.sum() > 0:
        cal_pred_to = _predict_ensemble(p_timeout_models, cal_X_arr)
        isotonic_mapping = _fit_isotonic(cal_pred_to, y_to_cal)
        if isotonic_mapping['enabled']:
            print(f'\n[v2 isotonic] timeout prob 校准: '
                  f'{len(isotonic_mapping["X_thresholds"])} 个 breakpoints '
                  f'(n_cal={len(y_to_cal)}, n_pos={int(y_to_cal.sum())})')
            with (out_dir / 'isotonic_timeout.json').open('w') as f:
                json.dump(isotonic_mapping, f, indent=2)

    # ---- D2: timeout 决策阈值校准 (现在在 cal 上, 不再依赖 val) ----
    timeout_decision_threshold = 0.5
    timeout_threshold_metrics = {}
    if p_timeout_models and y_to_cal.sum() > 0:
        cal_pred_to = _predict_ensemble(p_timeout_models, cal_X_arr)
        if isotonic_mapping['enabled']:
            cal_pred_to = _apply_isotonic(isotonic_mapping, cal_pred_to)
        best_f1 = -1.0
        for thresh in np.linspace(0.05, 0.95, 19):
            pt = (cal_pred_to >= thresh).astype(int)
            tp = int(((pt == 1) & (y_to_cal == 1)).sum())
            fp = int(((pt == 1) & (y_to_cal == 0)).sum())
            fn = int(((pt == 0) & (y_to_cal == 1)).sum())
            if tp == 0:
                continue
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            if f1 > best_f1:
                best_f1 = f1
                timeout_decision_threshold = float(thresh)
                timeout_threshold_metrics = {
                    'f1': float(f1), 'recall': float(rec), 'precision': float(prec),
                }
        if best_f1 > 0:
            print(f'\n[v2 D2] timeout 决策阈值 (cal): '
                  f'threshold={timeout_decision_threshold:.3f}, '
                  f'F1={timeout_threshold_metrics["f1"]:.3f}, '
                  f'recall={timeout_threshold_metrics["recall"]:.3f}, '
                  f'precision={timeout_threshold_metrics["precision"]:.3f}')

    # ---- 不可靠桶检测 (用 val 后置 filter, 实现不变) ----
    unreliable_buckets: list[str] = []
    if X_val_bash is not None and val_bash_df is not None:
        try:
            from scipy.stats import spearmanr
        except ImportError:
            spearmanr = None
        if spearmanr is not None:
            log_p50_val = _predict_ensemble(quantile_models[0.5], X_val_bash.values)
            val_pred_p50 = np.expm1(log_p50_val)
            for sp_bucket, sp_paths in specialists.items():
                val_in_sp = (val_bash_df['bucket'] == sp_bucket).values
                if not val_in_sp.any():
                    continue
                sp_p50_path = out_dir / sp_paths['p50_file']
                if sp_p50_path.exists():
                    sp_b = lgb.Booster(model_file=str(sp_p50_path))
                    val_pred_p50[val_in_sp] = np.expm1(
                        sp_b.predict(X_val_bash.values[val_in_sp]))
            val_with_pred = val_bash_df.copy()
            val_with_pred['_pred'] = val_pred_p50
            for bucket, sub in val_with_pred.groupby('bucket'):
                if len(sub) < 5 or bucket in specialists:
                    continue
                try:
                    rho, _ = spearmanr(sub['_actual_t'], sub['_pred'])
                    if rho != rho:
                        continue
                    log_mae = float(np.abs(
                        np.log1p(sub['_actual_t'].to_numpy())
                        - np.log1p(np.clip(sub['_pred'].to_numpy(), 0, None))
                    ).mean())
                    fallback_mean_t = mean_t_by_bucket.get(bucket, 0.5)
                    fallback_log_mae = float(np.abs(
                        np.log1p(sub['_actual_t'].to_numpy())
                        - np.log1p(fallback_mean_t)
                    ).mean())
                    if rho < 0.0 and log_mae > fallback_log_mae * 1.05:
                        unreliable_buckets.append(bucket)
                except Exception:
                    pass
    if unreliable_buckets:
        print(f'\n[unreliable] {len(unreliable_buckets)} 个桶回退 bucket mean:\n  '
              f'{unreliable_buckets}')

    # ---- bundle ----
    bundle = {
        'version': 'v2',
        'feature_cols': feat_cols,
        'categorical_cols': [c for c in _CATEGORICAL_COLS if c in feat_cols],
        'cat_level_maps': cat_maps,
        'median_t_by_bucket': median_t_by_bucket,
        'mean_t_by_bucket': mean_t_by_bucket,
        'unreliable_buckets': unreliable_buckets,
        'log_mean_by_bucket': log_mean_by_bucket,
        'global_bash_median_t': float(
            train_df.loc[train_df['_family'] == 'bash', '_actual_t'].median()),
        'global_overall_median_t': float(train_df['_actual_t'].median()),
        'config': asdict(cfg),
        'n_seeds': cfg.n_seeds,
        'n_cal': len(cal_bash),
        'quantiles': [float(q) for q in cfg.quantiles],
        'p_timeout_trained': len(p_timeout_models) > 0,
        'p_timeout_scale_pos_weight': float(spw),
        'timeout_decision_threshold': timeout_decision_threshold,
        'timeout_threshold_metrics': timeout_threshold_metrics,
        'timeout_isotonic_enabled': isotonic_mapping['enabled'],
        'conformal_shift_log': conformal_shift,
        'specialists': specialists,
        # 兼容老 Predictor 字段 (v1 的 D3/Step E 还会用)
        'p95_timeout_signal_threshold_s': float(cfg.timeout_threshold_s),
        'p95_timeout_signal_slope_s': max(15.0, float(cfg.timeout_threshold_s) / 3),
        'hung_before_threshold_s': 60.0,
        'hung_before_slope_s': 20.0,
        # 老字段保留, 用 conformal_shift 替代但旧代码 fallback 用
        'p95_log_shift': conformal_shift.get('0.95', 0.0),
    }
    with (out_dir / 'bundle.json').open('w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    return bundle


# ---------------------------------------------------------------------------
# 推理: load bundle + predict
# ---------------------------------------------------------------------------

@dataclass
class Predictor:
    bundle: dict
    # v2: ensemble per quantile. v1 兼容: 单 booster 当作 1-elem list
    quantile_models: dict[float, list[lgb.Booster]]
    p_timeout_models: list[lgb.Booster]
    specialists: dict[str, dict]
    isotonic_mapping: dict
    # v1 兼容
    p50: Optional[lgb.Booster] = None
    p95: Optional[lgb.Booster] = None
    p_timeout: Optional[lgb.Booster] = None

    @classmethod
    def load(cls, run_dir: str | Path) -> 'Predictor':
        run_dir = Path(run_dir)
        with (run_dir / 'bundle.json').open('r', encoding='utf-8') as f:
            bundle = json.load(f)

        version = bundle.get('version', 'v1')
        quantile_models: dict[float, list[lgb.Booster]] = {}
        p_timeout_models: list[lgb.Booster] = []
        legacy_p50 = legacy_p95 = legacy_p_timeout = None

        if version == 'v2':
            # 多分位 ensemble: pXX_seedY.lgb
            quantiles = bundle.get('quantiles', [0.5, 0.95])
            n_seeds = int(bundle.get('n_seeds', 5))
            for q in quantiles:
                qi = int(round(q * 100))
                files = [run_dir / f'p{qi:02d}_seed{s}.lgb' for s in range(n_seeds)]
                files = [f for f in files if f.exists()]
                if not files:
                    raise RuntimeError(f'v2 bundle 缺 quantile q={q} 的 booster 文件')
                quantile_models[float(q)] = [
                    lgb.Booster(model_file=str(f)) for f in files]
            for s in range(n_seeds):
                f = run_dir / f'p_timeout_seed{s}.lgb'
                if f.exists():
                    p_timeout_models.append(lgb.Booster(model_file=str(f)))
            # 兼容老 API: legacy_p50/p95 指向 ensemble 首个
            if 0.5 in quantile_models:
                legacy_p50 = quantile_models[0.5][0]
            if 0.95 in quantile_models:
                legacy_p95 = quantile_models[0.95][0]
            if p_timeout_models:
                legacy_p_timeout = p_timeout_models[0]
        else:
            # v1 fallback: 单 booster per head
            p50_path = run_dir / 'p50.lgb'
            p95_path = run_dir / 'p95.lgb'
            p_to_path = run_dir / 'p_timeout.lgb'
            legacy_p50 = lgb.Booster(model_file=str(p50_path))
            legacy_p95 = lgb.Booster(model_file=str(p95_path))
            quantile_models = {0.5: [legacy_p50], 0.95: [legacy_p95]}
            if p_to_path.exists():
                legacy_p_timeout = lgb.Booster(model_file=str(p_to_path))
                p_timeout_models = [legacy_p_timeout]

        # 高方差桶 specialists (v1/v2 同结构)
        specialists = {}
        for bucket, info in (bundle.get('specialists') or {}).items():
            sp_p50_path = run_dir / info['p50_file']
            sp_p95_path = run_dir / info['p95_file']
            if sp_p50_path.exists() and sp_p95_path.exists():
                specialists[bucket] = {
                    'p50': lgb.Booster(model_file=str(sp_p50_path)),
                    'p95': lgb.Booster(model_file=str(sp_p95_path)),
                }

        # isotonic mapping (v2 only)
        isotonic_mapping = {'enabled': False, 'X_thresholds': [], 'y_thresholds': []}
        iso_path = run_dir / 'isotonic_timeout.json'
        if iso_path.exists():
            with iso_path.open() as f:
                isotonic_mapping = json.load(f)

        return cls(
            bundle=bundle,
            quantile_models=quantile_models,
            p_timeout_models=p_timeout_models,
            specialists=specialists,
            isotonic_mapping=isotonic_mapping,
            p50=legacy_p50, p95=legacy_p95, p_timeout=legacy_p_timeout,
        )

    def _lookup_const(self, bucket: str) -> float:
        return self.bundle['median_t_by_bucket'].get(
            bucket, self.bundle['global_overall_median_t'])

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """对一个 sample DataFrame 做预测, 返回新增 pred 列的副本。

        输出列:
          - pred_p50, pred_p95: 主要 quantile (v1 兼容)
          - pred_p10, pred_p25, pred_p75, pred_p90, pred_p99: 全 quantile (v2 only)
          - pred_p_timeout, pred_p_timeout_raw: 二分类输出 (raw=校准前; pred=合并 D3/Step E)
          - pred_expected: 期望耗时
        """
        df = df.copy()
        feat_cols = self.bundle['feature_cols']
        cat_maps = self.bundle.get('cat_level_maps', {})

        # 对齐到训练特征列, 缺的用 NaN
        for col in feat_cols:
            if col not in df.columns:
                df[col] = float('nan')
        Xall, _, _ = _prepare_xy(df[feat_cols], cat_level_maps=cat_maps)
        Xall = Xall[feat_cols]

        # 默认 fallback: 查表
        bucket_lookup = df['bucket'].map(self._lookup_const).fillna(
            self.bundle['global_overall_median_t'])
        quantiles = sorted(self.quantile_models.keys())
        for q in quantiles:
            qi = int(round(q * 100))
            df[f'pred_p{qi:02d}'] = bucket_lookup
        df['pred_p_timeout_raw'] = 0.0

        # bash 样本走 ML
        bash_mask = df['_family'] == 'bash'
        bash_idx = df.index[bash_mask]
        conformal_shift = self.bundle.get('conformal_shift_log') or {}
        # v1 fallback: 用 p95_log_shift 当 0.95 的 shift
        if not conformal_shift and 'p95_log_shift' in self.bundle:
            conformal_shift = {'0.95': float(self.bundle['p95_log_shift'])}

        if bash_mask.any():
            X_bash = Xall.loc[bash_mask].values
            # 主头 ensemble per quantile
            for q in quantiles:
                qi = int(round(q * 100))
                log_pred = _predict_ensemble(self.quantile_models[q], X_bash)
                shift = float(conformal_shift.get(f'{q:.2f}', 0.0))
                pred_t = np.expm1(log_pred + shift)
                df.loc[bash_mask, f'pred_p{qi:02d}'] = pred_t

            # specialists 覆盖 p50/p95 (其他 quantile 保持主头)
            if self.specialists:
                bash_buckets = df.loc[bash_idx, 'bucket'].values
                shift_p50 = float(conformal_shift.get('0.50', 0.0))
                shift_p95 = float(conformal_shift.get('0.95', 0.0))
                for sp_bucket, models in self.specialists.items():
                    sp_mask_local = bash_buckets == sp_bucket
                    if not sp_mask_local.any():
                        continue
                    X_sp = Xall.loc[bash_idx[sp_mask_local]].values
                    df.loc[bash_idx[sp_mask_local], 'pred_p50'] = np.expm1(
                        models['p50'].predict(X_sp) + shift_p50)
                    df.loc[bash_idx[sp_mask_local], 'pred_p95'] = np.expm1(
                        models['p95'].predict(X_sp) + shift_p95)

            # 不可靠桶: 回退 bucket mean
            unreliable = self.bundle.get('unreliable_buckets') or []
            mean_by = self.bundle.get('mean_t_by_bucket') or {}
            if unreliable:
                bash_buckets = df.loc[bash_idx, 'bucket'].values
                for unr_bucket in unreliable:
                    unr_mask = bash_buckets == unr_bucket
                    if not unr_mask.any():
                        continue
                    fallback_mean = float(mean_by.get(
                        unr_bucket,
                        self.bundle.get('global_bash_median_t', 0.5)))
                    df.loc[bash_idx[unr_mask], 'pred_p50'] = fallback_mean

            # timeout 二分类: ensemble + isotonic
            if self.p_timeout_models:
                raw = _predict_ensemble(self.p_timeout_models, X_bash)
                if self.isotonic_mapping.get('enabled'):
                    raw = _apply_isotonic(self.isotonic_mapping, raw)
                df.loc[bash_mask, 'pred_p_timeout_raw'] = raw

        # 数值卫生 (clip + 单调 P50 <= P95)
        for q in quantiles:
            qi = int(round(q * 100))
            df[f'pred_p{qi:02d}'] = df[f'pred_p{qi:02d}'].clip(lower=0.0)
        df['pred_p_timeout_raw'] = df['pred_p_timeout_raw'].clip(0.0, 1.0)
        # 保证单调: 每个高分位 >= 低分位
        for i, q in enumerate(quantiles[1:], start=1):
            qi = int(round(q * 100))
            q_prev = quantiles[i - 1]
            qi_prev = int(round(q_prev * 100))
            df[f'pred_p{qi:02d}'] = np.maximum(
                df[f'pred_p{qi:02d}'].to_numpy(),
                df[f'pred_p{qi_prev:02d}'].to_numpy(),
            )

        # D3: P95 兜底 timeout 信号 (v1 保留)
        p95_th = float(self.bundle.get('p95_timeout_signal_threshold_s', 100.0))
        p95_sl = float(self.bundle.get('p95_timeout_signal_slope_s', 30.0))
        if 'pred_p95' not in df.columns:
            df['pred_p95'] = df.get('pred_p50', 0.0)
        z = np.clip((df['pred_p95'].to_numpy() - p95_th) / p95_sl, -50, 50)
        p95_signal = 1.0 / (1.0 + np.exp(-z))
        df['pred_p_timeout_p95_signal'] = p95_signal

        # Step E: 同 trajectory "已 hang 过" 硬规则 (v1 保留)
        e1_th = float(self.bundle.get('hung_before_threshold_s', 60.0))
        e1_sl = float(self.bundle.get('hung_before_slope_s', 20.0))
        e1_signal = np.zeros(len(df))
        if 'e1_same_bucket_log_mean' in df.columns:
            same_log_mean = df['e1_same_bucket_log_mean'].to_numpy()
            same_log_last = df.get('e1_same_bucket_last',
                                    pd.Series(np.full(len(df),
                                                       np.nan))).to_numpy()
            same_cnt = (df.get('e1_same_bucket_count',
                               pd.Series(np.zeros(len(df))))
                        .to_numpy())
            valid = (~np.isnan(same_log_mean)) & (same_cnt > 0)
            mean_t = np.where(valid,
                              np.expm1(np.where(valid, same_log_mean, 0)),
                              0.0)
            last_valid = ~np.isnan(same_log_last)
            last_t = np.where(last_valid,
                              np.expm1(np.where(last_valid, same_log_last, 0)),
                              0.0)
            proxy_t = np.maximum(mean_t, last_t)
            z2 = np.clip((proxy_t - e1_th) / e1_sl, -50, 50)
            e1_signal = np.where(valid, 1.0 / (1.0 + np.exp(-z2)), 0.0)
        df['pred_p_timeout_hung_signal'] = e1_signal

        df['pred_p_timeout'] = np.maximum.reduce([
            df['pred_p_timeout_raw'].to_numpy(),
            p95_signal,
            e1_signal,
        ])

        # 期望值 = (1-Pto) * P50 + Pto * 300
        df['pred_expected'] = (
            (1 - df['pred_p_timeout']) * df['pred_p50']
            + df['pred_p_timeout'] * 300.0)
        return df
