"""Trace 加载 + 训练样本展开。

输入: SWE-bench 风格 trace JSON (List[Trajectory])。
输出: pandas.DataFrame, 每行一个 tool call 样本, 含全部特征 + label。

特征构造严格按 trajectory 内顺序 causal 滚动。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from bucket import classify, family
from features import TrajectoryState, _extract_static_features


def load_trajectories(path: str | Path) -> list[dict]:
    with Path(path).open('r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f'输入不是 JSON 数组: {path}')
    return data


def iter_tool_calls(trajectory: dict):
    """按消息顺序遍历 trajectory 内所有 (assist_idx, tool_call, t, observation)。

    observation 是这次 tool_call 对应的 role=tool 消息文本 (Format A),
    若没匹配上则为空字符串。用于 Phase 3 C 类工件特征。
    """
    msgs_field = trajectory.get('messages', '')
    msgs = json.loads(msgs_field) if isinstance(msgs_field, str) else msgs_field
    if not isinstance(msgs, list):
        return

    # 第一遍: 建 tool_call_id -> observation 映射
    obs_by_tcid: dict[str, str] = {}
    for m in msgs:
        if m.get('role') != 'tool':
            continue
        tcids = m.get('tool_call_ids') or []
        content = m.get('content')
        text = ''
        if isinstance(content, list):
            parts = []
            for ci in content:
                if isinstance(ci, dict) and ci.get('type') == 'text':
                    parts.append(ci.get('text', ''))
            text = '\n'.join(p for p in parts if p)
        elif isinstance(content, str):
            text = content
        for tcid in tcids:
            if tcid:
                obs_by_tcid[tcid] = text

    for assist_idx, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        tcs = m.get('tool_calls')
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            t = tc.get('execution_time_seconds')
            if t is None:
                continue
            tcid = tc.get('id') or ''
            obs = obs_by_tcid.get(tcid, '')
            yield assist_idx, tc, float(t), obs


def compute_global_bucket_log_means(trajectories: list[dict]
                                    ) -> dict[str, float]:
    """从训练集算 log(1+t) 在每个桶内的均值。用于 E2 异桶迁移特征。

    注意: 这一步 leakage 微弱 (每个 trajectory 对自己桶 mean 的贡献很小)。
    严格做法是 leave-one-trajectory-out, 但 199 条数据太小, 简化处理。
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for traj in trajectories:
        for _, tc, t, _obs in iter_tool_calls(traj):
            b = classify(tc)
            sums[b] = sums.get(b, 0.0) + math.log1p(t)
            counts[b] = counts.get(b, 0) + 1
    return {b: sums[b] / counts[b] for b in sums}


def compute_global_bucket_medians(trajectories: list[dict]
                                  ) -> dict[str, float]:
    """每桶 train 真值 median, 用于查表 fallback。"""
    grouped: dict[str, list[float]] = {}
    for traj in trajectories:
        for _, tc, t, _obs in iter_tool_calls(traj):
            b = classify(tc)
            grouped.setdefault(b, []).append(t)
    medians = {}
    for b, ts in grouped.items():
        ts.sort()
        medians[b] = ts[len(ts) // 2]
    return medians


def build_samples(trajectories: list[dict],
                  global_bucket_log_means: dict[str, float],
                  timeout_threshold_s: float = 60.0
                  ) -> pd.DataFrame:
    """每条 trajectory 按顺序滚动, 输出每个 tool call 一行样本。

    严格 causal: 第 j 个 call 的特征只看 0..j-1 的真值。
    """
    rows = []
    for traj_idx, traj in enumerate(trajectories):
        traj_id = traj.get('traj_id') or traj.get('instance_id') or f'traj_{traj_idx}'
        state = TrajectoryState(global_bucket_log_means, timeout_threshold_s)
        for assist_idx, tc, actual_t, obs_text in iter_tool_calls(traj):
            feats = state.extract(tc)
            row = dict(feats)
            row['_traj_id'] = traj_id
            row['_assist_idx'] = assist_idx
            row['_actual_t'] = actual_t
            row['_label_log_t'] = math.log1p(actual_t)
            row['_label_is_timeout'] = int(actual_t >= timeout_threshold_s)
            row['_family'] = family(feats['bucket'])
            rows.append(row)
            state.observe(tc, actual_t, obs_text)
    return pd.DataFrame(rows)
