"""从 licht-v2 跑出的 scheduler_timestamps 抽 T2 排队时间真值.

T2 定义: prefill 节点收到请求到调度器真正把它从 waiting 队列拉起来执行 prefill
的时长 (秒).
   T2 = waiting_to_running - Request_arrival_time

每条 trajectory 会有 N 轮:
   round 0:  首条 prompt (新会话) -> 不在我们预测目标里 (用户原话「第2+轮」)
   round 1+: 每次 decode 结束 + tool 跑完 -> 重新回到 prefill 队列
我们要预测的是 round >= 1 (i.e. arrival_idx >= 1)。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def parse_traj_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 (arrival, wtr, departure) 三元组组合成每轮一条 record.

    严格按顺序遍历; 偶尔会出现 arrival 没对齐 wtr/departure 的尾部 round
    (实验未结束) -> 丢掉只有 arrival 的最后那个。
    """
    rounds: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}
    for ev in events:
        if 'Request_arrival_time' in ev:
            if cur and 'departure' in cur:
                rounds.append(cur)
            cur = {'arrival': ev['Request_arrival_time']}
        elif 'waiting_to_running' in ev:
            cur['wtr'] = ev['waiting_to_running']
            cur['prompt_length'] = ev.get('prompt_length')
            cur['hit_length'] = ev.get('hit_length')
        elif 'Request_departure_time' in ev:
            cur['departure'] = ev['Request_departure_time']
            cur['num_gen'] = ev.get('num_generation_tokens')
    if cur and 'departure' in cur:
        rounds.append(cur)
    return rounds


def build_round_rows(prefill_data: dict[str, list[dict]],
                     decode_data: dict[str, list[dict]] | None
                     ) -> pd.DataFrame:
    """对每个 traj 的每轮生成一行 (含 T2 真值 + prefill/decode 元数据)."""
    rows: list[dict[str, Any]] = []
    for traj_id, events in prefill_data.items():
        pf_rounds = parse_traj_events(events)
        dc_rounds = parse_traj_events(decode_data.get(traj_id, [])) if decode_data else []
        for k, r in enumerate(pf_rounds):
            t2_s = r['wtr'] - r['arrival']
            t_pf_s = r['departure'] - r['wtr']
            row = {
                'traj_id': traj_id,
                'round_idx': k,
                'pf_arrival': r['arrival'],
                'pf_wtr': r['wtr'],
                'pf_departure': r['departure'],
                'T2_s': t2_s,
                'T_prefill_s': t_pf_s,
                'prompt_length': r.get('prompt_length'),
                'hit_length': r.get('hit_length'),
                'hit_rate': ((r.get('hit_length') or 0)
                             / max(1, r.get('prompt_length') or 1)),
            }
            # decode 信息: round k 的 decode (跟在 round k prefill 后面)
            if k < len(dc_rounds):
                dc = dc_rounds[k]
                row['dc_arrival'] = dc.get('arrival')
                row['dc_wtr'] = dc.get('wtr')
                row['dc_departure'] = dc.get('departure')
                row['T_decode_s'] = ((dc.get('departure') or 0)
                                     - (dc.get('wtr') or 0))
                row['decode_num_gen'] = dc.get('num_gen')
                # T1 (tool) = next_round_prefill_arrival - this_round_decode_departure
                if (k + 1 < len(pf_rounds) and dc.get('departure') is not None):
                    row['T1_tool_s'] = (pf_rounds[k + 1]['arrival']
                                        - dc['departure'])
                else:
                    row['T1_tool_s'] = float('nan')
            else:
                row['dc_arrival'] = row['dc_wtr'] = row['dc_departure'] = None
                row['T_decode_s'] = float('nan')
                row['decode_num_gen'] = None
                row['T1_tool_s'] = float('nan')
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def compute_queue_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """对每个 (traj, round) 在它的 pf_arrival 时刻, 算 prefill 队列状态.

    特征 (按 prefill 节点 timeline):
      qs_n_waiting   : 在我之前 arrival 但还没 wtr 的其他 traj 数
      qs_n_running   : 此刻正在 prefill 中 (wtr <= now < departure)
      qs_max_wait_others_s : 比我等更久的最大已等时长 (HOL block 估计)
      qs_sum_prompt_ahead  : 等在我前面的请求 prompt 总长度 (token-volume 拥堵)
    """
    # 准备每个 (traj, round) 在 prefill 时间轴的事件
    # 用 ndarray 加速: 排序后双指针扫
    events = df[['traj_id', 'round_idx', 'pf_arrival', 'pf_wtr',
                 'pf_departure', 'prompt_length']].copy()
    events = events.sort_values('pf_arrival').reset_index(drop=True)

    n_waitings = []
    n_runnings = []
    max_waits = []
    sum_prompts_ahead = []

    arrivals = events['pf_arrival'].to_numpy()
    wtrs = events['pf_wtr'].to_numpy()
    deps = events['pf_departure'].to_numpy()
    prompts = events['prompt_length'].fillna(0).to_numpy()

    n = len(events)
    for i in range(n):
        t = arrivals[i]
        # 谁在 t 时刻处于 waiting (arrival < t <= wtr)?
        # 谁在 t 时刻处于 running (wtr <= t < departure)?
        # 这里简单 O(n) 扫描; 数据量 ~万级一次性脚本可承受
        mask_wait = (arrivals <= t) & (wtrs > t)
        mask_wait[i] = False  # 排除自己
        mask_run = (wtrs <= t) & (deps > t)
        mask_run[i] = False
        n_w = int(mask_wait.sum())
        n_r = int(mask_run.sum())
        if n_w > 0:
            wait_durs = t - arrivals[mask_wait]
            max_wait = float(wait_durs.max())
            sp_ahead = float(prompts[mask_wait].sum())
        else:
            max_wait = 0.0
            sp_ahead = 0.0
        n_waitings.append(n_w)
        n_runnings.append(n_r)
        max_waits.append(max_wait)
        sum_prompts_ahead.append(sp_ahead)

    events['qs_n_waiting'] = n_waitings
    events['qs_n_running'] = n_runnings
    events['qs_max_wait_others_s'] = max_waits
    events['qs_sum_prompt_ahead'] = sum_prompts_ahead

    # merge 回原 df
    key = ['traj_id', 'round_idx']
    merged = df.merge(
        events[key + ['qs_n_waiting', 'qs_n_running',
                      'qs_max_wait_others_s', 'qs_sum_prompt_ahead']],
        on=key, how='left',
    )
    return merged


def report_stats(df: pd.DataFrame, only_round_ge: int = 1) -> None:
    print(f'\n=== 全集统计 ===')
    print(f'  trajectories: {df["traj_id"].nunique()}')
    print(f'  total rounds: {len(df)}')
    print(f'  rounds/traj: '
          f'p50={df.groupby("traj_id").size().median():.0f}, '
          f'max={df.groupby("traj_id").size().max()}, '
          f'mean={df.groupby("traj_id").size().mean():.1f}')

    sub = df[df['round_idx'] >= only_round_ge].copy()
    print(f'\n=== round >= {only_round_ge} 子集 (T2 预测目标) ===')
    print(f'  n_rounds = {len(sub)}')
    for col, unit in [('T2_s', 's'), ('T_prefill_s', 's'),
                       ('T_decode_s', 's'), ('T1_tool_s', 's'),
                       ('prompt_length', 'tok'), ('hit_rate', ''),
                       ('qs_n_waiting', ''), ('qs_n_running', ''),
                       ('qs_sum_prompt_ahead', 'tok')]:
        if col not in sub.columns:
            continue
        s = sub[col].dropna()
        if s.empty:
            continue
        print(f'  {col:30s}  p10={s.quantile(0.1):.3f} p50={s.median():.3f} '
              f'p90={s.quantile(0.9):.3f} p99={s.quantile(0.99):.3f} '
              f'max={s.max():.3f}  ({unit})')
    print(f'\nT2 分布 (round >= {only_round_ge}):')
    t2 = sub['T2_s']
    bins = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        c = int(((t2 >= lo) & (t2 < hi)).sum())
        if c:
            print(f'    [{lo:>6.3f}s, {hi:>6.3f}s):  {c:>6d}  '
                  f'({100*c/len(sub):5.1f}%)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefill-ts', default=(
        '/data/whr/vllm-continuum/examples/online_serving/'
        'disaggregated_serving_p2p_nccl_xpyd/continuum_exp/prefill_20003/'
        'scheduler_timestamps'))
    ap.add_argument('--decode-ts', default=(
        '/data/whr/vllm-continuum/examples/online_serving/'
        'disaggregated_serving_p2p_nccl_xpyd/continuum_exp/decode_20005/'
        'scheduler_timestamps'))
    ap.add_argument('--out', default=(
        '/data/whr/vllm-continuum/tool_call_time/t2/t2_ground_truth.parquet'))
    ap.add_argument('--out-csv', default=(
        '/data/whr/vllm-continuum/tool_call_time/t2/t2_ground_truth.csv'))
    ap.add_argument('--only-round-ge', type=int, default=1,
                    help='只统计 round_idx >= 此值 (默认 1, 跳 round 0 首次 prefill)')
    args = ap.parse_args()

    print(f'读取 prefill: {args.prefill_ts}')
    pf = json.load(open(args.prefill_ts))
    print(f'  trajs: {len(pf)}')
    print(f'读取 decode:  {args.decode_ts}')
    dc = json.load(open(args.decode_ts))
    print(f'  trajs: {len(dc)}')

    print(f'\n构造 round-level rows...')
    df = build_round_rows(pf, dc)
    print(f'  rows: {len(df)}')

    print(f'\n计算队列状态特征...')
    df = compute_queue_state_features(df)

    report_stats(df, only_round_ge=args.only_round_ge)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    df.to_csv(args.out_csv, index=False)
    print(f'\n写入:')
    print(f'  {args.out}')
    print(f'  {args.out_csv}')


if __name__ == '__main__':
    sys.exit(main())
