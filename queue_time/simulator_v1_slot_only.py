"""LICHTv2 调度算法模拟器 - 离线预测 T2 排队时间.

输入: t2_ground_truth.parquet (含 9610 个 round-level 真值)
输出: 对每个 round 模拟出 T2_sim, 与 T2_actual 对比

模型假设
--------
1. **槽位模型**: prefill 节点最多并发 M 个请求 (vLLM 实际是 KV cache 容量 +
   max_num_seqs 联合限制, 经验上 p50 ≈ 12, p99 ≈ 24, 这里用 M 作可调参数).
2. **Oracle 服务时间**: 我们让每个被模拟器调度起来的请求使用其 *ground-truth*
   `pf_departure - pf_wtr` 作为服务时长. 这隔离 T2 预测错误 与 T_prefill 预测错误.
3. **无未来 arrival**: 模拟器只看 target 请求到达时已经在系统里的请求.
   未来 arrival 不可见 -> 系统性低估 T2 (论文上是 baseline residual).
4. **licht-v2 评分**: 直接复用 scheduler.py 的公式 (常数从代码取)
       score = A * log(1+K) + B * (1+K)^(-α) * max(twait - Tmax, 0)
   A=3.0, B=1.0, Tmax=120s, α=0.5.
5. **暂不模拟 backfill**: backfill 让小请求跳队, 是 T2 短尾的主因. 看模拟器
   误差分布后决定是否补.

公开 API
--------
    LichtV2Simulator(df, M=15)
       .predict_t2(target_idx) -> float (秒)
       .predict_batch(indices) -> np.ndarray
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ----- licht-v2 常数 (与 scheduler.py 一致) -----
LICHT_A: float = 3.0
LICHT_B: float = 1.0
LICHT_TMAX_S: float = 120.0
LICHT_ALPHA: float = 0.5


def licht_score(K: int, t_wait: float) -> float:
    """licht-v2 prefill 优先级评分 (大 = 优先)."""
    K = max(K, 0)
    wait_term = max(t_wait - LICHT_TMAX_S, 0.0)
    round_decay = (1.0 + K) ** (-LICHT_ALPHA)
    return LICHT_A * math.log1p(K) + LICHT_B * round_decay * wait_term


@dataclass(order=True)
class _RunningItem:
    """heap 元素: (finish_time, tiebreak_idx)."""
    finish_time: float
    idx: int = field(compare=False)


@dataclass
class _WaitingItem:
    """waiting 队列项. 比 dict 快, 缓存友好."""
    idx: int
    arrival: float
    K: int
    pf_duration: float
    is_target: bool = False


class LichtV2Simulator:
    """对单条 round 做 T2 模拟. 输入是全量 round 表 (含 oracle 字段)."""

    def __init__(self, df: pd.DataFrame, M: int = 15):
        """df 必须包含: pf_arrival, pf_wtr, pf_departure, round_idx, prompt_length."""
        required = {'pf_arrival', 'pf_wtr', 'pf_departure', 'round_idx',
                    'prompt_length'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'df 缺字段: {missing}')
        self.M = M
        # numpy 化以加速
        self.pf_arr = df['pf_arrival'].to_numpy()
        self.pf_wtr = df['pf_wtr'].to_numpy()
        self.pf_dep = df['pf_departure'].to_numpy()
        self.K = df['round_idx'].to_numpy()
        # 按 arrival 升序排过的 indices, 用作 candidate set 筛选
        self._sorted_arrival_idx = np.argsort(self.pf_arr)
        self._sorted_arrival_vals = self.pf_arr[self._sorted_arrival_idx]
        self._n = len(df)

    def _snapshot_at(self, target_idx: int) -> tuple[list[_RunningItem], list[_WaitingItem]]:
        """在 target 到达时刻取系统快照: running + waiting (不含 target 自己)."""
        t0 = self.pf_arr[target_idx]
        # 候选: pf_arrival <= t0 (找到 sorted_arrival 中 <= t0 的全部 idx, 比逐行
        # 检查快很多)
        cutoff = np.searchsorted(self._sorted_arrival_vals, t0, side='right')
        cand = self._sorted_arrival_idx[:cutoff]

        running: list[_RunningItem] = []
        waiting: list[_WaitingItem] = []
        for i in cand:
            if i == target_idx:
                continue
            wtr = self.pf_wtr[i]
            dep = self.pf_dep[i]
            if wtr <= t0:
                # 已经在 prefill, 但还没结束
                if dep > t0:
                    running.append(_RunningItem(dep, int(i)))
                # 已结束 dep <= t0 的不进 snapshot
            else:
                # 已 arrival 但还没 wtr -> 还在 waiting
                waiting.append(_WaitingItem(
                    idx=int(i),
                    arrival=float(self.pf_arr[i]),
                    K=int(self.K[i]),
                    pf_duration=float(dep - wtr),
                ))
        return running, waiting

    def predict_t2(self, target_idx: int, return_diag: bool = False
                   ) -> float | dict:
        """模拟 target 请求的 T2. return_diag=True 返回诊断 dict."""
        t0 = float(self.pf_arr[target_idx])
        running, waiting = self._snapshot_at(target_idx)
        n_waiting_init = len(waiting)
        n_running_init = len(running)

        # 把 target 加进 waiting
        target_pf_dur = float(self.pf_dep[target_idx] - self.pf_wtr[target_idx])
        target_K = int(self.K[target_idx])
        waiting.append(_WaitingItem(
            idx=target_idx, arrival=t0, K=target_K,
            pf_duration=target_pf_dur, is_target=True,
        ))

        heapq.heapify(running)
        now = t0
        steps = 0
        admit_log = []

        while waiting:
            # 槽位满, 等下一次释放
            while len(running) >= self.M:
                top = heapq.heappop(running)
                now = max(now, top.finish_time)
            # 槽位空, 取最高分 admit
            best_score = -float('inf')
            best_idx = -1
            for i, w in enumerate(waiting):
                # arrival 还没到的请求不能 admit (理论上 waiting 全是 <= now,
                # 因为我们没注入未来; 但 target 自己 arrival==t0<=now 也成立)
                if w.arrival > now:
                    continue
                s = licht_score(w.K, now - w.arrival)
                # tiebreak: 早到优先 (arrival 越小越好)
                if (s, -w.arrival) > (best_score, -waiting[best_idx].arrival
                                       if best_idx >= 0 else float('inf')):
                    best_score = s
                    best_idx = i
            if best_idx < 0:
                break  # 不应该发生
            admitted = waiting.pop(best_idx)
            steps += 1
            if admitted.is_target:
                t2 = max(now - t0, 0.0)
                if return_diag:
                    return {
                        'T2_sim': t2,
                        'n_waiting_init': n_waiting_init,
                        'n_running_init': n_running_init,
                        'steps_to_admit': steps,
                        'score_at_admit': best_score,
                    }
                return t2
            # 入 running
            finish = now + admitted.pf_duration
            heapq.heappush(running, _RunningItem(finish, admitted.idx))
            admit_log.append((admitted.idx, now, finish))

        # 不应该走到这里
        if return_diag:
            return {
                'T2_sim': float('nan'),
                'n_waiting_init': n_waiting_init,
                'n_running_init': n_running_init,
                'steps_to_admit': steps,
                'score_at_admit': float('nan'),
            }
        return float('nan')

    def predict_batch(self, indices: np.ndarray | list[int]) -> np.ndarray:
        """批量预测. 返回 T2_sim ndarray (与 indices 同序)."""
        out = np.empty(len(indices), dtype=float)
        for i, idx in enumerate(indices):
            out[i] = self.predict_t2(int(idx))
        return out
