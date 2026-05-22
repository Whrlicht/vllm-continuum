"""LICHTv2 调度算法 step-level 模拟器 - 完全离散, 不引入 wall-clock 估算.

核心约束
========
1. 用 trace 里 iteration_stats 的 2315 个真实 step 边界做仿真时钟.
   每次 simulator 推进 = 一次真实 scheduler step 调用.
2. 全部基于 step index, 不算秒数. 因此预测目标是 "等多少 step 才被 admit",
   等价于 ground truth = step_ends 在 (pf_arrival, pf_wtr] 之间的数量.
3. running 请求的 num_computed 严格按 "已经经过几个 step boundary" 推算:
       chunks_done = current_step_id - admit_step_id_r
4. running 释放规则照搬 LICHTV2: 完成最后一 chunk 后下一个 step 释放
       (即 release_step = admit_step_id_r + R).
5. 候选 admit 用真实 LICHTV2 的三道关 (Guard 1/2/3),
   apply_to_timeline 与 can_admit 严格 mirror.

step index 对齐
==============
step_end[s] = iteration_stats[s].timestamp = step s 的 model forward 结束时刻
schedule call of step s+1 runs at ≈ step_end[s].

对请求 i:
  arrival_step_i = 第一个 k 使 step_end[k] >= pf_arrival_i
  admit_step_i  = 第一个 k 使 step_end[k] >= pf_wtr_i  (即 admit 发生在 step admit_step_i+1 的 schedule call 中)
  Hmm wait — 让 schedule call k 处理在 (step_end[k-1], step_end[k]] 里到达的请求.
  对应: arrival_step_i = 第一个 k 使 step_end[k] >= pf_arrival_i (因为 t_a 在 (step_end[k-1], step_end[k]])
        admit_step_i  = 第一个 k 使 step_end[k] >= pf_wtr_i

real_step_count = admit_step_i - arrival_step_i.

仿真:
  从 step = arrival_step_target 开始, 每次 iter 推进一步.
  iter 0 时, waiting 含 target + 所有 arrival <= step_end[arrival_step_target] 但未 admit 的.
  iter k=admit_step_i - arrival_step_target 时 target admit → return k.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np
import pandas as pd


LICHT_A: float = 3.0
LICHT_B: float = 1.0
LICHT_TMAX_S: float = 120.0
LICHT_ALPHA: float = 0.5

NUM_GPU_BLOCKS: int = 16853
BLOCK_SIZE: int = 16
CHUNK_SIZE_TOKENS: int = 5242
MAX_ALLOC_PER_STEP_BLOCKS: int = 16621
MAX_NUM_SEQS: int = 256

LICHTV2_N: int = 50
LICHTV2_LONG_TAIL_HEADROOM_BLOCKS: int = int(0.025 * NUM_GPU_BLOCKS)
LICHTV2_MAX_LONG_BRIDGE: int = 2


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def licht_score(K: int, t_wait_s: float) -> float:
    """licht-v2 score (传 t_wait 秒, 用真实 wall-clock 算 hunger).
    虽然仿真是 step driven, hunger 项需要秒数, 用真实 step_end 推得的"已等多久"算."""
    K = max(K, 0)
    wait_term = max(t_wait_s - LICHT_TMAX_S, 0.0)
    round_decay = (1.0 + K) ** (-LICHT_ALPHA)
    return LICHT_A * math.log1p(K) + LICHT_B * round_decay * wait_term


def licht_v2_R_at(num_tokens: int, current_offset: int) -> int:
    remaining = max(num_tokens - current_offset, 0)
    return _ceil_div(remaining, CHUNK_SIZE_TOKENS)


def licht_v2_B_at(num_tokens: int, current_offset: int, t: int) -> int:
    Ri = licht_v2_R_at(num_tokens, current_offset)
    if not (0 <= t < Ri):
        return 0
    remaining_now = max(num_tokens - current_offset, 0)
    cum_t = current_offset + min(CHUNK_SIZE_TOKENS * (t + 1), remaining_now)
    cum_prev = current_offset + (
        min(CHUNK_SIZE_TOKENS * t, remaining_now) if t > 0 else 0)
    return max(_ceil_div(cum_t, BLOCK_SIZE) - _ceil_div(cum_prev, BLOCK_SIZE), 0)


def licht_v2_release_blocks(num_tokens: int, admit_anchor: int) -> int:
    net = max(num_tokens - admit_anchor, 0)
    return _ceil_div(net, BLOCK_SIZE)


@dataclass
class _Running:
    idx: int
    num_tokens: int       # P
    admit_anchor: int     # = hit_length
    admit_step_id: int    # 在 step admit_step_id 的 schedule call 被 admit
    num_computed: int = 0
    R_total: int = 0
    is_long_tail: bool = False
    released_at_step: int = -1
    evictable_prefix: int = 0

    def __post_init__(self):
        self.R_total = licht_v2_R_at(self.num_tokens, self.admit_anchor)
        self.is_long_tail = self.R_total > LICHTV2_N
        if self.released_at_step < 0:
            self.released_at_step = self.admit_step_id + self.R_total
        if self.num_computed == 0:
            self.num_computed = self.admit_anchor
        # evictable_prefix 默认 0, 与 _Waiting 一致

    def update_at_step(self, current_step: int) -> None:
        chunks_done = max(0, current_step - self.admit_step_id)
        R_actual = max(self.released_at_step - self.admit_step_id, 1)
        R_formula = licht_v2_R_at(self.num_tokens, self.admit_anchor)
        chunks_done = min(chunks_done, R_actual)
        total_tokens = max(self.num_tokens - self.admit_anchor, 0)
        if R_actual != R_formula and R_actual > 0:
            # 3.4% case: real did variable chunks.  Interpolate evenly
            # so sim's num_computed roughly matches real's.
            tokens_progressed = (total_tokens * chunks_done) // R_actual
        else:
            # 96.6% case: standard 5242/step.
            tokens_progressed = chunks_done * CHUNK_SIZE_TOKENS
        self.num_computed = self.admit_anchor + min(
            tokens_progressed, total_tokens)
        self.num_computed = min(self.num_computed, self.num_tokens)


@dataclass
class _Waiting:
    idx: int
    pf_arrival_s: float
    K: int
    num_tokens: int
    hit_length: int
    arrival_step_id: int  # 此请求最早能进 schedule call 的 step
    is_target: bool = False
    R_required: int = 0
    is_long_tail: bool = False
    evictable_prefix: int = 0  # admit 时 touch 的 prefix cache 块数

    def __post_init__(self):
        self.R_required = licht_v2_R_at(self.num_tokens, self.hit_length)
        self.is_long_tail = self.R_required > LICHTV2_N
        # evictable_prefix = admit 时在 free queue 中的 prefix cache 块数.
        # 默认 0: 在 multi-turn 场景下, prefix 多被并发请求共享, ref_cnt > 0
        # → 不在 free queue. 取 0 比 ceil(H/bs) 经验上更准.
        self.evictable_prefix = 0


class LichtV2StepSimulator:
    """完全 step-driven 的 LICHTV2 仿真器.

    主接口:
        sim = LichtV2StepSimulator(df, step_ends)
        sim.predict_step_count(target_idx) -> int (0, 1, 2, ...)
    """

    def __init__(self, df: pd.DataFrame, step_ends: np.ndarray,
                 max_sim_steps: int = 1000,
                 admit_event_oracle: dict | None = None,
                 probe_oracle: dict | None = None,
                 first_probe_step_oracle: dict | None = None,
                 real_free_oracle: dict | None = None,
                 probe_seq_oracle: dict | None = None,
                 deployment_realistic: bool = False):
        required = {'pf_arrival', 'pf_wtr', 'pf_departure', 'round_idx',
                    'prompt_length', 'hit_length'}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f'df 缺字段: {miss}')
        self.df = df.reset_index(drop=True)
        self.step_ends = np.asarray(step_ends, dtype=float)
        self.max_sim_steps = max_sim_steps

        # 预先把每个 round 的 arrival_step_id / admit_step_id 算出来 (一次性)
        arr = self.df['pf_arrival'].to_numpy()
        wtr = self.df['pf_wtr'].to_numpy()
        # arrival_step_id: 第一个 k 使 step_ends[k] > pf_arrival.
        # NOTE: pf_arrival is when the request hit the engine; the scheduler
        # may not actually probe it until 1+ steps later (HTTP/queue lag,
        # plus boundary jitter at sub-millisecond resolution can shift a
        # request from step N to N+1).  When first_probe_step_oracle is
        # supplied, we override per-(traj, K) with the FIRST probe step
        # observed in real monitoring — eliminates the +1 alignment bug
        # that cascades into wrong admit ordering.
        self.arrival_step = np.searchsorted(self.step_ends, arr, side='right')
        if first_probe_step_oracle:
            traj_ids = self.df['traj_id'].to_numpy()
            rounds = self.df['round_idx'].to_numpy().astype(int)
            for i in range(len(self.df)):
                key = (traj_ids[i], int(rounds[i]))
                if key in first_probe_step_oracle:
                    self.arrival_step[i] = first_probe_step_oracle[key]
        # admit_step_id: 第一个 k 使 step_ends[k] > pf_wtr
        self.admit_step = np.searchsorted(self.step_ends, wtr, side='right')
        # 真实 departure 落在哪个 step: 第一个 k 使 step_ends[k] > pf_departure
        # 用它代替 admit_step + R 来推断 release 时刻 (避免 R 公式与实际不符的 5% 误差)
        dep = self.df['pf_departure'].to_numpy()
        self.departure_step = np.searchsorted(self.step_ends, dep, side='right')
        # P, H, K 转 ndarray
        self.P = self.df['prompt_length'].fillna(0).to_numpy().astype(int)
        self.H = self.df['hit_length'].fillna(0).to_numpy().astype(int)
        self.K = self.df['round_idx'].to_numpy().astype(int)
        self.pf_arr = arr
        # idx -> (traj_id, round_idx) for oracle lookup
        self._traj_id = self.df['traj_id'].to_numpy()
        # Oracles from monitoring data:
        #   admit_event_oracle[(traj_id, K)] -> evictable_prefix at admit
        #     used to set evictable_prefix on running snapshots at sim start
        #   probe_oracle[(traj_id, K, step_id)] -> evictable_prefix at probe
        #     used to refresh each waiting candidate's evictable_prefix
        #     at every sim step where it's checked (so we DON'T have to
        #     assume the value is invariant between probe step and admit
        #     step — see earlier discussion).
        self.admit_event_oracle = admit_event_oracle or {}
        self.probe_oracle = probe_oracle or {}
        #   real_free_oracle[step_id] -> free_blocks_before_admit of the
        #     FIRST probe at that step (== real's `block_pool.get_num_free_blocks()`
        #     POST-running-loop, PRE-any-waiting-admits).  Used to anchor
        #     sim's `future_free[0]` at the EXACT same value as real's
        #     LICHTV2 timeline at the start of the waiting loop.
        #     Eliminates the +5 baseline offset (perpetually-pinned system
        #     prompt blocks) and any per-step chunk-tracking drift.
        self.real_free_oracle = real_free_oracle or {}
        #   probe_seq_oracle[step_id] -> list of (traj_id, K, will_admit)
        #     in real's probe order at that step.  When set, sim's admit
        #     loop replays real's non-target admits in real's order
        #     (D-2 verification mode: validates sim's can_admit logic
        #     against target with environment perfectly aligned to real).
        self.probe_seq_oracle = probe_seq_oracle or {}
        # Deployment-realistic mode: disable oracles scheduler doesn't have
        # at deployment time:
        #   - Oracle #1: future arrivals (sim only sees reqs in queue at
        #     target's start_step; no new arrivals during simulation)
        #   - Oracle #6: real's dep_step for `released_at_step` (sim uses
        #     R_formula prediction instead)
        #   - Oracle #7: D-2 probe_seq replay (sim makes its own admit
        #     decisions throughout — equivalent to probe_seq_oracle = None)
        self.deployment_realistic = deployment_realistic

    def _build_timeline(self, running: list[_Running],
                        current_step: int | None = None
                        ) -> tuple[list[int], list[int], int]:
        """严格 mirror scheduler.py _licht_v2_build_timeline.

        每个 running 的物理占用 = (num_computed - admit_anchor) 部分的 alloc + evictable_prefix.
        evictable_prefix 在 admit 时离开 free queue, 在 release (t=R_i) 时回到 free.

        When current_step is provided AND real_free_oracle has data for it,
        the BASE current_free is set so that `future_free[0]` (post-running-
        loop, pre-waiting-admits) equals real's first-probe `free_blocks_before_admit`
        at that step.  This ensures sim's `_licht_v2_can_admit` at step
        `current_step` reads the SAME `future_free[0]` value as the real
        LICHTV2 scheduler saw — eliminating the +5 baseline offset (system
        prompt cache always pinned) and any per-step chunk-tracking drift.
        """
        N = LICHTV2_N

        # IMPORTANT: real's `_licht_v2_build_timeline` uses R_formula
        # (via _licht_v2_R_at) for its own future predictions — real
        # does NOT use R_actual.  R_actual is only what actually
        # happens, but real's `can_admit` decision relies on R_formula
        # PREDICTIONS.  For sim to faithfully replicate real's predictions
        # (and thus real's admit decisions), sim must also use R_formula
        # everywhere in build_timeline — NOT R_actual.
        per_r: list[int] = [licht_v2_R_at(r.num_tokens, r.num_computed)
                            for r in running]

        # sum_t0 uses same B_at as the delta loop below — keeps oracle
        # anchor consistent.
        sum_t0 = sum(
            licht_v2_B_at(r.num_tokens, r.num_computed, 0)
            for r in running
        )

        if (current_step is not None
                and current_step in self.real_free_oracle):
            current_free = (self.real_free_oracle[current_step] + sum_t0)
        else:
            blocks_held = sum(
                _ceil_div(max(r.num_computed - r.admit_anchor, 0), BLOCK_SIZE)
                + r.evictable_prefix
                for r in running
            )
            current_free = NUM_GPU_BLOCKS - blocks_held

        future_free = [0] * (N + 1)
        future_alloc = [0] * (N + 1)
        prev = current_free
        for t in range(N + 1):
            delta_free = 0
            delta_alloc = 0
            for r, Ri in zip(running, per_r):
                cur = r.num_computed
                if t < Ri:
                    bit = licht_v2_B_at(r.num_tokens, cur, t)
                    delta_free -= bit
                    delta_alloc += bit
                elif t == Ri:
                    delta_free += (
                        licht_v2_release_blocks(r.num_tokens, r.admit_anchor)
                        + r.evictable_prefix)
            future_free[t] = prev + delta_free
            future_alloc[t] = delta_alloc
            prev = future_free[t]
        return future_free, future_alloc, current_free

    def _can_admit(self, w: _Waiting,
                   future_free: list[int], future_alloc: list[int],
                   n_long_running: int) -> bool:
        """严格 mirror scheduler.py _licht_v2_can_admit."""
        N = LICHTV2_N
        Rj = w.R_required
        if Rj <= 0:
            return True
        if w.is_long_tail and n_long_running + 1 > LICHTV2_MAX_LONG_BRIDGE:
            return False
        threshold = LICHTV2_LONG_TAIL_HEADROOM_BLOCKS if w.is_long_tail else 0
        max_alloc = MAX_ALLOC_PER_STEP_BLOCKS
        ep = w.evictable_prefix

        cum_delta = 0
        for t in range(N + 1):
            if t == 0:
                cum_delta -= ep  # admit 时 touch 的 prefix cache 块离开 free queue
            bit_j = 0
            if t < Rj:
                bit_j = licht_v2_B_at(w.num_tokens, w.hit_length, t)
                cum_delta -= bit_j
            elif t == Rj:
                cum_delta += licht_v2_release_blocks(w.num_tokens, w.hit_length) + ep
            if future_free[t] + cum_delta < threshold:
                return False
            if t < Rj and future_alloc[t] + bit_j > max_alloc:
                return False
        return True

    def _apply_to_timeline(self, w: _Waiting,
                           future_free: list[int], future_alloc: list[int]
                           ) -> None:
        """严格 mirror scheduler.py _licht_v2_apply_to_timeline."""
        N = LICHTV2_N
        Rj = w.R_required
        if Rj <= 0:
            return
        ep = w.evictable_prefix
        cum_delta = 0
        for t in range(N + 1):
            if t == 0:
                cum_delta -= ep
            if t < Rj:
                bit = licht_v2_B_at(w.num_tokens, w.hit_length, t)
                cum_delta -= bit
                future_alloc[t] += bit
            elif t == Rj:
                cum_delta += licht_v2_release_blocks(w.num_tokens, w.hit_length) + ep
            future_free[t] += cum_delta

    def predict_step_count(self, target_idx: int) -> int:
        """预测 target 从 arrival 到 admit 等几个 step."""
        target_idx = int(target_idx)
        t_a_target = self.pf_arr[target_idx]
        start_step = int(self.arrival_step[target_idx])

        # 初始化 running / waiting / future_arrivals
        # running: pf_wtr_i <= step_ends[start_step - 1] (即 admit_step < start_step)
        # waiting: arrival_step <= start_step AND admit_step > start_step (即在 start_step 还没 admit)
        # 但要把 target 加进去
        running: list[_Running] = []
        waiting: list[_Waiting] = []
        future: list[_Waiting] = []  # arrival_step > start_step

        admit_step_arr = self.admit_step
        arrival_step_arr = self.arrival_step

        for i in range(len(self.df)):
            if i == target_idx:
                continue
            arr_s = int(arrival_step_arr[i])
            admit_s = int(admit_step_arr[i])

            if admit_s < start_step:
                # 真实 self.running 移除 = step admit + R (last chunk step 结束).
                # data 里 pf_departure 落在 step admit + R - 1 (=dep_s), 所以
                # released_at = dep_s + 1.
                P_i = int(self.P[i]); H_i = int(self.H[i])
                if self.deployment_realistic:
                    # No oracle #6: derive released_at from R_formula, not
                    # real's dep_step.  Sim doesn't know future dep at
                    # deployment time.
                    R_total = licht_v2_R_at(P_i, H_i)
                    released_at = admit_s + R_total
                else:
                    dep_s = int(self.departure_step[i])
                    released_at = dep_s + 1
                if released_at <= start_step:
                    continue  # 已释放
                ep_i = self.admit_event_oracle.get(
                    (self._traj_id[i], int(self.K[i])), 0)
                r = _Running(
                    idx=i, num_tokens=P_i, admit_anchor=H_i,
                    admit_step_id=admit_s, released_at_step=released_at,
                    evictable_prefix=ep_i,
                )
                r.update_at_step(start_step)
                running.append(r)
            elif arr_s < start_step:
                # 严格在 start_step 这一 call 之前到达 -> 已在 waiting 队列
                waiting.append(_Waiting(
                    idx=i, pf_arrival_s=float(self.pf_arr[i]),
                    K=int(self.K[i]),
                    num_tokens=int(self.P[i]), hit_length=int(self.H[i]),
                    arrival_step_id=arr_s,
                ))
            else:
                # arr_s >= start_step: future arrival.
                # If deployment_realistic, scheduler at deployment time
                # CAN'T see future arrivals — skip these entirely.
                if self.deployment_realistic:
                    continue
                future.append(_Waiting(
                    idx=i, pf_arrival_s=float(self.pf_arr[i]),
                    K=int(self.K[i]),
                    num_tokens=int(self.P[i]), hit_length=int(self.H[i]),
                    arrival_step_id=arr_s,
                ))
        future.sort(key=lambda w: w.arrival_step_id)

        # 加 target
        target_w = _Waiting(
            idx=target_idx, pf_arrival_s=float(t_a_target),
            K=int(self.K[target_idx]),
            num_tokens=int(self.P[target_idx]),
            hit_length=int(self.H[target_idx]),
            arrival_step_id=start_step,
            is_target=True,
        )
        waiting.append(target_w)

        # 仿真主循环
        fa_ptr = 0
        for sim_step in range(self.max_sim_steps):
            current_step = start_step + sim_step

            # 更新 running 的 num_computed; 处理 release
            new_running = []
            n_long = 0
            for r in running:
                if current_step >= r.released_at_step:
                    continue  # 已释放
                r.update_at_step(current_step)
                if licht_v2_R_at(r.num_tokens, r.num_computed) > LICHTV2_N:
                    n_long += 1
                new_running.append(r)
            running = new_running

            # 注入 future arrival: arrival_step_id <= current_step
            while fa_ptr < len(future) and future[fa_ptr].arrival_step_id <= current_step:
                waiting.append(future[fa_ptr])
                fa_ptr += 1

            # Refresh each waiting candidate's evictable_prefix from
            # the probe oracle at THIS step.  Without this, sim is stuck
            # assuming evictable_prefix=0 — see discussion in chat
            # 2026-05-13 about probe-level monitoring.  Falls back to 0
            # when the candidate has no probe record at this step
            # (rare; happens at sim boundaries where sim and reality
            # disagree on which step the candidate is in the queue).
            for w in waiting:
                key = (self._traj_id[w.idx], w.K, current_step)
                w.evictable_prefix = self.probe_oracle.get(key, 0)

            # build timeline (pass current_step so the BASE can be anchored
            # to real LICHTV2's `_lv2_current_free` at that step via oracle).
            future_free, future_alloc, current_free = self._build_timeline(
                running, current_step=current_step)

            # waiting loop with backfill
            t_wall_now = self.step_ends[current_step - 1] if current_step > 0 else self.step_ends[0]
            # Helper: add a candidate to running after admit decision.
            def _admit_candidate(w: _Waiting) -> None:
                self._apply_to_timeline(w, future_free, future_alloc)
                idx = w.idx
                if self.deployment_realistic:
                    # No oracle #6: predict release via R_formula
                    R_total = licht_v2_R_at(w.num_tokens, w.hit_length)
                else:
                    R_total = (int(self.departure_step[idx])
                                - int(self.admit_step[idx]) + 1)
                released_at = current_step + max(R_total, 1)
                r = _Running(
                    idx=idx,
                    num_tokens=w.num_tokens,
                    admit_anchor=w.hit_length,
                    admit_step_id=current_step,
                    released_at_step=released_at,
                    evictable_prefix=w.evictable_prefix,
                )
                r.num_computed = w.hit_length
                running.append(r)

            # If probe_seq_oracle has data for this step, run REPLAY MODE:
            # iterate non-target admits in REAL's probe order, only let
            # target use its own can_admit (D-2 verification mode).
            seq = self.probe_seq_oracle.get(current_step) if self.probe_seq_oracle else None
            if seq is not None:
                target_jK = (self._traj_id[target_idx], int(self.K[target_idx]))
                waiting_by_jK = {(self._traj_id[w.idx], w.K): w for w in waiting}
                target_checked_in_seq = False
                target_admitted_in_seq = False
                for traj_jid, K_real, will in seq:
                    if len(running) >= MAX_NUM_SEQS: break
                    key = (traj_jid, int(K_real))
                    if key == target_jK:
                        target_checked_in_seq = True
                        w_target = waiting_by_jK.get(key)
                        if w_target is None: continue
                        n_long_now = sum(1 for r in running
                                          if licht_v2_R_at(r.num_tokens, r.num_computed) > LICHTV2_N)
                        if self._can_admit(w_target, future_free, future_alloc, n_long_now):
                            target_admitted_in_seq = True
                            break
                        continue
                    if not will:
                        continue
                    w = waiting_by_jK.get(key)
                    if w is None:
                        continue
                    waiting.remove(w)
                    if w.is_long_tail: n_long += 1
                    _admit_candidate(w)
                if target_admitted_in_seq:
                    return sim_step
                # Fallback target check: if target wasn't in real's probe seq
                # for this step (e.g., real had already admitted target by
                # this step in its own trace), sim still needs to check
                # whether target should admit now under sim's can_admit.
                # Without this, target could be stuck in sim waiting forever.
                if not target_checked_in_seq:
                    w_target = next((w for w in waiting if w.is_target), None)
                    if w_target is not None:
                        n_long_now = sum(1 for r in running
                                          if licht_v2_R_at(r.num_tokens, r.num_computed) > LICHTV2_N)
                        if self._can_admit(w_target, future_free, future_alloc, n_long_now):
                            return sim_step
            else:
                # No probe seq oracle (or this step has no probe data) —
                # fall back to standard score-rank admit loop.
                while waiting:
                    if len(running) >= MAX_NUM_SEQS:
                        break
                    scored = sorted(
                        range(len(waiting)),
                        key=lambda j: (
                            -licht_score(waiting[j].K,
                                          t_wall_now - waiting[j].pf_arrival_s),
                            waiting[j].pf_arrival_s,
                        ),
                    )
                    admit_idx = -1
                    for j in scored:
                        if self._can_admit(waiting[j], future_free, future_alloc, n_long):
                            admit_idx = j
                            break
                    if admit_idx < 0:
                        break
                    admitted = waiting.pop(admit_idx)
                    if admitted.is_target:
                        return sim_step
                    if admitted.is_long_tail:
                        n_long += 1
                    _admit_candidate(admitted)

        return -1  # 仿真步数耗尽

    def predict_batch(self, indices) -> np.ndarray:
        out = np.empty(len(indices), dtype=int)
        for i, idx in enumerate(indices):
            out[i] = self.predict_step_count(int(idx))
        return out


def load_step_ends() -> np.ndarray:
    path = ('/data/whr/vllm-continuum/examples/online_serving/'
            'disaggregated_serving_p2p_nccl_xpyd/continuum_exp/prefill_20003/'
            'monitoring_timestamps')
    with open(path) as f:
        d = json.load(f)
    return np.array(sorted([x['timestamp'] for x in d['iteration_stats']]))
