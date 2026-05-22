"""单步追踪 simulator 在某 case 下为什么 admit 失败."""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from simulator import (
    LichtV2Simulator, _Running, _Waiting,
    licht_score, licht_v2_R_at, licht_v2_B_at, licht_v2_release_blocks,
    LICHTV2_N, MAX_NUM_SEQS, MAX_ALLOC_PER_STEP_BLOCKS,
    LICHTV2_MAX_LONG_BRIDGE, LICHTV2_LONG_TAIL_HEADROOM_BLOCKS,
    NUM_GPU_BLOCKS, BLOCK_SIZE, CHUNK_SIZE_TOKENS, _ceil_div,
)


def debug_target(df: pd.DataFrame, target_idx: int):
    sim = LichtV2Simulator(df)
    t0 = float(df.iloc[target_idx]['pf_arrival'])
    print(f'\n=== Debug target idx={target_idx} ===')
    print(f'traj={df.iloc[target_idx]["traj_id"][:50]}')
    print(f'K={df.iloc[target_idx]["round_idx"]}, prompt={df.iloc[target_idx]["prompt_length"]}, '
          f'hit={df.iloc[target_idx]["hit_length"]}, T2_actual={df.iloc[target_idx]["T2_s"]:.3f}s')

    running, waiting, future_arrivals, emp_step = sim._snapshot_at(target_idx)
    print(f'\n初始 snapshot: n_running={len(running)}, n_waiting={len(waiting)} (不含 target), '
          f'n_future_arrivals={len(future_arrivals)}, empirical_step_s={emp_step:.3f}s')

    # 加 target
    P_tgt = int(df.iloc[target_idx]['prompt_length'])
    H_tgt = int(df.iloc[target_idx]['hit_length'])
    pf_dur_tgt = float(df.iloc[target_idx]['pf_departure'] - df.iloc[target_idx]['pf_wtr'])
    K_tgt = int(df.iloc[target_idx]['round_idx'])
    target_w = _Waiting(idx=target_idx, arrival=t0, K=K_tgt, num_tokens=P_tgt,
                        hit_length=H_tgt, pf_duration=pf_dur_tgt, is_target=True)
    waiting.append(target_w)

    R_tgt = licht_v2_R_at(P_tgt, H_tgt)
    blocks_tgt = licht_v2_release_blocks(P_tgt, H_tgt)
    print(f'\ntarget: R={R_tgt}, total_blocks_needed={blocks_tgt}')
    print(f'  B_at(t=0) = {licht_v2_B_at(P_tgt, H_tgt, 0)} blocks  (1 chunk)')
    print(f'  score = {licht_score(K_tgt, 0):.3f}')

    # 看 iter 0 state
    blocks_held = sum(_ceil_div(max(r.num_computed - r.admit_anchor, 0), BLOCK_SIZE) for r in running)
    current_free = NUM_GPU_BLOCKS - blocks_held
    print(f'\niter 0:')
    print(f'  blocks_held_sum (excl prefix): {blocks_held}')
    print(f'  current_free: {current_free} / {NUM_GPU_BLOCKS}')

    future_free, future_alloc = sim._build_timeline(running, current_free)
    print(f'  future_free[0..10]: {future_free[:10]}')
    print(f'  future_alloc[0..10]: {future_alloc[:10]}')

    # 排 waiting 按 score
    now = t0
    scored = sorted(range(len(waiting)),
                    key=lambda j: (-licht_score(waiting[j].K, now - waiting[j].arrival),
                                    waiting[j].arrival))
    print(f'\n  waiting (按 score 排):')
    n_long = sum(1 for r in running if licht_v2_R_at(r.num_tokens, r.num_computed) > LICHTV2_N)
    print(f'  n_long_running = {n_long}')
    for j in scored[:15]:
        w = waiting[j]
        s = licht_score(w.K, now - w.arrival)
        can = sim._can_admit(w, future_free, future_alloc, n_long)
        marker = '←TARGET' if w.is_target else ''
        Rj = licht_v2_R_at(w.num_tokens, w.hit_length)
        # 详细看 target 哪一关挂的
        if w.is_target:
            print(f'    score={s:6.2f}  K={w.K:>3d}  P={w.num_tokens:>6d}  H={w.hit_length:>6d}  '
                  f'R={Rj}  can_admit={can}  {marker}')
            # 逐关检查
            cum_delta = 0
            for t in range(LICHTV2_N + 1):
                if t == 0:
                    cum_delta -= w.evictable_prefix
                bit_j = 0
                if t < Rj:
                    bit_j = licht_v2_B_at(w.num_tokens, w.hit_length, t)
                    cum_delta -= bit_j
                elif t == Rj:
                    cum_delta += (licht_v2_release_blocks(w.num_tokens, w.hit_length)
                                   + w.evictable_prefix)
                # 检查 Guard 2
                ff = future_free[t] + cum_delta
                if ff < 0:
                    print(f'    Guard 2 FAIL at t={t}: future_free[{t}]={future_free[t]} '
                          f'+ cum_delta={cum_delta} = {ff} < 0')
                    break
                # 检查 Guard 3
                if t < Rj:
                    if future_alloc[t] + bit_j > MAX_ALLOC_PER_STEP_BLOCKS:
                        print(f'    Guard 3 FAIL at t={t}: future_alloc[{t}]={future_alloc[t]} '
                              f'+ B_j={bit_j} > {MAX_ALLOC_PER_STEP_BLOCKS}')
                        break
        else:
            print(f'    score={s:6.2f}  K={w.K:>3d}  P={w.num_tokens:>6d}  R={Rj}  can={can}')


if __name__ == '__main__':
    df = pd.read_parquet('/data/whr/vllm-continuum/queue_time/t2/t2_ground_truth_with_steps.parquet')
    # 找一个 +19 的 case
    bad = df[(df['round_idx'] >= 1) & (df['prompt_length'] == 11343)].head(1)
    if len(bad):
        debug_target(df, bad.index[0])
