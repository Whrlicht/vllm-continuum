"""详细追踪一个失败 case 的每一 sim 迭代, 看 target 为什么 admit 不上."""
import sys
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from simulator import (
    LichtV2StepSimulator, _Running, _Waiting, licht_score,
    licht_v2_R_at, licht_v2_B_at, licht_v2_release_blocks,
    LICHTV2_N, MAX_NUM_SEQS, MAX_ALLOC_PER_STEP_BLOCKS,
    LICHTV2_MAX_LONG_BRIDGE, LICHTV2_LONG_TAIL_HEADROOM_BLOCKS,
    NUM_GPU_BLOCKS, BLOCK_SIZE, CHUNK_SIZE_TOKENS, _ceil_div,
    load_step_ends,
)


def trace_target(df, step_ends, target_idx, max_iters=5):
    sim = LichtV2StepSimulator(df, step_ends)
    print(f'\n=== Target idx={target_idx} ===')
    target = df.iloc[target_idx]
    print(f'traj={target["traj_id"][:45]}, K={target["round_idx"]}, P={target["prompt_length"]}, '
          f'hit={target["hit_length"]}, T2_actual={target["T2_s"]:.3f}s, '
          f'real_step={target["real_step_count"]}')

    start_step = int(sim.arrival_step[target_idx])
    admit_step_target = int(sim.admit_step[target_idx])
    print(f'start_step (sim iter 0)={start_step}, admit_step={admit_step_target}, '
          f'real diff={admit_step_target - start_step}')

    # 重建 initial state (复制 predict_step_count 的逻辑但不返回)
    running, waiting, future = [], [], []
    for i in range(len(df)):
        if i == target_idx: continue
        arr_s = int(sim.arrival_step[i])
        admit_s = int(sim.admit_step[i])
        if admit_s < start_step:
            P_i = int(sim.P[i]); H_i = int(sim.H[i])
            R_i = licht_v2_R_at(P_i, H_i)
            if admit_s + R_i <= start_step: continue
            r = _Running(idx=i, num_tokens=P_i, admit_anchor=H_i, admit_step_id=admit_s)
            r.update_at_step(start_step)
            running.append(r)
        elif arr_s <= start_step:
            waiting.append(_Waiting(
                idx=i, pf_arrival_s=float(sim.pf_arr[i]), K=int(sim.K[i]),
                num_tokens=int(sim.P[i]), hit_length=int(sim.H[i]),
                arrival_step_id=arr_s,
            ))
        else:
            future.append(_Waiting(
                idx=i, pf_arrival_s=float(sim.pf_arr[i]), K=int(sim.K[i]),
                num_tokens=int(sim.P[i]), hit_length=int(sim.H[i]),
                arrival_step_id=arr_s,
            ))
    future.sort(key=lambda w: w.arrival_step_id)
    target_w = _Waiting(idx=target_idx, pf_arrival_s=float(target['pf_arrival']),
                        K=int(target['round_idx']), num_tokens=int(target['prompt_length']),
                        hit_length=int(target['hit_length']), arrival_step_id=start_step,
                        is_target=True)
    waiting.append(target_w)

    print(f'iter 0 start: n_running={len(running)}, n_waiting={len(waiting)} (含 target), n_future={len(future)}')

    fa_ptr = 0
    for sim_step in range(max_iters):
        current_step = start_step + sim_step
        # 更新 running, 处理 release
        new_running = []
        n_long = 0
        for r in running:
            if current_step >= r.released_at_step: continue
            r.update_at_step(current_step)
            if licht_v2_R_at(r.num_tokens, r.num_computed) > LICHTV2_N: n_long += 1
            new_running.append(r)
        running = new_running

        while fa_ptr < len(future) and future[fa_ptr].arrival_step_id <= current_step:
            waiting.append(future[fa_ptr]); fa_ptr += 1

        future_free, future_alloc, current_free = sim._build_timeline(running)
        t_wall_now = step_ends[current_step - 1] if current_step > 0 else step_ends[0]

        print(f'\n--- iter {sim_step} (current_step={current_step}, t_wall={t_wall_now:.1f}) ---')
        print(f'  running={len(running)} (n_long={n_long})')
        print(f'  waiting={len(waiting)}, current_free={current_free} / {NUM_GPU_BLOCKS}')
        print(f'  future_free[0..6]={future_free[:7]}')
        print(f'  future_alloc[0..6]={future_alloc[:7]}')

        # waiting 按 score 排
        scored = sorted(range(len(waiting)),
                        key=lambda j: (-licht_score(waiting[j].K, t_wall_now - waiting[j].pf_arrival_s),
                                        waiting[j].pf_arrival_s))
        # 这一 iter 内 admit 多少 (整个 waiting loop)
        admits_this_iter = []
        target_outcome = 'not_processed'
        target_failed_guard = None

        attempt_no = 0
        while waiting:
            if len(running) >= MAX_NUM_SEQS: break
            attempt_no += 1
            scored_now = sorted(range(len(waiting)),
                                key=lambda j: (-licht_score(waiting[j].K, t_wall_now - waiting[j].pf_arrival_s),
                                                waiting[j].pf_arrival_s))
            admit_idx = -1
            # 打印 attempt 时的 future_free 前 5 项
            if attempt_no <= 13:
                print(f'  attempt #{attempt_no}: future_free[0..4]={future_free[:5]}')
            for rank, j in enumerate(scored_now):
                w = waiting[j]
                if sim._can_admit(w, future_free, future_alloc, n_long):
                    if attempt_no <= 13:
                        print(f'    -> rank {rank}: K={w.K} P={w.num_tokens} can=True ADMIT')
                    admit_idx = j; break
                else:
                    if attempt_no <= 13:
                        # 打印失败原因
                        Rj = w.R_required
                        cum_delta = 0
                        fail_t = None
                        for t in range(LICHTV2_N + 1):
                            bit_j = 0
                            if t < Rj:
                                bit_j = licht_v2_B_at(w.num_tokens, w.hit_length, t)
                                cum_delta -= bit_j
                            elif t == Rj:
                                cum_delta += licht_v2_release_blocks(w.num_tokens, w.hit_length)
                            if future_free[t] + cum_delta < 0:
                                fail_t = (t, future_free[t], cum_delta); break
                        if fail_t:
                            print(f'    rank {rank}: K={w.K} P={w.num_tokens} FAIL Guard2 @ t={fail_t[0]} ff={fail_t[1]} cum={fail_t[2]}')
            if admit_idx < 0: break
            admitted = waiting.pop(admit_idx)
            if admitted.is_target:
                target_outcome = f'admitted_at_iter_{sim_step}'
                print(f'  ★ TARGET admitted at iter {sim_step} (pos in score={target_pos_in_score})')
                return
            admits_this_iter.append((admitted.K, admitted.num_tokens, admitted.idx))
            sim._apply_to_timeline(admitted, future_free, future_alloc)
            if admitted.is_long_tail: n_long += 1
            r = _Running(idx=admitted.idx, num_tokens=admitted.num_tokens,
                         admit_anchor=admitted.hit_length, admit_step_id=current_step)
            r.num_computed = admitted.hit_length
            running.append(r)

        print(f'  admits_this_iter ({len(admits_this_iter)}):')
        for k, p, i in admits_this_iter:
            print(f'    K={k:>3d}  P={p:>6d}  idx={i}')
        print(f'  target outcome: {target_outcome}')
        if target_failed_guard:
            print(f'  target failed: {target_failed_guard}')


if __name__ == '__main__':
    df = pd.read_parquet('/data/whr/vllm-continuum/queue_time/t2/t2_ground_truth_with_steps.parquet')
    step_ends = load_step_ends()
    # K=12, P=11343, real=0, sim=19 case
    target_df = df[(df['prompt_length'] == 11343) & (df['round_idx'] == 12) & (df['real_step_count'] == 0)]
    if len(target_df):
        trace_target(df, step_ends, int(target_df.index[0]), max_iters=3)
