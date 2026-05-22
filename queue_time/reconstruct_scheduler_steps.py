"""Reconstruct scheduler-step granularity step_ends from R constraints.

Given:
  - For each request, pf_wtr and pf_departure (both scheduler-step boundary times)
  - R_i = ceil((P_i - H_i) / CHUNK_SIZE) chunks = scheduler steps spent in prefill
  - Under licht-v2: no preemption, so admit_step + R_i - 1 = depart_step exactly

Walk events in wall-time order, assign integer step indices using R-constraints.
Silent steps (no admit/depart) are inferred via R-constraints from overlapping requests.

Outputs:
  - scheduler_step_ends.npy: shape (S_max+1,), step_ends[k] = wall time of end of step k
  - t2_ground_truth_sched_steps.parquet: ground truth augmented with real_step_count_sched
"""
import numpy as np
import pandas as pd

CHUNK = 5242
INPUT_PATH = '/data/whr/vllm-continuum/queue_time/t2/t2_ground_truth.parquet'
OUTPUT_PARQUET = '/data/whr/vllm-continuum/queue_time/t2/t2_ground_truth_sched_steps.parquet'
OUTPUT_STEP_ENDS = '/data/whr/vllm-continuum/queue_time/t2/scheduler_step_ends.npy'


def main():
    df = pd.read_parquet(INPUT_PATH).reset_index(drop=True)
    df['R'] = (
        np.ceil((df['prompt_length'] - df['hit_length']) / CHUNK)
        .astype(int).clip(lower=1)
    )
    print(f'Loaded {len(df)} rounds')
    print(f'  R distribution: p50={int(df.R.quantile(0.5))}, '
          f'p90={int(df.R.quantile(0.9))}, p99={int(df.R.quantile(0.99))}, '
          f'max={df.R.max()}')

    # Collect events: (wall_time, type, request_idx, R)
    events = []
    for i, row in df.iterrows():
        events.append((float(row['pf_wtr']),       'admit',  i, int(row['R'])))
        events.append((float(row['pf_departure']), 'depart', i, int(row['R'])))
    # Sort by time, then admit before depart at same time
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'admit' else 1))

    # Group consecutive events with the same wall time
    groups = []  # list of (t, [(etype, ridx, R), ...])
    cur_t = None
    cur_evs = []
    for t, et, ri, R in events:
        if cur_t is None or t != cur_t:
            if cur_evs:
                groups.append((cur_t, cur_evs))
            cur_t = t
            cur_evs = []
        cur_evs.append((et, ri, R))
    if cur_evs:
        groups.append((cur_t, cur_evs))
    print(f'  unique event timestamps: {len(groups)}')

    # Constraint propagation: walk groups in time order
    S_admit = {}        # req_idx -> S(admit)
    unique_times = []   # wall time per group
    S_at_time = []      # S index assigned to each group
    prev_S = -1
    for t, evs in groups:
        # Minimum: previous time's S + 1
        candidates = [prev_S + 1]
        # Tighten via R-constraints from departing requests
        for et, ri, R in evs:
            if et == 'depart' and ri in S_admit:
                candidates.append(S_admit[ri] + R - 1)
        S_t = max(candidates)
        # Record admits seen at this S
        for et, ri, R in evs:
            if et == 'admit':
                S_admit[ri] = S_t
        unique_times.append(t)
        S_at_time.append(S_t)
        prev_S = S_t

    unique_times = np.array(unique_times)
    S_at_time = np.array(S_at_time, dtype=int)
    S_max = int(S_at_time.max())
    print(f'  reconstructed scheduler steps: {S_max + 1}')

    # Build per-step wall-time array step_ends[k] for k=0..S_max
    # Anchored: step_ends[S] = event time for S in S_at_time
    # Silent steps in between: linear interpolation
    step_ends = np.full(S_max + 1, np.nan)
    for t, S in zip(unique_times, S_at_time):
        step_ends[S] = t

    known_mask = ~np.isnan(step_ends)
    known_idx = np.where(known_mask)[0]
    known_t = step_ends[known_mask]
    # Linear interpolation across silent steps
    step_ends = np.interp(np.arange(S_max + 1), known_idx, known_t)
    diffs = np.diff(step_ends)
    print(f'  per-step duration: mean={diffs.mean()*1000:.1f}ms, '
          f'p50={np.median(diffs)*1000:.1f}ms, '
          f'p99={np.quantile(diffs, 0.99)*1000:.1f}ms')

    # Recompute real_step_count under scheduler-step granularity
    # Use side='right' to mirror simulator's searchsorted
    arr_steps = np.searchsorted(step_ends, df['pf_arrival'].to_numpy(), side='right')
    wtr_steps = np.searchsorted(step_ends, df['pf_wtr'].to_numpy(), side='right')
    dep_steps = np.searchsorted(step_ends, df['pf_departure'].to_numpy(), side='right')
    df['real_step_count_sched'] = (wtr_steps - arr_steps).astype(int)
    # Also recompute R from event-derived (just for sanity)
    df['R_derived'] = (dep_steps - wtr_steps + 1).astype(int)

    # Sanity check: R_derived should be very close to R (formula)
    delta = (df['R_derived'] - df['R']).abs()
    print(f'\n  R sanity check: |R_derived - R_formula| stats')
    print(f'    exact match: {(delta == 0).sum()} / {len(df)} '
          f'({100*(delta == 0).mean():.1f}%)')
    print(f'    within ±1: {(delta <= 1).sum()} '
          f'({100*(delta <= 1).mean():.1f}%)')
    print(f'    max delta: {delta.max()}')

    # Save
    np.save(OUTPUT_STEP_ENDS, step_ends)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f'\nSaved:\n  {OUTPUT_STEP_ENDS}\n  {OUTPUT_PARQUET}')

    sub = df[df['round_idx'] >= 1]
    print(f'\n=== real_step_count_sched 分布 (round>=1, n={len(sub)}) ===')
    print(f'  p50={int(sub.real_step_count_sched.quantile(0.5))}')
    print(f'  p90={int(sub.real_step_count_sched.quantile(0.9))}')
    print(f'  p99={int(sub.real_step_count_sched.quantile(0.99))}')
    print(f'  max={int(sub.real_step_count_sched.max())}')
    print(f'  mean={sub.real_step_count_sched.mean():.2f}')

    print(f'\n  value counts (top 30):')
    print(sub['real_step_count_sched'].value_counts().sort_index().head(30))


if __name__ == '__main__':
    main()
