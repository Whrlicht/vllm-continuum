"""Step-count 准确性评估: simulator (step-driven, future arrival oracle) vs trace 真值.

Optional: pass --use-oracle to plug `licht_admit_probes` / `licht_admit_events`
from monitoring_timestamps into the simulator, giving per-step evictable_prefix
truth.  Default off — to compare baseline vs oracle accuracy side-by-side
just run twice.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulator import LichtV2StepSimulator, load_step_ends


MONITORING_PATH = (
    '/data/whr/vllm-continuum/examples/online_serving/'
    'disaggregated_serving_p2p_nccl_xpyd/continuum_exp/'
    'prefill_20003/monitoring_timestamps')


def build_oracles(step_ends: np.ndarray, build_probe_seq: bool = False
                  ) -> tuple[dict, dict, dict, dict, dict]:
    """Read licht_admit_probes / licht_admit_events and build oracles
    keyed by (traj_id, round_idx, ...) for the simulator:

      admit_event_oracle[(traj_id, K)]            -> evictable_prefix@admit
      probe_oracle      [(traj_id, K, step_id)]   -> evictable_prefix@probe
      first_probe_step_oracle[(traj_id, K)]       -> first scheduler step
                                                     where this request was
                                                     actually probed
      real_free_oracle[step_id]                   -> free_blocks_before_admit
                                                     of FIRST probe at this step
                                                     (= real LICHTV2's future_free[0]
                                                     post-running-loop)
    """
    with open(MONITORING_PATH) as f:
        mon = json.load(f)
    probes = mon.get('licht_admit_probes', [])
    events = mon.get('licht_admit_events', [])
    print(f'  loaded {len(probes)} probes, {len(events)} admit events')

    admit_event_oracle: dict = {}
    for e in events:
        key = (e['job_id'], int(e['agent_round']))
        admit_event_oracle.setdefault(key, int(e['evictable_prefix']))

    probe_ts = np.array([p['timestamp'] for p in probes])
    probe_steps = np.searchsorted(step_ends, probe_ts, side='right')

    probe_oracle: dict = {}
    first_probe_step: dict = {}
    real_free_oracle: dict = {}
    for p, s in zip(probes, probe_steps):
        s = int(s)
        key3 = (p['job_id'], int(p['agent_round']), s)
        probe_oracle.setdefault(key3, int(p['evictable_prefix']))
        key2 = (p['job_id'], int(p['agent_round']))
        prev = first_probe_step.get(key2)
        if prev is None or s < prev:
            first_probe_step[key2] = s
        # Capture the FIRST probe's free_blocks_before_admit per step
        # (i.e., post-running-loop, pre-any-waiting-admits).  This is
        # exactly the value real LICHTV2 sees as future_free[0] when it
        # enters the waiting loop.
        if s not in real_free_oracle:
            real_free_oracle[s] = int(p['free_blocks_before_admit'])
    print(f'  built admit_event_oracle:      {len(admit_event_oracle)} entries')
    print(f'  built probe_oracle:            {len(probe_oracle)} entries')
    print(f'  built first_probe_step_oracle: {len(first_probe_step)} entries')
    print(f'  built real_free_oracle:        {len(real_free_oracle)} entries')

    # D-2 mode: per-step probe sequence in real's order (sorted by timestamp).
    # Each entry: (traj_id, K, will_admit).  Sim's admit loop replays this
    # for non-target candidates, only invoking sim's can_admit for target —
    # validates sim's can_admit logic with the environment perfectly aligned.
    probe_seq_oracle: dict = {}
    if build_probe_seq:
        from collections import defaultdict
        per_step = defaultdict(list)
        for p, s in zip(probes, probe_steps):
            per_step[int(s)].append(
                (p['job_id'], int(p['agent_round']),
                 bool(p['will_admit']), float(p['timestamp'])))
        for s, items in per_step.items():
            items.sort(key=lambda x: x[3])
            probe_seq_oracle[s] = [(t, k, w) for (t, k, w, _) in items]
        print(f'  built probe_seq_oracle:        {len(probe_seq_oracle)} entries')

    return (admit_event_oracle, probe_oracle, first_probe_step,
            real_free_oracle, probe_seq_oracle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-oracle', action='store_true',
                    help='plug evictable_prefix oracle from probes/events')
    ap.add_argument('--d2-replay', action='store_true',
                    help='D-2 verification: replay real probe sequence for '
                         'non-target candidates; only target uses sim can_admit')
    ap.add_argument('--deployment-realistic', action='store_true',
                    help='Disable oracles scheduler does NOT have at '
                         'deployment time: future arrivals (#1), real '
                         'dep_step / R_actual (#6), probe_seq replay (#7). '
                         'Keep block_pool real-time state oracles (#2-#5).')
    args = ap.parse_args()

    gt_path = '/data/whr/vllm-continuum/queue_time/t2/t2_ground_truth_with_steps.parquet'
    print(f'读取真值: {gt_path}')
    df = pd.read_parquet(gt_path).reset_index(drop=True)
    print(f'  全集 rows: {len(df)}')
    eval_mask = (df['round_idx'] >= 1).to_numpy()
    print(f'  评估子集 (round >= 1): {eval_mask.sum()}')

    step_ends = load_step_ends()
    print(f'读取 {len(step_ends)} 个真实 step boundary, '
          f'mean step duration {np.diff(step_ends).mean():.2f}s')

    if args.use_oracle:
        print(f'\n构建 oracles (probes / events / first-probe-step / real_free) ...')
        admit_event_oracle, probe_oracle, fp_oracle, real_free, probe_seq = (
            build_oracles(step_ends, build_probe_seq=args.d2_replay))
        # Realign real_step_count to first-probe frame so it's directly
        # comparable to sim's prediction (which now starts from
        # first_probe_step too).  Without this, 20% of rows have a stale
        # pf_arrival-based real_step_count that is off by 1.
        traj_ids = df['traj_id'].to_numpy()
        rounds = df['round_idx'].to_numpy().astype(int)
        fp_steps = np.array([fp_oracle.get((traj_ids[i], int(rounds[i])),
                                            int(df['real_arrival_step'].iloc[i]))
                             for i in range(len(df))])
        df['first_probe_step'] = fp_steps
        df['real_step_count_pf'] = df['real_step_count'].copy()
        df['real_step_count'] = (df['real_admit_step'].to_numpy()
                                  - fp_steps).clip(min=0).astype(int)
        n_changed = (df['real_step_count'] != df['real_step_count_pf']).sum()
        print(f'  realigned real_step_count: {n_changed} rows changed '
              f'({100*n_changed/len(df):.1f}%)')
        sim = LichtV2StepSimulator(
            df, step_ends,
            admit_event_oracle=admit_event_oracle,
            probe_oracle=probe_oracle,
            first_probe_step_oracle=fp_oracle,
            real_free_oracle=real_free,
            probe_seq_oracle=(probe_seq if args.d2_replay
                              and not args.deployment_realistic else None),
            deployment_realistic=args.deployment_realistic,
        )
        if args.deployment_realistic:
            mode_tag = 'DEPLOYMENT-REALISTIC'
        elif args.d2_replay:
            mode_tag = 'WITH-ORACLE+D2-REPLAY'
        else:
            mode_tag = 'WITH-ORACLE'
    else:
        sim = LichtV2StepSimulator(df, step_ends)
        mode_tag = 'BASELINE'
    print(f'\n=== mode: {mode_tag} ===')

    t0 = time.time()
    print(f'\n批量预测 step count (全 oracle: future arrival + 真实 step boundary)...')
    pred = sim.predict_batch(np.arange(len(df)))
    elapsed = time.time() - t0
    print(f'  耗时 {elapsed:.0f}s ({elapsed/len(df)*1000:.1f} ms/sample)')

    df['sim_step_count'] = pred
    df['step_err'] = df['sim_step_count'] - df['real_step_count']

    sub = df[df['round_idx'] >= 1].copy()
    err = sub['step_err'].to_numpy()
    print(f'\n=== 全集 round >= 1 (n={len(sub)}) ===')
    print(f'  exact match:    {(err == 0).sum():>5d}  ({100*(err == 0).mean():.2f}%)')
    print(f'  within ±1:      {(np.abs(err) <= 1).sum():>5d}  '
          f'({100*(np.abs(err) <= 1).mean():.2f}%)')
    print(f'  within ±2:      {(np.abs(err) <= 2).sum():>5d}  '
          f'({100*(np.abs(err) <= 2).mean():.2f}%)')
    print(f'  mean abs err:   {np.abs(err).mean():.3f} steps')
    print(f'  bias:           {err.mean():+.3f}')

    print(f'\n=== 按真实 step 分层 ===')
    for s in [0, 1, 2, 3, 5, 10, 20]:
        sub_s = sub[sub['real_step_count'] == s]
        if len(sub_s) >= 20:
            err_s = sub_s['step_err'].to_numpy()
            exact = (err_s == 0).mean()
            within1 = (np.abs(err_s) <= 1).mean()
            print(f'  real_step={s:>3d}  n={len(sub_s):>5d}  exact={exact*100:5.1f}%  '
                  f'within±1={within1*100:5.1f}%  '
                  f'pred p50={int(np.median(sub_s["sim_step_count"]))}  '
                  f'mean_err={err_s.mean():+.2f}')

    print(f'\n=== 按 K 分层 ===')
    for K_range in [(1, 1), (2, 3), (4, 7), (8, 15), (16, 999)]:
        sub_k = sub[(sub['round_idx'] >= K_range[0])
                     & (sub['round_idx'] <= K_range[1])]
        if len(sub_k) >= 30:
            err_k = sub_k['step_err'].to_numpy()
            exact = (err_k == 0).mean()
            print(f'  K∈[{K_range[0]:>2d},{K_range[1]:>3d}]  n={len(sub_k):>4d}  '
                  f'exact={exact*100:5.1f}%  within±1={(np.abs(err_k) <= 1).mean()*100:5.1f}%  '
                  f'mean_err={err_k.mean():+.2f}')

    # 错得最大的 10 个
    print(f'\n=== 错得最大的 10 个 case ===')
    worst = sub.iloc[np.argsort(-np.abs(sub['step_err'].to_numpy()))[:10]]
    for _, row in worst.iterrows():
        print(f'  traj={row["traj_id"][:35]:35s}  K={row["round_idx"]:>3d}  '
              f'real={row["real_step_count"]:>3d}  sim={row["sim_step_count"]:>3d}  '
              f'err={row["step_err"]:+d}  '
              f'P={row["prompt_length"]:>6d}  hit={row["hit_rate"]:.2f}')

    out_path = '/data/whr/vllm-continuum/queue_time/sim_step_predictions.csv'
    sub[['traj_id', 'round_idx', 'real_step_count', 'sim_step_count',
          'step_err', 'T2_s', 'prompt_length', 'hit_length']].to_csv(out_path, index=False)
    print(f'\n误差表写入: {out_path}')


if __name__ == '__main__':
    main()
