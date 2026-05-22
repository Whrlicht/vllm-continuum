#!/usr/bin/env python3
"""Compare LICHTV3 K_queue predictions against the actual queue wait
observed at the prefill scheduler.

INPUTS
------
  /data/whr/vllm-continuum/output/v3_predictions.jsonl
      One record per decode-finish on the decode instance.  Keyed by
      (traj_id, agent_round, request_id_K).  K_queue_pred is the
      predicted number of prefill scheduler steps that round (K+1)
      will wait in the WAITING queue.

  /data/whr/vllm-continuum/output/v3_kqueue_actual.jsonl
      One record per WAITING -> RUNNING transition on prefill.  Keyed
      by request_id.  Carries `k_queue_actual` = admit_step - arrival_step
      and `wait_wall_s` = wall-clock seconds spent in waiting queue.

LINK
----
  pred(traj_id, K)  predicts  actual(traj_id, K+1).request_id
  → join by (traj_id, K+1) on the predictions table to get the
    K+1 request_id, then lookup actual by request_id.

OUTPUTS
-------
  Stdout summary (overall + per-K_queue_actual + worst cases).
  Optional --out-jsonl writes one merged record per matched pair.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict


PRED_LOG = "/data/whr/vllm-continuum/output/v3_predictions.jsonl"
ACTUAL_LOG = "/data/whr/vllm-continuum/output/v3_kqueue_actual.jsonl"


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"missing: {path}", file=sys.stderr)
        return []
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sv = sorted(values)
    k = max(0, min(len(sv) - 1, int(round((len(sv) - 1) * p / 100.0))))
    return sv[k]


def fmt_dist(label: str, values: list[float], unit: str = "") -> str:
    if not values:
        return f"{label}: (no data)"
    return (f"{label}: n={len(values)} "
            f"min={min(values):.3f}{unit} "
            f"p50={pct(values, 50):.3f}{unit} "
            f"mean={statistics.fmean(values):.3f}{unit} "
            f"p90={pct(values, 90):.3f}{unit} "
            f"p95={pct(values, 95):.3f}{unit} "
            f"max={max(values):.3f}{unit}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-log", default=PRED_LOG)
    ap.add_argument("--actual-log", default=ACTUAL_LOG)
    ap.add_argument("--out-jsonl", default=None,
                    help="optional path to write merged records")
    ap.add_argument("--show-worst", type=int, default=10,
                    help="print top-N largest |pred - actual| cases")
    args = ap.parse_args()

    preds = load_jsonl(args.pred_log)
    actuals = load_jsonl(args.actual_log)
    if not preds:
        print("no predictions — re-run the server with LICHTV3 decode "
              "enabled and let it write v3_predictions.jsonl")
        return 1
    if not actuals:
        print("no actuals — re-run prefill with "
              "LICHT_V3_LOG_KQUEUE_ACTUAL=1 (default) and check the path")
        return 1

    # Build (traj_id, round) -> pred record
    pred_by_tk: dict[tuple[str, int], dict] = {}
    for p in preds:
        t = p.get("traj_id")
        k = p.get("agent_round")
        if t is None or k is None:
            continue
        pred_by_tk[(str(t), int(k))] = p

    # Build request_id -> actual record
    actual_by_req: dict[str, dict] = {}
    for a in actuals:
        rid = a.get("request_id")
        if rid is None:
            continue
        # Last-write-wins (in case a request was re-admitted somehow)
        actual_by_req[rid] = a

    # Pair pred(K) with actual(K+1): pred predicts the NEXT round's
    # prefill queue wait.  We look up the K+1 record in predictions
    # to fetch its request_id, then look up that id in actuals.
    pairs: list[dict] = []
    n_pred_total = 0
    n_no_next_round_pred = 0
    n_no_actual_for_next = 0
    for (traj, k), p in pred_by_tk.items():
        n_pred_total += 1
        p_next = pred_by_tk.get((traj, k + 1))
        if p_next is None:
            n_no_next_round_pred += 1
            continue
        next_rid = p_next.get("request_id")
        if next_rid is None:
            n_no_next_round_pred += 1
            continue
        actual = actual_by_req.get(next_rid)
        if actual is None:
            n_no_actual_for_next += 1
            continue
        pairs.append({
            "traj_id": traj,
            "agent_round_pred": k,
            "agent_round_actual": k + 1,
            "request_id_at_K": p.get("request_id"),
            "request_id_at_K_plus_1": next_rid,
            "K_queue_pred": p.get("K_queue_pred"),
            "K_queue_actual": actual.get("k_queue_actual"),
            "wait_wall_s": actual.get("wait_wall_s"),
            "num_prompt_tokens": actual.get("num_prompt_tokens"),
            "num_running_at_admit": actual.get("num_running_at_admit"),
            "num_waiting_at_admit": actual.get("num_waiting_at_admit"),
            "T_tool_p50": p.get("T_tool_p50"),
            "T_tool_family": p.get("T_tool_family"),
        })

    print(f"=== JOIN STATS ===")
    print(f"  preds total                       : {n_pred_total}")
    print(f"  actuals total                     : {len(actuals)}")
    print(f"  paired (K with actual on K+1)     : {len(pairs)}")
    print(f"  preds without K+1 pred (last rnd) : {n_no_next_round_pred}")
    print(f"  preds with K+1 but no actual      : {n_no_actual_for_next}")

    if not pairs:
        print("\nno paired records — cannot compare")
        if args.out_jsonl:
            with open(args.out_jsonl, "w") as f:
                pass
        return 0

    # Distributions
    preds_v = [p["K_queue_pred"] for p in pairs
               if p["K_queue_pred"] is not None]
    actuals_v = [p["K_queue_actual"] for p in pairs
                 if p["K_queue_actual"] is not None]
    waits = [p["wait_wall_s"] for p in pairs
             if p["wait_wall_s"] is not None]

    print()
    print("=== PRED vs ACTUAL DISTRIBUTIONS ===")
    print("  " + fmt_dist("K_queue_pred  ", preds_v))
    print("  " + fmt_dist("K_queue_actual", actuals_v))
    print("  " + fmt_dist("wait_wall_s   ", waits, "s"))

    # Pred=0 means "expect immediate admit". Was it actually immediate?
    pred_zero = [p for p in pairs if p["K_queue_pred"] == 0
                 and p["K_queue_actual"] is not None]
    actual_zero_among_pred0 = sum(1 for p in pred_zero
                                  if p["K_queue_actual"] == 0)
    actual_le1_among_pred0 = sum(1 for p in pred_zero
                                 if p["K_queue_actual"] <= 1)
    print()
    print(f"=== PRED=0 SANITY ===")
    print(f"  pred==0 cases               : {len(pred_zero)}")
    if pred_zero:
        cov0 = 100.0 * actual_zero_among_pred0 / len(pred_zero)
        cov1 = 100.0 * actual_le1_among_pred0 / len(pred_zero)
        actuals_when_pred0 = [p["K_queue_actual"] for p in pred_zero]
        print(f"  of which actual == 0        : "
              f"{actual_zero_among_pred0}  ({cov0:.1f}%)")
        print(f"  of which actual <= 1        : "
              f"{actual_le1_among_pred0}  ({cov1:.1f}%)")
        print(f"  actual_when_pred0 max       : "
              f"{max(actuals_when_pred0)}")
        print(f"  actual_when_pred0 mean      : "
              f"{statistics.fmean(actuals_when_pred0):.2f}")

    # Errors
    err_pairs = [(p["K_queue_pred"], p["K_queue_actual"], p)
                 for p in pairs
                 if p["K_queue_pred"] is not None
                 and p["K_queue_actual"] is not None]
    abs_err = [abs(pr - ac) for pr, ac, _ in err_pairs]
    signed_err = [pr - ac for pr, ac, _ in err_pairs]
    over = sum(1 for s in signed_err if s > 0)
    under = sum(1 for s in signed_err if s < 0)
    exact = sum(1 for s in signed_err if s == 0)

    print()
    print(f"=== ERROR (pred - actual) ===")
    print(f"  n               : {len(err_pairs)}")
    print(f"  exact match     : {exact}  ({100*exact/len(err_pairs):.1f}%)")
    print(f"  overestimate    : {over}   ({100*over/len(err_pairs):.1f}%)")
    print(f"  underestimate   : {under}  ({100*under/len(err_pairs):.1f}%)")
    print(f"  |err| dist      : "
          f"p50={pct(abs_err, 50):.2f}  p90={pct(abs_err, 90):.2f}  "
          f"max={max(abs_err)}")
    print(f"  signed err mean : {statistics.fmean(signed_err):+.3f}")
    print(f"  signed err p50  : {pct(signed_err, 50):+.3f}")

    # Per-actual bucket
    by_actual: dict[int, list[int]] = defaultdict(list)
    for pr, ac, _ in err_pairs:
        by_actual[ac].append(pr)
    print()
    print("=== PRED DIST PER ACTUAL ===")
    print(f"  {'actual':>8s}  {'n':>5s}  {'pred_p50':>9s}  "
          f"{'pred_p90':>9s}  {'pred_max':>9s}")
    for ac in sorted(by_actual.keys()):
        prs = by_actual[ac]
        print(f"  {ac:>8d}  {len(prs):>5d}  {pct(prs,50):>9.1f}  "
              f"{pct(prs,90):>9.1f}  {max(prs):>9.0f}")

    # Worst cases
    if args.show_worst > 0:
        worst = sorted(err_pairs, key=lambda x: abs(x[0] - x[1]),
                       reverse=True)[:args.show_worst]
        print()
        print(f"=== TOP {len(worst)} WORST CASES ===")
        for pr, ac, p in worst:
            print(f"  pred={pr:>3d}  actual={ac:>3d}  "
                  f"wait={p['wait_wall_s']:.2f}s  "
                  f"prompt_tok={p['num_prompt_tokens']}  "
                  f"run_at_admit={p['num_running_at_admit']}  "
                  f"wait_at_admit={p['num_waiting_at_admit']}  "
                  f"traj={p['traj_id'][:50]}  K+1={p['agent_round_actual']}")

    # Write merged JSONL
    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"\nwrote merged: {args.out_jsonl} ({len(pairs)} records)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
