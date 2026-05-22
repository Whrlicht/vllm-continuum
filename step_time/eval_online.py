#!/usr/bin/env python3
"""Replay ground_truth.jsonl through the online RLS predictor in TIME
ORDER, predict-then-observe each step, evaluate convergence and final
accuracy.

This is the strict online evaluation:
  - At each step t, we have x_t (computable in scheduler at schedule time)
  - Predict ŷ_t using ONLY data from steps 0..t-1 (= no future leak)
  - Observe true y_t after step finishes
  - Use (x_t, y_t) to update model for next prediction

What "online" buys vs offline batch
-----------------------------------
* No train/test split — every step that isn't step 0 contributes both
  a held-out prediction (before model sees its y) AND a training
  sample (after model sees its y).  Equivalent to leave-one-out across
  the time dimension, fully causal.
* Convergence over time is visible: early predictions are wild, late
  predictions stable.  We split into quintiles to see this.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

from online_predictor import OnlineStepTimePredictor


DEFAULT_GT = "/data/whr/vllm-continuum/step_time/ground_truth.jsonl"


def pct(v, p):
    if not v:
        return float("nan")
    sv = sorted(v)
    k = max(0, min(len(sv) - 1, int(round((len(sv) - 1) * p / 100.0))))
    return sv[k]


def fmt_dist(label, vals, unit="s"):
    if not vals:
        return f"{label}: (empty)"
    return (f"{label}: n={len(vals):>4d}  "
            f"p50={pct(vals,50):>7.3f}{unit}  "
            f"p90={pct(vals,90):>7.3f}{unit}  "
            f"p99={pct(vals,99):>7.3f}{unit}  "
            f"max={max(vals):>7.3f}{unit}  "
            f"mean={statistics.fmean(vals):>7.3f}{unit}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default=DEFAULT_GT)
    ap.add_argument("--lam", type=float, default=0.995,
                    help="forgetting factor λ (default 0.995)")
    ap.add_argument("--init-uncertainty", type=float, default=1000.0)
    ap.add_argument("--features",
                    default=("num_scheduled_tokens,num_running,"
                             "sum_L_sq,attention_waves"),
                    help="comma-separated feature names (read from ground "
                         "truth records).  Default = final 4-feature set "
                         "(user-confirmed 2026-05-19).")
    ap.add_argument("--out-log", default=None,
                    help="optional JSONL with per-step predictions")
    args = ap.parse_args()

    # Load
    records = []
    with open(args.ground_truth) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"loaded {len(records)} steps from {args.ground_truth}")

    # Skip records with missing num_scheduled_tokens (= no admit-event
    # match).  These would be steps where we have no compute-load info.
    records = [r for r in records if r.get("num_scheduled_tokens") is not None]
    print(f"  {len(records)} usable (have num_scheduled_tokens)")

    feature_names = [f.strip() for f in args.features.split(",") if f.strip()]
    print(f"  using features: {feature_names}")
    p = OnlineStepTimePredictor(
        feature_names=feature_names,
        forgetting_factor=args.lam,
        init_uncertainty=args.init_uncertainty)

    predictions = []  # one entry per step
    for r in records:
        y_true = float(r["duration_s"])
        y_pred = p.predict(r)
        predictions.append({
            "step_id": r["step_id"],
            "running": int(r.get("num_running", 0)),
            "actual": y_true,
            "pred": y_pred,
            "n_seen_before_predict": p.n_observed,
            "abs_err": (abs(y_pred - y_true) if y_pred is not None
                        else None),
        })
        # Observe (= what scheduler does after step finishes)
        p.observe(r, y_true)

    # Filter for prediction stats: skip step 0 (no prediction)
    valid = [pr for pr in predictions if pr["pred"] is not None]
    print(f"\npredictions: {len(valid)} / {len(predictions)}")
    print(f"  step 0 has no prediction (cold start) — as designed\n")

    abs_errs = [pr["abs_err"] for pr in valid]
    rel_errs = [pr["abs_err"] / pr["actual"]
                for pr in valid if pr["actual"] > 0.01]

    print("=== Overall ===")
    print("  " + fmt_dist("abs_err  ", abs_errs))
    print("  " + fmt_dist("rel_err  ", [r * 100 for r in rel_errs], "%"))
    for band in (0.3, 0.5, 1.0, 2.0):
        n_hit = sum(1 for e in abs_errs if e <= band)
        print(f"  within ±{band:.1f}s: {n_hit}/{len(abs_errs)} = "
              f"{100 * n_hit / len(abs_errs):.1f}%")

    # Convergence: 5 quintiles
    print("\n=== Convergence (5 quintiles, in time order) ===")
    n = len(valid)
    q = max(n // 5, 1)
    for i in range(5):
        lo, hi = i * q, ((i + 1) * q if i < 4 else n)
        sub = valid[lo:hi]
        if not sub:
            continue
        errs = [pr["abs_err"] for pr in sub]
        rels = [pr["abs_err"] / pr["actual"]
                for pr in sub if pr["actual"] > 0.01]
        print(f"  quintile {i+1} (step ~{sub[0]['step_id']}..{sub[-1]['step_id']}, "
              f"n={len(sub)}):  "
              f"|err| p50={pct(errs, 50):.3f}s  p90={pct(errs, 90):.3f}s  "
              f"MAPE p50={pct(rels, 50) * 100:.1f}%")

    # Within-band by quintile
    print("\n=== ±0.5s hit rate by quintile ===")
    for i in range(5):
        lo, hi = i * q, ((i + 1) * q if i < 4 else n)
        sub = valid[lo:hi]
        if not sub:
            continue
        hit = sum(1 for pr in sub if pr["abs_err"] <= 0.5)
        print(f"  quintile {i+1}: {hit}/{len(sub)} = "
              f"{100 * hit / len(sub):.1f}%")

    # Final model state
    print("\n=== Final model state ===")
    s = p.stats()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # Per-bucket diagnostics (num_running)
    print("\n=== |err| vs num_running (after warm-up = last 80% of steps) ===")
    warm_start = int(n * 0.2)
    warm = valid[warm_start:]
    from collections import defaultdict
    by_run = defaultdict(list)
    for pr in warm:
        by_run[pr["running"]].append(pr["abs_err"])
    print(f"  {'running':>8s}  {'n':>5s}  {'|err|_p50':>10s}  {'|err|_p90':>10s}")
    for nr in sorted(by_run.keys()):
        errs = by_run[nr]
        if len(errs) >= 5:
            print(f"  {nr:>8d}  {len(errs):>5d}  "
                  f"{pct(errs, 50):>9.3f}s  {pct(errs, 90):>9.3f}s")

    # Optional output log
    if args.out_log:
        with open(args.out_log, "w") as f:
            for pr in predictions:
                f.write(json.dumps(pr) + "\n")
        print(f"\nsaved per-step log: {args.out_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
