#!/usr/bin/env python3
"""Offline analysis of ShadowScheduler StepTimeModel predictions.

Reads `/data/whr/vllm-continuum/output/v3_step_time.jsonl`.  Each line:
  {ts, step_id, tokens, predicted_s, predicted_no_ar_s, actual_s,
   abs_err_s, model: {alpha, beta, rho, n_obs, ...}}

Reports:
  - Overall error distribution (with vs without AR correction)
  - Convergence: error in first 20% vs last 20% of observations
  - Per-token-bucket accuracy
  - Final model params
  - AR uplift: how much does the rho term improve over plain linear
"""
import json
import statistics
import sys
from collections import defaultdict


PATH = "/data/whr/vllm-continuum/output/v3_step_time.jsonl"


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def pct(v, p):
    if not v:
        return float("nan")
    sv = sorted(v)
    k = max(0, min(len(sv) - 1, int(round((len(sv) - 1) * p / 100.0))))
    return sv[k]


def dist(label, v, unit="s"):
    if not v:
        print(f"  {label}: (empty)")
        return
    print(f"  {label}: n={len(v):4d}  "
          f"min={min(v):.4f}{unit}  "
          f"p50={pct(v,50):.4f}{unit}  "
          f"mean={statistics.fmean(v):.4f}{unit}  "
          f"p90={pct(v,90):.4f}{unit}  "
          f"p99={pct(v,99):.4f}{unit}  "
          f"max={max(v):.4f}{unit}")


def mape(pred, actual):
    if actual <= 0:
        return float("nan")
    return abs(pred - actual) / actual * 100.0


def main():
    rows = load(PATH)
    if not rows:
        print(f"no data at {PATH}")
        return 1
    print(f"loaded {len(rows)} step observations\n")

    actuals = [r["actual_s"] for r in rows]
    preds = [r["predicted_s"] for r in rows]
    preds_no_ar = [r["predicted_no_ar_s"] for r in rows]
    abs_err = [abs(p - a) for p, a in zip(preds, actuals)]
    abs_err_no_ar = [abs(p - a) for p, a in zip(preds_no_ar, actuals)]
    mapes = [mape(p, a) for p, a in zip(preds, actuals)
             if a > 0]
    mapes_no_ar = [mape(p, a) for p, a in zip(preds_no_ar, actuals)
                   if a > 0]
    tokens = [r["tokens"] for r in rows]

    print("=== ACTUAL DURATION ===")
    dist("actual  ", actuals)
    print("\n=== TOKENS ===")
    dist("tokens  ", tokens, unit="")
    print("\n=== ABSOLUTE ERROR (predicted - actual) ===")
    dist("|err| AR    ", abs_err)
    dist("|err| no-AR ", abs_err_no_ar)
    print("\n=== MAPE (mean abs % err) ===")
    dist("MAPE  AR    ", mapes, unit="%")
    dist("MAPE  no-AR ", mapes_no_ar, unit="%")

    # AR uplift
    n_better = sum(1 for ar, nar in zip(abs_err, abs_err_no_ar) if ar < nar)
    n_worse  = sum(1 for ar, nar in zip(abs_err, abs_err_no_ar) if ar > nar)
    n_equal  = sum(1 for ar, nar in zip(abs_err, abs_err_no_ar) if ar == nar)
    print(f"\n=== AR vs no-AR per-step (head-to-head) ===")
    print(f"  AR better : {n_better} ({100*n_better/len(rows):.1f}%)")
    print(f"  AR worse  : {n_worse} ({100*n_worse/len(rows):.1f}%)")
    print(f"  tie       : {n_equal} ({100*n_equal/len(rows):.1f}%)")

    # Convergence: split into quintiles
    print("\n=== CONVERGENCE (per quintile of observation order) ===")
    n = len(rows)
    q = max(n // 5, 1)
    for i in range(5):
        lo = i * q
        hi = (i + 1) * q if i < 4 else n
        sub = [abs(r["predicted_s"] - r["actual_s"]) for r in rows[lo:hi]]
        sub_mape = [mape(r["predicted_s"], r["actual_s"]) for r in rows[lo:hi]
                    if r["actual_s"] > 0]
        if not sub:
            continue
        print(f"  quintile {i+1}  n={len(sub):4d}  "
              f"|err| p50={pct(sub,50):.4f}s  "
              f"MAPE p50={pct(sub_mape,50):.2f}%  "
              f"MAPE p90={pct(sub_mape,90):.2f}%")

    # Per-token bucket
    print("\n=== PER-TOKEN-BUCKET ACCURACY ===")
    buckets = [(0, 100, "0-100"),
               (100, 500, "100-500"),
               (500, 1500, "500-1.5K"),
               (1500, 4000, "1.5K-4K"),
               (4000, 8000, "4K-8K"),
               (8000, 999999, "8K+")]
    by_bucket = defaultdict(list)
    for r in rows:
        for lo, hi, name in buckets:
            if lo <= r["tokens"] < hi:
                by_bucket[name].append(r)
                break
    for _, _, name in buckets:
        sub = by_bucket.get(name, [])
        if not sub:
            continue
        sub_err = [abs(r["predicted_s"] - r["actual_s"]) for r in sub]
        sub_actual = [r["actual_s"] for r in sub]
        sub_mape = [mape(r["predicted_s"], r["actual_s"]) for r in sub
                    if r["actual_s"] > 0]
        print(f"  {name:>10s}  n={len(sub):4d}  "
              f"actual p50={pct(sub_actual,50):.3f}s  "
              f"|err| p50={pct(sub_err,50):.4f}s  "
              f"MAPE p50={pct(sub_mape,50):.2f}%")

    # Final model state
    last = rows[-1]
    m = last.get("model", {})
    print(f"\n=== FINAL MODEL STATE (last obs, step_id={last.get('step_id')}) ===")
    print(f"  alpha   : {m.get('alpha', float('nan')):.6f} s")
    print(f"  beta    : {m.get('beta', float('nan')):.3e} s/token  "
          f"(= {m.get('beta', float('nan'))*1e6:.2f} us/token)")
    print(f"  rho     : {m.get('rho', float('nan')):.3f}")
    print(f"  n_obs   : {m.get('n_obs')}")
    print(f"  n_refits: {m.get('n_refits')}")
    print(f"  resid_std: {m.get('residual_std', float('nan')):.4f} s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
