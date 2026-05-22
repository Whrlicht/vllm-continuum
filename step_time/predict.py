#!/usr/bin/env python3
"""Baseline step-time predictor for prefill scheduler steps.

Trains a simple linear model on per-step features extracted from
prefill_20003 monitoring_timestamps (= ground_truth.jsonl).

Why linear (not deeper):
  - The dominant feature is num_running (+0.40 correlation).  Tokens
    barely correlate (+0.02) because prefill is mostly batch-saturated.
  - 2204 samples is enough for a linear fit but limited for non-linear
    models without overfitting.
  - The user can swap in LightGBM / MLP later via the same train()/
    predict() interface.

Train/test split: time-ordered 80/20 (= first 80% train, last 20% test
— mimics deployment where we train on history and predict future).

Features used:
  - num_prompt_tokens        (compute load)
  - num_generation_tokens
  - num_running              (batch size, dominant)
  - num_waiting              (admission pressure indicator)
  - num_finished_requests    (per-step cleanup overhead)
  - kv_cache_usage           (memory pressure → potential swap delays)
  - prefix_cache_hits        (skipped compute)
  - effective_tokens         (= prompt - cache_hits, "real compute")

Output:
  - <out_dir>/model.json     (alpha, beta vector keyed by feature)
  - <out_dir>/eval.txt       (train/test stats, per-bucket diagnostics)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Optional


DEFAULT_GT = "/data/whr/vllm-continuum/step_time/ground_truth.jsonl"
DEFAULT_OUT_DIR = "/data/whr/vllm-continuum/step_time"
FEATURES = [
    # PRIMARY: per-step actual compute (derived from licht_admit_events
    # or logged by scheduler — see extract_ground_truth.py).  Replaces
    # `num_prompt_tokens` from iteration_stats which measured finished-
    # prefill prompt-total, not per-step compute (corr +0.02 vs +0.47).
    "num_scheduled_tokens",
    "num_new_admits",
    "num_new_admit_tokens",
    "num_generation_tokens",
    "num_running",
    "num_waiting",
    "num_finished_requests",
    "kv_cache_usage",
    "prefix_cache_hits",
]


def load_records(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Fill defaults for features that may be None when derivation
            # didn't have admit-event data for that step.
            for f in FEATURES:
                if r.get(f) is None:
                    r[f] = 0
            out.append(r)
    return out


def build_X_y(records: list[dict]) -> tuple[list[list[float]], list[float]]:
    X = [[float(r[f]) for f in FEATURES] + [1.0]   # 1.0 = bias
         for r in records]
    y = [float(r["duration_s"]) for r in records]
    return X, y


def normal_eq_fit(X: list[list[float]],
                  y: list[float]) -> list[float]:
    """Solve linear regression via numpy normal equations.
    Returns weights vector (= one per feature + bias)."""
    import numpy as np
    Xa = np.asarray(X, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    # Ridge regularisation to avoid singular when features collinear.
    n_feat = Xa.shape[1]
    lam = 1e-3
    A = Xa.T @ Xa + lam * np.eye(n_feat)
    b = Xa.T @ ya
    w = np.linalg.solve(A, b)
    return w.tolist()


def predict(record: dict, weights: list[float]) -> float:
    x = [float(record.get(f, 0) or 0) for f in FEATURES] + [1.0]
    return sum(x[i] * weights[i] for i in range(len(x)))


def pct(v: list[float], p: float) -> float:
    if not v:
        return float("nan")
    sv = sorted(v)
    k = max(0, min(len(sv) - 1, int(round((len(sv) - 1) * p / 100.0))))
    return sv[k]


def evaluate(records: list[dict], weights: list[float], label: str) -> str:
    abs_errs = []
    rel_errs = []
    for r in records:
        pred = predict(r, weights)
        actual = r["duration_s"]
        abs_errs.append(abs(pred - actual))
        if actual > 0.01:
            rel_errs.append(abs(pred - actual) / actual)
    n = len(abs_errs)
    lines = [
        f"\n=== {label} (n={n}) ===",
        f"  abs_err:  p50={pct(abs_errs, 50):.3f}s  "
        f"p90={pct(abs_errs, 90):.3f}s  "
        f"p99={pct(abs_errs, 99):.3f}s  "
        f"max={max(abs_errs):.3f}s",
        f"  rel_err:  p50={100*pct(rel_errs, 50):.1f}%  "
        f"p90={100*pct(rel_errs, 90):.1f}%  "
        f"p99={100*pct(rel_errs, 99):.1f}%",
        f"  mean abs err: {statistics.fmean(abs_errs):.3f}s",
        f"  mean rel err: {100*statistics.fmean(rel_errs):.1f}%",
    ]
    # within-band hit rate
    for band in (0.5, 1.0, 2.0):
        hits = sum(1 for e in abs_errs if e <= band)
        lines.append(f"  within ±{band:.1f}s: {hits}/{n} = "
                     f"{100*hits/n:.1f}%")
    return "\n".join(lines)


def diagnose_residuals(records: list[dict], weights: list[float]) -> str:
    """Per-num_running bucket residual analysis."""
    from collections import defaultdict
    by_run: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        p = predict(r, weights)
        by_run[r["num_running"]].append((p, r["duration_s"]))
    lines = ["\n=== Residuals per num_running bucket ===",
             "  running   n     pred_p50   actual_p50   bias       |err|_p50"]
    for nr in sorted(by_run.keys()):
        rs = by_run[nr]
        if len(rs) < 5:
            continue
        preds = [p for p, _ in rs]
        actuals = [a for _, a in rs]
        biases = [p - a for p, a in rs]
        abs_errs = [abs(p - a) for p, a in rs]
        lines.append(
            f"   {nr:>3d}   {len(rs):>4d}   {pct(preds,50):>7.2f}s   "
            f"{pct(actuals,50):>7.2f}s    "
            f"{statistics.fmean(biases):>+6.2f}s    "
            f"{pct(abs_errs,50):>5.2f}s")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default=DEFAULT_GT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="last fraction held out for test (time-ordered)")
    args = ap.parse_args()

    records = load_records(args.ground_truth)
    if not records:
        print("ERROR: no ground truth records", file=sys.stderr)
        return 1
    n = len(records)
    split = int(n * (1 - args.test_frac))
    train = records[:split]
    test = records[split:]
    print(f"loaded {n} records, train={len(train)}, test={len(test)}")

    X_tr, y_tr = build_X_y(train)
    weights = normal_eq_fit(X_tr, y_tr)

    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.json")
    model = {f: weights[i] for i, f in enumerate(FEATURES)}
    model["bias"] = weights[-1]
    with open(model_path, "w") as f:
        json.dump({"features": FEATURES, "weights": model}, f, indent=2)
    print(f"\nweights:")
    for k, v in model.items():
        print(f"  {k:>30s}: {v:+.6e}")

    # Evaluate
    eval_lines = []
    eval_lines.append(evaluate(train, weights, "TRAIN"))
    eval_lines.append(evaluate(test, weights, "TEST"))
    eval_lines.append(diagnose_residuals(test, weights))
    eval_text = "\n".join(eval_lines)
    print(eval_text)
    eval_path = os.path.join(args.out_dir, "eval.txt")
    with open(eval_path, "w") as f:
        f.write("Step-time predictor evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(eval_text)
    print(f"\nsaved: {model_path}, {eval_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
