#!/usr/bin/env python3
"""Evaluate the three LICHTV3 runtime predictions with the CORRECT
ground-truth semantics (validated 2026-05-20).

Three predictions
-----------------
1. STEP TIME   — RLS+correction predictor (online_step_time_predictor.py).
   Log: v3_step_time.jsonl.  Self-contained (predicted vs actual per step).

2. K_QUEUE     — binary "will this request enter RUNNING next prefill
   step?" (1=True, 2=False).  Log: v3_shadow_predictions.jsonl.
   Two stages:
     - Stage1 = prediction BEFORE the request arrives at prefill
                (decode-side foresight, estimated tokens).  This is the
                genuinely useful quick-admit-vs-wait classifier.
     - Stage2 = prediction AFTER arrival, re-simmed each step while it
                waits; the stored value is the last one before admit, so
                it acts as an "imminent-admit detector".  k=1 requests
                admit on arrival and never reach Stage2.
   Ground truth: actual_k_queue = admit_step - arrival_step (total wait).
     k=1 → should have been admitted next step (True).
     k>1 → had to wait (False at arrival).

3. TOOL TIME   — T_tool_p50/p95/p_timeout per (traj_id, agent_round).
   Pred log: v3_predictions.jsonl.
   Ground truth = execution_time_seconds (PURE tool exec time), nested in
   each assistant action message's tool_calls[0] in the trace file.
   NOTE: do NOT use shadow's real_tool_time_s — that is
   decode_finish→prefill_arrival and includes step-wait, NOT pure exec.
   Metrics follow tool_call_time/evaluate.py: log-MAE (not raw MAE,
   because tool time spans many orders), P95 coverage, timeout recall.

Usage
-----
  # everything (auto-detects logs in this dir, trace in trace_data/)
  python3 eval_v3_predictions.py

  # pick one
  python3 eval_v3_predictions.py --which step_time
  python3 eval_v3_predictions.py --which kqueue
  python3 eval_v3_predictions.py --which tool_time

  # custom paths
  python3 eval_v3_predictions.py \
      --step-time-log  output/v3_step_time.jsonl \
      --shadow-log     output/v3_shadow_predictions.jsonl \
      --pred-log       output/v3_predictions.jsonl \
      --trace          trace_data/swe_bench_sample_500_tool_clean_with_timings.json \
      --timeout-s      300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict

ROOT = "/data/whr/vllm-continuum"
DEF_STEP_TIME = f"{ROOT}/output/v3_step_time.jsonl"
DEF_SHADOW = f"{ROOT}/output/v3_shadow_predictions.jsonl"
DEF_PRED = f"{ROOT}/output/v3_predictions.jsonl"
DEF_TRACE = f"{ROOT}/trace_data/swe_bench_sample_500_tool_clean_with_timings.json"


def pct(v, p):
    if not v:
        return float("nan")
    sv = sorted(v)
    k = max(0, min(len(sv) - 1, int(round((len(sv) - 1) * p / 100.0))))
    return sv[k]


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        print(f"  WARN: missing {path}", file=sys.stderr)
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def log1p_ae(pred, act):
    return abs(math.log1p(max(pred, 0.0)) - math.log1p(max(act, 0.0)))


# ---------------------------------------------------------------------------
# 1. STEP TIME
# ---------------------------------------------------------------------------

def eval_step_time(path):
    print("=" * 72)
    print("1. STEP TIME  (RLS + correction)")
    print("=" * 72)
    recs = load_jsonl(path)
    pairs = [(r["actual_s"], r["predicted_s"]) for r in recs
             if r.get("predicted_s") is not None
             and r.get("actual_s") is not None]
    if not pairs:
        print("  no predictions found")
        return
    print(f"  steps logged: {len(recs)}, with prediction: {len(pairs)} "
          f"(step 0 cold-start skipped)")
    ys = [a for a, _ in pairs]
    abs_e = [abs(a - p) for a, p in pairs]
    abs_pe = [abs(a - p) / a for a, p in pairs if a > 0.01]
    n = len(pairs)
    under = sum(1 for a, p in pairs if p < a)
    print(f"\n  MAE        : {statistics.fmean(abs_e):.4f}s")
    print(f"  |err| p50  : {pct(abs_e, 50):.4f}s   p90: {pct(abs_e, 90):.4f}s   "
          f"p99: {pct(abs_e, 99):.4f}s   max: {max(abs_e):.4f}s")
    print(f"  MAPE       : {100 * statistics.fmean(abs_pe):.2f}%   "
          f"WAPE: {100 * sum(abs_e) / sum(ys):.2f}%   "
          f"P90 APE: {100 * pct(abs_pe, 90):.2f}%")
    for b in (0.3, 0.5, 1.0, 2.0):
        h = sum(1 for e in abs_e if e <= b)
        print(f"  within ±{b}s : {100 * h / n:.1f}%")
    print(f"  under-rate : {100 * under / n:.1f}%   "
          f"over-rate: {100 * (n - under) / n:.1f}%")
    # convergence
    q = n // 5
    print("  convergence (|err| p50 per quintile): ", end="")
    print("  ".join(f"Q{i+1}={pct(abs_e[i*q:(i+1)*q if i<4 else n], 50):.3f}s"
                    for i in range(5)))
    if recs and "model" in recs[-1] and "correction" in recs[-1]["model"]:
        c = recs[-1]["model"]["correction"]["c"]
        print(f"  correction c_final: {c:.4f}")


# ---------------------------------------------------------------------------
# 2. K_QUEUE (binary)
# ---------------------------------------------------------------------------

def eval_kqueue(path):
    print("=" * 72)
    print("2. K_QUEUE  (binary: 1=admit next step True, 2=False)")
    print("=" * 72)
    sp = load_jsonl(path)
    if not sp:
        print("  no shadow predictions found")
        return
    print(f"  shadow records: {len(sp)}")
    print("  ground truth: actual_k_queue==1 → quick admit (want pred 1);"
          "  >1 → waited (want pred 2 at arrival)")

    for key, label in [("pred_k_queue_stage1", "STAGE1 (pre-arrival foresight)"),
                       ("pred_k_queue_stage2", "STAGE2 (post-arrival, near admit)")]:
        cm = defaultdict(int)
        for r in sp:
            v = r.get(key)
            a = r.get("actual_k_queue")
            if v is None or a is None:
                continue
            ak = "k=1" if a == 1 else "k>1"
            pk = "p1" if v == 1 else ("p2" if v == 2 else None)
            if pk:
                cm[(ak, pk)] += 1
        n_k1 = cm[("k=1", "p1")] + cm[("k=1", "p2")]
        n_kg = cm[("k>1", "p1")] + cm[("k>1", "p2")]

        # Two evaluation modes depending on what reached this stage:
        #   - both k=1 and k>1 present  → quick-vs-wait CLASSIFIER (the
        #     prediction is "admit next step?" judged at arrival, so k=1
        #     should be pred=1 and k>1 should be pred=2).
        #   - only k>1 present          → IMMINENT-ADMIT DETECTOR.  The
        #     stored value is the LAST prediction before admit, which
        #     SHOULD be pred=1 (the request does admit next step from
        #     that point).  Judging it against "k>1 ⇒ pred=2" is wrong
        #     here — the right metric is the detection rate (% pred=1).
        print(f"\n  {label}:")
        if n_k1 > 0:
            # CLASSIFIER mode
            print(f"    mode: quick-vs-wait classifier")
            print(f"    {'':16s} {'pred=1(True)':>13s} {'pred=2(False)':>13s}")
            print(f"    actual k=1 ({n_k1:5d}) {cm[('k=1','p1')]:>13d} "
                  f"{cm[('k=1','p2')]:>13d}   "
                  f"correct(want 1)={100*cm[('k=1','p1')]/n_k1:.1f}%")
            print(f"    actual k>1 ({n_kg:5d}) {cm[('k>1','p1')]:>13d} "
                  f"{cm[('k>1','p2')]:>13d}   "
                  f"correct(want 2)={100*cm[('k>1','p2')]/max(n_kg,1):.1f}%")
            tp, fn = cm[("k=1", "p1")], cm[("k=1", "p2")]
            fp, tn = cm[("k>1", "p1")], cm[("k>1", "p2")]
            tot = tp + fn + fp + tn
            acc = (tp + tn) / max(tot, 1)
            print(f"    binary acc (quick vs wait): {100*acc:.1f}%   "
                  f"recall={100*tp/max(tp+fn,1):.0f}% "
                  f"precision={100*tp/max(tp+fp,1):.0f}%")
        else:
            # DETECTOR mode (only waited reqs reach here)
            detect = cm[("k>1", "p1")]
            miss = cm[("k>1", "p2")]
            tot = detect + miss
            print(f"    mode: imminent-admit detector  "
                  f"(only waited reqs reach here; k=1 admit on arrival)")
            print(f"    records: {tot} (all k>1)")
            print(f"    detected (pred=1 at admit): {detect}/{tot} = "
                  f"{100*detect/max(tot,1):.1f}%   ← detection rate")
            print(f"    missed   (pred=2 at admit): {miss}/{tot} = "
                  f"{100*miss/max(tot,1):.1f}%")


# ---------------------------------------------------------------------------
# 3. TOOL TIME
# ---------------------------------------------------------------------------

def build_exec_time_index(trace_path):
    """(traj_id, action_round) -> execution_time_seconds (pure tool exec)."""
    idx = {}
    if not os.path.exists(trace_path):
        print(f"  WARN: trace not found {trace_path}", file=sys.stderr)
        return idx
    trace = json.load(open(trace_path))
    for d in trace:
        tid = d.get("traj_id")
        msgs = d.get("messages")
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                continue
        if not isinstance(msgs, list):
            continue
        rnd = 0
        for m in msgs:
            if (isinstance(m, dict) and m.get("role") == "assistant"
                    and m.get("message_type") == "action"):
                tcs = m.get("tool_calls") or []
                et = (tcs[0].get("execution_time_seconds")
                      if tcs and isinstance(tcs[0], dict) else None)
                if et is not None:
                    idx[(tid, rnd)] = float(et)
                rnd += 1
    return idx


# ---- paper_eval.py methodology (Method 1/2/3), pure-python port ----

_HEAVY_BASH_PREFIXES = ("bash::pip::", "bash::conda::", "bash::apt::",
                        "bash::bg_server")
_HEAVY_BASH_BUCKETS = {
    "bash::python::script_repro", "bash::python::module_mypy",
    "bash::python::pytest_full_discovery", "bash::python::unittest_discover",
    "bash::python::module_other", "bash::find::exec_grep",
}


def tier_of(bucket):
    if not bucket:
        return "unknown"
    if bucket == "submit":
        return "constant_submit"
    if bucket.startswith("editor::"):
        return "constant_editor"
    if bucket.startswith("bash::light::") or bucket == "bash::cd_only":
        return "light_bash"
    if bucket.startswith("bash::find::") and "exec_grep" not in bucket:
        return "light_bash"
    if bucket.startswith("bash::git::"):
        return "light_bash"
    if bucket in _HEAVY_BASH_BUCKETS or bucket.startswith(_HEAVY_BASH_PREFIXES):
        return "heavy_bash"
    return "normal_bash"


def _acc_within(rs, abs_tol, rel_tol):
    """Method 2: fraction with |pred-actual| <= max(abs_tol, rel_tol*actual)."""
    if not rs:
        return float("nan")
    hit = 0
    for r in rs:
        tol = max(abs_tol, rel_tol * r["gt"])
        if abs(r["p50"] - r["gt"]) <= tol:
            hit += 1
    return hit / len(rs)


def _spearman(rs):
    """Rank correlation between pred p50 and actual, pure python."""
    if len(rs) < 3:
        return float("nan")
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    pr = ranks([r["p50"] for r in rs])
    ar = ranks([r["gt"] for r in rs])
    n = len(rs)
    mp = sum(pr) / n
    ma = sum(ar) / n
    num = sum((pr[i] - mp) * (ar[i] - ma) for i in range(n))
    dp = sum((x - mp) ** 2 for x in pr) ** 0.5
    da = sum((x - ma) ** 2 for x in ar) ** 0.5
    return num / (dp * da) if dp and da else float("nan")


def eval_tool_time(pred_path, trace_path, timeout_s):
    print("=" * 72)
    print("3. TOOL TIME  (vs pure execution_time_seconds; paper_eval.py method)")
    print("=" * 72)
    exec_time = build_exec_time_index(trace_path)
    print(f"  ground-truth exec times indexed: {len(exec_time)} (traj,round)")
    preds = load_jsonl(pred_path)
    if not preds:
        print("  no T_tool predictions found")
        return

    rows = []
    for p in preds:
        tid, rnd = p.get("traj_id"), p.get("agent_round")
        p50 = p.get("T_tool_p50")
        if tid is None or rnd is None or p50 is None:
            continue
        gt = exec_time.get((tid, rnd))
        if gt is None:
            continue
        bucket = p.get("T_tool_bucket")
        rows.append(dict(p50=p50, p95=p.get("T_tool_p95"),
                         ptmo=p.get("T_tool_p_timeout"), gt=gt,
                         src=p.get("T_tool_source"), bucket=bucket,
                         tier=tier_of(bucket),
                         is_to=1 if gt >= timeout_s else 0))

    # Method 1: per-bucket empirical P95 from this run's actuals; override
    # the (median-ish) runtime p95 for constant editor/submit families so
    # P95 coverage isn't definitionally capped at ~50%.
    by_bucket_actuals = defaultdict(list)
    for r in rows:
        by_bucket_actuals[r["bucket"]].append(r["gt"])
    bucket_p95 = {b: pct(v, 95) for b, v in by_bucket_actuals.items()
                  if len(v) >= 3}
    for r in rows:
        r["p95_raw"] = r["p95"]
        if r["tier"] in ("constant_editor", "constant_submit"):
            bp = bucket_p95.get(r["bucket"])
            if bp is not None:
                r["p95"] = max(bp, r["p50"])  # keep p95 >= p50

    def report(rs, label):
        if not rs:
            print(f"\n  [{label}] no matched rows")
            return
        log_ae = [log1p_ae(r["p50"], r["gt"]) for r in rs]
        n_p95 = sum(1 for r in rs if r["p95"] is not None)
        cov = sum(1 for r in rs if r["p95"] is not None and r["gt"] <= r["p95"])
        cov_raw = sum(1 for r in rs if r["p95_raw"] is not None
                      and r["gt"] <= r["p95_raw"])
        n_to = sum(r["is_to"] for r in rs)
        print(f"\n  [{label}] n={len(rs)}  actual: p50={pct([r['gt'] for r in rs],50):.3f}s "
              f"mean={statistics.fmean([r['gt'] for r in rs]):.3f}s")
        print(f"    log-MAE        : {statistics.fmean(log_ae):.4f}  "
              f"(p95={pct(log_ae,95):.4f})")
        print(f"    Spearman ρ     : {_spearman(rs):.4f}")
        # print(f"    Acc@(5ms,5%)   : {100*_acc_within(rs,0.005,0.05):.1f}%")
        # print(f"    Acc@(10ms,10%) : {100*_acc_within(rs,0.010,0.10):.1f}%")
        # print(f"    Acc@(50ms,20%) : {100*_acc_within(rs,0.050,0.20):.1f}%")
        print(f"    P95 cover      : {100*cov/max(n_p95,1):.1f}%  "
            #   f"(raw runtime p95: {100*cov_raw/max(n_p95,1):.1f}%; "
              f"target ≥95%)")
        if n_to:
            tp = sum(1 for r in rs if r["ptmo"] and r["ptmo"] >= 0.5 and r["is_to"])
            fp = sum(1 for r in rs if r["ptmo"] and r["ptmo"] >= 0.5 and not r["is_to"])
            fn = sum(1 for r in rs if (not r["ptmo"] or r["ptmo"] < 0.5) and r["is_to"])
            print(f"    timeout (≥{timeout_s:.0f}s, n={n_to}): "
                  f"recall@0.5={100*tp/max(tp+fn,1):.0f}%  "
                  f"precision@0.5={100*tp/max(tp+fp,1):.0f}%  FP={fp}")

    report(rows, "ALL")

    # Method 3: tier stratification (the headline view — don't read the
    # aggregate alone; constant tools dominate the count).
    print("\n  --- by tier (Method 3) ---")
    print(f"    {'tier':<17s} {'n':>5s} {'act_mean':>9s} {'logMAE':>7s} "
          f"{'ρ':>6s} {'Acc5ms':>7s} {'Acc50ms':>8s} {'P95cov':>7s}")
    tiers = defaultdict(list)
    for r in rows:
        tiers[r["tier"]].append(r)
    order = ["constant_submit", "constant_editor", "light_bash",
             "normal_bash", "heavy_bash", "unknown"]
    for t in order:
        rs = tiers.get(t)
        if not rs:
            continue
        log_ae = [log1p_ae(r["p50"], r["gt"]) for r in rs]
        n_p95 = sum(1 for r in rs if r["p95"] is not None)
        cov = sum(1 for r in rs if r["p95"] is not None and r["gt"] <= r["p95"])
        print(f"    {t:<17s} {len(rs):>5d} "
              f"{statistics.fmean([r['gt'] for r in rs]):>8.3f}s "
              f"{statistics.fmean(log_ae):>7.4f} {_spearman(rs):>6.3f} "
              f"{100*_acc_within(rs,0.005,0.05):>6.1f}% "
              f"{100*_acc_within(rs,0.050,0.20):>7.1f}% "
              f"{100*cov/max(n_p95,1):>6.1f}%")

    # Source split (ml vs bucket_median) kept for completeness.
    print("\n  --- by predictor source ---")
    report([r for r in rows if r["src"] == "ml"], "ML")
    report([r for r in rows if r["src"] == "bucket_median"], "bucket_median")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "step_time", "kqueue", "tool_time"])
    ap.add_argument("--step-time-log", default=DEF_STEP_TIME)
    ap.add_argument("--shadow-log", default=DEF_SHADOW)
    ap.add_argument("--pred-log", default=DEF_PRED)
    ap.add_argument("--trace", default=DEF_TRACE)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    args = ap.parse_args()

    if args.which in ("all", "step_time"):
        eval_step_time(args.step_time_log)
        print()
    if args.which in ("all", "kqueue"):
        eval_kqueue(args.shadow_log)
        print()
    if args.which in ("all", "tool_time"):
        eval_tool_time(args.pred_log, args.trace, args.timeout_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
