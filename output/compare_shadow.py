#!/usr/bin/env python3
"""Compare ShadowScheduler predictions against ground truth K_queue.

Joins:
  v3_shadow_predictions.jsonl   shadow pred_k_queue_stage1/2 by request_id
  v3_kqueue_actual.jsonl        ground-truth k_queue_actual by request_id
  v3_predictions.jsonl          legacy K_queue_pred  by (traj_id, agent_round_next)
"""
import json
import statistics
import sys
from collections import Counter, defaultdict


def load_jsonl(path):
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


def main():
    shadow = load_jsonl("/data/whr/vllm-continuum/output/v3_shadow_predictions.jsonl")
    actuals = load_jsonl("/data/whr/vllm-continuum/output/v3_kqueue_actual.jsonl")
    legacy = load_jsonl("/data/whr/vllm-continuum/output/v3_predictions.jsonl")

    print(f"shadow: {len(shadow)}  actuals: {len(actuals)}  legacy: {len(legacy)}")

    # Index actuals by request_id
    actual_by_rid = {a["request_id"]: a for a in actuals if a.get("request_id")}
    # Index legacy by (traj_id, agent_round + 1) so we can compare same row
    legacy_by_tk = {}
    for L in legacy:
        if L.get("traj_id") is None or L.get("agent_round") is None:
            continue
        legacy_by_tk[(L["traj_id"], L["agent_round"] + 1)] = L

    # Build paired records: shadow pred for next round vs actual on that next round
    pairs = []
    for s in shadow:
        rid = s.get("request_id")
        if not rid:
            continue
        a = actual_by_rid.get(rid)
        if a is None:
            continue
        k_actual = a.get("k_queue_actual")
        if k_actual is None:
            continue
        # Legacy: K_queue_pred at decode-finish of round K-1 predicts next K
        L = legacy_by_tk.get((s["traj_id"], s["agent_round_next"]))
        legacy_kq = L.get("K_queue_pred") if L else None
        pairs.append({
            "request_id": rid,
            "traj_id": s["traj_id"],
            "agent_round_next": s["agent_round_next"],
            "bucket": s.get("bucket"),
            "est_num_tokens": s.get("est_num_tokens"),
            "real_num_tokens": s.get("real_num_tokens"),
            "est_tool_time_s": s.get("est_tool_time_s"),
            "real_tool_time_s": s.get("real_tool_time_s"),
            "est_eta_step": s.get("est_eta_step"),
            "real_arrival_step": s.get("real_arrival_step"),
            "pred_stage1": s.get("pred_k_queue_stage1"),
            "pred_stage2": s.get("pred_k_queue_stage2"),
            "actual": k_actual,
            "wait_wall_s": a.get("wait_wall_s"),
            "legacy_K_queue_pred": legacy_kq,
        })

    print(f"paired (shadow ∩ actuals): {len(pairs)}")
    if not pairs:
        return 1

    # Distributions
    actual_dist = [p["actual"] for p in pairs]
    stage1 = [p["pred_stage1"] for p in pairs if p["pred_stage1"] is not None]
    stage2 = [p["pred_stage2"] for p in pairs if p["pred_stage2"] is not None]
    legacy_v = [p["legacy_K_queue_pred"] for p in pairs
                if p["legacy_K_queue_pred"] is not None]

    print("\n=== DISTRIBUTIONS ===")
    print(f"actual  : n={len(actual_dist):4d}  p50={pct(actual_dist,50)}  p90={pct(actual_dist,90)}  p95={pct(actual_dist,95)}  max={max(actual_dist)}  mean={statistics.fmean(actual_dist):.2f}")
    print(f"stage1  : n={len(stage1):4d}  p50={pct(stage1,50)}  p90={pct(stage1,90)}  p95={pct(stage1,95)}  max={max(stage1)}  mean={statistics.fmean(stage1):.2f}")
    if stage2:
        print(f"stage2  : n={len(stage2):4d}  p50={pct(stage2,50)}  p90={pct(stage2,90)}  p95={pct(stage2,95)}  max={max(stage2)}  mean={statistics.fmean(stage2):.2f}")
    else:
        print(f"stage2  : n=0  (no stage2 corrections logged)")
    print(f"legacy  : n={len(legacy_v):4d}  p50={pct(legacy_v,50)}  p90={pct(legacy_v,90)}  p95={pct(legacy_v,95)}  max={max(legacy_v)}  mean={statistics.fmean(legacy_v):.2f}")

    # Error breakdown
    print("\n=== ERROR (pred - actual) ===")
    for name, getter in [
        ("stage1", lambda p: p["pred_stage1"]),
        ("legacy", lambda p: p["legacy_K_queue_pred"]),
    ]:
        errs = [(getter(p), p["actual"]) for p in pairs
                if getter(p) is not None and p["actual"] is not None]
        if not errs:
            continue
        diff = [pr - ac for pr, ac in errs]
        abs_diff = [abs(d) for d in diff]
        exact = sum(1 for d in diff if d == 0)
        over = sum(1 for d in diff if d > 0)
        under = sum(1 for d in diff if d < 0)
        print(f"\n  [{name}]  n={len(errs)}")
        print(f"    exact      : {exact:4d} ({100*exact/len(errs):.1f}%)")
        print(f"    over       : {over:4d} ({100*over/len(errs):.1f}%)")
        print(f"    under      : {under:4d} ({100*under/len(errs):.1f}%)")
        print(f"    |err| p50  : {pct(abs_diff,50)}")
        print(f"    |err| p90  : {pct(abs_diff,90)}")
        print(f"    |err| p99  : {pct(abs_diff,99)}")
        print(f"    |err| max  : {max(abs_diff)}")
        print(f"    bias mean  : {statistics.fmean(diff):+.3f}")

    # Joint distribution: stage1 -> actual
    print("\n=== JOINT (stage1 → actual) ===")
    joint = defaultdict(Counter)
    for p in pairs:
        if p["pred_stage1"] is not None:
            joint[p["pred_stage1"]][p["actual"]] += 1
    for s in sorted(joint):
        items = sorted(joint[s].items())
        print(f"  stage1={s:3d}  → " + "  ".join(f"actual={a}:{n}" for a, n in items))

    # Cap (=large pred) cases
    cap_cases = [p for p in pairs
                 if p["pred_stage1"] is not None and p["pred_stage1"] > 50]
    if cap_cases:
        print(f"\n=== STAGE1 CAP/HIGH (>50): {len(cap_cases)} ===")
        for c in cap_cases[:5]:
            print(f"  s1={c['pred_stage1']}  actual={c['actual']}  "
                  f"est_tok={c['est_num_tokens']}  real_tok={c['real_num_tokens']}  "
                  f"traj={c['traj_id'][:40]}  K+1={c['agent_round_next']}")

    # Estimation error: est_num_tokens vs real
    print("\n=== ESTIMATION ERROR ===")
    tok_errs = [p["est_num_tokens"] - p["real_num_tokens"]
                for p in pairs
                if p["est_num_tokens"] is not None
                and p["real_num_tokens"] is not None]
    if tok_errs:
        print(f"  est - real num_tokens : n={len(tok_errs)}  "
              f"p50={pct(tok_errs,50):+.0f}  p90={pct(tok_errs,90):+.0f}  "
              f"p99={pct(tok_errs,99):+.0f}  "
              f"min={min(tok_errs):+.0f}  max={max(tok_errs):+.0f}  "
              f"mean={statistics.fmean(tok_errs):+.0f}")
    tool_errs = [p["est_tool_time_s"] - p["real_tool_time_s"]
                 for p in pairs
                 if p["est_tool_time_s"] is not None
                 and p["real_tool_time_s"] is not None]
    if tool_errs:
        print(f"  est - real tool_time  : n={len(tool_errs)}  "
              f"p50={pct(tool_errs,50):+.3f}s  p90={pct(tool_errs,90):+.3f}s  "
              f"min={min(tool_errs):+.3f}s  max={max(tool_errs):+.3f}s")

    # Per-bucket stage1 vs actual
    print("\n=== PER BUCKET ===")
    by_bucket = defaultdict(list)
    for p in pairs:
        if p["pred_stage1"] is not None and p["actual"] is not None:
            by_bucket[p["bucket"]].append(p["pred_stage1"] - p["actual"])
    for b in sorted(by_bucket, key=lambda b: -len(by_bucket[b]))[:10]:
        v = by_bucket[b]
        exact = sum(1 for d in v if d == 0)
        print(f"  {b:<35s}  n={len(v):4d}  exact={100*exact/len(v):.1f}%  "
              f"bias_mean={statistics.fmean(v):+.2f}  "
              f"|err|_p90={pct([abs(d) for d in v], 90)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
