#!/usr/bin/env python3
"""Extract per-step ground truth from a prefill instance's
monitoring_timestamps dump.

Source format
-------------
The prefill side writes `monitoring_timestamps` (JSON) with these top
level keys:
  - iteration_stats   : list, one entry per scheduler step
       {timestamp, num_prompt_tokens, num_generation_tokens,
        num_finished_requests, num_preempted_reqs}
  - scheduler_stats   : list, 1:1 with iteration_stats (recorded ~ms later)
       {timestamp, num_running, num_waiting, num_waiting_for_remote_kvs,
        num_preempted, kv_cache_usage, prefix_cache_queries,
        prefix_cache_hits}

The pair (iteration_stats[i], scheduler_stats[i]) describes a single
prefill scheduler step.  The step's wall-clock duration is the diff
between consecutive iteration_stats timestamps.

Output
------
JSONL at /data/whr/vllm-continuum/step_time/ground_truth.jsonl.  One
line per step (i >= 1) with:
  - step_id            : sequential index
  - ts                 : iteration_stats[i].timestamp (= step END)
  - duration_s         : ts[i] - ts[i-1] (= time spent doing step i)
  - num_prompt_tokens  : new prompt tokens scheduled this step
  - num_generation_tokens : decoded tokens this step
  - total_tokens       : sum of above
  - num_running        : RUNNING queue size at step start
  - num_waiting        : WAITING queue size at step start
  - num_waiting_for_remote_kvs : in-flight KV migration count
  - num_finished_requests : reqs finished this step
  - num_preempted_reqs : reqs preempted this step
  - kv_cache_usage     : fraction of total blocks used
  - prefix_cache_queries : new prompt tokens queried for prefix match
  - prefix_cache_hits  : hits in prefix cache
  - prefix_cache_hit_rate : hits / max(queries, 1)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


CHUNK_SIZE_TOKENS = 5242   # LICHTV2 default chunk size


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _b_at(num_compute_tokens: int, chunk_idx: int,
          chunk_size: int = CHUNK_SIZE_TOKENS) -> int:
    """Tokens in the chunk_idx-th chunk of a request with
    `num_compute_tokens` total to compute (= num_prompt - hit_length).
    Mirrors prefill's _licht_v2_B_at (in token count, not block count)."""
    if num_compute_tokens <= 0 or chunk_idx < 0:
        return 0
    total_chunks = _ceil_div(num_compute_tokens, chunk_size)
    if chunk_idx >= total_chunks:
        return 0
    # All chunks except last are full chunk_size; last is remainder.
    if chunk_idx < total_chunks - 1:
        return chunk_size
    # Last chunk
    return num_compute_tokens - chunk_idx * chunk_size


def _derive_per_step_compute(data: dict, iter_stats: list,
                              block_tile: int = 16,
                              gpu_capacity_blocks: int = 216) -> dict:
    """Reconstruct per-step compute features from licht_admit_events +
    request_stats.

    L_i semantics
    -------------
    L_i = tokens_this_step for request i = the chunk that this req
    computes in this step (NOT cumulative length, NOT full prompt
    length).

    Final feature set (user-confirmed 2026-05-19):
      - num_scheduled_tokens : Σ chunk_tokens this step
      - num_new_admits / num_new_admit_tokens : first-chunk only
      - sum_L_sq              : Σ chunk_i² (chunk-distribution variance)
      - attention_waves       : max(1.0, tiles / gpu_capacity_blocks)
                                where tiles = Σ n_i·(n_i+1)/2,
                                      n_i = chunk_i // block_tile
                                (float, NOT ceil — captures fractional
                                 GPU sweep load)

      Split-attention features (added 2026-05-19, user idea):
      - sum_new_admit_sq      : Σ chunk_i² over reqs whose k==0 at this
                                step (= first chunk of just-admitted req).
                                Approximates self-attention surge of
                                fresh prefill.
      - sum_chunk_x_lcum      : Σ chunk_i × L_cum_before_i over reqs
                                whose k>=1 at this step (= continuing
                                chunked prefill).  L_cum_before =
                                hit + Σ chunks_0..k-1 (all KV already
                                in cache when this chunk starts).
                                Approximates cross-attention cost.

    Dropped intentionally:
      - sum_L         (≡ num_scheduled_tokens, perfectly collinear)
      - max_L         (saturated at chunk_size ≥90% of steps, ~0 variance)
      - attention_tiles (≡ attention_waves × capacity, redundant)
    """
    admit_events = data.get("licht_admit_events", [])
    request_stats = data.get("request_stats", {}) or {}

    iter_ts = [float(it["timestamp"]) for it in iter_stats]
    import bisect

    def ts_to_step(t: float) -> int:
        return bisect.bisect_left(iter_ts, t)

    reqs = []
    for evt in admit_events:
        rid = evt.get("request_id")
        if rid is None:
            continue
        ts = float(evt["timestamp"])
        admit_step = ts_to_step(ts)
        if admit_step >= len(iter_ts):
            continue
        rs = request_stats.get(rid) or {}
        total_prompt = int(rs.get("num_prompt_tokens", 0))
        hit = int(evt.get("num_computed_at_admit", 0))
        compute_tokens = max(total_prompt - hit, 0)
        if compute_tokens == 0:
            R = 1
        else:
            R = _ceil_div(compute_tokens, CHUNK_SIZE_TOKENS)
        reqs.append({
            "admit_step": admit_step,
            "hit_length": hit,
            "compute_tokens": compute_tokens,
            "R": R,
        })

    per_step = {}
    for r in reqs:
        admit_step = r["admit_step"]
        hit = r["hit_length"]
        compute_tokens = r["compute_tokens"]
        R = r["R"]
        # L_cum_before this step's chunk: starts at hit (= prefix cache
        # KV already in cache when req admitted), grows by completed
        # chunks.
        L_cum_before = hit
        for k in range(R):
            step_idx = admit_step + k
            if step_idx >= len(iter_ts):
                break
            tokens_this_step = _b_at(compute_tokens, k)
            L_i = tokens_this_step  # chunk this step, NOT cumulative

            d = per_step.setdefault(step_idx, {
                "num_scheduled_tokens": 0,
                "num_new_admits": 0,
                "num_new_admit_tokens": 0,
                "num_active_reqs": 0,
                "sum_L_sq": 0,
                "sum_new_admit_sq": 0,
                "sum_chunk_x_lcum": 0,
                "_tiles": 0,  # internal scratch — not written out
            })
            d["num_scheduled_tokens"] += tokens_this_step
            d["num_active_reqs"] += 1
            d["sum_L_sq"] += L_i * L_i
            n_i = L_i // block_tile
            d["_tiles"] += (n_i * (n_i + 1)) // 2
            if k == 0:
                d["num_new_admits"] += 1
                d["num_new_admit_tokens"] += tokens_this_step
                # New-admit self-attention surge: chunk² (user idea —
                # ignore hit-cache cross-attn for first-chunk reqs since
                # most newly-admitted reqs have small hit anyway).
                d["sum_new_admit_sq"] += L_i * L_i
            else:
                # Continuing chunked prefill: chunk attending to all
                # previously-cached KV (= hit + sum of chunks 0..k-1).
                d["sum_chunk_x_lcum"] += L_i * L_cum_before

            # Update L_cum_before AFTER recording this step (so next
            # iteration sees post-chunk-k KV total).
            L_cum_before += tokens_this_step

    # attention_waves = max(1.0, tiles / capacity)  — float, captures
    # fractional GPU sweeps (= "how many full SM-sweeps the attention
    # workload requires").  Floored at 1.0 so empty/tiny steps don't get
    # 0 and create division-by-zero downstream.
    cap = max(int(gpu_capacity_blocks), 1)
    for d in per_step.values():
        tiles = d.pop("_tiles")
        d["attention_waves"] = max(1.0, tiles / cap)
    return per_step


DEFAULT_SRC = (
    "/data/whr/vllm-continuum/examples/online_serving/"
    "disaggregated_serving_p2p_nccl_xpyd/continuum_exp/"
    "prefill_20003/monitoring_timestamps")
DEFAULT_OUT = "/data/whr/vllm-continuum/step_time/ground_truth.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="path to monitoring_timestamps JSON file")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="output JSONL path")
    ap.add_argument("--gpu-capacity", type=int, default=216,
                    help="GPU attention block capacity = num_SMs × "
                         "blocks_per_SM.  Default 216 = A800/A100 with "
                         "108 SMs × 2 CTAs/SM (FA2 forward typical).  "
                         "Override for other hardware.")
    ap.add_argument("--block-tile", type=int, default=16,
                    help="attention tile size in tokens (default 16)")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        print(f"ERROR: source not found: {args.src}", file=sys.stderr)
        return 1
    with open(args.src) as f:
        data = json.load(f)
    iter_stats = data.get("iteration_stats", [])
    sched_stats = data.get("scheduler_stats", [])
    if not iter_stats:
        print("ERROR: empty iteration_stats", file=sys.stderr)
        return 1
    if len(iter_stats) != len(sched_stats):
        print(f"WARN: iter_stats len={len(iter_stats)} != "
              f"sched_stats len={len(sched_stats)}; will join by index",
              file=sys.stderr)
    n = min(len(iter_stats), len(sched_stats))

    # step_compute_log: NEW (added 2026-05-19) — per-step actual compute
    # load logged by scheduler.schedule().  Falls back to a derivation
    # from licht_admit_events + request_stats if absent.
    step_compute_log = data.get("step_compute_log", [])
    step_compute_by_id: dict[int, dict] = {
        int(e["step_id"]): e for e in step_compute_log
        if e.get("step_id") is not None
    }
    derived_compute_by_step: dict[int, dict] = {}
    if step_compute_log:
        print(f"step_compute_log has {len(step_compute_log)} entries; "
              "using as primary compute load feature")
    else:
        print("step_compute_log missing — deriving per-step compute from "
              "licht_admit_events + request_stats")
        derived_compute_by_step = _derive_per_step_compute(
            data, iter_stats,
            block_tile=args.block_tile,
            gpu_capacity_blocks=args.gpu_capacity)
        print(f"  derived for {len(derived_compute_by_step)} steps "
              f"(block_tile={args.block_tile}, "
              f"gpu_capacity={args.gpu_capacity})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_written = 0
    with open(args.out, "w") as fout:
        for i in range(1, n):  # skip step 0 (no prior ts → no duration)
            it_prev = iter_stats[i - 1]
            it = iter_stats[i]
            ss = sched_stats[i] if i < len(sched_stats) else {}
            duration = float(it["timestamp"]) - float(it_prev["timestamp"])
            if duration <= 0:
                continue
            queries = int(ss.get("prefix_cache_queries", 0) or 0)
            hits = int(ss.get("prefix_cache_hits", 0) or 0)
            # Per-step compute: prefer step_compute_log (= scheduler-emitted,
            # exact), else fall back to derivation from admit events.
            sc = step_compute_by_id.get(i) or step_compute_by_id.get(i + 1)
            derived = derived_compute_by_step.get(i)
            rec = {
                "step_id": i,
                "ts": float(it["timestamp"]),
                "duration_s": duration,
                # ---- "old" features (kept for back-compat) ----
                "iteration_num_prompt_tokens": int(
                    it.get("num_prompt_tokens", 0)),  # = finished prefill
                "num_generation_tokens": int(
                    it.get("num_generation_tokens", 0)),
                # ---- per-step actual compute (exact OR derived) ----
                "num_scheduled_tokens": (int(sc["num_scheduled_tokens"])
                                          if sc else
                                          (int(derived["num_scheduled_tokens"])
                                           if derived else None)),
                "num_new_admits": (int(sc["num_new_admits"])
                                    if sc else
                                    (int(derived["num_new_admits"])
                                     if derived else None)),
                "num_new_admit_tokens": (int(sc["num_new_admit_tokens"])
                                          if sc else
                                          (int(derived["num_new_admit_tokens"])
                                           if derived else None)),
                # ---- attention / length features (from derivation) ----
                "num_active_reqs": (int(derived["num_active_reqs"])
                                     if derived else None),
                "sum_L_sq": (int(derived["sum_L_sq"])
                             if derived else None),
                "sum_new_admit_sq": (int(derived["sum_new_admit_sq"])
                                      if derived else None),
                "sum_chunk_x_lcum": (int(derived["sum_chunk_x_lcum"])
                                      if derived else None),
                "attention_waves": (float(derived["attention_waves"])
                                     if derived else None),
                "compute_source": ("logged" if sc
                                    else ("derived" if derived
                                          else "missing")),
                # ---- queue state at step start ----
                "num_running": int(
                    sc["num_running"] if sc else ss.get("num_running", 0)),
                "num_waiting": int(
                    sc["num_waiting"] if sc else ss.get("num_waiting", 0)),
                "num_waiting_for_remote_kvs": int(
                    ss.get("num_waiting_for_remote_kvs", 0)),
                "num_finished_requests": int(
                    it.get("num_finished_requests", 0)),
                "num_preempted_reqs": int(it.get("num_preempted_reqs", 0)),
                "kv_cache_usage": float(ss.get("kv_cache_usage", 0.0)),
                "prefix_cache_queries": queries,
                "prefix_cache_hits": hits,
                "prefix_cache_hit_rate": (hits / queries
                                          if queries > 0 else 0.0),
            }
            fout.write(json.dumps(rec) + "\n")
            n_written += 1

    print(f"wrote {n_written} step records → {args.out}")
    # Quick stats
    with open(args.out) as f:
        durs = [json.loads(l)["duration_s"] for l in f if l.strip()]
    durs.sort()
    if durs:
        n = len(durs)
        print(f"\n=== duration_s distribution (n={n}) ===")
        print(f"  min   = {durs[0]:.4f}s")
        print(f"  p25   = {durs[n // 4]:.4f}s")
        print(f"  p50   = {durs[n // 2]:.4f}s")
        print(f"  p75   = {durs[3 * n // 4]:.4f}s")
        print(f"  p90   = {durs[9 * n // 10]:.4f}s")
        print(f"  p99   = {durs[int(0.99 * n)]:.4f}s")
        print(f"  max   = {durs[-1]:.4f}s")
        print(f"  total = {sum(durs):.1f}s ({sum(durs) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
