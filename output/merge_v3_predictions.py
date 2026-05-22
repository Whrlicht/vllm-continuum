#!/usr/bin/env python3
"""Merge LICHTV3 per-round predictions into multiturn_trace_client.json.

Reads:
  /data/whr/vllm-continuum/output/v3_predictions.jsonl  (one record per
    server-side prediction, written by decode_manager._write_prediction_record)
  /data/whr/vllm-continuum/output/multiturn_trace_client.json  (one record per
    trajectory; each `results[i].rounds[j]` already has
    `execution_time_seconds` = trace's recorded tool exec ground truth)

Writes (in-place):
  /data/whr/vllm-continuum/output/multiturn_trace_client.json  with each
    round augmented by:
      v3_T_tool_p50, v3_T_tool_p95, v3_T_tool_p_timeout,
      v3_T_tool_bucket, v3_T_tool_family, v3_T_tool_source,
      v3_K_queue_pred, v3_next_round_num_blocks

Join key: (traj_id, agent_round) — agent_round in predictions matches
round_index in client rounds (both 0-indexed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


PRED_LOG = "/data/whr/vllm-continuum/output/v3_predictions.jsonl"
CLIENT_JSON = "/data/whr/vllm-continuum/output/multiturn_trace_client.json"


def load_predictions(path: str) -> dict[tuple[str, int], dict]:
    """Index predictions by (traj_id, agent_round).  If duplicates,
    keep the latest (last-write-wins by ts)."""
    out: dict[tuple[str, int], dict] = {}
    if not os.path.exists(path):
        print(f"prediction log not found: {path}", file=sys.stderr)
        return out
    n_total = 0
    n_skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue
            n_total += 1
            traj = rec.get("traj_id")
            k = rec.get("agent_round")
            if traj is None or k is None:
                n_skipped += 1
                continue
            key = (str(traj), int(k))
            prev = out.get(key)
            if prev is None or rec.get("ts", 0) >= prev.get("ts", 0):
                out[key] = rec
    print(f"loaded {n_total} prediction records "
          f"({len(out)} unique (traj, round); {n_skipped} skipped)")
    return out


def merge(client_path: str, preds: dict[tuple[str, int], dict],
          out_path: str | None = None,
          out_jsonl_with_actual: str | None = None) -> None:
    with open(client_path) as f:
        client = json.load(f)
    out_path = out_path or client_path
    results = client.get("results", [])
    n_rounds = 0
    n_matched = 0
    by_traj_match: dict[str, int] = defaultdict(int)
    # If requested, also produce a JSONL where each line is a prediction
    # record + the client-recorded actual `execution_time_seconds`.
    actual_jsonl: list[dict] = []
    for traj in results:
        traj_id = traj.get("traj_id")
        for rd in traj.get("rounds", []):
            n_rounds += 1
            K = rd.get("round_index")
            if traj_id is None or K is None:
                continue
            key = (str(traj_id), int(K))
            pred = preds.get(key)
            if pred is None:
                continue
            n_matched += 1
            by_traj_match[traj_id] += 1
            rd["v3_T_tool_p50"] = pred.get("T_tool_p50")
            rd["v3_T_tool_p95"] = pred.get("T_tool_p95")
            rd["v3_T_tool_p_timeout"] = pred.get("T_tool_p_timeout")
            rd["v3_T_tool_bucket"] = pred.get("T_tool_bucket")
            rd["v3_T_tool_family"] = pred.get("T_tool_family")
            rd["v3_T_tool_source"] = pred.get("T_tool_source")
            rd["v3_K_queue_pred"] = pred.get("K_queue_pred")
            rd["v3_next_round_num_blocks"] = pred.get(
                "next_round_num_blocks")
            # Build augmented JSONL line
            actual_jsonl.append({
                **pred,
                "actual_execution_time_seconds":
                    rd.get("execution_time_seconds"),
                "actual_request_latency_s":
                    rd.get("request_latency_s"),
                "client_request_start_time":
                    rd.get("request_start_time"),
                "client_request_end_time":
                    rd.get("request_end_time"),
            })
    print(f"merged: rounds_total={n_rounds}  rounds_with_pred={n_matched}  "
          f"trajs_with_any_pred={sum(1 for v in by_traj_match.values() if v > 0)}")
    with open(out_path, "w") as f:
        json.dump(client, f)
    print(f"wrote: {out_path}")
    if out_jsonl_with_actual:
        with open(out_jsonl_with_actual, "w") as f:
            for rec in actual_jsonl:
                f.write(json.dumps(rec) + "\n")
        print(f"wrote: {out_jsonl_with_actual} ({len(actual_jsonl)} records "
              f"with actual)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-log", default=PRED_LOG)
    ap.add_argument("--client-json", default=CLIENT_JSON)
    ap.add_argument("--out", default=None,
                    help="output JSON; default = overwrite --client-json")
    ap.add_argument(
        "--out-jsonl-with-actual",
        default="/data/whr/vllm-continuum/output/v3_predictions_with_actual.jsonl",
        help="ALSO write a JSONL with prediction + client actual "
             "(execution_time_seconds + request_latency_s) per line")
    args = ap.parse_args()
    preds = load_predictions(args.pred_log)
    if not preds:
        print("no predictions to merge; aborting")
        return 1
    merge(args.client_json, preds, args.out,
          out_jsonl_with_actual=args.out_jsonl_with_actual)
    return 0


if __name__ == "__main__":
    sys.exit(main())
