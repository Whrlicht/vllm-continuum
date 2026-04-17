# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import builtins
import json
import os
import time
from collections import defaultdict
from typing import Any, Optional


class MonitoringRecorder:
    """Lightweight recorder for offline analysis.

    Writes a single JSON file to RUN_OUTPUT_DIR/monitoring_timestamps.
    """

    def __init__(self) -> None:
        self.request_meta: dict[str, dict[str, Any]] = {}
        self.request_stats: dict[str, dict[str, Any]] = {}
        self.scheduler_stats: list[dict[str, Any]] = []
        self.iteration_stats: list[dict[str, Any]] = []
        self.tool_call_times: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.output_dir: Optional[str] = None

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def _get_output_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        return os.environ.get("RUN_OUTPUT_DIR", "./continuum_exp")

    def record_request_meta(self, request: Any) -> None:
        # Avoid overwriting if a request_id is reused unexpectedly.
        if request.request_id in self.request_meta:
            return
        self.request_meta[request.request_id] = {
            "job_id": request.job_id,
            "proxy_request_id": getattr(request, "proxy_request_id", None),
            "agent_round": getattr(request, "agent_round", None),
            "request_id": request.request_id,
            "arrival_time": request.arrival_time,
            "is_last_step": request.is_last_step,
            "last_func_call": request.last_func_call,
            "this_func_call": request.this_func_call,
        }

    def record_tool_call_time(self, job_id: Optional[str], func_call: str,
                              exec_time: float) -> None:
        if job_id is None:
            return
        self.tool_call_times[str(job_id)].append({
            "func_call": func_call,
            "exec_time": exec_time,
        })

    def record_scheduler_stats(
        self,
        *,
        timestamp: float,
        num_running: int,
        num_waiting: int,
        num_waiting_for_remote_kvs: int,
        num_preempted: int,
        kv_cache_usage: float,
        prefix_cache_queries: int,
        prefix_cache_hits: int,
    ) -> None:
        self.scheduler_stats.append({
            "timestamp": timestamp,
            "num_running": num_running,
            "num_waiting": num_waiting,
            "num_waiting_for_remote_kvs": num_waiting_for_remote_kvs,
            "num_preempted": num_preempted,
            "kv_cache_usage": kv_cache_usage,
            "prefix_cache_queries": prefix_cache_queries,
            "prefix_cache_hits": prefix_cache_hits,
        })

    def record_iteration_stats(self, iteration_stats: Any) -> None:
        self.iteration_stats.append({
            "timestamp": iteration_stats.iteration_timestamp,
            "num_prompt_tokens": iteration_stats.num_prompt_tokens,
            "num_generation_tokens": iteration_stats.num_generation_tokens,
            "num_finished_requests": len(iteration_stats.finished_requests),
            "num_preempted_reqs": iteration_stats.num_preempted_reqs,
        })

    def record_finished_request(self, req_state: Any, finish_reason: Any,
                                iteration_timestamp: float) -> None:
        stats = req_state.stats
        if stats is None:
            return

        meta = self.request_meta.get(req_state.request_id, {})

        queued_ts = stats.queued_ts or None
        scheduled_ts = stats.scheduled_ts or None
        first_token_ts = stats.first_token_ts or None
        last_token_ts = stats.last_token_ts or None

        e2e_latency = iteration_timestamp - stats.arrival_time
        queued_time = None
        prefill_time = None
        decode_time = None
        inference_time = None

        if queued_ts is not None and scheduled_ts is not None:
            queued_time = scheduled_ts - queued_ts
        if scheduled_ts is not None and first_token_ts is not None:
            prefill_time = first_token_ts - scheduled_ts
        if first_token_ts is not None and last_token_ts is not None:
            decode_time = last_token_ts - first_token_ts
        if scheduled_ts is not None and last_token_ts is not None:
            inference_time = last_token_ts - scheduled_ts

        self.request_stats[req_state.request_id] = {
            "request_id": req_state.request_id,
            "job_id": meta.get("job_id"),
            "proxy_request_id": meta.get("proxy_request_id"),
            "agent_round": meta.get("agent_round"),
            "arrival_time": stats.arrival_time,
            "finish_time": iteration_timestamp,
            "finish_reason": finish_reason.name if finish_reason else None,
            "num_prompt_tokens": len(req_state.prompt_token_ids),
            "num_generation_tokens": stats.num_generation_tokens,
            "first_token_latency": stats.first_token_latency,
            "e2e_latency": e2e_latency,
            "queued_time": queued_time,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "inference_time": inference_time,
            "first_token_ts": first_token_ts,
            "last_token_ts": last_token_ts,
        }

    def dump(self) -> None:
        output_dir = self._get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, "monitoring_timestamps")
        tmp_path = f"{final_path}.tmp.{os.getpid()}.{time.time_ns()}"
        existing: dict[str, Any] = {}
        if os.path.exists(final_path):
            try:
                with builtins.open(final_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        merged_request_meta = dict(existing.get("request_meta") or {})
        merged_request_meta.update(self.request_meta)

        merged_request_stats = dict(existing.get("request_stats") or {})
        merged_request_stats.update(self.request_stats)

        merged_scheduler_stats = list(existing.get("scheduler_stats") or [])
        merged_scheduler_stats.extend(self.scheduler_stats)

        merged_iteration_stats = list(existing.get("iteration_stats") or [])
        merged_iteration_stats.extend(self.iteration_stats)

        merged_tool_call_times = dict(existing.get("tool_call_times") or {})
        for job_id, entries in self.tool_call_times.items():
            merged_tool_call_times.setdefault(job_id, [])
            merged_tool_call_times[job_id].extend(entries)

        with builtins.open(tmp_path, "w") as f:
            json.dump({
                "request_meta": merged_request_meta,
                "request_stats": merged_request_stats,
                "scheduler_stats": merged_scheduler_stats,
                "iteration_stats": merged_iteration_stats,
                "tool_call_times": merged_tool_call_times,
            }, f, indent=2)
        os.replace(tmp_path, final_path)


monitoring_recorder = MonitoringRecorder()
