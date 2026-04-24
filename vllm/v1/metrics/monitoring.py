# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import builtins
import json
import os
import time
from collections import defaultdict
from typing import Any, Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]


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
        self.delay_free_stats: dict[str, dict[str, Any]] = {}
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

    def record_delay_free_start(self, request_id: str,
                               job_id: Optional[str]) -> None:
        self.delay_free_stats[request_id] = {
            "request_id": request_id,
            "job_id": job_id,
            "delay_free_start_ts": time.time(),
        }

    def record_delay_free_end(self, request_id: str,
                              worker_timestamps: Optional[dict[str,
                                                               float]] = None
                              ) -> None:
        entry = self.delay_free_stats.get(request_id)
        if entry is None:
            return
        now = time.time()
        entry["deferred_cleanup_ts"] = now
        start = entry.get("delay_free_start_ts")

        # If bg thread already freed blocks, use block_freed_ts as
        # the real end time.  Otherwise fall back to now (no bg thread).
        block_freed = (worker_timestamps or {}).get("block_freed_ts")
        real_end = block_freed if block_freed is not None else now
        entry["delay_free_end_ts"] = real_end
        if start is not None:
            entry["total_delay_free_duration"] = real_end - start
            entry["deferred_cleanup_lag"] = now - real_end
        if worker_timestamps:
            entry.update(worker_timestamps)
            #
            # BLOCK_MIGRATE (decode-pull) mode timeline:
            #
            #   delay_free_start_ts     P-scheduler: marks delay-free
            #   bridge_popped_ts        D-worker: pulls bridge metadata
            #                           via BRIDGE_POP RPC
            #   migration_launch_ts     D-worker: cudaMemcpy2DAsync issued
            #   migration_complete_ts   D-worker: cuda event done
            #                           (poll thread detects immediately)
            #   release_received_ts     P-listener: RELEASE RPC arrived
            #   finished_sending_ts     P-scheduler: bg free thread drains
            #                           queue, kv_cache_manager.free() done
            #   delay_free_end_ts       P-scheduler: block_freed_ts
            #                           (== finished_sending_ts when bg
            #                           thread is active)
            #
            # bridge_staged_ts is also recorded by P-worker at wait_for_save
            # for bookkeeping, but it is anchored at CPU-time before GPU
            # forward sync, so "staged → popped" is NOT a meaningful
            # delay-free metric (it includes whatever GPU work happened
            # between wait_for_save and update_from_output).  Always use
            # wait_bridge_pop = popped - delay_free_start_ts instead.
            #
            # Non-block (PUT_ASYNC/GET) mode:
            #   delay_free_start_ts
            #   send_complete_ts        all layer tensors transferred
            #   delay_free_end_ts
            #
            popped = worker_timestamps.get("bridge_popped_ts")
            launch = worker_timestamps.get("migration_launch_ts")
            complete = worker_timestamps.get("migration_complete_ts")
            rel_recv = worker_timestamps.get("release_received_ts")
            fin_sending = worker_timestamps.get("finished_sending_ts")
            send_complete = worker_timestamps.get("send_complete_ts")

            # -- BLOCK_MIGRATE: segments that SUM to total --
            #
            # total_delay_free = wait_migration + schedule_wait
            #   (delay_free_end = block_freed_ts when bg thread is active)
            #
            #   wait_migration    delay_free_start → release_received
            #     ├ wait_bridge_pop     delay_free_start → bridge_popped
            #     ├ cuda_memcpy         bridge_popped    → migration_complete
            #     └ release_rpc_total   migration_complete → release_received
            #   schedule_wait     release_received → block_freed
            #
            #   deferred_cleanup_lag    block_freed → deferred_cleanup_ts
            #     (blocks already free, just del requests / monitoring)
            #

            # Segment 1: delay_free_start → release_received
            if start is not None and rel_recv is not None:
                entry["wait_migration"] = rel_recv - start

            # Sub-segments of segment 1
            if start is not None and popped is not None:
                entry["wait_bridge_pop"] = popped - start
            if popped is not None and complete is not None:
                entry["cuda_memcpy_duration"] = complete - popped
            if complete is not None and rel_recv is not None:
                entry["release_rpc_total"] = rel_recv - complete

            # Segment 2: release_received → block_freed (bg thread drain)
            if rel_recv is not None and fin_sending is not None:
                entry["schedule_wait"] = fin_sending - rel_recv

            # -- IPC resolve cost (kept for rare "launch hang" debugging) --
            if popped is not None and launch is not None:
                entry["ipc_setup_duration"] = launch - popped

            # -- Non-block durations --
            if start is not None and send_complete is not None:
                entry["wait_send_complete_duration"] = send_complete - start
                entry["scheduler_free_lag"] = now - send_complete

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

    def record_scheduler_stats_from_snapshot(self,
                                             scheduler_stats: Any) -> None:
        if scheduler_stats is None:
            return

        prefix_cache_stats = getattr(scheduler_stats, "prefix_cache_stats", None)
        self.record_scheduler_stats(
            timestamp=time.time(),
            num_running=int(getattr(scheduler_stats, "num_running_reqs", 0)),
            num_waiting=int(getattr(scheduler_stats, "num_waiting_reqs", 0)),
            num_waiting_for_remote_kvs=int(
                getattr(scheduler_stats, "num_waiting_for_remote_kvs", 0)),
            num_preempted=int(getattr(scheduler_stats, "num_preempted", 0)),
            kv_cache_usage=float(getattr(scheduler_stats, "kv_cache_usage", 0.0)),
            prefix_cache_queries=int(getattr(prefix_cache_stats, "queries", 0)),
            prefix_cache_hits=int(getattr(prefix_cache_stats, "hits", 0)),
        )

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
        lock_path = f"{final_path}.lock"
        tmp_path = f"{final_path}.tmp.{os.getpid()}.{time.time_ns()}"

        # Multiple processes (API + engine core) can dump concurrently.
        # Locking keeps read-merge-write atomic and avoids data loss.
        with builtins.open(lock_path, "a") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
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

                merged_delay_free_stats = dict(
                    existing.get("delay_free_stats") or {})
                merged_delay_free_stats.update(self.delay_free_stats)

                with builtins.open(tmp_path, "w") as f:
                    json.dump({
                        "request_meta": merged_request_meta,
                        "request_stats": merged_request_stats,
                        "scheduler_stats": merged_scheduler_stats,
                        "iteration_stats": merged_iteration_stats,
                        "tool_call_times": merged_tool_call_times,
                        "delay_free_stats": merged_delay_free_stats,
                    },
                              f,
                              indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, final_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


monitoring_recorder = MonitoringRecorder()
