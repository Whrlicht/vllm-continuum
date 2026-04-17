#!/usr/bin/env python3
"""
Calculate average job duration from scheduler_timestamps.

Each job's duration is calculated from the first Request_arrival_time
to the last Request_departure_time.
"""

import argparse
import bisect
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def resolve_input_dir(input_dir_arg: str) -> Path:
    """Resolve input directory from cwd first, then repository root.

    This makes commands robust when launched from subdirectories like
    continuum_exp/ while passing repo-root-relative paths (e.g. examples/...).
    """
    raw = Path(input_dir_arg).expanduser()
    if raw.is_absolute():
        return raw

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parent.parent
    repo_candidate = (repo_root / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate

    # Keep previous behavior in spirit: return cwd-relative resolved path,
    # but include clearer diagnostics later if it does not exist.
    return cwd_candidate


def load_scheduler_timestamps(input_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load scheduler_timestamps from the input directory."""
    timestamp_file = Path(input_dir) / "scheduler_timestamps"

    if not timestamp_file.exists():
        raise FileNotFoundError(f"scheduler_timestamps not found in {input_dir}")

    with open(timestamp_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {timestamp_file}: {exc}. "
                "This usually means the file was truncated/corrupted during "
                "concurrent writes. In disaggregated mode, ensure each "
                "instance uses a distinct RUN_OUTPUT_DIR."
            ) from exc

    return data


def load_monitoring_timestamps(
        input_dir: str) -> Optional[Dict[str, Any]]:
    """Load monitoring_timestamps if present."""
    monitoring_file = Path(input_dir) / "monitoring_timestamps"
    if not monitoring_file.exists():
        return None
    with open(monitoring_file, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {monitoring_file}: {exc}. "
                "In disaggregated mode, ensure each instance writes to its own "
                "RUN_OUTPUT_DIR."
            ) from exc


def discover_instance_dirs(input_dir: Path) -> List[Path]:
    """Find instance subdirectories containing scheduler_timestamps."""
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    if (input_dir / "scheduler_timestamps").exists():
        return [input_dir]

    discovered = [
        child for child in input_dir.iterdir()
        if child.is_dir() and (child / "scheduler_timestamps").exists()
    ]
    return sorted(discovered, key=lambda p: p.name)


def detect_instance_role(instance_dir: Path) -> Optional[str]:
    """Infer role from instance directory name."""
    name = instance_dir.name.lower()
    if name.startswith("prefill_"):
        return "prefill"
    if name.startswith("decode_"):
        return "decode"
    return None


def load_companion_role_timestamps(
        instance_dir: Path) -> Tuple[Optional[str], Optional[Dict[str, List[Dict[str, Any]]]]]:
    """Load companion role scheduler_timestamps under same parent dir.

    For prefill_* instance, try decode_* sibling; for decode_*, try prefill_*.
    """
    role = detect_instance_role(instance_dir)
    if role not in ("prefill", "decode"):
        return None, None

    parent = instance_dir.parent
    target_prefix = "decode_" if role == "prefill" else "prefill_"
    candidates = sorted(
        child for child in parent.iterdir()
        if child.is_dir() and child.name.startswith(target_prefix)
        and (child / "scheduler_timestamps").exists())
    if not candidates:
        return None, None

    companion_dir = candidates[0]
    companion_role = "decode" if role == "prefill" else "prefill"
    return companion_role, load_scheduler_timestamps(str(companion_dir))


def compute_decode_to_next_prefill_pauses(
        prefill_timestamps: Dict[str, List[Dict[str, Any]]],
        decode_timestamps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[float]]:
    """Compute tool pauses as: decode round i departure -> prefill round i+1 arrival."""
    pauses_by_job: Dict[str, List[float]] = {}
    all_job_ids = set(prefill_timestamps) | set(decode_timestamps)

    for job_id in all_job_ids:
        prefill_rounds, _, _ = _extract_rounds(prefill_timestamps.get(job_id, []))
        decode_rounds, _, _ = _extract_rounds(decode_timestamps.get(job_id, []))

        pauses: List[float] = []
        max_idx = min(len(decode_rounds), max(0, len(prefill_rounds) - 1))
        for idx in range(max_idx):
            decode_departure = decode_rounds[idx].get("departure_time")
            next_prefill_arrival = prefill_rounds[idx + 1].get("arrival_time")
            if decode_departure is None or next_prefill_arrival is None:
                continue
            pause = next_prefill_arrival - decode_departure
            if pause >= 0:
                pauses.append(pause)

        pauses_by_job[job_id] = pauses

    return pauses_by_job


def apply_canonical_toolcall_times(
        job_round_metrics: Dict[str, Any],
        canonical_pauses: Dict[str, List[float]]) -> None:
    """Override per-job tool_call_times with decode->next-prefill pauses."""
    for job_id, job_metrics in job_round_metrics.items():
        pauses = canonical_pauses.get(job_id, [])
        job_metrics["tool_call_times"] = pauses
        job_metrics["avg_tool_call_time"] = (
            sum(pauses) / len(pauses) if pauses else 0)
        job_metrics["tool_call_time_semantics"] = (
            "decode_departure_to_next_prefill_arrival")


def percentile_sorted(data: List[float], p: float) -> float:
    """Calculate percentile p (0-100) from sorted data."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


def summarize_distribution(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": 0,
            "min": 0,
            "max": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
        }
    values_sorted = sorted(values)
    return {
        "count": len(values_sorted),
        "mean": sum(values_sorted) / len(values_sorted),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "p50": percentile_sorted(values_sorted, 50),
        "p95": percentile_sorted(values_sorted, 95),
        "p99": percentile_sorted(values_sorted, 99),
    }


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safely divide two values and return None on invalid inputs."""
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _first_not_none(*values: Any) -> Any:
    """Return the first non-None value from inputs."""
    for value in values:
        if value is not None:
            return value
    return None


def _prompt_bucket(prompt_length: Optional[float]) -> str:
    """Bucket prompt length into coarse ranges."""
    if prompt_length is None:
        return "unknown"
    if prompt_length < 1000:
        return "0-1k"
    if prompt_length < 2000:
        return "1k-2k"
    if prompt_length < 4000:
        return "2k-4k"
    if prompt_length < 8000:
        return "4k-8k"
    return "8k+"


def _gen_bucket(num_generation_tokens: Optional[float]) -> str:
    """Bucket generation token counts for grouping analysis."""
    if num_generation_tokens is None:
        return "unknown"
    if num_generation_tokens < 128:
        return "0-128"
    if num_generation_tokens < 512:
        return "128-512"
    if num_generation_tokens < 1024:
        return "512-1k"
    return "1k+"


def _tool_time_bucket(actual_pause: Optional[float]) -> str:
    """Bucket tool pause duration into user-facing ranges."""
    if actual_pause is None:
        return "unknown"
    if actual_pause < 1:
        return "0-1s"
    if actual_pause < 5:
        return "1-5s"
    if actual_pause < 30:
        return "5-30s"
    return "30s+"


def _build_pin_intervals(
        pinned_times: List[float], unpinned_times: List[float]) -> List[Tuple[float, float]]:
    """Pair pin/unpin timestamps into non-overlapping pin intervals."""
    if not pinned_times or not unpinned_times:
        return []

    sorted_pins = sorted(pinned_times)
    sorted_unpins = sorted(unpinned_times)
    intervals: List[Tuple[float, float]] = []
    unpin_idx = 0

    for pin_time in sorted_pins:
        while unpin_idx < len(sorted_unpins) and sorted_unpins[unpin_idx] < pin_time:
            unpin_idx += 1
        if unpin_idx >= len(sorted_unpins):
            break
        unpin_time = sorted_unpins[unpin_idx]
        if unpin_time >= pin_time:
            intervals.append((pin_time, unpin_time))
            unpin_idx += 1

    return intervals


def _interval_overlap_duration(
        start: Optional[float], end: Optional[float],
        intervals: List[Tuple[float, float]]) -> Optional[float]:
    """Compute overlap duration between [start, end] and a list of intervals."""
    if start is None or end is None or end < start:
        return None
    total = 0.0
    for s, e in intervals:
        overlap_start = max(start, s)
        overlap_end = min(end, e)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def _extract_event_time(event: Dict[str, Any], keys: List[str]) -> Optional[float]:
    """Fetch a timestamp-like value from candidate keys in one event."""
    for key in keys:
        value = event.get(key)
        if value is not None:
            return value
    return None


def _extract_optional_numeric(event: Dict[str, Any], keys: List[str]) -> Optional[float]:
    """Fetch optional numeric-like value from candidate keys in one event."""
    for key in keys:
        if key in event and event[key] is not None:
            return event[key]
    return None


def calculate_job_duration(job_history: List[Dict[str, Any]]) -> float:
    """
    Calculate duration for a single job.

    Duration = last Request_departure_time - first Request_arrival_time
    """
    arrival_times = []
    departure_times = []

    for event in job_history:
        if "Request_arrival_time" in event:
            arrival_times.append(event["Request_arrival_time"])
        elif "Request_departure_time" in event:
            departure_times.append(event["Request_departure_time"])

    if not arrival_times or not departure_times:
        return None

    first_arrival = min(arrival_times)
    last_departure = max(departure_times)

    duration = last_departure - first_arrival
    return duration


def calculate_average_duration(timestamps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Calculate average job duration across all jobs."""
    job_durations = {}
    all_arrival_times = []
    all_departure_times = []

    for job_id, history in timestamps.items():
        duration = calculate_job_duration(history)
        if duration is not None:
            job_durations[job_id] = duration

        # Collect all arrival and departure times across all jobs
        for event in history:
            if "Request_arrival_time" in event:
                all_arrival_times.append(event["Request_arrival_time"])
            elif "Request_departure_time" in event:
                all_departure_times.append(event["Request_departure_time"])

    if not job_durations:
        return {
            "num_jobs": 0,
            "average_duration": 0,
            "total_duration": 0,
            "min_duration": 0,
            "max_duration": 0,
            "median_duration": 0,
            "percentile_95": 0,
            "percentile_99": 0,
            "job_durations": {}
        }

    durations = sorted(job_durations.values())

    # Calculate total_duration as largest departure - smallest arrival
    total_duration = max(all_departure_times) - min(all_arrival_times) if all_arrival_times and all_departure_times else 0

    # Calculate median
    n = len(durations)
    if n % 2 == 0:
        median_duration = (durations[n // 2 - 1] + durations[n // 2]) / 2
    else:
        median_duration = durations[n // 2]

    # Calculate percentiles
    percentile_95 = percentile_sorted(durations, 95)
    percentile_99 = percentile_sorted(durations, 99)

    return {
        "num_jobs": len(job_durations),
        "average_duration": sum(durations) / len(durations),
        "total_duration": total_duration,
        "min_duration": min(durations),
        "max_duration": max(durations),
        "median_duration": median_duration,
        "percentile_95": percentile_95,
        "percentile_99": percentile_99,
        "job_durations": job_durations
    }


def _extract_rounds(
        job_history: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]],
                                                    List[float], List[float]]:
    """Extract round-level timeline and scheduling context from job history."""
    rounds: List[Dict[str, Any]] = []
    pinned_times: List[float] = []
    unpinned_times: List[float] = []

    def _latest_open_round() -> Optional[Dict[str, Any]]:
        for round_obj in reversed(rounds):
            if round_obj.get("departure_time") is None:
                return round_obj
        return rounds[-1] if rounds else None

    for event in job_history:
        if "Request_arrival_time" in event:
            rounds.append({
                "arrival_time": event["Request_arrival_time"],
                "run_start": None,
                "departure_time": None,
                "prompt_length": None,
                "hit_length": None,
                "run_type": None,
                "ttl_estimate": _extract_optional_numeric(
                    event, ["ttl_estimate", "ttl", "estimated_ttl"]),
                "first_token_time": _extract_event_time(
                    event, ["first_token_time", "first_token_ts"]),
                "finish_time": _extract_event_time(
                    event, ["finish_time", "request_finish_time"]),
                "num_generation_tokens": _extract_optional_numeric(
                    event, ["num_generation_tokens", "output_tokens", "gen_tokens"]),
                "total_preempted_block_time": _extract_optional_numeric(
                    event, ["total_preempted_block_time", "preempted_block_time"]),
                "total_remote_kv_wait_time": _extract_optional_numeric(
                    event, ["total_remote_kv_wait_time", "remote_kv_wait_time"]),
                "schedule_skip_count": _extract_optional_numeric(
                    event, ["schedule_skip_count"]),
                "skip_reason_breakdown": event.get("skip_reason_breakdown"),
                "waited_for_remote_kv": event.get("waited_for_remote_kv"),
            })
        elif "waiting_to_running" in event or "evicted_to_running" in event:
            key = "waiting_to_running" if "waiting_to_running" in event else \
                "evicted_to_running"
            if rounds:
                for r in reversed(rounds):
                    if r["run_start"] is None:
                        r["run_start"] = event[key]
                        r["prompt_length"] = _first_not_none(
                            event.get("prompt_length"), r.get("prompt_length"))
                        r["hit_length"] = _first_not_none(
                            event.get("hit_length"), r.get("hit_length"))
                        r["run_type"] = key
                        r["ttl_estimate"] = _first_not_none(
                            r.get("ttl_estimate"),
                            _extract_optional_numeric(
                                event, ["ttl_estimate", "ttl", "estimated_ttl"]))
                        r["first_token_time"] = _first_not_none(
                            r.get("first_token_time"),
                            _extract_event_time(
                                event, ["first_token_time", "first_token_ts"]))
                        r["finish_time"] = _first_not_none(
                            r.get("finish_time"),
                            _extract_event_time(
                                event, ["finish_time", "request_finish_time"]))
                        r["num_generation_tokens"] = _first_not_none(
                            r.get("num_generation_tokens"),
                            _extract_optional_numeric(
                                event,
                                ["num_generation_tokens", "output_tokens", "gen_tokens"]))
                        r["total_preempted_block_time"] = _first_not_none(
                            r.get("total_preempted_block_time"),
                            _extract_optional_numeric(
                                event,
                                ["total_preempted_block_time", "preempted_block_time"]))
                        r["total_remote_kv_wait_time"] = _first_not_none(
                            r.get("total_remote_kv_wait_time"),
                            _extract_optional_numeric(
                                event,
                                ["total_remote_kv_wait_time", "remote_kv_wait_time"]))
                        r["schedule_skip_count"] = _first_not_none(
                            r.get("schedule_skip_count"),
                            _extract_optional_numeric(event, ["schedule_skip_count"]))
                        if r.get("skip_reason_breakdown") is None and \
                                event.get("skip_reason_breakdown") is not None:
                            r["skip_reason_breakdown"] = event.get(
                                "skip_reason_breakdown")
                        r["waited_for_remote_kv"] = _first_not_none(
                            r.get("waited_for_remote_kv"),
                            event.get("waited_for_remote_kv"))
                        break
        elif "Request_departure_time" in event:
            if rounds:
                for r in reversed(rounds):
                    if r["departure_time"] is None:
                        r["departure_time"] = event["Request_departure_time"]
                        r["num_generation_tokens"] = _first_not_none(
                            r.get("num_generation_tokens"),
                            _extract_optional_numeric(
                                event,
                                ["num_generation_tokens", "output_tokens", "gen_tokens"]))
                        r["finish_time"] = _first_not_none(
                            r.get("finish_time"),
                            _extract_event_time(
                                event, ["finish_time", "request_finish_time"]))
                        r["first_token_time"] = _first_not_none(
                            r.get("first_token_time"),
                            _extract_event_time(
                                event, ["first_token_time", "first_token_ts"]))
                        r["ttl_estimate"] = _first_not_none(
                            r.get("ttl_estimate"),
                            _extract_optional_numeric(
                                event, ["ttl_estimate", "ttl", "estimated_ttl"]))
                        break
        elif "pinned_time" in event:
            pinned_times.append(event["pinned_time"])
        elif "unpinned_time" in event:
            unpinned_times.append(event["unpinned_time"])
        else:
            # Handle context events that may carry TTL/first token/finish info.
            target_round = _latest_open_round()
            if target_round is not None:
                ttl_estimate = _extract_optional_numeric(
                    event, ["ttl_estimate", "ttl", "estimated_ttl"])
                if target_round.get("ttl_estimate") is None and ttl_estimate is not None:
                    target_round["ttl_estimate"] = ttl_estimate
                first_token_time = _extract_event_time(
                    event, ["first_token_time", "first_token_ts"])
                if target_round.get("first_token_time") is None and first_token_time is not None:
                    target_round["first_token_time"] = first_token_time
                finish_time = _extract_event_time(
                    event, ["finish_time", "request_finish_time"])
                if target_round.get("finish_time") is None and finish_time is not None:
                    target_round["finish_time"] = finish_time
    return rounds, pinned_times, unpinned_times


def calculate_job_round_metrics(
        timestamps: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute round-level metrics per job, including TTL and pause diagnostics."""
    per_job = {}
    for job_id, history in timestamps.items():
        rounds, pinned_times, unpinned_times = _extract_rounds(history)
        pin_intervals = _build_pin_intervals(pinned_times, unpinned_times)

        tool_call_times = []
        waiting_bubbles = []
        recompute_ratios = []
        kv_reuse = []
        tool_pause_kv_retained = []
        ttl_errors = []
        ttl_gap_count = 0
        ttl_theoretical_hit_count = 0
        ttl_actual_hit_count = 0
        ttl_miss_after_pin_count = 0
        ttl_theoretical_hit_but_actual_miss_count = 0
        ttl_wasted_pin_count = 0

        for idx, r in enumerate(rounds):
            if r["arrival_time"] is not None and r["run_start"] is not None:
                waiting_bubbles.append(r["run_start"] - r["arrival_time"])

            if r["prompt_length"]:
                hit_length = r["hit_length"] or 0
                recompute_ratio = max(
                    0.0,
                    (r["prompt_length"] - hit_length) / r["prompt_length"],
                )
                recompute_ratios.append(recompute_ratio)
                kv_reuse.append(hit_length > 0)

            if idx > 0:
                prev = rounds[idx - 1]
                actual_pause = None
                if prev["departure_time"] is not None and r["arrival_time"] is not None:
                    actual_pause = r["arrival_time"] - prev["departure_time"]

                if prev["departure_time"] is not None:
                    if actual_pause is not None:
                        tool_call_times.append(actual_pause)

                pause_start = prev["departure_time"]
                pause_end = r["arrival_time"]
                pinned_duration_actual = _interval_overlap_duration(
                    pause_start, pause_end, pin_intervals)
                ttl_estimate = _first_not_none(
                    r.get("ttl_estimate"), prev.get("ttl_estimate"))
                ttl_error = (actual_pause - ttl_estimate
                             if actual_pause is not None and ttl_estimate is not None
                             else None)
                ttl_relative_error = _safe_div(ttl_error, actual_pause)
                ttl_coverage_margin = (ttl_estimate - actual_pause
                                       if actual_pause is not None and ttl_estimate is not None
                                       else None)
                ttl_theoretical_hit = (actual_pause <= ttl_estimate
                                       if actual_pause is not None and ttl_estimate is not None
                                       else None)
                hit_length = r.get("hit_length") or 0
                ttl_actual_hit = ((actual_pause <= ttl_estimate) and hit_length > 0
                                  if actual_pause is not None and ttl_estimate is not None
                                  else (hit_length > 0 if actual_pause is not None else None))
                ttl_underestimated = (ttl_error > 0 if ttl_error is not None else None)
                ttl_overestimated = (ttl_error < 0 if ttl_error is not None else None)
                ttl_miss_after_pin = (pinned_duration_actual is not None and
                                      pinned_duration_actual > 0 and
                                      hit_length <= 0)
                ttl_theoretical_hit_but_actual_miss = (
                    ttl_theoretical_hit is True and hit_length <= 0)
                ttl_hit_ratio = _safe_div(r.get("hit_length") or 0,
                                          r.get("prompt_length"))
                ttl_wasted_pin_flag = (
                    pinned_duration_actual is not None and
                    pinned_duration_actual > 0 and
                    not (ttl_actual_hit is True)
                )

                r["actual_pause"] = actual_pause
                r["ttl_estimate"] = ttl_estimate
                r["ttl_error"] = ttl_error
                r["ttl_relative_error"] = ttl_relative_error
                r["ttl_coverage_margin"] = ttl_coverage_margin
                r["ttl_theoretical_hit"] = ttl_theoretical_hit
                r["ttl_actual_hit"] = ttl_actual_hit
                r["ttl_underestimated"] = ttl_underestimated
                r["ttl_overestimated"] = ttl_overestimated
                r["ttl_miss_after_pin"] = ttl_miss_after_pin
                r["ttl_theoretical_hit_but_actual_miss"] = \
                    ttl_theoretical_hit_but_actual_miss
                r["pinned_duration_actual"] = pinned_duration_actual
                r["ttl_hit_ratio"] = ttl_hit_ratio
                r["ttl_wasted_pin_flag"] = ttl_wasted_pin_flag

                if actual_pause is not None:
                    ttl_gap_count += 1
                if ttl_error is not None:
                    ttl_errors.append(ttl_error)
                if ttl_theoretical_hit is True:
                    ttl_theoretical_hit_count += 1
                if ttl_actual_hit is True:
                    ttl_actual_hit_count += 1
                if ttl_miss_after_pin:
                    ttl_miss_after_pin_count += 1
                if ttl_theoretical_hit_but_actual_miss:
                    ttl_theoretical_hit_but_actual_miss_count += 1
                if ttl_wasted_pin_flag:
                    ttl_wasted_pin_count += 1

                retained = bool(ttl_actual_hit)
                tool_pause_kv_retained.append(retained)
            else:
                r["actual_pause"] = None
                r["ttl_error"] = None
                r["ttl_relative_error"] = None
                r["ttl_coverage_margin"] = None
                r["ttl_theoretical_hit"] = None
                r["ttl_actual_hit"] = None
                r["ttl_underestimated"] = None
                r["ttl_overestimated"] = None
                r["ttl_miss_after_pin"] = None
                r["ttl_theoretical_hit_but_actual_miss"] = None
                r["pinned_duration_actual"] = None
                r["ttl_hit_ratio"] = _safe_div(r.get("hit_length") or 0,
                                                r.get("prompt_length"))
                r["ttl_wasted_pin_flag"] = None

        ttl_gap_denominator = ttl_gap_count if ttl_gap_count > 0 else 1

        per_job[job_id] = {
            "num_rounds": len(rounds),
            "rounds": rounds,
            "tool_call_times": tool_call_times,
            "waiting_bubbles": waiting_bubbles,
            "prefill_recompute_ratios": recompute_ratios,
            "kv_reuse_flags": kv_reuse,
            "tool_pause_kv_retained": tool_pause_kv_retained,
            "avg_tool_call_time": (sum(tool_call_times) / len(tool_call_times)
                                   if tool_call_times else 0),
            "avg_waiting_bubble": (sum(waiting_bubbles) / len(waiting_bubbles)
                                   if waiting_bubbles else 0),
            "avg_prefill_recompute_ratio": (sum(recompute_ratios) /
                                            len(recompute_ratios)
                                            if recompute_ratios else 0),
            "kv_reuse_rate": (sum(1 for v in kv_reuse if v) / len(kv_reuse)
                              if kv_reuse else 0),
            "tool_pause_kv_retained_rate":
                (sum(1 for v in tool_pause_kv_retained if v) /
                 len(tool_pause_kv_retained)
                 if tool_pause_kv_retained else 0),
            "ttl_gap_count": ttl_gap_count,
            "ttl_errors": ttl_errors,
            "ttl_theoretical_hit_rate": ttl_theoretical_hit_count / ttl_gap_denominator,
            "ttl_actual_hit_rate": ttl_actual_hit_count / ttl_gap_denominator,
            "ttl_miss_after_pin_rate": ttl_miss_after_pin_count / ttl_gap_denominator,
            "ttl_theoretical_hit_but_actual_miss_rate":
                ttl_theoretical_hit_but_actual_miss_count / ttl_gap_denominator,
            "ttl_wasted_pin_rate": ttl_wasted_pin_count / ttl_gap_denominator,
        }
    return per_job


def _build_monitoring_request_index(
        monitoring: Optional[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Build a best-effort index from monitoring request_stats keyed by (job_id, round_idx)."""
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not monitoring:
        return index

    request_stats = list((monitoring.get("request_stats") or {}).values())
    for rs in request_stats:
        job_key = _first_not_none(
            rs.get("job_id"),
            rs.get("request_id"),
            rs.get("trace_id"),
            rs.get("sample_id"),
        )
        round_idx = _first_not_none(
            rs.get("round_idx"),
            rs.get("turn_idx"),
            rs.get("request_idx"),
        )
        if job_key is None or round_idx is None:
            continue
        index[(str(job_key), int(round_idx))] = rs
    return index


def _build_monitoring_arrival_fallback_index(
        job_round_metrics: Dict[str, Any],
        monitoring: Optional[Dict[str, Any]],
        max_delta_seconds: float = 1.0) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Best-effort fallback matching by arrival_time when IDs are unavailable."""
    if not monitoring:
        return {}

    request_stats = list((monitoring.get("request_stats") or {}).values())
    req_with_arrival = [
        (idx, rs.get("arrival_time"), rs)
        for idx, rs in enumerate(request_stats)
        if rs.get("arrival_time") is not None
    ]
    req_with_arrival.sort(key=lambda x: x[1])

    req_arrivals = [item[1] for item in req_with_arrival]
    req_used = set()

    rounds_with_arrival: List[Tuple[str, int, float]] = []
    for job_id, job_metrics in job_round_metrics.items():
        rounds = job_metrics.get("rounds") or []
        for idx, round_obj in enumerate(rounds):
            arrival_time = round_obj.get("arrival_time")
            if arrival_time is not None:
                rounds_with_arrival.append((str(job_id), idx, arrival_time))

    rounds_with_arrival.sort(key=lambda x: x[2])
    assignments: Dict[Tuple[str, int], Dict[str, Any]] = {}

    for job_id, round_idx, arrival_time in rounds_with_arrival:
        pos = bisect.bisect_left(req_arrivals, arrival_time)
        candidates = []
        for candidate_pos in range(max(0, pos - 8), min(len(req_arrivals), pos + 8)):
            req_idx, req_arrival, req_obj = req_with_arrival[candidate_pos]
            if req_idx in req_used:
                continue
            delta = abs(req_arrival - arrival_time)
            if delta <= max_delta_seconds:
                candidates.append((delta, req_idx, req_obj))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        _, matched_req_idx, matched_req_obj = candidates[0]
        req_used.add(matched_req_idx)
        assignments[(job_id, round_idx)] = matched_req_obj

    return assignments


def build_round_records(
        timestamps: Dict[str, List[Dict[str, Any]]],
        job_round_metrics: Dict[str, Any],
        monitoring: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten per-job rounds into unified request/round records for diagnosis."""
    del timestamps  # Keeps signature future-proof while preserving compatibility.

    monitoring_index = _build_monitoring_request_index(monitoring)
    monitoring_arrival_fallback = _build_monitoring_arrival_fallback_index(
        job_round_metrics,
        monitoring,
    )
    records: List[Dict[str, Any]] = []

    for job_id, job_metrics in job_round_metrics.items():
        rounds = job_metrics.get("rounds") or []
        num_rounds = len(rounds)

        for idx, round_obj in enumerate(rounds):
            rs = monitoring_index.get((str(job_id), idx))
            if rs is None:
                rs = monitoring_arrival_fallback.get((str(job_id), idx), {})

            arrival_time = _first_not_none(round_obj.get("arrival_time"), rs.get("arrival_time"))
            run_start = _first_not_none(round_obj.get("run_start"), rs.get("run_start"))
            departure_time = _first_not_none(
                round_obj.get("departure_time"),
                round_obj.get("finish_time"),
                rs.get("finish_time"),
            )
            first_token_time = _first_not_none(round_obj.get("first_token_time"), rs.get("first_token_time"))

            queue_wait = (run_start - arrival_time
                          if run_start is not None and arrival_time is not None
                          else None)
            prefill_duration = (first_token_time - run_start
                                if first_token_time is not None and run_start is not None
                                else None)
            decode_duration = (departure_time - first_token_time
                               if departure_time is not None and first_token_time is not None
                               else None)

            ttft = _first_not_none(
                rs.get("first_token_latency"),
                (first_token_time - arrival_time
                 if first_token_time is not None and arrival_time is not None else None),
            )
            e2e = _first_not_none(
                rs.get("e2e_latency"),
                (departure_time - arrival_time
                 if departure_time is not None and arrival_time is not None else None),
            )

            total_preempted_block_time = _first_not_none(
                round_obj.get("total_preempted_block_time"),
                rs.get("total_preempted_block_time"),
                rs.get("preempted_block_time"),
                0,
            )
            total_remote_kv_wait_time = _first_not_none(
                round_obj.get("total_remote_kv_wait_time"),
                rs.get("total_remote_kv_wait_time"),
                rs.get("remote_kv_wait_time"),
                0,
            )
            schedule_skip_count = _first_not_none(
                round_obj.get("schedule_skip_count"),
                rs.get("schedule_skip_count"),
                0,
            )
            skip_reason_breakdown = _first_not_none(
                round_obj.get("skip_reason_breakdown"),
                rs.get("skip_reason_breakdown"),
                {},
            )

            if num_rounds == 1:
                round_type = "first"
            elif idx == 0:
                round_type = "first"
            elif idx == num_rounds - 1:
                round_type = "final"
            elif round_obj.get("actual_pause") is not None:
                round_type = "tool_return"
            else:
                round_type = "middle"

            if round_obj.get("ttl_actual_hit") is True:
                ttl_state = "hit"
            elif round_obj.get("ttl_miss_after_pin") is True:
                ttl_state = "miss_after_pin"
            elif round_obj.get("actual_pause") is not None and \
                    (round_obj.get("pinned_duration_actual") in (0, 0.0, None)):
                ttl_state = "never_pinned"
            else:
                ttl_state = "unknown"

            was_preempted = (
                total_preempted_block_time is not None and total_preempted_block_time > 0
            ) or (round_obj.get("run_type") == "evicted_to_running")

            waited_for_remote_kv = (
                (total_remote_kv_wait_time is not None and total_remote_kv_wait_time > 0) or
                bool(round_obj.get("waited_for_remote_kv"))
            )

            prompt_length = _first_not_none(
                round_obj.get("prompt_length"), rs.get("num_prompt_tokens"))
            hit_length = _first_not_none(
                round_obj.get("hit_length"), rs.get("hit_length"), 0)
            num_generation_tokens = _first_not_none(
                round_obj.get("num_generation_tokens"), rs.get("num_generation_tokens"))

            record = {
                "job_id": job_id,
                "round_idx": idx,
                "arrival_time": arrival_time,
                "run_start": run_start,
                "departure_time": departure_time,
                "prompt_length": prompt_length,
                "hit_length": hit_length,
                "run_type": round_obj.get("run_type"),
                "ttft": ttft,
                "e2e": e2e,
                "queue_wait_to_first_schedule": queue_wait,
                "prefill_duration": prefill_duration,
                "decode_duration": decode_duration,
                "total_preempted_block_time": total_preempted_block_time,
                "total_remote_kv_wait_time": total_remote_kv_wait_time,
                "schedule_skip_count": schedule_skip_count,
                "skip_reason_breakdown": skip_reason_breakdown,
                "actual_pause": round_obj.get("actual_pause"),
                "ttl_estimate": round_obj.get("ttl_estimate"),
                "ttl_error": round_obj.get("ttl_error"),
                "ttl_relative_error": round_obj.get("ttl_relative_error"),
                "ttl_coverage_margin": round_obj.get("ttl_coverage_margin"),
                "ttl_theoretical_hit": round_obj.get("ttl_theoretical_hit"),
                "ttl_actual_hit": round_obj.get("ttl_actual_hit"),
                "ttl_underestimated": round_obj.get("ttl_underestimated"),
                "ttl_overestimated": round_obj.get("ttl_overestimated"),
                "ttl_miss_after_pin": round_obj.get("ttl_miss_after_pin"),
                "ttl_theoretical_hit_but_actual_miss":
                    round_obj.get("ttl_theoretical_hit_but_actual_miss"),
                "pinned_duration_actual": round_obj.get("pinned_duration_actual"),
                "ttl_hit_ratio": _first_not_none(
                    round_obj.get("ttl_hit_ratio"),
                    _safe_div(hit_length or 0, prompt_length),
                ),
                "ttl_wasted_pin_flag": round_obj.get("ttl_wasted_pin_flag"),
                "round_type": round_type,
                "ttl_state": ttl_state,
                "was_preempted": was_preempted,
                "waited_for_remote_kv": waited_for_remote_kv,
                "prompt_bucket": _prompt_bucket(prompt_length),
                "gen_bucket": _gen_bucket(num_generation_tokens),
                "tool_time_bucket": _tool_time_bucket(round_obj.get("actual_pause")),
            }
            records.append(record)

    return records


def summarize_records_by_group(
        records: List[Dict[str, Any]], group_key: str,
        numeric_fields: List[str]) -> Dict[str, Any]:
    """Summarize numeric fields by a categorical group key over round records."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        group_value = record.get(group_key)
        group_name = str(group_value) if group_value is not None else "unknown"
        if group_name not in grouped:
            grouped[group_name] = {"count": 0}
            for field in numeric_fields:
                grouped[group_name][field] = []

        grouped[group_name]["count"] += 1
        for field in numeric_fields:
            value = record.get(field)
            if isinstance(value, (int, float)):
                grouped[group_name][field].append(value)

    summary: Dict[str, Any] = {}
    for group_name, stats in grouped.items():
        summary[group_name] = {"count": stats["count"]}
        for field in numeric_fields:
            summary[group_name][field] = summarize_distribution(stats[field])
    return summary


def summarize_ttl_metrics(round_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate TTL estimation quality metrics across all round gaps."""
    ttl_error_values = [
        r["ttl_error"] for r in round_records
        if isinstance(r.get("ttl_error"), (int, float))
    ]
    ttl_gap_records = [r for r in round_records if r.get("actual_pause") is not None]
    ttl_gap_count = len(ttl_gap_records)

    def _rate(flag_key: str) -> float:
        if ttl_gap_count == 0:
            return 0.0
        return sum(1 for r in ttl_gap_records if r.get(flag_key) is True) / ttl_gap_count

    return {
        "ttl_error_distribution": summarize_distribution(ttl_error_values),
        "theoretical_hit_rate": _rate("ttl_theoretical_hit"),
        "actual_hit_rate": _rate("ttl_actual_hit"),
        "miss_after_pin_rate": _rate("ttl_miss_after_pin"),
        "theoretical_hit_but_actual_miss_rate": _rate(
            "ttl_theoretical_hit_but_actual_miss"),
        "wasted_pin_rate": _rate("ttl_wasted_pin_flag"),
        "ttl_gap_count": ttl_gap_count,
    }


def build_tail_analysis(
        round_records: List[Dict[str, Any]], top_k: int = 20) -> Dict[str, Any]:
    """Build top-k slow request diagnostics for TTFT, E2E, and queue wait."""
    fields_to_keep = [
        "job_id",
        "round_idx",
        "ttft",
        "e2e",
        "queue_wait_to_first_schedule",
        "prefill_duration",
        "decode_duration",
        "actual_pause",
        "ttl_estimate",
        "ttl_error",
        "ttl_state",
        "round_type",
        "prompt_length",
        "hit_length",
        "ttl_hit_ratio",
        "run_type",
        "was_preempted",
        "waited_for_remote_kv",
        "schedule_skip_count",
    ]

    def _top_by(metric_key: str) -> List[Dict[str, Any]]:
        eligible = [
            r for r in round_records
            if isinstance(r.get(metric_key), (int, float))
        ]
        eligible.sort(key=lambda x: x.get(metric_key, 0), reverse=True)
        top = eligible[:top_k]
        return [{k: row.get(k) for k in fields_to_keep} for row in top]

    return {
        "top_ttft_requests": _top_by("ttft"),
        "top_e2e_requests": _top_by("e2e"),
        "top_queue_wait_requests": _top_by("queue_wait_to_first_schedule"),
    }


def calculate_monitoring_metrics(
    monitoring: Optional[Dict[str, Any]],
    timestamps: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    instance_role: Optional[str] = None) -> Dict[str, Any]:
    if not monitoring:
        monitoring = {}

    request_stats = list((monitoring.get("request_stats") or {}).values())
    scheduler_stats = monitoring.get("scheduler_stats") or []
    iteration_stats = monitoring.get("iteration_stats") or []

    ttft_values = []
    e2e_values = []
    tbt_values = []
    total_gen_tokens = 0
    arrival_times = []
    finish_times = []

    for rs in request_stats:
        if rs.get("first_token_latency") is not None:
            ttft_values.append(rs["first_token_latency"])
        if rs.get("e2e_latency") is not None:
            e2e_values.append(rs["e2e_latency"])
        decode_time = rs.get("decode_time")
        num_gen = rs.get("num_generation_tokens", 0)
        if decode_time is not None and num_gen and num_gen > 1:
            tbt_values.append(decode_time / (num_gen - 1))
        if rs.get("num_generation_tokens") is not None:
            total_gen_tokens += rs["num_generation_tokens"]
        if rs.get("arrival_time") is not None:
            arrival_times.append(rs["arrival_time"])
        if rs.get("finish_time") is not None:
            finish_times.append(rs["finish_time"])

    duration = 0.0
    if arrival_times and finish_times:
        duration = max(finish_times) - min(arrival_times)

    request_throughput = (len(request_stats) / duration
                          if duration > 0 else 0)
    output_throughput = (total_gen_tokens / duration
                         if duration > 0 else 0)

    kv_usage = [s.get("kv_cache_usage", 0) for s in scheduler_stats]
    kv_usage_pct = [v * 100 for v in kv_usage]

    prefix_queries = sum(
        s.get("prefix_cache_queries", 0) for s in scheduler_stats)
    prefix_hits = sum(
        s.get("prefix_cache_hits", 0) for s in scheduler_stats)

    num_running = [s.get("num_running", 0) for s in scheduler_stats]
    num_waiting = [s.get("num_waiting", 0) for s in scheduler_stats]
    num_waiting_remote = [
        s.get("num_waiting_for_remote_kvs", 0) for s in scheduler_stats
    ]
    num_preempted = [s.get("num_preempted", 0) for s in scheduler_stats]
    num_swapped = [a + b for a, b in zip(num_waiting_remote, num_preempted)]

    result = {
        "ttft_seconds": summarize_distribution(ttft_values),
        "tbt_seconds": summarize_distribution(tbt_values),
        "e2e_latency_seconds": summarize_distribution(e2e_values),
        "request_throughput_rps": request_throughput,
        "output_throughput_tps": output_throughput,
        "kv_cache_usage_percent": summarize_distribution(kv_usage_pct),
        "prefix_cache_totals": {
            "queries": prefix_queries,
            "hits": prefix_hits,
            "hit_rate": (prefix_hits / prefix_queries
                         if prefix_queries > 0 else 0),
        },
        "num_requests_running": summarize_distribution(num_running),
        "num_requests_waiting": summarize_distribution(num_waiting),
        "num_requests_waiting_for_remote_kvs": summarize_distribution(
            num_waiting_remote),
        "num_requests_preempted": summarize_distribution(num_preempted),
        "num_requests_swapped": summarize_distribution(num_swapped),
        "scheduler_stats": scheduler_stats,
        "iteration_stats": iteration_stats,
    }

    # Fallback path for disaggregated runs where request_stats may be empty
    # in scheduler-side dump files.
    if timestamps:
        fallback_stage_latency = []
        fallback_prefill_latency = []
        fallback_decode_tbt = []
        fallback_total_gen_tokens = 0
        fallback_arrivals = []
        fallback_departures = []

        for history in timestamps.values():
            rounds, _, _ = _extract_rounds(history)
            for round_obj in rounds:
                arrival_time = round_obj.get("arrival_time")
                run_start = round_obj.get("run_start")
                departure_time = round_obj.get("departure_time")
                num_gen = round_obj.get("num_generation_tokens")

                if arrival_time is not None:
                    fallback_arrivals.append(arrival_time)
                if departure_time is not None:
                    fallback_departures.append(departure_time)

                if arrival_time is not None and departure_time is not None:
                    stage_latency = departure_time - arrival_time
                    if stage_latency >= 0:
                        fallback_stage_latency.append(stage_latency)
                        if instance_role == "prefill":
                            fallback_prefill_latency.append(stage_latency)

                if instance_role == "decode" and \
                        run_start is not None and departure_time is not None and \
                        departure_time >= run_start:
                    decode_duration = departure_time - run_start
                    if isinstance(num_gen, (int, float)) and num_gen > 0:
                        fallback_decode_tbt.append(decode_duration / num_gen)
                        fallback_total_gen_tokens += num_gen
                    else:
                        # Legacy files may not have num_generation_tokens.
                        # Keep chart non-empty with a conservative fallback.
                        fallback_decode_tbt.append(decode_duration)

        # E2E fallback: role-local request latency from scheduler timestamps.
        if result["e2e_latency_seconds"].get("count", 0) == 0 and fallback_stage_latency:
            result["e2e_latency_seconds"] = summarize_distribution(
                fallback_stage_latency)

        # TTFT fallback semantics for prefill role.
        if instance_role == "prefill" and \
                result["ttft_seconds"].get("count", 0) == 0 and fallback_prefill_latency:
            result["ttft_seconds"] = summarize_distribution(
                fallback_prefill_latency)

        # TBT fallback semantics for decode role.
        if instance_role == "decode" and \
                result["tbt_seconds"].get("count", 0) == 0 and fallback_decode_tbt:
            result["tbt_seconds"] = summarize_distribution(fallback_decode_tbt)

        # Throughput fallback if request_stats were empty.
        if result.get("request_throughput_rps", 0) == 0 and \
                fallback_arrivals and fallback_departures:
            total_window = max(fallback_departures) - min(fallback_arrivals)
            if total_window > 0:
                req_count = len(fallback_departures)
                result["request_throughput_rps"] = req_count / total_window
                if result.get("output_throughput_tps", 0) == 0 and fallback_total_gen_tokens > 0:
                    result["output_throughput_tps"] = fallback_total_gen_tokens / total_window

        result["metric_fallback"] = {
            "used_scheduler_fallback": True,
            "instance_role": instance_role,
            "ttft_semantics": "prefill_stage_latency_departure_minus_arrival",
            "tbt_semantics": (
                "decode_stage_duration_div_num_generation_tokens "
                "(or decode_stage_duration if token count missing)"),
            "e2e_semantics": "role_local_latency_departure_minus_arrival",
        }

    return result


def save_results(results: Dict[str, Any], output_dir: str):
    """Save results to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

    # Also print summary to console
    print("\n=== Job Duration Statistics ===")
    print(f"Number of jobs: {results['num_jobs']}")
    print(f"Average duration: {results['average_duration']:.2f} seconds")
    print(f"Total duration: {results['total_duration']:.2f} seconds")
    print(f"Min duration: {results['min_duration']:.2f} seconds")
    print(f"Max duration: {results['max_duration']:.2f} seconds")
    print(f"Median duration: {results['median_duration']:.2f} seconds")
    print(f"95th percentile duration: {results['percentile_95']:.2f} seconds")
    print(f"99th percentile duration: {results['percentile_99']:.2f} seconds")
    monitoring_metrics = results.get("monitoring_metrics") or {}
    if monitoring_metrics:
        print("\n=== Monitoring Metrics ===")
        print("TTFT (s) p50/p95/p99:",
              f"{monitoring_metrics['ttft_seconds']['p50']:.4f}/"
              f"{monitoring_metrics['ttft_seconds']['p95']:.4f}/"
              f"{monitoring_metrics['ttft_seconds']['p99']:.4f}")
        print("E2E (s) p50/p95/p99:",
              f"{monitoring_metrics['e2e_latency_seconds']['p50']:.4f}/"
              f"{monitoring_metrics['e2e_latency_seconds']['p95']:.4f}/"
              f"{monitoring_metrics['e2e_latency_seconds']['p99']:.4f}")
        print("Request throughput (req/s):",
              f"{monitoring_metrics['request_throughput_rps']:.2f}")
        print("Output throughput (tok/s):",
              f"{monitoring_metrics['output_throughput_tps']:.2f}")
        print("KV cache usage p50/p95/p99 (%):",
              f"{monitoring_metrics['kv_cache_usage_percent']['p50']:.2f}/"
              f"{monitoring_metrics['kv_cache_usage_percent']['p95']:.2f}/"
              f"{monitoring_metrics['kv_cache_usage_percent']['p99']:.2f}")

    ttl_summary = results.get("ttl_metrics_summary") or {}
    if ttl_summary:
        print("\n=== TTL Metrics Summary ===")
        print("TTL theoretical hit rate:",
              f"{ttl_summary.get('theoretical_hit_rate', 0.0):.4f}")
        print("TTL actual hit rate:",
              f"{ttl_summary.get('actual_hit_rate', 0.0):.4f}")
        print("TTL miss-after-pin rate:",
              f"{ttl_summary.get('miss_after_pin_rate', 0.0):.4f}")
        print("TTL theoretical-hit-but-actual-miss rate:",
              f"{ttl_summary.get('theoretical_hit_but_actual_miss_rate', 0.0):.4f}")

    grouped = (results.get("grouped_metrics") or {}).get("by_ttl_state") or {}
    if grouped:
        print("\n=== TTFT by TTL State (p95/p99) ===")
        for state, stats in grouped.items():
            ttft_stats = stats.get("ttft") or {}
            print(
                f"{state}: p95={ttft_stats.get('p95', 0):.4f}, "
                f"p99={ttft_stats.get('p99', 0):.4f}, "
                f"count={stats.get('count', 0)}"
            )

    top_ttft = (results.get("tail_analysis") or {}).get("top_ttft_requests") or []
    if top_ttft:
        print("\n=== Top 5 Slow TTFT Requests ===")
        for row in top_ttft[:5]:
            print(
                f"job={row.get('job_id')}, round={row.get('round_idx')}, "
                f"ttft={row.get('ttft')}, e2e={row.get('e2e')}, "
                f"queue_wait={row.get('queue_wait_to_first_schedule')}, "
                f"ttl_state={row.get('ttl_state')}, round_type={row.get('round_type')}"
            )
    print("=" * 35)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average job duration from scheduler_timestamps"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/continuum_exp",
        help="Directory containing scheduler_timestamps file (default: ./continuum_exp)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis",
        help="Directory to save results (default: ./continuum_exp)"
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="",
        help=("Instance directory name under --input-dir, e.g. prefill_20003 "
              "or decode_20005")
    )
    parser.add_argument(
        "--all-instances",
        action="store_true",
        help=("Analyze all instance subdirectories under --input-dir that "
              "contain scheduler_timestamps")
    )

    args = parser.parse_args()

    input_dir = resolve_input_dir(args.input_dir)
    output_dir = Path(args.output_dir)

    def _run_one(instance_input_dir: Path, instance_output_dir: Path) -> None:
        print(f"Loading scheduler_timestamps from {instance_input_dir}...")
        timestamps = load_scheduler_timestamps(str(instance_input_dir))
        monitoring = load_monitoring_timestamps(str(instance_input_dir))
        instance_role = detect_instance_role(instance_input_dir)
        companion_role, companion_timestamps = load_companion_role_timestamps(
            instance_input_dir)

        print(f"Calculating job durations for {len(timestamps)} jobs...")
        results = calculate_average_duration(timestamps)
        results["job_round_metrics"] = calculate_job_round_metrics(timestamps)

        # In disaggregated mode, normalize tool-call gaps to the canonical
        # decode(i) departure -> prefill(i+1) arrival definition.
        if companion_timestamps is not None and \
                {instance_role, companion_role} == {"prefill", "decode"}:
            if instance_role == "prefill":
                prefill_timestamps = timestamps
                decode_timestamps = companion_timestamps
            else:
                prefill_timestamps = companion_timestamps
                decode_timestamps = timestamps
            canonical_pauses = compute_decode_to_next_prefill_pauses(
                prefill_timestamps,
                decode_timestamps,
            )
            apply_canonical_toolcall_times(
                results["job_round_metrics"], canonical_pauses)

        results["monitoring_metrics"] = calculate_monitoring_metrics(
            monitoring,
            timestamps=timestamps,
            instance_role=instance_role,
        )
        results["source_input_dir"] = str(instance_input_dir)
        results["instance_role"] = instance_role

        round_records = build_round_records(
            timestamps,
            results["job_round_metrics"],
            monitoring,
        )

        grouped_numeric_fields = [
            "ttft",
            "e2e",
            "queue_wait_to_first_schedule",
            "prefill_duration",
            "decode_duration",
            "actual_pause",
            "ttl_error",
            "ttl_hit_ratio",
        ]

        results["ttl_metrics_summary"] = summarize_ttl_metrics(round_records)
        results["round_records"] = round_records
        results["grouped_metrics"] = {
            "by_ttl_state": summarize_records_by_group(
                round_records, "ttl_state", grouped_numeric_fields),
            "by_round_type": summarize_records_by_group(
                round_records, "round_type", grouped_numeric_fields),
            "by_prompt_bucket": summarize_records_by_group(
                round_records, "prompt_bucket", grouped_numeric_fields),
            "by_tool_time_bucket": summarize_records_by_group(
                round_records, "tool_time_bucket", grouped_numeric_fields),
            "by_was_preempted": summarize_records_by_group(
                round_records, "was_preempted", grouped_numeric_fields),
        }
        results["tail_analysis"] = build_tail_analysis(round_records, top_k=20)

        save_results(results, str(instance_output_dir))

    if args.all_instances:
        instance_dirs = discover_instance_dirs(input_dir)
        if not instance_dirs:
            raise FileNotFoundError(
                "No scheduler_timestamps found under "
                f"{input_dir}. Original --input-dir was {args.input_dir}.")
        print(f"Found {len(instance_dirs)} instance directories.")
        for instance_dir in instance_dirs:
            print(f"\n=== Analyzing instance: {instance_dir.name} ===")
            _run_one(instance_dir, output_dir / instance_dir.name)
        return

    if args.instance:
        selected_input_dir = input_dir / args.instance
    else:
        instance_dirs = discover_instance_dirs(input_dir)
        if not instance_dirs:
            selected_input_dir = input_dir
        elif len(instance_dirs) == 1:
            selected_input_dir = instance_dirs[0]
        elif input_dir in instance_dirs:
            selected_input_dir = input_dir
        else:
            discovered = ", ".join(d.name for d in instance_dirs)
            raise ValueError(
                "Multiple instance directories found. "
                f"Use --instance or --all-instances. Found: {discovered}")

    _run_one(selected_input_dir, output_dir)


if __name__ == "__main__":
    main()
