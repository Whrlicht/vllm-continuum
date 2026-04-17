#!/usr/bin/env python3
"""Summarize continuum_exp/results.json into a human-friendly report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt_sec(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}s"


def _fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def _get(d: Dict[str, Any], path: str, default: Any = 0) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _line(label: str, value: str, width: int = 28) -> str:
    return f"{label:<{width}} {value}"


def _section(title: str) -> str:
    return f"\n== {title} =="


def summarize_results(results: Dict[str, Any], top_k: int) -> str:
    lines: List[str] = []

    lines.append(_section("Job Duration"))
    lines.append(_line("num_jobs", str(results.get("num_jobs", 0))))
    lines.append(_line("average", _fmt_sec(results.get("average_duration", 0))))
    lines.append(_line("total", _fmt_sec(results.get("total_duration", 0))))
    lines.append(_line("min", _fmt_sec(results.get("min_duration", 0))))
    lines.append(_line("median", _fmt_sec(results.get("median_duration", 0))))
    lines.append(_line("p95", _fmt_sec(results.get("percentile_95", 0))))
    lines.append(_line("p99", _fmt_sec(results.get("percentile_99", 0))))
    lines.append(_line("max", _fmt_sec(results.get("max_duration", 0))))

    monitoring = results.get("monitoring_metrics") or {}
    if monitoring:
        lines.append(_section("Latency"))
        lines.append(_line("TTFT p50/p95/p99",
                           f"{_fmt_sec(_get(monitoring, 'ttft_seconds.p50'))} / "
                           f"{_fmt_sec(_get(monitoring, 'ttft_seconds.p95'))} / "
                           f"{_fmt_sec(_get(monitoring, 'ttft_seconds.p99'))}"))
        lines.append(_line("TBT p50/p95/p99",
                           f"{_fmt_sec(_get(monitoring, 'tbt_seconds.p50'))} / "
                           f"{_fmt_sec(_get(monitoring, 'tbt_seconds.p95'))} / "
                           f"{_fmt_sec(_get(monitoring, 'tbt_seconds.p99'))}"))
        lines.append(_line("E2E p50/p95/p99",
                           f"{_fmt_sec(_get(monitoring, 'e2e_latency_seconds.p50'))} / "
                           f"{_fmt_sec(_get(monitoring, 'e2e_latency_seconds.p95'))} / "
                           f"{_fmt_sec(_get(monitoring, 'e2e_latency_seconds.p99'))}"))

        lines.append(_section("Throughput"))
        lines.append(_line("request throughput", f"{_fmt_num(_get(monitoring, 'request_throughput_rps'))} req/s"))
        lines.append(_line("output throughput", f"{_fmt_num(_get(monitoring, 'output_throughput_tps'))} tok/s"))

        lines.append(_section("Scheduler"))
        lines.append(_line("running reqs p50/p95/p99",
                           f"{_fmt_num(_get(monitoring, 'num_requests_running.p50'))} / "
                           f"{_fmt_num(_get(monitoring, 'num_requests_running.p95'))} / "
                           f"{_fmt_num(_get(monitoring, 'num_requests_running.p99'))}"))
        lines.append(_line("waiting reqs p50/p95/p99",
                           f"{_fmt_num(_get(monitoring, 'num_requests_waiting.p50'))} / "
                           f"{_fmt_num(_get(monitoring, 'num_requests_waiting.p95'))} / "
                           f"{_fmt_num(_get(monitoring, 'num_requests_waiting.p99'))}"))
        lines.append(_line("swapped reqs p50/p95/p99",
                           f"{_fmt_num(_get(monitoring, 'num_requests_swapped.p50'))} / "
                           f"{_fmt_num(_get(monitoring, 'num_requests_swapped.p95'))} / "
                           f"{_fmt_num(_get(monitoring, 'num_requests_swapped.p99'))}"))
        lines.append(_line("KV cache usage p50/p95/p99",
                           f"{_fmt_num(_get(monitoring, 'kv_cache_usage_percent.p50'))}% / "
                           f"{_fmt_num(_get(monitoring, 'kv_cache_usage_percent.p95'))}% / "
                           f"{_fmt_num(_get(monitoring, 'kv_cache_usage_percent.p99'))}%"))
        lines.append(_line("prefix cache hit rate",
                           f"{_fmt_num(_get(monitoring, 'prefix_cache_totals.hit_rate') * 100)}%"))
        lines.append(_line("prefix cache queries",
                           str(_get(monitoring, 'prefix_cache_totals.queries'))))
        lines.append(_line("prefix cache hits",
                           str(_get(monitoring, 'prefix_cache_totals.hits'))))

    job_round = results.get("job_round_metrics") or {}
    if job_round:
        # Build a few per-job rankings
        by_duration = sorted((
            (jid, d) for jid, d in (results.get("job_durations") or {}).items()
        ), key=lambda x: x[1], reverse=True)
        by_rounds = sorted((
            (jid, m.get("num_rounds", 0)) for jid, m in job_round.items()
        ), key=lambda x: x[1], reverse=True)
        by_wait = sorted((
            (jid, m.get("avg_waiting_bubble", 0)) for jid, m in job_round.items()
        ), key=lambda x: x[1], reverse=True)
        by_tool = sorted((
            (jid, m.get("avg_tool_call_time", 0)) for jid, m in job_round.items()
        ), key=lambda x: x[1], reverse=True)
        by_recompute = sorted((
            (jid, m.get("avg_prefill_recompute_ratio", 0)) for jid, m in job_round.items()
        ), key=lambda x: x[1], reverse=True)

        lines.append(_section(f"Top Jobs (top {top_k})"))
        lines.append("Longest duration:")
        for jid, v in by_duration[:top_k]:
            lines.append(f"  job {jid}: {_fmt_sec(v)}")

        lines.append("Most rounds:")
        for jid, v in by_rounds[:top_k]:
            lines.append(f"  job {jid}: {v} rounds")

        lines.append("Largest waiting bubble:")
        for jid, v in by_wait[:top_k]:
            lines.append(f"  job {jid}: {_fmt_sec(v)}")

        lines.append("Largest tool-call time:")
        for jid, v in by_tool[:top_k]:
            lines.append(f"  job {jid}: {_fmt_sec(v)}")

        lines.append("Highest prefill recompute ratio:")
        for jid, v in by_recompute[:top_k]:
            lines.append(f"  job {jid}: {_fmt_num(v * 100)}%")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize results.json into a human-friendly report")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to results.json")
    parser.add_argument(
        "--output",
        default="",
        help="Optional output file path for the report text")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K jobs to show in rankings (default: 5)")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r") as f:
        results = json.load(f)

    report = summarize_results(results, args.top_k)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)


if __name__ == "__main__":
    main()
