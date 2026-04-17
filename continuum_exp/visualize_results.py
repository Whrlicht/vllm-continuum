#!/usr/bin/env python3
"""Visualize results.json as charts (PNG) and a simple HTML dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_input_path(path_arg: str) -> Path:
    """Resolve input path from cwd first, then repository root."""
    raw = Path(path_arg).expanduser()
    if raw.is_absolute():
        return raw

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parent.parent
    repo_candidate = (repo_root / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate


def _get(d: Dict[str, Any], path: str, default: Any = 0) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _bar_p50_p95_p99(title: str, stats: Dict[str, Any], out_path: Path) -> None:
    labels = ["p50", "p95", "p99"]
    values = [stats.get("p50", 0), stats.get("p95", 0), stats.get("p99", 0)]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, values, color=["#4C78A8", "#F58518", "#E45756"])
    ax.set_title(title)
    ax.set_ylabel("seconds")
    _save_fig(fig, out_path)


def _bar_simple(title: str, labels: List[str], values: List[float], out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _save_fig(fig, out_path)


def _time_series(title: str, x: List[float], ys: List[Tuple[str, List[float]]], out_path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, y in ys:
        if y:
            ax.plot(x, y, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right")
    _save_fig(fig, out_path)


def _histogram(title: str, values: List[float], out_path: Path, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(values, bins=30, color="#72B7B2")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    _save_fig(fig, out_path)


def _topk_bar(title: str, items: List[Tuple[str, float]], out_path: Path, ylabel: str) -> None:
    labels = [f"job {k}" for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color="#E45756")
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.invert_yaxis()
    _save_fig(fig, out_path)


def _normalize_time(ts: List[float]) -> List[float]:
    if not ts:
        return []
    t0 = min(ts)
    return [t - t0 for t in ts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize results.json")
    parser.add_argument(
        "--input",
        default="",
        help="Path to results.json for single-instance visualization")
    parser.add_argument(
        "--input-dir",
        default="",
        help=("Directory containing results.json or per-instance subdirectories "
              "with results.json"))
    parser.add_argument(
        "--all-instances",
        action="store_true",
        help="Visualize all instance subdirectories under --input-dir")
    parser.add_argument("--output-dir", required=True, help="Output directory for charts")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K jobs in charts")
    args = parser.parse_args()

    def _render_one(input_path: Path, out_dir: Path) -> None:
        _ensure_dir(out_dir)

        with input_path.open("r") as f:
            results = json.load(f)

        monitoring = results.get("monitoring_metrics") or {}
        job_round = results.get("job_round_metrics") or {}

        generated = []

        # Latency p50/p95/p99
        if monitoring:
            _bar_p50_p95_p99("TTFT (Prefill Stage, s)", monitoring.get("ttft_seconds", {}), out_dir / "latency_ttft.png")
            _bar_p50_p95_p99("TBT (Decode Per Token, s)", monitoring.get("tbt_seconds", {}), out_dir / "latency_tbt.png")
            _bar_p50_p95_p99("E2E (Role-Local, s)", monitoring.get("e2e_latency_seconds", {}), out_dir / "latency_e2e.png")
            generated += ["latency_ttft.png", "latency_tbt.png", "latency_e2e.png"]

            # Throughput
            _bar_simple(
                "Throughput",
                ["request", "output"],
                [
                    _get(monitoring, "request_throughput_rps"),
                    _get(monitoring, "output_throughput_tps"),
                ],
                out_dir / "throughput.png",
                "per second",
            )
            generated.append("throughput.png")

            # Scheduler time series
            scheduler_stats = monitoring.get("scheduler_stats") or []
            if scheduler_stats:
                times = [s.get("timestamp", 0) for s in scheduler_stats]
                x = _normalize_time(times)
                running = [s.get("num_running", 0) for s in scheduler_stats]
                waiting = [s.get("num_waiting", 0) for s in scheduler_stats]
                swapped = [
                    (s.get("num_waiting_for_remote_kvs", 0) + s.get("num_preempted", 0))
                    for s in scheduler_stats
                ]
                kv_usage_pct = [s.get("kv_cache_usage", 0) * 100 for s in scheduler_stats]

                _time_series("Requests in Scheduler", x,
                             [("running", running), ("waiting", waiting), ("swapped", swapped)],
                             out_dir / "scheduler_reqs.png",
                             "requests")
                _time_series("KV Cache Usage", x,
                             [("usage%", kv_usage_pct)],
                             out_dir / "kv_usage.png",
                             "%")
                generated += ["scheduler_reqs.png", "kv_usage.png"]

        # Job-level stats
        if job_round:
            num_rounds = [m.get("num_rounds", 0) for m in job_round.values()]
            avg_wait = [m.get("avg_waiting_bubble", 0) for m in job_round.values()]
            avg_tool = [m.get("avg_tool_call_time", 0) for m in job_round.values()]
            recompute = [m.get("avg_prefill_recompute_ratio", 0) * 100 for m in job_round.values()]
            kv_reuse = [m.get("kv_reuse_rate", 0) * 100 for m in job_round.values()]

            _histogram("Job Rounds", num_rounds, out_dir / "job_rounds_hist.png", "rounds")
            _histogram("Avg Waiting Bubble (s)", avg_wait, out_dir / "job_waiting_hist.png", "seconds")
            _histogram("Avg Tool Call Time (s)", avg_tool, out_dir / "job_toolcall_hist.png", "seconds")
            _histogram("Prefill Recompute Ratio (%)", recompute, out_dir / "job_recompute_hist.png", "percent")
            _histogram("KV Reuse Rate (%)", kv_reuse, out_dir / "job_kv_reuse_hist.png", "percent")
            generated += [
                "job_rounds_hist.png",
                "job_waiting_hist.png",
                "job_toolcall_hist.png",
                "job_recompute_hist.png",
                "job_kv_reuse_hist.png",
            ]

            # Top-K longest jobs
            job_durations = results.get("job_durations") or {}
            top = sorted(job_durations.items(), key=lambda x: x[1], reverse=True)[:args.top_k]
            if top:
                _topk_bar("Top Job Durations", top, out_dir / "top_job_durations.png", "seconds")
                generated.append("top_job_durations.png")

        # Write a simple HTML
        html_path = out_dir / "index.html"
        with html_path.open("w") as f:
            f.write("<html><head><meta charset='utf-8'><title>Results Dashboard</title></head><body>\n")
            f.write("<h1>Results Dashboard</h1>\n")
            for img in generated:
                f.write(f"<div><img src='{img}' style='max-width: 100%; height: auto;'/></div>\n")
            f.write("</body></html>\n")

        print(f"Wrote {len(generated)} charts to {out_dir}")
        print(f"HTML dashboard: {html_path}")

    if args.all_instances:
        if not args.input_dir:
            raise ValueError("--all-instances requires --input-dir")
        input_dir = resolve_input_path(args.input_dir)
        candidates = [
            p for p in sorted(input_dir.iterdir(), key=lambda x: x.name)
            if p.is_dir() and (p / "results.json").exists()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No instance results.json found under {input_dir}")
        base_output_dir = Path(args.output_dir)
        for candidate in candidates:
            print(f"\n=== Visualizing instance: {candidate.name} ===")
            _render_one(candidate / "results.json", base_output_dir / candidate.name)
        return

    if args.input:
        selected_input = resolve_input_path(args.input)
    elif args.input_dir:
        input_dir = resolve_input_path(args.input_dir)
        selected_input = input_dir / "results.json"
    else:
        raise ValueError("Provide --input or --input-dir")

    _render_one(selected_input, Path(args.output_dir))


if __name__ == "__main__":
    main()

