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
from matplotlib.patches import Patch


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


def _percentile_sorted(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] + (k - f) * (values[c] - values[f])


def _summarize_distribution(values: List[float]) -> Dict[str, Any]:
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

    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "mean": sum(sorted_values) / len(sorted_values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p50": _percentile_sorted(sorted_values, 50),
        "p95": _percentile_sorted(sorted_values, 95),
        "p99": _percentile_sorted(sorted_values, 99),
    }


def _derive_prefill_exec_ttft_stats(
        results: Dict[str, Any], instance_role: str) -> Dict[str, Any]:
    monitoring = results.get("monitoring_metrics") or {}
    default_stats = monitoring.get("ttft_seconds", {}) if isinstance(
        monitoring, dict) else {}

    if instance_role != "prefill":
        return default_stats

    round_records = results.get("round_records") or []
    prefill_durations: List[float] = []
    for record in round_records:
        if not isinstance(record, dict):
            continue
        value = record.get("prefill_duration")
        if isinstance(value, (int, float)) and value >= 0:
            prefill_durations.append(float(value))

    if prefill_durations:
        return _summarize_distribution(prefill_durations)
    return default_stats


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


def _plot_decode_to_next_prefill_start_gantt(
        breakdown: Dict[str, Any], out_path: Path) -> bool:
    """Plot four-stage latency decomposition as three Gantt-like bars."""
    snapshots = breakdown.get("gantt_snapshots_seconds")
    if not isinstance(snapshots, dict):
        return False

    transitions_complete = int(breakdown.get("transitions_complete") or 0)
    if transitions_complete <= 0:
        scanned = int(breakdown.get("transitions_scanned") or 0)
        matched = int(
            breakdown.get("transitions_with_prefill_round_match") or 0)
        stage4_queue = int(
            breakdown.get("transitions_with_stage4_queue") or 0)
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.axis("off")
        ax.text(
            0.5,
            0.62,
            "No complete decode->next-prefill transitions found.",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.40,
            ("scanned={scanned}, matched={matched}, "
             "stage4_queue={stage4_queue}, complete={transitions_complete}").format(
                 scanned=scanned,
                 matched=matched,
                 stage4_queue=stage4_queue,
                 transitions_complete=transitions_complete,
            ),
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.text(
            0.5,
            0.20,
            "Rerun analyze.py after client output is generated and ensure it matches this prefill/decode run.",
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
        )
        _save_fig(fig, out_path)
        return True

    cases = [("mean", "Average"), ("p50", "P50"), ("p95", "P95")]
    stage_defs = [
        ("decode_to_client_response_end_s", "decode->client_end"),
        ("client_tool_sleep_s", "client_sleep(tool)"),
        ("client_send_to_proxy_receive_s", "client->proxy"),
        ("proxy_receive_to_prefill_run_start_s", "proxy->prefill_start"),
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    rows: List[Tuple[str, List[float], float]] = []
    for key, label in cases:
        row = snapshots.get(key)
        if not isinstance(row, dict):
            continue
        values = [max(0.0, float(row.get(stage_key, 0) or 0))
                  for stage_key, _ in stage_defs]
        total = max(0.0, float(
            row.get("total_decode_to_next_prefill_start_s", sum(values)) or 0))
        rows.append((label, values, total))

    if not rows:
        return False

    max_total = max((row[2] for row in rows), default=0.0)
    if max_total <= 0:
        max_total = 1.0

    fig, axes = plt.subplots(len(rows), 1,
                             figsize=(10, 1.9 * len(rows) + 1.0),
                             sharex=True)
    if len(rows) == 1:
        axes = [axes]

    for ax, (label, values, total) in zip(axes, rows):
        left = 0.0
        tiny_text_y = [0.34, 0.46, 0.58, 0.70]
        for idx, value in enumerate(values):
            ax.barh([0], [value], left=[left], height=0.5,
                    color=colors[idx], edgecolor="white")
            if value >= max_total * 0.05:
                ax.text(left + value / 2, 0, f"{value:.3f}s",
                        ha="center", va="center", fontsize=8)
            elif value > 0:
                anchor_x = left + value
                ax.text(anchor_x, tiny_text_y[idx], f"{value:.4f}s",
                        ha="left", va="bottom", fontsize=7,
                        color=colors[idx], clip_on=False)
            left += value

        stage_summary = " | ".join(
            f"{stage_defs[i][1]}={values[i]:.4f}s" for i in range(len(stage_defs)))
        ax.text(
            0.01,
            0.88,
            stage_summary,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
        )

        ax.set_yticks([0])
        ax.set_yticklabels([label])
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_xlim(0, max_total * 1.12)
        ax.text(min(total, max_total * 1.05), 0.32,
                f"total={total:.3f}s", fontsize=8, ha="right")

    axes[-1].set_xlabel("seconds")
    fig.suptitle("Decode End -> Next Prefill Start: 4-Stage Breakdown", y=0.985)

    handles = [Patch(facecolor=colors[idx], label=label)
               for idx, (_, label) in enumerate(stage_defs)]
    fig.legend(handles=handles,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.955),
               ncol=2,
               frameon=False)
    fig.subplots_adjust(top=0.80, hspace=0.35)
    _save_fig(fig, out_path)
    return True


def _plot_queue_time_by_next_round(
        queue_stats: List[Dict[str, Any]], out_path: Path) -> bool:
    """Plot next-round index vs stage-4 queue time (mean/p50/p95)."""
    xs: List[int] = []
    ys_mean: List[float] = []
    ys_p50: List[float] = []
    ys_p95: List[float] = []

    for row in queue_stats:
        if not isinstance(row, dict):
            continue
        round_idx = row.get("next_round_idx")
        if not isinstance(round_idx, (int, float)):
            continue
        xs.append(int(round_idx))
        ys_mean.append(float(row.get("mean", 0) or 0))
        ys_p50.append(float(row.get("p50", 0) or 0))
        ys_p95.append(float(row.get("p95", 0) or 0))

    if not xs:
        return False

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys_mean, marker="o", color="#4C78A8", label="mean")
    ax.plot(xs, ys_p50, marker="s", color="#F58518", label="p50")
    ax.plot(xs, ys_p95, marker="^", color="#E45756", label="p95")
    ax.set_title("Next Round Index vs Stage-4 Queue Time")
    ax.set_xlabel("next round index")
    ax.set_ylabel("seconds")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left")
    _save_fig(fig, out_path)
    return True


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
        instance_role = str(results.get("instance_role") or "").lower()
        transition_breakdown = results.get("decode_to_next_prefill_start_breakdown") or {}
        ttft_stats = _derive_prefill_exec_ttft_stats(results, instance_role)

        generated = []

        # Latency p50/p95/p99
        if monitoring:
            _bar_p50_p95_p99("Prefill Execution (No Queue, s)", ttft_stats, out_dir / "latency_ttft.png")
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

        if instance_role == "prefill" and isinstance(transition_breakdown, dict):
            gantt_name = "decode_to_next_prefill_start_gantt.png"
            if _plot_decode_to_next_prefill_start_gantt(
                    transition_breakdown, out_dir / gantt_name):
                generated.append(gantt_name)

            queue_name = "round_vs_stage4_queue_time.png"
            queue_stats = transition_breakdown.get(
                "queue_time_by_next_round_seconds") or []
            if _plot_queue_time_by_next_round(queue_stats, out_dir / queue_name):
                generated.append(queue_name)
        elif instance_role == "prefill":
            # Avoid showing stale charts from previous runs when breakdown is absent.
            for stale_name in (
                    "decode_to_next_prefill_start_gantt.png",
                    "round_vs_stage4_queue_time.png"):
                stale_path = out_dir / stale_name
                if stale_path.exists():
                    stale_path.unlink()

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

