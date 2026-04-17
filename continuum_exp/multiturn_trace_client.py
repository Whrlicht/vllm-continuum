#!/usr/bin/env python3
"""Multi-turn trace-driven chat client for vLLM OpenAI-compatible API.

Features:
- Configurable base_url/model_name.
- Two task dispatch modes:
  1) fixed concurrency
  2) Poisson arrivals with jps (jobs per second)
- Multi-turn replay per trajectory.
- Inter-round delay uses execution_time_seconds from trace messages.
- Next-round prompt is composed from previous prompt + previous output.
- Unified trace_replay toggle in request body.
- Saves all outputs to output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiohttp


@dataclass
class RoundSpec:
    assistant_round_index: int
    assistant_message_index: int
    execution_time_seconds: float


@dataclass
class TrajectorySample:
    traj_id: str
    instance_id: str
    messages: list[dict[str, Any]]
    rounds: list[RoundSpec]


@dataclass
class TextComparison:
    exact_match: bool
    normalized_match: bool
    first_diff_index: int
    expected_length: int
    actual_length: int


@dataclass
class ClientConfig:
    base_url: str
    model_name: str
    trace_path: str
    output_dir: str
    max_trajectories: int
    max_rounds_per_trajectory: int
    request_timeout_s: float
    trace_replay: bool
    dispatch_mode: str
    concurrency: int
    jps: float
    poisson_max_concurrency: int
    temperature: float
    top_p: float
    max_tokens_per_round: int
    job_id_field: str
    output_mode: str
    prefill_monitoring_path: str
    decode_monitoring_path: str
    enable_monitoring_enrich: bool


def parse_args() -> ClientConfig:
    parser = argparse.ArgumentParser(
        description="Replay multi-turn conversations from trace dataset")

    # Interfaces requested by user.
    parser.add_argument("--base-url",
                        default="http://localhost:10234/v1",
                        help="OpenAI-compatible base URL")
    parser.add_argument(
        "--model-name",
        default="/data/huggingface/models--meta-llama--Llama-3.1-8B-Instruct",
        help="Served model name/path")

    parser.add_argument(
        "--trace-path",
        default="trace_data/swe_bench_sample_100_with_timings.json",
        help="Path to trace dataset JSON")
    parser.add_argument("--output-dir",
                        default="output",
                        help="Directory to write output JSON")
    parser.add_argument("--max-trajectories",
                        type=int,
                        default=0,
                        help="0 means all trajectories")
    parser.add_argument("--max-rounds-per-trajectory",
                        type=int,
                        default=0,
                        help="0 means full rounds derived from trace")
    parser.add_argument("--request-timeout-s",
                        type=float,
                        default=300.0,
                        help="HTTP timeout per request")

    # Unified trace_replay switch.
    parser.add_argument("--trace-replay",
                        dest="trace_replay",
                        action="store_true",
                        help="Enable trace_replay=true in request body")
    parser.add_argument("--no-trace-replay",
                        dest="trace_replay",
                        action="store_false",
                        help="Disable trace_replay in request body")
    parser.set_defaults(trace_replay=True)

    # Distribution control.
    parser.add_argument("--dispatch-mode",
                        choices=["fixed", "poisson"],
                        default="fixed",
                        help="Task dispatch strategy")
    parser.add_argument("--concurrency",
                        type=int,
                        default=32,
                        help="Used in fixed mode")
    parser.add_argument("--jps",
                        type=float,
                        default=1.0,
                        help="Poisson jobs per second")
    parser.add_argument("--poisson-max-concurrency",
                        type=int,
                        default=64,
                        help="Concurrency cap in poisson mode")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-tokens-per-round",
        type=int,
        default=0,
        help=("Max generated tokens per round. "
              "Use 0 to let server decide."),
    )
    parser.add_argument(
        "--job-id-field",
        choices=["traj_id", "instance_id"],
        default="traj_id",
        help=("Which trace id to send as request job_id for scheduler/"
              "monitoring grouping."),
    )
    parser.add_argument(
        "--output-mode",
        choices=["concise", "full"],
        default="concise",
        help=("concise: smaller output json; full: include large raw fields "
              "(prompt_messages/raw_response/expected text)."),
    )
    parser.add_argument(
        "--prefill-monitoring-path",
        default="",
        help=("Optional prefill monitoring_timestamps path for enrichment. "
              "If empty, client tries auto-discovery."),
    )
    parser.add_argument(
        "--decode-monitoring-path",
        default="",
        help=("Optional decode monitoring_timestamps path for enrichment. "
              "If empty, client tries auto-discovery."),
    )
    parser.add_argument(
        "--disable-monitoring-enrich",
        action="store_true",
        help=("Disable post-run enrichment using monitoring_timestamps."),
    )

    ns = parser.parse_args()
    return ClientConfig(
        base_url=ns.base_url.rstrip("/"),
        model_name=ns.model_name,
        trace_path=ns.trace_path,
        output_dir=ns.output_dir,
        max_trajectories=ns.max_trajectories,
        max_rounds_per_trajectory=ns.max_rounds_per_trajectory,
        request_timeout_s=ns.request_timeout_s,
        trace_replay=ns.trace_replay,
        dispatch_mode=ns.dispatch_mode,
        concurrency=max(1, ns.concurrency),
        jps=max(1e-6, ns.jps),
        poisson_max_concurrency=max(1, ns.poisson_max_concurrency),
        temperature=ns.temperature,
        top_p=ns.top_p,
        max_tokens_per_round=max(0, ns.max_tokens_per_round),
        job_id_field=ns.job_id_field,
        output_mode=ns.output_mode,
        prefill_monitoring_path=ns.prefill_monitoring_path,
        decode_monitoring_path=ns.decode_monitoring_path,
        enable_monitoring_enrich=not ns.disable_monitoring_enrich,
    )


def _flatten_content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"),
                                                                str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item,
                                            ensure_ascii=False,
                                            sort_keys=True))
            else:
                parts.append(_flatten_content_to_text(item))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def parse_messages_field(messages_field: Any) -> list[dict[str, Any]]:
    if isinstance(messages_field, list):
        return [m for m in messages_field if isinstance(m, dict)]
    if isinstance(messages_field, str):
        try:
            parsed = json.loads(messages_field)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [m for m in parsed if isinstance(m, dict)]
    return []


def extract_round_specs(messages: list[dict[str, Any]]) -> list[RoundSpec]:
    rounds: list[RoundSpec] = []
    assistant_round_index = 0
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        exec_seconds = 0.0
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            call_times: list[float] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                raw = tc.get("execution_time_seconds")
                if isinstance(raw, (int, float)):
                    call_times.append(float(raw))
            if call_times:
                # Use max as conservative delay for this round.
                exec_seconds = max(call_times)
        rounds.append(
            RoundSpec(
                assistant_round_index=assistant_round_index,
                assistant_message_index=msg_idx,
                execution_time_seconds=max(0.0, exec_seconds),
            ))
        assistant_round_index += 1

    # Keep at least one round per trajectory.
    if not rounds:
        rounds = [
            RoundSpec(
                assistant_round_index=0,
                assistant_message_index=len(messages),
                execution_time_seconds=0.0,
            )
        ]
    return rounds


def normalize_chat_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = str(msg.get("role", "user"))
    out: dict[str, Any] = {"role": role}

    content = msg.get("content")
    if isinstance(content, (str, list)):
        out["content"] = content
    else:
        out["content"] = _flatten_content_to_text(content)

    # Keep prompts schema-minimal for compatibility across chat templates.
    return out


def build_round_prompt_messages(
    messages: list[dict[str, Any]],
    assistant_message_index: int,
) -> list[dict[str, Any]]:
    prefix = messages[:assistant_message_index]
    return [normalize_chat_message(m) for m in prefix]


def extract_expected_assistant_output(
    messages: list[dict[str, Any]],
    assistant_message_index: int,
) -> str:
    if assistant_message_index < 0 or assistant_message_index >= len(messages):
        return ""
    msg = messages[assistant_message_index]
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _flatten_content_to_text(content)
        if text:
            return text

    # Fallback for datasets where assistant text is split into other fields.
    parts: list[str] = []
    for key in ("thought", "action"):
        if key in msg:
            val = _flatten_content_to_text(msg.get(key))
            if val:
                parts.append(val)
    return "\n".join(parts)


def _normalize_for_compare(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def compare_text(expected: str, actual: str) -> TextComparison:
    exp_norm = _normalize_for_compare(expected)
    act_norm = _normalize_for_compare(actual)

    first_diff = -1
    limit = min(len(expected), len(actual))
    for i in range(limit):
        if expected[i] != actual[i]:
            first_diff = i
            break
    if first_diff == -1 and len(expected) != len(actual):
        first_diff = limit

    return TextComparison(
        exact_match=(expected == actual),
        normalized_match=(exp_norm == act_norm),
        first_diff_index=first_diff,
        expected_length=len(expected),
        actual_length=len(actual),
    )


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_proxy_timing_headers(headers: "aiohttp.typedefs.LooseHeaders") -> dict[str, float]:
    mapping = {
        "proxy_req_recv_ts": "X-Disagg-Proxy-Req-Recv-Ts",
        "proxy_prefill_start_ts": "X-Disagg-Proxy-Prefill-Start-Ts",
        "proxy_prefill_done_ts": "X-Disagg-Proxy-Prefill-Done-Ts",
        "proxy_decode_start_ts": "X-Disagg-Proxy-Decode-Start-Ts",
    }
    out: dict[str, float] = {}
    for key, header_name in mapping.items():
        parsed = _safe_float(headers.get(header_name))
        if parsed is not None:
            out[key] = parsed
    return out


def _annotate_client_inter_round_gaps(rounds: list[dict[str, Any]]) -> None:
    for i in range(len(rounds) - 1):
        current = rounds[i]
        nxt = rounds[i + 1]
        current_end = _safe_float(current.get("request_end_time"))
        next_start = _safe_float(nxt.get("request_start_time"))
        sleep_target = _safe_float(current.get("execution_time_seconds"))
        if current_end is None or next_start is None:
            continue
        inter_gap = next_start - current_end
        current["client_inter_round_gap_to_next_request_s"] = inter_gap
        current["client_inter_round_sleep_target_s"] = sleep_target
        if sleep_target is not None:
            current["client_inter_round_extra_over_sleep_s"] = (
                inter_gap - sleep_target)


def _auto_discover_monitoring_paths() -> tuple[Optional[Path], Optional[Path]]:
    repo_root = Path(__file__).resolve().parent.parent
    default_root = repo_root / (
        "examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/continuum_exp"
    )
    if not default_root.exists():
        return None, None

    prefill_candidates = sorted(
        p for p in default_root.glob("prefill_*/monitoring_timestamps") if p.is_file()
    )
    decode_candidates = sorted(
        p for p in default_root.glob("decode_*/monitoring_timestamps") if p.is_file()
    )
    prefill_path = prefill_candidates[0] if prefill_candidates else None
    decode_path = decode_candidates[0] if decode_candidates else None
    return prefill_path, decode_path


def _load_json_if_exists(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _build_proxy_request_index(monitoring: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta = monitoring.get("request_meta") or {}
    stats = monitoring.get("request_stats") or {}
    by_proxy: dict[str, dict[str, Any]] = {}

    for request_id, meta_obj in meta.items():
        if not isinstance(meta_obj, dict):
            continue
        proxy_request_id = meta_obj.get("proxy_request_id")
        if not proxy_request_id:
            continue
        stat_obj = stats.get(request_id)
        if not isinstance(stat_obj, dict):
            stat_obj = {}

        entry = {
            "request_id": request_id,
            "job_id": meta_obj.get("job_id"),
            "agent_round": meta_obj.get("agent_round"),
            "arrival_time": _safe_float(
                stat_obj.get("arrival_time", meta_obj.get("arrival_time"))),
            "finish_time": _safe_float(stat_obj.get("finish_time")),
            "first_token_ts": _safe_float(stat_obj.get("first_token_ts")),
            "last_token_ts": _safe_float(stat_obj.get("last_token_ts")),
        }

        existing = by_proxy.get(str(proxy_request_id))
        if existing is None:
            by_proxy[str(proxy_request_id)] = entry
            continue

        old_finish = _safe_float(existing.get("finish_time"))
        new_finish = _safe_float(entry.get("finish_time"))
        if new_finish is not None and (
                old_finish is None or new_finish >= old_finish):
            by_proxy[str(proxy_request_id)] = entry

    return by_proxy


def _enrich_results_with_monitoring(
    cfg: ClientConfig,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not cfg.enable_monitoring_enrich:
        return {"enabled": False, "reason": "disabled_by_flag"}

    prefill_path = Path(cfg.prefill_monitoring_path) \
        if cfg.prefill_monitoring_path else None
    decode_path = Path(cfg.decode_monitoring_path) \
        if cfg.decode_monitoring_path else None

    if prefill_path is None or decode_path is None:
        auto_prefill, auto_decode = _auto_discover_monitoring_paths()
        prefill_path = prefill_path or auto_prefill
        decode_path = decode_path or auto_decode

    prefill_monitoring = _load_json_if_exists(prefill_path)
    decode_monitoring = _load_json_if_exists(decode_path)

    if not prefill_monitoring or not decode_monitoring:
        return {
            "enabled": True,
            "enriched": False,
            "reason": "monitoring_files_missing_or_invalid",
            "prefill_monitoring_path": str(prefill_path) if prefill_path else "",
            "decode_monitoring_path": str(decode_path) if decode_path else "",
        }

    prefill_index = _build_proxy_request_index(prefill_monitoring)
    decode_index = _build_proxy_request_index(decode_monitoring)

    rounds_total = 0
    rounds_with_proxy_id = 0
    rounds_with_decode_finish = 0
    rounds_with_prefill_arrival = 0
    transition_with_server_times = 0

    for traj in results:
        rounds = traj.get("rounds") or []
        _annotate_client_inter_round_gaps(rounds)

        for round_obj in rounds:
            rounds_total += 1
            route = round_obj.get("disagg_route") or {}
            if not isinstance(route, dict):
                continue
            proxy_request_id = route.get("proxy_request_id")
            if not proxy_request_id:
                continue
            rounds_with_proxy_id += 1

            decode_info = decode_index.get(str(proxy_request_id))
            prefill_info = prefill_index.get(str(proxy_request_id))

            request_start = _safe_float(round_obj.get("request_start_time"))
            request_end = _safe_float(round_obj.get("request_end_time"))
            proxy_recv = _safe_float(route.get("proxy_req_recv_ts"))

            if decode_info is not None:
                round_obj["decode_server_request_id"] = decode_info.get("request_id")
                decode_finish = _safe_float(decode_info.get("finish_time"))
                if decode_finish is not None:
                    round_obj["decode_server_finish_time"] = decode_finish
                    rounds_with_decode_finish += 1
                    if request_end is not None:
                        round_obj["decode_departure_to_client_response_end_s"] = (
                            request_end - decode_finish)

            if prefill_info is not None:
                round_obj["prefill_server_request_id"] = prefill_info.get("request_id")
                prefill_arrival = _safe_float(prefill_info.get("arrival_time"))
                if prefill_arrival is not None:
                    round_obj["prefill_server_arrival_time"] = prefill_arrival
                    rounds_with_prefill_arrival += 1
                    if request_start is not None:
                        round_obj["client_send_to_prefill_arrival_s"] = (
                            prefill_arrival - request_start)
                    if proxy_recv is not None:
                        round_obj["proxy_receive_to_prefill_arrival_s"] = (
                            prefill_arrival - proxy_recv)

            if request_start is not None and proxy_recv is not None:
                round_obj["client_send_to_proxy_receive_s"] = (
                    proxy_recv - request_start)

        for i in range(len(rounds) - 1):
            current = rounds[i]
            nxt = rounds[i + 1]
            decode_finish = _safe_float(current.get("decode_server_finish_time"))
            next_prefill_arrival = _safe_float(nxt.get("prefill_server_arrival_time"))
            if decode_finish is None or next_prefill_arrival is None:
                continue

            transition_with_server_times += 1
            transition_gap = next_prefill_arrival - decode_finish
            current["decode_departure_to_next_prefill_arrival_server_s"] = transition_gap
            exec_s = _safe_float(current.get("execution_time_seconds"))
            if exec_s is not None:
                current["decode_to_next_prefill_arrival_minus_exec_s"] = (
                    transition_gap - exec_s)

    return {
        "enabled": True,
        "enriched": True,
        "prefill_monitoring_path": str(prefill_path) if prefill_path else "",
        "decode_monitoring_path": str(decode_path) if decode_path else "",
        "prefill_index_size": len(prefill_index),
        "decode_index_size": len(decode_index),
        "rounds_total": rounds_total,
        "rounds_with_proxy_request_id": rounds_with_proxy_id,
        "rounds_with_decode_finish": rounds_with_decode_finish,
        "rounds_with_prefill_arrival": rounds_with_prefill_arrival,
        "transitions_with_server_times": transition_with_server_times,
    }


def load_trajectories(path: str,
                      max_trajectories: int = 0) -> list[TrajectorySample]:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = raw if isinstance(raw, list) else [raw]
    out: list[TrajectorySample] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        traj_id = str(entry.get("traj_id") or entry.get("instance_id") or "")
        instance_id = str(entry.get("instance_id") or traj_id)
        if not traj_id:
            continue

        messages = parse_messages_field(entry.get("messages"))
        rounds = extract_round_specs(messages)

        out.append(
            TrajectorySample(
                traj_id=traj_id,
                instance_id=instance_id,
                messages=messages,
                rounds=rounds,
            ))

        if max_trajectories > 0 and len(out) >= max_trajectories:
            break

    return out


async def chat_once(
    cfg: ClientConfig,
    session: aiohttp.ClientSession,
    prompt_messages: list[dict[str, Any]],
    traj_id: str,
    instance_id: str,
    assistant_round_index: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": cfg.model_name,
        "messages": prompt_messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "stream": False,
    }
    if cfg.max_tokens_per_round > 0:
        payload["max_tokens"] = cfg.max_tokens_per_round
    if cfg.trace_replay:
        payload["trace_replay"] = True
        payload["traj_id"] = traj_id
        job_id = traj_id if cfg.job_id_field == "traj_id" else instance_id
        payload["vllm_xargs"] = {
            "trace_replay_assistant_round": int(assistant_round_index),
            "job_id": job_id,
        }

    url = f"{cfg.base_url}/chat/completions"
    try:
        async with session.post(url,
                                json=payload,
                                timeout=cfg.request_timeout_s) as resp:
            disagg_route = {
                "prefill_http": resp.headers.get("X-Disagg-Prefill-HTTP"),
                "decode_http": resp.headers.get("X-Disagg-Decode-HTTP"),
                "proxy_request_id": resp.headers.get("X-Disagg-Request-Id"),
                "route_token": resp.headers.get("X-Disagg-Route-Token"),
            }
            disagg_route.update(_extract_proxy_timing_headers(resp.headers))
            disagg_route = {
                key: value for key, value in disagg_route.items()
                if value not in (None, "")
            }
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}: {text}")
            if not text:
                return {"_disagg_route": disagg_route} if disagg_route else {}

            parsed = json.loads(text)
            if isinstance(parsed, dict) and disagg_route:
                parsed["_disagg_route"] = disagg_route
            return parsed
    except aiohttp.ClientError as e:
        raise RuntimeError(str(e)) from e


def extract_output_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""

    msg = first.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            return content

    text = first.get("text")
    return text if isinstance(text, str) else ""


async def run_single_trajectory(
    sample: TrajectorySample,
    cfg: ClientConfig,
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    start_ts = time.time()
    rounds = sample.rounds
    if cfg.max_rounds_per_trajectory > 0:
        rounds = rounds[:cfg.max_rounds_per_trajectory]
    per_round: list[dict[str, Any]] = []

    for i, round_spec in enumerate(rounds):
        prompt_messages = build_round_prompt_messages(
            sample.messages,
            round_spec.assistant_message_index,
        )

        req_start = time.time()
        ok = True
        err_msg = ""
        raw_response: dict[str, Any] = {}
        output_text = ""
        disagg_route: dict[str, Any] | None = None

        try:
            raw_response = await chat_once(
                cfg,
                session,
                prompt_messages,
                sample.traj_id,
                sample.instance_id,
                round_spec.assistant_round_index,
            )
            output_text = extract_output_text(raw_response)
            route_meta = raw_response.get("_disagg_route")
            if isinstance(route_meta, dict):
                disagg_route = route_meta
        except Exception as e:  # noqa: BLE001
            ok = False
            err_msg = str(e)

        req_end = time.time()
        expected_output = extract_expected_assistant_output(
            sample.messages,
            round_spec.assistant_message_index,
        )
        comparison = compare_text(expected_output, output_text)

        round_result = {
            "round_index": i,
            "request_start_time": req_start,
            "request_end_time": req_end,
            "request_latency_s": req_end - req_start,
            "execution_time_seconds": round_spec.execution_time_seconds,
            "assistant_round_index": round_spec.assistant_round_index,
            "assistant_message_index": round_spec.assistant_message_index,
            "response_id": (raw_response.get("id")
                             if isinstance(raw_response, dict) else None),
            "output": output_text,
            "comparison": asdict(comparison),
            "ok": ok,
            "error": err_msg,
        }
        if disagg_route:
            round_result["disagg_route"] = disagg_route

        if cfg.output_mode == "full":
            round_result["prompt_messages"] = prompt_messages
            round_result["expected_assistant_output"] = expected_output
            round_result["raw_response"] = raw_response
        else:
            # Concise mode: keep lightweight diagnostics only.
            round_result["prompt_message_count"] = len(prompt_messages)
            round_result["prompt_tail_roles"] = [
                m.get("role", "unknown") for m in prompt_messages[-6:]
            ]
            round_result["expected_output_length"] = len(expected_output)
            usage = raw_response.get("usage") if isinstance(raw_response,
                                                             dict) else None
            if isinstance(usage, dict):
                round_result["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }

        per_round.append(round_result)

        if not ok:
            break

        # Delay to next round based on trace execution_time_seconds.
        if i < len(rounds) - 1 and round_spec.execution_time_seconds > 0:
            sleep_start = time.time()
            await asyncio.sleep(round_spec.execution_time_seconds)
            sleep_end = time.time()
            round_result["sleep_start_time"] = sleep_start
            round_result["sleep_end_time"] = sleep_end
            round_result["actual_sleep_time_s"] = sleep_end - sleep_start
            round_result["sleep_overrun_s"] = (
                (sleep_end - sleep_start) - round_spec.execution_time_seconds)

    _annotate_client_inter_round_gaps(per_round)

    end_ts = time.time()
    return {
        "traj_id": sample.traj_id,
        "instance_id": sample.instance_id,
        "num_rounds_requested": len(rounds),
        "num_rounds_completed": len(per_round),
        "start_time": start_ts,
        "end_time": end_ts,
        "duration_s": end_ts - start_ts,
        "rounds": per_round,
        "success": all(r.get("ok", False) for r in per_round),
    }


async def run_fixed_dispatch(
    samples: list[TrajectorySample],
    cfg: ClientConfig,
    session: aiohttp.ClientSession,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(cfg.concurrency)
    results: list[dict[str, Any]] = []

    async def worker(sample: TrajectorySample) -> None:
        async with sem:
            result = await run_single_trajectory(sample, cfg, session)
            results.append(result)

    tasks = [asyncio.create_task(worker(s)) for s in samples]
    await asyncio.gather(*tasks)
    return results


async def run_poisson_dispatch(
    samples: list[TrajectorySample],
    cfg: ClientConfig,
    session: aiohttp.ClientSession,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(cfg.poisson_max_concurrency)
    launched: list[asyncio.Task] = []
    results: list[dict[str, Any]] = []

    async def guarded_run(sample: TrajectorySample) -> None:
        async with sem:
            result = await run_single_trajectory(sample, cfg, session)
            results.append(result)

    for i, sample in enumerate(samples):
        launched.append(asyncio.create_task(guarded_run(sample)))
        if i < len(samples) - 1:
            inter_arrival = random.expovariate(cfg.jps)
            await asyncio.sleep(inter_arrival)

    await asyncio.gather(*launched)
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_trajectories": 0,
            "num_success": 0,
            "num_failed": 0,
            "avg_duration_s": 0,
            "total_duration_s": 0,
        }

    durations = [float(r.get("duration_s", 0.0)) for r in results]
    num_success = sum(1 for r in results if r.get("success", False))
    return {
        "num_trajectories": len(results),
        "num_success": num_success,
        "num_failed": len(results) - num_success,
        "avg_duration_s": sum(durations) / len(durations),
        "total_duration_s": sum(durations),
    }


def dump_output(
    cfg: ClientConfig,
    results: list[dict[str, Any]],
    timing_enrichment_summary: Optional[dict[str, Any]] = None,
) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"multiturn_trace_client_{ts}.json"

    payload = {
        "config": asdict(cfg),
        "record_scope": "end_to_end_client",
        "record_note": (
            "Each round is one client-visible request to base_url. In "
            "disaggregated mode, per-round disagg_route (if present) records "
            "which prefill/decode node handled that request. Additional "
            "timing fields include client inter-round gaps and, when "
            "monitoring enrichment is available, decode-finish/client-end and "
            "client-send/prefill-arrival diagnostics."
        ),
        "summary": summarize(results),
        "timing_enrichment_summary": timing_enrichment_summary or {},
        "results": results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


async def async_main(cfg: ClientConfig) -> int:
    samples = load_trajectories(cfg.trace_path, cfg.max_trajectories)
    if not samples:
        print("No valid trajectories found in trace dataset")
        return 1

    print(f"Loaded {len(samples)} trajectories")
    print(f"Dispatch mode: {cfg.dispatch_mode}")

    t0 = time.time()
    # Avoid connector-level caps masking the intended client concurrency.
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector,
                                     timeout=timeout) as session:
        if cfg.dispatch_mode == "fixed":
            results = await run_fixed_dispatch(samples, cfg, session)
        else:
            results = await run_poisson_dispatch(samples, cfg, session)
    t1 = time.time()

    timing_enrichment_summary = _enrich_results_with_monitoring(cfg, results)

    out_path = dump_output(
        cfg,
        results,
        timing_enrichment_summary=timing_enrichment_summary,
    )
    summary = summarize(results)
    print("Run finished")
    print(f"Elapsed: {t1 - t0:.3f}s")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if timing_enrichment_summary:
        print("Timing enrichment summary:")
        print(json.dumps(timing_enrichment_summary, ensure_ascii=False, indent=2))
    print(f"Output saved to: {out_path}")
    return 0


def main() -> None:
    cfg = parse_args()
    code = asyncio.run(async_main(cfg))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
