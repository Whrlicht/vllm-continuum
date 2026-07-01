# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LICHTV3 decode-side coordinator (回传 REMOVED 2026-05-21 — predictors only).

This is the only object the scheduler imports.  It owns:
  - the ToolTimePredictorWrapper (tool-call time predictor)
  - the ShadowScheduler (K_queue stage1/2 predictor + step-time model;
    event-driven mirror of prefill's scheduler)

API exposed to the scheduler:
  - `on_round_finished(req, decode_finish_ts)`:
        invoked once per request that just completed a decode round.
        Runs the predictors (T_tool, K_queue, step-time) and writes the
        prediction record.  All non-fatal on error.  NO KV transfer.
  - `shutdown()`, `bind_connector()`, `drain_pending_releases()` (no-op),
    `should_retain_gpu_blocks()` (always False) — kept for scheduler parity.

Behaviour gating:
  - Construction is cheap; loading the tool predictor happens lazily on
    first use.
  - The whole pipeline is wrapped in a try/except so a bug in v3 cannot
    wedge the decode scheduler.  Any failure logs and returns silently.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from vllm.logger import init_logger

from .predictors import ToolTimePredictorWrapper
from .shadow_scheduler import ShadowScheduler
# 回传 REMOVED (2026-05-21): connector_bridge / prewarm / tier_planner /
# tier_storage / warm_pool deleted.  The dead orchestration methods that
# referenced them remain only as never-called shells (safe under
# `from __future__ import annotations`).

logger = init_logger(__name__)


@dataclass
class _PendingTool:
    """Tracks a request currently in (decode-finished, awaiting next
    prefill) state.  Used as ghost arrivals and to remember per-request
    state (tier handle, token sequence, target prefill address)
    until the prewarm fires."""
    request_id: str
    decode_finish_ts: float
    eta_ts: float
    agent_round_next: int
    num_prompt_tokens_estimate: int
    # 回传 REMOVED: tier_handle unused (kept as Any for dataclass parity).
    tier_handle: Optional[Any] = None
    token_ids: Optional[list[int]] = None
    prefill_address: Optional[str] = None
    # job_id of the request — used by D3 (arrival-based cancel) to
    # match a future round of the same agent conversation.
    job_id: Optional[str] = None
    # Whether _real_push has already consumed this entry.
    pushed: bool = False


class LichtV3DecodeManager:
    """Top-level v3 coordinator.  One instance per decode scheduler."""

    def __init__(self,
                 max_slots: int,
                 block_size: int,
                 tier_cfg: Optional[Any] = None,   # 回传 REMOVED: unused
                 tool_predictor_run_dir: Optional[str] = None,
                 default_t_tool_s: float = 5.0,
                 kv_cache_manager=None,
                 kv_caches_by_layer: Optional[dict] = None,
                 model_name_or_path: Optional[str] = None,
                 tokenizer_mode: str = "auto",
                 trust_remote_code: bool = False,
                 tokenizer_revision: Optional[str] = None):
        self.block_size = max(block_size, 1)
        # 回传 REMOVED (2026-05-21): no tier planner/config.
        # Tokenizer is needed only when a real predictor is loaded.
        self._model_name_or_path = model_name_or_path
        self._tokenizer_mode = tokenizer_mode
        self._trust_remote_code = trust_remote_code
        self._tokenizer_revision = tokenizer_revision
        tok_provider = self._build_tokenizer_provider() \
            if model_name_or_path else None
        if tool_predictor_run_dir:
            self.tool_predictor = ToolTimePredictorWrapper.from_run_dir(
                tool_predictor_run_dir,
                default_t_tool_s=default_t_tool_s,
                tokenizer_provider=tok_provider)
        else:
            self.tool_predictor = ToolTimePredictorWrapper.with_fallback(
                default_t_tool_s=default_t_tool_s)
        # 回传 REMOVED: no warm pool.
        # Resolved lazily on first push since the kv_caches_by_layer
        # dict is populated only after the connector's
        # register_kv_caches finishes (which runs after Scheduler init).
        self._kv_cache_manager = kv_cache_manager
        self._kv_caches_by_layer = kv_caches_by_layer or {}
        # Decode-side GPU retention: when a request's tier == "gpu", we
        # ask the scheduler (via callback) to skip the normal free path
        # so the KV blocks stay valid until our push consumes them.
        self._gpu_retained_req_ids: set[str] = set()
        # Bound after construction by the scheduler.
        self._release_retained_cb: Optional[Any] = None
        # Tier objects are constructed on first push when layer dict
        # becomes non-empty.  Until then `_stub_push` runs as fallback.
        self._tiers_initialised = False
        # Scheduler-side connector (role=SCHEDULER), bound by the scheduler.
        self._connector: Optional[Any] = None
        # 回传 REMOVED: no prewarm scheduler thread.
        # In-flight tool calls — used as ghost arrivals.
        self._pending_lock = threading.Lock()
        self._pending: dict[str, _PendingTool] = {}
        # Cross-thread release queue: the prewarm thread cannot safely
        # touch the scheduler's block_pool (concurrent mutation races
        # with scheduler.allocate_slots etc. cause linked-list
        # corruption and stalls).  Instead, when push completes for a
        # GPU-retained request, prewarm ENQUEUES the request_id here;
        # the scheduler thread drains this queue in `schedule()` and
        # runs `_v3_release_retained` itself.
        import queue as _queue_mod
        self._pending_releases: _queue_mod.SimpleQueue = \
            _queue_mod.SimpleQueue()
        # Per-job last-round info for measurement-based history observe.
        # When round N+1's on_round_finished fires, we look up round N's
        # tc + decode_finish_ts from here, compute measured actual_t,
        # and observe.  This makes the tracker's TrajectoryState use
        # REAL execution times in its E1-E5 history features, matching
        # how the predictor was trained.
        self._job_last_round: dict[str, tuple[dict, float]] = {}
        # Counters (for log only; promote to metrics later).
        self.counters: dict[str, int] = {
            "rounds_seen": 0,
            "tier_gpu": 0, "tier_cpu": 0, "tier_ssd": 0, "tier_drop": 0,
        }
        # LICHTV3 ShadowScheduler — event-driven mirror of prefill
        # state.  SOLE K_queue predictor (legacy
        # ProductionStepCountPredictor was removed once shadow hit 100%
        # accuracy in the trace_replay validation).  Enabled by env
        # LICHT_V3_USE_SHADOW_SCHED=1 (default in launch script).  If
        # disabled, on_round_finished falls back to k_queue=1, which is
        # the modal true value under low concurrency.
        self.shadow_scheduler: Optional[ShadowScheduler] = None
        if os.environ.get("LICHT_V3_USE_SHADOW_SCHED", "0") == "1":
            try:
                self.shadow_scheduler = ShadowScheduler(
                    max_slots=max_slots,
                    block_size=block_size,
                )
                logger.info(
                    "LICHTV3 ShadowScheduler enabled "
                    "(sub_addr=%s, pred_log=%s)",
                    self.shadow_scheduler._sub_addr or "(none)",
                    self.shadow_scheduler.pred_log_path)
            except Exception as e:
                logger.warning(
                    "LICHTV3 ShadowScheduler init failed: %s — "
                    "decode will keep using legacy predictor only", e)
                self.shadow_scheduler = None
        # Truncate prediction log so re-runs don't accumulate stale records.
        try:
            pred_log = os.environ.get(
                "LICHT_V3_PRED_LOG", self._PRED_LOG_PATH_DEFAULT)
            os.makedirs(os.path.dirname(pred_log), exist_ok=True)
            open(pred_log, "w").close()
            logger.info(
                "LICHTV3 prediction log truncated: %s", pred_log)
        except Exception as e:  # pragma: no cover
            logger.warning(
                "LICHTV3 prediction log truncate failed: %s", e)
        logger.info(
            "LICHTV3 decode manager initialised (predictors only, 回传 "
            "removed) (max_slots=%d, block_size=%d)",
            max_slots, block_size)

    # ------------------------------------------------------------------
    # Per-round prediction log writer (for offline merge w/ client JSON)
    # ------------------------------------------------------------------

    _PRED_LOG_PATH_DEFAULT = (
        "/data/whr/vllm-continuum/output/v3_predictions.jsonl")

    def _write_prediction_record(self, *, request, agent_round: int,
                                 num_prompt_tokens: int, num_blocks: int,
                                 k_queue: int,
                                 t_tool_full: Optional[dict],
                                 decode_finish_ts: float) -> None:
        """Append a JSONL line per round with all predictor outputs.
        Used by a downstream merge script that joins this file with the
        client's multiturn_trace_client.json (which has the per-round
        ground-truth execution_time_seconds)."""
        try:
            path = os.environ.get(
                "LICHT_V3_PRED_LOG", self._PRED_LOG_PATH_DEFAULT)
            traj_id = getattr(request, "traj_id", None)
            job_id = getattr(request, "job_id", None)
            req_id = getattr(request, "request_id", None)
            rec = {
                "ts": float(decode_finish_ts),
                "traj_id": str(traj_id) if traj_id is not None else None,
                "job_id":  str(job_id)  if job_id  is not None else None,
                "request_id": str(req_id) if req_id is not None else None,
                "agent_round": int(agent_round),
                "num_prompt_tokens_current": int(num_prompt_tokens),
                "next_round_num_blocks": int(num_blocks),
                "K_queue_pred": int(k_queue),
                "T_tool_p50":       (None if t_tool_full is None
                                     else float(t_tool_full["p50"])),
                "T_tool_p95":       (None if t_tool_full is None
                                     else float(t_tool_full["p95"])),
                "T_tool_p_timeout": (None if t_tool_full is None
                                     else float(t_tool_full["p_timeout"])),
                "T_tool_bucket":    (None if t_tool_full is None
                                     else t_tool_full.get("bucket")),
                "T_tool_family":    (None if t_tool_full is None
                                     else t_tool_full.get("family")),
                "T_tool_source":    (None if t_tool_full is None
                                     else t_tool_full.get("source")),
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            logger.debug("LICHTV3 _write_prediction_record failed: %s", e)

    # ------------------------------------------------------------------
    # Public hook from scheduler
    # ------------------------------------------------------------------

    def on_round_finished(self, request: Any,
                          decode_finish_ts: Optional[float] = None) -> None:
        """Called by the decode scheduler whenever a request reaches a
        round boundary (== request finish in vLLM terms).  Performs the
        full v3 pipeline in one shot.  Never raises.
        """
        try:
            self._on_round_finished_inner(request, decode_finish_ts)
        except Exception as e:  # pragma: no cover
            logger.warning("LICHTV3 on_round_finished error: %s", e)

    def _dmgr_prof(self, seg: str, dt: float, bump: bool = False) -> None:
        """DMGR-PROF (LICHT_STEP_PROFILE=1): 拆 on_round_finished 的四段
        (observe / predict / shadow / write), 每 LICHT_UPD_PROFILE_N(默认200)
        个 finish 汇总一次 → 定位那 ~140ms/finish 到底谁吃的。"""
        try:
            if os.environ.get("LICHT_STEP_PROFILE") != "1":
                return
            acc = getattr(self, "_dmgr_prof_buf", None)
            if acc is None:
                acc = self._dmgr_prof_buf = {}
                self._dmgr_prof_n = 0
            acc[seg] = acc.get(seg, 0.0) + dt
            if not bump:
                return
            self._dmgr_prof_n += 1
            if self._dmgr_prof_n < int(
                    os.environ.get("LICHT_UPD_PROFILE_N", "200")):
                return
            n = self._dmgr_prof_n
            parts = " ".join(
                f"{k}={v * 1000.0:.0f}ms(avg{v / n * 1000.0:.2f})"
                for k, v in sorted(acc.items()))
            logger.info("DMGR-PROF finishes=%d | per-seg total(per-finish "
                        "avg ms): %s", n, parts)
            self._dmgr_prof_buf = {}
            self._dmgr_prof_n = 0
        except Exception:  # pragma: no cover - profiling must never break
            pass

    def _on_round_finished_inner(self, request: Any,
                                 decode_finish_ts: Optional[float]
                                 ) -> None:
        if decode_finish_ts is None:
            decode_finish_ts = time.time()
        self.counters["rounds_seen"] += 1
        req_id = getattr(request, "request_id", None)
        if req_id is None:
            return
        # Heartbeat: log a one-line counter summary every 20 rounds so
        # the user can see v3 is alive without waiting for shutdown.
        if self.counters["rounds_seen"] % 20 == 0:
            logger.info(
                "LICHTV3 heartbeat: rounds_seen=%d "
                "tier_gpu=%d cpu=%d ssd=%d drop=%d "
                "skipped_no_tool=%d no_snapshot=%d cancelled=%d "
                "prewarm_fire_cancelled=%d "
                "watchdog_demoted=%d watchdog_dropped=%d "
                "push_success=%d push_fail=%d "
                "v3_pushback_enqueued=%d v3_pushback_armed=%d "
                "v3_offload_enqueued=%d "
                "v3_retention_released=%d v3_release_by_signal=%d",
                self.counters["rounds_seen"],
                self.counters.get("tier_gpu", 0),
                self.counters.get("tier_cpu", 0),
                self.counters.get("tier_ssd", 0),
                self.counters.get("tier_drop", 0),
                self.counters.get("skipped_no_tool", 0),
                self.counters.get("no_snapshot", 0),
                self.counters.get("cancelled_next_round_arrived", 0),
                self.counters.get("prewarm_fire_cancelled", 0),
                self.counters.get("watchdog_demoted", 0),
                self.counters.get("watchdog_dropped", 0),
                self.counters.get("push_success", 0),
                self.counters.get("push_fail", 0),
                self.counters.get("v3_pushback_enqueued", 0),
                self.counters.get("v3_pushback_armed", 0),
                self.counters.get("v3_offload_enqueued", 0),
                self.counters.get("v3_retention_released", 0),
                self.counters.get("v3_release_by_signal", 0))
        # D3: if any earlier round of the same agent conversation has a
        # pending v3 push, cancel it — the next round has already
        # arrived (we're seeing its decode finish), so the push window
        # has been missed.  Frees CPU/SSD storage and (rarely) GPU
        # retention from that earlier round.
        job_id_now = getattr(request, "job_id", None)
        self._cancel_pending_for_job(
            str(job_id_now) if job_id_now is not None else None,
            exclude_request_id=req_id)
        agent_round_curr = max(getattr(request, "agent_round", 0) or 0, 0)
        num_prompt_tokens = int(getattr(request, "num_tokens", 0)
                                or getattr(request, "num_prompt_tokens", 0)
                                or 0)
        # ---- Measurement-based observe of PREVIOUS round ----
        # Provide all three pieces of data that TrajectoryState.observe
        # consumed at training time:
        #   (a) tool_call dict of the PREVIOUS round (parsed earlier
        #       from its model output, saved in _job_last_round)
        #   (b) actual_t = wall-clock elapsed from PREVIOUS round's
        #       decode_finish to CURRENT round's pf_arrival.  This is
        #       the inter-round gap which equals (tool exec time +
        #       client→prefill HTTP/queue overhead), NOT including the
        #       current round's prefill+decode work.  Closer to training
        #       target than using dc_dep→dc_dep (which adds an extra
        #       T_prefill+T_decode of bias = 10-30s for typical
        #       requests).  Some ~5s engine_core queue bias remains but
        #       it is consistent across rounds so trajectory features
        #       are not relatively biased.
        #   (c) observation_text = the most recent <tool> body in
        #       THIS round's prompt (= the previous round's tool
        #       output), which TrajectoryState uses to update
        #       FileCache (C features) and E5 obs signals.
        # The combination makes TrajectoryState's state evolve in the
        # same way it did during training (`build_samples`'s observe
        # loop), so E1-E5 + C features are accurate.
        job_id_str = (str(job_id_now) if job_id_now is not None else None)
        if job_id_str:
            prev = self._job_last_round.pop(job_id_str, None)
            if prev is not None:
                prev_tc, prev_decode_finish_ts = prev
                # Prefer request.arrival_time (set by scheduler at
                # request_arrives() = pf_arrival of CURRENT round).
                # Falls back to decode_finish_ts if unset (unusual).
                cur_pf_arrival = float(
                    getattr(request, "arrival_time", None)
                    or decode_finish_ts)
                measured_t = max(cur_pf_arrival - prev_decode_finish_ts,
                                 0.0)
                # Pull obs text from the just-arrived round's prompt.
                _t_obs = time.perf_counter()
                obs_text = ""
                try:
                    from .features_adapter import extract_observation_text
                    prompt_tail = self.tool_predictor.decode_prompt_tail(
                        request)
                    obs_text = extract_observation_text(prompt_tail)
                except Exception as e:  # pragma: no cover
                    logger.debug(
                        "LICHTV3 obs_text extraction failed: %s", e)
                self.tool_predictor.observe_for_job(
                    job_id_str, prev_tc, measured_t,
                    observation_text=obs_text)
                self._dmgr_prof("observe", time.perf_counter() - _t_obs)
        # ---- Extract THIS round's tc once; reuse for predict + save ----
        _t_ex = time.perf_counter()
        cur_tc = self.tool_predictor.extract_tool_call_for_request(request)
        # extract_tc = detokenize 输出 + 解析 tool_call(怀疑慢在 tokenizer.decode)
        _t_pf = time.perf_counter()
        self._dmgr_prof("extract_tc", _t_pf - _t_ex)
        # Step 1: predict T_tool.  Returns dict with p50/p95/p_timeout,
        # or None when no tool call detected (trajectory ended).
        t_tool_full = self.tool_predictor.predict_full_for_request(
            request, tc=cur_tc)
        # predict_full = feature_row(建特征) + pd.DataFrame + ML predict_df。
        # bump=True: 此处所有 finish 都会到(在 SKIP 判断之前)→ 每 finish 计一次。
        self._dmgr_prof("predict_full", time.perf_counter() - _t_pf, bump=True)
        if t_tool_full is None:
            t_tool_s = None
        else:
            t_tool_s = float(t_tool_full["p50"])
        if t_tool_s is None:
            # Make this case observable.  Includes diagnostics:
            #   - n_out_tokens:    # of decoded output tokens (if 0,
            #                      detokenize path is starving)
            #   - traj_id / agent_round: trace_replay context
            #   - preview:         last 200 chars of decoded output
            self.counters["skipped_no_tool"] = (
                self.counters.get("skipped_no_tool", 0) + 1)
            try:
                n_out = len(getattr(request, "output_token_ids", []) or [])
            except Exception:
                n_out = -1
            traj_id = getattr(request, "traj_id", None)
            trace_on = getattr(request, "trace_replay_enabled", False)
            preview = ""
            if self.counters["skipped_no_tool"] <= 5:
                try:
                    preview = self.tool_predictor._decode_output(
                        request)[-200:].replace("\n", "\\n")
                except Exception:
                    preview = ""
            logger.info(
                "LICHTV3 SKIP req=%s reason=no_tool_call "
                "n_out_tokens=%d trace=%s traj_id=%s round=%d "
                "preview='%s'",
                req_id, n_out, trace_on,
                (str(traj_id)[:40] if traj_id else "None"),
                agent_round_curr, preview)
            return  # treat as "no next round expected"
        # Remember this round's tc + decode_finish_ts so the next
        # round's on_round_finished can observe with measured actual_t.
        if cur_tc is not None and job_id_str:
            self._job_last_round[job_id_str] = (cur_tc, decode_finish_ts)
        # Step 2: estimate next-round prompt size (used by tier-planner
        # for blocks-to-retain).
        eta_ts = decode_finish_ts + t_tool_s
        try:
            n_output = len(getattr(request, "output_token_ids", []) or [])
        except Exception:
            n_output = 0
        # tool result tokens estimate — fallback constant; ShadowScheduler
        # maintains a per-bucket rolling median internally but we don't
        # query it here (its purpose is to refine its OWN simulation).
        EST_TOOL_RESULT_TOKENS = int(
            os.environ.get("LICHT_V3_EST_TOOL_RESULT_TOKENS", "200"))
        next_prompt_tokens = (num_prompt_tokens + n_output
                              + EST_TOOL_RESULT_TOKENS)
        # Step 3: predict K_queue via ShadowScheduler (sole predictor).
        # Returns step count (1 = admitted in very next schedule call,
        # which is the floor under low concurrency).  When shadow is
        # disabled or unable to predict, fall back to k_queue=1 — the
        # most common true value, so the fallback is conservative-ish.
        k_queue = 1
        # t_step_end_s: predicted wall-clock end of the CURRENT prefill
        # step (= step start + step-time prediction).  Feeds tier Gate 1
        # (arrival vs current-step-end).  0.0 = unavailable → Gate 1
        # fails → KV sinks to CPU/SSD (conservative).
        t_step_end_s = 0.0
        if self.shadow_scheduler is not None:
            _t_shadow = time.perf_counter()
            try:
                traj_id_for_shadow = (
                    str(getattr(request, "traj_id", None)
                        or getattr(request, "job_id", None)
                        or req_id))
                bucket = (str(t_tool_full.get("bucket", "unknown"))
                          if t_tool_full else "unknown")
                t_tool_p95 = (float(t_tool_full.get("p95", 0.0))
                              if t_tool_full else 0.0)
                shadow_k = self.shadow_scheduler.on_decode_finish(
                    traj_id=traj_id_for_shadow,
                    agent_round_curr=agent_round_curr,
                    decode_finish_ts=decode_finish_ts,
                    curr_prompt_tokens=int(num_prompt_tokens),
                    n_output=int(n_output),
                    t_tool_p50_s=float(t_tool_s),
                    bucket=bucket,
                    t_tool_p95_s=t_tool_p95,
                )
                if shadow_k is not None:
                    k_queue = int(shadow_k)
                _tse = self.shadow_scheduler.predicted_step_end_ts()
                if _tse is not None:
                    t_step_end_s = float(_tse)
                self._dmgr_prof("shadow", time.perf_counter() - _t_shadow)
            except Exception as e:
                logger.debug("ShadowScheduler.on_decode_finish err: %s "
                             "(falling back to k_queue=%d)", e, k_queue)
        # Step 5: tier-plan THIS request alone (batch=1).  Real batch
        # planning across simultaneously finishing requests is a future
        # improvement; per-request planning is conservative.
        # NOTE: blocks-to-retain reflects what we want to push to prefill
        # for the NEXT round (which sees next_prompt_tokens), NOT the
        # current round.  Using the bigger number keeps capacity decisions
        # conservative.
        num_blocks = (next_prompt_tokens + self.block_size - 1) // self.block_size
        # ----------------------------------------------------------------
        # Write a per-round prediction JSONL record so post-process can
        # merge against the client's multiturn_trace_client.json (which
        # has `execution_time_seconds` = the trace's ground truth).
        # File lives in $LICHT_V3_PRED_LOG or default /data/whr/vllm-continuum/output/v3_predictions.jsonl
        # ----------------------------------------------------------------
        _t_wr = time.perf_counter()
        self._write_prediction_record(
            request=request, agent_round=agent_round_curr,
            num_prompt_tokens=num_prompt_tokens,
            num_blocks=num_blocks, k_queue=k_queue,
            t_tool_full=t_tool_full, decode_finish_ts=decode_finish_ts)
        self._dmgr_prof("write", time.perf_counter() - _t_wr)
        # ====================================================================
        # 回传 REMOVED (2026-05-21, user request): LICHT-V3 is now LICHT-V2 +
        # the three predictors (prediction OUTPUTS only).  decode no longer
        # offloads KV to CPU/SSD, no tier decision, no arm, no push-back.
        # Everything above (tool/K_queue/step预测 + prediction JSON) stays;
        # the KV simply frees normally after the prefill→decode handoff, as
        # in plain disaggregated serving.
        # ====================================================================
        return

    # ------------------------------------------------------------------
    # D3 arrival-based cancel: same job_id next round → cancel pending
    # ------------------------------------------------------------------

    def _cancel_pending_for_job(self, job_id: Optional[str],
                                exclude_request_id: str) -> None:
        if not job_id:
            return
        cancellable_ids: list[str] = []
        with self._pending_lock:
            for rid, p in self._pending.items():
                if rid == exclude_request_id:
                    continue
                if p.job_id == job_id and not p.pushed:
                    cancellable_ids.append(rid)
        for rid in cancellable_ids:
            self._cancel_pending(rid, reason="next-round-arrived")

    def _cancel_pending(self, request_id: str, reason: str) -> None:
        # Predictor bookkeeping only: drop the in-flight tool entry.  回传
        # cleanup (prewarm/tier/warm-pool/retention) REMOVED 2026-05-21.
        with self._pending_lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return
        self.counters[f"cancelled_{reason.replace('-','_')}"] = (
            self.counters.get(
                f"cancelled_{reason.replace('-','_')}", 0) + 1)

    def _build_tokenizer_provider(self):
        """Return a zero-arg callable that returns the model's
        tokenizer (loaded lazily and cached process-globally)."""
        model = self._model_name_or_path
        mode = self._tokenizer_mode
        trust = self._trust_remote_code
        rev = self._tokenizer_revision

        def _provider():
            from .features_adapter import get_tokenizer
            return get_tokenizer(model, mode, trust, rev)

        return _provider

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        # 回传 REMOVED: no prewarm thread to shut down.
        if self.shadow_scheduler is not None:
            try:
                self.shadow_scheduler.shutdown()
            except Exception as e:
                logger.debug("shadow shutdown err: %s", e)
        logger.info(
            "LICHTV3 decode manager shutdown. Counters: %s", self.counters)

    # ------------------------------------------------------------------
    # Scheduler integration for GPU-tier retention
    # ------------------------------------------------------------------

    def bind_release_retained_cb(self, cb) -> None:
        """Scheduler passes a callback that frees blocks for a
        previously-retained request.  Stored here but ONLY called by
        the scheduler thread itself via `drain_pending_releases()` —
        never directly from the prewarm thread (would race with
        scheduler block_pool ops)."""
        self._release_retained_cb = cb

    def bind_connector(self, connector) -> None:
        """Scheduler passes its SCHEDULER-role connector (kept for parity;
        回传 removed so it's no longer used to enqueue push-backs)."""
        self._connector = connector

    def drain_pending_releases(self) -> None:
        """回传 REMOVED (2026-05-21): no GPU-tier retention to release.
        Kept as a no-op since the scheduler still calls it each step."""
        return

    def should_retain_gpu_blocks(self, request_id: str) -> bool:
        # 回传 REMOVED: never retain (normal delay-free path).
        return False
