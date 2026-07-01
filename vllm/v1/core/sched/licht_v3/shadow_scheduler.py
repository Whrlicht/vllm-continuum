# SPDX-License-Identifier: Apache-2.0
"""ShadowScheduler — decode-side mirror of prefill's scheduler.

Goal
----
Predict per-request K_queue (= scheduler steps between arrival in
prefill waiting and admission to running) in the SAME UNIT prefill
actually measures, with continuous correction as prefill state evolves.

Replaces the per-decode-finish single-pass admit-position approach in
`ProductionStepCountPredictor`.  Differences:

1. Persistent state — single instance, updated incrementally by prefill
   StepEvent messages.
2. Step-driven admit simulator — actually walks step-by-step, models
   running queue progress, releases blocks when running reqs finish.
3. Sees future arrivals — own `pending_arrivals` for in-flight tool
   calls + prefill's `waiting_now` snapshot for everything else.
4. Two-stage correction — Stage 1 estimates at decode-finish, Stage 2
   replaces estimates with real values once prefill receives the
   request and reports it in StepEvent.waiting_now / admitted.

Env vars
--------
* LICHT_V3_USE_SHADOW_SCHED=1     enable
* LICHT_V3_STEP_EVENT_SUB_ADDR    ZMQ sub address (e.g. tcp://10.0.0.1:5559)
* LICHT_V3_SHADOW_PRED_LOG        path for prediction/correction log
* LICHT_V3_DEFAULT_TOOL_RESULT_TOKENS  fallback when bucket median missing
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

from vllm.logger import init_logger

from .online_step_time_predictor import OnlineStepTimePredictor
from .step_event import ReqSnapshot, StepEvent, decode_step_event

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RunningState:
    """A request mirrored from prefill's RUNNING queue."""
    request_id: str
    num_tokens: int            # total prompt tokens
    admit_step: int
    r_total: int               # full chunks needed (prefill phase only)
    r_remaining: int           # prefill chunks left.  0 ⇒ DECODE phase
                                # (no more chunks, and no scheduler-predictable
                                # release: blocks held until EOS, which we
                                # can't forecast — treat as static).
    blocks_needed: int         # ceil(num_tokens / block_size)
    traj_id: Optional[str] = None
    agent_round: Optional[int] = None
    in_decode_phase: bool = False   # True if r_remaining was 0 at mirror time


@dataclass
class WaitingState:
    """A request mirrored from prefill's WAITING queue."""
    request_id: str
    num_tokens: int
    arrival_step: int
    blocks_needed: int
    traj_id: Optional[str] = None
    agent_round: Optional[int] = None
    # Prefix-cache hit on prefill (P5): tokens already cached (incl. KV
    # pushed back by回传) → fewer chunks to compute.  evictable_prefix =
    # the subset of hit blocks in the free queue (consumed at admit).
    # Both come straight from the prefill StepEvent's ReqSnapshot.
    hit_length: int = 0
    evictable_prefix: int = 0


@dataclass
class Pending:
    """A tool call registered locally by on_decode_finish, not yet seen
    by prefill (its next-round request hasn't reached prefill yet)."""
    traj_id: str
    agent_round_next: int      # K+1 — the round whose K_queue we predict
    decode_finish_ts: float
    # Stage-1 estimates
    est_num_tokens: int        # next-round prompt size estimate
    est_eta_step: int          # estimated arrival step on prefill
    est_tool_time_s: float     # T_tool estimate that fed eta
    bucket: str                # T1 bucket — for per-bucket result-token stats
    curr_prompt_tokens: int    # this round's prompt size (constant)
    n_output: int              # this round's assistant output tokens
    # T_tool predictions — p50 drives est_eta_step, p95 is the deadline
    # past which we treat this pending as OVERDUE (likely long-tail
    # mispredict) and stop letting it influence other targets' Stage 1
    # competition.  The pending itself is kept until its own Stage 2
    # fires (when prefill confirms arrival via StepEvent).
    t_tool_p50_s: float = 0.0
    t_tool_p95_s: float = 0.0
    # Stage-1 prediction (initial)
    pred_k_queue_stage1: Optional[int] = None
    # Stage-2 corrections (filled when prefill actually sees the request)
    real_num_tokens: Optional[int] = None
    real_arrival_step: Optional[int] = None
    real_tool_time_s: Optional[float] = None
    pred_k_queue_stage2: Optional[int] = None
    # Final ground truth (filled on admit)
    real_admit_step: Optional[int] = None
    actual_k_queue: Optional[int] = None
    # Bookkeeping
    request_id: Optional[str] = None    # filled at Stage 2
    # 回传 REMOVED (2026-05-21): no push-back fields on the pending.


# ---------------------------------------------------------------------------
# ShadowScheduler
# ---------------------------------------------------------------------------

class ShadowScheduler:
    """Decode-side mirror of prefill's scheduler + step-driven K_queue
    simulator.  Thread-safe (state mutated only by ZMQ subscriber thread
    and by on_decode_finish; lock guards both).
    """

    _DEFAULT_PRED_LOG = (
        "/data/whr/vllm-continuum/output/v3_shadow_predictions.jsonl")

    def __init__(self,
                 max_slots: int = 32,
                 block_size: int = 16,
                 total_kv_blocks: int = 16853,
                 sub_addr: Optional[str] = None,
                 pred_log_path: Optional[str] = None):
        # State (guarded by self._lock)
        self._lock = threading.RLock()
        self.sim_step = 0
        self.step_wall_ts = time.time()
        self.sec_per_step = 0.05      # bootstrap; replaced by StepEvent
        # Real mirrors
        self.sim_running: dict[str, RunningState] = {}
        self.sim_waiting_real: dict[str, WaitingState] = {}
        # My pendings (registered by decode)
        self.pendings: dict[tuple[str, int], Pending] = {}
        # 回传 REMOVED (2026-05-21): no push-back wiring/registry.  The shadow
        # is now PURELY a predictor (K_queue stage1/2 + step-time), no
        # transfer.
        # Constants — updated from StepEvent on first message
        self.max_slots = max_slots
        self.block_size = max(block_size, 1)
        self.total_kv_blocks = max(total_kv_blocks, 1)
        # LICHTV2 timeline + constants — mirrored from each StepEvent.
        # The simulator left-shifts these per simulated step and fills
        # the tail with total_kv_blocks ("future unknown, assume max").
        self.prefill_future_free: list[int] = []
        self.prefill_future_alloc: list[int] = []
        self.lichtv2_n: int = 50
        self.chunk_size_tokens: int = 5242
        self.max_alloc_per_step_blocks: int = 0
        self.long_tail_headroom_blocks: int = 0
        self.long_running_count: int = 0
        self.max_long_bridge: int = 2
        self.score_a: float = 3.0
        self.score_b: float = 1.0
        self.score_tmax_s: float = 120.0
        self.round_decay_alpha: float = 0.5
        # Bucket stats for tool-result-tokens estimate (Stage 1 input)
        self.bucket_result_tokens: dict[str, list[int]] = {}
        self.default_result_tokens = int(os.environ.get(
            "LICHT_V3_DEFAULT_TOOL_RESULT_TOKENS", "200"))
        # Step-time predictor (RLS + multiplicative correction, see
        # online_step_time_predictor.py).  Features per step: 2-tuple of
        # (num_scheduled_tokens, num_running).  Cold start: predict()
        # returns None until first observe() lands.
        self.step_time_model = OnlineStepTimePredictor()
        # The PREVIOUS StepEvent's (features, emit_ts, prediction).
        # When the NEXT StepEvent arrives, its emit_ts - prev_emit_ts
        # gives the actual duration of the step whose features were
        # prev_features.
        self._prev_step_for_time: Optional[dict] = None
        self._step_time_log_path = os.environ.get(
            "LICHT_V3_STEP_TIME_LOG",
            "/data/whr/vllm-continuum/output/v3_step_time.jsonl")
        try:
            os.makedirs(os.path.dirname(self._step_time_log_path),
                        exist_ok=True)
            open(self._step_time_log_path, "w").close()
        except Exception as e:
            logger.warning("step_time log truncate failed: %s", e)
        # Logging
        self.pred_log_path = (pred_log_path
                              or os.environ.get("LICHT_V3_SHADOW_PRED_LOG",
                                                self._DEFAULT_PRED_LOG))
        try:
            os.makedirs(os.path.dirname(self.pred_log_path), exist_ok=True)
            open(self.pred_log_path, "w").close()
        except Exception as e:
            logger.warning("shadow pred log truncate failed: %s", e)
        # ZMQ sub thread
        self._sub_addr = sub_addr or os.environ.get(
            "LICHT_V3_STEP_EVENT_SUB_ADDR", "")
        self._stop = threading.Event()
        self._sub_thread: Optional[threading.Thread] = None
        if self._sub_addr:
            self._start_sub_thread()
        else:
            logger.info("ShadowScheduler: no sub addr — running in "
                        "register-only mode (no prefill state mirror).")
        # Background timer thread: every window_s, wake up and call
        # _maybe_resim so predictions get refreshed even during prefill
        # idle periods (no StepEvent stream to drive resim otherwise).
        self._timer_thread: Optional[threading.Thread] = None
        # Counters
        self.counters = {
            "step_events": 0,
            "register": 0,
            "stage2_match": 0,
            "stage2_no_match": 0,
            "predictions_written": 0,
            "resim_runs": 0,
            "resim_skipped_window": 0,
        }
        # Rate-limited resim: events mark `dirty`; resim only fires when
        # at least `_resim_window_s` has passed since the last resim.
        # The NEW pending's own prediction in on_decode_finish runs
        # immediately (bypasses the window) since the caller needs the
        # return value.
        self._resim_window_s = float(os.environ.get(
            "LICHT_V3_RESIM_WINDOW_S", "0.500"))   # 500ms default
        self._needs_resim = False
        self._last_resim_ts = 0.0
        # Tracks stage2 match count at last resim — used to force a resim
        # whenever a Stage-2 correction happens, so corrections show up
        # in pred_k_queue_stage2 even when the window says "skip".
        self._stage2_match_at_last_resim = 0
        # Start background timer for window-based resim during idle prefill.
        self._start_window_timer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start_sub_thread(self) -> None:
        def _run():
            try:
                import zmq
            except ImportError:
                logger.warning("ShadowScheduler: zmq not available; sub "
                               "thread disabled")
                return
            ctx = zmq.Context()
            sock = ctx.socket(zmq.SUB)
            sock.connect(self._sub_addr)
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.setsockopt(zmq.RCVTIMEO, 200)
            logger.info("ShadowScheduler: subscribed to StepEvent at %s",
                        self._sub_addr)
            while not self._stop.is_set():
                try:
                    buf = sock.recv()
                except zmq.Again:
                    continue
                except Exception as e:
                    logger.warning("StepEvent recv error: %s", e)
                    continue
                try:
                    evt = decode_step_event(buf)
                    self.on_step_event(evt)
                except Exception as e:
                    logger.warning("StepEvent decode/apply error: %s", e)
            try:
                sock.close()
                ctx.term()
            except Exception:
                pass
        self._sub_thread = threading.Thread(
            target=_run, daemon=True, name="LichtV3-ShadowSub")
        self._sub_thread.start()

    def _start_window_timer(self) -> None:
        """Background thread: every _resim_window_s, call _maybe_resim
        with dirty mark.  Ensures predictions stay fresh even when no
        StepEvent / decode-finish arrives for a while."""
        def _run():
            while not self._stop.is_set():
                if self._stop.wait(self._resim_window_s):
                    break
                try:
                    with self._lock:
                        if self.pendings:
                            self._mark_dirty_for_resim()
                            self._maybe_resim(force=False)
                except Exception as e:
                    logger.debug("window timer resim err: %s", e)
        self._timer_thread = threading.Thread(
            target=_run, daemon=True, name="LichtV3-ShadowTimer")
        self._timer_thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._sub_thread is not None:
            self._sub_thread.join(timeout=2.0)
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Event entry points
    # ------------------------------------------------------------------

    def on_step_event(self, evt: StepEvent) -> None:
        with self._lock:
            self.counters["step_events"] += 1
            self.sim_step = evt.step_id
            self.step_wall_ts = evt.step_wall_ts
            if evt.sec_per_step_recent > 0:
                self.sec_per_step = evt.sec_per_step_recent
            if evt.max_num_seqs > 0:
                self.max_slots = evt.max_num_seqs
            if evt.block_size > 0:
                self.block_size = evt.block_size
            if evt.total_kv_blocks > 0:
                self.total_kv_blocks = evt.total_kv_blocks
            # LICHTV2 timeline + constants — overwritten each StepEvent.
            if evt.future_free:
                self.prefill_future_free = list(evt.future_free)
                self.prefill_future_alloc = list(evt.future_alloc)
            if evt.lichtv2_horizon_n > 0:
                self.lichtv2_n = evt.lichtv2_horizon_n
            if evt.chunk_size_tokens > 0:
                self.chunk_size_tokens = evt.chunk_size_tokens
            if evt.max_alloc_per_step_blocks > 0:
                self.max_alloc_per_step_blocks = evt.max_alloc_per_step_blocks
            self.long_tail_headroom_blocks = evt.long_tail_headroom_blocks
            self.long_running_count = evt.long_running_count
            if evt.max_long_bridge > 0:
                self.max_long_bridge = evt.max_long_bridge
            self.score_a = evt.score_a
            self.score_b = evt.score_b
            self.score_tmax_s = evt.score_tmax_s
            self.round_decay_alpha = evt.round_decay_alpha

            # 0) Step-time observe-then-predict cycle.
            #    When this StepEvent (step N) arrives, the duration of the
            #    PREVIOUS step (step N-1) is known: this_emit_ts - prev_emit_ts.
            #    Its tokens were `prev.num_scheduled_tokens_this_step`.
            #    Verify the prediction we made on prev's arrival, then
            #    feed the observation to the model, then make a fresh
            #    prediction for step N (whose tokens are this event's).
            self._handle_step_time(evt)

            # 1) Overwrite real mirrors from authoritative snapshot
            self._refresh_real_mirrors(evt)

            # 2) Stage-2 correction: match prefill's waiting/admitted
            #    against my pendings.  Earlier matches (waiting_now) are
            #    preferred over later (admitted) so we get the longest
            #    correction window.
            self._stage2_match(evt)

            # 3) Re-simulate BEFORE finalize so pred_k_queue_stage2 is
            #    captured on pendings that admit in the same StepEvent
            #    where their Stage-2 correction happened.
            #
            #    Force the resim (bypass window) when:
            #      - a Stage 2 correction just happened this event, OR
            #      - prefill admitted / finished / preempted something
            #        this event (real state changed → other in-flight
            #        predictions may need update).
            #    Otherwise (empty step), normal window logic applies.
            had_stage2_match = (
                self.counters["stage2_match"]
                > self._stage2_match_at_last_resim)
            state_changed = bool(
                evt.admitted or evt.finished or evt.preempted)
            # Compute keys about to be finalized — skip them in resim so
            # their pred_k_queue_stage2 is preserved from the previous
            # resim (which doesn't have the off-by-one of "advance one
            # step past the actual admit").
            admit_keys_to_skip = {
                (r.traj_id, r.agent_round)
                for r in evt.admitted
                if r.traj_id is not None and r.agent_round is not None
            }
            self._mark_dirty_for_resim()
            self._maybe_resim(force=had_stage2_match or state_changed,
                              skip_keys=admit_keys_to_skip)

            # 4) Finalize predictions for any pending that just admitted
            self._finalize_on_admit(evt)
            # 回传 REMOVED (2026-05-21): no firing here — shadow only predicts.

    def on_decode_finish(self,
                         traj_id: str,
                         agent_round_curr: int,
                         decode_finish_ts: float,
                         curr_prompt_tokens: int,
                         n_output: int,
                         t_tool_p50_s: float,
                         bucket: str,
                         t_tool_p95_s: float = 0.0,
                         ) -> Optional[int]:
        """Stage 1: register a pending arrival, estimate, predict.

        Returns the initial K_queue prediction (in scheduler steps) or
        None if the simulator cannot produce a meaningful answer.
        """
        with self._lock:
            self.counters["register"] += 1
            agent_round_next = agent_round_curr + 1
            key = (traj_id, agent_round_next)
            # Estimate next-round prompt tokens.
            est_result_tokens = self._estimate_result_tokens(bucket)
            est_num_tokens = (curr_prompt_tokens + n_output
                              + est_result_tokens)
            # NOTE: we no longer convert T_tool seconds to a sim_step
            # estimate (sec_per_step is a crude rolling average and the
            # conversion introduces a long-tail error source).  Instead
            # the simulator injects every non-overdue, non-long-tool
            # pending as an immediate competitor at sim_step.  Long-tool
            # pendings (T_tool_p50 > threshold) are filtered out
            # separately inside the simulator.  est_eta_step is kept on
            # the Pending struct only as a debug field.
            sec_until_arrival = max(t_tool_p50_s, 0.0)
            est_eta_step = self.sim_step + 1
            # p95 defaults to 2× p50 if caller didn't pass one (back-compat).
            p95 = (t_tool_p95_s
                   if t_tool_p95_s > 0 else max(2 * t_tool_p50_s, 1.0))
            pending = Pending(
                traj_id=traj_id,
                agent_round_next=agent_round_next,
                decode_finish_ts=decode_finish_ts,
                est_num_tokens=est_num_tokens,
                est_eta_step=est_eta_step,
                est_tool_time_s=sec_until_arrival,
                bucket=bucket,
                curr_prompt_tokens=curr_prompt_tokens,
                n_output=n_output,
                t_tool_p50_s=float(t_tool_p50_s),
                t_tool_p95_s=float(p95),
            )
            self.pendings[key] = pending
            # ① Always immediate: predict for the new pending so caller
            #    (tier_planner) gets a return value right now.
            try:
                k = self._simulate_k_queue(pending)
                pending.pred_k_queue_stage1 = k
            except Exception as e:
                logger.warning("Stage1 simulation failed: %s", e)
                k = None
            # ② Mark dirty so other in-flight pendings get re-simulated
            #    later (within the resim window).
            self._mark_dirty_for_resim()
            return k

    # ------------------------------------------------------------------
    # Internal: step-time observe/predict
    # ------------------------------------------------------------------

    def _handle_step_time(self, evt: StepEvent) -> None:
        """Observe the duration of the just-completed step (if we have
        a prediction outstanding for it), then make a fresh prediction
        for the step that just started.

        Timing model
        ------------
        StepEvent is emitted at the END of schedule() (before worker
        execute).  So between two StepEvents N and N+1 the wall time
        covers: worker.execute(N) + post-process + input-queue drain +
        schedule(N+1).  This entire interval = "step N's duration"
        from the model's point of view, and the tokens that produced
        it are `num_scheduled_tokens` from StepEvent N.
        """
        prev = self._prev_step_for_time
        if prev is not None:
            actual_duration = max(evt.step_wall_ts - prev["emit_ts"], 1e-6)
            prev_features = prev["features"]
            predicted = prev["pred"]
            predicted_base = prev["pred_base"]
            # Update model with the observed (features, duration) pair.
            self.step_time_model.observe(prev_features, actual_duration)
            # Snapshot model state AFTER observation for logging.
            st = self.step_time_model.stats()
            try:
                with open(self._step_time_log_path, "a") as f:
                    f.write(json.dumps({
                        "ts": evt.step_wall_ts,
                        "step_id": prev["step_id"],
                        "features": prev_features,
                        "predicted_s": predicted,
                        "predicted_base_s": predicted_base,
                        "actual_s": actual_duration,
                        "abs_err_s": (None if predicted is None
                                       else abs(predicted - actual_duration)),
                        "model": {
                            "n_observed": st["n_observed"],
                            "weights_scaled": st["weights_scaled"],
                            "correction": st["correction"],
                        },
                    }) + "\n")
            except Exception:
                pass
        # Make a fresh prediction for THIS step.  Build the features
        # dict from the just-arrived StepEvent.  Its duration will be
        # measured when the next StepEvent arrives.
        cur_features = {
            "num_scheduled_tokens":
                int(evt.num_scheduled_tokens_this_step),
            "num_running": len(evt.running_now),
        }
        pred = self.step_time_model.predict(cur_features)
        pred_base = self.step_time_model.predict_base(cur_features)
        self._prev_step_for_time = {
            "emit_ts": evt.step_wall_ts,
            "step_id": evt.step_id,
            "features": cur_features,
            "pred": pred,
            "pred_base": pred_base,
        }

    def predicted_step_end_ts(self) -> Optional[float]:
        """Predicted wall-clock END of the current prefill step =
        latest StepEvent emit_ts (real step start anchor) + the
        step-time predictor's duration estimate for this step.

        Used by the decode-side tier decision (Gate 1: does a returning
        request arrive before this step ends?).  Returns None during
        cold start (no step-time prediction yet)."""
        prev = self._prev_step_for_time
        if prev is None:
            return None
        emit = prev.get("emit_ts")
        pred = prev.get("pred")
        if emit is None or pred is None:
            return None
        return float(emit) + float(pred)

    # ------------------------------------------------------------------
    # Internal: mirror refresh + Stage-2 correction
    # ------------------------------------------------------------------

    def _refresh_real_mirrors(self, evt: StepEvent) -> None:
        """Mirror prefill state and classify each running req's phase.

        Matches LICHTV2 backfill semantics:
          - prefill phase (r_remaining > 0): chunked alloc per step,
            release at t=R (delay-free).  These reqs are dynamic in the
            simulator's timeline.
          - decode phase (r_remaining == 0): blocks held STATICALLY for
            the entire lookahead horizon.  LICHTV2's timeline doesn't
            predict decode-phase EOS either; both real LICHTV2 and our
            simulator are intentionally conservative here.
        """
        new_running: dict[str, RunningState] = {}
        for r in evt.running_now:
            nt = r.num_prompt_tokens
            blocks = max(
                (nt + self.block_size - 1) // self.block_size, 1)
            r_total = self._r_total(nt, r.hit_length)
            admit_step = (r.admit_step
                          if r.admit_step is not None
                          else self.sim_step)
            # Prefer prefill-supplied r_remaining (exact, from
            # _licht_v2_R_at on real num_computed_tokens).  Fallback
            # to a coarse derivation from admit_step.
            if r.r_remaining is not None:
                prefill_rem = int(r.r_remaining)
            else:
                prefill_rem = max(
                    r_total - (self.sim_step - admit_step), 0)
            in_decode = (prefill_rem == 0
                         and r.request_id not in evt.finished)
            new_running[r.request_id] = RunningState(
                request_id=r.request_id,
                num_tokens=nt,
                admit_step=admit_step,
                r_total=r_total,
                r_remaining=prefill_rem,
                blocks_needed=blocks,
                traj_id=r.traj_id,
                agent_round=r.agent_round,
                in_decode_phase=in_decode,
            )
        self.sim_running = new_running

        new_waiting: dict[str, WaitingState] = {}
        for w in evt.waiting_now:
            nt = w.num_prompt_tokens
            blocks = max(
                (nt + self.block_size - 1) // self.block_size, 1)
            arr = (w.arrival_step
                   if w.arrival_step is not None else self.sim_step)
            new_waiting[w.request_id] = WaitingState(
                request_id=w.request_id,
                num_tokens=nt,
                arrival_step=arr,
                blocks_needed=blocks,
                traj_id=w.traj_id,
                agent_round=w.agent_round,
                hit_length=int(getattr(w, "hit_length", 0) or 0),
                evictable_prefix=int(getattr(w, "evictable_prefix", 0) or 0),
            )
        self.sim_waiting_real = new_waiting

    def _stage2_match(self, evt: StepEvent) -> None:
        """When prefill reports a request in waiting_now or admitted
        that matches one of our pendings (by traj_id + agent_round),
        replace estimates with real values.

        Only the first match wins per pending — subsequent occurrences
        in later StepEvents are ignored once Stage 2 has run.
        """
        for batch, source in ((evt.waiting_now, "waiting"),
                              (evt.admitted, "admitted")):
            for r in batch:
                if r.traj_id is None or r.agent_round is None:
                    continue
                key = (r.traj_id, r.agent_round)
                p = self.pendings.get(key)
                if p is None or p.real_num_tokens is not None:
                    continue  # not ours, or already corrected
                self.counters["stage2_match"] += 1
                p.real_num_tokens = r.num_prompt_tokens
                p.real_arrival_step = (r.arrival_step
                                       if r.arrival_step is not None
                                       else self.sim_step)
                p.real_tool_time_s = max(
                    evt.step_wall_ts - p.decode_finish_ts, 0.0)
                p.request_id = r.request_id
                # _stage2_match is purely PREDICTION correction (K_queue) +
                # bucket stats (回传 removed 2026-05-21).
                # Update bucket stat: actual tool_result_tokens
                actual_result_tokens = max(
                    p.real_num_tokens - p.curr_prompt_tokens - p.n_output,
                    0)
                self._update_bucket_result_tokens(
                    p.bucket, actual_result_tokens)

    def _finalize_on_admit(self, evt: StepEvent) -> None:
        """When a pending's request is in evt.admitted, compute the
        ground-truth K_queue and write the prediction record."""
        for r in evt.admitted:
            if r.traj_id is None or r.agent_round is None:
                continue
            key = (r.traj_id, r.agent_round)
            p = self.pendings.get(key)
            if p is None:
                continue  # not one of ours (e.g. round 0)
            arrival = (p.real_arrival_step
                       if p.real_arrival_step is not None
                       else (r.arrival_step
                             if r.arrival_step is not None
                             else self.sim_step))
            admit = self.sim_step
            actual_kq = max(admit - arrival, 0)
            p.real_admit_step = admit
            p.actual_k_queue = actual_kq
            self._write_prediction_record(p)
            del self.pendings[key]

    def _resim_all_in_flight(
            self,
            skip_keys: Optional[set] = None) -> None:
        """Re-simulate all in-flight pendings.

        `skip_keys` — set of (traj_id, agent_round) to NOT resim, used
        when a pending is about to be finalized in the same StepEvent
        (its simulator would advance one extra step past the actual
        admit and overstate K_queue by 1).  The previous resim's value
        for that pending is left intact and used by finalize_on_admit.
        """
        for p in self.pendings.values():
            if p.actual_k_queue is not None:
                continue
            if (skip_keys is not None
                    and (p.traj_id, p.agent_round_next) in skip_keys):
                continue
            try:
                k = self._simulate_k_queue(p)
                if p.real_num_tokens is None:
                    # No Stage-2 yet → keep updating Stage 1 prediction
                    p.pred_k_queue_stage1 = k
                else:
                    p.pred_k_queue_stage2 = k
            except Exception as e:
                logger.debug("re-sim failed for %s/%d: %s",
                             p.traj_id, p.agent_round_next, e)

    def _mark_dirty_for_resim(self) -> None:
        self._needs_resim = True

    def _maybe_resim(self, *, force: bool = False,
                     skip_keys: Optional[set] = None) -> None:
        """Rate-limited re-simulation of all in-flight pendings.

        Fires only when:
          - There's a pending dirty marker AND the resim window has
            elapsed since the last run, OR
          - `force=True` (caller bypasses window — used when a Stage-2
            correction MUST be reflected in pred_k_queue_stage2 of the
            current StepEvent before finalize-on-admit runs).
        `skip_keys` is forwarded to `_resim_all_in_flight`.
        """
        if not self._needs_resim and not force:
            return
        now = time.monotonic()
        if (not force) and (now - self._last_resim_ts
                            < self._resim_window_s):
            self.counters["resim_skipped_window"] += 1
            return
        self._resim_all_in_flight(skip_keys=skip_keys)
        self._needs_resim = False
        self._last_resim_ts = now
        self._stage2_match_at_last_resim = self.counters["stage2_match"]
        self.counters["resim_runs"] += 1

    # ------------------------------------------------------------------
    # Internal: step-driven admit simulator
    # ------------------------------------------------------------------

    def _simulate_k_queue(self, target_pending: Pending,
                          max_sim_steps: int = 1) -> int:
        """Single-round binary admit predictor for `target_pending`.

        Returns:
          1 — predicted to admit in the very next prefill scheduler step
          2 — predicted NOT to admit in the next prefill step
              (field name kept for backward-compat with existing
              pred_k_queue_stage1 / stage2 / actual_k_queue analysis,
              but interpretation is now binary: 1 = True, 2 = False)

        Periodic resim (every 500ms / every StepEvent) refreshes this
        binary prediction.  False → True transition naturally signals
        "ready to admit imminently".  No multi-round / step-count
        guessing — which was the source of stage>1 inaccuracy.

        Mechanism (mirrors prefill's LICHTV2 scheduling for ONE step):
          1. Shift prefill timeline left by 1 (= "next step's timeline").
          2. Advance sim_running by one chunk.
          3. Build candidate list = real_waiting + target + filtered
             in-flight pendings.
          4. LICHT score sort, try _lv2_can_admit, _lv2_apply_to_timeline.
          5. If target admitted in this round → return 1; else 2.

        Filters (user-confirmed design):
          - Stage-2 corrected pendings: skip (already in mirror).
          - T_tool_p50 > LICHT_V3_LONG_TOOL_THRESHOLD_S (default 5s):
            skip (long tool, won't compete soon).
          - Overdue (wall_clock - decode_finish_ts > p95): skip.
        """
        _ = max_sim_steps  # kept for signature compat; ignored
        N = self.lichtv2_n
        block_size = max(self.block_size, 1)
        chunk = max(self.chunk_size_tokens, 1)
        max_alloc = max(self.max_alloc_per_step_blocks, 1)
        headroom = self.long_tail_headroom_blocks
        long_bridge = max(self.max_long_bridge, 1)
        target_key = (target_pending.traj_id,
                      target_pending.agent_round_next)
        long_tool_threshold_s = float(os.environ.get(
            "LICHT_V3_LONG_TOOL_THRESHOLD_S", "5.0"))

        # ----- 1) Initial state copies -----
        sim_ff = list(self.prefill_future_free) or [
            self.total_kv_blocks] * (N + 1)
        sim_fa = list(self.prefill_future_alloc) or [0] * (N + 1)
        sim_running = [
            _running_copy(r) for r in self.sim_running.values()
            if (r.traj_id, r.agent_round) != target_key
        ]
        sim_waiting: list[_SimWaiting] = []
        target_real_hit = 0
        target_real_ev = 0
        for w in self.sim_waiting_real.values():
            if (w.traj_id, w.agent_round) == target_key:
                # Target has already arrived → remember prefill's REAL reported
                # prefix hit (system prefix + blocks not yet LRU-evicted).
                # Used below for stage2 instead of a pre-arrival estimate.
                target_real_hit = int(getattr(w, "hit_length", 0) or 0)
                target_real_ev = int(getattr(w, "evictable_prefix", 0) or 0)
                continue
            sim_waiting.append(_SimWaiting(
                request_id=w.request_id, num_tokens=w.num_tokens,
                arrival_step=w.arrival_step,
                blocks_needed=w.blocks_needed,
                traj_id=w.traj_id, agent_round=w.agent_round,
                # P5: real prefix hit reported by prefill (ground truth,
                # reflects whatever the prefix cache holds incl.回传'd KV).
                hit=w.hit_length, evictable=w.evictable_prefix))
        long_count = self.long_running_count

        # ----- 2) Target placement -----
        target_arrival_step: Optional[int] = None
        target_already_in_waiting = (
            target_pending.real_arrival_step is not None
            and target_pending.real_arrival_step <= self.sim_step
        )
        num_tok_t = (target_pending.real_num_tokens
                     if target_pending.real_num_tokens is not None
                     else target_pending.est_num_tokens)
        # 回传 REMOVED (2026-05-21): no push-back → a returning request does NOT
        # hit its full prior-round prefix (the old code assumed it did →
        # over-predicted quick-admits → K_queue regression).  Prefix model:
        #   stage2 (target ALREADY arrived): use prefill's REAL reported hit —
        #     ground truth, includes the shared system prefix AND any prior
        #     blocks not yet LRU-evicted.
        #   stage1 (pre-arrival): the prefix cache state is invisible to the
        #     shadow, so give no conversational-prefix credit (measured median
        #     real hit ≈ system prefix only, ~80 tok / <10%).
        if target_already_in_waiting:
            hit_t = min(target_real_hit, max(num_tok_t - 1, 0))
            ev_t = target_real_ev
        else:
            hit_t = 0
            ev_t = 0
        if target_already_in_waiting:
            target_arrival_step = int(target_pending.real_arrival_step)
            sim_waiting.append(_SimWaiting(
                request_id=f"target:{target_key[0]}:{target_key[1]}",
                num_tokens=num_tok_t,
                arrival_step=target_arrival_step,
                blocks_needed=max(
                    (num_tok_t + block_size - 1) // block_size, 1),
                traj_id=target_key[0], agent_round=target_key[1],
                _is_target=True, hit=hit_t, evictable=ev_t))
        else:
            # Stage 1: target arrives "between sim_step and sim_step+1",
            # i.e., arrival_step = sim_step.  Inject as immediate
            # competitor (we don't model when along the tool execution
            # it actually arrives — that conversion was a mistake).
            target_arrival_step = self.sim_step
            sim_waiting.append(_SimWaiting(
                request_id=f"target:{target_key[0]}:{target_key[1]}",
                num_tokens=num_tok_t,
                arrival_step=target_arrival_step,
                blocks_needed=max(
                    (num_tok_t + block_size - 1) // block_size, 1),
                traj_id=target_key[0], agent_round=target_key[1],
                _is_target=True, hit=hit_t, evictable=ev_t))

        # ----- 3) Other in-flight pendings as competition -----
        # Precise arrival check (replaces the old crude "T_tool > 5s →
        # exclude" filter): a pending becomes a waiting competitor in the
        # NEXT prefill step iff its predicted arrival time falls at or
        # before the END of the CURRENT real step.
        #
        #   t_step_end = real_step_start + predicted_step_duration
        #              = latest StepEvent emit_ts (option A, ground-truth
        #                anchor) + step-time predictor's estimate for this
        #                step.
        #   t_arrive   = decode_finish_ts + T_tool_p50  (send latency is
        #                empirically negligible → ignored)
        #
        # Anchoring to the REAL step start each StepEvent (NOT accumulating
        # predicted durations across steps) bounds the error to a single
        # step's prediction (~0.3s) instead of compounding.  Long-tool
        # requests are no longer dropped wholesale — they are re-checked
        # every StepEvent and injected only once they are about to arrive.
        now_wall = time.time()
        t_step_end = None
        prev_st = self._prev_step_for_time
        if prev_st is not None:
            _emit = prev_st.get("emit_ts")
            _pred = prev_st.get("pred")
            if _emit is not None and _pred is not None:
                t_step_end = _emit + _pred
        for p in self.pendings.values():
            if p is target_pending:
                continue
            if p.actual_k_queue is not None:
                continue
            if p.real_arrival_step is not None:
                continue  # Stage-2 corrected → already in mirror
            # ⑤ Overdue / timeout: past the p95 deadline and still not
            #    arrived → treat as long-tail/stuck, don't compete.
            if (p.t_tool_p95_s > 0
                    and now_wall - p.decode_finish_ts > p.t_tool_p95_s):
                continue
            # ④ Precise arrival: will it have arrived by this step's end?
            if t_step_end is not None:
                t_arrive = p.decode_finish_ts + p.t_tool_p50_s
                if t_arrive > t_step_end:
                    continue  # not arriving by this step's end → not yet a competitor
            elif p.t_tool_p50_s > long_tool_threshold_s:
                # Cold start (no step-time prediction yet): fall back to
                # the old crude long-tool threshold.
                continue
            # 回传 REMOVED: no full-prefix credit (see target above).
            hit_p = 0
            ev_p = 0
            sim_waiting.append(_SimWaiting(
                request_id=f"pending:{p.traj_id}:{p.agent_round_next}",
                num_tokens=p.est_num_tokens,
                arrival_step=self.sim_step,
                blocks_needed=max(
                    (p.est_num_tokens + block_size - 1) // block_size, 1),
                traj_id=p.traj_id, agent_round=p.agent_round_next,
                _is_target=False, hit=hit_p, evictable=ev_p))

        # ----- 4) SINGLE-round simulation -----
        #         Binary semantic: predict whether target admits in the
        #         very next prefill step.  Return 1 = True (admit next),
        #         2 = False (not admit next).  Periodic re-sim refreshes
        #         this prediction every window (500ms / StepEvent), so
        #         the False → True transition naturally captures
        #         "imminent admit" once it becomes true.
        sim_step_cursor = self.sim_step + 1

        # (a) Shift timeline left; fill tail with max blocks.
        sim_ff = sim_ff[1:] + [self.total_kv_blocks]
        sim_fa = sim_fa[1:] + [0]

        # (b) Advance running.  Prefill-phase reqs progress one chunk;
        #     decode-phase reqs stay static.  Releases reflected by
        #     dropping reqs whose r_remaining went below 0 — the
        #     prefill-side timeline already encoded the matching
        #     +release event at t=R.
        still_running: list[_SimRunning] = []
        for r in sim_running:
            if r.in_decode_phase:
                still_running.append(r)
                continue
            r.r_remaining -= 1
            if r.r_remaining < 0:
                continue  # released
            still_running.append(r)
        sim_running = still_running
        long_count = sum(1 for r in sim_running if r.r_remaining > N)

        # (c) Admit pass — strict LICHTV2 order + 3 Guards.
        if sim_waiting:
            while sim_waiting and len(sim_running) < self.max_slots:
                # LICHT score: higher = higher priority.
                t_wall_now = sim_step_cursor * self.sec_per_step
                def _score(w: _SimWaiting) -> float:
                    arr_t = w.arrival_step * self.sec_per_step
                    K = max(w.agent_round or 0, 0)
                    return _lv2_score(K, max(t_wall_now - arr_t, 0.0),
                                       self.score_a, self.score_b,
                                       self.score_tmax_s,
                                       self.round_decay_alpha)
                sim_waiting.sort(key=_score, reverse=True)
                pick_idx = -1
                for i, w in enumerate(sim_waiting):
                    ok = _lv2_can_admit(
                        candidate_num_tokens=w.num_tokens,
                        candidate_hit=w.hit,
                        candidate_evictable_prefix=w.evictable,
                        future_free=sim_ff,
                        future_alloc=sim_fa,
                        N=N, chunk_size=chunk, block_size=block_size,
                        max_alloc_per_step=max_alloc,
                        long_tail_headroom=headroom,
                        long_count=long_count,
                        max_long_bridge=long_bridge)
                    if ok:
                        pick_idx = i
                        break
                if pick_idx < 0:
                    break
                w = sim_waiting.pop(pick_idx)
                _lv2_apply_to_timeline(
                    candidate_num_tokens=w.num_tokens,
                    candidate_hit=w.hit,
                    candidate_evictable_prefix=w.evictable,
                    future_free=sim_ff,
                    future_alloc=sim_fa,
                    N=N, chunk_size=chunk, block_size=block_size)
                if w._is_target:
                    return 1   # binary True: target admits next step
                r_total = _lv2_R_at(w.num_tokens, w.hit, chunk)
                if r_total > N:
                    long_count += 1
                sim_running.append(_SimRunning(
                    request_id=w.request_id,
                    num_tokens=w.num_tokens,
                    admit_step=sim_step_cursor,
                    r_total=r_total,
                    r_remaining=r_total,
                    blocks_needed=max(
                        (w.num_tokens + block_size - 1) // block_size, 1),
                ))

        # Target was not admitted in this round → binary False.
        return 2

    def _r_total(self, num_tokens: int, hit_length: int) -> int:
        """Chunked-prefill chunks needed.  Uses the prefill-synced
        chunk_size_tokens (real per-step chunk, dynamic-aware) instead of a
        hardcoded 5242, so the shadow fallback R matches prefill."""
        remaining = max(num_tokens - hit_length, 0)
        chunk = max(getattr(self, "chunk_size_tokens", 5242), 1)
        return max((remaining + chunk - 1) // chunk, 1)

    # ------------------------------------------------------------------
    # Bucket stats
    # ------------------------------------------------------------------

    def _estimate_result_tokens(self, bucket: str) -> int:
        samples = self.bucket_result_tokens.get(bucket)
        if not samples:
            return self.default_result_tokens
        sv = sorted(samples)
        return int(sv[len(sv) // 2])

    def _update_bucket_result_tokens(self, bucket: str,
                                     actual_tokens: int) -> None:
        if actual_tokens <= 0 or actual_tokens > 100_000:
            return  # sanity bound
        lst = self.bucket_result_tokens.setdefault(bucket, [])
        lst.append(int(actual_tokens))
        # Keep last 200 samples for rolling median.
        if len(lst) > 200:
            del lst[:len(lst) - 200]

    # ------------------------------------------------------------------
    # Prediction log
    # ------------------------------------------------------------------

    def _write_prediction_record(self, p: Pending) -> None:
        try:
            rec = {
                "ts": time.time(),
                "traj_id": p.traj_id,
                "agent_round_next": p.agent_round_next,
                "request_id": p.request_id,
                "bucket": p.bucket,
                # estimates
                "est_num_tokens": p.est_num_tokens,
                "est_eta_step": p.est_eta_step,
                "est_tool_time_s": p.est_tool_time_s,
                # reality (from Stage 2 + admit)
                "real_num_tokens": p.real_num_tokens,
                "real_arrival_step": p.real_arrival_step,
                "real_tool_time_s": p.real_tool_time_s,
                "real_admit_step": p.real_admit_step,
                # predictions
                "pred_k_queue_stage1": p.pred_k_queue_stage1,
                "pred_k_queue_stage2": p.pred_k_queue_stage2,
                "actual_k_queue": p.actual_k_queue,
            }
            with open(self.pred_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.counters["predictions_written"] += 1
        except Exception as e:
            logger.debug("shadow pred log write failed: %s", e)


# ---------------------------------------------------------------------------
# Internal simulator dataclasses (kept private — used in _simulate_k_queue)
# ---------------------------------------------------------------------------

@dataclass
class _SimRunning:
    request_id: str
    num_tokens: int
    admit_step: int
    r_total: int
    r_remaining: int
    blocks_needed: int
    in_decode_phase: bool = False


@dataclass
class _SimWaiting:
    request_id: str
    num_tokens: int
    arrival_step: int
    blocks_needed: int
    traj_id: Optional[str]
    agent_round: Optional[int]
    _is_target: bool = False
    # P5: prefix-cache hit modelled on prefill (回传'd prefix).  hit =
    # tokens already cached (chunking starts after it); evictable = hit
    # blocks consumed from the free queue at admit.
    hit: int = 0
    evictable: int = 0


@dataclass
class _SimPending:
    key: tuple
    num_tokens: int
    blocks_needed: int
    eta_step: int
    is_target: bool


def _running_copy(r: RunningState) -> _SimRunning:
    return _SimRunning(
        request_id=r.request_id,
        num_tokens=r.num_tokens,
        admit_step=r.admit_step,
        r_total=r.r_total,
        r_remaining=r.r_remaining,
        blocks_needed=r.blocks_needed,
        in_decode_phase=r.in_decode_phase,
    )


# ---------------------------------------------------------------------------
# LICHTV2 admission helpers (mirror of prefill scheduler functions)
# ---------------------------------------------------------------------------

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _lv2_R_at(num_tokens: int, current_offset: int,
              chunk_size: int) -> int:
    remaining = max(num_tokens - current_offset, 0)
    if remaining <= 0:
        return 0
    return _ceil_div(remaining, chunk_size)


def _lv2_release_blocks(num_tokens: int, num_computed_at_admit: int,
                        block_size: int) -> int:
    net = max(num_tokens - num_computed_at_admit, 0)
    return _ceil_div(net, block_size)


def _lv2_B_at(num_tokens: int, current_offset: int, t: int,
              chunk_size: int, block_size: int) -> int:
    """Blocks newly allocated by request at future step `t`, relative
    to NOW (offset = current_offset).  Mirror of _licht_v2_B_at."""
    Ri = _lv2_R_at(num_tokens, current_offset, chunk_size)
    if not (0 <= t < Ri):
        return 0
    remaining_now = max(num_tokens - current_offset, 0)
    cum_t = current_offset + min(chunk_size * (t + 1), remaining_now)
    cum_prev = current_offset + (
        min(chunk_size * t, remaining_now) if t > 0 else 0)
    return max(_ceil_div(cum_t, block_size)
               - _ceil_div(cum_prev, block_size), 0)


def _lv2_score(K: int, t_wait_s: float, a: float, b: float,
               tmax_s: float, alpha: float) -> float:
    """Mirror of LICHT prefill score (line 67 of queue_time/simulator.py).
    Higher = higher priority."""
    K = max(K, 0)
    wait_term = max(t_wait_s - tmax_s, 0.0)
    round_decay = (1.0 + K) ** (-alpha)
    return a * math.log1p(K) + b * round_decay * wait_term


def _lv2_can_admit(candidate_num_tokens: int, candidate_hit: int,
                   candidate_evictable_prefix: int,
                   future_free: list[int], future_alloc: list[int],
                   N: int, chunk_size: int, block_size: int,
                   max_alloc_per_step: int,
                   long_tail_headroom: int, long_count: int,
                   max_long_bridge: int) -> bool:
    """Mirror of _licht_v2_can_admit.  Returns True if candidate fits
    under all three LICHTV2 guards."""
    Rj = _lv2_R_at(candidate_num_tokens, candidate_hit, chunk_size)
    if Rj <= 0:
        return True  # nothing to schedule, let regular path handle
    long_tail = Rj > N
    # Guard 1: long-tail concurrency cap (only for long-tail candidates)
    if long_tail:
        if long_count + 1 > max_long_bridge:
            return False
    # Guard 2 threshold: headroom ONLY for long-tail candidates, else 0
    # (mirror of prefill's _licht_v2_can_admit line 1032-1034).
    threshold = long_tail_headroom if long_tail else 0
    cum_delta = 0
    for t in range(0, N + 1):
        if t == 0:
            cum_delta -= candidate_evictable_prefix
        bit_j = 0
        if t < Rj:
            bit_j = _lv2_B_at(candidate_num_tokens, candidate_hit, t,
                              chunk_size, block_size)
            cum_delta -= bit_j
        elif t == Rj:
            cum_delta += (_lv2_release_blocks(
                candidate_num_tokens, candidate_hit, block_size)
                + candidate_evictable_prefix)
        # Guard 2: block availability
        ff = future_free[t] if t < len(future_free) else 0
        if ff + cum_delta < threshold:
            return False
        # Guard 3: per-step alloc budget
        if t < Rj:
            fa = future_alloc[t] if t < len(future_alloc) else 0
            if fa + bit_j > max_alloc_per_step:
                return False
    return True


def _lv2_apply_to_timeline(candidate_num_tokens: int, candidate_hit: int,
                            candidate_evictable_prefix: int,
                            future_free: list[int],
                            future_alloc: list[int],
                            N: int, chunk_size: int,
                            block_size: int) -> None:
    """Mirror of _licht_v2_apply_to_timeline.  Mutates future_free /
    future_alloc IN PLACE to commit candidate's events."""
    Rj = _lv2_R_at(candidate_num_tokens, candidate_hit, chunk_size)
    if Rj <= 0:
        return
    cum_delta = 0
    for t in range(0, N + 1):
        if t == 0:
            cum_delta -= candidate_evictable_prefix
        if t < Rj:
            bit = _lv2_B_at(candidate_num_tokens, candidate_hit, t,
                            chunk_size, block_size)
            cum_delta -= bit
            if t < len(future_alloc):
                future_alloc[t] += bit
        elif t == Rj:
            cum_delta += (_lv2_release_blocks(
                candidate_num_tokens, candidate_hit, block_size)
                + candidate_evictable_prefix)
        if t < len(future_free):
            future_free[t] += cum_delta
