# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LICHTV3 runtime predictors.

`ProductionStepCountPredictor`
    In-process port of `queue_time/simulator_v1_slot_only.py` that
    consumes a `LichtV3PrefillSnapshot` (built from live scheduler
    state) plus a list of "ghost" arrivals known on the decode side
    (requests currently doing tool calls that will hit prefill soon).
    Output is `steps_to_admit` (int) — explicitly NOT seconds.  Step
    count is the metric the offline simulator predicts most accurately
    when the horizon is small, and the v3 decision rule does not need
    a wall-clock T_admit.

`ToolTimePredictorWrapper`
    Thin wrapper around `tool_call_time.train.Predictor`.  At runtime
    the assistant's tool_calls list arrives only after the decode loop
    has produced the message; this wrapper accepts whatever feature
    bundle the caller assembles and returns T_tool in seconds.  Until
    the feature-extraction adapter from tool_call_time/features.py is
    wired into the decode hook, this falls back to the bucket-median
    table from the bundle (still trained, just no feature signal).
"""
from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Step-count predictor (offline simulator port)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _RunningItem:
    finish_time: float
    idx: int = field(compare=False)


@dataclass
class _WaitingItem:
    idx: int
    arrival: float
    K: int
    pf_duration: float
    is_target: bool = False


@dataclass(frozen=True)
class GhostArrival:
    """A request currently in tool-call phase that decode predicts will
    arrive at prefill at `eta_ts` with round number `agent_round`."""
    request_id: str
    eta_ts: float
    agent_round: int
    num_prompt_tokens_estimate: int


@dataclass(frozen=True)
class TargetReq:
    """The request we are scoring this turn."""
    request_id: str
    eta_ts: float            # = decode_finish_ts + T_tool
    agent_round: int         # next round = current + 1
    num_prompt_tokens_estimate: int


def licht_score(K: int, t_wait: float, *, a: float, b: float,
                tmax_s: float, alpha: float) -> float:
    K = max(K, 0)
    wait_term = max(t_wait - tmax_s, 0.0)
    round_decay = (1.0 + K) ** (-alpha)
    return a * math.log1p(K) + b * round_decay * wait_term


class ProductionStepCountPredictor:
    """Pure-step LICHTV2 admission simulator.

    Mirrors the prefill scheduler's `_licht_v2_can_admit` /
    `_licht_v2_apply_to_timeline` logic against the snapshot's
    `licht_v2_future_free` / `licht_v2_future_alloc` arrays.  The
    timeline is indexed by SCHEDULER STEP (0..N), not seconds — no
    `chunk_step_s` constant, no heap advance in time domain.

    Algorithm:
      1. Build a candidate list:
           (waiting reqs from snapshot) + ghost arrivals + target
      2. Compute LICHT score for each.  `wait_term` is in real seconds
         and uses the snapshot's `arrival_ts` field for waiting reqs;
         ghosts and target have wait_term = 0 (haven't arrived yet).
      3. Sort candidates by score descending (= LICHT prefill score
         picker order).
      4. Walk candidates: for each, run can_admit against the local
         timeline copy.  If yes, apply_to_timeline and increment an
         admit counter.  When target admits, return the counter.
         If can_admit fails for target, return N (cap).
    """

    def __init__(self,
                 max_slots: int,
                 # Kept for API compatibility; only used by the
                 # fallback path when the snapshot has no v2 timeline.
                 avg_chunk_step_s: float = 0.5):
        self.max_slots = max(max_slots, 1)
        self.avg_chunk_step_s = max(avg_chunk_step_s, 1e-3)

    def predict(self,
                snapshot: dict,
                ghost_arrivals: list[GhostArrival],
                target: TargetReq) -> int:
        """Return predicted K_queue = position (0-based) of target in
        the LICHT-score-sorted admit sequence, or N when target cannot
        be admitted within the lookahead horizon."""
        future_free = snapshot.get("licht_v2_future_free")
        future_alloc = snapshot.get("licht_v2_future_alloc")
        if not future_free or not future_alloc:
            # licht-v2 not active on prefill — fall back to slot model.
            return self._fallback_slot_predict(
                snapshot, ghost_arrivals, target)

        c = snapshot["constants"]
        A = float(c["score_a"])
        B = float(c["score_b"])
        Tmax = float(c["score_tmax_s"])
        Alpha = float(c["round_decay_alpha"])
        N = int(c["lichtv2_horizon_n"])
        chunk_tokens = max(int(c["chunk_size_tokens"]), 1)
        block_size = max(int(snapshot["block_size"]), 1)
        total_blocks = int(snapshot.get("total_kv_blocks", 0))
        max_alloc_per_step = max(
            int(snapshot["max_num_batched_tokens"]) // block_size, 1)
        # LICHTV2_LONG_TAIL_HEADROOM_RATIO is 0.025 in scheduler config.
        long_tail_threshold = int(0.025 * total_blocks)
        # LICHTV2_MAX_LONG_BRIDGE = 2 in scheduler config.
        max_long_bridge = 2
        snap_ts = float(snapshot["timestamp"])

        # ---- assemble candidates ----
        # Each entry: (request_id, K, num_tokens, score, is_target).
        # num_tokens = total prompt tokens (= the "request.num_tokens"
        # licht-v2 uses to compute R/B).
        candidates = []
        for w in snapshot["waiting"]:
            n_tok = int(w["num_prompt_tokens"])
            K = int(w["agent_round"])
            twait = max(snap_ts - float(w["arrival_ts"]), 0.0)
            score = _score(A, B, Tmax, Alpha, K, twait)
            candidates.append(_Candidate(
                request_id=w["request_id"], K=K,
                num_tokens=n_tok, score=score, is_target=False))
        for g in ghost_arrivals:
            if g.request_id == target.request_id:
                continue
            n_tok = max(int(g.num_prompt_tokens_estimate), 1)
            # twait at snapshot moment: positive iff ghost's eta is in
            # the past (it should be in the queue already by now);
            # zero iff ghost has not yet arrived.  Real scheduler scores
            # at admit moment with the actual elapsed twait — using
            # snap_ts is the best estimate when arrival is still future.
            g_twait = max(snap_ts - float(g.eta_ts), 0.0)
            candidates.append(_Candidate(
                request_id=g.request_id, K=g.agent_round,
                num_tokens=n_tok,
                score=_score(A, B, Tmax, Alpha, g.agent_round, g_twait),
                is_target=False))
        n_tok_t = max(int(target.num_prompt_tokens_estimate), 1)
        # Same twait treatment for target.  Note: this score is computed
        # at snap_ts moment.  Real ranking later might give target a
        # larger boost as twait grows beyond Tmax — but unless target
        # is already long-waiting at snap_ts, the initial twait is 0.
        t_twait = max(snap_ts - float(target.eta_ts), 0.0)
        candidates.append(_Candidate(
            request_id=target.request_id, K=target.agent_round,
            num_tokens=n_tok_t,
            score=_score(A, B, Tmax, Alpha, target.agent_round, t_twait),
            is_target=True))

        # ---- sort by LICHT score desc, tie-break stable on request_id ----
        candidates.sort(key=lambda c_: (-c_.score, c_.request_id))

        # ---- count current long-running on snapshot ----
        long_running = sum(
            1 for r in snapshot["running"]
            if int(r["r_remaining_chunks"]) > N)

        # ---- mutable timeline copies ----
        ff = list(future_free)
        fa = list(future_alloc)

        # ---- walk candidates, simulating admits ----
        admit_count = 0
        for cand in candidates:
            ok, deltas, alloc_per_t = _can_admit_step_domain(
                cand.num_tokens, chunk_tokens, block_size,
                ff, fa, N, max_alloc_per_step,
                long_tail_threshold, long_running, max_long_bridge)
            if not ok:
                if cand.is_target:
                    return N
                continue
            # Apply: update timeline copies + long_running counter.
            _apply_step_domain(ff, fa, deltas, alloc_per_t, N)
            R_j = _chunks_needed(cand.num_tokens, chunk_tokens)
            if R_j > N:
                long_running += 1
            if cand.is_target:
                return admit_count
            admit_count += 1

        # Should not reach here (target was in candidates) — defensive.
        return N

    # ------------------------------------------------------------------
    # Fallback (slot-only, used when v2 timeline is absent)
    # ------------------------------------------------------------------

    def _fallback_slot_predict(
            self, snapshot, ghost_arrivals, target) -> int:
        """Lightweight slot-based fallback when licht_v2 is not active.
        Uses a coarse step-count proxy with avg_chunk_step_s for heap
        advance — same model as before the rewrite."""
        c = snapshot["constants"]
        A = float(c["score_a"])
        B = float(c["score_b"])
        Tmax = float(c["score_tmax_s"])
        Alpha = float(c["round_decay_alpha"])
        snap_ts = float(snapshot["timestamp"])
        chunk_step_s = self.avg_chunk_step_s

        running_heap: list[_RunningItem] = []
        next_idx = 0
        for r in snapshot["running"]:
            remaining_s = max(int(r["r_remaining_chunks"]), 0) * chunk_step_s
            finish_time = snap_ts + remaining_s
            running_heap.append(_RunningItem(finish_time, next_idx))
            next_idx += 1
        heapq.heapify(running_heap)

        waiting: list[_WaitingItem] = []
        for w in snapshot["waiting"]:
            pf_duration = max(int(w["r_full_chunks"]), 1) * chunk_step_s
            waiting.append(_WaitingItem(
                idx=next_idx, arrival=float(w["arrival_ts"]),
                K=int(w["agent_round"]), pf_duration=pf_duration))
            next_idx += 1

        chunk_tokens = max(int(c["chunk_size_tokens"]), 1)
        for g in ghost_arrivals:
            if g.request_id == target.request_id:
                continue
            chunks = max((g.num_prompt_tokens_estimate + chunk_tokens - 1)
                         // chunk_tokens, 1)
            waiting.append(_WaitingItem(
                idx=next_idx, arrival=g.eta_ts, K=g.agent_round,
                pf_duration=chunks * chunk_step_s))
            next_idx += 1

        target_chunks = max((target.num_prompt_tokens_estimate
                              + chunk_tokens - 1) // chunk_tokens, 1)
        waiting.append(_WaitingItem(
            idx=next_idx, arrival=target.eta_ts, K=target.agent_round,
            pf_duration=target_chunks * chunk_step_s, is_target=True))

        now = snap_ts
        steps = 0
        max_steps = 1024
        while waiting and steps < max_steps:
            while len(running_heap) >= self.max_slots and running_heap:
                top = heapq.heappop(running_heap)
                if top.finish_time > now:
                    now = top.finish_time
            best_score = -float("inf")
            best_pos = -1
            best_arrival = float("inf")
            for i, w in enumerate(waiting):
                if w.arrival > now:
                    continue
                s = licht_score(w.K, now - w.arrival,
                                a=A, b=B, tmax_s=Tmax, alpha=Alpha)
                if (s > best_score
                        or (s == best_score and w.arrival < best_arrival)):
                    best_score = s
                    best_arrival = w.arrival
                    best_pos = i
            if best_pos < 0:
                next_arrival = min(w.arrival for w in waiting)
                now = max(now, next_arrival)
                continue
            admitted = waiting.pop(best_pos)
            steps += 1
            if admitted.is_target:
                return max(steps - 1, 0)
            heapq.heappush(
                running_heap,
                _RunningItem(now + admitted.pf_duration, admitted.idx))
        return max_steps


# ---------------------------------------------------------------------------
# Step-domain helpers (mirrors of Scheduler._licht_v2_*)
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    request_id: str
    K: int
    num_tokens: int
    score: float
    is_target: bool


def _score(a: float, b: float, tmax: float, alpha: float,
           K: int, t_wait: float) -> float:
    K = max(K, 0)
    wait_term = max(t_wait - tmax, 0.0)
    round_decay = (1.0 + K) ** (-alpha)
    return a * math.log1p(K) + b * round_decay * wait_term


def _chunks_needed(num_tokens: int, chunk_tokens: int) -> int:
    """Mirror of Scheduler._licht_v2_R_at(req, 0)."""
    n = max(int(num_tokens), 0)
    return (n + chunk_tokens - 1) // chunk_tokens


def _b_at(num_tokens: int, chunk_tokens: int, block_size: int,
          t: int) -> int:
    """Mirror of Scheduler._licht_v2_B_at(req, 0, t):
    blocks newly allocated by a fresh request at future step `t`.

    Treats num_computed_at_admit = 0 (no prefix-cache hit).  When the
    scheduler later admits with a non-zero prefix-cache hit the alloc
    profile is strictly smaller — using 0 is the conservative estimate."""
    if t < 0:
        return 0
    R = _chunks_needed(num_tokens, chunk_tokens)
    if t >= R:
        return 0
    cum_t = min(chunk_tokens * (t + 1), num_tokens)
    cum_prev = min(chunk_tokens * t, num_tokens) if t > 0 else 0
    return max(((cum_t + block_size - 1) // block_size)
                - ((cum_prev + block_size - 1) // block_size), 0)


def _release_blocks(num_tokens: int, block_size: int) -> int:
    """Mirror of Scheduler._licht_v2_release_blocks(req, 0)."""
    return (max(int(num_tokens), 0) + block_size - 1) // block_size


def _can_admit_step_domain(
        num_tokens: int, chunk_tokens: int, block_size: int,
        future_free: list, future_alloc: list,
        N: int, max_alloc_per_step: int,
        long_tail_threshold: int,
        long_running: int, max_long_bridge: int):
    """Mirror of Scheduler._licht_v2_can_admit but operates on local
    list copies.  Returns (ok, deltas_per_t, alloc_per_t) so the caller
    can apply on success without recomputing."""
    R_j = _chunks_needed(num_tokens, chunk_tokens)
    if R_j <= 0:
        # Nothing to schedule — treat as already-admitted no-op.
        return True, [0] * (N + 1), [0] * (N + 1)

    long_tail = R_j > N
    if long_tail and (long_running + 1 > max_long_bridge):
        return False, None, None

    threshold = long_tail_threshold if long_tail else 0

    deltas = [0] * (N + 1)
    alloc_per_t = [0] * (N + 1)
    cum_delta = 0
    for t in range(0, N + 1):
        bit_j = 0
        if t < R_j:
            bit_j = _b_at(num_tokens, chunk_tokens, block_size, t)
            cum_delta -= bit_j
            alloc_per_t[t] = bit_j
        elif t == R_j:
            cum_delta += _release_blocks(num_tokens, block_size)
        deltas[t] = cum_delta
        # Guard 2: block availability.
        if future_free[t] + cum_delta < threshold:
            return False, None, None
        # Guard 3: per-step alloc budget.
        if t < R_j and (future_alloc[t] + bit_j) > max_alloc_per_step:
            return False, None, None
    return True, deltas, alloc_per_t


def _apply_step_domain(future_free: list, future_alloc: list,
                       deltas: list, alloc_per_t: list, N: int) -> None:
    """In-place commit of a candidate's events on the timeline copies."""
    for t in range(0, N + 1):
        future_free[t] = future_free[t] + deltas[t]
        future_alloc[t] = future_alloc[t] + alloc_per_t[t]


# ---------------------------------------------------------------------------
# Tool-time predictor wrapper
# ---------------------------------------------------------------------------

class ToolTimePredictorWrapper:
    """Wraps tool_call_time.train.Predictor for runtime use.

    Two construction modes:
      - `from_run_dir(path, ...)`: load a trained bundle.  Production
        path: feed a vLLM `Request` to `.predict_for_request(req)` and
        we will detokenize → extract tool_call → build features →
        predict, end-to-end.
      - `with_fallback(default_t_tool_s)`: no predictor loaded; every
        prediction returns the constant.  Used when the model bundle
        cannot be loaded (env var unset, file missing, etc.).
    """

    def __init__(self,
                 predictor: Optional[Any] = None,
                 default_t_tool_s: float = 5.0,
                 tokenizer_provider: Optional[Any] = None,
                 trajectory_tracker: Optional[Any] = None):
        self._predictor = predictor
        self._default = max(default_t_tool_s, 0.0)
        self._fallback_warned = False
        self._tokenizer_provider = tokenizer_provider  # callable: () -> Tokenizer
        self._tracker = trajectory_tracker
        # Cached const-table predictor fields for submit/editor
        # families where ML is intentionally bypassed.
        self._median_t_by_bucket: dict[str, float] = {}
        self._global_overall_median_t: float = 0.0
        self._unreliable_buckets: set[str] = set()
        if predictor is not None:
            bundle = getattr(predictor, "bundle", {}) or {}
            self._median_t_by_bucket = dict(
                bundle.get("median_t_by_bucket", {}))
            self._global_overall_median_t = float(
                bundle.get("global_overall_median_t", default_t_tool_s))
            self._unreliable_buckets = set(
                bundle.get("unreliable_buckets", []) or [])

    @classmethod
    def from_run_dir(cls, run_dir: str | os.PathLike,
                     default_t_tool_s: float = 5.0,
                     tokenizer_provider: Optional[Any] = None,
                     ) -> "ToolTimePredictorWrapper":
        try:
            import sys
            tool_pkg = "/data/whr/vllm-continuum"
            if tool_pkg not in sys.path:
                sys.path.insert(0, tool_pkg)
            from tool_call_time.train import Predictor as _P  # type: ignore
            p = _P.load(Path(run_dir))
            logger.info("LICHTV3 ToolTimePredictor loaded from %s", run_dir)
            bundle = getattr(p, "bundle", {}) or {}
            # Build tracker using the bundle's global_bucket_log_means.
            global_mu = bundle.get("global_bucket_log_means", {}) or {}
            timeout_thresh = float(
                bundle.get("timeout_threshold_s", 60.0))
            from .features_adapter import TrajectoryTracker
            tracker = TrajectoryTracker(global_mu, timeout_thresh)
            return cls(predictor=p, default_t_tool_s=default_t_tool_s,
                       tokenizer_provider=tokenizer_provider,
                       trajectory_tracker=tracker)
        except Exception as e:
            logger.warning(
                "LICHTV3 ToolTimePredictor load failed (%s); "
                "falling back to constant T_tool=%.2fs", e, default_t_tool_s)
            return cls(predictor=None, default_t_tool_s=default_t_tool_s,
                       tokenizer_provider=tokenizer_provider)

    @classmethod
    def with_fallback(cls, default_t_tool_s: float = 5.0
                       ) -> "ToolTimePredictorWrapper":
        return cls(predictor=None, default_t_tool_s=default_t_tool_s)

    # ------------------------------------------------------------------
    # Production entry point
    # ------------------------------------------------------------------

    def extract_tool_call_for_request(self, request) -> Optional[dict]:
        """Parse the request's model output and return an OpenAI-format
        tool_call dict, or None if no tool_call is detected.

        Single path: regex over the detokenised model output.  Same
        contract as a real deployment (no trace shortcuts) — relies on
        the trace_replay client emitting a full ReAct-style assistant
        message (content + tool_call JSON) per round.
        """
        from .features_adapter import extract_tool_call
        text = self._decode_output(request)
        if not text:
            return None
        return extract_tool_call(text)

    def predict_for_request(self, request,
                            tc: Optional[dict] = None) -> Optional[float]:
        """Return only the p50 prediction (kept for back-compat).
        Internal callers should prefer `predict_full_for_request`."""
        res = self.predict_full_for_request(request, tc=tc)
        if res is None:
            return None
        return res.get("p50")

    def predict_full_for_request(
            self, request,
            tc: Optional[dict] = None) -> Optional[dict]:
        """End-to-end prediction from a vLLM v1 Request, returning a dict
        with ALL signals the bundle exposes:
          {
            "p50":        median tool exec time (seconds)
            "p95":        95th percentile (seconds)
            "p_timeout":  prob. of being a long-running outlier
            "bucket":     classified bucket string (e.g. bash::python::script_repro)
            "family":     bash / editor / submit / unknown
            "source":     "ml" | "bucket_median" | "fallback"
          }

        Returns None iff no tool_call detected (trajectory likely ended).

        Behaviour follows tool_call_time's design exactly:
          - bash family → ML predict_df pred_p50 + pred_p95 + pred_p_timeout
          - editor / submit family → bundle bucket median (lookup, no ML);
            p95 = p50 (we don't have a separate quantile table for them),
            p_timeout = 0.0 (training submit/editor never hit 60s).
          - no tool_call detected → return None.
        """
        if self._predictor is None or self._tracker is None:
            if not self._fallback_warned:
                logger.warning(
                    "LICHTV3 ToolTimePredictor running in fallback mode; "
                    "T_tool := %.2fs for every request.", self._default)
                self._fallback_warned = True
            return {
                "p50": self._default, "p95": self._default,
                "p_timeout": 0.0, "bucket": "fallback",
                "family": "fallback", "source": "fallback",
            }
        try:
            if tc is None:
                tc = self.extract_tool_call_for_request(request)
            if tc is None:
                return None
            job_id = (getattr(request, "job_id", None)
                       or getattr(request, "request_id", "_unknown_"))
            from .features_adapter import _ensure_tool_call_time_on_path
            _ensure_tool_call_time_on_path()
            from bucket import classify, family  # type: ignore
            bucket = classify(tc)
            fam = family(bucket)
            if fam == "submit" or fam == "editor":
                t = float(self._median_t_by_bucket.get(
                    bucket, self._global_overall_median_t))
                return {
                    "p50": t, "p95": t, "p_timeout": 0.0,
                    "bucket": bucket, "family": fam,
                    "source": "bucket_median",
                }
            # ML path (bash family) — run predict_df once, harvest all 3.
            row = self._tracker.feature_row(job_id, tc)
            import pandas as pd
            df = pd.DataFrame([row])
            out = self._predictor.predict_df(df)
            def _safe(col: str, default: float = 0.0) -> float:
                if col not in out.columns:
                    return default
                v = float(out[col].iloc[0])
                if not (v == v):  # NaN
                    return default
                return v
            p50 = _safe("pred_p50", self._global_overall_median_t)
            if p50 <= 0:
                p50 = float(self._global_overall_median_t)
            # p95 fallback: use p50 if column missing (older bundles).
            p95 = max(_safe("pred_p95", p50), p50)
            p_timeout = max(min(_safe("pred_p_timeout", 0.0), 1.0), 0.0)
            return {
                "p50": p50, "p95": p95, "p_timeout": p_timeout,
                "bucket": bucket, "family": fam, "source": "ml",
            }
        except Exception as e:
            logger.warning("LICHTV3 predict_for_request error: %s", e)
            return {
                "p50": self._default, "p95": self._default,
                "p_timeout": 0.0, "bucket": "error",
                "family": "error", "source": "error",
            }

    def observe_for_job(self, job_id: str, tc: dict,
                        actual_t: float,
                        observation_text: str = "") -> None:
        """Called by decode_manager once a round has actually run and we
        can measure (a) its true `actual_t` (= wall-clock elapsed
        between consecutive on_round_finished calls of the same job)
        and (b) its tool output text (= the most recent <tool> body
        in the next round's prompt).  Both feed TrajectoryState the
        same way training did, so E1-E5 + C features are accurate."""
        if self._tracker is None or tc is None:
            return
        try:
            self._tracker.observe(job_id, tc, float(actual_t),
                                  observation_text=observation_text)
        except Exception as e:  # pragma: no cover
            logger.debug(
                "LICHTV3 observe_for_job failed job=%s: %s", job_id, e)

    def decode_prompt_tail(self, request, max_tokens: int = 8192) -> str:
        """Detokenize the tail of `request.prompt_token_ids`.  Used by
        decode_manager to extract the most recent tool result for
        observation-text-aware history updates.  Bounded to keep
        per-round detokenization cheap (~ms for 8K tokens)."""
        if self._tokenizer_provider is None:
            return ""
        try:
            tok = self._tokenizer_provider()
            prompt_ids = list(getattr(request, "prompt_token_ids", []) or [])
            if not prompt_ids:
                return ""
            tail = prompt_ids[-max_tokens:]
            return tok.decode(tail, skip_special_tokens=False)
        except Exception as e:  # pragma: no cover
            logger.debug("LICHTV3 decode_prompt_tail failed: %s", e)
            return ""

    def predict(self, features_row: Optional[dict] = None
                 ) -> Optional[float]:
        """Legacy entry: takes a pre-built feature row.  Kept for unit
        tests; production code calls predict_for_request()."""
        if self._predictor is None or features_row is None:
            if not self._fallback_warned:
                logger.warning(
                    "LICHTV3 ToolTimePredictor running in fallback mode; "
                    "T_tool := %.2fs for every request.", self._default)
                self._fallback_warned = True
            return self._default
        try:
            import pandas as pd
            df = pd.DataFrame([features_row])
            out = self._predictor.predict_df(df)
            return float(out["pred_p50"].iloc[0])
        except Exception as e:
            logger.warning("LICHTV3 ToolTimePredictor predict failed: %s", e)
            return self._default

    # ------------------------------------------------------------------
    # Detokenization
    # ------------------------------------------------------------------

    def _decode_output(self, request) -> str:
        if self._tokenizer_provider is None:
            return ""
        try:
            tok = self._tokenizer_provider()
            output_ids = list(getattr(request, "output_token_ids", []) or [])
            if not output_ids:
                return ""
            return tok.decode(output_ids, skip_special_tokens=False)
        except Exception as e:
            logger.debug("LICHTV3 detokenize failed: %s", e)
            return ""
