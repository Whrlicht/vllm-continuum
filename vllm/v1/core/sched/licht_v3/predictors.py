# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LICHTV3 runtime tool-time predictor.

`ToolTimePredictorWrapper`
    Thin wrapper around `tool_call_time.train.Predictor`.  At runtime
    the assistant's tool_calls list arrives only after the decode loop
    has produced the message; this wrapper accepts whatever feature
    bundle the caller assembles and returns T_tool in seconds.  Until
    the feature-extraction adapter from tool_call_time/features.py is
    wired into the decode hook, this falls back to the bucket-median
    table from the bundle (still trained, just no feature signal).

NOTE: the legacy file-snapshot K_queue predictor
(`ProductionStepCountPredictor`) was removed once the event-driven
`ShadowScheduler` (step_event ZMQ path) became the sole K_queue
predictor.  Only the tool-time wrapper lives here now.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


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
