#!/usr/bin/env python3
"""Online step-time predictor — Recursive Least Squares (RLS) with
forgetting factor.

Design constraints (user-stated)
--------------------------------
* No offline pretraining.  System cold-starts with zero parameters and
  learns from each new step it sees.  This avoids the "trained on one
  GPU / model, deployed on another" generalisation problem.
* Features per step: x_t = [num_scheduled_tokens_t, num_running_t, 1]
  (last entry = bias).  num_scheduled_tokens here is the per-step
  compute load AFTER prefix-cache removal (= what GPU actually has
  to process).
* Target: y_t = step duration in seconds.
* Feedback delay 1 step (y_t only available after step t finishes).
* Step 0 makes no prediction (no data yet); model just observes (x_0,
  y_0) and starts predicting from step 1 onwards.

Why RLS (not SGD or batch refit)?
--------------------------------
RLS is mathematically equivalent to "rerun closed-form least squares
on all history at every step", but uses an O(d²) recurrence instead
of O(N·d² + d³).  For d=3 this is ~9 multiplies per update — micro-
second-level overhead.  No learning rate to tune.  Fast convergence.

Forgetting factor λ ∈ (0, 1]:
* λ = 1.0  : standard RLS, all history weighted equally.  Best when
            steady-state.
* λ < 1.0  : exponential decay of old data influence.  Adapts to slow
            drift (GPU thermal, memory fragmentation, traffic mix).
* Default λ = 0.995  →  half-life ≈ 138 steps (≈ 17 minutes if each
                        step is ~7s).  Slow adapt.
* λ = 0.99 →  half-life ≈ 69 steps (≈ 8 min).  Faster adapt.

API
---
    p = OnlineStepTimePredictor(forgetting_factor=0.995)
    # Per-step in scheduler:
    ŷ = p.predict(num_scheduled_tokens, num_running)
    # ŷ is None until first observation is recorded.
    # ... step executes, duration measured ...
    p.observe(num_scheduled_tokens, num_running, duration_s)
"""
from __future__ import annotations

from typing import Optional


class OnlineStepTimePredictor:
    """RLS with forgetting factor.  Arbitrary feature list, with an
    implicit bias term (= last weight, fed a constant 1.0).

    Feature scales
    --------------
    Each feature is divided by a `scale` constant before being fed to
    RLS.  This keeps the covariance matrix P well-conditioned when raw
    feature magnitudes span many orders (e.g. num_running ≈ 10 vs
    sum_L_sq ≈ 10^8 → ratio 10^7 → P matrix dominated by sum_L_sq,
    occasional updates blow up other coords).

    Defaults (final 2026-05-19, after 14-point scale scan):
        num_scheduled_tokens   / 1e6      (optimal p50 + w05 on prefill_20003)
        num_running            / 1
        bias                   / 1

    Multiplicative correction factor c_t (added 2026-05-19)
    -------------------------------------------------------
    On top of the RLS base prediction ỹ_t = w·x_t, we apply a slow
    EMA-tracked multiplicative correction c_t:

        ŷ_t   = c_{t-1} · ỹ_t                (final prediction)
        r_t   = y_t / ỹ_t                    (ratio of actual to BASE pred)
        c_t   = (1−α) · c_{t-1} + α · r_t    (EMA, α default 0.05)

    Intuition: RLS handles linear additive trends; c_t captures slow
    multiplicative drift the linear model can't (GPU thermal throttling,
    memory fragmentation, traffic-mix regime shifts).  Two timescales:
    RLS adapts feature weights step-by-step; c_t adjusts the global
    scale on a half-life of ~14 steps at α=0.05.
    """

    DEFAULT_FEATURES = ["num_scheduled_tokens", "num_running"]
    DEFAULT_SCALES = {
        "num_scheduled_tokens": 1.0e6,
    }

    def __init__(self,
                 feature_names: Optional[list] = None,
                 feature_scales: Optional[dict] = None,
                 forgetting_factor: float = 0.995,
                 init_uncertainty: float = 1000.0,
                 correction_alpha: float = 0.05,
                 correction_min_base_s: float = 0.5,
                 correction_clamp: tuple = (0.9, 1.02)):
        """
        Args:
          feature_names: ordered list of feature names to pull from the
            record dict at predict/observe time.  Default = the 4-
            feature set [num_scheduled_tokens, num_running, sum_L_sq,
            attention_waves].  A bias is implicitly added at the end
            (column of 1s), so the parameter vector length is
            len(feature_names) + 1.
          feature_scales: dict mapping feature_name → scale divisor.
            Features not in the dict get scale = 1.0.  Defaults pulled
            from DEFAULT_SCALES.
          forgetting_factor: λ ∈ (0, 1].  0.995 ≈ half-life of 138 steps.
          init_uncertainty: scalar α for initial covariance P_0 = α·I.
          correction_alpha: EMA rate for multiplicative correction c_t.
            0.05 default → half-life ≈ 14 steps.  Set to 0.0 to disable.
          correction_min_base_s: skip c update if base pred is below
            this (avoid spurious r when prediction near zero, e.g. cold
            start).  Default 0.5s.
          correction_clamp: (lo, hi) clamp on r before EMA, to bound
            single-step shocks.  Default (0.9, 1.02) — asymmetric: loose
            on the downside (over-prediction → quick correction down),
            tight on the upside (single under-prediction shouldn't push
            c back up).  Net effect: predictor slightly biased toward
            UNDER-estimation per user request 2026-05-19 (over-estimation
            is more dangerous in the K_queue scheduler).
        """
        import numpy as np
        if not (0.0 < forgetting_factor <= 1.0):
            raise ValueError(
                f"forgetting_factor must be in (0,1], got {forgetting_factor}")
        if not (0.0 <= correction_alpha <= 1.0):
            raise ValueError(
                f"correction_alpha must be in [0,1], got {correction_alpha}")
        self.feature_names = list(feature_names or self.DEFAULT_FEATURES)
        scales = dict(self.DEFAULT_SCALES)
        if feature_scales:
            scales.update(feature_scales)
        self.feature_scales = [float(scales.get(f, 1.0))
                               for f in self.feature_names]
        self.n_features = len(self.feature_names) + 1  # +1 = bias
        self.lam = float(forgetting_factor)
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.P = float(init_uncertainty) * np.eye(self.n_features,
                                                   dtype=np.float64)
        self.n_observed = 0
        # Multiplicative correction factor state
        self.correction_alpha = float(correction_alpha)
        self.correction_min_base_s = float(correction_min_base_s)
        self.correction_clamp = (float(correction_clamp[0]),
                                  float(correction_clamp[1]))
        self.c = 1.0
        self.n_c_updates = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_x(self, features: dict):
        import numpy as np
        vals = []
        for f, s in zip(self.feature_names, self.feature_scales):
            v = features.get(f, 0)
            if v is None:
                v = 0
            vals.append(float(v) / s)
        vals.append(1.0)  # bias
        return np.asarray(vals, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: dict) -> Optional[float]:
        """Final prediction = c_{t-1} · (w·x_t).  Returns None if no
        observation has been seen yet."""
        if self.n_observed == 0:
            return None
        x = self._make_x(features)
        base = float(x @ self.w)
        return self.c * base

    def predict_base(self, features: dict) -> Optional[float]:
        """Uncorrected base prediction (= just the RLS output, no c).
        Useful for diagnostics."""
        if self.n_observed == 0:
            return None
        x = self._make_x(features)
        return float(x @ self.w)

    def observe(self, features: dict, duration_s: float) -> None:
        """Update model with the just-observed (features_t, y_t) pair.
        Updates BOTH the correction factor c_t (using base pred, BEFORE
        RLS update) AND the RLS weights w."""
        import numpy as np
        x = self._make_x(features)
        y = float(duration_s)

        # 1) Update c_t using r_t = y_t / ỹ_t (base pred with PRE-update w).
        # Only after we've seen at least one prior step (so base pred is
        # meaningful) and only when base ≥ threshold (avoid div-by-near-0).
        if self.correction_alpha > 0.0 and self.n_observed > 0:
            base = float(x @ self.w)
            if base >= self.correction_min_base_s:
                r = y / base
                # Clamp single-step shocks to keep c stable.
                r = max(self.correction_clamp[0],
                        min(self.correction_clamp[1], r))
                self.c = ((1.0 - self.correction_alpha) * self.c
                          + self.correction_alpha * r)
                self.n_c_updates += 1

        # 2) Standard RLS update with forgetting factor.
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        k = Px / denom
        residual = y - float(x @ self.w)
        self.w = self.w + k * residual
        self.P = (1.0 / self.lam) * (self.P - np.outer(k, x) @ self.P)
        self.n_observed += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        # weights_scaled = what RLS stores internally (= per unit of
        #                  *normalised* feature).
        # weights_raw    = effective weight per unit of *raw* feature
        #                  (= scaled / scale).  This is what you'd
        #                  multiply against an unnormalised value.
        n = len(self.feature_names)
        weights_scaled = {self.feature_names[i]: float(self.w[i])
                          for i in range(n)}
        weights_scaled["bias"] = float(self.w[-1])
        weights_raw = {self.feature_names[i]:
                       float(self.w[i]) / self.feature_scales[i]
                       for i in range(n)}
        weights_raw["bias"] = float(self.w[-1])
        return {
            "n_observed": self.n_observed,
            "lambda": self.lam,
            "feature_names": list(self.feature_names),
            "feature_scales": {self.feature_names[i]: self.feature_scales[i]
                                for i in range(n)},
            "weights_scaled": weights_scaled,
            "weights_raw": weights_raw,
            "uncertainty_diag": {
                **{self.feature_names[i]: float(self.P[i, i])
                   for i in range(n)},
                "bias": float(self.P[-1, -1]),
            },
            "correction": {
                "c": self.c,
                "alpha": self.correction_alpha,
                "n_updates": self.n_c_updates,
            },
        }
