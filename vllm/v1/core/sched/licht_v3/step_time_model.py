# SPDX-License-Identifier: Apache-2.0
"""StepTimeModel — per-step wall-clock duration predictor (AR(1)).

Given a stream of (tokens, duration) observations from prefill StepEvents,
predict the duration of an upcoming step as:

    expected(n)  = alpha + beta * tokens(n)
    residual(n)  = duration(n) - expected(n)
    predict(n+1) = expected(n+1) + rho * residual(n)

* `alpha`, `beta`: refit every `refit_every` observations via least-squares
  over the most recent `window` observations.
* `rho`: lag-1 autocorrelation of residuals, clamped to [0, 0.9].
* Cold start: defaults `alpha=50ms`, `beta=10us/token`, `rho=0` until the
  first refit (after `min_obs_for_refit` observations).

The model is intentionally simple — for production-load step time the
linear-with-mild-autocorr structure captures most of the signal.  More
complex models (LightGBM, etc.) can replace it later behind the same
interface.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class StepTimeModelStats:
    alpha: float
    beta: float
    rho: float
    n_obs: int
    n_refits: int
    last_residual: float
    residual_std: float


class StepTimeModel:
    """AR(1) step-duration predictor with online refit."""

    def __init__(self,
                 alpha_init: float = 0.050,
                 beta_init: float = 1.0e-5,
                 window: int = 100,
                 refit_every: int = 20,
                 min_obs_for_refit: int = 20,
                 rho_max: float = 0.9):
        self.alpha = float(alpha_init)
        self.beta = float(beta_init)
        self.rho = 0.0
        self._last_residual = 0.0
        self._have_last_residual = False
        self._history: "deque[tuple[int, float]]" = deque(maxlen=window)
        self._residual_history: "deque[float]" = deque(maxlen=window)
        self._refit_every = max(refit_every, 1)
        self._min_obs_for_refit = max(min_obs_for_refit, 2)
        self._rho_max = rho_max
        self._steps_since_fit = 0
        self.n_obs = 0
        self.n_refits = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, tokens: int, duration: float) -> None:
        """Record one observed step.  May trigger a refit."""
        if tokens < 0 or duration <= 0:
            return
        # Compute residual under CURRENT params (before this update).
        expected = self.alpha + self.beta * tokens
        residual = duration - expected
        self._last_residual = residual
        self._have_last_residual = True
        self._history.append((int(tokens), float(duration)))
        self._residual_history.append(residual)
        self.n_obs += 1
        self._steps_since_fit += 1
        if (self._steps_since_fit >= self._refit_every
                and len(self._history) >= self._min_obs_for_refit):
            self._refit()
            self._steps_since_fit = 0

    def predict(self, tokens: int, *, use_ar: bool = True) -> float:
        """Predicted duration (seconds) for a step that will process
        `tokens` tokens.  When `use_ar` is True, applies the AR(1)
        correction using the most recent residual."""
        expected = self.alpha + self.beta * max(int(tokens), 0)
        if use_ar and self._have_last_residual and self.n_obs >= 5:
            expected += self.rho * self._last_residual
        # Floor at 1 ms to avoid pathological zero/negative outputs.
        return max(expected, 0.001)

    def stats(self) -> StepTimeModelStats:
        if self._residual_history:
            n = len(self._residual_history)
            mean = sum(self._residual_history) / n
            var = sum((r - mean) ** 2 for r in self._residual_history) / n
            std = var ** 0.5
        else:
            std = 0.0
        return StepTimeModelStats(
            alpha=self.alpha,
            beta=self.beta,
            rho=self.rho,
            n_obs=self.n_obs,
            n_refits=self.n_refits,
            last_residual=self._last_residual,
            residual_std=std,
        )

    # ------------------------------------------------------------------
    # Internal: refit via 2-variable least squares
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        """Closed-form least squares on `_history` to update alpha, beta,
        then estimate rho from lag-1 autocorrelation of resulting
        residuals.  Pure-Python (no numpy) — n=100 is trivial."""
        H = list(self._history)
        n = len(H)
        sum_t = sum(t for t, _ in H)
        sum_d = sum(d for _, d in H)
        sum_tt = sum(t * t for t, _ in H)
        sum_td = sum(t * d for t, d in H)
        denom = n * sum_tt - sum_t * sum_t
        if denom <= 1e-9:
            # All tokens identical → can't separate alpha from beta.
            # Keep beta, just update alpha to mean duration.
            self.alpha = sum_d / n
        else:
            beta_new = (n * sum_td - sum_t * sum_d) / denom
            alpha_new = (sum_d - beta_new * sum_t) / n
            # Sanity: clamp to positive (negative slope = pathological).
            self.beta = max(beta_new, 0.0)
            # Allow small negative alpha (e.g. fit artifact) but floor.
            self.alpha = max(alpha_new, 0.0)
        # Recompute residuals using new params.
        new_resid = [d - (self.alpha + self.beta * t) for t, d in H]
        # Lag-1 autocorrelation.
        if len(new_resid) >= 3:
            r0 = new_resid[:-1]
            r1 = new_resid[1:]
            mean0 = sum(r0) / len(r0)
            mean1 = sum(r1) / len(r1)
            num = sum((r0[i] - mean0) * (r1[i] - mean1)
                      for i in range(len(r0)))
            den0 = sum((r - mean0) ** 2 for r in r0) ** 0.5
            den1 = sum((r - mean1) ** 2 for r in r1) ** 0.5
            if den0 > 0 and den1 > 0:
                rho_new = num / (den0 * den1)
                self.rho = max(0.0, min(self._rho_max, rho_new))
            else:
                self.rho = 0.0
        # Refresh internal residual buffer with new params.
        self._residual_history.clear()
        for r in new_resid:
            self._residual_history.append(r)
        if new_resid:
            self._last_residual = new_resid[-1]
        self.n_refits += 1
