# SPDX-License-Identifier: Apache-2.0
"""OnlineStepTimePredictor — RLS step-duration predictor with
forgetting factor and multiplicative correction factor.

Drop-in replacement for the legacy AR(1) `StepTimeModel`.  See
docs in standalone copy at `step_time/online_predictor.py` for the
full design rationale and trace-based evaluation results.

Summary
-------
* Features: x_t = [num_scheduled_tokens / 1e6, num_running, 1].
* Target:   y_t = step duration in seconds.
* Update:   recursive least squares with forgetting factor λ=0.995.
* Bias correction: multiplicative slow-EMA factor c_t (α=0.05,
  clamp r ∈ [0.9, 1.02] — asymmetric, biased toward UNDER-estimation
  since over-estimation is more dangerous in K_queue scheduling).
* Cold start: w=0, P=1000·I, c=1.  Step 0 returns None for predict().

Final accuracy on prefill_20003 trace (2204 steps):
  MAE 0.596s, MAPE 6.23%, WAPE 7.37%, P90 APE 11.91%, w05 71.6%,
  under-rate 55.2% (intentional safety bias).
"""
from __future__ import annotations

from typing import Optional


class OnlineStepTimePredictor:
    """RLS with forgetting factor + multiplicative correction factor.

    `predict(features)` returns the corrected step duration in seconds,
    or `None` until the first observation has been recorded.

    `observe(features, duration_s)` updates both the correction factor
    c_t and the RLS weights w.

    `features` is a dict; only keys named in `feature_names` are read,
    so callers can pass extra fields without error.
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

    def predict(self, features: dict) -> Optional[float]:
        if self.n_observed == 0:
            return None
        x = self._make_x(features)
        base = float(x @ self.w)
        return self.c * base

    def predict_base(self, features: dict) -> Optional[float]:
        """Uncorrected base prediction (RLS only, no c).  For diagnostics."""
        if self.n_observed == 0:
            return None
        x = self._make_x(features)
        return float(x @ self.w)

    def observe(self, features: dict, duration_s: float) -> None:
        import numpy as np
        x = self._make_x(features)
        y = float(duration_s)
        # 1) Update c_t using r_t = y_t / ỹ_t with PRE-update weights.
        if self.correction_alpha > 0.0 and self.n_observed > 0:
            base = float(x @ self.w)
            if base >= self.correction_min_base_s:
                r = y / base
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

    def stats(self) -> dict:
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
            "correction": {
                "c": self.c,
                "alpha": self.correction_alpha,
                "n_updates": self.n_c_updates,
                "clamp": list(self.correction_clamp),
            },
        }
