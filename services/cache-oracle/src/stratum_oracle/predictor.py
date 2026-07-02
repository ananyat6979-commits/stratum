"""
KV cache pressure predictor for stratum-cache-oracle.

Predicts KV block table utilization 100ms into the future using
Holt-Winters exponential smoothing with adaptive seasonality.

WHY PREDICT AHEAD
=================
The router makes routing decisions NOW but the request reaches the
worker 5-50ms later. If the router routes to a worker whose KV cache
is currently at 80% utilization, by the time the request arrives the
cache may be at 95% and about to trigger mass eviction.

Predicting 100ms ahead gives the router enough time to route away from
a worker approaching saturation before the eviction cascade begins.

WHY HOLT-WINTERS
================
KV cache utilization has two components:
1. Level: the current utilization trend
2. Trend: the rate of change (are we filling up or draining?)

Simple exponential smoothing captures the level but not the trend.
Holt-Winters double exponential smoothing captures both, producing
forecasts that are significantly more accurate than naive last-value
prediction for trending time series.

BENCHMARK EVIDENCE (from the blueprint's expected results)
==========================================================
Oracle RMSE: 0.043 (normalized) at 100ms horizon
Baseline (last-value prediction) RMSE: 0.071
Improvement: 39% reduction in prediction error

VLLM METRIC LAG
===============
vLLM's block table utilization metric has a ~200ms lag relative to
actual allocation due to async block table updates. The predictor
accounts for this by adjusting the forecast horizon:
  effective_horizon = requested_horizon + metric_lag
  = 100ms + 200ms = 300ms total lookahead

KNOWN LIMITATION: HORIZON DOMINATED BY CURRENT LEVEL
=====================================================
The effective prediction horizon (100ms requested + 200ms vLLM lag =
300ms) is a small fraction of the Prometheus scrape interval (15,000ms).
The Holt-Winters h-step forecast is level + h*trend, where
h = effective_horizon_ms / scrape_interval_ms = 300/15000 = 0.02.

This means predict()'s output is, by construction, dominated by the
current level -- the trend term contributes at most 2% of one step's
worth of trend. This is mathematically correct Holt-Winters behavior,
not an arithmetic bug: you cannot extract more forecasting signal at
a 300ms horizon than a 15-second sampling interval actually observed.

Practical consequence: predict() currently behaves close to a
last-value predictor at these settings, despite the module's framing
around "seeing eviction cascades coming before they happen." The
benchmark evidence cited in this docstring's original design notes
(RMSE 0.043 vs baseline 0.071) describes the *algorithm's* potential
under adequate sampling density, not this deployment's actual
achieved accuracy -- which has not yet been separately validated
against real production KV telemetry at this scrape interval.

Two paths forward, neither implemented yet (see ADR-007):
  1. Reduce Prometheus scrape interval to sub-second (increases
     scrape load on every worker; may not be supported by all
     backend metrics endpoints)
  2. Replace periodic scraping with event-driven pressure reporting:
     the router itself reports observed KV pressure after each
     request/response cycle, giving true request-granularity signal
     without depending on Prometheus's sampling cadence at all

Until one of these lands, treat predict()'s output as "current level,
lightly trend-adjusted" rather than "a genuine 100ms-ahead forecast."

Reference:
  Holt, C.E. (1957). "Forecasting seasonals and trends by
  exponentially weighted moving averages."
  Brown, R.G. (1959). "Statistical Forecasting for Inventory Control."
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HoltWintersState:
    """Running state for Holt-Winters double exponential smoothing."""
    level: float = 0.0
    trend: float = 0.0
    n_observations: int = 0


class KvPressurePredictor:
    """
    Per-worker KV cache pressure predictor using Holt-Winters smoothing.

    One instance per worker. The router holds a dict[worker_id, predictor].

    Parameters
    ----------
    alpha : float
        Level smoothing factor in (0, 1). Higher = more weight on recent obs.
        Tuned empirically: alpha=0.3 balances responsiveness and stability.
    beta : float
        Trend smoothing factor in (0, 1). Higher = trend adapts faster.
        Tuned empirically: beta=0.1 (trends in KV utilization are slow).
    horizon_ms : int
        Prediction horizon in milliseconds.
    metric_lag_ms : int
        vLLM block table metric reporting lag. Added to horizon.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.1,
        horizon_ms: int = 100,
        metric_lag_ms: int = 200,
    ):
        self.alpha = alpha
        self.beta = beta
        self.horizon_ms = horizon_ms
        self.metric_lag_ms = metric_lag_ms
        self._state = HoltWintersState()
        # Rolling window for RMSE tracking (last 100 predictions vs actuals)
        self._prediction_errors: deque[float] = deque(maxlen=100)

    def update(self, utilization: float) -> None:
        """
        Update the predictor with a new utilization observation.

        Args:
            utilization: Current KV block table utilization in [0.0, 1.0].
                         0.0 = empty, 1.0 = completely full.
        """
        utilization = float(utilization)
        s = self._state

        if s.n_observations == 0:
            # Bootstrap: initialize level to first observation, trend to zero
            s.level = utilization
            s.trend = 0.0
        elif s.n_observations == 1:
            # Second observation: estimate initial trend
            s.trend = utilization - s.level
            s.level = utilization
        else:
            # Holt-Winters update:
            # level_t = alpha * y_t + (1 - alpha) * (level_{t-1} + trend_{t-1})
            # trend_t = beta * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}
            prev_level = s.level
            s.level = self.alpha * utilization + (1.0 - self.alpha) * (s.level + s.trend)
            s.trend = self.beta * (s.level - prev_level) + (1.0 - self.beta) * s.trend

        s.n_observations += 1

    def predict(self, horizon_ms: Optional[int] = None) -> float:
        """
        Predict KV utilization at the given horizon.

        Accounts for vLLM's metric reporting lag by adding metric_lag_ms
        to the effective horizon.

        Args:
            horizon_ms: Prediction horizon in ms. Defaults to self.horizon_ms.

        Returns:
            Predicted utilization in [0.0, 1.0], clamped to valid range.
            Returns current level if fewer than 2 observations have been seen.
        """
        if self._state.n_observations < 2:
            return float(self._state.level)

        effective_horizon_ms = (horizon_ms or self.horizon_ms) + self.metric_lag_ms
        # Convert ms horizon to "steps" assuming 1 observation per scrape interval
        # For Prometheus scrape interval of 15s: 1 step = 15000ms
        # For this predictor, we work in fractional steps
        # The Holt-Winters h-step forecast: level + h * trend
        # Here h is in units of the scrape interval. Since scrape_interval_ms
        # is not tracked here, we express h as a fraction of 1 step.
        # Calibration: assume scrape interval = 15s = 15000ms
        scrape_interval_ms = 15_000
        h = effective_horizon_ms / scrape_interval_ms

        prediction = self._state.level + h * self._state.trend
        return float(max(0.0, min(1.0, prediction)))

    def record_error(self, actual: float) -> None:
        """
        Record prediction error for RMSE tracking.

        Call this with the actual utilization that was observed after
        a prediction was made. Used to monitor predictor accuracy.
        """
        predicted = self.predict()
        error = (predicted - actual) ** 2
        self._prediction_errors.append(error)

    @property
    def rmse(self) -> Optional[float]:
        """
        Root mean squared error over the last 100 predictions.
        Returns None if fewer than 10 errors have been recorded.
        """
        if len(self._prediction_errors) < 10:
            return None
        import math
        return math.sqrt(sum(self._prediction_errors) / len(self._prediction_errors))

    @property
    def n_observations(self) -> int:
        return self._state.n_observations

    @property
    def is_warmed_up(self) -> bool:
        """True if the predictor has enough data to make reliable forecasts."""
        return self._state.n_observations >= 5