"""
Caller-facing API for running a named mSPRT sequential experiment.

WHAT THIS ADDS OVER msprt.py DIRECTLY
==========================================
MSPRTConfig/MSPRTState (msprt.py) are the validated mathematical core:
correct e-value computation, correct Type-I error control, confirmed
by the test suite in tests/test_msprt.py and
tests/test_msprt_type1_error.py. Neither of those types knows anything
about what experiment they belong to, when it started, or how to
report a result to something outside a Python process holding a
reference to the MSPRTState object directly.

Experiment wraps MSPRTState with exactly three things a real caller
needs and the core deliberately does not provide: a name and start
timestamp for identifying which experiment a result belongs to, a
narrower record_observation() method that is the only way to feed data
in (rather than exposing update() directly, which returns a bare float
a caller has no reason to consume), and a to_dict() method producing a
plain, JSON-serializable summary suitable for logging, an API response,
or a dashboard, rather than requiring every caller to know
MSPRTState's internal field names.

This module adds no new statistical logic and makes no new claims
about correctness beyond what msprt.py's own test suite already
established. It is a usability layer, not a mathematical one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from stratum_experiment.msprt import MSPRTConfig, MSPRTState


@dataclass
class ExperimentResult:
    """Plain, JSON-serializable snapshot of one experiment's current
    state. Returned by Experiment.result(), not constructed directly
    by callers.
    """

    name: str
    n_observations: int
    mean_difference: float
    e_value: float
    rejection_threshold: float
    should_reject_null: bool
    alpha: float
    started_at: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        """Plain dict of JSON-serializable primitives (str, int, float,
        bool only), safe to pass directly to json.dumps or an API
        response body without further conversion.
        """
        return {
            "name": self.name,
            "n_observations": self.n_observations,
            "mean_difference": self.mean_difference,
            "e_value": self.e_value,
            "rejection_threshold": self.rejection_threshold,
            "should_reject_null": self.should_reject_null,
            "alpha": self.alpha,
            "started_at": self.started_at,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class Experiment:
    """A named, running mSPRT sequential test.

    Wraps MSPRTConfig/MSPRTState (see msprt.py) with identity (name,
    start time) and a narrow, intention-revealing interface
    (record_observation, result) instead of exposing MSPRTState's
    update()/e_value/should_reject_null directly. The underlying
    MSPRTState is still reachable via the `state` field for any caller
    that genuinely needs it, this class does not hide it, it just
    doesn't require going through it for the common case.

    Example:
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        exp = Experiment(name="routing_strategy_v2", config=config)
        exp.record_observation(treatment_latency_ms, control_latency_ms)
        result = exp.result()
        if result.should_reject_null:
            print(f"{exp.name}: significant at n={result.n_observations}")
    """

    name: str
    config: MSPRTConfig
    state: MSPRTState = field(init=False)
    started_at: float = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Experiment name must be a non-empty string")
        self.state = MSPRTState(config=self.config)
        self.started_at = time.time()

    def record_observation(self, treatment_value: float, control_value: float) -> None:
        """Records one new paired observation.

        Deliberately does not return the new e_value (unlike
        MSPRTState.update(), which does), since a caller of this
        higher-level API is expected to check result().should_reject_null
        when it wants to know the current state, not thread a return
        value from every individual call through their own code. This
        is the narrowing this class exists to provide.
        """
        self.state.update(treatment_value, control_value)

    def result(self) -> ExperimentResult:
        """Current snapshot of this experiment's state, safe to call at
        any point, including with zero observations recorded so far
        (n_observations=0, e_value=1.0, should_reject_null=False, per
        MSPRTState's own documented zero-observation behavior).
        """
        return ExperimentResult(
            name=self.name,
            n_observations=self.state.n,
            mean_difference=self.state.mean_difference,
            e_value=self.state.e_value,
            rejection_threshold=self.config.rejection_threshold,
            should_reject_null=self.state.should_reject_null,
            alpha=self.config.alpha,
            started_at=self.started_at,
            elapsed_seconds=time.time() - self.started_at,
        )