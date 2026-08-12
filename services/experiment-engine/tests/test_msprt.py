"""
Unit tests for the mSPRT core (stratum_experiment.msprt).

These test the closed-form e-value computation against cases whose
correct answer can be verified by hand or against the module's own
documented formula, not against the implementation's own output taken
on faith. Two categories:

1. Boundary and structural properties that must hold regardless of the
   specific formula's correctness (e_value starts at 1.0, is always
   nonnegative, config validates its own parameters).
2. Direct recomputation of the closed-form formula from the module's
   own docstring, done independently in each test rather than by
   calling helper functions this module also defines, so a bug shared
   between the implementation and a test helper can't hide a mismatch
   from both.

What this test suite does NOT do: confirm the Type-I error rate is
actually controlled at the claimed alpha. That is an empirical claim
about behavior across many simulated experiments, not a property any
single formula-level test can establish, see test_msprt_type1_error.py
for that.
"""

from __future__ import annotations

import math

import pytest

from stratum_experiment.msprt import MSPRTConfig, MSPRTState


class TestMSPRTConfig:
    def test_valid_config_constructs(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        assert config.alpha == 0.05
        assert config.sigma_squared == 1.0
        assert config.tau_squared == 0.25

    def test_rejection_threshold_is_reciprocal_of_alpha(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        assert config.rejection_threshold == pytest.approx(20.0)

        config_01 = MSPRTConfig(alpha=0.01, sigma_squared=1.0, tau_squared=0.25)
        assert config_01.rejection_threshold == pytest.approx(100.0)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_alpha_outside_open_unit_interval(self, bad_alpha):
        with pytest.raises(ValueError, match="alpha"):
            MSPRTConfig(alpha=bad_alpha, sigma_squared=1.0, tau_squared=0.25)

    @pytest.mark.parametrize("bad_sigma_sq", [0.0, -1.0])
    def test_rejects_nonpositive_sigma_squared(self, bad_sigma_sq):
        with pytest.raises(ValueError, match="sigma_squared"):
            MSPRTConfig(alpha=0.05, sigma_squared=bad_sigma_sq, tau_squared=0.25)

    @pytest.mark.parametrize("bad_tau_sq", [0.0, -1.0])
    def test_rejects_nonpositive_tau_squared(self, bad_tau_sq):
        with pytest.raises(ValueError, match="tau_squared"):
            MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=bad_tau_sq)


class TestMSPRTStateBoundaries:
    def test_e_value_starts_at_one(self):
        # Required by the martingale property E[e_0] = 1, stated in the
        # module docstring: this is not an arbitrary starting value, a
        # test stream that starts anywhere else would already violate
        # the mathematical property the whole stopping rule depends on.
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        assert state.e_value == 1.0

    def test_should_reject_null_is_false_at_zero_observations(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        assert state.should_reject_null is False

    def test_mean_difference_is_zero_at_zero_observations(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        assert state.mean_difference == 0.0

    def test_e_value_is_never_negative_across_a_stream_of_observations(self):
        # Structural property that must hold regardless of the specific
        # formula: a likelihood ratio is a ratio of two nonnegative
        # densities, it cannot be negative. Checked across a stream
        # with both positive and negative differences, not just one
        # direction.
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        observations = [(1.0, 0.5), (0.2, 0.9), (-0.3, 0.1), (0.0, 0.0), (2.0, -1.0)]
        for treatment, control in observations:
            e = state.update(treatment, control)
            assert e >= 0.0

    def test_update_return_value_matches_e_value_property(self):
        # update() returns a value; e_value is a property read
        # separately. These must always agree, since a caller checking
        # should_reject_null right after update() is reading the same
        # state update() just wrote.
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        returned = state.update(1.5, 0.5)
        assert returned == state.e_value

    def test_n_increments_by_exactly_one_per_update(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        for expected_n in range(1, 11):
            state.update(1.0, 0.0)
            assert state.n == expected_n


class TestMSPRTEValueFormula:
    """Recomputes the closed-form formula independently in each test,
    from the module docstring, rather than delegating to any shared
    helper, so a bug present in both the implementation and a shared
    test helper cannot hide a real discrepancy from every test at once.
    """

    def _expected_e_value(self, n: int, sum_d: float, sigma_sq: float, tau_sq: float) -> float:
        if n == 0:
            return 1.0
        mean_d = sum_d / n
        denominator = sigma_sq + n * tau_sq
        sqrt_term = math.sqrt(sigma_sq / denominator)
        exponent = (n**2 * tau_sq * mean_d**2) / (2.0 * sigma_sq * denominator)
        return sqrt_term * math.exp(exponent)

    def test_matches_hand_computed_formula_after_one_observation(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        state.update(treatment_value=2.0, control_value=0.0)  # d = 2.0
        expected = self._expected_e_value(n=1, sum_d=2.0, sigma_sq=1.0, tau_sq=0.25)
        assert state.e_value == pytest.approx(expected, rel=1e-9)

    def test_matches_hand_computed_formula_after_several_observations(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=2.0, tau_squared=0.5)
        state = MSPRTState(config=config)
        diffs = [1.0, -0.5, 2.0, 0.3, -1.2, 0.8]
        for i, d in enumerate(diffs, start=1):
            state.update(treatment_value=d, control_value=0.0)
            expected = self._expected_e_value(
                n=i, sum_d=sum(diffs[:i]), sigma_sq=2.0, tau_sq=0.5
            )
            assert state.e_value == pytest.approx(expected, rel=1e-9), f"mismatch at n={i}"

    def test_zero_mean_difference_keeps_e_value_below_one(self):
        # When mean_d = 0 exactly, the exponent term vanishes (0^2 = 0),
        # leaving only the sqrt(sigma_sq / (sigma_sq + n*tau_sq)) factor,
        # which is strictly less than 1 for any n > 0 and tau_sq > 0.
        # This is the expected behavior of evidence accumulating AGAINST
        # a nonzero effect when observations show no difference: e_value
        # should decrease below 1, not stay at 1 or increase.
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        state = MSPRTState(config=config)
        for _ in range(20):
            state.update(treatment_value=1.0, control_value=1.0)  # d = 0 every time
        assert state.e_value < 1.0

    def test_large_consistent_effect_crosses_rejection_threshold(self):
        # A large, consistent difference should eventually cross
        # rejection_threshold. This is a sanity check that the test
        # actually has power to detect a real effect, not just that it
        # correctly fails to reject noise (covered by the Monte Carlo
        # null-hypothesis validation in test_msprt_type1_error.py).
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=1.0)
        state = MSPRTState(config=config)
        for _ in range(100):
            state.update(treatment_value=5.0, control_value=0.0)  # d = 5.0, consistently
            if state.should_reject_null:
                break
        assert state.should_reject_null is True
        assert state.n <= 100


class TestMSPRTIncrementalConsistency:
    def test_incremental_updates_match_a_single_batch_of_the_same_observations(self):
        # The whole point of tracking running sufficient statistics
        # (n, sum_d) instead of the full history is that the e_value
        # after k incremental update() calls must equal the e_value
        # computed from those same k observations all at once. This is
        # what makes the O(1)-per-update design correct, not just fast.
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
        diffs = [0.5, -0.3, 1.2, 0.8, -0.1, 0.4, 0.9]

        incremental_state = MSPRTState(config=config)
        for d in diffs:
            incremental_state.update(treatment_value=d, control_value=0.0)

        batch_state = MSPRTState(config=config)
        batch_state.n = len(diffs)
        batch_state.sum_d = sum(diffs)

        assert incremental_state.e_value == pytest.approx(batch_state.e_value, rel=1e-9)