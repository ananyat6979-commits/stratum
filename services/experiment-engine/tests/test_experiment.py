"""
Tests for the Experiment wrapper (experiment.py).

These deliberately do NOT re-test mSPRT's statistical correctness or
Type-I error control, that's already covered by test_msprt.py and
test_msprt_type1_error.py, and re-testing it here through an extra
layer of indirection would just be slower, duplicate coverage. These
test only what this module adds: name validation, the
record_observation/result interface, to_dict's JSON-serializability,
and that results delegate correctly to the underlying MSPRTState
rather than computing anything independently (a delegation bug, e.g.
result() reading a stale cached value, would not be caught by
msprt.py's own tests at all, since those never go through this class).
"""

from __future__ import annotations

import json

import pytest

from stratum_experiment.experiment import Experiment, ExperimentResult
from stratum_experiment.msprt import MSPRTConfig, MSPRTState


def _config() -> MSPRTConfig:
    return MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)


class TestExperimentConstruction:
    def test_valid_name_constructs(self):
        exp = Experiment(name="test_experiment", config=_config())
        assert exp.name == "test_experiment"

    @pytest.mark.parametrize("bad_name", ["", "   ", "\t\n"])
    def test_rejects_empty_or_whitespace_only_name(self, bad_name):
        with pytest.raises(ValueError, match="name"):
            Experiment(name=bad_name, config=_config())

    def test_starts_with_a_fresh_msprt_state(self):
        exp = Experiment(name="test", config=_config())
        assert isinstance(exp.state, MSPRTState)
        assert exp.state.n == 0

    def test_started_at_is_set_at_construction(self):
        import time

        before = time.time()
        exp = Experiment(name="test", config=_config())
        after = time.time()
        assert before <= exp.started_at <= after


class TestExperimentRecordObservation:
    def test_record_observation_increments_underlying_state(self):
        exp = Experiment(name="test", config=_config())
        exp.record_observation(1.0, 0.5)
        assert exp.state.n == 1

    def test_record_observation_does_not_return_a_value(self):
        # Deliberate API choice, see experiment.py's docstring: this
        # narrows MSPRTState.update()'s return value away, callers of
        # this class check result() instead.
        exp = Experiment(name="test", config=_config())
        returned = exp.record_observation(1.0, 0.5)
        assert returned is None

    def test_multiple_observations_accumulate(self):
        exp = Experiment(name="test", config=_config())
        for _ in range(5):
            exp.record_observation(1.0, 0.0)
        assert exp.state.n == 5


class TestExperimentResult:
    def test_result_at_zero_observations_matches_msprt_state_defaults(self):
        exp = Experiment(name="test", config=_config())
        result = exp.result()
        assert result.n_observations == 0
        assert result.e_value == 1.0
        assert result.should_reject_null is False
        assert result.mean_difference == 0.0

    def test_result_delegates_to_underlying_state_not_a_stale_copy(self):
        # Guards against a real bug class: result() computing from
        # values captured once rather than reading the live state.
        exp = Experiment(name="test", config=_config())
        result_before = exp.result()
        exp.record_observation(5.0, 0.0)
        exp.record_observation(5.0, 0.0)
        result_after = exp.result()
        assert result_after.n_observations > result_before.n_observations
        assert result_after.e_value != result_before.e_value

    def test_result_reflects_should_reject_null_from_underlying_state(self):
        config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=1.0)
        exp = Experiment(name="test", config=config)
        for _ in range(100):
            exp.record_observation(5.0, 0.0)
            if exp.result().should_reject_null:
                break
        assert exp.result().should_reject_null is True

    def test_result_includes_config_alpha_and_rejection_threshold(self):
        config = MSPRTConfig(alpha=0.01, sigma_squared=1.0, tau_squared=0.25)
        exp = Experiment(name="test", config=config)
        result = exp.result()
        assert result.alpha == 0.01
        assert result.rejection_threshold == pytest.approx(100.0)

    def test_elapsed_seconds_is_nonnegative_and_increases(self):
        import time

        exp = Experiment(name="test", config=_config())
        result_1 = exp.result()
        time.sleep(0.05)
        result_2 = exp.result()
        assert result_1.elapsed_seconds >= 0.0
        assert result_2.elapsed_seconds > result_1.elapsed_seconds


class TestExperimentResultToDict:
    def test_to_dict_returns_all_fields(self):
        exp = Experiment(name="test", config=_config())
        exp.record_observation(1.0, 0.0)
        d = exp.result().to_dict()
        expected_keys = {
            "name", "n_observations", "mean_difference", "e_value",
            "rejection_threshold", "should_reject_null", "alpha",
            "started_at", "elapsed_seconds",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_is_actually_json_serializable(self):
        # The real point of to_dict: not just "returns a dict" but
        # "returns a dict json.dumps will not choke on," since the
        # whole reason this exists is API responses and logging.
        exp = Experiment(name="test", config=_config())
        exp.record_observation(1.0, 0.0)
        d = exp.result().to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["name"] == "test"
        assert deserialized["n_observations"] == 1

    def test_to_dict_name_matches_experiment_name(self):
        exp = Experiment(name="my_special_experiment", config=_config())
        assert exp.result().to_dict()["name"] == "my_special_experiment"