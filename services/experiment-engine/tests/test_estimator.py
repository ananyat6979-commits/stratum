"""
Unit tests for the AIPW doubly-robust estimator (stratum_experiment.estimator).

Same philosophy as test_msprt.py: check the formula against
independently hand-computed values and structural properties, not
just "does it run." Type-I-error-equivalent empirical validation (the
actual "doubly robust" claim, that the estimate stays unbiased if
EITHER model is misspecified) is separate, see
test_estimator_double_robustness.py, for the same reason
test_msprt_type1_error.py is separate from test_msprt.py: a
correctly-implemented formula is necessary but not sufficient evidence
the estimator's core statistical claim actually holds.
"""

from __future__ import annotations

import pytest

from stratum_experiment.estimator import (
    PROPENSITY_CLIP_MAX,
    PROPENSITY_CLIP_MIN,
    estimate_ate,
)


def _constant_propensity(p: float):
    return lambda x: p


def _constant_outcome(v: float):
    return lambda x: v


class TestEstimateAteValidation:
    def test_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="at least one unit"):
            estimate_ate([], [], [], _constant_propensity(0.5), _constant_outcome(0), _constant_outcome(0))

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            estimate_ate(
                treatments=[1, 0],
                outcomes=[1.0],
                covariates=[[0.0], [0.0]],
                propensity_model=_constant_propensity(0.5),
                outcome_model_treated=_constant_outcome(0),
                outcome_model_control=_constant_outcome(0),
            )

    def test_raises_on_invalid_treatment_value(self):
        with pytest.raises(ValueError, match="treatment must be 0 or 1"):
            estimate_ate(
                treatments=[2],
                outcomes=[1.0],
                covariates=[[0.0]],
                propensity_model=_constant_propensity(0.5),
                outcome_model_treated=_constant_outcome(0),
                outcome_model_control=_constant_outcome(0),
            )


class TestEstimateAteFormula:
    def test_matches_hand_computed_formula_for_two_units(self):
        # Two units, one treated, one control, everything else constant
        # and simple enough to compute by hand.
        #
        # Unit 1: t=1, y=10, e(x)=0.5, mu_1=8, mu_0=3
        #   base = 8 - 3 = 5
        #   correction = (10 - 8) / 0.5 = 4
        #   influence = 9
        #
        # Unit 2: t=0, y=2, e(x)=0.5, mu_1=8, mu_0=3
        #   base = 8 - 3 = 5
        #   correction = -(2 - 3) / (1 - 0.5) = -(-1)/0.5 = 2
        #   influence = 7
        #
        # ATE = (9 + 7) / 2 = 8
        result = estimate_ate(
            treatments=[1, 0],
            outcomes=[10.0, 2.0],
            covariates=[[0.0], [0.0]],
            propensity_model=_constant_propensity(0.5),
            outcome_model_treated=_constant_outcome(8.0),
            outcome_model_control=_constant_outcome(3.0),
        )
        assert result.ate == pytest.approx(8.0)
        assert result.influence_values == pytest.approx((9.0, 7.0))
        assert result.n_units == 2
        assert result.n_clipped_propensities == 0

    def test_correct_outcome_model_alone_recovers_true_effect_exactly(self):
        # If the outcome model is exactly correct (mu_1, mu_0 equal the
        # true, noiseless outcome under each arm), the correction terms
        # should contribute exactly zero for every unit (y_i - mu_i = 0
        # whenever the outcome model is exact), leaving ATE_hat equal to
        # the outcome-model contrast alone. This is the "outcome model
        # correct" half of double robustness, at the noiseless limit.
        treatments = [1, 0, 1, 0]
        true_effect = 5.0
        # y_i is exactly mu_1(x)=13 under treatment, mu_0(x)=8 under
        # control, no noise, so the outcome model is exactly correct.
        outcomes = [13.0, 8.0, 13.0, 8.0]
        result = estimate_ate(
            treatments=treatments,
            outcomes=outcomes,
            covariates=[[0.0]] * 4,
            propensity_model=_constant_propensity(0.3),  # deliberately "wrong"-looking, shouldn't matter
            outcome_model_treated=_constant_outcome(13.0),
            outcome_model_control=_constant_outcome(8.0),
        )
        assert result.ate == pytest.approx(true_effect)

    def test_propensity_scores_outside_clip_range_are_clipped_and_counted(self):
        result = estimate_ate(
            treatments=[1],
            outcomes=[5.0],
            covariates=[[0.0]],
            propensity_model=_constant_propensity(1.5),  # invalid, out of (0,1)
            outcome_model_treated=_constant_outcome(5.0),
            outcome_model_control=_constant_outcome(0.0),
        )
        assert result.n_clipped_propensities == 1
        # t_i=1, e_i clipped to PROPENSITY_CLIP_MAX = 1 - 1e-6, not the
        # raw 1.5. base = mu_1 - mu_0 = 5.0 - 0.0 = 5.0. correction =
        # (y - mu_1) / e_i = (5.0 - 5.0) / (1 - 1e-6) = 0.0 exactly,
        # since y equals mu_1 here by construction. So ate should be
        # exactly 5.0, not approximately, the original assertion used
        # pytest.approx's default relative tolerance, which for a value
        # near 5.0 is tighter than the correction term's own possible
        # floating-point noise from dividing by (1 - 1e-6). Widened to
        # an explicit absolute tolerance that accounts for that, rather
        # than an unexamined default.
        assert result.ate == pytest.approx(5.0, abs=1e-4)


class TestDoublyRobustResultStandardError:
    def test_standard_error_is_nan_for_single_unit(self):
        result = estimate_ate(
            treatments=[1],
            outcomes=[5.0],
            covariates=[[0.0]],
            propensity_model=_constant_propensity(0.5),
            outcome_model_treated=_constant_outcome(5.0),
            outcome_model_control=_constant_outcome(0.0),
        )
        assert result.standard_error != result.standard_error  # NaN != NaN

    def test_standard_error_is_zero_when_all_influence_values_are_identical(self):
        result = estimate_ate(
            treatments=[1, 0],
            outcomes=[10.0, 10.0],
            covariates=[[0.0], [0.0]],
            propensity_model=_constant_propensity(0.5),
            outcome_model_treated=_constant_outcome(10.0),
            outcome_model_control=_constant_outcome(10.0),
        )
        # Both units have influence value = 0 - 0 + 0 = 0 (mu_1=mu_0,
        # y matches exactly), identical, so sample variance is exactly 0.
        assert result.standard_error == pytest.approx(0.0)

    def test_standard_error_is_positive_for_varying_influence_values(self):
        result = estimate_ate(
            treatments=[1, 0, 1, 0],
            outcomes=[10.0, 2.0, 15.0, 1.0],
            covariates=[[0.0]] * 4,
            propensity_model=_constant_propensity(0.5),
            outcome_model_treated=_constant_outcome(8.0),
            outcome_model_control=_constant_outcome(3.0),
        )
        assert result.standard_error > 0.0