"""
Monte Carlo validation of the actual "doubly robust" claim: that
estimate_ate stays approximately unbiased for the true average
treatment effect when EITHER the propensity model or the outcome
model is misspecified, not just when both are correct.

WHY THIS IS SEPARATE FROM test_estimator.py
================================================
test_estimator.py checks the AIPW formula computes correctly against
hand-derived values, with correctly-specified models throughout. That
establishes the arithmetic is right. It does NOT establish the
"doubly robust" property, which is specifically about what happens
when one of the two models is wrong. This is the estimator.py
equivalent of test_msprt_type1_error.py: a formula matching its
derivation is necessary but not sufficient evidence the estimator's
actual statistical guarantee holds.

METHOD
======
Simulates a population with a KNOWN true average treatment effect
(TRUE_ATE), where treatment assignment genuinely depends on covariates
(real confounding, not random assignment, since if assignment were
random a naive difference-in-means would already be unbiased and this
test would not distinguish AIPW from a much simpler estimator). Runs
estimate_ate() under three model-specification scenarios:

1. BOTH models correctly specified: the baseline case, both mechanisms
   should work together correctly.
2. Propensity model correct, outcome model deliberately WRONG (a
   constant, ignoring the true covariate-dependent outcome entirely):
   tests the "outcome model can be wrong" half of double robustness.
3. Outcome model correct, propensity model deliberately WRONG (a
   constant 0.5, ignoring the true covariate-dependent treatment
   assignment mechanism): tests the "propensity model can be wrong"
   half.

In all three cases, the estimate should be close to TRUE_ATE (checked
against a confidence interval derived from repeated simulation, not a
single point estimate, since a single simulation's estimate has
genuine sampling variance even from a correctly-specified estimator).
A FOURTH scenario, both models wrong, is included specifically to
confirm this test actually has power to detect bias when the theory
predicts it should exist, the same role
test_large_consistent_effect_crosses_rejection_threshold plays in
test_msprt.py.
"""

from __future__ import annotations

import random

import pytest

from stratum_experiment.estimator import estimate_ate

TRUE_ATE = 2.0
N_UNITS_PER_SIM = 2000
N_REPETITIONS = 200
SEED = 20260812


def _generate_population(rng: random.Random, n: int):
    """Generates one simulated population with real confounding.

    x is a single covariate, uniform on [-1, 1]. True propensity is a
    logistic function of x (higher x -> more likely to be treated),
    genuine confounding. True outcome model: y = 1.0 + 3.0*x + t*TRUE_ATE
    + noise, so x affects both treatment assignment and outcome, the
    actual definition of a confounder, and TRUE_ATE is the true causal
    effect by construction.
    """
    xs, ts, ys = [], [], []
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        true_propensity = 1.0 / (1.0 + pow(2.718281828, -2.0 * x))  # logistic(2x)
        t = 1 if rng.random() < true_propensity else 0
        noise = rng.gauss(0.0, 1.0)
        y = 1.0 + 3.0 * x + t * TRUE_ATE + noise
        xs.append([x])
        ts.append(t)
        ys.append(y)
    return ts, ys, xs


def _true_propensity_model(x):
    return 1.0 / (1.0 + pow(2.718281828, -2.0 * x[0]))


def _true_outcome_model_treated(x):
    return 1.0 + 3.0 * x[0] + TRUE_ATE


def _true_outcome_model_control(x):
    return 1.0 + 3.0 * x[0]


def _wrong_constant_propensity(x):
    return 0.5  # ignores x entirely, wrong whenever true propensity depends on x


def _wrong_constant_outcome_treated(x):
    return 0.0  # ignores x entirely, wrong whenever true outcome depends on x


def _wrong_constant_outcome_control(x):
    return 0.0


def _run_repeated_estimates(propensity_model, outcome_treated, outcome_control) -> list[float]:
    rng = random.Random(SEED)
    estimates = []
    for _ in range(N_REPETITIONS):
        ts, ys, xs = _generate_population(rng, N_UNITS_PER_SIM)
        result = estimate_ate(ts, ys, xs, propensity_model, outcome_treated, outcome_control)
        estimates.append(result.ate)
    return estimates


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _stderr_of_mean(values: list[float]) -> float:
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return (variance / len(values)) ** 0.5


class TestDoubleRobustness:
    def test_both_models_correct_recovers_true_ate(self):
        estimates = _run_repeated_estimates(
            _true_propensity_model, _true_outcome_model_treated, _true_outcome_model_control
        )
        mean_estimate = _mean(estimates)
        tolerance = 4 * _stderr_of_mean(estimates)
        assert mean_estimate == pytest.approx(TRUE_ATE, abs=tolerance), (
            f"mean AIPW estimate {mean_estimate:.4f} across {N_REPETITIONS} "
            f"simulations, both models correctly specified, is outside "
            f"tolerance of the true ATE {TRUE_ATE}. Baseline case failing "
            f"means the core formula itself is wrong, investigate "
            f"estimate_ate directly before touching anything else."
        )

    def test_outcome_model_wrong_propensity_correct_still_recovers_true_ate(self):
        # THE key doubly-robust claim, half 1: outcome model is
        # deliberately, badly wrong (a constant, ignoring x entirely,
        # when the truth is strongly x-dependent), propensity model is
        # exactly correct. Should still be unbiased for TRUE_ATE.
        estimates = _run_repeated_estimates(
            _true_propensity_model, _wrong_constant_outcome_treated, _wrong_constant_outcome_control
        )
        mean_estimate = _mean(estimates)
        tolerance = 4 * _stderr_of_mean(estimates)
        assert mean_estimate == pytest.approx(TRUE_ATE, abs=tolerance), (
            f"mean AIPW estimate {mean_estimate:.4f} with a deliberately "
            f"WRONG outcome model but CORRECT propensity model is outside "
            f"tolerance of the true ATE {TRUE_ATE}. This is the actual "
            f"'doubly robust' claim failing: the estimator should stay "
            f"unbiased when only the outcome model is wrong, and it did "
            f"not. A real correctness failure in the correction term, "
            f"not a tolerance issue, investigate before loosening this."
        )

    def test_propensity_model_wrong_outcome_correct_still_recovers_true_ate(self):
        # The key doubly-robust claim, half 2: propensity model is
        # deliberately wrong (constant 0.5, ignoring the true
        # x-dependent assignment mechanism), outcome model is exactly
        # correct. Should still be unbiased for TRUE_ATE.
        estimates = _run_repeated_estimates(
            _wrong_constant_propensity, _true_outcome_model_treated, _true_outcome_model_control
        )
        mean_estimate = _mean(estimates)
        tolerance = 4 * _stderr_of_mean(estimates)
        assert mean_estimate == pytest.approx(TRUE_ATE, abs=tolerance), (
            f"mean AIPW estimate {mean_estimate:.4f} with a deliberately "
            f"WRONG propensity model but CORRECT outcome model is outside "
            f"tolerance of the true ATE {TRUE_ATE}. This is the other "
            f"half of the doubly robust claim failing, investigate before "
            f"loosening this."
        )

    def test_both_models_wrong_shows_meaningful_bias(self):
        # Confirms this test suite actually has power to detect bias
        # when theory predicts it should exist: with BOTH models wrong,
        # there is no remaining correction mechanism, and the estimate
        # should be noticeably biased away from TRUE_ATE. If this test
        # fails (estimate stays close to TRUE_ATE even with both models
        # wrong), it does not mean the estimator is unusually robust,
        # it means something is wrong with this test's ability to
        # detect bias at all, undermining the three tests above too.
        estimates = _run_repeated_estimates(
            _wrong_constant_propensity, _wrong_constant_outcome_treated, _wrong_constant_outcome_control
        )
        mean_estimate = _mean(estimates)
        bias = abs(mean_estimate - TRUE_ATE)
        assert bias > 0.3, (
            f"mean AIPW estimate {mean_estimate:.4f} with BOTH models "
            f"wrong shows less than the expected meaningful bias from "
            f"TRUE_ATE={TRUE_ATE} (bias={bias:.4f}). This test exists to "
            f"confirm the suite can detect bias when it should be "
            f"present, if it can't, the three doubly-robust tests above "
            f"are not actually confirming anything."
        )