"""
Monte Carlo validation that mSPRT's Type-I error rate matches its
claimed alpha, under the null hypothesis, across continuous
monitoring.

WHY THIS IS SEPARATE FROM test_msprt.py
============================================
test_msprt.py checks that the e_value formula is computed correctly,
by comparing against an independently hand-derived closed form. That
is necessary but explicitly not sufficient (see msprt.py's own module
docstring): a correctly-implemented formula still needs empirical
confirmation that the STOPPING RULE built on top of it (reject the
first time e_value crosses 1/alpha) actually produces false positives
at rate alpha, not some other rate, when there is truly no effect.
This is what Ville's inequality claims mathematically; this test
checks it holds in this specific implementation, at finite,
practically-relevant sample sizes, not just in the infinite-sample
limit the inequality technically covers.

METHOD
======
Simulates N_SIMULATIONS independent "experiments," each drawing
MAX_OBSERVATIONS_PER_SIM paired (treatment, control) observations from
IDENTICAL distributions (true effect = 0, i.e. the null hypothesis is
true by construction). For each simulated experiment, checks after
every observation whether should_reject_null has become True, and if
so, records that experiment as a false positive and stops early
(continuous monitoring, not fixed-horizon checking).

The fraction of simulated experiments that ever reject is the
empirical Type-I error rate. This must be close to alpha, not exactly
equal (this is a finite Monte Carlo sample, sampling variance is
expected), checked against a binomial confidence interval around alpha
rather than a hard equality.

SAMPLE SIZE JUSTIFICATION
=============================
5000 simulations per alpha level, 3000-step horizon (see
MAX_OBSERVATIONS_PER_SIM above for why 3000 specifically). At
alpha=0.05, the standard error of the empirical rate is
sqrt(0.05 * 0.95 / 5000) ~= 0.00308.

TOLERANCE IS NOT CENTERED ON ALPHA ITSELF
=============================================
An earlier version of this suite checked the empirical rate against
alpha directly (e.g. 0.05 +/- 3 standard errors) and failed at every
alpha level tested except 0.01. Investigated directly rather than
loosened blindly: Ville's inequality (see msprt.py's module docstring) proves
P(exists n such that e_n >= 1/alpha) <= alpha, an upper bound,
not an equality. Whether the true rate sits close to that ceiling or
meaningfully below it depends on the specific martingale's behavior
under the null. A diagnostic run at a 3000-step horizon (500
simulations, see diagnose_horizon.py) directly checked this
configuration's e_value trajectory at alpha=0.05: it decays on average
under the null (checkpoint trace at step 1: 0.90, step 500: 0.09, step
1000: 0.08, with real variance, not a clean monotone collapse, rising
back to 0.165 by step 2000), and the measured asymptotic rejection
rate was 18/500 = 0.036, meaningfully below the 0.05 ceiling. This is
valid, expected behavior for a mixture-prior test whose average
trajectory drifts downward under the null, not a defect.

alpha=0.10's expected_rate was initially left unverified (set equal to
alpha itself, with an explicit note that no diagnostic had been run
for it). The first full run of this suite after that change measured
it directly, at the real N_SIMULATIONS=5000 scale rather than a
separate smaller diagnostic: 0.0696. The parametrization below now
uses that measured value, closing the gap the original comment flagged
rather than leaving it open. alpha=0.01's expected_rate is still
alpha itself, unverified against a direct measurement, since that case
has passed at every tolerance tried so far and a placeholder that
keeps passing doesn't carry the same urgency as one that was actively
failing, noted here explicitly as a real, remaining gap, not
silently assumed resolved by the other two being fixed.
"""

from __future__ import annotations

import random

import pytest

from stratum_experiment.msprt import MSPRTConfig, MSPRTState

N_SIMULATIONS = 5000
# 3000, not 500: a diagnostic run (see diagnose_horizon.py, not part of
# this suite) at a 3000-step horizon found 18/500 = 0.036 true rejection
# rate under the null, versus only 12/500 = 0.024 when the same run is
# artificially cut off at 500 steps, matching this suite's original
# 500-step result of 0.0298 closely. One third of true rejections
# (6 of 18) happened after step 500, including one as late as step
# 2800, confirming the shorter horizon was genuinely truncating real
# rejections, not just adding noise. 3000 is not an arbitrary increase,
# it is the horizon that diagnostic run was actually measured at.
MAX_OBSERVATIONS_PER_SIM = 3000

# Fixed seed: this test's entire point is confirming a specific
# implementation's behavior is stable and correct, not exploring
# random variation across runs. A fixed seed makes a failure
# reproducible and debuggable; an unseeded run would make "it failed
# once" impossible to investigate against the same data that produced
# the failure.
RANDOM_SEED = 20260811


def _run_one_null_simulation(
    config: MSPRTConfig, rng: random.Random, max_observations: int
) -> bool:
    """Runs one simulated experiment under the null (true effect = 0),
    checking should_reject_null after every observation. Returns True
    if the null was ever rejected (a false positive), False if it
    survived all max_observations without rejecting.

    Both treatment and control are drawn from the SAME distribution
    (mean 0, variance config.sigma_squared / 2 each, so their
    difference has variance config.sigma_squared, matching what the
    test's config declares as the known per-observation-difference
    variance), this is what "the null hypothesis is true" means
    concretely: no systematic difference between the two arms.
    """
    state = MSPRTState(config=config)
    per_arm_sd = (config.sigma_squared / 2.0) ** 0.5
    for _ in range(max_observations):
        treatment = rng.gauss(0.0, per_arm_sd)
        control = rng.gauss(0.0, per_arm_sd)
        state.update(treatment, control)
        if state.should_reject_null:
            return True
    return False


def _empirical_type1_rate(
    alpha: float, sigma_squared: float, tau_squared: float, n_simulations: int
) -> float:
    config = MSPRTConfig(alpha=alpha, sigma_squared=sigma_squared, tau_squared=tau_squared)
    rng = random.Random(RANDOM_SEED)
    false_positives = sum(
        1
        for _ in range(n_simulations)
        if _run_one_null_simulation(config, rng, MAX_OBSERVATIONS_PER_SIM)
    )
    return false_positives / n_simulations


class TestMSPRTType1ErrorControl:
    @pytest.mark.parametrize(
        "alpha,ceiling,expected_rate,tolerance",
        [
            # ceiling is the Ville's-inequality upper bound (alpha
            # itself). expected_rate is this configuration's actual
            # measured asymptotic rate at a 3000-step horizon, always
            # <= ceiling, per the module docstring's diagnostic run
            # summary. tolerance is centered on expected_rate, roughly
            # 3 standard errors at N_SIMULATIONS=5000, not on ceiling.
            # Only alpha=0.05 has a directly diagnosed expected_rate
            # (0.036, see module docstring); 0.01 and 0.10 use alpha
            # itself as expected_rate since no separate diagnostic run
            # was done for those configurations and the original
            # 0.01 case already passed against alpha directly, this is
            # noted as a real gap, not silently assumed identical
            # behavior, see the follow-up item this leaves open.
            (0.05, 0.05, 0.036, 0.0110),
            (0.01, 0.01, 0.01, 0.0080),
            # 0.10's expected_rate was originally left as alpha itself,
            # explicitly flagged at the time as unverified, no
            # diagnostic run had actually been done for this
            # configuration. The full test run that carried that
            # placeholder (see commit b9e13ad) measured the real rate
            # directly: 0.0696 at N_SIMULATIONS=5000, 3000-step
            # horizon. Replacing the placeholder with that measurement
            # rather than continuing to guess.
            (0.10, 0.10, 0.0696, 0.0130),
        ],
    )
    def test_empirical_type1_rate_stays_at_or_below_the_ville_ceiling(
        self, alpha, ceiling, expected_rate, tolerance
    ):
        empirical_rate = _empirical_type1_rate(
            alpha=alpha, sigma_squared=1.0, tau_squared=0.25, n_simulations=N_SIMULATIONS
        )
        # The hard, non-negotiable check: Ville's inequality is a proven
        # upper bound, this must never be exceeded by more than sampling
        # noise, regardless of where the true rate happens to sit below
        # it. A meaningful violation here is a real correctness failure.
        assert empirical_rate <= ceiling + tolerance, (
            f"empirical Type-I error rate {empirical_rate:.4f} exceeds "
            f"the Ville's-inequality ceiling of alpha={ceiling} by more "
            f"than sampling tolerance. This is the actual Type-I error "
            f"guarantee being violated, a real correctness failure, "
            f"investigate the stopping rule immediately, do not loosen "
            f"this bound."
        )
        # The softer, informational check: confirms the rate is in the
        # ballpark of what was directly measured for this configuration,
        # to catch a large unexplained shift (e.g. a future code change
        # that alters the martingale's typical trajectory) even though
        # it isn't itself a violation of the formal guarantee.
        assert empirical_rate == pytest.approx(expected_rate, abs=tolerance), (
            f"empirical Type-I error rate {empirical_rate:.4f} at "
            f"alpha={alpha} differs meaningfully from this "
            f"configuration's previously measured rate of "
            f"{expected_rate}. Below the Ville ceiling, so not a formal "
            f"guarantee violation, but a large unexplained shift from a "
            f"directly diagnosed baseline is still worth investigating "
            f"before assuming it's fine."
        )

    def test_empirical_type1_rate_stays_below_ville_ceiling_with_generous_tau(self):
        # A larger tau_squared (more spread-out prior on effect size)
        # means the test is more willing to accumulate evidence for
        # small true effects, which could plausibly (if the
        # implementation were wrong) inflate the false-positive rate
        # under continuous monitoring, since there's "more room" for
        # noise to look like a plausible small effect. Checked
        # separately from the main tau_squared=0.25 sweep above.
        #
        # No separate diagnostic run was done for tau_squared=4.0
        # specifically (see the alpha=0.05, tau=0.25 case above for
        # the one configuration actually diagnosed at a 3000-step
        # horizon). Checking only the hard Ville ceiling here, not a
        # tight expected-rate band, since tightening further without a
        # direct measurement at this tau value would be asserting
        # precision this suite doesn't actually have evidence for,
        # see the follow-up item this leaves open.
        empirical_rate = _empirical_type1_rate(
            alpha=0.05, sigma_squared=1.0, tau_squared=4.0, n_simulations=N_SIMULATIONS
        )
        assert empirical_rate <= 0.05 + 0.0110, (
            f"empirical Type-I error rate {empirical_rate:.4f} at "
            f"tau_squared=4.0 exceeds the Ville's-inequality ceiling by "
            f"more than sampling tolerance, a real correctness failure."
        )