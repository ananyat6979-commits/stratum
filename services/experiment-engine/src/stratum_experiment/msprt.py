"""
Mixture Sequential Probability Ratio Test (mSPRT) core.

WHAT THIS IS
============
A sequential hypothesis test that can be evaluated after every new
observation, rather than requiring a fixed sample size decided in
advance, while still controlling the Type-I error rate (false positive
rate) at a stated alpha. This is the property a fixed-horizon t-test
does NOT have: peeking at a fixed-horizon test's p-value repeatedly as
data arrives and stopping the moment it crosses 0.05 inflates the true
false-positive rate well above 0.05, a well-known and well-documented
failure mode. mSPRT is specifically designed to be safe to check after
every observation.

THE MATH, STATED PRECISELY
=============================
Given paired observations (treatment_i, control_i) for i = 1..n, define
the per-observation log-likelihood-ratio under a normal mixture prior
over the true effect size theta ~ N(0, tau^2):

    e_n = product over i=1..n of the mixture likelihood ratio

Under the null hypothesis (true effect = 0), e_n is a nonnegative
martingale with E[e_n] = 1 for all n. This is the property that makes
the test valid at any stopping time: by Ville's inequality, for any
alpha,

    P(exists n such that e_n >= 1/alpha) <= alpha

So "reject the null the first time e_n >= 1/alpha" controls the
Type-I error rate at alpha, regardless of when that crossing happens
or how the stopping decision was made, which is exactly the property a
fixed-horizon test lacks under repeated peeking.

This implementation uses the standard closed-form mixture for a known
observation variance (see Johari, Pekelis, Walsh 2015 for the derivation
this follows, and Robbins 1970 for the underlying martingale argument):
given per-observation difference d_i = treatment_i - control_i with
known variance sigma^2, and prior variance tau^2 on the true effect,

    Lambda_n = sqrt(sigma^2 / (sigma^2 + n*tau^2)) *
               exp( (n^2 * tau^2 * mean(d)^2) / (2*sigma^2*(sigma^2 + n*tau^2)) )

is the mixture likelihood ratio at n observations. This module computes
Lambda_n incrementally from a running sum and sum-of-squares, not by
recomputing from the full history on every call, since a real caller
evaluates this after every new observation and O(n) recomputation per
step would make the whole test O(n^2).

WHAT THIS MODULE DELIBERATELY DOES NOT DO
=============================================
- Does not estimate sigma^2 from data. Takes it as a required parameter.
  Estimating a nuisance variance online while also testing is a real and
  harder problem (see Waudby-Smith & Ramdas 2020 for a treatment that
  doesn't assume known variance); this implementation is honest about
  that limitation rather than silently plugging in a sample variance and
  losing the Type-I error guarantee without saying so.
- Does not choose tau^2 for you. tau^2 encodes a prior belief about
  plausible effect sizes; too large wastes power on effect sizes never
  observed in practice, too small under-detects real large effects. This
  module exposes it as a required parameter with worked guidance in the
  MSPRTConfig docstring, not a silent default.
- Does not itself claim Type-I error control is correct. That claim is
  validated separately, empirically, by Monte Carlo simulation, see
  the companion validation module this is designed to be checked
  against, not yet written as of this file's initial commit (see
  docs/SCOPE.md for status). A formula matching a paper's derivation is
  necessary but not sufficient evidence it's implemented correctly;
  only a simulation that empirically confirms the false-positive rate
  matches the claimed alpha closes that gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MSPRTConfig:
    """Configuration for one mSPRT sequential test.

    alpha: target Type-I error rate (false-positive rate). Standard
        choices are 0.05 or 0.01, same as a fixed-horizon test's
        significance level, but here it holds under continuous
        monitoring, not just at one fixed sample size.

    sigma_squared: the KNOWN (not estimated) variance of a single
        paired difference d_i = treatment_i - control_i. If this is
        wrong, the Type-I error guarantee is invalid, not approximately
        valid, invalid. In practice this is usually estimated from a
        prior, separate dataset (e.g. historical A/A test variance, or
        pre-experiment data), never from the same stream being tested,
        since using the test data itself to estimate its own variance
        reintroduces exactly the kind of statistical leakage mSPRT
        exists to avoid elsewhere.

    tau_squared: the prior variance on the true effect size. Encodes
        "how large an effect do I actually expect to see, if there is
        one." A reasonable starting point: if you'd consider an effect
        of size delta meaningful and plausible, set tau_squared such
        that delta is within roughly 1-2 standard deviations of the
        prior, i.e. tau_squared on the order of delta^2 to (delta/2)^2.
        Too small relative to the true effect loses power to detect it;
        too large relative to typical effect sizes wastes power on
        implausibly large effects that rarely occur. This is a real
        modeling choice, not a technicality, and should be set before
        looking at the data being tested, same reasoning as
        sigma_squared above.
    """

    alpha: float
    sigma_squared: float
    tau_squared: float

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.sigma_squared <= 0.0:
            raise ValueError(f"sigma_squared must be positive, got {self.sigma_squared}")
        if self.tau_squared <= 0.0:
            raise ValueError(f"tau_squared must be positive, got {self.tau_squared}")

    @property
    def rejection_threshold(self) -> float:
        """The e-value threshold 1/alpha above which the null is rejected.

        Derived directly from Ville's inequality (see module docstring):
        P(exists n such that e_n >= 1/alpha) <= alpha under the null,
        which is exactly the guarantee that makes "reject the first time
        e_n crosses this threshold" valid at any stopping time.
        """
        return 1.0 / self.alpha


@dataclass
class MSPRTState:
    """Running state for one sequential test, updated incrementally.

    Holds only the two sufficient statistics needed to compute the
    mixture likelihood ratio at any n (running sum and count), not the
    full observation history, so update() is O(1) per observation and
    the whole test is O(n) total across n observations, not O(n^2).
    """

    n: int = 0
    sum_d: float = 0.0
    config: MSPRTConfig = field(repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            raise ValueError("MSPRTState requires a config")

    def update(self, treatment_value: float, control_value: float) -> float:
        """Records one new paired observation, returns the updated e-value.

        d = treatment_value - control_value is the per-observation
        difference this test accumulates evidence about. Returns the
        mixture likelihood ratio Lambda_n after including this
        observation, which the caller compares against
        config.rejection_threshold to decide whether to stop.
        """
        d = treatment_value - control_value
        self.n += 1
        self.sum_d += d
        return self.e_value

    @property
    def e_value(self) -> float:
        """The current mixture likelihood ratio Lambda_n.

        Recomputed from the running sufficient statistics (n, sum_d) on
        every access rather than cached, since it's a cheap closed-form
        expression (a handful of arithmetic operations), not worth the
        complexity of cache invalidation for the marginal cost saved.

        Formula, restated from the module docstring with sigma^2 and
        tau^2 abbreviated s2 and t2:

            Lambda_n = sqrt(s2 / (s2 + n*t2)) *
                       exp( n^2 * t2 * mean_d^2 / (2 * s2 * (s2 + n*t2)) )

        where mean_d = sum_d / n. At n=0 (no observations yet), returns
        1.0, the correct starting value for a martingale with E[e_0] = 1.
        """
        if self.n == 0:
            return 1.0

        s2 = self.config.sigma_squared
        t2 = self.config.tau_squared
        n = self.n
        mean_d = self.sum_d / n

        denominator = s2 + n * t2
        sqrt_term = math.sqrt(s2 / denominator)
        exponent = (n**2 * t2 * mean_d**2) / (2.0 * s2 * denominator)

        # Guard against overflow in exp() for pathologically large
        # evidence (a huge true effect, or a huge n): exponent growing
        # large enough for exp() to overflow means the test has already
        # rejected many observations ago (e_value would already exceed
        # any reasonable rejection_threshold), so capping at a value
        # that comfortably exceeds any realistic 1/alpha is correct
        # behavior, not an approximation that changes the test's
        # conclusion.
        exponent = min(exponent, 700.0)  # exp(700) is already ~1e304, past any realistic 1/alpha

        return sqrt_term * math.exp(exponent)

    @property
    def should_reject_null(self) -> bool:
        """True once e_value has crossed config.rejection_threshold.

        Per Ville's inequality (see module docstring), stopping and
        rejecting the null the first time this becomes True controls
        the Type-I error rate at config.alpha, regardless of when it
        happens or how long the test has been running, which is the
        entire point of using a sequential test instead of a
        fixed-horizon one.
        """
        return self.e_value >= self.config.rejection_threshold

    @property
    def mean_difference(self) -> float:
        """The current running mean of treatment - control, for reporting.

        Not itself part of the hypothesis test (the e_value is), this is
        the effect-size estimate a caller would want to report alongside
        should_reject_null, e.g. "rejected the null after 340
        observations, mean difference 12.3ms."
        """
        if self.n == 0:
            return 0.0
        return self.sum_d / self.n