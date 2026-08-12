"""
Doubly-robust (augmented inverse propensity weighting, AIPW) estimator
for the average treatment effect (ATE) from observational data.

WHAT THIS IS
============
Given a dataset of units, each with a treatment assignment (1 or 0,
not necessarily randomized), an observed outcome, and a vector of
covariates, this estimates the causal effect of treatment on outcome,
correcting for the fact that treatment assignment may have depended on
those same covariates (confounding), which a naive difference-in-means
would not correct for.

THE ESTIMATOR, STATED PRECISELY
====================================
For unit i with covariates x_i, treatment t_i in {0, 1}, and observed
outcome y_i, define:
  - e(x_i): the propensity score, P(T=1 | X=x_i), estimated by a
    propensity model.
  - mu_1(x_i), mu_0(x_i): the expected outcome under treatment and
    control respectively, estimated by an outcome model.

The AIPW estimate of the average treatment effect is:

    ATE_hat = (1/n) * sum_i [
        (mu_1(x_i) - mu_0(x_i))
        + t_i * (y_i - mu_1(x_i)) / e(x_i)
        - (1 - t_i) * (y_i - mu_0(x_i)) / (1 - e(x_i))
    ]

This is "doubly robust" in a specific, checkable sense: if e(x) is
correctly specified but mu_1/mu_0 are not, the estimate is still
consistent (the correction terms fix the outcome model's bias). If
mu_1/mu_0 are correctly specified but e(x) is not, the estimate is
still consistent (the outcome model alone would already be unbiased,
the propensity-weighted correction terms have expectation zero). This
module's Monte Carlo validation (see the companion test file) checks
BOTH of these cases independently, deliberately misspecifying one
model while keeping the other correct, since checking only the
both-correct case would not actually test the "doubly" part of doubly
robust at all.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
=============================================
- Does not choose or fit the propensity/outcome models. Takes them as
  required, pre-fitted callables. Model selection (logistic regression
  vs. a tree-based model vs. anything else) is a separate, real
  modeling decision this module is agnostic to, deliberately, the same
  way msprt.py does not choose sigma_squared or tau_squared for the
  caller.
- Does not handle propensity scores at or near 0 or 1 gracefully
  beyond a hard validation check. A propensity score near the
  boundary means a unit that (near-)deterministically did or did not
  receive treatment, and dividing by e(x) or (1-e(x)) near zero
  explodes the variance of the estimate, a well-known, real failure
  mode of inverse-propensity-weighted estimators (see Robins,
  Rotnitzky, Zhao 1994 for the original AIPW derivation this follows).
  This module raises rather than silently producing an unstable
  number.
- Does not itself compute a standard error or confidence interval.
  The point estimate (ATE_hat) and, separately, the per-unit influence
  function values (needed to construct a standard error via their
  sample variance) are both exposed, so a caller can compute a CI, but
  this module does not assert a specific inference procedure is
  correct without that being separately validated, the same
  discipline applied to msprt.py's Type-I error claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

PROPENSITY_CLIP_MIN = 1e-6
PROPENSITY_CLIP_MAX = 1.0 - 1e-6


@dataclass(frozen=True)
class DoublyRobustResult:
    """Result of one AIPW estimation.

    ate: the point estimate of the average treatment effect.
    influence_values: per-unit influence function values (the summand
        inside the ATE_hat formula's sum, before averaging), exposed so
        a caller can compute a standard error via
        sqrt(variance(influence_values) / n) if they want one, without
        this module asserting that specific inference procedure is
        itself validated here.
    n_units: number of units the estimate was computed from.
    n_clipped_propensities: how many units had a propensity score
        outside [PROPENSITY_CLIP_MIN, PROPENSITY_CLIP_MAX] and were
        clipped to that boundary rather than raising. Zero is the
        expected, healthy value; a nonzero count is a real signal the
        propensity model is producing near-degenerate scores for some
        units, worth investigating, not silently ignored, which is why
        it's reported rather than only clipped.
    """

    ate: float
    influence_values: tuple[float, ...]
    n_units: int
    n_clipped_propensities: int

    @property
    def standard_error(self) -> float:
        """Standard error of the ATE estimate, via the sample variance
        of the influence function values divided by n. This is the
        standard asymptotic variance estimator for AIPW (see Robins,
        Rotnitzky, Zhao 1994), not itself independently re-derived
        here, exposed as a convenience since influence_values alone
        requires a caller to know to do this computation themselves.
        """
        n = self.n_units
        if n <= 1:
            return float("nan")
        mean_influence = sum(self.influence_values) / n
        variance = sum((v - mean_influence) ** 2 for v in self.influence_values) / (n - 1)
        return (variance / n) ** 0.5


def estimate_ate(
    treatments: Sequence[int],
    outcomes: Sequence[float],
    covariates: Sequence[Sequence[float]],
    propensity_model: Callable[[Sequence[float]], float],
    outcome_model_treated: Callable[[Sequence[float]], float],
    outcome_model_control: Callable[[Sequence[float]], float],
) -> DoublyRobustResult:
    """Computes the AIPW estimate of the average treatment effect.

    Args:
        treatments: sequence of 0/1 treatment indicators, one per unit.
        outcomes: sequence of observed outcomes, one per unit.
        covariates: sequence of covariate vectors, one per unit. Each
            element is itself a sequence (e.g. a list of floats), the
            shape a caller's fitted models expect.
        propensity_model: callable taking one unit's covariates,
            returning P(T=1 | X), a float in (0, 1). Already fitted,
            this function does not fit it.
        outcome_model_treated: callable taking one unit's covariates,
            returning the predicted outcome under treatment,
            mu_1(x). Already fitted.
        outcome_model_control: callable taking one unit's covariates,
            returning the predicted outcome under control, mu_0(x).
            Already fitted.

    Raises:
        ValueError: if treatments/outcomes/covariates have mismatched
            lengths, or if the input is empty.
    """
    n = len(treatments)
    if n == 0:
        raise ValueError("estimate_ate requires at least one unit, got zero")
    if len(outcomes) != n or len(covariates) != n:
        raise ValueError(
            f"treatments, outcomes, and covariates must have the same length, "
            f"got {n}, {len(outcomes)}, {len(covariates)}"
        )

    influence_values: list[float] = []
    n_clipped = 0

    for t_i, y_i, x_i in zip(treatments, outcomes, covariates):
        if t_i not in (0, 1):
            raise ValueError(f"treatment must be 0 or 1, got {t_i}")

        e_i = propensity_model(x_i)
        if not (PROPENSITY_CLIP_MIN <= e_i <= PROPENSITY_CLIP_MAX):
            n_clipped += 1
            e_i = min(max(e_i, PROPENSITY_CLIP_MIN), PROPENSITY_CLIP_MAX)

        mu_1_i = outcome_model_treated(x_i)
        mu_0_i = outcome_model_control(x_i)

        # The AIPW summand, stated directly from the module docstring's
        # formula: the outcome-model contrast, plus a propensity-
        # weighted correction term for whichever arm this unit was
        # actually observed in.
        base = mu_1_i - mu_0_i
        if t_i == 1:
            correction = (y_i - mu_1_i) / e_i
        else:
            correction = -(y_i - mu_0_i) / (1.0 - e_i)

        influence_values.append(base + correction)

    ate = sum(influence_values) / n

    return DoublyRobustResult(
        ate=ate,
        influence_values=tuple(influence_values),
        n_units=n,
        n_clipped_propensities=n_clipped,
    )