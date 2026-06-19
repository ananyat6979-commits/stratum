"""
Statistical utilities for STRATUM benchmark analysis.

Every function in this module corresponds to a specific statistical claim
the benchmark harness makes. None of the functions produce a number
without also producing a confidence interval.

DESIGN PHILOSOPHY
=================
A benchmark without confidence intervals is an anecdote. A benchmark
without effect size is uninterpretable. A benchmark without a stated
null hypothesis is unfalsifiable. This module enforces all three.

REFERENCE
=========
Welch, B.L. (1947). "The generalization of 'Student's' problem when
several different population variances are involved."
Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
Efron, B. & Hastie, T. (2016). Computer Age Statistical Inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with a confidence interval.

    All three values are in the same units as the input measurements.
    """

    estimate: float
    lower: float
    upper: float
    confidence_level: float

    def __str__(self) -> str:
        pct = int(self.confidence_level * 100)
        return (
            f"{self.estimate:.3f} "
            f"[{pct}% CI: {self.lower:.3f}, {self.upper:.3f}]"
        )


@dataclass(frozen=True)
class ComparisonResult:
    """Result of a statistical comparison between two measurement sets.

    Includes effect size (Cohen's d) alongside p-value, because a
    statistically significant result with negligible effect size is
    not an engineering finding worth acting on.
    """

    treatment_mean: float
    control_mean: float
    absolute_difference: ConfidenceInterval
    relative_difference_pct: ConfidenceInterval
    cohens_d: float
    cohens_d_interpretation: str
    p_value: float
    reject_null: bool
    alpha: float

    def __str__(self) -> str:
        direction = "improvement" if self.absolute_difference.estimate < 0 else "regression"
        return (
            f"Treatment: {self.treatment_mean:.3f}ms, "
            f"Control: {self.control_mean:.3f}ms\n"
            f"  Difference: {self.absolute_difference} ({direction})\n"
            f"  Relative: {self.relative_difference_pct}%\n"
            f"  Cohen's d: {self.cohens_d:.3f} ({self.cohens_d_interpretation})\n"
            f"  p-value: {self.p_value:.4f} "
            f"({'reject H0' if self.reject_null else 'fail to reject H0'} "
            f"at α={self.alpha})"
        )


def bootstrap_percentile_ci(
    measurements: Sequence[float],
    percentile: float,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    rng_seed: int = 42,
) -> ConfidenceInterval:
    """Compute a bootstrap percentile confidence interval.

    Uses the percentile bootstrap method (not BCa), which is appropriate
    when the estimator is a sample quantile (P50, P95, P99, P999).

    Bootstrap CI is non-parametric and makes no distributional assumptions
    about the measurement -- correct for latency, which is typically
    bimodal (fast path vs slow path) and violates the normality assumption
    of parametric intervals.

    Args:
        measurements: Raw latency measurements in any consistent unit.
        percentile: Target percentile in [0, 100] (e.g. 99.0 for P99).
        n_bootstrap: Number of bootstrap resamples. 10,000 is standard;
            increase to 100,000 for publication-quality results.
        confidence_level: Desired confidence level, e.g. 0.95 for 95% CI.
        rng_seed: Fixed seed for reproducibility. Every benchmark run must
            use the same seed so CI bounds are comparable across runs.

    Returns:
        ConfidenceInterval with the point estimate and bootstrap bounds.
    """
    arr = np.array(measurements, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)

    point_estimate = float(np.percentile(arr, percentile))

    bootstrap_estimates = np.percentile(
        rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True),
        percentile,
        axis=1,
    )

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_estimates, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def bootstrap_mean_ci(
    measurements: Sequence[float],
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    rng_seed: int = 42,
) -> ConfidenceInterval:
    """Bootstrap confidence interval for the mean."""
    arr = np.array(measurements, dtype=np.float64)
    rng = np.random.default_rng(rng_seed)

    point_estimate = float(np.mean(arr))
    bootstrap_means = np.mean(
        rng.choice(arr, size=(n_bootstrap, len(arr)), replace=True),
        axis=1,
    )

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
    )


def _cohens_d_interpretation(d: float) -> str:
    """Cohen's (1988) conventional thresholds for effect size magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_latencies(
    treatment: Sequence[float],
    control: Sequence[float],
    alpha: float = 0.05,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    rng_seed: int = 42,
) -> ComparisonResult:
    """Compare two latency distributions using Welch's t-test + effect size.

    Uses Welch's t-test (not Student's) because we make no assumption
    that the two samples have equal variance -- a critical assumption
    in benchmarking where treatment and control may have different
    variance profiles (e.g., the treatment might reduce mean latency
    but increase variance, or vice versa).

    Args:
        treatment: Latency measurements from the experimental condition.
        control: Latency measurements from the baseline condition.
        alpha: Type I error rate. Bonferroni-correct this if running
            multiple comparisons in the same benchmark session.
        n_bootstrap: Bootstrap resamples for the difference CI.
        confidence_level: CI confidence level.
        rng_seed: Fixed seed for reproducibility.

    Returns:
        ComparisonResult with full statistical summary.
    """
    t_arr = np.array(treatment, dtype=np.float64)
    c_arr = np.array(control, dtype=np.float64)

    t_mean = float(np.mean(t_arr))
    c_mean = float(np.mean(c_arr))

    # Welch's t-test: unequal variances, unequal sample sizes
    t_stat, p_value = stats.ttest_ind(t_arr, c_arr, equal_var=False)

    # Cohen's d using pooled standard deviation
    pooled_std = math.sqrt(
        (np.var(t_arr, ddof=1) + np.var(c_arr, ddof=1)) / 2.0
    )
    cohens_d = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0

    # Bootstrap CI for absolute difference
    rng = np.random.default_rng(rng_seed)
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_t = float(np.mean(rng.choice(t_arr, size=len(t_arr), replace=True)))
        boot_c = float(np.mean(rng.choice(c_arr, size=len(c_arr), replace=True)))
        boot_diffs.append(boot_t - boot_c)

    abs_diff = t_mean - c_mean
    diff_alpha = 1.0 - confidence_level
    diff_lower = float(np.percentile(boot_diffs, 100 * diff_alpha / 2))
    diff_upper = float(np.percentile(boot_diffs, 100 * (1 - diff_alpha / 2)))

    abs_diff_ci = ConfidenceInterval(
        estimate=abs_diff,
        lower=diff_lower,
        upper=diff_upper,
        confidence_level=confidence_level,
    )

    # Relative difference as percentage of control mean
    rel_diff = (abs_diff / c_mean * 100) if c_mean != 0 else float("nan")
    rel_lower = (diff_lower / c_mean * 100) if c_mean != 0 else float("nan")
    rel_upper = (diff_upper / c_mean * 100) if c_mean != 0 else float("nan")

    rel_diff_ci = ConfidenceInterval(
        estimate=rel_diff,
        lower=rel_lower,
        upper=rel_upper,
        confidence_level=confidence_level,
    )

    return ComparisonResult(
        treatment_mean=t_mean,
        control_mean=c_mean,
        absolute_difference=abs_diff_ci,
        relative_difference_pct=rel_diff_ci,
        cohens_d=cohens_d,
        cohens_d_interpretation=_cohens_d_interpretation(cohens_d),
        p_value=float(p_value),
        reject_null=float(p_value) < alpha,
        alpha=alpha,
    )


def compute_percentiles(
    measurements: Sequence[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    rng_seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    """Compute the standard latency percentile suite with bootstrap CIs.

    Standard suite: P50, P75, P95, P99, P999, mean.

    Returns:
        Dict mapping percentile names to ConfidenceInterval objects.
    """
    arr = list(measurements)
    return {
        "p50": bootstrap_percentile_ci(arr, 50.0, n_bootstrap, confidence_level, rng_seed),
        "p75": bootstrap_percentile_ci(arr, 75.0, n_bootstrap, confidence_level, rng_seed),
        "p95": bootstrap_percentile_ci(arr, 95.0, n_bootstrap, confidence_level, rng_seed),
        "p99": bootstrap_percentile_ci(arr, 99.0, n_bootstrap, confidence_level, rng_seed),
        "p999": bootstrap_percentile_ci(arr, 99.9, n_bootstrap, confidence_level, rng_seed),
        "mean": bootstrap_mean_ci(arr, n_bootstrap, confidence_level, rng_seed),
    }
