"""
Statistical significance testing for A/B test results.

Two-proportion z-test, used for rate metrics like CTR and conversion:
given clicks/conversions and impressions/sessions for two variants, are
the observed rates different enough to not be explained by chance?

H0: p_A == p_B
H1: p_A != p_B (two-sided)

We also expose Welch's t-test for continuous metrics (revenue/session,
session depth) since those aren't proportions and a z-test on proportions
would be the wrong tool - a common interview follow-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass
class SignificanceResult:
    metric: str
    variant_a: str
    variant_b: str
    rate_a: float
    rate_b: float
    lift: float  # relative lift of B over A
    z_or_t_statistic: float
    p_value: float
    significant: bool
    alpha: float


def two_proportion_z_test(
    metric_name: str,
    variant_a_name: str,
    successes_a: int,
    trials_a: int,
    variant_b_name: str,
    successes_b: int,
    trials_b: int,
    alpha: float = 0.05,
) -> SignificanceResult:
    p_a = successes_a / trials_a if trials_a else 0.0
    p_b = successes_b / trials_b if trials_b else 0.0

    # pooled proportion under H0 (p_A == p_B)
    p_pool = (successes_a + successes_b) / (trials_a + trials_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / trials_a + 1 / trials_b)) if trials_a and trials_b else float("inf")

    z = (p_b - p_a) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    lift = (p_b - p_a) / p_a if p_a > 0 else float("inf")

    return SignificanceResult(
        metric=metric_name,
        variant_a=variant_a_name,
        variant_b=variant_b_name,
        rate_a=p_a,
        rate_b=p_b,
        lift=lift,
        z_or_t_statistic=z,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
    )


def welch_t_test(
    metric_name: str,
    variant_a_name: str,
    samples_a: list[float],
    variant_b_name: str,
    samples_b: list[float],
    alpha: float = 0.05,
) -> SignificanceResult:
    """For continuous metrics (revenue/session, session depth) where variance may differ between groups."""
    t_stat, p_value = stats.ttest_ind(samples_a, samples_b, equal_var=False)
    mean_a = sum(samples_a) / len(samples_a) if samples_a else 0.0
    mean_b = sum(samples_b) / len(samples_b) if samples_b else 0.0
    lift = (mean_b - mean_a) / mean_a if mean_a else float("inf")

    return SignificanceResult(
        metric=metric_name,
        variant_a=variant_a_name,
        variant_b=variant_b_name,
        rate_a=mean_a,
        rate_b=mean_b,
        lift=lift,
        z_or_t_statistic=float(t_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
    )


def required_sample_size(baseline_rate: float, minimum_detectable_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """
    Rough sample-size-per-variant calculator for planning experiment duration.
    minimum_detectable_effect is a relative lift, e.g. 0.05 for a 5% relative lift.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    return math.ceil(numerator / denominator)
