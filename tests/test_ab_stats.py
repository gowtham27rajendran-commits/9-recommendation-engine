import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ab_testing.stats import required_sample_size, two_proportion_z_test, welch_t_test


def test_identical_rates_are_not_significant():
    result = two_proportion_z_test("ctr", "A", 100, 1000, "B", 100, 1000)
    assert not result.significant
    assert abs(result.p_value - 1.0) < 0.05


def test_large_clear_difference_is_significant():
    # 5% CTR vs 15% CTR on large samples should be an obvious win for B
    result = two_proportion_z_test("ctr", "A", 500, 10_000, "B", 1500, 10_000)
    assert result.significant
    assert result.p_value < 0.05
    assert result.lift > 0


def test_small_sample_noisy_difference_is_not_significant():
    # tiny sample sizes shouldn't produce false-positive significance
    result = two_proportion_z_test("ctr", "A", 2, 20, "B", 3, 20)
    assert not result.significant


def test_welch_t_test_detects_revenue_difference():
    samples_a = [10.0, 12.0, 9.0, 11.0, 10.5] * 20
    samples_b = [20.0, 22.0, 19.0, 21.0, 20.5] * 20
    result = welch_t_test("revenue_per_session", "A", samples_a, "B", samples_b)
    assert result.significant
    assert result.rate_b > result.rate_a


def test_required_sample_size_increases_for_smaller_effects():
    n_large_effect = required_sample_size(baseline_rate=0.05, minimum_detectable_effect=0.20)
    n_small_effect = required_sample_size(baseline_rate=0.05, minimum_detectable_effect=0.05)
    assert n_small_effect > n_large_effect
