import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ab_testing.router import DEFAULT_CONFIG, assign_variant, bucket_for_user


def test_bucket_is_deterministic():
    for uid in [1, 42, 999, 123456]:
        b1 = bucket_for_user(uid)
        b2 = bucket_for_user(uid)
        assert b1 == b2


def test_bucket_in_range():
    for uid in range(1000):
        b = bucket_for_user(uid)
        assert 0 <= b < 100


def test_assign_variant_is_stable_across_calls():
    for uid in range(500):
        v1 = assign_variant(uid)
        v2 = assign_variant(uid)
        assert v1 == v2


def test_assign_variant_returns_known_variant():
    valid_names = {v.name for v in DEFAULT_CONFIG.variants}
    for uid in range(500):
        assert assign_variant(uid) in valid_names


def test_traffic_split_is_roughly_even():
    """With 10k users and a 25/25/25/25 split, each variant should land
    within a reasonable tolerance of 25% (this is a statistical property
    of SHA256 hashing, not a hard guarantee, hence the tolerance)."""
    from collections import Counter

    counts = Counter(assign_variant(uid) for uid in range(10_000))
    for variant, count in counts.items():
        share = count / 10_000
        assert 0.20 < share < 0.30, f"{variant} got {share:.2%}, expected ~25%"


def test_config_validation_catches_gaps():
    from app.ab_testing.router import ABTestConfig, Variant

    bad_config = ABTestConfig(
        experiment_name="broken",
        variants=[Variant(name="a", traffic_start=0, traffic_end=40), Variant(name="b", traffic_start=50, traffic_end=100)],
    )
    try:
        bad_config.validate()
        assert False, "expected ValueError for gap in traffic ranges"
    except ValueError:
        pass


def test_config_validation_passes_for_default():
    DEFAULT_CONFIG.validate()  # should not raise
