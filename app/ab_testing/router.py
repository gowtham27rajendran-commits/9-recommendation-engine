"""
A/B test variant router.

Uses a deterministic hash of user_id (not random-per-request) so the same
user always lands in the same variant across sessions. This is the exact
point made in the README's interview talking points: random-per-request
assignment would let a single user see different algorithms on different
visits, which contaminates the experiment (you can't attribute a
conversion to a variant if the user bounced between variants).

hash(user_id) % 100 buckets users into [0, 100). Each variant owns a
contiguous range of buckets, which makes traffic splits easy to reason
about and easy to change without re-bucketing everyone (e.g. moving from
50/50 to 80/20 as a variant becomes the presumed winner).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Variant:
    name: str
    traffic_start: int  # inclusive, 0-99
    traffic_end: int  # exclusive, 0-100


@dataclass
class ABTestConfig:
    experiment_name: str
    variants: list[Variant] = field(default_factory=list)
    salt: str = "v1"  # bump salt to force re-bucketing (e.g. new experiment iteration)

    def validate(self) -> None:
        ranges = sorted((v.traffic_start, v.traffic_end) for v in self.variants)
        expected_start = 0
        for start, end in ranges:
            if start != expected_start:
                raise ValueError(f"Traffic ranges must be contiguous and cover [0,100); gap at {expected_start}")
            expected_start = end
        if expected_start != 100:
            raise ValueError("Traffic ranges must sum to 100")


DEFAULT_CONFIG = ABTestConfig(
    experiment_name="collab_vs_content_v1",
    variants=[
        Variant(name="collaborative", traffic_start=0, traffic_end=25),
        Variant(name="content_based", traffic_start=25, traffic_end=50),
        Variant(name="hybrid", traffic_start=50, traffic_end=75),
        Variant(name="popularity", traffic_start=75, traffic_end=100),
    ],
)


def bucket_for_user(user_id: int, salt: str = "v1") -> int:
    """Deterministic, stable bucket in [0, 100) for a given user_id."""
    h = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return int(h, 16) % 100


def assign_variant(user_id: int, config: ABTestConfig = DEFAULT_CONFIG) -> str:
    bucket = bucket_for_user(user_id, salt=config.salt)
    for variant in config.variants:
        if variant.traffic_start <= bucket < variant.traffic_end:
            return variant.name
    raise RuntimeError(f"No variant covers bucket {bucket} - check config.validate()")
