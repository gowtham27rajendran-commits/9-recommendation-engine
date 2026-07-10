"""
Popularity baseline: global (or segment-level) click counts.

This is the ultimate fallback - <1ms latency, always available, works
for any user including brand-new ones with zero profile data. Used both
as its own A/B variant and as the fallback inside the hybrid model when
collaborative signal is unavailable and content signal is thin.
"""
from __future__ import annotations

from collections import Counter


def build_popularity_ranking(interactions: list[dict], segment_key: str | None = None) -> dict[str | None, list[tuple[int, float]]]:
    """
    Returns a dict mapping segment -> ranked [(item_id, score), ...].
    If segment_key is None, returns a single global ranking under key None.
    """
    counters: dict[str | None, Counter] = {}
    for ev in interactions:
        seg = None if segment_key is None else ev.get(segment_key)
        counters.setdefault(seg, Counter())[ev["item_id"]] += ev["weight"]

    rankings = {}
    for seg, counter in counters.items():
        total = sum(counter.values()) or 1
        rankings[seg] = [(item_id, count / total) for item_id, count in counter.most_common()]
    return rankings


def recommend(ranking: list[tuple[int, float]], k: int = 100, exclude_item_ids: set[int] | None = None) -> list[tuple[int, float]]:
    exclude_item_ids = exclude_item_ids or set()
    results = []
    for item_id, score in ranking:
        if item_id in exclude_item_ids:
            continue
        results.append((item_id, score))
        if len(results) >= k:
            break
    return results
