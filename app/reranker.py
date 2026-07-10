"""
Re-ranker: takes the top-100 candidate set from whichever algorithm the
A/B router picked, and applies business rules before returning top-N.

Two rules implemented, matching the README:
1. Promoted items - sponsored items get boosted into view (capped, so we
   don't turn the whole feed into ads).
2. Diversity - greedy category-diversification (a simplified MMR:
   Maximal Marginal Relevance) so results aren't 10 near-duplicate items
   from one category, which tanks session depth even if each individual
   item scored well.
"""
from __future__ import annotations


def rerank(
    candidates: list[tuple[int, float]],
    items_by_id: dict[int, dict],
    top_n: int = 10,
    max_promoted: int = 2,
    diversity_lambda: float = 0.3,
) -> list[dict]:
    """
    candidates: [(item_id, score), ...] already sorted by relevance, best first.
    diversity_lambda: 0 = pure relevance ranking, 1 = pure diversity ranking.
    """
    if not candidates:
        return []

    # Step 1: promoted-item boost. Pull promoted items up near the top,
    # but cap how many can appear so it doesn't dominate the feed.
    promoted = [c for c in candidates if items_by_id.get(c[0], {}).get("promoted")]
    organic = [c for c in candidates if not items_by_id.get(c[0], {}).get("promoted")]
    promoted_to_inject = promoted[:max_promoted]
    remaining_pool = [c for c in candidates if c not in promoted_to_inject]

    # interleave: promoted items get slots 1 and ~4 rather than all stacked at top,
    # so the page doesn't read as "here are the ads, then the real results"
    seeded: list[tuple[int, float]] = []
    if promoted_to_inject:
        seeded.append(promoted_to_inject[0])
    pool_iter = iter([c for c in organic if c not in seeded])

    # Step 2: greedy diversity selection (simplified MMR) over category
    selected: list[tuple[int, float]] = list(seeded)
    remaining = [c for c in remaining_pool if c not in seeded]

    max_score = max((s for _, s in candidates), default=1.0) or 1.0

    while remaining and len(selected) < top_n:
        # inject the 2nd promoted item partway down, once
        if len(promoted_to_inject) > 1 and promoted_to_inject[1] not in selected and len(selected) == 3:
            selected.append(promoted_to_inject[1])
            remaining = [c for c in remaining if c != promoted_to_inject[1]]
            continue

        selected_categories = [items_by_id.get(iid, {}).get("category") for iid, _ in selected]

        def mmr_score(candidate):
            item_id, score = candidate
            norm_relevance = score / max_score
            category = items_by_id.get(item_id, {}).get("category")
            category_penalty = selected_categories.count(category) * 0.25
            return (1 - diversity_lambda) * norm_relevance - diversity_lambda * category_penalty

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    final = selected[:top_n]
    return [
        {
            "item_id": item_id,
            "score": round(score, 4),
            "title": items_by_id.get(item_id, {}).get("title"),
            "category": items_by_id.get(item_id, {}).get("category"),
            "promoted": items_by_id.get(item_id, {}).get("promoted", False),
        }
        for item_id, score in final
    ]
