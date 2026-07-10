"""
Hybrid model: weighted blend of collaborative + content-based scores.

Key property called out in the README: "naturally degrades to
content-based when collaborative signals are sparse." We implement that
degradation explicitly rather than hoping the blend does it implicitly:
if a user has no ALS factors (never seen at training time), collaborative
weight is redistributed entirely to content-based. If a user also has no
content profile (no history, no onboarding picks), we fall back further
to popularity so we never return an empty list.
"""
from __future__ import annotations

import numpy as np

from app.models import collaborative, content_based, popularity


def hybrid_recommend(
    als_model: collaborative.ALSModel,
    content_index: content_based.ContentIndex,
    popularity_ranking: list[tuple[int, float]],
    user_id: int,
    content_user_vector: np.ndarray,
    k: int = 100,
    exclude_item_ids: set[int] | None = None,
    collab_weight: float = 0.6,
    content_weight: float = 0.4,
) -> list[tuple[int, float]]:
    exclude_item_ids = exclude_item_ids or set()

    collab_scores = als_model.score_user(user_id)  # None if cold-start
    has_collab = collab_scores is not None

    content_norm = np.linalg.norm(content_user_vector)
    has_content = content_norm > 0

    if not has_collab and not has_content:
        # total cold start: fall back to popularity baseline entirely
        return popularity.recommend(popularity_ranking, k=k, exclude_item_ids=exclude_item_ids)

    if not has_collab:
        # degrade to pure content-based
        return content_based.recommend(content_index, content_user_vector, k=k, exclude_item_ids=exclude_item_ids)

    # both signals available (or collab available, content thin) -> blend
    content_scores = content_index.item_vectors @ (content_user_vector / content_norm) if has_content else np.zeros(len(content_index.item_ids))

    def _normalize(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    collab_norm_scores = _normalize(collab_scores)
    content_norm_scores = _normalize(content_scores)

    w_collab, w_content = collab_weight, content_weight
    if not has_content:
        w_collab, w_content = 1.0, 0.0

    blended = w_collab * collab_norm_scores + w_content * content_norm_scores

    ranked_idx = np.argsort(-blended)
    results = []
    for idx in ranked_idx:
        item_id = content_index.item_ids[idx]
        if item_id in exclude_item_ids:
            continue
        results.append((item_id, float(blended[idx])))
        if len(results) >= k:
            break
    return results
