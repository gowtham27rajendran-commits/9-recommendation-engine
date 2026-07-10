"""
Content-based filtering.

Works even for cold-start users/items because it never needs interaction
history to produce a score - only item features (category + tags) and,
for a user, either a taste vector (existing user) or onboarding category
picks (brand-new user). This is what the README means by "cold start: good".

Approach: one-hot encode category + tags into a sparse feature space,
build a per-user profile vector as the weighted average of the feature
vectors of items they've interacted with (or their declared onboarding
categories for new users), then rank items by cosine similarity to that
profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContentIndex:
    item_ids: list[int]
    item_vectors: np.ndarray  # (n_items, n_features), L2-normalized
    feature_names: list[str]
    item_id_to_idx: dict


def build_content_index(items: list[dict]) -> ContentIndex:
    all_tags = sorted({t for it in items for t in it["tags"]} | {it["category"] for it in items})
    feature_names = all_tags
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    item_ids = [it["item_id"] for it in items]
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}

    mat = np.zeros((len(items), len(feature_names)))
    for row, it in enumerate(items):
        mat[row, feat_idx[it["category"]]] += 2.0  # primary category weighted higher
        for tag in it["tags"]:
            mat[row, feat_idx[tag]] += 1.0

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms

    return ContentIndex(
        item_ids=item_ids,
        item_vectors=mat,
        feature_names=feature_names,
        item_id_to_idx=item_id_to_idx,
    )


def user_profile_from_history(index: ContentIndex, item_ids_with_weights: list[tuple[int, float]]) -> np.ndarray:
    """Weighted average of item vectors the user has interacted with."""
    if not item_ids_with_weights:
        return np.zeros(len(index.feature_names))
    vec = np.zeros(len(index.feature_names))
    total_w = 0.0
    for item_id, w in item_ids_with_weights:
        idx = index.item_id_to_idx.get(item_id)
        if idx is None:
            continue
        vec += w * index.item_vectors[idx]
        total_w += w
    if total_w > 0:
        vec /= total_w
    return vec


def user_profile_from_onboarding(index: ContentIndex, categories: list[str]) -> np.ndarray:
    """Cold-start path: build a profile straight from onboarding category picks."""
    vec = np.zeros(len(index.feature_names))
    feat_idx = {name: i for i, name in enumerate(index.feature_names)}
    for cat in categories:
        if cat in feat_idx:
            vec[feat_idx[cat]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def recommend(index: ContentIndex, user_vector: np.ndarray, k: int = 100, exclude_item_ids: set[int] | None = None) -> list[tuple[int, float]]:
    exclude_item_ids = exclude_item_ids or set()
    norm = np.linalg.norm(user_vector)
    if norm == 0:
        # no signal at all -> caller should fall back to popularity baseline
        return []
    scores = index.item_vectors @ (user_vector / norm)
    ranked_idx = np.argsort(-scores)
    results = []
    for idx in ranked_idx:
        item_id = index.item_ids[idx]
        if item_id in exclude_item_ids:
            continue
        results.append((item_id, float(scores[idx])))
        if len(results) >= k:
            break
    return results
