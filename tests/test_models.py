import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import collaborative, content_based, popularity
from app.models.hybrid import hybrid_recommend
from app.reranker import rerank


ITEMS = [
    {"item_id": 1, "title": "A", "category": "electronics", "tags": ["electronics", "audio"], "promoted": False},
    {"item_id": 2, "title": "B", "category": "electronics", "tags": ["electronics", "video"], "promoted": True},
    {"item_id": 3, "title": "C", "category": "books", "tags": ["books", "fiction"], "promoted": False},
    {"item_id": 4, "title": "D", "category": "books", "tags": ["books", "nonfiction"], "promoted": False},
    {"item_id": 5, "title": "E", "category": "kitchen", "tags": ["kitchen"], "promoted": True},
]
ITEMS_BY_ID = {it["item_id"]: it for it in ITEMS}

INTERACTIONS = [
    {"user_id": 1, "item_id": 1, "action": "click", "weight": 3, "timestamp": 1},
    {"user_id": 1, "item_id": 2, "action": "purchase", "weight": 10, "timestamp": 2},
    {"user_id": 2, "item_id": 3, "action": "click", "weight": 3, "timestamp": 3},
    {"user_id": 2, "item_id": 4, "action": "view", "weight": 1, "timestamp": 4},
]


def test_content_index_cold_start_from_onboarding():
    index = content_based.build_content_index(ITEMS)
    vec = content_based.user_profile_from_onboarding(index, ["electronics"])
    recs = content_based.recommend(index, vec, k=5)
    top_item_id = recs[0][0]
    assert ITEMS_BY_ID[top_item_id]["category"] == "electronics"


def test_content_recommend_excludes_seen_items():
    index = content_based.build_content_index(ITEMS)
    vec = content_based.user_profile_from_onboarding(index, ["electronics"])
    recs = content_based.recommend(index, vec, k=5, exclude_item_ids={2})
    assert all(item_id != 2 for item_id, _ in recs)


def test_als_cold_start_returns_none():
    model = collaborative.train_als(
        interactions=INTERACTIONS,
        n_users_total=2,
        n_items_total=5,
        user_ids=[1, 2],
        item_ids=[1, 2, 3, 4, 5],
        n_factors=4,
        n_iterations=3,
    )
    assert model.score_user(user_id=999) is None  # unseen user
    assert model.score_user(user_id=1) is not None


def test_popularity_ranking_orders_by_weight():
    rankings = popularity.build_popularity_ranking(INTERACTIONS)
    global_ranking = rankings[None]
    ranked_ids = [item_id for item_id, _ in global_ranking]
    # item 2 has weight 10 (purchase), should outrank item 4 with weight 1 (view)
    assert ranked_ids.index(2) < ranked_ids.index(4)


def test_hybrid_degrades_to_popularity_for_total_cold_start():
    model = collaborative.train_als(
        interactions=INTERACTIONS, n_users_total=2, n_items_total=5,
        user_ids=[1, 2], item_ids=[1, 2, 3, 4, 5], n_factors=4, n_iterations=3,
    )
    index = content_based.build_content_index(ITEMS)
    pop_ranking = popularity.build_popularity_ranking(INTERACTIONS)[None]
    empty_vec = np.zeros(len(index.feature_names))

    recs = hybrid_recommend(model, index, pop_ranking, user_id=999, content_user_vector=empty_vec, k=5)
    assert recs  # should not be empty - falls back to popularity
    assert [i for i, _ in recs] == [i for i, _ in pop_ranking][:5]


def test_reranker_caps_promoted_items():
    candidates = [(i, 1.0 / i) for i in [1, 2, 3, 4, 5]]  # item 2 and 5 are promoted
    results = rerank(candidates, ITEMS_BY_ID, top_n=5, max_promoted=2)
    promoted_count = sum(1 for r in results if r["promoted"])
    assert promoted_count <= 2


def test_reranker_returns_requested_top_n():
    candidates = [(i, 1.0 / i) for i in [1, 2, 3, 4, 5]]
    results = rerank(candidates, ITEMS_BY_ID, top_n=3)
    assert len(results) == 3


def test_reranker_handles_empty_candidates():
    assert rerank([], ITEMS_BY_ID, top_n=5) == []
