"""
Offline evaluation: hold out each user's most recent N interactions,
train/serve on the rest, and measure whether the held-out items show up
in the recommendations (and how highly ranked they are).

Recall@K = (# held-out items recommended in top K) / (# held-out items)
NDCG@K   = rewards hits near the top of the list more than hits near
           the bottom (a hit at rank 1 counts more than a hit at rank 10)

Run: python scripts/offline_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import collaborative, content_based, popularity
from app.models.hybrid import hybrid_recommend

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
K = 10
HOLDOUT_N = 3  # hold out the last 3 interactions per eligible user


def recall_at_k(recommended_ids: list[int], held_out_ids: set[int], k: int) -> float:
    if not held_out_ids:
        return None
    top_k = set(recommended_ids[:k])
    return len(top_k & held_out_ids) / len(held_out_ids)


def ndcg_at_k(recommended_ids: list[int], held_out_ids: set[int], k: int) -> float:
    if not held_out_ids:
        return None
    dcg = 0.0
    for rank, item_id in enumerate(recommended_ids[:k], start=1):
        if item_id in held_out_ids:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(held_out_ids), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    users = json.loads((DATA_DIR / "users.json").read_text())
    items = json.loads((DATA_DIR / "items.json").read_text())
    interactions = json.loads((DATA_DIR / "interactions.json").read_text())

    # sort each user's interactions by time, hold out the last HOLDOUT_N
    by_user: dict[int, list[dict]] = {}
    for ev in interactions:
        by_user.setdefault(ev["user_id"], []).append(ev)
    for uid in by_user:
        by_user[uid].sort(key=lambda e: e["timestamp"])

    train_interactions = []
    held_out_by_user: dict[int, set[int]] = {}
    train_history_by_user: dict[int, list[tuple[int, float]]] = {}

    for uid, evs in by_user.items():
        if len(evs) <= HOLDOUT_N:
            train_interactions.extend(evs)
            train_history_by_user[uid] = [(e["item_id"], e["weight"]) for e in evs]
            continue
        train = evs[:-HOLDOUT_N]
        held = evs[-HOLDOUT_N:]
        train_interactions.extend(train)
        train_history_by_user[uid] = [(e["item_id"], e["weight"]) for e in train]
        held_out_by_user[uid] = {e["item_id"] for e in held}

    print(f"Eligible users for eval (>{HOLDOUT_N} interactions): {len(held_out_by_user)}")

    user_ids = [u["user_id"] for u in users]
    item_ids = [it["item_id"] for it in items]
    users_by_id = {u["user_id"]: u for u in users}

    print("Training ALS on the train split (this holds out recent interactions, so it'll differ slightly from the served model)...")
    als_model = collaborative.train_als(
        interactions=train_interactions,
        n_users_total=len(user_ids),
        n_items_total=len(item_ids),
        user_ids=user_ids,
        item_ids=item_ids,
        n_factors=32,
        n_iterations=15,
    )
    content_index = content_based.build_content_index(items)
    popularity_rankings = popularity.build_popularity_ranking(train_interactions)
    global_popularity = popularity_rankings.get(None, [])

    def content_vector_for(uid: int) -> np.ndarray:
        history = train_history_by_user.get(uid, [])
        if history:
            return content_based.user_profile_from_history(content_index, history)
        user = users_by_id.get(uid)
        if user and user.get("is_new_user"):
            return content_based.user_profile_from_onboarding(content_index, user.get("onboarding_categories", []))
        return np.zeros(len(content_index.feature_names))

    variants = {
        "collaborative": lambda uid: [i for i, _ in als_model.recommend(uid, k=K)],
        "content_based": lambda uid: [i for i, _ in content_based.recommend(content_index, content_vector_for(uid), k=K)],
        "hybrid": lambda uid: [
            i
            for i, _ in hybrid_recommend(als_model, content_index, global_popularity, uid, content_vector_for(uid), k=K)
        ],
        "popularity": lambda uid: [i for i, _ in popularity.recommend(global_popularity, k=K)],
    }

    print(f"\n{'Variant':<15} {'Recall@'+str(K):>12} {'NDCG@'+str(K):>12}  (n={len(held_out_by_user)} users)")
    print("-" * 42)
    for variant_name, recommend_fn in variants.items():
        recalls, ndcgs = [], []
        for uid, held in held_out_by_user.items():
            recs = recommend_fn(uid)
            r = recall_at_k(recs, held, K)
            n = ndcg_at_k(recs, held, K)
            if r is not None:
                recalls.append(r)
                ndcgs.append(n)
        print(f"{variant_name:<15} {np.mean(recalls):>12.4f} {np.mean(ndcgs):>12.4f}")


if __name__ == "__main__":
    main()
