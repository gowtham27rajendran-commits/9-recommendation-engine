"""
Generates synthetic users, items, and interactions so the whole system
is runnable without any external dataset.

Design notes (why synthetic data is shaped this way):
- Items belong to categories with feature vectors -> lets content-based
  filtering actually mean something (similar items share categories/tags).
- Users have latent taste vectors over categories -> interactions are
  sampled with a bias towards categories the user likes, which gives the
  collaborative filter real signal to recover (instead of pure noise).
- A "long tail" popularity skew is injected via a power-law over items,
  so the popularity baseline and cold-start behavior are realistic.
"""
import json
import random
from pathlib import Path

import numpy as np

RNG_SEED = 42
N_USERS = 2000
N_ITEMS = 500
N_CATEGORIES = 12
CATEGORIES = [f"cat_{i}" for i in range(N_CATEGORIES)]
INTERACTIONS_PER_USER_RANGE = (5, 80)  # heavy-tailed engagement
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_items(rng: np.random.Generator) -> list[dict]:
    items = []
    # power-law popularity weight per item, used only to bias interaction sampling
    popularity_weights = rng.pareto(a=1.5, size=N_ITEMS) + 0.1
    for item_id in range(N_ITEMS):
        primary_cat = int(rng.integers(0, N_CATEGORIES))
        # each item has a primary category plus 0-2 secondary tags
        n_secondary = int(rng.integers(0, 3))
        secondary = rng.choice(
            [c for c in range(N_CATEGORIES) if c != primary_cat],
            size=n_secondary,
            replace=False,
        ).tolist()
        tags = [primary_cat] + secondary
        items.append(
            {
                "item_id": item_id,
                "title": f"Item {item_id}",
                "category": CATEGORIES[primary_cat],
                "tags": [CATEGORIES[t] for t in tags],
                "price": round(float(rng.uniform(5, 200)), 2),
                "popularity_weight": float(popularity_weights[item_id]),
                "promoted": bool(rng.random() < 0.04),  # ~4% of catalog is sponsored
            }
        )
    return items


def make_users(rng: np.random.Generator) -> list[dict]:
    users = []
    for user_id in range(N_USERS):
        # each user has a taste distribution over categories (dirichlet -> sums to 1)
        taste = rng.dirichlet(alpha=np.full(N_CATEGORIES, 0.4))
        users.append(
            {
                "user_id": user_id,
                "taste_vector": taste.tolist(),
                "is_new_user": bool(rng.random() < 0.1),  # 10% cold-start users
                "onboarding_categories": [
                    CATEGORIES[i] for i in np.argsort(-taste)[:2]
                ],
            }
        )
    return users


def make_interactions(rng: np.random.Generator, users: list[dict], items: list[dict]) -> list[dict]:
    interactions = []
    item_cat_idx = np.array([CATEGORIES.index(it["category"]) for it in items])
    item_pop = np.array([it["popularity_weight"] for it in items])
    event_id = 0

    for user in users:
        if user["is_new_user"]:
            # cold-start users get almost no history by definition
            n_interactions = int(rng.integers(0, 3))
        else:
            n_interactions = int(rng.integers(*INTERACTIONS_PER_USER_RANGE))

        taste = np.array(user["taste_vector"])
        # score = how much this item's category matches user taste, blended with
        # global popularity so popular items still get oversampled a bit
        cat_affinity = taste[item_cat_idx]
        score = 0.7 * cat_affinity + 0.3 * (item_pop / item_pop.sum())
        prob = score / score.sum()

        chosen = rng.choice(len(items), size=min(n_interactions, len(items)), replace=False, p=prob)
        for item_idx in chosen:
            action = rng.choice(
                ["view", "click", "add_to_cart", "purchase"],
                p=[0.55, 0.30, 0.10, 0.05],
            )
            interactions.append(
                {
                    "event_id": event_id,
                    "user_id": user["user_id"],
                    "item_id": int(items[item_idx]["item_id"]),
                    "action": action,
                    "weight": {"view": 1, "click": 3, "add_to_cart": 5, "purchase": 10}[action],
                    "timestamp": int(rng.integers(1_700_000_000, 1_720_000_000)),
                }
            )
            event_id += 1

    interactions.sort(key=lambda x: x["timestamp"])
    return interactions


def main():
    rng = np.random.default_rng(RNG_SEED)
    random.seed(RNG_SEED)

    DATA_DIR.mkdir(exist_ok=True)
    items = make_items(rng)
    users = make_users(rng)
    interactions = make_interactions(rng, users, items)

    (DATA_DIR / "items.json").write_text(json.dumps(items, indent=2))
    (DATA_DIR / "users.json").write_text(json.dumps(users, indent=2))
    (DATA_DIR / "interactions.json").write_text(json.dumps(interactions, indent=2))

    print(f"Generated {len(users)} users, {len(items)} items, {len(interactions)} interactions")
    print(f"Cold-start users: {sum(u['is_new_user'] for u in users)}")
    print(f"Written to {DATA_DIR}")


if __name__ == "__main__":
    main()
