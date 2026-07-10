"""
Trains the collaborative filter and persists it to data/als_model.pkl.
Run via: python scripts/train_als.py
This is the offline half of "10ms (precomputed)" latency for the
collaborative variant - all the expensive factorization work happens
here, ahead of time, so serving is just a dot product.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.collaborative import save_model, train_als

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    users = json.loads((DATA_DIR / "users.json").read_text())
    items = json.loads((DATA_DIR / "items.json").read_text())
    interactions = json.loads((DATA_DIR / "interactions.json").read_text())

    user_ids = [u["user_id"] for u in users]
    item_ids = [it["item_id"] for it in items]

    print(f"Training ALS on {len(user_ids)} users x {len(item_ids)} items, {len(interactions)} interactions...")
    start = time.time()
    model = train_als(
        interactions=interactions,
        n_users_total=len(user_ids),
        n_items_total=len(item_ids),
        user_ids=user_ids,
        item_ids=item_ids,
        n_factors=32,
        n_iterations=15,
        reg=0.1,
        alpha=15.0,
    )
    elapsed = time.time() - start
    print(f"Trained in {elapsed:.1f}s")

    save_model(model)
    print(f"Saved model to {DATA_DIR / 'als_model.pkl'}")

    # sanity check: show top-5 recommendations for a random known user
    sample_user = user_ids[0]
    recs = model.recommend(sample_user, k=5)
    print(f"Sample recs for user {sample_user}: {recs}")


if __name__ == "__main__":
    main()
