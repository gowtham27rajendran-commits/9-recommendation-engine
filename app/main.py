"""
Recommendation Engine API.

Wires together the full pipeline from the README's architecture diagram:

  User Request -> API Gateway (FastAPI route)
              -> A/B Test Router (assign user to variant)
              -> Variant model produces candidate set (top-100)
              -> Re-Ranker (business rules: promoted items, diversity)
              -> Top-N Results
              -> Event Logger (Kafka / local fallback) -> offline evaluation

Run with: uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ab_testing.router import DEFAULT_CONFIG, assign_variant
from app.ab_testing.stats import two_proportion_z_test
from app.event_logger import get_event_logger
from app.models import collaborative, content_based, popularity
from app.models.hybrid import hybrid_recommend
from app.reranker import rerank
from app.schemas import (
    InteractionEvent,
    RecommendationResponse,
    SignificanceResponse,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ----------------------------------------------------------------------------
# App state: loaded once at startup. In a real deployment these would be
# refreshed on a schedule (e.g. retrain ALS nightly, reload artifacts via a
# model registry) rather than only at process boot.
# ----------------------------------------------------------------------------
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE.clear()


app = FastAPI(
    title="Recommendation Engine with A/B Testing",
    description="Collaborative filtering + content-based filtering, A/B tested in production.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def load_artifacts():
    users = json.loads((DATA_DIR / "users.json").read_text())
    items = json.loads((DATA_DIR / "items.json").read_text())
    interactions = json.loads((DATA_DIR / "interactions.json").read_text())

    items_by_id = {it["item_id"]: it for it in items}
    users_by_id = {u["user_id"]: u for u in users}

    # user -> [(item_id, weight), ...] history, needed to build content profiles
    history_by_user: dict[int, list[tuple[int, float]]] = {}
    for ev in interactions:
        history_by_user.setdefault(ev["user_id"], []).append((ev["item_id"], ev["weight"]))

    als_model_path = DATA_DIR / "als_model.pkl"
    if not als_model_path.exists():
        raise RuntimeError(
            "ALS model not found. Run `python scripts/train_als.py` before starting the server."
        )
    als_model = collaborative.load_model(als_model_path)
    content_index = content_based.build_content_index(items)
    popularity_rankings = popularity.build_popularity_ranking(interactions)
    global_popularity = popularity_rankings.get(None, [])

    STATE.update(
        users_by_id=users_by_id,
        items_by_id=items_by_id,
        history_by_user=history_by_user,
        als_model=als_model,
        content_index=content_index,
        global_popularity=global_popularity,
        event_logger=get_event_logger(),
    )
    print(f"Loaded {len(users)} users, {len(items)} items, ALS model, content index, popularity baseline.")
    if STATE["event_logger"].degraded:
        print("NOTE: Kafka not reachable - event logging is falling back to data/events.jsonl")


def _content_user_vector(user_id: int) -> np.ndarray:
    content_index = STATE["content_index"]
    user = STATE["users_by_id"].get(user_id)
    history = STATE["history_by_user"].get(user_id, [])

    if history:
        return content_based.user_profile_from_history(content_index, history)
    if user and user.get("is_new_user"):
        return content_based.user_profile_from_onboarding(content_index, user.get("onboarding_categories", []))
    return np.zeros(len(content_index.feature_names))


def _get_candidates(variant: str, user_id: int, exclude_item_ids: set[int]) -> list[tuple[int, float]]:
    if variant == "collaborative":
        candidates = STATE["als_model"].recommend(user_id, k=100, exclude_item_ids=exclude_item_ids)
        if not candidates:
            # cold-start user with no ALS factors: this is exactly the failure
            # mode called out in the README table ("Collaborative: Cold Start Poor").
            # We still return something rather than an empty page, but a real
            # A/B report should track this fallback rate as its own metric.
            candidates = popularity.recommend(STATE["global_popularity"], k=100, exclude_item_ids=exclude_item_ids)
        return candidates

    if variant == "content_based":
        user_vec = _content_user_vector(user_id)
        candidates = content_based.recommend(STATE["content_index"], user_vec, k=100, exclude_item_ids=exclude_item_ids)
        if not candidates:
            candidates = popularity.recommend(STATE["global_popularity"], k=100, exclude_item_ids=exclude_item_ids)
        return candidates

    if variant == "hybrid":
        user_vec = _content_user_vector(user_id)
        return hybrid_recommend(
            STATE["als_model"],
            STATE["content_index"],
            STATE["global_popularity"],
            user_id,
            user_vec,
            k=100,
            exclude_item_ids=exclude_item_ids,
        )

    if variant == "popularity":
        return popularity.recommend(STATE["global_popularity"], k=100, exclude_item_ids=exclude_item_ids)

    raise HTTPException(status_code=500, detail=f"Unknown variant '{variant}'")


@app.get("/")
def root():
    return {
        "service": "Recommendation Engine with A/B Testing",
        "endpoints": ["/recommendations/{user_id}", "/interactions", "/experiments/results", "/health"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "users_loaded": len(STATE.get("users_by_id", {})),
        "items_loaded": len(STATE.get("items_by_id", {})),
        "event_logger_degraded": STATE.get("event_logger").degraded if STATE.get("event_logger") else None,
    }


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, top_n: int = 10):
    start = time.perf_counter()
    request_id = str(uuid.uuid4())

    if user_id not in STATE["users_by_id"]:
        raise HTTPException(status_code=404, detail=f"Unknown user_id {user_id}")

    variant = assign_variant(user_id, DEFAULT_CONFIG)
    already_seen = {item_id for item_id, _ in STATE["history_by_user"].get(user_id, [])}

    candidates = _get_candidates(variant, user_id, exclude_item_ids=already_seen)
    results = rerank(candidates, STATE["items_by_id"], top_n=top_n)

    latency_ms = (time.perf_counter() - start) * 1000
    STATE["event_logger"].log_recommendation_served(
        user_id=user_id, variant=variant, item_ids=[r["item_id"] for r in results], request_id=request_id
    )

    return RecommendationResponse(
        request_id=request_id,
        user_id=user_id,
        variant=variant,
        results=results,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/interactions")
def log_interaction(event: InteractionEvent):
    STATE["event_logger"].log_interaction(
        user_id=event.user_id,
        item_id=event.item_id,
        action=event.action,
        variant=event.variant,
        request_id=event.request_id,
    )
    return {"status": "logged"}


@app.get("/experiments/results", response_model=list[SignificanceResponse])
def experiment_results(metric: str = "ctr", alpha: float = 0.05):
    """
    Reads back logged events (from data/events.jsonl, the local fallback path)
    and computes pairwise significance between every variant vs the current
    control ("popularity", the simplest baseline) for the requested metric.

    metric=ctr: click-through rate (clicks / recommendations_served)
    """
    events_path = DATA_DIR / "events.jsonl"
    if not events_path.exists():
        return []

    served_by_variant: dict[str, int] = {}
    clicks_by_variant: dict[str, int] = {}

    with open(events_path) as f:
        for line in f:
            ev = json.loads(line)
            variant = ev.get("variant")
            if ev.get("event_type") == "recommendation_served":
                served_by_variant[variant] = served_by_variant.get(variant, 0) + len(ev.get("item_ids", []))
            elif ev.get("event_type") == "interaction" and ev.get("action") == "click":
                clicks_by_variant[variant] = clicks_by_variant.get(variant, 0) + 1

    control = "popularity"
    if control not in served_by_variant:
        return []

    results = []
    for variant, trials in served_by_variant.items():
        if variant == control:
            continue
        result = two_proportion_z_test(
            metric_name=metric,
            variant_a_name=control,
            successes_a=clicks_by_variant.get(control, 0),
            trials_a=served_by_variant.get(control, 0),
            variant_b_name=variant,
            successes_b=clicks_by_variant.get(variant, 0),
            trials_b=trials,
            alpha=alpha,
        )
        results.append(SignificanceResponse(**result.__dict__))
    return results
