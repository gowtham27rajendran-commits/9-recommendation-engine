import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers startup event, loads real artifacts from data/
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["users_loaded"] > 0


def test_recommendations_returns_top_n(client):
    resp = client.get("/recommendations/0?top_n=5")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 5
    assert body["variant"] in {"collaborative", "content_based", "hybrid", "popularity"}
    assert body["latency_ms"] < 500  # generous CI-safe ceiling; local runs are sub-ms


def test_recommendations_unknown_user_404s(client):
    resp = client.get("/recommendations/999999999")
    assert resp.status_code == 404


def test_same_user_gets_same_variant_across_requests(client):
    v1 = client.get("/recommendations/7?top_n=3").json()["variant"]
    v2 = client.get("/recommendations/7?top_n=3").json()["variant"]
    assert v1 == v2


def test_log_interaction(client):
    resp = client.post(
        "/interactions",
        json={"user_id": 0, "item_id": 1, "action": "click", "variant": "popularity"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "logged"


def test_log_interaction_rejects_bad_action(client):
    resp = client.post(
        "/interactions",
        json={"user_id": 0, "item_id": 1, "action": "not_a_real_action", "variant": "popularity"},
    )
    assert resp.status_code == 422  # pydantic pattern validation should reject this
