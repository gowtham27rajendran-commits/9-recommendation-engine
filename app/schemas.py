from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    item_id: int
    title: str | None = None
    category: str | None = None
    score: float
    promoted: bool = False


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: int
    variant: str
    results: list[RecommendationItem]
    latency_ms: float


class InteractionEvent(BaseModel):
    user_id: int
    item_id: int
    action: str = Field(pattern="^(view|click|add_to_cart|purchase)$")
    variant: str
    request_id: str | None = None


class SignificanceResponse(BaseModel):
    metric: str
    variant_a: str
    variant_b: str
    rate_a: float
    rate_b: float
    lift: float
    p_value: float
    significant: bool
    alpha: float
