# Recommendation Engine with A/B Testing

A production recommendation engine combining collaborative filtering and content-based filtering, with built-in A/B testing to measure which algorithm performs best.

## Architecture

```
User Request → API Gateway
                    ↓
             A/B Test Router (assign user to variant)
                    ↓
        ┌──────────┴──────────┐
   Variant A               Variant B
  (Collab Filter)      (Content-Based)
        └──────────┬──────────┘
                    ↓
            Candidate Set (top-100)
                    ↓
             Re-Ranker (business rules: promoted items, diversity)
                    ↓
             Top-N Results
                    ↓
             Event Logger (Kafka) → offline evaluation
```

## Algorithms

| Algorithm | Signals Used | Cold Start | Latency |
|---|---|---|---|
| Collaborative Filtering (ALS) | User-item interactions | ❌ Poor | 10ms (precomputed) |
| Content-Based | Item features, user profile | ✅ Good | 20ms |
| Hybrid (weighted blend) | Both | ⚠️ Medium | 25ms |
| Popularity Baseline | Global/segment clicks | ✅ Good | <1ms |

## A/B Testing Design

- Users assigned to variants via deterministic hash (stable across sessions)
- Metrics tracked: CTR, conversion, session depth, revenue/session
- Statistical significance via two-proportion z-test
- Auto-promote winning variant after N days or α=0.05

## Running Locally

```bash
pip install -r requirements.txt
python scripts/train_als.py   # train collaborative filter
uvicorn app.main:app --reload
```

## Interview Talking Points

**"How do you solve the cold start problem?"**
New users: content-based on onboarding preferences + popularity baseline. New items: content-based only until 10+ interactions. The hybrid model naturally degrades to content-based when collaborative signals are sparse.

**"How do you run A/B tests without bias?"**
Deterministic hash of user_id % 100 assigns variant — same user always sees the same variant. Random assignment per-request would mean a user sees different algorithms each visit, contaminating results.

**"How do you evaluate recommendation quality offline?"**
Hold out last N interactions per user. Measure Recall@K (did recommended items include held-out items?) and NDCG@K (were they ranked highly?). Online metrics (CTR, conversion) are ground truth but expensive to run experiments for.
