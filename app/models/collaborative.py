"""
Implicit-feedback Alternating Least Squares (ALS) collaborative filter.

Implemented from scratch with numpy (no external ALS library) so the
math is fully inspectable/interview-defensible:

Given a user-item interaction weight matrix R (implicit confidence, not
explicit ratings), we factorize R ~= U @ V^T where U is (n_users x k)
and V is (n_items x k). Following Hu, Koren & Volinsky (2008)
"Collaborative Filtering for Implicit Feedback Datasets":

  confidence c_ui = 1 + alpha * r_ui
  preference  p_ui = 1 if r_ui > 0 else 0

We alternately solve closed-form least squares for U (holding V fixed)
and V (holding U fixed), each row independently, with L2 regularization.
This is what makes it "precomputed / 10ms latency" at serve time: once
trained, a recommendation is just a dot product + top-k, no gradient
descent at request time.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "als_model.pkl"


@dataclass
class ALSModel:
    user_factors: np.ndarray  # (n_users, k)
    item_factors: np.ndarray  # (n_items, k)
    user_id_to_idx: dict
    item_id_to_idx: dict
    idx_to_item_id: dict

    def score_user(self, user_id: int) -> np.ndarray | None:
        """Return raw scores over all items for a known user, else None (cold start)."""
        idx = self.user_id_to_idx.get(user_id)
        if idx is None:
            return None
        return self.user_factors[idx] @ self.item_factors.T

    def recommend(self, user_id: int, k: int = 100, exclude_item_ids: set[int] | None = None) -> list[tuple[int, float]]:
        scores = self.score_user(user_id)
        if scores is None:
            return []
        exclude_item_ids = exclude_item_ids or set()
        ranked_idx = np.argsort(-scores)
        results = []
        for idx in ranked_idx:
            item_id = self.idx_to_item_id[idx]
            if item_id in exclude_item_ids:
                continue
            results.append((item_id, float(scores[idx])))
            if len(results) >= k:
                break
        return results


def _build_confidence_matrix(interactions: list[dict], user_id_to_idx: dict, item_id_to_idx: dict, alpha: float = 15.0) -> csr_matrix:
    rows, cols, vals = [], [], []
    # aggregate weights per (user, item) pair first
    agg: dict[tuple[int, int], float] = {}
    for ev in interactions:
        u, i = ev["user_id"], ev["item_id"]
        if u not in user_id_to_idx or i not in item_id_to_idx:
            continue
        key = (user_id_to_idx[u], item_id_to_idx[i])
        agg[key] = agg.get(key, 0.0) + ev["weight"]

    for (u_idx, i_idx), w in agg.items():
        rows.append(u_idx)
        cols.append(i_idx)
        vals.append(1.0 + alpha * w)  # confidence, Hu et al.

    n_users = len(user_id_to_idx)
    n_items = len(item_id_to_idx)
    return csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


def train_als(
    interactions: list[dict],
    n_users_total: int,
    n_items_total: int,
    user_ids: list[int],
    item_ids: list[int],
    n_factors: int = 32,
    n_iterations: int = 15,
    reg: float = 0.1,
    alpha: float = 15.0,
    seed: int = 42,
) -> ALSModel:
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    idx_to_item_id = {i: iid for iid, i in item_id_to_idx.items()}

    C = _build_confidence_matrix(interactions, user_id_to_idx, item_id_to_idx, alpha=alpha)
    n_users, n_items = C.shape

    rng = np.random.default_rng(seed)
    U = 0.01 * rng.standard_normal((n_users, n_factors))
    V = 0.01 * rng.standard_normal((n_items, n_factors))

    C_dense = C.toarray()  # fine at this scale (2000 x 500); a real deployment would batch this
    P = (C_dense > 0).astype(np.float64)  # preference: 1 where any interaction occurred

    I_k = np.eye(n_factors)

    for iteration in range(n_iterations):
        # --- fix V, solve for U ---
        # Uses the standard ALS-for-implicit-feedback trick: instead of
        # forming the full (n_items x n_items) diagonal confidence matrix
        # per user (O(n^2) memory/compute), we exploit
        # V.T @ diag(c_u - 1) @ V == (V * (c_u - 1)[:, None]).T @ V
        # which is just elementwise scaling + a matmul.
        VtV = V.T @ V
        for u in range(n_users):
            c_u = C_dense[u] + 1.0  # confidence, +1 baseline for zero entries
            p_u = P[u]
            weighted_V = V * (c_u - 1.0)[:, None]
            A = VtV + weighted_V.T @ V + reg * I_k
            b = V.T @ (c_u * p_u)
            U[u] = np.linalg.solve(A, b)

        # --- fix U, solve for V ---
        UtU = U.T @ U
        for i in range(n_items):
            c_i = C_dense[:, i] + 1.0
            p_i = P[:, i]
            weighted_U = U * (c_i - 1.0)[:, None]
            A = UtU + weighted_U.T @ U + reg * I_k
            b = U.T @ (c_i * p_i)
            V[i] = np.linalg.solve(A, b)

    return ALSModel(
        user_factors=U,
        item_factors=V,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        idx_to_item_id=idx_to_item_id,
    )


def save_model(model: ALSModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = MODEL_PATH) -> ALSModel:
    with open(path, "rb") as f:
        return pickle.load(f)
