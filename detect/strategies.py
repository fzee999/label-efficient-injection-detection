"""Query strategies: each picks the next k pool items from a loop Context."""
from __future__ import annotations

import numpy as np

HYBRID_PREFILTER = 10   # hybrid: take the 10*k most uncertain, then spread out


def _unlabeled(ctx) -> np.ndarray:
    return np.flatnonzero(~ctx.labeled)


def random(ctx, k: int) -> np.ndarray:
    """Next k unlabeled items in the seed's shuffled order."""
    return ctx.order[~ctx.labeled[ctx.order]][:k]


def _uncertainty_order(ctx, cand: np.ndarray) -> np.ndarray:
    """Candidates sorted most-uncertain first; random tie-break."""
    p = ctx.clf.predict_proba(ctx.x[cand])[:, 1]
    return cand[np.lexsort((ctx.rng.random(len(cand)), np.abs(p - 0.5)))]


def uncertainty(ctx, k: int) -> np.ndarray:
    """The k items whose predicted probability is closest to 0.5."""
    return _uncertainty_order(ctx, _unlabeled(ctx))[:k]


def _k_center(x: np.ndarray, cand: np.ndarray, centers: np.ndarray, k: int) -> np.ndarray:
    """Greedy k-center: repeatedly take the candidate farthest from every center."""
    dist = 1 - (x[cand] @ x[centers].T).max(axis=1)   # cosine distance to nearest center
    picked = []
    for _ in range(k):
        j = int(dist.argmax())
        picked.append(cand[j])
        dist = np.minimum(dist, 1 - x[cand] @ x[cand[j]])
        dist[j] = -1
    return np.array(picked)


def diversity(ctx, k: int) -> np.ndarray:
    """k items that best cover the pool, ignoring the model."""
    return _k_center(ctx.x, _unlabeled(ctx), np.flatnonzero(ctx.labeled), k)


def hybrid(ctx, k: int) -> np.ndarray:
    """Most uncertain shortlist, then k-center within it."""
    shortlist = _uncertainty_order(ctx, _unlabeled(ctx))[:HYBRID_PREFILTER * k]
    return _k_center(ctx.x, shortlist, np.flatnonzero(ctx.labeled), k)


STRATEGIES = {"random": random, "uncertainty": uncertainty, "diversity": diversity, "hybrid": hybrid}
