"""The active-learning loop: label a seed set, then query, retrain, log, repeat."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RESULTS
from .curve import fit, score_row
from .strategies import STRATEGIES

SEED_SIZE, BATCH, BUDGET = 20, 20, 600   # pre-registered protocol


@dataclass
class Context:
    """What a strategy may look at when choosing the next batch."""
    x: np.ndarray            # pool vectors
    labeled: np.ndarray      # bool mask over the pool
    clf: object              # model trained on the labeled set
    order: np.ndarray        # this seed's shuffled pool order
    rng: np.random.Generator


class PerfectLabeler:
    """Returns the dataset's labels; every label counts as a human label."""
    name = "perfect"

    def __init__(self, y: np.ndarray):
        self.y = y

    def label(self, idx: np.ndarray) -> tuple[np.ndarray, int]:
        return self.y[idx], len(idx)


class LLMLabeler:
    """Returns the LLM's frozen labels from step 11; no human labels."""
    name = "llm"

    def __init__(self, y: np.ndarray):
        lab = pd.read_csv(RESULTS / "labeler" / "pool_labels.csv").sort_values("pool_index")
        if len(lab) != len(y) or not (lab.pool_index.to_numpy() == np.arange(len(y))).all():
            raise ValueError("LLM labels do not cover the pool exactly")
        self.y = lab.label.to_numpy()
        self.p = lab.p_injection.to_numpy()

    def label(self, idx: np.ndarray) -> tuple[np.ndarray, int]:
        return self.y[idx], 0


class RoutedLabeler:
    """The LLM labels; texts it is unsure about (confidence < tau) go to the human."""

    def __init__(self, y: np.ndarray, tau: float = 0.9):
        llm = LLMLabeler(y)
        to_human = np.maximum(llm.p, 1 - llm.p) < tau
        self.y = np.where(to_human, y, llm.y)   # the labels this labeler returns
        self.to_human = to_human
        self.name = f"routed{round(tau * 100)}"

    def label(self, idx: np.ndarray) -> tuple[np.ndarray, int]:
        return self.y[idx], int(self.to_human[idx].sum())


def make_labeler(name: str, y: np.ndarray):
    """'perfect', 'llm', or 'routedNN' (threshold NN/100)."""
    if name == "perfect":
        return PerfectLabeler(y)
    if name == "llm":
        return LLMLabeler(y)
    if name.startswith("routed"):
        return RoutedLabeler(y, int(name[len("routed"):]) / 100)
    raise ValueError(f"unknown labeler {name!r}")


def run(strategy: str, labeler, seed: int, x: np.ndarray, y_true: np.ndarray, sets: dict):
    """One full run. Returns (per-round rows, queried pool rows)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))           # same order as the random curve
    labeled = np.zeros(len(x), dtype=bool)
    y_obs = np.full(len(x), -1)
    rows, queries, human = [], [], 0

    batch = order[:SEED_SIZE]
    for rnd in range(BUDGET // BATCH):
        got, h = labeler.label(batch)
        y_obs[batch], labeled[batch], human = got, True, human + h
        queries += [{"seed": seed, "round": rnd, "pool_index": int(i)} for i in batch]

        idx = np.flatnonzero(labeled)
        clf = fit(x[idx], y_obs[idx])
        xb = x[batch]
        rows.append({
            "strategy": strategy, "labeler": labeler.name, "seed": seed, "round": rnd,
            "budget": int(labeled.sum()), "human_labels": human,
            "label_errors": int((y_obs[idx] != y_true[idx]).sum()),
            "injections_labeled": int(y_true[idx].sum()),
            "batch_injection_share": round(float(y_true[batch].mean()), 3),
            "batch_mean_cos": round(float((xb @ xb.T)[np.triu_indices(len(xb), 1)].mean()), 4),
            **score_row(clf, sets),
        })
        if labeled.sum() >= BUDGET:
            break
        batch = STRATEGIES[strategy](Context(x, labeled, clf, order, rng), BATCH)
    return rows, queries
