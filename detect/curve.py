"""Learning curves: train on a labeled subset, score the evaluation sets."""
from __future__ import annotations

import numpy as np

from .model import make_classifier, scores

CEILING_F1 = 0.983          # full-data test F1 (step 05)
CEILING_FAR_F1 = 0.947      # full-data F1 on the far test texts
TARGET_F1 = round(0.95 * CEILING_F1, 3)   # pre-registered target
# Added after the random curve showed the 95% target is reached in ~25 labels:
SECONDARY_TARGETS = {"test_99": ("test_f1", round(0.99 * CEILING_F1, 3)),
                     "far_99": ("test_far_f1", round(0.99 * CEILING_FAR_F1, 3))}
AUC_RANGE = (20, 600)       # protocol budgets
REPORT_SETS = ["test", "test_far", "deepset", "lab"]


class Constant:
    """Fallback when the labeled set has one class: predict its rate."""

    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.p), np.full(len(x), self.p)])


def fit(x: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        return Constant(float(y.mean()))
    return make_classifier().fit(x, y)


def score_row(clf, sets: dict) -> dict:
    row = {}
    for name in REPORT_SETS:
        xs, ys = sets[name]
        s = scores(ys, clf.predict_proba(xs)[:, 1])
        row[f"{name}_f1"], row[f"{name}_auroc"] = s["f1"], s["auroc"]
    return row


def labels_to_target(budgets: list[int], f1: list[float], target: float = TARGET_F1) -> int | None:
    """First budget at which F1 reaches the target; None if never."""
    return next((b for b, f in zip(budgets, f1) if f >= target), None)


def curve_area(budgets: list[int], f1: list[float], lo: int = AUC_RANGE[0], hi: int = AUC_RANGE[1]) -> float:
    """Area under the F1 curve between lo and hi labels, scaled to 0-1."""
    pts = [(b, f) for b, f in zip(budgets, f1) if lo <= b <= hi]
    b, f = zip(*pts)
    return float(np.trapezoid(f, b) / (hi - lo))


def median_censored(vals: list) -> str:
    """Median where None means 'not reached within the budget'."""
    x = sorted(np.inf if v is None else v for v in vals)
    m = float(np.median(x))
    return f"> {AUC_RANGE[1]}" if np.isinf(m) else f"{m:g}"


def summarise(df) -> dict:
    """Per-seed labels-to-target and curve area for one strategy/labeler."""
    out = {"target_95": [], **{k: [] for k in SECONDARY_TARGETS}, "auc_20_600": []}
    for _, g in df.groupby("seed"):
        g = g.sort_values("budget")
        b = g["budget"].tolist()
        out["target_95"].append(labels_to_target(b, g["test_f1"].tolist()))
        for key, (col, target) in SECONDARY_TARGETS.items():
            out[key].append(labels_to_target(b, g[col].tolist(), target))
        out["auc_20_600"].append(round(curve_area(b, g["test_f1"].tolist()), 4))
    return out
