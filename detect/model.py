"""The classifier, the scoring, and the evaluation sets."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .embed import embeddings
from .splits import load_split

FAR_COSINE = 0.90  # test texts with no pool neighbour this close (robustness check)


def make_classifier(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000)


def scores(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Injection-class metrics; AUROC only when both classes are present."""
    pred = (prob >= threshold).astype(int)
    both = len(np.unique(y)) == 2
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "n": int(len(y)),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "auroc": round(float(roc_auc_score(y, prob)), 4) if both else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluation_sets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Name -> (vectors, labels) for every set and subgroup we report."""
    sets = {}
    test, xt = load_split("test"), embeddings("test")
    sets["test"] = (xt, test["label"].to_numpy())
    far = (xt @ embeddings("pool").T).max(axis=1) < FAR_COSINE
    sets["test_far"] = (xt[far], test["label"].to_numpy()[far])

    d, xd = load_split("ood_deepset"), embeddings("ood_deepset")
    sets["deepset"] = (xd, d["label"].to_numpy())
    en = (d["lang"] == "en").to_numpy()
    sets["deepset_en"] = (xd[en], d["label"].to_numpy()[en])
    sets["deepset_non_en"] = (xd[~en], d["label"].to_numpy()[~en])

    lab, xl = load_split("ood_lab"), embeddings("ood_lab")
    sets["lab"] = (xl, lab["label"].to_numpy())
    for kind in ["direct", "indirect", "leakage"]:
        m = (lab["kind"] == kind).to_numpy()
        sets[f"lab_{kind}"] = (xl[m], lab["label"].to_numpy()[m])
    benign = (lab["label"] == 0).to_numpy()
    sets["lab_benign"] = (xl[benign], lab["label"].to_numpy()[benign])
    return sets


def evaluate(clf, sets: dict) -> dict:
    return {name: scores(y, clf.predict_proba(x)[:, 1]) for name, (x, y) in sets.items()}
