"""RQ5 / H5: does the best detector transfer to outside attacks?

Best detector = highest mean test F1 at 600 labels across all runs (fixed before
computing transfer). H5 made precise beforehand: OOD recall >= 10 points below
in-distribution recall, and indirect (document) payloads lowest of the three types.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.curve import fit  # noqa: E402
from detect.embed import embeddings  # noqa: E402
from detect.loop import make_labeler  # noqa: E402
from detect.model import FAR_COSINE, make_classifier  # noqa: E402
from detect.splits import load_split  # noqa: E402

BEST = ("hybrid", "routed90")
REFERENCES = [("uncertainty", "perfect"), ("hybrid", "llm")]
OUT = RESULTS / "transfer"


def eval_frames() -> dict:
    test, lab, deep = load_split("test"), load_split("ood_lab"), load_split("ood_deepset")
    xt, xl, xd = embeddings("test"), embeddings("ood_lab"), embeddings("ood_deepset")
    far = (xt @ embeddings("pool").T).max(axis=1) < FAR_COSINE
    return {"test": (xt, test), "test_far": (xt[far], test[far].reset_index(drop=True)),
            "lab": (xl, lab), "deepset": (xd, deep)}


def recall(p, y):
    return float(((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1))


def flag_rate(p, mask):
    return float((p[mask] >= 0.5).mean()) if mask.any() else float("nan")


def measures(clf, frames) -> dict:
    out = {}
    for name in ("test", "test_far"):
        x, df = frames[name]
        out[f"{name}_recall"] = recall(clf.predict_proba(x)[:, 1], df.label.to_numpy())
    x, lab = frames["lab"]
    p = clf.predict_proba(x)[:, 1]
    for kind in ("direct", "indirect", "leakage"):
        m = (lab.kind == kind).to_numpy()
        out[f"lab_{kind}_recall"] = float((p[m] >= 0.5).mean())
    out["lab_attack_recall"] = recall(p, lab.label.to_numpy())
    out["lab_benign_questions_flagged"] = flag_rate(p, (lab.kind == "benign_question").to_numpy())
    out["lab_policy_docs_flagged"] = flag_rate(p, (lab.kind == "policy_doc").to_numpy())
    out["lab_auroc"] = float(roc_auc_score(lab.label, p))
    x, deep = frames["deepset"]
    p = clf.predict_proba(x)[:, 1]
    y = deep.label.to_numpy()
    en = (deep.lang == "en").to_numpy()
    out["deepset_recall"] = recall(p, y)
    out["deepset_en_recall"] = recall(p[en], y[en])
    out["deepset_non_en_recall"] = recall(p[~en], y[~en])
    out["deepset_roleplay_recall"] = recall(p[deep.roleplay.to_numpy()], y[deep.roleplay.to_numpy()])
    out["deepset_auroc"] = float(roc_auc_score(y, p))
    return out


def final_models(strategy: str, labeler: str):
    x, y = embeddings("pool"), load_split("pool")["label"].to_numpy()
    labels = make_labeler(labeler, y).y
    q = pd.read_csv(RESULTS / "queries" / f"{strategy}_{labeler}.csv")
    return [fit(x[g.pool_index.to_numpy()], labels[g.pool_index.to_numpy()]) for _, g in q.groupby("seed")]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = eval_frames()
    rows, per_attack = [], []
    lab = frames["lab"][1]
    for strategy, labeler in [BEST, *REFERENCES]:
        for seed, clf in enumerate(final_models(strategy, labeler)):
            rows.append({"detector": f"{strategy} / {labeler}", "seed": seed, **measures(clf, frames)})
            if (strategy, labeler) == BEST:
                p = clf.predict_proba(frames["lab"][0])[:, 1]
                per_attack += [{"seed": seed, "id": i, "kind": k, "score": s}
                               for i, k, s in zip(lab.id, lab.kind, p)]
    full = make_classifier().fit(embeddings("pool"), load_split("pool")["label"].to_numpy())
    rows.append({"detector": "all 8,084 dataset labels", "seed": -1, **measures(full, frames)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "transfer_by_seed.csv", index=False)
    summary = df.drop(columns="seed").groupby("detector", sort=False).mean().round(3)
    summary.to_csv(OUT / "transfer_summary.csv")

    pa = pd.DataFrame(per_attack)
    misses = (pa.assign(missed=pa.score < 0.5).groupby(["id", "kind"])
              .agg(mean_score=("score", "mean"), seeds_missed=("missed", "sum")).reset_index())
    misses.to_csv(OUT / "best_detector_per_lab_text.csv", index=False)

    print(summary.T.to_string())
    best = df[df.detector == f"{BEST[0]} / {BEST[1]}"]
    gap = 100 * (best.test_recall - best.lab_attack_recall)
    kinds = {k: best[f"lab_{k}_recall"].mean() for k in ("direct", "indirect", "leakage")}
    print(f"\nH5 part 1: test recall minus lab attack recall = {gap.mean():.1f} points "
          f"(range {gap.min():.1f} to {gap.max():.1f}); needs >= 10")
    print(f"H5 part 2: lowest-recall attack type = {min(kinds, key=kinds.get)} {kinds}; needs 'indirect'")
    attacks = misses[misses.kind.isin(["direct", "indirect", "leakage"])].sort_values("mean_score")
    print("\nlab attacks missed on most seeds (best detector):")
    print(attacks[attacks.seeds_missed >= 5].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
