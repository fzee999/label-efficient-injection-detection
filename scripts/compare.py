"""Compare strategies for one labeler: targets, curve area, paired wins vs random.

    .venv\\Scripts\\python scripts\\compare.py --labeler perfect
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.curve import median_censored, summarise  # noqa: E402
from detect.embed import embeddings  # noqa: E402
from detect.loop import make_labeler  # noqa: E402
from detect.splits import load_split  # noqa: E402
from detect.strategies import STRATEGIES  # noqa: E402


BUDGETS = [40, 100, 200, 600]
NEAR_COPY = 0.95


def labeled_set_stats(strategy: str, labeler: str, x: np.ndarray, y: np.ndarray, observed: np.ndarray) -> dict:
    """Over the final 600 labels: wrong labels by direction, and near-duplicates."""
    q = pd.read_csv(RESULTS / "queries" / f"{strategy}_{labeler}.csv")
    rows = []
    for _, g in q.groupby("seed"):
        i = g.pool_index.to_numpy()
        sim = x[i] @ x[i].T
        np.fill_diagonal(sim, 0)
        rows.append([((y[i] == 1) & (observed[i] == 0)).sum(), ((y[i] == 0) & (observed[i] == 1)).sum(),
                     (sim.max(axis=1) >= NEAR_COPY).sum()])
    a = np.mean(rows, axis=0)
    return {"wrong_label_missed_injection": round(float(a[0]), 1),
            "wrong_label_false_injection": round(float(a[1]), 1),
            "near_copies_in_labeled_set": round(float(a[2]), 1)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labeler", default="perfect")
    args = p.parse_args()

    curves = {s: pd.read_csv(f) for s in STRATEGIES
              if (f := RESULTS / "curves" / f"al_{s}_{args.labeler}.csv").exists()}
    base = summarise(curves["random"])["auc_20_600"]
    x, y = embeddings("pool"), load_split("pool")["label"].to_numpy()
    observed = make_labeler(args.labeler, y).y
    rows = []
    for s, df in curves.items():
        sm = summarise(df)
        auc = pd.Series(sm["auc_20_600"])
        later = df[df["round"] > 0]
        at600 = df[df.budget == 600]
        row = {"strategy": s}
        for key in ("target_95", "test_99", "far_99", "target_95_human", "test_99_human", "far_99_human"):
            row[key] = median_censored(sm[key])
            row[f"{key}_reached"] = sum(v is not None for v in sm[key])
        row["auc_mean"], row["auc_std"] = round(auc.mean(), 4), round(auc.std(), 4)
        row["auc_wins_vs_random"] = int(sum(a > b for a, b in zip(sm["auc_20_600"], base)))
        for b in BUDGETS:
            row[f"f1_at_{b}"] = round(df[df.budget == b].test_f1.mean(), 3)
        row["far_f1_at_600"] = round(at600.test_far_f1.mean(), 3)
        row["deepset_f1_at_600"] = round(at600.deepset_f1.mean(), 3)
        row["deepset_auroc_at_600"] = round(at600.deepset_auroc.mean(), 3)
        row["lab_f1_at_600"] = round(at600.lab_f1.mean(), 3)
        row["batch_cos"] = round(later.batch_mean_cos.mean(), 3)
        row["batch_injection_pct"] = round(100 * later.batch_injection_share.mean(), 1)
        row.update(labeled_set_stats(s, args.labeler, x, y, observed))
        rows.append(row)

    table = pd.DataFrame(rows)
    out = RESULTS / f"strategy_comparison_{args.labeler}.csv"
    table.to_csv(out, index=False)
    (RESULTS / f"strategy_comparison_{args.labeler}.json").write_text(
        json.dumps(rows, indent=2) + "\n")
    print(table.set_index("strategy").T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
