"""Random-labeling learning curve over 10 seeds; writes results/curves/random_perfect.csv."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detect.config import RESULTS, SEEDS  # noqa: E402
from detect.curve import TARGET_F1, fit, score_row, summarise  # noqa: E402
from detect.embed import embeddings  # noqa: E402
from detect.model import evaluation_sets  # noqa: E402
from detect.splits import load_split  # noqa: E402

BUDGETS = list(range(10, 610, 10)) + [800, 1000, 1500, 2000, 3000, 4000, 6000, 8084]


def main() -> int:
    x, y = embeddings("pool"), load_split("pool")["label"].to_numpy()
    sets = evaluation_sets()
    rows, t0 = [], time.perf_counter()
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(y))  # nested: budget b = first b items
        for b in BUDGETS:
            idx = order[:b]
            rows.append({"strategy": "random", "labeler": "perfect", "seed": seed, "budget": b,
                         "injections_labeled": int(y[idx].sum()), **score_row(fit(x[idx], y[idx]), sets)})
        print(f"seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "curves"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "random_perfect.csv", index=False)

    show = [10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 600, 1000, 2000, 8084]
    g = df[df.budget.isin(show)].groupby("budget")
    summary = pd.DataFrame({
        "test_f1": g.test_f1.mean().round(3).astype(str) + " ± " + g.test_f1.std().round(3).astype(str),
        "test_far_f1": g.test_far_f1.mean().round(3),
        "deepset_f1": g.deepset_f1.mean().round(3),
        "deepset_auroc": g.deepset_auroc.mean().round(3),
        "lab_f1": g.lab_f1.mean().round(3),
        "min_injections": g.injections_labeled.min(),
    })
    print(f"\nmean ± std over {len(SEEDS)} seeds; target test F1 = {TARGET_F1}\n")
    print(summary.to_string())

    summary = summarise(df)
    (out / "random_perfect_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print()
    for key, vals in summary.items():
        done = [v for v in vals if v is not None]
        mid = np.median(done) if done else float("nan")
        print(f"{key:11} median {mid:8.4g}   per seed {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
