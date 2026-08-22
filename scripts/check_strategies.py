"""Property checks for every strategy, plus one seed of each for a first look."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.curve import Constant, fit  # noqa: E402
from detect.embed import embeddings  # noqa: E402
from detect.loop import BATCH, Context, PerfectLabeler, run  # noqa: E402
from detect.model import evaluation_sets  # noqa: E402
from detect.splits import load_split  # noqa: E402
from detect.strategies import STRATEGIES, uncertainty  # noqa: E402


def check(name: str, ok: bool) -> bool:
    print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return ok


def main() -> int:
    x, y = embeddings("pool"), load_split("pool")["label"].to_numpy()
    sets = evaluation_sets()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(x))
    labeled = np.zeros(len(x), bool)
    labeled[order[:40]] = True
    clf = fit(x[labeled], y[labeled])
    good = True

    for name, select in STRATEGIES.items():
        ctx = Context(x, labeled, clf, order, np.random.default_rng(1))
        picks = select(ctx, BATCH)
        again = select(Context(x, labeled, clf, order, np.random.default_rng(1)), BATCH)
        good &= check(f"{name}: {BATCH} distinct, all unlabeled, repeatable",
                      len(set(picks)) == BATCH and not labeled[picks].any() and np.array_equal(picks, again))

    p = clf.predict_proba(x)[:, 1]
    gap = np.abs(p - 0.5)
    picks = uncertainty(Context(x, labeled, clf, order, np.random.default_rng(1)), BATCH)
    good &= check("uncertainty picks the 20 unlabeled items nearest 0.5",
                  gap[picks].max() <= np.sort(gap[~labeled])[BATCH - 1] + 1e-12)

    one_class = Context(x, labeled, Constant(0.0), order, np.random.default_rng(2))
    good &= check("uncertainty survives a one-class model (random tie-break)",
                  len(set(uncertainty(one_class, BATCH))) == BATCH)

    rows = {s: run(s, PerfectLabeler(y), 0, x, y, sets)[0] for s in STRATEGIES}
    base = pd.read_csv(RESULTS / "curves" / "random_perfect.csv").query("seed == 0").set_index("budget")
    r = pd.DataFrame(rows["random"]).set_index("budget")
    good &= check("random strategy still reproduces the step 06 curve",
                  float((r.test_f1 - base.loc[r.index, "test_f1"]).abs().max()) == 0.0)

    print("\nseed 0, perfect labels:")
    print(f"{'strategy':12} {'F1@40':>6} {'F1@100':>7} {'F1@600':>7} {'batch cos':>9} {'batch inj%':>10}")
    for s, rr in rows.items():
        d = pd.DataFrame(rr).set_index("budget")
        later = d[d["round"] > 0]
        print(f"{s:12} {d.loc[40, 'test_f1']:6.3f} {d.loc[100, 'test_f1']:7.3f} {d.loc[600, 'test_f1']:7.3f} "
              f"{later.batch_mean_cos.mean():9.3f} {100 * later.batch_injection_share.mean():9.0f}%")
    print("\nALL CHECKS PASSED" if good else "\nSOME CHECKS FAILED")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
