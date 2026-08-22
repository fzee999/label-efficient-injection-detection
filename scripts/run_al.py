"""Run one strategy x labeler over all seeds.

    .venv\\Scripts\\python scripts\\run_al.py --strategy random --labeler perfect
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from detect.config import RESULTS, SEEDS  # noqa: E402
from detect.curve import median_censored, summarise  # noqa: E402
from detect.embed import embeddings  # noqa: E402
from detect.loop import PerfectLabeler, run  # noqa: E402
from detect.strategies import STRATEGIES  # noqa: E402
from detect.model import evaluation_sets  # noqa: E402
from detect.splits import load_split  # noqa: E402

LABELERS = {"perfect": PerfectLabeler}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    p.add_argument("--labeler", choices=sorted(LABELERS), default="perfect")
    args = p.parse_args()

    x, y = embeddings("pool"), load_split("pool")["label"].to_numpy()
    sets = evaluation_sets()
    labeler = LABELERS[args.labeler](y)

    rows, queries, t0 = [], [], time.perf_counter()
    for seed in SEEDS:
        r, q = run(args.strategy, labeler, seed, x, y, sets)
        rows += r
        queries += q
        print(f"seed {seed}: {len(r)} rounds, final test F1 {r[-1]['test_f1']:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    name = f"{args.strategy}_{args.labeler}"
    curves, qdir = RESULTS / "curves", RESULTS / "queries"
    curves.mkdir(parents=True, exist_ok=True)
    qdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(curves / f"al_{name}.csv", index=False)
    pd.DataFrame(queries).to_csv(qdir / f"{name}.csv", index=False)
    summary = summarise(df)
    (curves / f"al_{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for key, vals in summary.items():
        reached = sum(v is not None for v in vals)
        print(f"{key:11} median {median_censored(vals):>7}  reached {reached}/{len(vals)}  per seed {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
