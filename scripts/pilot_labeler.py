"""Pilot the LLM labeler on 100 random pool texts; writes results/labeler/pilot.csv."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.labeler import label_text  # noqa: E402
from detect.splits import load_split  # noqa: E402

N, PILOT_SEED = 100, 7


def main() -> int:
    pool = load_split("pool")
    idx = np.random.default_rng(PILOT_SEED).choice(len(pool), N, replace=False)
    rows = []
    for k, i in enumerate(idx, 1):
        out = label_text(pool.text.iloc[i])
        rows.append({"pool_index": int(i), "true": int(pool.label.iloc[i]), **out})
        if k % 20 == 0:
            print(f"{k}/{N}", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS / "labeler"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pilot.csv", index=False)

    ok = df[df.ok]
    acc = (ok.label == ok.true).mean()
    tp = int(((ok.label == 1) & (ok.true == 1)).sum()); fp = int(((ok.label == 1) & (ok.true == 0)).sum())
    fn = int(((ok.label == 0) & (ok.true == 1)).sum()); tn = int(((ok.label == 0) & (ok.true == 0)).sum())
    print(f"\nanswered with A/B: {len(ok)}/{N}   accuracy {acc:.3f}   TP {tp} FP {fp} FN {fn} TN {tn}")
    print(f"median seconds per text {df.seconds.median():.2f}   letter mass median {df.letter_mass.median():.3f}")
    conf = np.maximum(ok.p_injection, 1 - ok.p_injection)
    for lo, hi in [(0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"  confidence {lo:.2f}-{min(hi, 1):.2f}: {m.sum():3} texts, accuracy {(ok.label[m] == ok.true[m]).mean():.3f}")
    if (~df.ok).any():
        print("non-A/B replies:", df[~df.ok].raw.tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
