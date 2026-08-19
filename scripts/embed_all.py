"""Embed every split once and sanity-check the vectors."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.embed import embeddings  # noqa: E402
from detect.splits import load_split  # noqa: E402

SPLITS = ["pool", "test", "ood_deepset", "ood_lab"]


def main() -> int:
    report = {}
    vecs = {}
    for name in SPLITS:
        t0 = time.perf_counter()
        x = embeddings(name)
        vecs[name] = x
        norms = np.linalg.norm(x, axis=1)
        report[name] = {"rows": int(x.shape[0]), "dim": int(x.shape[1]),
                        "seconds": round(time.perf_counter() - t0, 1),
                        "nan": int(np.isnan(x).sum()),
                        "norm_min": round(float(norms.min()), 4), "norm_max": round(float(norms.max()), 4)}
        print(name, report[name], flush=True)

    # does meaning separate the classes? nearest pool neighbour of each test text
    pool_y = load_split("pool")["label"].to_numpy()
    test_y = load_split("test")["label"].to_numpy()
    sim = vecs["test"] @ vecs["pool"].T
    nn, best = sim.argmax(axis=1), sim.max(axis=1)
    same = pool_y[nn] == test_y
    report["test_nearest_pool_neighbour_same_label_pct"] = round(100 * float(same.mean()), 1)

    # near-copies across the split, which exact-match cleaning cannot catch
    bands = {}
    for lo, hi in [(0.99, 2), (0.95, 0.99), (0.90, 0.95), (0.0, 0.90)]:
        m = (best >= lo) & (best < hi)
        bands[f"{lo:.2f}+" if hi == 2 else f"{lo:.2f}-{hi:.2f}"] = {
            "test_texts": int(m.sum()), "nn_same_label_pct": round(100 * float(same[m].mean()), 1)}
    report["test_nearest_pool_cosine_bands"] = bands

    pp = vecs["pool"] @ vecs["pool"].T
    np.fill_diagonal(pp, 0)
    report["pool_texts_with_near_copy_in_pool"] = int((pp.max(axis=1) >= 0.99).sum())

    unit = lambda v: v / np.linalg.norm(v)
    centroid = {c: unit(vecs["pool"][pool_y == c].mean(axis=0)) for c in (0, 1)}
    report["mean_cos_to_own_centroid"] = {
        f"class{c}": round(float((vecs["pool"][pool_y == c] @ centroid[c]).mean()), 3) for c in (0, 1)}
    report["cos_between_class_centroids"] = round(float(centroid[0] @ centroid[1]), 3)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "embedding_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in SPLITS}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
