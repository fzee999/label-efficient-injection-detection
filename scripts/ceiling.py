"""Full-data ceiling: train on every pool label, score every evaluation set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.embed import embeddings  # noqa: E402
from detect.model import evaluate, evaluation_sets, make_classifier  # noqa: E402
from detect.splits import load_split  # noqa: E402

GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def main() -> int:
    x, y = embeddings("pool"), load_split("pool")["label"].to_numpy()
    sets = evaluation_sets()

    # is the fixed C=1 a fair choice? 5-fold cross-validation on the pool only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cv_f1 = {C: round(float(cross_val_score(make_classifier(C), x, y, cv=cv, scoring="f1").mean()), 4)
             for C in GRID}
    best_C = max(cv_f1, key=cv_f1.get)

    report = {"pool_rows": len(y), "cv_f1_by_C": cv_f1, "best_C": best_C,
              "C1": evaluate(make_classifier(1.0).fit(x, y), sets)}
    if best_C != 1.0:
        report["best_C_scores"] = evaluate(make_classifier(best_C).fit(x, y), sets)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ceiling.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"cross-validated F1 by C: {cv_f1}  -> best C = {best_C}\n")
    print(f"{'set':16} {'n':>5} {'F1':>7} {'prec':>7} {'recall':>7} {'AUROC':>7} {'FPR':>7}")
    for name, s in report["C1"].items():
        fmt = lambda v: f"{v:7.3f}" if v is not None else "      -"
        print(f"{name:16} {s['n']:5} {fmt(s['f1'])} {fmt(s['precision'])} {fmt(s['recall'])} "
              f"{fmt(s['auroc'])} {fmt(s['false_positive_rate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
