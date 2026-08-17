"""Write the fixed splits to data/splits and the lab OOD set to data/ood/lab.csv.

    .venv\\Scripts\\python scripts\\build_splits.py --lab "<path to Prompt Injection Lab>"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from detect.config import RESULTS, SPLITS  # noqa: E402
from detect.splits import LAB_FILE, build_ood_deepset, build_pool_and_test  # noqa: E402


_ISO = r"\d{4}-\d{2}-\d{2}"


def _undate(text: str) -> str:
    """Drop the dates in the source texts; they carry no signal for detection."""
    text = re.sub(rf"Effective {_ISO} · ", "", text)
    text = re.sub(rf" \({_ISO}\)", "", text)
    return re.sub(_ISO, "", text)


def build_lab(lab: Path) -> tuple[pd.DataFrame, dict]:
    """Attacks are injections (label 1); benign questions and policy docs are not (0)."""
    rows = []
    for f in sorted((lab / "data" / "attacks").glob("*.toml")):
        for a in tomllib.loads(f.read_text(encoding="utf-8"))["attack"]:
            # an indirect attack reaches a detector as document text, so score its payload
            text = a["payload"] if a["type"] == "indirect" else a["prompt"]
            rows.append({"id": a["id"], "text": text, "label": 1, "kind": a["type"]})
    sys.path.insert(0, str(lab))
    from lab.benign import BENIGN  # noqa: E402
    for b in BENIGN:
        rows.append({"id": b.id, "text": b.question, "label": 0, "kind": "benign_question"})
    for f in sorted((lab / "data" / "docs").glob("*.md")):
        rows.append({"id": f.stem, "text": f.read_text(encoding="utf-8"), "label": 0, "kind": "policy_doc"})
    df = pd.DataFrame(rows)
    df["text"] = df["text"].map(_undate)
    return df, df["kind"].value_counts().to_dict()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lab", required=True, type=Path)
    args = p.parse_args()

    SPLITS.mkdir(parents=True, exist_ok=True)
    LAB_FILE.parent.mkdir(parents=True, exist_ok=True)

    pool, test, log_main = build_pool_and_test()
    deepset, log_deepset = build_ood_deepset()
    lab, log_lab = build_lab(args.lab)

    pool.to_csv(SPLITS / "pool.csv", index=False)
    test.to_csv(SPLITS / "test.csv", index=False)
    deepset.to_csv(SPLITS / "ood_deepset.csv", index=False)
    lab.to_csv(LAB_FILE, index=False)

    report = {"pool_and_test": log_main, "ood_deepset": log_deepset, "ood_lab": log_lab,
              "label_balance": {
                  "pool_injection_pct": round(100 * pool["label"].mean(), 1),
                  "test_injection_pct": round(100 * test["label"].mean(), 1),
                  "ood_deepset_injection_pct": round(100 * deepset["label"].mean(), 1)}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
