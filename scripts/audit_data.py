"""Audit the datasets: size, balance, duplicates, label conflicts, overlap, length."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.data import SOURCES, load, normalize  # noqa: E402

EMBED_CHAR_LIMIT = 2048 * 4  # ~4 chars per token; longer texts get truncated


def audit_split(df: pd.DataFrame) -> dict:
    norm = df["text"].map(normalize)
    words = df["text"].str.split().str.len()
    chars = df["text"].str.len()
    conflicts = df.assign(n=norm).groupby("n")["label"].nunique()
    non_ascii = df["text"].map(lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1))
    return {
        "rows": len(df),
        "injection": int(df["label"].sum()),
        "injection_pct": round(100 * df["label"].mean(), 1),
        "empty": int((df["text"].str.strip() == "").sum()),
        "exact_duplicate_rows": int(df["text"].duplicated().sum()),
        "normalized_duplicate_rows": int(norm.duplicated().sum()),
        "texts_with_conflicting_labels": int((conflicts > 1).sum()),
        "words_median": int(words.median()),
        "words_p95": int(words.quantile(0.95)),
        "words_max": int(words.max()),
        "over_embed_limit": int((chars > EMBED_CHAR_LIMIT).sum()),
        "mostly_non_ascii": int((non_ascii > 0.2).sum()),
    }


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    exact = set(a["text"]) & set(b["text"])
    norm = set(a["text"].map(normalize)) & set(b["text"].map(normalize))
    return {"exact": len(exact), "normalized": len(norm)}


def main() -> int:
    report = {}
    for name, src in SOURCES.items():
        splits = {s: load(name, s) for s in src["splits"]}
        report[name] = {s: audit_split(df) for s, df in splits.items()}
        report[name]["train_test_overlap"] = overlap(splits["train"], splits["test"])
    report["cross_dataset_overlap"] = {
        "safeguard_train_vs_deepset_test": overlap(load("safeguard", "train"), load("deepset", "test")),
        "safeguard_train_vs_deepset_train": overlap(load("safeguard", "train"), load("deepset", "train")),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
