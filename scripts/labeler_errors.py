"""Score the LLM labeler against the dataset labels: accuracy, confusion, calibration, error tags."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.labeler import dev_indices  # noqa: E402
from detect.splits import load_split  # noqa: E402

TAGS = {   # H2's terms first, then patterns added after reading the disagreements
    "mentions_instructions": r"\b(?:instructions?|rules?|prompts?|guidelines?|polic(?:y|ies)|system)\b",
    "mentions_ai": r"\b(?:AI|assistant|chatbot|bot|language model|ChatGPT|GPT)\b",
    "override_phrase": r"\b(?:ignore|disregard|forget|override|bypass)\b",
    "persona": r"\b(?:you are|act as|pretend|role ?play|from now on)\b",
    "sensitive_request": r"\b(?:confidential|passwords?|personal|private|medical records|credit card|secret|hack\w*)\b",
    "task_format": r"(?:in this task|your task is|you will be given|you are given|^given (?:a|an|the) )",
    "act_as_request": r"\b(?:act as|i want you to act)\b",
    "jailbreak_persona": r"\b(?:no restrictions|without any rules|stay in character|unfiltered|DAN|jailbreak|developer mode|two (?:different|distinct) (?:separate )?(?:ways|answers))\b",
    "completion_leak": r"\b(?:starts? with|begins? with|complete (?:this|the) (?:statement|sentence))\b",
}
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def main() -> int:
    pool = load_split("pool")
    lab = pd.read_csv(RESULTS / "labeler" / "pool_labels.csv").set_index("pool_index").sort_index()
    df = pool.join(lab, rsuffix="_llm").rename(columns={"label": "true", "label_llm": "llm"})
    df["dev"] = False
    df.loc[dev_indices(pool["label"]), "dev"] = True
    df["words"] = df.text.str.split().str.len()
    for tag, pat in TAGS.items():
        df[tag] = df.text.str.contains(pat, flags=re.I, regex=True)
    df["long"] = df.words > 200
    df["conf"] = np.maximum(df.p_injection, 1 - df.p_injection)

    s = df[~df.dev]   # score on texts never used to shape the prompt
    tp = int(((s.llm == 1) & (s.true == 1)).sum()); fp = int(((s.llm == 1) & (s.true == 0)).sum())
    fn = int(((s.llm == 0) & (s.true == 1)).sum()); tn = int(((s.llm == 0) & (s.true == 0)).sum())
    report = {
        "scored_texts": len(s), "excluded_dev_texts": int(df.dev.sum()),
        "accuracy": round(float((s.llm == s.true).mean()), 4),
        "cohen_kappa": round(float(cohen_kappa_score(s.true, s.llm)), 4),
        "auroc": round(float(roc_auc_score(s.true, s.p_injection)), 4),
        "precision": round(tp / (tp + fp), 4), "recall": round(tp / (tp + fn), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "errors": {"false_positive_share": round(fp / (fp + fn), 3), "false_negative_share": round(fn / (fp + fn), 3)},
    }

    report["calibration"] = []
    for lo, hi in BINS:
        m = (s.conf >= lo) & (s.conf < hi)
        report["calibration"].append({"confidence": f"{lo:.2f}-{min(hi, 1):.2f}", "texts": int(m.sum()),
                                      "accuracy": round(float((s.llm[m] == s.true[m]).mean()), 4) if m.any() else None})
    report["coverage"] = []
    for t in THRESHOLDS:
        hi_conf = s.conf >= t
        report["coverage"].append({"threshold": t, "share_confident": round(float(hi_conf.mean()), 4),
                                   "accuracy_confident": round(float((s.llm[hi_conf] == s.true[hi_conf]).mean()), 4),
                                   "accuracy_rest": round(float((s.llm[~hi_conf] == s.true[~hi_conf]).mean()), 4)})

    inj, ben = s[s.true == 1], s[s.true == 0]
    fns, fps = inj[inj.llm == 0], ben[ben.llm == 1]
    report["tags_missed_injections"] = {t: {"all_injections_pct": round(100 * inj[t].mean(), 1),
                                            "missed_pct": round(100 * fns[t].mean(), 1)} for t in [*TAGS, "long"]}
    report["tags_false_alarms"] = {t: {"all_benign_pct": round(100 * ben[t].mean(), 1),
                                       "false_alarm_pct": round(100 * fps[t].mean(), 1)} for t in [*TAGS, "long"]}

    out = RESULTS / "labeler"
    (out / "accuracy.json").write_text(json.dumps(report, indent=2) + "\n")
    tagged = df[["true", "llm", "p_injection", "conf", "dev", "words", *TAGS, "long"]]
    tagged[tagged.true != tagged.llm].to_csv(out / "disagreements.csv")   # row numbers only, no text
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
