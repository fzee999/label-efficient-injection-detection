"""Evidence for every LLM label: its raw answer, both letter probabilities, the dataset
label and the routing decision; checked against the published labeler scores.

    .venv\\Scripts\\python scripts\\evidence.py

Writes results/labeler/evidence.csv (no texts; committed) and, with the texts,
results/local/labeler_evidence_with_text.csv and results/local/labeler_transcripts.md
(git-ignored: the source dataset's license is not stated, so its texts stay local).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.labeler import dev_indices  # noqa: E402
from detect.splits import load_split  # noqa: E402

LAB = RESULTS / "labeler"
LOCAL = RESULTS / "local"
TAU = 0.9
NAMES = {1: "injection", 0: "benign"}


def build() -> tuple[pd.DataFrame, pd.Series]:
    pool = load_split("pool")
    raw = pd.read_csv(LAB / "pool_labels.csv").sort_values("pool_index").set_index("pool_index")
    assert len(raw) == len(pool), "pool_labels.csv does not cover the pool"
    p, mass = raw.p_injection, raw.letter_mass
    conf = np.maximum(p, 1 - p)
    dev = np.zeros(len(pool), bool)
    dev[dev_indices(pool.label.to_numpy())] = True
    ev = pd.DataFrame({
        "pool_index": raw.index,
        "source_row": pool.row.to_numpy(),
        "dataset_label": pool.label.to_numpy(),
        "llm_answer": raw.raw.astype(str).str.strip(),
        "p_a": (p * mass).round(4),
        "p_b": ((1 - p) * mass).round(4),
        "letter_mass": mass,
        "p_injection": p,
        "confidence": conf.round(4),
        "llm_label": raw.label,
        "agrees": raw.label.to_numpy() == pool.label.to_numpy(),
        "routed_0_9": np.where(conf >= TAU, "llm", "human"),
        "dev_text": dev,
        "answer_ok": raw.ok,
        "seconds": raw.seconds,
    })
    return ev, pool.text


def check(ev: pd.DataFrame) -> list[str]:
    """Recompute the published labeler scores from the evidence rows."""
    acc = json.loads((LAB / "accuracy.json").read_text())
    s = ev[~ev.dev_text]
    y, l = s.dataset_label, s.llm_label
    got = {"tp": int(((y == 1) & (l == 1)).sum()), "fp": int(((y == 0) & (l == 1)).sum()),
           "fn": int(((y == 1) & (l == 0)).sum()), "tn": int(((y == 0) & (l == 0)).sum())}
    kept = s[s.routed_0_9 == "llm"]
    cov = next(c for c in acc["coverage"] if c["threshold"] == TAU)
    letter = s.llm_answer.str.upper().map({"A": 1, "B": 0})
    lines = [
        f"scored texts: {len(s)} (published {acc['scored_texts']})",
        f"confusion: {got} (published {acc['confusion']})",
        f"accuracy: {s.agrees.mean():.4f} (published {acc['accuracy']})",
        f"kept by the LLM at {TAU}: {len(kept) / len(s):.4f} at {kept.agrees.mean():.4f} correct "
        f"(published {cov['share_confident']} at {cov['accuracy_confident']})",
        f"answer letter disagrees with p >= 0.5: {int((letter != l).sum())} of {len(s)}",
        f"no letter among the top tokens: {int((~s.answer_ok).sum())}",
    ]
    assert len(s) == acc["scored_texts"] and got == acc["confusion"], "evidence does not match accuracy.json"
    assert round(len(kept) / len(s), 4) == cov["share_confident"], "coverage does not match accuracy.json"
    return lines


def transcript(ev: pd.DataFrame, texts: pd.Series, summary: list[str]) -> str:
    meta = json.loads((LAB / "pool_labels.json").read_text())
    out = ["# LLM labeler transcripts", "",
           f"Model `{meta['model']}`, prompt fingerprint `{meta['prompt_fingerprint'][:16]}`. "
           "Each entry: the text exactly as labeled, the dataset's label, the LLM's one-token answer "
           "(A = injection, B = benign), p(injection) = P(A) / (P(A) + P(B)), and whether routing "
           f"at {TAU} keeps the LLM's label or sends the text to the human.", "",
           "## Check against the published scores", "", *[f"- {x}" for x in summary], ""]

    def entry(r) -> list[str]:
        body = str(texts.iloc[r.pool_index]).replace("````", "` ` ` `")
        return [f"### pool {r.pool_index} · source row {r.source_row}",
                f"dataset **{NAMES[r.dataset_label]}** · LLM answered **{r.llm_answer}** "
                f"({NAMES[r.llm_label]}) · P(A) {r.p_a:.4f}, P(B) {r.p_b:.4f} · "
                f"p(injection) {r.p_injection:.4f} · confidence {r.confidence:.4f} · "
                f"{'kept' if r.routed_0_9 == 'llm' else 'sent to the human'} at {TAU}", "",
                "````", body, "````", ""]

    wrong = ev[~ev.agrees & ~ev.dev_text].sort_values("confidence", ascending=False)
    out += [f"## Every disagreement ({len(wrong)}), most confident first", ""]
    for r in wrong.itertuples():
        out += entry(r)
    out += ["## The 40 development texts (used to choose the answer format; not scored)", ""]
    for r in ev[ev.dev_text].itertuples():
        out += entry(r)
    return "\n".join(out)


def main() -> int:
    ev, texts = build()
    summary = check(ev)
    print("\n".join(summary))
    ev.to_csv(LAB / "evidence.csv", index=False)
    LOCAL.mkdir(parents=True, exist_ok=True)
    ev.insert(2, "text", texts.to_numpy())
    ev.to_csv(LOCAL / "labeler_evidence_with_text.csv", index=False)
    (LOCAL / "labeler_transcripts.md").write_text(transcript(ev, texts, summary), encoding="utf-8")
    print(f"wrote {LAB / 'evidence.csv'} and {LOCAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
