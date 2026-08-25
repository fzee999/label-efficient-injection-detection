"""Figures not made by earlier scripts: labeler calibration, the feedback loop,
human labels to target, and transfer by attack type.

    .venv\\Scripts\\python scripts\\final_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.style import COLORS, DPI, INK, INK2, REFERENCE, SURFACE, apply, tidy  # noqa: E402

FIGS = RESULTS / "figures"
STRATS = ["random", "uncertainty", "diversity", "hybrid"]
LABELERS = {"perfect": "dataset labels", "llm": "LLM labels only", "routed90": "LLM, human below 0.9"}
KEPT, ROUTED = "#52514e", "#c9c8c3"
LIGHT = 0.45   # alpha for the hatched (secondary) part of a bar


def save(fig, name: str) -> None:
    path = FIGS / name
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def calibration() -> None:
    acc = json.loads((RESULTS / "labeler" / "accuracy.json").read_text())
    bins = acc["calibration"]
    edges = [tuple(float(v) for v in b["confidence"].split("-")) for b in bins]
    at_09 = next(c for c in acc["coverage"] if c["threshold"] == 0.9)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
    tidy(ax)
    x = np.arange(len(bins))
    ax.bar(x, [b["accuracy"] for b in bins], width=0.72, edgecolor=SURFACE, linewidth=2,
           color=[KEPT if lo >= 0.9 else ROUTED for lo, _ in edges])
    ax.scatter(x, [(lo + hi) / 2 for lo, hi in edges], marker="_", s=420, linewidths=1.6, color=INK, zorder=3)
    for xi, b in zip(x, bins):
        ax.text(xi, b["accuracy"] + 0.015, f"{b['accuracy']:.0%}", ha="center", fontsize=8, color=INK)
    ax.axvline(3.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(3.42, 1.1, "sent to the human", ha="right", fontsize=8, color=INK2)
    ax.text(3.58, 1.1, f"LLM label kept: {at_09['share_confident']:.0%} of texts, "
            f"{at_09['accuracy_confident']:.1%} correct", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{lo:.2f}–{hi:.2f}\n{b['texts']:,}" for (lo, hi), b in zip(edges, bins)], fontsize=8)
    ax.set_xlabel("LLM's confidence in its own label (texts per bin below)")
    ax.set_ylabel("share matching the dataset label")
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(handles=[Line2D([], [], marker="_", markersize=14, linestyle="none", color=INK,
                              label="accuracy if confidence were exact")],
              frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.93))
    ax.set_title(f"The LLM labeler's confidence tracks its accuracy ({acc['scored_texts']:,} pool texts)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "labeler_calibration.png")


def feedback_loop() -> None:
    comp = {lab: pd.read_csv(RESULTS / f"strategy_comparison_{lab}.csv").set_index("strategy") for lab in LABELERS}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [2, 3]})
    w = 0.2
    for ax, labs in [(a, ["llm", "routed90"]), (b, list(LABELERS))]:
        tidy(ax)
        for i, s in enumerate(STRATS):
            xs = np.arange(len(labs)) + (i - 1.5) * w
            if ax is a:
                missed = np.array([comp[lab].loc[s, "wrong_label_missed_injection"] for lab in labs])
                false = np.array([comp[lab].loc[s, "wrong_label_false_injection"] for lab in labs])
                ax.bar(xs, missed, w, color=COLORS[s], edgecolor=SURFACE, linewidth=1, label=s)
                ax.bar(xs, false, w, bottom=missed, color=COLORS[s], alpha=LIGHT, hatch="////",
                       edgecolor=SURFACE, linewidth=1)
                tops = missed + false
            else:
                tops = np.array([comp[lab].loc[s, "near_copies_in_labeled_set"] for lab in labs])
                ax.bar(xs, tops, w, color=COLORS[s], edgecolor=SURFACE, linewidth=1)
            for xi, t in zip(xs, tops):
                ax.text(xi, t + 2, f"{t:.0f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(labs)), [LABELERS[lab] for lab in labs])
    a.set_title("Wrong labels among the 600 each strategy chose", loc="left", fontsize=10)
    a.set_ylabel("labels (mean of 10 seeds)")
    a.legend(handles=[Patch(facecolor=INK2, label="injection labeled benign"),
                      Patch(facecolor=INK2, alpha=LIGHT, hatch="////", edgecolor=SURFACE,
                            label="benign labeled injection")],
             frameon=False, fontsize=8, loc="upper right")
    b.set_title("Near-copies among those 600 (another chosen text at cosine ≥ 0.95)", loc="left", fontsize=10)
    b.set_ylabel("texts (mean of 10 seeds)")
    handles, names = a.get_legend_handles_labels()
    fig.legend(handles[:4], names[:4], loc="upper left", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.05, 0.95))
    fig.suptitle("With LLM labels, uncertainty sampling piles up the labeler's misses and keeps asking about them",
                 x=0.06, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, "feedback_loop.png")


def human_labels() -> None:
    names = {"perfect": "all from the human", "routed90": "human below 0.9", "routed80": "human below 0.8"}
    t = pd.read_csv(RESULTS / "analysis" / "labels_to_target.csv")
    t = t[t.labeler.isin(names) & (t.strategy != "diversity")].copy()
    t["human"] = pd.to_numeric(t.human_to_test_99, errors="coerce")
    t["total"] = pd.to_numeric(t.test_99.str.split().str[0], errors="coerce")
    t = t.dropna(subset=["human"]).sort_values("human")

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    tidy(ax, "x")
    y = np.arange(len(t))
    colors = [COLORS[s] for s in t.strategy]
    ax.barh(y, t.total, 0.66, color=colors, alpha=0.25, edgecolor=SURFACE, linewidth=2)
    ax.barh(y, t.human, 0.66, color=colors, edgecolor=SURFACE, linewidth=2)
    for yi, r in zip(y, t.itertuples()):
        txt = f"{r.human:g}" if r.labeler == "perfect" else f"{r.human:g} human, of {r.total:g} labels"
        ax.text(r.total + 4, yi, txt, va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{r.strategy}, {names[r.labeler]}" for r in t.itertuples()])
    ax.set_xlabel("labels to reach 99% of the full-data F1 (median of 10 seeds)")
    ax.set_xlim(0, 330)
    ax.legend(handles=[Patch(facecolor=INK2, label="from the human"),
                       Patch(facecolor=INK2, alpha=0.25, label="from the LLM")],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Human labels needed for a near-ceiling detector", loc="left", fontsize=10)
    fig.text(0.01, 0.01, "Every run shown reaches the target on all 10 seeds. With LLM labels alone, random and "
             "uncertainty sampling never reach it within 600;\ndiversity sampling never reaches it with any labeler.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "human_labels_to_target.png")


def transfer() -> None:
    sets = [("test_recall", "Test set"), ("test_far_recall", "Test, no close pool match"),
            ("lab_indirect_recall", "Lab: document payloads"), ("lab_leakage_recall", "Lab: leakage"),
            ("lab_direct_recall", "Lab: direct attacks"), ("deepset_recall", "deepset")]
    detectors = [("all 8,084 dataset labels", "all 8,084 dataset labels", REFERENCE, None),
                 ("uncertainty / perfect", "uncertainty, 600 dataset labels", COLORS["uncertainty"], None),
                 ("hybrid / routed90", "hybrid, 600 labels, human below 0.9 (best)", COLORS["hybrid"], None),
                 ("hybrid / llm", "hybrid, 600 LLM labels", COLORS["hybrid"], "////")]
    t = pd.read_csv(RESULTS / "transfer" / "transfer_summary.csv").set_index("detector")

    fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=SURFACE)
    tidy(ax, "x")
    h = 0.19
    for i, (key, label, color, hatch) in enumerate(detectors):
        ys = np.arange(len(sets)) + (i - 1.5) * h
        vals = [t.loc[key, col] for col, _ in sets]
        ax.barh(ys, vals, h, color=color, alpha=LIGHT if hatch else 1, hatch=hatch,
                edgecolor=SURFACE, linewidth=1, label=label)
        if key == "hybrid / routed90":
            for yi, v in zip(ys, vals):
                ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=7.5, color=INK)
    ax.axhline(1.5, color=INK2, linewidth=0.8)
    ax.text(1.12, 0.5, "in\ndistribution", fontsize=8, color=INK2, va="center", ha="right")
    ax.text(1.12, 3.5, "outside", fontsize=8, color=INK2, va="center", ha="right")
    ax.set_yticks(range(len(sets)), [name for _, name in sets])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("attacks caught (recall, mean of 10 seeds)")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2)
    ax.set_title("Detectors catch attacks that look like the training data, and miss disguised ones",
                 loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "transfer_recall.png")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    apply()
    calibration()
    feedback_loop()
    human_labels()
    transfer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
