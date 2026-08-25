"""Sweep the routing threshold: human effort versus detector quality.

    .venv\\Scripts\\python scripts\\sweep_threshold.py

Runs any missing routed runs, then writes results/threshold_sweep.csv and
results/figures/threshold_sweep.png. LLM-only (no human) and dataset labels
(all human) are the two endpoints.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from detect.config import RESULTS, ROOT  # noqa: E402
from detect.curve import median_censored, summarise  # noqa: E402
from detect.strategies import STRATEGIES  # noqa: E402
from detect.style import COLORS, DPI, INK2, SURFACE, apply, tidy  # noqa: E402

TAUS = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def ensure_run(strategy: str, labeler: str) -> pd.DataFrame:
    path = RESULTS / "curves" / f"al_{strategy}_{labeler}.csv"
    if not path.exists():
        print(f"running {strategy} / {labeler}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_al.py"),
                        "--strategy", strategy, "--labeler", labeler], check=True, capture_output=True)
    return pd.read_csv(path)


def row(strategy: str, labeler: str, tau, perfect_f1: float) -> dict:
    df = ensure_run(strategy, labeler)
    end = df[df.budget == 600]
    sm = summarise(df)
    return {"strategy": strategy, "labeler": labeler, "tau": tau,
            "human_pct": round(100 * end.human_labels.mean() / 600, 1),
            "wrong_labels": round(end.label_errors.mean(), 1),
            "f1_600": round(end.test_f1.mean(), 4),
            "gap_vs_dataset_labels": round(end.test_f1.mean() - perfect_f1, 4),
            "far_f1_600": round(end.test_far_f1.mean(), 4),
            "lab_f1_600": round(end.lab_f1.mean(), 4),
            "auc": round(float(pd.Series(sm["auc_20_600"]).mean()), 4),
            "human_to_99": median_censored(sm["test_99_human"]),
            "reached_99": sum(v is not None for v in sm["test_99"])}


PLOTTED = ["random", "uncertainty", "hybrid"]   # diversity stays below 0.95 at every threshold
TAU_LABEL_OFFSET = {"random": (2, -11), "uncertainty": (5, -2), "hybrid": (-4, 6)}


def plot(table: pd.DataFrame, path: Path) -> None:
    apply()
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    tidy(ax, "both")
    for s in PLOTTED:
        g = table[table.strategy == s].sort_values("human_pct")
        ax.plot(g.human_pct, g.f1_600, color=COLORS[s], linewidth=2, marker="o", markersize=5, label=s)
        for _, r in g[g.labeler.str.startswith("routed") & (g.tau < 0.99)].iterrows():
            ax.annotate(f"{r.tau:g}", (r.human_pct, r.f1_600), textcoords="offset points",
                        xytext=TAU_LABEL_OFFSET[s], fontsize=7, color=INK2)
    ax.axvline(30, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(30.8, ax.get_ylim()[0] + 0.001, "H4: 30% human", color=INK2, fontsize=8, va="bottom")
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0\nLLM only", "20", "40", "60", "80", "100\nall human"])
    ax.set_xlabel("share of the 600 labels that came from the human (%); point labels = threshold")
    ax.set_ylabel("test F1 at 600 labels (mean of 10 seeds)")
    ax.set_title("Human effort vs detector quality, by routing threshold\n"
                 "(diversity stays below 0.95 at every threshold; see the table)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def main() -> int:
    rows = []
    for s in STRATEGIES:
        perfect = ensure_run(s, "perfect")
        perfect_f1 = perfect[perfect.budget == 600].test_f1.mean()
        rows.append(row(s, "llm", 0.5, perfect_f1))
        rows += [row(s, f"routed{round(t * 100)}", t, perfect_f1) for t in TAUS]
        rows.append(row(s, "perfect", 1.0, perfect_f1))
    table = pd.DataFrame(rows)
    table["h4_pass"] = (table.gap_vs_dataset_labels.abs() <= 0.005) & (table.human_pct < 30)
    table.to_csv(RESULTS / "threshold_sweep.csv", index=False)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    plot(table, RESULTS / "figures" / "threshold_sweep.png")
    print(table.drop(columns=["labeler"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
