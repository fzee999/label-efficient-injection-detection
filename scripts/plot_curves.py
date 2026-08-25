"""Learning curves per strategy: test F1 and far-text F1, mean with ±1 std band.

    .venv\\Scripts\\python scripts\\plot_curves.py --labeler perfect
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.curve import CEILING_F1, CEILING_FAR_F1  # noqa: E402
from detect.strategies import STRATEGIES  # noqa: E402
from detect.style import COLORS, DPI, INK, INK2, SURFACE, apply, tidy  # noqa: E402

PANELS = [("test_f1", "Test set", CEILING_F1),
          ("test_far_f1", "Test texts with no close pool match", CEILING_FAR_F1)]
TITLES = {"perfect": "dataset labels", "llm": "LLM labels"}


def title_for(labeler: str) -> str:
    if labeler.startswith("routed"):
        return f"LLM labels, human below confidence {int(labeler[6:]) / 100:g}"
    return TITLES.get(labeler, labeler)


def y_range(curves: dict, col: str, ceiling: float) -> tuple[float, float]:
    """From just below the lowest mean (after the first round) to just above the ceiling."""
    lows = [df[df.budget >= 40].groupby("budget")[col].mean().min() for df in curves.values()]
    lo = np.floor((min(lows) - 0.015) / 0.02) * 0.02
    return float(lo), float(max(ceiling + 0.015, 1.0 if ceiling > 0.97 else ceiling + 0.02))


def spread_labels(ys: dict, gap: float) -> dict:
    """Nudge end-of-line labels apart so they don't overlap."""
    items = sorted(ys.items(), key=lambda kv: kv[1])
    out, last = {}, -1e9
    for name, y in items:
        y = max(y, last + gap)
        out[name], last = y, y
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labeler", default="perfect")
    args = p.parse_args()
    curves = {s: pd.read_csv(f) for s in STRATEGIES
              if (f := RESULTS / "curves" / f"al_{s}_{args.labeler}.csv").exists()}

    apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE, sharex=True)
    for ax, (col, title, ceiling) in zip(axes, PANELS):
        ylim = y_range(curves, col, ceiling)
        tidy(ax)
        ends = {}
        for s, df in curves.items():
            g = df.groupby("budget")[col]
            m, sd = g.mean(), g.std()
            ax.fill_between(m.index, m - sd, m + sd, color=COLORS[s], alpha=0.12, linewidth=0)
            ax.plot(m.index, m.values, color=COLORS[s], linewidth=2, label=s)
            ends[s] = float(m.iloc[-1])
        pad = (ylim[1] - ylim[0]) * 0.008
        for y, txt, va in [(ceiling, "all 8,084 labels", "bottom"), (0.99 * ceiling, "99% of that", "top")]:
            ax.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(25, y + (pad if va == "bottom" else -pad), txt, color=INK2, fontsize=8, va=va)
        for s, y in spread_labels(ends, (ylim[1] - ylim[0]) * 0.035).items():
            ax.plot([607], [y], marker="o", markersize=5, color=COLORS[s], clip_on=False)
            ax.text(617, y, s, color=INK, fontsize=8.5, va="center", clip_on=False)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(20, 600)
        ax.set_ylim(*ylim)
        ax.set_xlabel("labels used")
    axes[0].set_ylabel("F1, injection class (mean of 10 seeds, band = ±1 std)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=len(names), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.96, 1.0))
    fig.suptitle(f"Active learning vs random labeling ({title_for(args.labeler)})",
                 x=0.06, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 0.96, 1))

    out = RESULTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"curves_{args.labeler}.png"
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
