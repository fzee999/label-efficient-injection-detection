"""Formal analysis: paired tests across seeds, equivalence checks, labels-to-target, headline figure.

Tests were chosen at analysis time (the pre-registration fixed hypotheses, not tests):
six primary comparisons on curve area, Holm-corrected together; everything else exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import t as student_t, ttest_rel, wilcoxon  # noqa: E402

from detect.config import RESULTS  # noqa: E402
from detect.curve import CEILING_F1, median_censored, summarise  # noqa: E402

STRATS = ["random", "uncertainty", "diversity", "hybrid"]
LABELERS = ["perfect", "llm", "routed80", "routed90"]
EQUIV_MARGIN = 0.005   # fixed in step 14
OUT = RESULTS / "analysis"


def per_seed(strategy: str, labeler: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "curves" / f"al_{strategy}_{labeler}.csv")
    end = df[df.budget == 600].set_index("seed").sort_index()
    return pd.DataFrame({"auc": summarise(df)["auc_20_600"], "f1_600": end.test_f1.to_numpy(),
                         "far_f1_600": end.test_far_f1.to_numpy(), "lab_f1_600": end.lab_f1.to_numpy()},
                        index=end.index)


def paired(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> dict:
    d = a - b
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    half = student_t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)
    return {"mean_diff": m, "ci_low": m - half, "ci_high": m + half, "wins": int((d > 0).sum()),
            "t_p": float(ttest_rel(a, b).pvalue) if sd > 0 else 1.0,
            "wilcoxon_p": float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0}


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {(s, l): per_seed(s, l) for s in STRATS for l in LABELERS}

    primary = [
        ("uncertainty vs random, dataset labels", data[("uncertainty", "perfect")].auc, data[("random", "perfect")].auc),
        ("hybrid vs random, dataset labels", data[("hybrid", "perfect")].auc, data[("random", "perfect")].auc),
        ("uncertainty vs random, LLM labels", data[("uncertainty", "llm")].auc, data[("random", "llm")].auc),
        ("hybrid vs random, LLM labels", data[("hybrid", "llm")].auc, data[("random", "llm")].auc),
        ("H3: uncertainty's loss vs random's loss (dataset -> LLM labels)",
         data[("uncertainty", "perfect")].auc - data[("uncertainty", "llm")].auc,
         data[("random", "perfect")].auc - data[("random", "llm")].auc),
        ("uncertainty vs random, routed labels 0.9", data[("uncertainty", "routed90")].auc, data[("random", "routed90")].auc),
    ]
    rows = [{"family": "primary", "comparison": name, "metric": "curve area", **paired(a.to_numpy(), b.to_numpy())}
            for name, a, b in primary]
    for r, adj in zip(rows, holm([r["t_p"] for r in rows])):
        r["holm_p"], r["significant"] = adj, adj < 0.05

    in_primary = {("uncertainty", "perfect", "auc"), ("hybrid", "perfect", "auc"), ("uncertainty", "llm", "auc"),
                  ("hybrid", "llm", "auc"), ("uncertainty", "routed90", "auc")}
    for l in LABELERS:
        for s in ["uncertainty", "diversity", "hybrid"]:
            for metric in ["auc", "f1_600", "far_f1_600", "lab_f1_600"]:
                if (s, l, metric) in in_primary:
                    continue
                rows.append({"family": "exploratory", "comparison": f"{s} vs random, {l}", "metric": metric,
                             **paired(data[(s, l)][metric].to_numpy(), data[("random", l)][metric].to_numpy()),
                             "holm_p": None, "significant": None})
    tests = pd.DataFrame(rows)
    tests.to_csv(OUT / "paired_tests.csv", index=False)

    equiv = []
    for l in ["routed80", "routed90"]:
        for s in STRATS:
            r = paired(data[(s, l)].f1_600.to_numpy(), data[(s, "perfect")].f1_600.to_numpy(), level=0.90)
            equiv.append({"strategy": s, "labeler": l, "mean_diff": r["mean_diff"], "ci90_low": r["ci_low"],
                          "ci90_high": r["ci_high"],
                          "equivalent": r["ci_low"] > -EQUIV_MARGIN and r["ci_high"] < EQUIV_MARGIN})
    pd.DataFrame(equiv).to_csv(OUT / "equivalence_routed_vs_dataset.csv", index=False)

    lt = []
    for l in LABELERS:
        for s in STRATS:
            sm = summarise(pd.read_csv(RESULTS / "curves" / f"al_{s}_{l}.csv"))
            lt.append({"strategy": s, "labeler": l,
                       **{k: f"{median_censored(sm[k])} ({sum(v is not None for v in sm[k])}/10)"
                          for k in ["target_95", "test_99", "far_99"]},
                       "human_to_test_99": median_censored(sm["test_99_human"]),
                       "auc_mean": round(float(np.mean(sm["auc_20_600"])), 4)})
    pd.DataFrame(lt).to_csv(OUT / "labels_to_target.csv", index=False)

    headline_figure(OUT.parent / "figures" / "headline_curves.png")

    pd.set_option("display.width", 200)
    fmt = tests[tests.family == "primary"][["comparison", "mean_diff", "ci_low", "ci_high", "wins", "t_p", "wilcoxon_p", "holm_p", "significant"]]
    print(fmt.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print(pd.DataFrame(equiv).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return 0


def headline_figure(path: Path) -> None:
    colors = {"random": "#2a78d6", "uncertainty": "#eb6834", "hybrid": "#eda100"}
    panels = [("perfect", "Dataset labels"), ("llm", "LLM labels only"), ("routed90", "LLM, human below 0.9")]
    surface, ink2, grid = "#fcfcfb", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink2,
                         "xtick.color": ink2, "ytick.color": ink2})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, facecolor=surface)
    for ax, (lab, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        for s, c in colors.items():
            df = pd.read_csv(RESULTS / "curves" / f"al_{s}_{lab}.csv")
            g = df.groupby("budget").test_f1
            m, sd, n = g.mean(), g.std(), g.count()
            half = student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            ax.fill_between(m.index, m - half, m + half, color=c, alpha=0.18, linewidth=0)
            ax.plot(m.index, m.values, color=c, linewidth=2, label=s)
        ax.axhline(CEILING_F1, color=ink2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(20, 600)
        ax.set_ylim(0.90, 0.995)
        ax.set_xlabel("labels used")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].text(25, CEILING_F1 + 0.001, "all 8,084 dataset labels", color=ink2, fontsize=8, va="bottom")
    axes[0].set_ylabel("test F1 (mean of 10 seeds, band = 95% CI)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.98, 1.0))
    fig.suptitle("Which strategy wins depends on who labels", x=0.06, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=surface)


if __name__ == "__main__":
    raise SystemExit(main())
