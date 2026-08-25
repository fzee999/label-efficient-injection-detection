"""Shared figure style: one color per strategy across every figure."""
from __future__ import annotations

import matplotlib.pyplot as plt

# validated categorical slots 1-4, light surface (adjacent CVD ΔE ≥ 9.1)
COLORS = {"random": "#2a78d6", "uncertainty": "#eb6834", "diversity": "#1baf7a", "hybrid": "#eda100"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REFERENCE = "#8a8984"   # the full-data model
DPI = 200


def apply() -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "hatch.color": SURFACE, "hatch.linewidth": 1.2})


def tidy(ax, grid_axis: str = "y") -> None:
    """Surface, light grid, no top/right spines."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
