"""Shared look for Member B's report figures: the reference palette's first
two categorical slots (validated for light mode with the dataviz skill's
validate_palette.js), recessive hairline chrome, and text in ink colours."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

ORACLE = "#2a78d6"  # categorical slot 1
REAL_MODEL = "#eb6834"  # categorical slot 2

DPI = 150
# At 150 dpi one point is about 2.1 px: lines ~2 px, markers ~8 px, rings ~2 px.
LINE_WIDTH = 1.0
MARKER_SIZE = 4.0
RING_WIDTH = 1.0


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.5,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
        }
    )


def recede(ax) -> None:
    """Hairline, solid, recessive: horizontal gridlines and a baseline only."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
