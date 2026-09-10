"""The project's one plotting style — IDIS navy/gold, no red/green, no default grid.

Call ``apply_style()`` once at the top of a notebook or script, then plot normally.
Copy of the style rules used across the GPU-MPS and BO studies — kept identical so
figures in this study drop straight into the same report.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# --- palette (hex) -----------------------------------------------------------
NAVY = "#2F3761"       # primary series
GOLD = "#8A730F"       # secondary series
MID = "#5B6196"        # midpoint for navy->gold colormaps
GREY = "#D5D7DF"       # light grey (legend edges, fills)
GREY_DASH = "#9AA3B5"  # dashed reference lines / fit curves
CREAM = "#F7F4EF"      # figure background
WHITE = "#FFFFFF"      # axes background

# active vs decoy accents (kept inside palette family)
ACTIVE = GOLD
DECOY = NAVY
WARN = "#B03A2E"       # reserved for "baseline to beat" reference lines

# per-target colour map — stable colour per target across notebooks
TARGET_COLORS = {
    "2XU3": NAVY, "3I06": GOLD, "4A5S": "#B89A3A", "4L7G": "#5B6196",
    "4QB3": "#7A8AB5", "5HU9": "#9E822A", "8ELC": "#3F4B7A", "9D9I": "#A18C4C",
    "9SI4": "#6C7699",
}


def navy_gold_cmap():
    """Continuous navy -> mid -> gold colormap (for heatmaps / gradients)."""
    return mpl.colors.LinearSegmentedColormap.from_list(
        "navy_gold", [NAVY, MID, "#B89A3A", GOLD]
    )


def apply_style():
    """Apply the house rcParams. Safe to call more than once."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "savefig.bbox": "tight",
        "axes.facecolor": WHITE,
        "axes.edgecolor": NAVY,
        "axes.labelcolor": NAVY,
        "axes.titlecolor": NAVY,
        "axes.titleweight": "bold",
        "axes.grid": False,
        "axes.prop_cycle": cycler(color=[NAVY, GOLD, GREY_DASH]),
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "legend.frameon": True,
        "legend.facecolor": CREAM,
        "legend.edgecolor": GREY,
        "font.size": 12,
    })
