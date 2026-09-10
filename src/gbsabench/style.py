"""The package's one plotting style — navy/gold, no red/green, no grid.

Mirror of the MPS study's `mpsbench.style`: call ``apply_style()`` once at the top of a notebook,
then plot normally. The look is defined here in exactly one place.

Rules baked in (house style):
  * two-colour palette: NAVY (primary) + GOLD (secondary); never red or green
  * no gridlines; cream figure background, white axes
  * reference lines are dashed grey (GREY_DASH)
  * titles are set by the caller and kept out of rcParams, so publication figures can omit them
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# --- palette (hex) -----------------------------------------------------------
NAVY = "#2F3761"       # primary series
GOLD = "#8A730F"       # secondary series
MID = "#5B6196"        # midpoint for navy->gold colormaps
GREY = "#D5D7DF"       # light grey (legend edges, fills, bands)
GREY_DASH = "#9AA3B5"  # dashed reference lines / fit curves
CREAM = "#F7F4EF"      # figure background
WHITE = "#FFFFFF"      # axes background


def navy_gold_cmap():
    """A continuous navy -> mid -> gold colormap (for heatmaps / gradients)."""
    return mpl.colors.LinearSegmentedColormap.from_list("navy_gold", [NAVY, MID, "#B89A3A", GOLD])


def cream_navy_cmap():
    """Sequential heatmap cmap: cream (low) -> gold (mid) -> navy (high).

    Use when zero should visually read as "empty/absent" (light cream) and the
    extreme should read as "loaded" (navy). Lighter overall than
    :func:`navy_gold_cmap`; better contrast for annotated numbers.
    """
    return mpl.colors.LinearSegmentedColormap.from_list(
        "cream_navy", [CREAM, "#EADFAE", GOLD, MID, NAVY])


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
        "axes.grid": False,               # house rule: no grid
        "axes.prop_cycle": cycler(color=[NAVY, GOLD]),
        "text.color": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "legend.frameon": True,
        "legend.facecolor": CREAM,
        "legend.edgecolor": GREY,
        "font.size": 12,
    })
