#!/usr/bin/env python
"""Scatter of BEDROC vs every GBSA full-factorial parameter.

One panel per tested parameter (igb, intdiel, saltcon, surften). Each point is
ONE (combo x target) BEDROC@20 measurement -> 48 combos x 8 targets = 384 points
per panel. Points coloured by interior dielectric (the dominant factor) so the
vertical spread in the other panels is visibly explained by di1/di2/di4.
Grey dashed = docking median baseline; gold dashes = per-level median BEDROC.

Input : data/derived/bedroc_all_combos_per_target.csv  (from bedroc_doe.py)
Output: figures/study1_bedroc_param_scatter.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/otras/hcx/lwa/projects/gbsa-study")
NAVY, GOLD, GREYD = "#2F3761", "#8A730F", "#8A8A8A"
DI_COLORS = {1: "#B8BDD0", 2: "#2F3761", 4: "#8A730F"}   # light-navy -> navy -> gold
DOCK_MED = 0.436

per = pd.read_csv(REPO / "data/derived/bedroc_all_combos_per_target.csv")
y = "bedroc20_gbsa"
params = [("igb", "GB model (igb)"), ("intdiel", "interior dielectric"),
          ("saltcon", "salt conc (M)"), ("surften", "surface tension")]

plt.rcParams.update({"axes.grid": False, "font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
rng = np.random.default_rng(0)          # deterministic jitter
for ax, (col, label) in zip(axes.ravel(), params):
    levels = sorted(per[col].unique())
    xpos = {lv: i for i, lv in enumerate(levels)}
    for _, r in per.iterrows():
        xj = xpos[r[col]] + rng.uniform(-0.16, 0.16)
        ax.scatter(xj, r[y], s=26, color=DI_COLORS[int(r["intdiel"])],
                   edgecolor="white", linewidth=0.3, alpha=0.85, zorder=2)
    for lv in levels:                    # per-level median tick
        med = per[per[col] == lv][y].median()
        ax.plot([xpos[lv] - 0.28, xpos[lv] + 0.28], [med, med],
                color="black", lw=2.2, zorder=4)
    ax.axhline(DOCK_MED, ls="--", color=GREYD, lw=1.5, zorder=1)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel(label); ax.set_ylabel("BEDROC (α=20)")
    ax.set_title(f"{label}", fontsize=10)
    ax.set_ylim(-0.02, 1.02)

# legend (intdiel colour key + baselines)
handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=DI_COLORS[d],
                      label=f"intdiel={d}") for d in (1, 2, 4)]
handles += [plt.Line2D([], [], color="black", lw=2.2, label="per-level median"),
            plt.Line2D([], [], color=GREYD, ls="--", lw=1.5, label=f"docking = {DOCK_MED:.3f}")]
fig.legend(handles=handles, ncol=5, fontsize=9, frameon=False,
           loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.suptitle("BEDROC (α=20) vs each full-factorial GBSA parameter  —  "
             "384 (combo × target) points, coloured by interior dielectric", fontsize=11)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(REPO / "figures/study1_bedroc_param_scatter.png", dpi=150)
print("wrote figures/study1_bedroc_param_scatter.png")
print("points per panel:", len(per))
