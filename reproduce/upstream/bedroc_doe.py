#!/usr/bin/env python
"""BEDROC-only comparison of ALL 48 GBSA physics combos vs AutoDock-Vina docking.

Response metric = BEDROC (early recognition), NOT Kendall tau.
Actives = designated strong binders (bins top+active); inactives = measured weak
binders (bin inactive) -- the newbench set has NO decoys, so "inactive" == weak binder.

Inputs  (RAW, measured):
  data/raw/gbsa_dG_raw.csv  : complex_id,target,combo,mean_dG_kcalmol,n_frames  (199 x 48)
  data/raw/metadata.csv     : complex_id,target,pchembl,bin,is_active,docking_score
Outputs (DERIVED):
  data/derived/bedroc_all_combos_per_target.csv
  data/derived/bedroc_all_combos_summary.csv
  figures/study1_bedroc_doe_maineffects.png
  figures/study1_bedroc_combo_ranking.png
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit.ML.Scoring.Scoring import CalcBEDROC

ROOT = Path(__file__).resolve().parents[1]
NAVY, GOLD, GREYD = "#2F3761", "#8A730F", "#8A8A8A"
ALPHAS = [8.0, 20.0, 80.0]
LOCKED = "igb2_di4_salt0.15_st0.0072"

def parse_combo(c):
    m = re.match(r"igb(\d+)_di(\d+)_salt([\d.]+)_st([\d.]+)", c)
    return dict(igb=int(m[1]), intdiel=int(m[2]), saltcon=float(m[3]), surften=float(m[4]))

def bedroc_from_scores(activity_labels, predicted_score, alpha):
    """activity_labels, predicted_score: 1D arrays; higher predicted_score = predicted more active."""
    order = np.argsort(-predicted_score, kind="mergesort")          # stable, descending
    seq = [(int(a),) for a in np.asarray(activity_labels)[order]]
    return CalcBEDROC(seq, 0, alpha)

# ---- load + join ---------------------------------------------------------
gbsa = pd.read_csv(ROOT / "data/raw/gbsa_dG_raw.csv")
meta = pd.read_csv(ROOT / "data/raw/metadata.csv")[
    ["complex_id", "target", "is_active", "docking_score"]]
df = gbsa.merge(meta, on=["complex_id", "target"], how="left", validate="many_to_one")
assert df["is_active"].notna().all(), "unjoined complexes"
df["is_active"] = df["is_active"].astype(int)
targets = sorted(df["target"].unique())

# ---- docking BEDROC per target (on the GBSA-covered complex set) ----------
dock = {}
for t in targets:
    sub = df[(df.target == t) & (df.combo == df.combo.iloc[0])]  # one combo = full complex list
    dock[t] = {a: bedroc_from_scores(sub.is_active.values, -sub.docking_score.values, a)
               for a in ALPHAS}

# ---- GBSA BEDROC per (combo, target) -------------------------------------
rows = []
for combo, gc in df.groupby("combo"):
    p = parse_combo(combo)
    for t in targets:
        sub = gc[gc.target == t]
        na, ni = int(sub.is_active.sum()), int((1 - sub.is_active).sum())
        rec = dict(combo=combo, **p, target=t, n=len(sub), n_active=na, n_inactive=ni)
        for a in ALPHAS:
            # GBSA: more negative dG = stronger binder -> predicted score = -dG
            rec[f"bedroc{int(a)}_gbsa"] = bedroc_from_scores(sub.is_active.values,
                                                             -sub.mean_dG_kcalmol.values, a)
            rec[f"bedroc{int(a)}_dock"] = dock[t][a]
        rows.append(rec)
per = pd.DataFrame(rows)
per.to_csv(ROOT / "data/derived/bedroc_all_combos_per_target.csv", index=False)

# ---- summary per combo (median across targets, wins vs docking) ----------
summ = []
for combo, gc in per.groupby("combo"):
    p = parse_combo(combo)
    rec = dict(combo=combo, **p, n_targets=gc.target.nunique())
    for a in ALPHAS:
        g = gc[f"bedroc{int(a)}_gbsa"]; d = gc[f"bedroc{int(a)}_dock"]
        rec[f"gbsa_median_bedroc{int(a)}"] = g.median()
        rec[f"dock_median_bedroc{int(a)}"] = d.median()
        rec[f"delta_bedroc{int(a)}"] = g.median() - d.median()
        rec[f"wins_bedroc{int(a)}"] = int((g.values > d.values).sum())
    summ.append(rec)
summ = pd.DataFrame(summ).sort_values("gbsa_median_bedroc20", ascending=False)
summ["is_locked"] = summ.combo == LOCKED
summ.to_csv(ROOT / "data/derived/bedroc_all_combos_summary.csv", index=False)

dock_med20 = float(np.median([dock[t][20.0] for t in targets]))

# ---- FIG 1: DoE main effects (BEDROC@20) ---------------------------------
plt.rcParams.update({"axes.grid": False, "font.size": 10})
params = [("igb", "GB model (igb)"), ("intdiel", "interior dielectric"),
          ("saltcon", "salt conc (M)"), ("surften", "surface tension")]
fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, (col, label) in zip(axes.ravel(), params):
    levels = sorted(per[col].unique())
    med = [per[per[col] == lv]["bedroc20_gbsa"].median() for lv in levels]
    q1 = [per[per[col] == lv]["bedroc20_gbsa"].quantile(.25) for lv in levels]
    q3 = [per[per[col] == lv]["bedroc20_gbsa"].quantile(.75) for lv in levels]
    x = range(len(levels))
    ax.fill_between(x, q1, q3, color=NAVY, alpha=0.15, zorder=1)
    ax.plot(x, med, "-D", color=NAVY, ms=8, lw=1.8, zorder=3)
    ax.axhline(dock_med20, ls="--", color=GREYD, lw=1.4, zorder=2,
               label=f"docking median = {dock_med20:.3f}")
    ax.set_xticks(list(x)); ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel(label); ax.set_ylabel("BEDROC (α=20)")
    ax.set_title(f"main effect: {label}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
fig.suptitle("GBSA parameter main effects on early recognition (BEDROC α=20)  —  "
             "median over 8 targets, IQR band", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(ROOT / "figures/study1_bedroc_doe_maineffects.png", dpi=150)

# ---- FIG 2: all 48 combos ranked vs docking ------------------------------
fig, ax = plt.subplots(figsize=(7, 11))
order = summ.sort_values("gbsa_median_bedroc20")
y = range(len(order))
colors = [GOLD if lk else NAVY for lk in order.is_locked]
ax.barh(list(y), order.gbsa_median_bedroc20, color=colors, height=0.7)
ax.axvline(dock_med20, ls="--", color=GREYD, lw=1.6,
           label=f"docking median BEDROC = {dock_med20:.3f}")
ax.set_yticks(list(y)); ax.set_yticklabels(order.combo, fontsize=6)
ax.set_xlabel("median BEDROC (α=20) over 8 targets")
ax.set_title("All 48 GBSA physics combos vs docking (gold = locked combo)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(ROOT / "figures/study1_bedroc_combo_ranking.png", dpi=150)

# ---- console report ------------------------------------------------------
n_beat = int((summ.gbsa_median_bedroc20 > dock_med20).sum())
print(f"targets (n): {len(targets)}  ->  {targets}")
print(f"docking median BEDROC@20 = {dock_med20:.3f}")
print(f"combos beating docking (median BEDROC@20): {n_beat}/48")
print("\nTOP 8 combos by median BEDROC@20:")
cols = ["combo", "gbsa_median_bedroc20", "delta_bedroc20", "wins_bedroc20"]
print(summ.head(8)[cols].to_string(index=False))
print("\nLOCKED combo row:")
print(summ[summ.is_locked][cols].to_string(index=False))
print("\nWORST 3:")
print(summ.tail(3)[cols].to_string(index=False))
print("\nwrote: data/derived/bedroc_all_combos_{per_target,summary}.csv + 2 figures")
