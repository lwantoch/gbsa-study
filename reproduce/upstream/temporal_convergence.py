#!/usr/bin/env python
"""Temporal convergence of GBSA ranking: two questions the study never answered.

  Q1  Minimum PRODUCTION TIME (cumulative [0,T]) for GBSA to beat docking BEDROC.
  Q2  How many FRAMES are needed (subsample the full 30 ns) to preserve BEDROC.

Metric = BEDROC (alpha=20) ONLY.  Actives = strong binders (bins top+active);
inactives = measured WEAK binders (bin inactive; no decoys).

Source = LOCAL per-frame data (validated to reproduce data/raw/gbsa_dG_raw.csv):
  newbench15_gbsa/factorial_out/<complex_id>/perframe_delta.parquet
  -> 199 complexes x 48 combos x 3001 frames, TOTAL per frame, ~10 ps/frame.
Analysis on the LOCKED combo (reported combo).  No re-running, no trajectories needed.

Outputs (DERIVED):
  data/derived/temporal_bedroc_vs_time.csv
  data/derived/temporal_bedroc_vs_frames.csv
  figures/study1_temporal_min_production_time.png
  figures/study1_temporal_frames_needed.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit.ML.Scoring.Scoring import CalcBEDROC

REPO = Path("/home/otras/hcx/lwa/projects/gbsa-study")
FOUT = Path("/home/otras/hcx/lwa/newbench15_gbsa/factorial_out")
NAVY, GOLD, GREYD = "#2F3761", "#8A730F", "#8A8A8A"
LOCKED = "igb2_di4_salt0.15_st0.0072"
ALPHA = 20.0
PS_PER_FRAME = 10.0            # documented BASE_PS; 3001 frames ~ 30 ns

def bedroc(labels, score_higher_is_active):
    order = np.argsort(-score_higher_is_active, kind="mergesort")
    return CalcBEDROC([(int(a),) for a in np.asarray(labels)[order]], 0, ALPHA)

# ---- labels + docking baseline (same complex set) ------------------------
meta = pd.read_csv(REPO / "data/raw/metadata.csv")[
    ["complex_id", "target", "is_active", "docking_score"]]
raw = pd.read_csv(REPO / "data/raw/gbsa_dG_raw.csv")[["complex_id", "target"]].drop_duplicates()
lab = raw.merge(meta, on=["complex_id", "target"], how="left")
lab["is_active"] = lab["is_active"].astype(int)
targets = sorted(lab.target.unique())

dock_bedroc = {}
for t in targets:
    s = lab[lab.target == t]
    dock_bedroc[t] = bedroc(s.is_active.values, -s.docking_score.values)
DOCK_MED = float(np.median(list(dock_bedroc.values())))

# ---- load LOCKED-combo per-frame TOTAL for all 199 complexes -------------
totals = {}   # complex_id -> np.array TOTAL ordered by frame
for cid in lab.complex_id:
    pq = FOUT / cid / "perframe_delta.parquet"
    if not pq.exists():
        continue
    pf = pd.read_parquet(pq, columns=["combo", "frame", "TOTAL"])
    sub = pf[pf.combo == LOCKED].sort_values("frame")
    if len(sub):
        totals[cid] = sub["TOTAL"].to_numpy()
NFRAMES = int(np.median([len(v) for v in totals.values()]))
print(f"loaded {len(totals)} complexes, locked combo, {NFRAMES} frames each")
print(f"docking median BEDROC@20 = {DOCK_MED:.3f}  (target set n={len(targets)})")

def bedroc_per_target(pred_score_by_cid):
    """pred_score_by_cid: dict cid->GBSA score (higher=more active). Return median BEDROC over targets + per-target."""
    per = {}
    for t in targets:
        s = lab[lab.target == t]
        cids = [c for c in s.complex_id if c in pred_score_by_cid]
        if len(cids) < 3:
            continue
        ss = s[s.complex_id.isin(cids)]
        score = np.array([pred_score_by_cid[c] for c in ss.complex_id])
        per[t] = bedroc(ss.is_active.values, score)
    return float(np.median(list(per.values()))), per

# ---- Q1: cumulative time [0,T] -------------------------------------------
frame_grid = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750,
              1000, 1250, 1500, 2000, 2500, 3001]
frame_grid = [n for n in frame_grid if n <= NFRAMES]
rows_t = []
for n in frame_grid:
    # GBSA predicted activity = -mean(dG over first n frames); more negative dG -> higher score
    pred = {c: -v[:n].mean() for c, v in totals.items()}
    med, per = bedroc_per_target(pred)
    rows_t.append(dict(n_frames=n, ns=n * PS_PER_FRAME / 1000.0, median_bedroc=med,
                       **{f"bedroc_{t}": per.get(t, np.nan) for t in targets}))
tt = pd.DataFrame(rows_t)
tt.to_csv(REPO / "data/derived/temporal_bedroc_vs_time.csv", index=False)

# smallest T (ns) whose median BEDROC >= docking AND stays >= for all larger T
above = tt.median_bedroc.values >= DOCK_MED
min_t_ns = None
for i in range(len(tt)):
    if above[i:].all():
        min_t_ns = tt.ns.values[i]; min_t_frames = tt.n_frames.values[i]; break

# ---- Q2: subsample k evenly-spaced frames from the FULL 30 ns -------------
k_grid = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, NFRAMES]
k_grid = sorted(set(k for k in k_grid if k <= NFRAMES))
full_pred = {c: -v.mean() for c, v in totals.items()}
full_med, _ = bedroc_per_target(full_pred)
rows_k = []
for k in k_grid:
    idx = np.unique(np.linspace(0, NFRAMES - 1, k).round().astype(int))
    pred = {c: -v[idx].mean() for c, v in totals.items() if len(v) >= NFRAMES}
    med, per = bedroc_per_target(pred)
    rows_k.append(dict(k_frames=k, median_bedroc=med,
                       pct_of_full=100 * med / full_med if full_med else np.nan))
kk = pd.DataFrame(rows_k)
kk.to_csv(REPO / "data/derived/temporal_bedroc_vs_frames.csv", index=False)
# fewest frames within 2% of full-trajectory BEDROC
within = kk[kk.median_bedroc >= 0.98 * full_med]
min_frames = int(within.k_frames.min()) if len(within) else NFRAMES

# ---- FIG Q1 --------------------------------------------------------------
plt.rcParams.update({"axes.grid": False, "font.size": 10})
fig, ax = plt.subplots(figsize=(8, 5.2))
for t in targets:                                   # per-target thin lines
    ax.plot(tt.ns, tt[f"bedroc_{t}"], color=NAVY, alpha=0.18, lw=1)
ax.plot(tt.ns, tt.median_bedroc, "-D", color=NAVY, ms=6, lw=2, label="GBSA median (8 targets)")
ax.axhline(DOCK_MED, ls="--", color=GREYD, lw=1.6, label=f"docking median = {DOCK_MED:.3f}")
if min_t_ns is not None:
    ax.axvline(min_t_ns, ls=":", color=GOLD, lw=1.8)
    ax.plot([min_t_ns], [DOCK_MED], "o", color=GOLD, ms=9, zorder=5,
            label=f"beats docking from {min_t_ns:.1f} ns ({min_t_frames} frames)")
ax.set_xlabel("cumulative production time  [0, T]  (ns)")
ax.set_ylabel("BEDROC (α=20)")
ax.set_title("Q1  Minimum production time for GBSA to beat docking (locked combo)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(REPO / "figures/study1_temporal_min_production_time.png", dpi=150)

# ---- FIG Q2 --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(kk.k_frames, kk.median_bedroc, "-D", color=NAVY, ms=6, lw=2, label="GBSA median BEDROC")
ax.axhline(full_med, ls="-", color=NAVY, lw=1, alpha=0.4, label=f"full-traj ({NFRAMES} fr) = {full_med:.3f}")
ax.axhline(0.98 * full_med, ls="--", color=GREYD, lw=1.4, label="98% of full")
ax.axhline(DOCK_MED, ls=":", color=GREYD, lw=1.4, label=f"docking = {DOCK_MED:.3f}")
ax.axvline(min_frames, ls=":", color=GOLD, lw=1.8, label=f"{min_frames} frames reach 98%")
ax.set_xscale("log")
ax.set_xlabel("number of frames (evenly spaced over full 30 ns)")
ax.set_ylabel("BEDROC (α=20)")
ax.set_title("Q2  How many frames are needed (locked combo)", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(REPO / "figures/study1_temporal_frames_needed.png", dpi=150)

# ---- report --------------------------------------------------------------
print("\n=== Q1  min production time to beat docking ===")
print(tt[["n_frames", "ns", "median_bedroc"]].to_string(index=False))
print(f"-> beats docking ({DOCK_MED:.3f}) from {min_t_ns} ns ({min_t_frames} frames) onward")
print("\n=== Q2  frames needed (subsample of full 30 ns) ===")
print(kk.to_string(index=False))
print(f"-> {min_frames} frames reach >=98% of full-trajectory BEDROC ({full_med:.3f})")
print("\nwrote 2 derived CSVs + 2 figures")
