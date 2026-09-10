#!/usr/bin/env python3
"""SI figures — per-target × per-analysis, A4 portrait, 30 mini-plots per page.

Structure:  figures/SI/{TARGET}/{NN}_{analysis}.png
For each (target, analysis), one A4 portrait page with a 3 × 10 grid of tiny
plots — one per ligand — so you can eyeball every complex individually.

Actives GOLD (thick border + line), decoys NAVY (thin line).
Small font, minimal decoration; axis labels only on the leftmost column and
bottom row to save ink.
"""
from __future__ import annotations
import sys, os
from pathlib import Path

ROOT = Path('/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study')
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from discovery9.style import apply_style, NAVY, GOLD, GREY, GREY_DASH as GREYD, CREAM, WHITE
from discovery9.paths import RAW, FIGURES
from discovery9.io    import load_features, load_metadata

apply_style()

SI = FIGURES / "SI"
SI.mkdir(parents=True, exist_ok=True)

df = load_features()
meta = load_metadata()[['complex_id','target','is_active','pchembl','docking_score']]
df = df.merge(meta, on=['complex_id','target'], how='left', suffixes=('','_meta'))
if 'is_active_meta' in df.columns:
    df['is_active'] = df['is_active_meta']

TARGETS = sorted(df.target.unique())

# (order, title, unit, filename_stem, source_file, source_col)
ANALYSES = [
    (1, 'Ligand pose drift',                   'Å',  'ligand_drift',         'timeseries.parquet',       'lig_drift_A'),
    (2, 'Ligand COM displacement',             'Å',  'ligand_com_disp',      'timeseries.parquet',       'lig_com_disp_A'),
    (3, 'Ligand buried SASA',                  'Å²', 'ligand_buried_sasa',   'timeseries.parquet',       'lig_buried_sasa_A2'),
    (4, 'vdW contacts (<4 Å heavy-heavy)',     '#',  'vdw_contacts',         'timeseries.parquet',       'vdw_contacts'),
    (5, 'Active-site backbone RMSD',           'Å',  'active_site_rmsd',     'timeseries.parquet',       'rmsd_as_bb_A'),
    (6, 'H-bonds ligand ↔ active site',        '#',  'hbonds',               'hbond_timeseries.parquet', 'n_hb'),
    (7, 'Ligand internal RMSD',                'Å',  'ligand_internal_rmsd', 'timeseries.parquet',       'lig_internal_rmsd_A'),
    (8, 'Protein Rg',                          'Å',  'protein_rg',           'timeseries.parquet',       'protein_rg_A'),
]

A4_PORTRAIT = (8.27, 11.69)   # inches
GRID_ROWS, GRID_COLS = 10, 3   # 30 mini-plots max

def draw_grid(target: str, order: int, title: str, unit: str, stem: str,
              source_file: str, source_col: str):
    """One A4 portrait page: 3 × 10 mini-plots, one per ligand."""
    # Column-major layout: column 1 = actives, columns 2+3 = decoys.
    # Order actives by pKi descending (strongest first), decoys by complex_id.
    sub_all = df[df.target == target].copy()
    n_lig = len(sub_all)
    n_act = int((sub_all.is_active == True).sum())
    actives = sub_all[sub_all.is_active == True].sort_values('pchembl', ascending=False)
    decoys  = sub_all[sub_all.is_active == False].sort_values('complex_id')
    unlabeled = sub_all[sub_all.is_active.isna()].sort_values('complex_id')
    # Layout: col 0 = actives (10 cells), cols 1+2 = decoys (up to 20 cells), overflow = unlabeled
    col0 = list(actives.itertuples(index=False))                     # up to 10 actives
    remaining = list(decoys.itertuples(index=False)) + list(unlabeled.itertuples(index=False))
    col1 = remaining[:GRID_ROWS]                                     # next 10
    col2 = remaining[GRID_ROWS:GRID_ROWS*2]                          # next 10
    columns = [col0, col1, col2]

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=A4_PORTRAIT, sharex=True)
    fig.suptitle(f'{target}  ·  SI-{order}: {title}  [{unit}]  ·  '
                 f'{n_act} actives (GOLD frame), {n_lig - n_act} decoys',
                 fontsize=10, color=NAVY, fontweight='bold', y=0.995)

    # Preload all series to compute global y-range for uniform scale
    all_y = []
    col_ts = [[], [], []]
    for c, col in enumerate(columns):
        for row in col:
            f = RAW / 'complex_analyses' / target / row.complex_id / source_file
            if not f.exists():
                col_ts[c].append(None); continue
            ts = pd.read_parquet(f)
            if source_col not in ts.columns:
                col_ts[c].append(None); continue
            col_ts[c].append(ts)
            all_y.extend(ts[source_col].values.tolist())
    if not all_y:
        plt.close(fig); return None, 0
    y_lo, y_hi = np.nanmin(all_y), np.nanmax(all_y)
    y_pad = 0.05 * (y_hi - y_lo + 1e-9)
    y_lo, y_hi = y_lo - y_pad, y_hi + y_pad

    for c in range(GRID_COLS):
        col = columns[c]
        for r in range(GRID_ROWS):
            ax = axes[r, c]
            if r >= len(col):
                ax.axis('off'); continue
            row = col[r]
            ts = col_ts[c][r]
            is_act = (c == 0)  # column 0 is actives by construction
            line_color = GOLD if is_act else NAVY
            for spine in ax.spines.values():
                spine.set_color(GOLD if is_act else GREYD)
                spine.set_linewidth(1.5 if is_act else 0.6)
            if ts is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', color=GREYD, transform=ax.transAxes, fontsize=6)
            else:
                t = ts.time_ps / 1000.0
                y = ts[source_col].values
                ax.plot(t, y, color=line_color, lw=1.0 if is_act else 0.7, alpha=0.95)
            ax.set_ylim(y_lo, y_hi)
            ax.tick_params(labelsize=6, length=2, pad=1)
            cid_short = row.complex_id[:8]
            pki_str = f' pKi={row.pchembl:.1f}' if pd.notna(getattr(row, 'pchembl', np.nan)) else ''
            ax.set_title(f'{cid_short}{pki_str}', fontsize=6.5, pad=2,
                         color=NAVY, fontweight='bold' if is_act else 'normal')
            if c != 0: ax.set_yticklabels([])
            if r != GRID_ROWS - 1: ax.set_xticklabels([])

    # Common axis labels
    fig.text(0.5, 0.005, 'time [ns]', ha='center', fontsize=10, color=NAVY)
    fig.text(0.005, 0.5, f'{title}  [{unit}]', va='center', rotation='vertical', fontsize=10, color=NAVY)

    plt.subplots_adjust(hspace=0.35, wspace=0.15, left=0.06, right=0.99, top=0.965, bottom=0.03)

    tdir = SI / target
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / f'{order:02d}_{stem}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=CREAM)
    plt.close(fig)
    return out, n_lig

count = 0
for t in TARGETS:
    for (order, title, unit, stem, src_file, src_col) in ANALYSES:
        try:
            out, n = draw_grid(t, order, title, unit, stem, src_file, src_col)
            if out:
                count += 1
                print(f'  {out.relative_to(ROOT)}   ({n} ligs)')
        except Exception as e:
            print(f'  ERR {t} / {stem}: {e}')

print(f'\nWrote {count} A4 SI figures across {len(TARGETS)} targets to {SI}/')
