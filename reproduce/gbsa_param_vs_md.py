#!/usr/bin/env python3
"""Do individual GBSA parameters prefer particular MD-signatures?

For each GBSA parameter axis (igb, intdiel, saltcon, surften), split combos
into two groups by that parameter value, take the delta of per-target BEDROC α=20,
and correlate that per-target delta with per-target MD features (Spearman ρ).

Positive ρ → high value of the MD feature → prefers the higher parameter setting.
Report the ranked table + p-values (permutation).
"""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy.stats import spearmanr, rankdata
warnings.filterwarnings('ignore')

ROOT = Path('/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study')
STUDY = ROOT / 'gbsa-study-updated/gbsa-study-updated/gbsa-study'

# per-(GBSA combo × target) BEDROC α=20
bt = pd.read_csv(STUDY / 'data/derived/temporal/bedroc_all_combos_per_target.csv')

# per-target MD fingerprint (mean of the 30 complexes' features)
df = pd.read_parquet(ROOT / 'features.parquet')
FP_COLS = [
    'rmsd_bb_mean_A','rmsd_as_bb_mean_A','protein_rg_mean_A','as_ca_rmsf_mean_A',
    'lig_drift_mean_A','lig_com_disp_max_A','lig_escape_frac',
    'lig_internal_rmsd_mean_A','lig_rmsf_mean_A',
    'lig_buried_sasa_mean_A2','vdw_contacts_mean',
    'n_hb_mean','hb_persistence_frac','salt_bridges_lp_mean',
    'ifp_tanimoto_median_vs_ref','lig_binding_modes_2A','lig_orient_autocorr_mean',
    'lig_rg_mean_A','lig_asphericity_mean','lig_dipole_mean_eA',
    'active_site_formal_charge','protein_formal_charge','n_active_site_residues',
    'ligand_partial_charge_sum',
]
FP = df.groupby('target')[FP_COLS].mean()

targets = sorted(set(FP.index) & set(bt.target.unique()))
FP = FP.loc[targets]
print(f'{len(targets)} targets, {bt.combo.nunique()} GBSA combos')
print(f'aligned targets: {targets}')

# For each GBSA param axis, compute per-target average BEDROC at each level.
# Delta_pref(target) = BEDROC(level_high) - BEDROC(level_low)  averaged over other params
# Then Spearman across targets between MD feature and delta.

def perm_pvalue(ρ, x, y, n_perm=5000, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    stat_abs = abs(ρ)
    ct = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        rp, _ = spearmanr(x, yp)
        if abs(rp) >= stat_abs: ct += 1
    return ct / n_perm

axes = {
    'igb':    ('igb',     [1,2,5,7,8]),
    'intdiel':('intdiel', [1,2,4]),
    'saltcon':('saltcon', [0.0, 0.15]),
    'surften':('surften', [0.0, 0.0072]),
}

print()
print('=' * 70)
print(' Per-target GBSA parameter preferences  vs  MD signature ')
print('=' * 70)
print(' delta_pref(target) = BEDROC α=20 at higher level  −  BEDROC at lower level')
print(' (marginalised over other params)')
print()

all_rows = []
for ax_name, (col, levels) in axes.items():
    print(f'\n--- axis: {col}  (levels {levels}) ---')
    if len(levels) == 2:
        pairs = [(levels[0], levels[1])]
    else:
        pairs = [(min(levels), max(levels))]  # only extremes for readability
    for lo, hi in pairs:
        sub = bt[bt[col].isin([lo, hi])]
        # per-target mean BEDROC at each level
        agg = sub.groupby(['target', col]).bedroc20_gbsa.mean().unstack(col)
        delta = (agg[hi] - agg[lo]).reindex(targets)
        print(f'\n  Δ = BEDROC(@ {col}={hi}) − BEDROC(@ {col}={lo})  per target:')
        for t in targets:
            print(f'    {t}: {delta[t]:+.3f}')
        # correlation of delta with each MD feature
        rows = []
        for feat in FP_COLS:
            x = FP[feat].values
            y = delta.values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 5: continue
            ρ, _ = spearmanr(x[m], y[m])
            p = perm_pvalue(ρ, x[m], y[m], n_perm=5000)
            rows.append({'feature': feat, 'spearman_r': ρ, 'perm_p': p, 'n_targets': int(m.sum())})
        rank_df = pd.DataFrame(rows).sort_values('spearman_r', key=lambda s: -s.abs())
        rank_df['axis'] = ax_name; rank_df['pair'] = f'{lo}→{hi}'
        all_rows.append(rank_df)
        print(f'  Top-8 MD-feature correlations with the Δ (largest |ρ| first):')
        print(rank_df.head(8).round(3).to_string(index=False))

all_r = pd.concat(all_rows, ignore_index=True).sort_values('spearman_r', key=lambda s: -s.abs())
all_r.to_csv(ROOT / 'gbsa_param_vs_md_correlations.csv', index=False)
print()
print('=' * 70)
print(' STRONGEST correlations across all (axis × feature) pairs ')
print('=' * 70)
strong = all_r[all_r.perm_p < 0.05].head(20)
if len(strong) == 0:
    print('  none: no pair reaches permutation p < 0.05 with only 8 targets.')
    print('\n  Top 10 by |ρ| regardless of significance:')
    print(all_r.head(10).round(3).to_string(index=False))
else:
    print(strong.round(3).to_string(index=False))
print(f'\nWrote {ROOT}/gbsa_param_vs_md_correlations.csv')
