#!/usr/bin/env python3
"""V2: correlate MD + REAL ligand chemistry with GBSA parameter preferences.
Now with BH-FDR correction and honest reporting.

Also tries ML: per-complex GBSA prediction with MD + ligand-chem features.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')

ROOT = Path('/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study')
STUDY = ROOT / 'gbsa-study-updated/gbsa-study-updated/gbsa-study'

df = pd.read_parquet(ROOT / 'features.parquet')
lig = pd.read_parquet(ROOT / 'ligand_chem.parquet')
bt = pd.read_csv(STUDY / 'data/derived/temporal/bedroc_all_combos_per_target.csv')
gbsa_raw = pd.read_csv(STUDY / 'data/raw/gbsa_dG_raw.csv').rename(columns={'mean_dG_kcalmol':'gbsa_dG'})
meta = pd.read_csv(STUDY / 'data/raw/metadata.csv')

full = df.merge(lig, on=['target','complex_id'], how='inner')
print(f'joined MD+lig-chem: {len(full)} complexes  ·  cols {full.shape[1]}')

# MD + ligand-chem features (per-target aggregate = mean)
MD_FEATS = [
    'rmsd_bb_mean_A','rmsd_as_bb_mean_A','protein_rg_mean_A','as_ca_rmsf_mean_A',
    'lig_drift_mean_A','lig_com_disp_max_A','lig_escape_frac',
    'lig_internal_rmsd_mean_A','lig_rmsf_mean_A',
    'lig_buried_sasa_mean_A2','vdw_contacts_mean',
    'n_hb_mean','hb_persistence_frac','salt_bridges_lp_mean',
    'ifp_tanimoto_median_vs_ref','lig_binding_modes_2A','lig_orient_autocorr_mean',
    'lig_rg_mean_A','lig_asphericity_mean','lig_dipole_mean_eA',
    'active_site_formal_charge','protein_formal_charge','n_active_site_residues',
]
LIG_FEATS = [
    'lig_MW','lig_n_heavy','lig_rot_bonds','lig_HBD','lig_HBA','lig_all_rings',
    'lig_LogP','lig_TPSA','lig_fraction_sp3','lig_partial_q_abs_sum',
]
ALL_FEATS = MD_FEATS + LIG_FEATS

FP = full.groupby('target')[ALL_FEATS].mean()
targets = sorted(set(FP.index) & set(bt.target.unique()))
FP = FP.loc[targets]
print(f'{len(targets)} targets, {len(ALL_FEATS)} features ({len(MD_FEATS)} MD + {len(LIG_FEATS)} lig-chem)')

def perm_p(rho, x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed); s = abs(rho); ct = 0
    for _ in range(n):
        yp = rng.permutation(y)
        r, _ = spearmanr(x, yp)
        if abs(r) >= s: ct += 1
    return ct/n

axes = {
    'igb 1→8':     ('igb',     1, 8),
    'intdiel 1→4': ('intdiel', 1, 4),
    'salt 0→0.15': ('saltcon', 0.0, 0.15),
    'surften 0→0.0072': ('surften', 0.0, 0.0072),
}

rho_mat = pd.DataFrame(index=ALL_FEATS, columns=list(axes.keys()), dtype=float)
p_mat = pd.DataFrame(index=ALL_FEATS, columns=list(axes.keys()), dtype=float)

for ax_name, (col, lo, hi) in axes.items():
    sub = bt[bt[col].isin([lo, hi])]
    agg = sub.groupby(['target', col]).bedroc20_gbsa.mean().unstack(col)
    delta = (agg[hi] - agg[lo]).reindex(targets)
    for feat in ALL_FEATS:
        x = FP[feat].values; y = delta.values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 5:
            rho_mat.loc[feat, ax_name] = np.nan; p_mat.loc[feat, ax_name] = np.nan; continue
        rho, _ = spearmanr(x[m], y[m])
        rho_mat.loc[feat, ax_name] = rho
        p_mat.loc[feat, ax_name] = perm_p(rho, x[m], y[m], n=5000)

# BH-FDR
flat_p = p_mat.stack().dropna()
_, q, _, _ = multipletests(flat_p.values, method='fdr_bh')
q_mat = p_mat.astype(float).copy()
for (feat, ax), qv in zip(flat_p.index, q):
    q_mat.loc[feat, ax] = qv

# Long-form ranked table
rows = []
for feat in rho_mat.index:
    for ax in rho_mat.columns:
        rows.append({'feature': feat, 'axis': ax,
                     'is_lig_chem': feat in LIG_FEATS,
                     'spearman_r': rho_mat.loc[feat, ax],
                     'perm_p': p_mat.loc[feat, ax],
                     'bh_q': q_mat.loc[feat, ax]})
rank = pd.DataFrame(rows).dropna(subset=['spearman_r']).sort_values('bh_q')
print('\n=== Ranked correlations, BH-FDR corrected across 4×{} = {} tests ==='.format(len(ALL_FEATS), 4*len(ALL_FEATS)))
print(rank.head(15).round(3).to_string(index=False))

print('\n=== Signals surviving BH-FDR q<0.10 ===')
strong = rank[rank.bh_q < 0.10]
if strong.empty:
    print('  none. best q =', rank.bh_q.min().round(3))
else:
    print(strong.round(3).to_string(index=False))

print('\n=== Best correlation per axis (regardless of q) ===')
for ax in axes:
    sub = rank[rank.axis == ax].head(3)
    print(f'\n  --- {ax} ---')
    print(sub[['feature','spearman_r','perm_p','bh_q']].round(3).to_string(index=False))

rank.to_csv(ROOT / 'gbsa_param_vs_md_v2_correlations.csv', index=False)
rho_mat.to_csv(ROOT / 'gbsa_param_vs_md_v2_rho_matrix.csv')
q_mat.to_csv(ROOT / 'gbsa_param_vs_md_v2_q_matrix.csv')
print(f'\nsaved 3 CSVs to {ROOT}/')

# ============================================================
# ML: does adding ligand-chem improve per-complex GBSA prediction?
# ============================================================
print('\n' + '='*70)
print(' ML: MD alone vs MD+ligand-chem for per-complex GBSA prediction ')
print('='*70)

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score

LOCKED = 'igb2_di4_salt0.15_st0.0072'
g_locked = gbsa_raw[gbsa_raw.combo == LOCKED][['complex_id','target','gbsa_dG']]
data = full.merge(g_locked, on=['complex_id','target'], how='inner')
data = data.dropna(subset=['gbsa_dG'])
print(f'GBSA @ locked combo: {len(data)} labeled complexes')

def loto_r2(feature_cols, name):
    X = data[feature_cols].astype(float).fillna(data[feature_cols].median(numeric_only=True))
    y = data.gbsa_dG.values
    g = data.target.values
    logo = LeaveOneGroupOut()
    pred = np.full(len(data), np.nan)
    for tr, te in logo.split(X, y, g):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.04, l2_regularization=0.5, random_state=0)
        m.fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    tmp = pd.DataFrame({'y': y, 'p': pred, 'target': g})
    per_t_r2 = tmp.groupby('target').apply(lambda g: r2_score(g.y, g.p), include_groups=False)
    per_t_r = tmp.groupby('target').apply(lambda g: np.corrcoef(g.y, g.p)[0,1], include_groups=False)
    print(f'  {name:32s}  per-target Pearson r median: {per_t_r.median():.3f}   mean: {per_t_r.mean():.3f}')
    return per_t_r

r_md   = loto_r2(MD_FEATS,                'MD features only')
r_lig  = loto_r2(LIG_FEATS,               'ligand-chem only')
r_both = loto_r2(MD_FEATS + LIG_FEATS,    'MD + ligand-chem')

# Save comparison
comp = pd.DataFrame({'MD_only': r_md, 'lig_only': r_lig, 'MD+lig': r_both}).round(3)
comp.to_csv(ROOT / 'gbsa_prediction_r_by_target.csv')
print('\nper-target Pearson r (predicting GBSA @ locked combo, LOTO):')
print(comp.to_string())
