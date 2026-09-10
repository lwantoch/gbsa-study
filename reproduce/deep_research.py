#!/usr/bin/env python3
"""Deep-research: is there an ML approach that connects MD features to
per-target BEDROC lift? Reports honest results including "no signal".

CV: LeaveOneGroupOut on 'target'.
Baselines: random combo, locked-global combo, oracle (per-target argmax).
Models: constant/mean, Logistic/Ridge, RandomForest, HistGradientBoosting.
"""
import json, os, sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.dummy import DummyClassifier, DummyRegressor

ROOT = Path('/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study')
STUDY = ROOT / 'gbsa-study-updated' / 'gbsa-study-updated' / 'gbsa-study'

def bedroc(scores, labels, alpha=20.0):
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=int)
    m = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[m], labels[m]
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels): return np.nan
    order = np.argsort(-scores, kind='stable')
    labels = labels[order]
    N = len(labels); Ra = n_pos / N
    ranks = np.where(labels == 1)[0] + 1
    num = np.sum(np.exp(-alpha * ranks / N))
    denom = Ra * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1)
    Rf = num / denom if denom > 0 else np.nan
    factor = Ra * np.sinh(alpha/2) / (np.cosh(alpha/2) - np.cosh(alpha/2 - alpha*Ra))
    return Rf * factor + 1 / (1 - np.exp(alpha*(1-Ra)))

# ---------- 1. LOAD ----------
print('=' * 70); print(' LOAD '); print('=' * 70)
df = pd.read_parquet(ROOT / 'features.parquet')
print(f'features.parquet: {len(df)} complexes × {df.shape[1]} cols')
meta = pd.read_csv(STUDY / 'data/raw/metadata.csv')
gbsa_raw = pd.read_csv(STUDY / 'data/raw/gbsa_dG_raw.csv').rename(columns={'mean_dG_kcalmol':'gbsa_dG'})
bpartial = pd.read_csv(STUDY / 'data/derived/study2/bedroc20_partial.csv')
BTC = bpartial.pivot(index='target', columns='config', values='bedroc20')
BTC = BTC.dropna(axis=1, thresh=int(0.7 * len(BTC)))
print(f'BEDROC α=20 per (target × sp-config): {BTC.shape}')

FP_COLS = [
    'rmsd_bb_mean_A','rmsd_bb_std_A','rmsd_as_bb_mean_A','rmsd_as_bb_std_A',
    'protein_rg_mean_A','as_ca_rmsf_mean_A','as_ca_rmsf_max_A',
    'lig_drift_mean_A','lig_drift_std_A','lig_drift_last_A',
    'lig_com_disp_max_A','lig_escape_frac',
    'lig_internal_rmsd_mean_A','lig_rmsf_mean_A','lig_rmsf_max_A',
    'lig_buried_sasa_mean_A2','lig_buried_sasa_std_A2',
    'vdw_contacts_mean','vdw_contacts_std','n_hb_mean','n_hb_std','hb_persistence_frac',
    'salt_bridges_lp_mean','ifp_tanimoto_median_vs_ref','ifp_tanimoto_last_vs_ref','ifp_tanimoto_entropy',
    'lig_binding_modes_1A','lig_binding_modes_2A','lig_orient_autocorr_mean','lig_orient_autocorr_last',
    'lig_rg_mean_A','lig_asphericity_mean','lig_dipole_mean_eA','lig_dipole_std_eA',
    'coulomb_mean_arb','coulomb_std_arb',
    'active_site_formal_charge','protein_formal_charge','ligand_partial_charge_sum',
    'protein_n_titratable','n_active_site_residues',
]
FP_mean = df.groupby('target')[FP_COLS].mean()
FP_std  = df.groupby('target')[FP_COLS].std().add_suffix('_std')
FP = pd.concat([FP_mean, FP_std], axis=1)
tgts = sorted(set(FP.index) & set(BTC.index))
FP = FP.loc[tgts]; BTC = BTC.loc[tgts]
print(f'aligned: {len(tgts)} targets')

# ---------- 2. BASELINES ----------
print(); print('=' * 70); print(' BASELINES '); print('=' * 70)
oracle = BTC.max(axis=1)
locked_config = BTC.mean(axis=0).idxmax()
locked = BTC[locked_config]
rng = np.random.default_rng(0)
random_bedroc_means = []
for _ in range(500):
    picks = [rng.choice(BTC.columns[BTC.iloc[t].notna()]) for t in range(len(BTC))]
    random_bedroc_means.append(np.nanmean([BTC.iloc[i][picks[i]] for i in range(len(BTC))]))
random_baseline = float(np.mean(random_bedroc_means))
print(f'Oracle              panel mean: {oracle.mean():.3f}  (max possible)')
print(f'Locked ({locked_config})       panel mean: {locked.mean():.3f}  (current recipe)')
print(f'Random combo (500)  panel mean: {random_baseline:.3f}')

# ---------- 3. P1: predict per-config BEDROC from per-target MD fingerprint ----------
print(); print('=' * 70)
print(' P1: predict per-(target,sp-config) BEDROC from target MD fingerprint ')
print('=' * 70)

p1_results = {}
def eval_p1(name, factory):
    pred = pd.DataFrame(index=BTC.index, columns=BTC.columns, dtype=float)
    X = FP.fillna(FP.median()).values
    loo = LeaveOneOut()
    for c in BTC.columns:
        y = BTC[c].values; mask = np.isfinite(y)
        if mask.sum() < 5: continue
        Xc, yc = X[mask], y[mask]
        for tr, te in loo.split(Xc):
            m = factory(); m.fit(Xc[tr], yc[tr])
            te_orig = np.where(mask)[0][te]
            pred.iloc[te_orig, list(BTC.columns).index(c)] = m.predict(Xc[te])
    picked_true = []
    for t in pred.index:
        row = pred.loc[t].dropna()
        picked_true.append(BTC.loc[t, row.idxmax()] if not row.empty else np.nan)
    picked = pd.Series(picked_true, index=pred.index)
    lift = picked.mean() - locked.mean()
    p1_results[name] = picked
    print(f'  {name:20s}  panel mean = {picked.mean():.3f}  lift vs locked = {lift:+.3f}')

eval_p1('P1 mean',       lambda: DummyRegressor(strategy='mean'))
eval_p1('P1 RidgeCV',    lambda: RidgeCV())
eval_p1('P1 RandomForest',lambda: RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0, n_jobs=-1))
eval_p1('P1 HGBT',       lambda: HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, random_state=0))

# ---------- 4. P2: predict per-complex is_active from MD features (LOTO) ----------
print(); print('=' * 70)
print(' P2: per-complex is_active from MD features (LOTO cross-val) ')
print('=' * 70)

data = df.merge(meta[['complex_id','target','pchembl','is_active','docking_score']],
                on=['target','complex_id'], how='inner', suffixes=('','_meta'))
if 'is_active_meta' in data.columns:
    data['is_active'] = data['is_active_meta']
    data = data.drop(columns=[c for c in data.columns if c.endswith('_meta')])
data = data.dropna(subset=['is_active'])
print(f'labeled complexes: {len(data)} across {data.target.nunique()} targets')

X = data[FP_COLS].astype(float).fillna(data[FP_COLS].median(numeric_only=True))
y = data.is_active.astype(bool).astype(int).values
groups = data.target.values

p2 = {}
def eval_p2(name, factory):
    logo = LeaveOneGroupOut()
    pred = np.full(len(data), np.nan)
    for tr, te in logo.split(X, y, groups):
        m = factory(); m.fit(X.iloc[tr], y[tr])
        if hasattr(m, 'predict_proba'):
            pred[te] = m.predict_proba(X.iloc[te])[:, 1]
        else:
            pred[te] = m.decision_function(X.iloc[te])
    tmp = data.assign(ml=pred).groupby('target').apply(
        lambda g: bedroc(g.ml.values, g.is_active.astype(int).values), include_groups=False)
    aucs = data.assign(ml=pred).groupby('target').apply(
        lambda g: roc_auc_score(g.is_active.astype(int).values, g.ml.values)
                  if g.is_active.nunique() > 1 else np.nan, include_groups=False)
    print(f'  {name:28s}  BEDROC = {tmp.mean():.3f}   AUC = {aucs.mean():.3f}')
    p2[name] = tmp

eval_p2('P2 constant',            lambda: DummyClassifier(strategy='prior'))
eval_p2('P2 logistic',            lambda: LogisticRegression(max_iter=1000, C=0.5))
eval_p2('P2 random forest',       lambda: RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0, n_jobs=-1))
eval_p2('P2 HGBT',                lambda: HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.04, l2_regularization=0.5, random_state=0))

# MD composite (no fit)
COMP = {'lig_drift_mean_A':-1,'lig_escape_frac':-1,'hb_persistence_frac':+1,
        'vdw_contacts_mean':+1,'ifp_tanimoto_median_vs_ref':+1,'lig_binding_modes_2A':-1}
z = data.copy()
for c in COMP:
    g = z.groupby('target')[c]
    z[c+'_z'] = (z[c] - g.transform('mean')) / (g.transform('std') + 1e-9)
z['md_composite'] = sum(z[c+'_z'] * s for c,s in COMP.items())
comp_bedroc = z.groupby('target').apply(
    lambda g: bedroc(g.md_composite.values, g.is_active.astype(int).values), include_groups=False)
p2['MD-composite'] = comp_bedroc
print(f'  {"MD-composite (no fit)":28s}  BEDROC = {comp_bedroc.mean():.3f}')

# docking baseline
dock_bedroc = data.groupby('target').apply(
    lambda g: bedroc((-g.docking_score).values, g.is_active.astype(int).values), include_groups=False)
print(f'  {"docking_score (baseline)":28s}  BEDROC = {dock_bedroc.mean():.3f}')

# GBSA baseline @ locked combo
locked_gbsa_combo = 'igb2_di4_salt0.15_st0.0072'
g_locked = gbsa_raw[gbsa_raw.combo == locked_gbsa_combo][['complex_id','target','gbsa_dG']]
data_g = data.merge(g_locked, on=['complex_id','target'], how='left')
gbsa_bedroc = data_g.dropna(subset=['gbsa_dG']).groupby('target').apply(
    lambda g: bedroc((-g.gbsa_dG).values, g.is_active.astype(int).values), include_groups=False)
print(f'  {"GBSA @ locked combo":28s}  BEDROC = {gbsa_bedroc.mean():.3f}')

# ---------- 5. P3: full-factorial ML ----------
print(); print('=' * 70)
print(' P3: full-factorial ML — MD × GBSA combo × per-combo dG → is_active ')
print('=' * 70)

def parse_combo(s):
    d = {}
    for tok in s.split('_'):
        if tok.startswith('igb'):  d['combo_igb']  = int(tok[3:])
        elif tok.startswith('di'):  d['combo_diel'] = int(tok[2:])
        elif tok.startswith('salt'):d['combo_salt'] = float(tok[4:])
        elif tok.startswith('st'):  d['combo_st']   = float(tok[2:])
    return d
combos_all = gbsa_raw.combo.drop_duplicates().to_frame().reset_index(drop=True)
combos_all = combos_all.join(pd.DataFrame([parse_combo(s) for s in combos_all.combo]))
factorial = (gbsa_raw[['complex_id','target','combo','gbsa_dG']]
             .merge(combos_all, on='combo')
             .merge(df[['complex_id','target'] + FP_COLS], on=['complex_id','target'])
             .merge(meta[['complex_id','target','is_active','pchembl','docking_score']],
                    on=['complex_id','target']))
factorial = factorial.dropna(subset=['is_active','gbsa_dG'])
print(f'factorial rows: {len(factorial):,} | combos: {factorial.combo.nunique()} | targets: {factorial.target.nunique()}')

FEAT_FACT = FP_COLS + ['combo_igb','combo_diel','combo_salt','combo_st','gbsa_dG']
Xf = factorial[FEAT_FACT].astype(float).fillna(factorial[FEAT_FACT].median(numeric_only=True))
yf = factorial.is_active.astype(bool).astype(int).values
gf = factorial.target.values

logo = LeaveOneGroupOut()
pred = np.full(len(factorial), np.nan)
for tr, te in logo.split(Xf, yf, gf):
    m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.04, l2_regularization=0.5, random_state=0)
    m.fit(Xf.iloc[tr], yf[tr])
    pred[te] = m.predict_proba(Xf.iloc[te])[:, 1]
fd = factorial.copy(); fd['ml'] = pred
rows = []
for t, g in fd.groupby('target'):
    per_combo = {c: bedroc(gc.ml.values, gc.is_active.astype(int).values) for c, gc in g.groupby('combo')}
    vals = {k:v for k,v in per_combo.items() if np.isfinite(v)}
    rows.append({
        'target': t,
        'picks_combo': max(vals, key=vals.get) if vals else None,
        'picks_bedroc': max(vals.values()) if vals else np.nan,
        'ensemble_bedroc': bedroc(g.groupby('complex_id').ml.mean().values,
                                   g.groupby('complex_id').is_active.first().astype(int).values),
    })
p3 = pd.DataFrame(rows)
print(f'  P3 HGBT — ML picks GBSA combo:    BEDROC = {p3.picks_bedroc.mean():.3f}')
print(f'  P3 HGBT — ML ensemble over combos: BEDROC = {p3.ensemble_bedroc.mean():.3f}')

# ---------- 6. SUMMARY ----------
print(); print('=' * 70); print(' SUMMARY (higher BEDROC = better) '); print('=' * 70)
rows = [
    ('random combo (500 draws)',             random_baseline),
    (f'LOCKED sp-config ({locked_config})',  locked.mean()),
    ('ORACLE (per-target argmax, upper bnd)', oracle.mean()),
]
for k, v in p1_results.items():
    rows.append((k, v.mean()))
for k, v in p2.items():
    rows.append((k, v.mean()))
rows += [
    ('docking_score baseline',           dock_bedroc.mean()),
    ('GBSA @ locked combo baseline',     gbsa_bedroc.mean()),
    ('P3 HGBT full-fact, ML-picks combo', p3.picks_bedroc.mean()),
    ('P3 HGBT full-fact, ML-ensemble',    p3.ensemble_bedroc.mean()),
]
summary = pd.DataFrame(rows, columns=['approach','panel_bedroc']).round(3)
print(summary.to_string(index=False))

# --- write out ---
summary.to_csv(ROOT / 'deep_research_summary.csv', index=False)

# per-target detail
per_target = pd.DataFrame({'oracle': oracle, 'locked_sp': locked}, index=BTC.index)
for k, v in p1_results.items():
    per_target[k] = v
per_target.to_csv(ROOT / 'deep_research_p1_per_target.csv')

p2_df = pd.DataFrame(p2)
p2_df['docking'] = dock_bedroc
p2_df['gbsa_locked'] = gbsa_bedroc
p2_df.round(3).to_csv(ROOT / 'deep_research_p2_per_target.csv')

p3.to_csv(ROOT / 'deep_research_p3_per_target.csv', index=False)

print(f'\nSaved 4 result CSVs to {ROOT}/deep_research_*.csv')
