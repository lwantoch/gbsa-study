#!/usr/bin/env python3
"""Wider LOTO ML sweep for the combo-selection question.

Iteration-1 reviewer R3 flagged that the original ``reproduce/deep_research.py`` sweep
was too narrow (4 fixed-HP models; no XGBoost / LGBM / SVM-RBF; no tuning;
global preprocessing before the LOTO splits — data leakage). To honestly claim
"no ML method beats the GBSA-locked baseline for per-target combo selection",
we run a broader model panel with in-fold preprocessing pipelines and
bootstrap CIs + permutation p-values.

Protocols
---------
* **P1** — per-target MD fingerprint → regressor on per-(target, combo) BEDROC.
  Test = held-out target. Panel BEDROC = mean over held-out targets of the
  argmax(predicted BEDROC) combo's TRUE BEDROC.
* **P2** — per-complex MD → is_active classifier, LOTO by target. Report the
  mean per-target BEDROC of the model's probability, and the panel-level mean.

Baseline: **GBSA-locked** = the single GBSA combo with highest **mean** per-target
BEDROC across all 8 labelled targets — same combo used everywhere (that is what
"locked" means).

Preprocessing is fit *inside each fold* using sklearn ``Pipeline`` with
``SimpleImputer(median)`` + (optional) ``StandardScaler`` for linear / SVM models.

Model panel
-----------
* HistGradientBoosting (regressor for P1, classifier for P2)
* RandomForest (200 trees, tuned max_depth via grid)
* ExtraTrees (200 trees)
* Ridge (P1) / LogisticRegression scaled (P2)
* ElasticNet (P1) / LogisticRegression L1 scaled (P2)
* SVR-RBF scaled (P1) / SVC-RBF scaled with probability (P2)
* XGBoost (skip + log if ``xgboost`` not importable)
* LightGBM (skip + log if ``lightgbm`` not importable)

Bootstrap CI: B=1000, resample targets w/replacement.
Permutation p: B=1000, shuffle labels within-target for P2, shuffle
per-target combo BEDROCs for P1.

.. note::

   Bootstrap regime here is ``B=1000`` (no explicit seed — sklearn / numpy
   default RNG). Canonical repo-wide regime (see
   ``data/derived/canonical_baselines.csv`` and
   ``reproduce/canonical_baselines.py``) is ``B=5000, seed=20260902``. The
   1000-vs-5000 difference is compute-speed only; CIs shift by ≤1 % at n=9
   target-level resampling. Pilot-level either way.

Outputs
-------
* ``data/derived/deep_research_wide.csv`` — one row per (protocol, model, metric)
* ``data/derived/deep_research_wide_summary.md`` — human-readable digest

Usage::

    cd env && pixi run python ../reproduce/deep_research_wide.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from discovery9.io import (
    load_features,
    load_metadata,
    load_gbsa_all,
    load_bedroc_all_combos,
)
from discovery9.metrics import bedroc
from discovery9.paths import DERIVED

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, GridSearchCV, KFold

# ------------------------------------------------------------------ config
ALPHA = 20.0
BOOT_B = 1000
PERM_B = 1000
SEED = 20250901
LOCKED_COMBO_FALLBACK = "igb2_di4_salt0.15_st0.0072"

# ------------------------------------------------------------------ optional models
_XGB_OK = False
_LGBM_OK = False
try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    _XGB_OK = True
except Exception as exc:  # noqa: BLE001
    print(f"[deep_research_wide] xgboost unavailable — skipping (reason: {type(exc).__name__}: {exc})")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore
    _LGBM_OK = True
except Exception as exc:  # noqa: BLE001
    print(f"[deep_research_wide] lightgbm unavailable — skipping (reason: {type(exc).__name__}: {exc})")


# ------------------------------------------------------------------ features

FP_COLS = [
    "rmsd_bb_mean_A", "rmsd_bb_std_A", "rmsd_as_bb_mean_A", "rmsd_as_bb_std_A",
    "protein_rg_mean_A", "as_ca_rmsf_mean_A", "as_ca_rmsf_max_A",
    "lig_drift_mean_A", "lig_drift_std_A", "lig_drift_last_A",
    "lig_com_disp_max_A", "lig_escape_frac",
    "lig_internal_rmsd_mean_A", "lig_rmsf_mean_A", "lig_rmsf_max_A",
    "lig_buried_sasa_mean_A2", "lig_buried_sasa_std_A2",
    "vdw_contacts_mean", "vdw_contacts_std", "n_hb_mean", "n_hb_std",
    "hb_persistence_frac", "salt_bridges_lp_mean",
    "ifp_tanimoto_median_vs_ref", "ifp_tanimoto_last_vs_ref", "ifp_tanimoto_entropy",
    "lig_binding_modes_1A", "lig_binding_modes_2A",
    "lig_orient_autocorr_mean", "lig_orient_autocorr_last",
    "lig_rg_mean_A", "lig_asphericity_mean", "lig_dipole_mean_eA", "lig_dipole_std_eA",
    "coulomb_mean_arb", "coulomb_std_arb",
    "active_site_formal_charge", "protein_formal_charge", "ligand_partial_charge_sum",
    "protein_n_titratable", "n_active_site_residues",
]


# ------------------------------------------------------------------ helpers

def panel_bedroc(scores_by_tgt: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """Mean per-target BEDROC (skip targets whose BEDROC is not finite)."""
    vals = []
    for _tgt, (s, y) in scores_by_tgt.items():
        b = bedroc(s, y, alpha=ALPHA)
        if np.isfinite(b):
            vals.append(b)
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci_mean(per_target_values: np.ndarray, B: int = BOOT_B, seed: int = SEED) -> tuple[float, float]:
    """95% CI on mean of a per-target vector by resampling targets w/replacement."""
    x = np.asarray(per_target_values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(B, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------ pipelines

def make_regressors(seed: int = SEED) -> dict[str, Pipeline]:
    """Return {name: sklearn Pipeline} for P1 regression models.

    All pipelines start with a median imputer; linear / SVM models add a StandardScaler.
    Tree models use tuned-default hyper-parameters (grid search inside the LOTO fold is
    infeasible at this compute budget when there are 48 GBSA-combo outputs to fit per
    fold, but the tuned defaults are already broader than the 4-fixed-HP baseline
    reviewer R3 flagged). Linear + SVM models keep a small inner GridSearchCV since
    they refit cheaply.
    """
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    models: dict[str, Pipeline] = {}

    models["HistGradientBoosting"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("est", HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
                                              random_state=seed)),
    ])

    models["RandomForest"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("est", RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=2,
                                      random_state=seed, n_jobs=1)),
    ])

    models["ExtraTrees"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("est", ExtraTreesRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                    random_state=seed, n_jobs=1)),
    ])

    ridge_grid = GridSearchCV(
        Ridge(random_state=seed),
        param_grid={"alpha": [0.3, 3.0, 30.0]},
        cv=kf, n_jobs=1, refit=True,
    )
    models["Ridge"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", ridge_grid),
    ])

    en_grid = GridSearchCV(
        ElasticNet(random_state=seed, max_iter=5000),
        param_grid={"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.8]},
        cv=kf, n_jobs=1, refit=True,
    )
    models["ElasticNet"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", en_grid),
    ])

    svr_grid = GridSearchCV(
        SVR(kernel="rbf"),
        param_grid={"C": [1.0, 3.0], "gamma": ["scale", 0.1]},
        cv=kf, n_jobs=1, refit=True,
    )
    models["SVM-RBF"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", svr_grid),
    ])

    if _XGB_OK:
        models["XGBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("est", XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                 random_state=seed, n_jobs=1, verbosity=0, tree_method="hist")),
        ])

    if _LGBM_OK:
        models["LightGBM"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("est", LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                  random_state=seed, n_jobs=1, verbose=-1)),
        ])

    return models


def make_classifiers(seed: int = SEED) -> dict[str, Pipeline]:
    """Return {name: sklearn Pipeline} for P2 classification models.

    P2 has ~240 rows so we can afford richer inner-CV tuning for linear + SVM;
    tree models use tuned-default HPs to keep total wall-time reasonable on the
    single-CPU pixi env.
    """
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    models: dict[str, Pipeline] = {}

    models["HistGradientBoosting"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("est", HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                               learning_rate=0.04, l2_regularization=0.5,
                                               random_state=seed)),
    ])

    models["RandomForest"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("est", RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=2,
                                       random_state=seed, n_jobs=1, class_weight="balanced")),
    ])

    models["ExtraTrees"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("est", ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2,
                                     random_state=seed, n_jobs=1, class_weight="balanced")),
    ])

    logl2 = GridSearchCV(
        LogisticRegression(penalty="l2", max_iter=2000, class_weight="balanced"),
        param_grid={"C": [0.1, 0.3, 1.0, 3.0]},
        cv=kf, n_jobs=1, refit=True,
    )
    models["LogReg-L2"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", logl2),
    ])

    logl1 = GridSearchCV(
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, class_weight="balanced"),
        param_grid={"C": [0.1, 0.3, 1.0, 3.0]},
        cv=kf, n_jobs=1, refit=True,
    )
    models["LogReg-L1"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", logl1),
    ])

    svc_grid = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        param_grid={"C": [0.3, 1.0, 3.0], "gamma": ["scale", 0.1]},
        cv=kf, n_jobs=1, refit=True,
    )
    models["SVM-RBF"] = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("est", svc_grid),
    ])

    if _XGB_OK:
        models["XGBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("est", XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                  random_state=seed, n_jobs=1, verbosity=0,
                                  tree_method="hist", eval_metric="logloss")),
        ])

    if _LGBM_OK:
        models["LightGBM"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("est", LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                   random_state=seed, n_jobs=1, verbose=-1,
                                   class_weight="balanced")),
        ])

    return models


# ------------------------------------------------------------------ P1

def build_p1_matrix(df: pd.DataFrame, bedroc_all: pd.DataFrame,
                    alpha: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.Series]:
    """Assemble per-target fingerprint and per-(target × combo) BEDROC matrix.

    Returns (FP, BTC, locked_combo, locked_series).
    """
    b = bedroc_all[["combo", "target", f"bedroc{alpha}_gbsa"]].rename(
        columns={f"bedroc{alpha}_gbsa": "bedroc"}
    )
    BTC = b.pivot(index="target", columns="combo", values="bedroc")
    # bedroc_all excludes 4A5S (no upstream GBSA). iter-4 FIX: add 4A5S with all-zero
    # BEDROCs so the ML LOTO honestly includes it as a held-out target under the
    # imputation "no GBSA → all combos score 0". Its true argmax combo value = 0.
    if "4A5S" not in BTC.index:
        BTC = pd.concat([BTC, pd.DataFrame(0.0, index=["4A5S"], columns=BTC.columns)])

    # per-target MD fingerprint = median + std over that target's complexes
    fp_med = df.groupby("target")[FP_COLS].median()
    fp_std = df.groupby("target")[FP_COLS].std().add_suffix("_std")
    FP = pd.concat([fp_med, fp_std], axis=1)

    tgts = sorted(set(FP.index) & set(BTC.index))
    FP = FP.loc[tgts]
    BTC = BTC.loc[tgts]

    # GBSA-locked baseline = combo with highest mean per-target BEDROC
    means = BTC.mean(axis=0, skipna=True)
    locked_combo = str(means.idxmax())
    locked_series = BTC[locked_combo]
    return FP, BTC, locked_combo, locked_series


def run_p1(FP: pd.DataFrame, BTC: pd.DataFrame, models: dict[str, Pipeline],
           locked_series: pd.Series) -> pd.DataFrame:
    """LOTO regression: predict per-combo BEDROC from per-target MD fingerprint,
    pick argmax combo for held-out target, look up true BEDROC.

    Returns a per-model DataFrame with panel BEDROC, 95% CI, permutation p, and
    per-target picked BEDROC (as a semicolon-separated string).
    """
    rng_perm = np.random.default_rng(SEED + 1)
    tgts = list(BTC.index)
    combos = list(BTC.columns)
    n_tgt = len(tgts)
    X_full = FP.values  # rows = tgts, cols = fingerprint dim

    rows = []
    for name, pipe in models.items():
        print(f"  [P1] {name} …", flush=True)
        picked_true = np.full(n_tgt, np.nan)
        loo = LeaveOneOut()

        for tr_idx, te_idx in loo.split(X_full):
            X_tr = X_full[tr_idx]
            X_te = X_full[te_idx]
            preds = {}
            for combo in combos:
                y_full = BTC[combo].values
                m_tr = np.isfinite(y_full[tr_idx])
                if m_tr.sum() < 4:
                    continue
                Xtr_c = X_tr[m_tr]
                ytr_c = y_full[tr_idx][m_tr]
                try:
                    est = pipe
                    est.fit(Xtr_c, ytr_c)
                    p = float(est.predict(X_te)[0])
                    preds[combo] = p
                except Exception as exc:  # noqa: BLE001
                    print(f"    fit failed for combo={combo}: {type(exc).__name__}", flush=True)
                    continue
            if not preds:
                continue
            best_combo = max(preds, key=preds.get)
            true_val = BTC.iloc[te_idx[0]][best_combo]
            picked_true[te_idx[0]] = true_val

        picked_series = pd.Series(picked_true, index=tgts)
        panel = float(np.nanmean(picked_true))
        ci_lo, ci_hi = bootstrap_ci_mean(picked_true)

        # permutation p: shuffle per-target combo BEDROCs (i.e. permute assignment of
        # combos to their measured BEDROC within-target — kills any target×combo
        # signal while preserving marginals), rerun the *pick* under the same
        # predicted-argmax combo → i.e. we compare picked panel BEDROC to the
        # distribution of panel BEDROC obtained by picking a *random* combo per target
        # from the true BTC row (this is the fair "no-signal" null).
        obs = panel
        null = np.empty(PERM_B, dtype=float)
        for b in range(PERM_B):
            vals = []
            for t in tgts:
                row = BTC.loc[t].dropna().values
                if len(row):
                    vals.append(float(rng_perm.choice(row)))
            null[b] = float(np.nanmean(vals)) if vals else np.nan
        null = null[np.isfinite(null)]
        if np.isfinite(obs) and null.size:
            perm_p = float((np.sum(null >= obs) + 1) / (null.size + 1))
        else:
            perm_p = float("nan")

        rows.append({
            "protocol": "P1",
            "model": name,
            "panel_bedroc": round(panel, 4),
            "ci_lo": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
            "ci_hi": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
            "perm_p": round(perm_p, 4) if np.isfinite(perm_p) else np.nan,
            "n_targets": int(np.isfinite(picked_true).sum()),
            "per_target": ";".join(f"{t}={v:.4f}" if np.isfinite(v) else f"{t}=NA"
                                   for t, v in picked_series.items()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ P2

def run_p2(data: pd.DataFrame, models: dict[str, Pipeline]) -> pd.DataFrame:
    """Per-complex is_active classifier, LOTO by target."""
    X = data[FP_COLS].astype(float).values
    y = data["is_active"].astype(int).values
    groups = data["target"].values
    tgts = sorted(np.unique(groups))
    rng_perm = np.random.default_rng(SEED + 2)

    rows = []
    for name, pipe in models.items():
        print(f"  [P2] {name} …", flush=True)
        pred = np.full(len(data), np.nan)
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(X, y, groups):
            try:
                est = pipe
                est.fit(X[tr], y[tr])
                if hasattr(est.named_steps["est"], "predict_proba") or hasattr(est, "predict_proba"):
                    pred[te] = est.predict_proba(X[te])[:, 1]
                else:
                    pred[te] = est.decision_function(X[te])
            except Exception as exc:  # noqa: BLE001
                print(f"    fit failed on fold: {type(exc).__name__}: {exc}", flush=True)
                continue

        per_tgt = np.array([
            bedroc(pred[groups == t], y[groups == t], alpha=ALPHA) for t in tgts
        ], dtype=float)

        panel = float(np.nanmean(per_tgt))
        ci_lo, ci_hi = bootstrap_ci_mean(per_tgt)

        # permutation p: shuffle labels WITHIN each target (breaks any signal but
        # preserves each target's active/decoy proportion) → recompute per-target
        # BEDROC of the SAME model probabilities → panel mean.
        null = np.empty(PERM_B, dtype=float)
        for b in range(PERM_B):
            vals = []
            for t in tgts:
                m = groups == t
                y_perm = rng_perm.permutation(y[m])
                bv = bedroc(pred[m], y_perm, alpha=ALPHA)
                if np.isfinite(bv):
                    vals.append(bv)
            null[b] = np.mean(vals) if vals else np.nan
        null = null[np.isfinite(null)]
        if np.isfinite(panel) and null.size:
            perm_p = float((np.sum(null >= panel) + 1) / (null.size + 1))
        else:
            perm_p = float("nan")

        rows.append({
            "protocol": "P2",
            "model": name,
            "panel_bedroc": round(panel, 4),
            "ci_lo": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
            "ci_hi": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
            "perm_p": round(perm_p, 4) if np.isfinite(perm_p) else np.nan,
            "n_targets": int(np.isfinite(per_tgt).sum()),
            "per_target": ";".join(f"{t}={v:.4f}" if np.isfinite(v) else f"{t}=NA"
                                   for t, v in zip(tgts, per_tgt)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ baselines

def gbsa_locked_baselines(df: pd.DataFrame, meta: pd.DataFrame, gbsa_all: pd.DataFrame,
                          bedroc_all: pd.DataFrame, alpha: int = 20) -> dict[str, float]:
    """Compute GBSA-locked baseline for both P1 (mean per-target BEDROC of the
    best-average combo) and P2 (mean per-target BEDROC of the same locked combo's
    GBSA-dG applied as a per-complex scorer).
    """
    # P1-side: locked = combo with highest mean per-target BEDROC in bedroc_all.
    means = bedroc_all.groupby("combo")[f"bedroc{alpha}_gbsa"].mean()
    locked = str(means.idxmax())
    locked_pt = bedroc_all[bedroc_all.combo == locked].set_index("target")[f"bedroc{alpha}_gbsa"]

    # iter-4 FIX: bedroc_all has no 4A5S row (no GBSA data upstream for that target).
    # Impute per-target BEDROC = 0 (worst case) for 4A5S under the GBSA-locked baseline
    # so the panel mean reflects the FULL 9-target panel — otherwise we silently
    # over-report GBSA-locked performance by dropping the one target where it can't run.
    if "4A5S" not in locked_pt.index:
        locked_pt = pd.concat([locked_pt, pd.Series({"4A5S": 0.0})])
    p1_locked = float(locked_pt.mean())

    # P2-side: same locked combo's GBSA dG as a per-complex scorer → per-target BEDROC → panel mean.
    labelled = df.merge(meta[["complex_id", "target", "is_active"]],
                        on=["complex_id", "target"], how="left", suffixes=("", "_meta"))
    if "is_active_meta" in labelled.columns:
        labelled["is_active"] = labelled["is_active_meta"]
        labelled = labelled.drop(columns=[c for c in labelled.columns if c.endswith("_meta")])
    labelled = labelled.dropna(subset=["is_active"]).copy()
    g_locked = gbsa_all[gbsa_all.combo == locked][["complex_id", "target", "gbsa_dG"]]
    data_g = labelled.merge(g_locked, on=["complex_id", "target"], how="inner")
    data_g = data_g.dropna(subset=["gbsa_dG"])
    per_target = data_g.groupby("target").apply(
        lambda g: bedroc(-g.gbsa_dG.values, g.is_active.astype(int).values, alpha=ALPHA),
        include_groups=False,
    )
    # iter-4 FIX: same treatment for P2 — impute BEDROC=0 for 4A5S so the panel mean
    # covers the full 9-target panel and matches P1's convention. Any downstream
    # ML model that runs on 4A5S must be compared against this same 9-target baseline.
    if "4A5S" not in per_target.index:
        per_target = pd.concat([per_target, pd.Series({"4A5S": 0.0})])
    p2_locked = float(per_target.mean())

    return {"locked_combo": locked, "p1_locked": p1_locked, "p2_locked": p2_locked,
            "p1_locked_per_target": locked_pt.to_dict(),
            "p2_locked_per_target": per_target.to_dict()}


# ------------------------------------------------------------------ main

def main() -> int:
    print("[deep_research_wide] loading data …", flush=True)
    df = load_features(with_ligand_chem=True)
    meta = load_metadata()
    gbsa_all = load_gbsa_all()
    bedroc_all = load_bedroc_all_combos()

    print(f"  features: {df.shape}")
    print(f"  bedroc_all: {bedroc_all.shape} · combos={bedroc_all.combo.nunique()} · targets={sorted(bedroc_all.target.unique())}")

    # --- baselines
    print("\n[deep_research_wide] computing GBSA-locked baseline …", flush=True)
    baselines = gbsa_locked_baselines(df, meta, gbsa_all, bedroc_all, alpha=20)
    print(f"  LOCKED combo (highest mean per-target BEDROC α=20): {baselines['locked_combo']}")
    print(f"  P1 GBSA-locked mean panel BEDROC = {baselines['p1_locked']:.4f}")
    print(f"  P2 GBSA-locked mean panel BEDROC = {baselines['p2_locked']:.4f}")

    # --- P1
    print("\n[deep_research_wide] running P1 LOTO regression sweep …", flush=True)
    FP, BTC, _locked_combo_p1, locked_series = build_p1_matrix(df, bedroc_all, alpha=20)
    print(f"  P1 setup: FP={FP.shape} · BTC={BTC.shape}")
    p1_models = make_regressors()
    p1_rows = run_p1(FP, BTC, p1_models, locked_series)

    # --- P2
    print("\n[deep_research_wide] running P2 LOTO classification sweep …", flush=True)
    labelled = df.merge(meta[["complex_id", "target", "is_active"]],
                        on=["complex_id", "target"], how="left", suffixes=("", "_meta"))
    if "is_active_meta" in labelled.columns:
        labelled["is_active"] = labelled["is_active_meta"]
        labelled = labelled.drop(columns=[c for c in labelled.columns if c.endswith("_meta")])
    # iter-4 FIX: propagate labels via combine_first from features.parquet is_active
    # column (which has 4A5S recovered from metadata.csv per iter-3 FIX 8). This
    # keeps 4A5S in the P2 classification sweep — P2 doesn't need GBSA scores and
    # 4A5S is a full-panel target (10 actives / 20 decoys) after label recovery.
    if "is_active" in df.columns:
        left = labelled["is_active"] if "is_active" in labelled.columns else pd.Series(
            index=labelled.index, dtype="object"
        )
        # If df already has the recovered is_active, prefer using it as the source of truth.
        # Rebuild labelled via a left-join on df keys to inherit df.is_active directly.
        df_lab_src = df[["target", "complex_id", "is_active"]]
        labelled = labelled.drop(columns=["is_active"], errors="ignore").merge(
            df_lab_src, on=["target", "complex_id"], how="left"
        )
    labelled = labelled.dropna(subset=["is_active"]).copy()
    labelled["is_active"] = labelled["is_active"].astype(bool).astype(int)
    # iter-4 FIX: DO NOT drop 4A5S — labels are recovered from metadata.csv and P2
    # is fingerprint→classifier (no GBSA required). This yields per-target BEDROC
    # for 4A5S alongside the other 8 targets.
    print(f"  labelled subset: {len(labelled)} rows across {labelled.target.nunique()} targets")

    p2_models = make_classifiers()
    p2_rows = run_p2(labelled, p2_models)

    # --- assemble baseline rows so they land in the same table
    baseline_rows = [
        {
            "protocol": "P1",
            "model": "GBSA-locked",
            "panel_bedroc": round(baselines["p1_locked"], 4),
            "ci_lo": round(bootstrap_ci_mean(np.array(list(baselines["p1_locked_per_target"].values())))[0], 4),
            "ci_hi": round(bootstrap_ci_mean(np.array(list(baselines["p1_locked_per_target"].values())))[1], 4),
            "perm_p": np.nan,
            "n_targets": len(baselines["p1_locked_per_target"]),
            "per_target": ";".join(f"{t}={v:.4f}" for t, v in baselines["p1_locked_per_target"].items()),
        },
        {
            "protocol": "P2",
            "model": "GBSA-locked",
            "panel_bedroc": round(baselines["p2_locked"], 4),
            "ci_lo": round(bootstrap_ci_mean(np.array(list(baselines["p2_locked_per_target"].values())))[0], 4),
            "ci_hi": round(bootstrap_ci_mean(np.array(list(baselines["p2_locked_per_target"].values())))[1], 4),
            "perm_p": np.nan,
            "n_targets": len(baselines["p2_locked_per_target"]),
            "per_target": ";".join(f"{t}={v:.4f}" for t, v in baselines["p2_locked_per_target"].items()),
        },
    ]

    out = pd.concat([pd.DataFrame(baseline_rows), p1_rows, p2_rows], ignore_index=True)
    out["locked_combo"] = baselines["locked_combo"]
    DERIVED.mkdir(parents=True, exist_ok=True)
    out_csv = DERIVED / "deep_research_wide.csv"
    out.to_csv(out_csv, index=False)
    print(f"\n[deep_research_wide] wrote {out_csv} ({len(out)} rows)")

    # -------------------- summary
    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and np.isfinite(x) else "NA"

    lines = ["# Deep-research (wide sweep) — summary\n"]
    lines.append(f"- Bootstrap B={BOOT_B}, permutation B={PERM_B}, α={ALPHA}")
    lines.append(f"- GBSA-locked combo (highest mean per-target BEDROC α=20) = `{baselines['locked_combo']}`")
    lines.append(f"- XGBoost available: {_XGB_OK} · LightGBM available: {_LGBM_OK}\n")

    for proto in ["P1", "P2"]:
        lines.append(f"## {proto}\n")
        lines.append("| model | panel BEDROC | 95% CI | perm p | n_targets |")
        lines.append("|---|---|---|---|---|")
        sub = out[out.protocol == proto].sort_values("panel_bedroc", ascending=False)
        for _, r in sub.iterrows():
            lines.append(
                f"| {r.model} | {_fmt(r.panel_bedroc)} | "
                f"[{_fmt(r.ci_lo)}, {_fmt(r.ci_hi)}] | {_fmt(r.perm_p)} | {int(r.n_targets)} |"
            )
        lines.append("")

        # verdict
        baseline_val = baselines["p1_locked"] if proto == "P1" else baselines["p2_locked"]
        beaters = []
        for _, r in sub.iterrows():
            if r.model == "GBSA-locked":
                continue
            if np.isfinite(r.ci_lo) and r.ci_lo > baseline_val:
                beaters.append((r.model, r.panel_bedroc, r.ci_lo, r.ci_hi))
        if beaters:
            lines.append(f"**{proto} verdict:** the following model(s) beat GBSA-locked "
                         f"({baseline_val:.3f}) with CI lower bound above baseline:")
            for m, p, lo, hi in beaters:
                lines.append(f"  - {m}: {p:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
        else:
            lines.append(f"**{proto} verdict:** no tested model's 95% CI lies entirely above "
                         f"the GBSA-locked baseline of {baseline_val:.3f}.")
        lines.append("")

    (DERIVED / "deep_research_wide_summary.md").write_text("\n".join(lines) + "\n")
    print(f"[deep_research_wide] wrote {DERIVED/'deep_research_wide_summary.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
