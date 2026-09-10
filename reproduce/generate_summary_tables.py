#!/usr/bin/env python3
"""
Generate `tables/panel_bedroc_summary.csv` and `tables/rank_fusion_sweep.csv`.

Both tables consolidate panel-level early-enrichment numbers referenced by
README.md. Sources:

  panel_bedroc_summary.csv
    - Deep-research ML sweep       -> data/derived/deep_research_wide.csv
    - Hardened single-feature +
      docking + GBSA-locked        -> data/derived/hardened_claim_b.csv
                                      (+ deep_research_summary.csv for the
                                       docking-baseline number if missing)

  rank_fusion_sweep.csv
    - LOTO K=1..K=max Borda rank fusion of the top single features from
      `data/derived/single_feature_bedroc.csv`; regenerated in-place by this
      script since NB 17 stores its sweep only in cell output.

Run:
    cd env && pixi run python ../reproduce/generate_summary_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ---------- paths (self-contained; no discovery9 import required) ----------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DERIVED = ROOT / "data" / "derived"
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

GBSA_STUDY = ROOT / "data" / "external" / "gbsa-study"


# ---------- helpers ----------

def bedroc(scores, labels, alpha: float = 20.0) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    m = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[m], labels[m]
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return np.nan
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    N = len(labels)
    Ra = n_pos / N
    ranks = np.where(labels == 1)[0] + 1
    num = np.sum(np.exp(-alpha * ranks / N))
    denom = Ra * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1)
    Rf = num / denom if denom > 0 else np.nan
    factor = Ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra))
    return Rf * factor + 1 / (1 - np.exp(alpha * (1 - Ra)))


# ---------- panel_bedroc_summary.csv ----------

def build_panel_bedroc_summary() -> pd.DataFrame:
    rows = []

    # ML rows from deep_research_wide.csv (naive condition, per-model)
    wide = pd.read_csv(DERIVED / "deep_research_wide.csv")
    for _, r in wide.iterrows():
        rows.append({
            "score": f"{r['protocol']}::{r['model']}",
            "condition": "naive",
            "panel_bedroc": float(r["panel_bedroc"]),
            "ci_lo": float(r["ci_lo"]) if pd.notna(r["ci_lo"]) else np.nan,
            "ci_hi": float(r["ci_hi"]) if pd.notna(r["ci_hi"]) else np.nan,
            "perm_p": float(r["perm_p"]) if pd.notna(r["perm_p"]) else np.nan,
            "n_targets": int(r["n_targets"]),
        })

    # Single-feature (hardened, all seven conditions)
    hard = pd.read_csv(DERIVED / "hardened_claim_b.csv")
    for _, r in hard.iterrows():
        rows.append({
            "score": f"feat::{r['feature']}",
            "condition": r["condition"],
            "panel_bedroc": float(r["BEDROC"]),
            "ci_lo": float(r["ci_lo"]) if pd.notna(r["ci_lo"]) else np.nan,
            "ci_hi": float(r["ci_hi"]) if pd.notna(r["ci_hi"]) else np.nan,
            "perm_p": float(r["perm_p"]) if pd.notna(r["perm_p"]) else np.nan,
            "n_targets": int(r["n_targets"]),
        })

    # Docking + GBSA-locked + MD-composite from deep_research_summary.csv (LEGACY 8T CONVENTION)
    # Values here use the 8-target-subset convention from `reproduce/deep_research.py` --
    # kept for NB 13's cross-notebook comparison table. Canonical 9-target values are
    # appended below from `data/derived/canonical_baselines.csv`.
    summ = pd.read_csv(DERIVED / "deep_research_summary.csv")
    label_map = {
        "docking_score baseline": "baseline::docking_8T_legacy",
        "GBSA @ locked combo baseline": "baseline::gbsa_locked_8T_legacy",
        "MD-composite": "baseline::md_composite",
        "LOCKED sp-config (sp08)": "baseline::gbsa_sp08",
        "random combo (500 draws)": "baseline::random_combo",
        "ORACLE (per-target argmax, upper bnd)": "baseline::oracle_upper_bound",
    }
    for _, r in summ.iterrows():
        approach = r["approach"]
        if approach in label_map:
            rows.append({
                "score": label_map[approach],
                "condition": "naive",
                "panel_bedroc": float(r["panel_bedroc"]),
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "perm_p": np.nan,
                "n_targets": 8,
            })

    # Canonical baselines (single source of truth) — see reproduce/canonical_baselines.py
    canonical_path = DERIVED / "canonical_baselines.csv"
    if canonical_path.exists():
        canon = pd.read_csv(canonical_path, comment="#")
        for _, r in canon.iterrows():
            rows.append({
                "score": f"canonical::{r['baseline_name']}",
                "condition": "naive",
                "panel_bedroc": float(r["panel_bedroc"]),
                "ci_lo": float(r["ci_low"]) if pd.notna(r["ci_low"]) and r["ci_low"] != "" else np.nan,
                "ci_hi": float(r["ci_high"]) if pd.notna(r["ci_high"]) and r["ci_high"] != "" else np.nan,
                "perm_p": np.nan,
                "n_targets": int(r["n_targets"]),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["condition", "panel_bedroc"], ascending=[True, False]).reset_index(drop=True)
    return df


# ---------- rank_fusion_sweep.csv ----------

def build_rank_fusion_sweep() -> pd.DataFrame:
    """
    Reproduce the NB 17 K=1..K_max LOTO Borda rank-fusion sweep.

    We use the same feature set NB 17 uses (MD + ligand-chem), pick the best
    single feature per LOTO fold on the training targets, then average the
    top-K feature ranks on the held-out target.
    """
    features = pd.read_parquet(DERIVED / "features.parquet")
    lig_chem_path = DERIVED / "ligand_chem.parquet"
    if lig_chem_path.exists():
        lig = pd.read_parquet(lig_chem_path)
        # only bring columns not already present
        drop_cols = [c for c in lig.columns if c in features.columns and c not in ("complex_id", "target")]
        lig = lig.drop(columns=drop_cols, errors="ignore")
        features = features.merge(lig, on=["complex_id", "target"], how="left")

    meta = pd.read_csv(f"{GBSA_STUDY}/data/raw/metadata.csv")
    _noconflict = features.drop(columns=["is_active", "pchembl"], errors="ignore")
    full = _noconflict.merge(
        meta[["complex_id", "target", "is_active", "pchembl", "docking_score"]],
        on=["complex_id", "target"], how="left",
    )
    # iter-4 FIX: include 4A5S — its is_active labels are recovered in metadata.csv
    # (10 actives / 20 decoys); dropping it hides that the "single-feature always
    # wins" story is fragile on the bromodomain-adjacent 4A5S subpanel.
    full = full.dropna(subset=["is_active"])
    targets = sorted(full.target.unique())

    MD_FEATS = [
        "rmsd_bb_mean_A", "rmsd_bb_std_A", "rmsd_as_bb_mean_A", "rmsd_as_bb_std_A",
        "protein_rg_mean_A", "protein_rg_std_A", "as_ca_rmsf_mean_A", "as_ca_rmsf_max_A",
        "lig_drift_mean_A", "lig_drift_std_A", "lig_drift_last_A",
        "lig_com_disp_mean_A", "lig_com_disp_max_A", "lig_com_disp_last_A", "lig_escape_frac",
        "lig_internal_rmsd_mean_A", "lig_internal_rmsd_std_A", "lig_internal_rmsd_last_A",
        "lig_rmsf_mean_A", "lig_rmsf_max_A", "lig_buried_sasa_mean_A2", "lig_buried_sasa_std_A2",
        "vdw_contacts_mean", "vdw_contacts_std", "n_hb_mean", "n_hb_std", "hb_persistence_frac",
        "salt_bridges_lp_mean", "ifp_tanimoto_median_vs_ref", "ifp_tanimoto_last_vs_ref",
        "ifp_tanimoto_entropy",
        "lig_binding_modes_1A", "lig_binding_modes_2A",
        "lig_orient_autocorr_mean", "lig_orient_autocorr_last",
        "lig_rg_mean_A", "lig_asphericity_mean", "lig_dipole_mean_eA", "lig_dipole_std_eA",
        "coulomb_mean_arb", "coulomb_std_arb",
    ]
    LIG_FEATS = [
        "lig_MW", "lig_n_heavy", "lig_rot_bonds", "lig_HBD", "lig_HBA",
        "lig_all_rings", "lig_LogP", "lig_TPSA", "lig_fraction_sp3",
        "lig_partial_q_abs_sum",
    ]
    ALL_FEATS = [f for f in (MD_FEATS + LIG_FEATS) if f in full.columns]

    def dir_and_panel(w_train, feat):
        per_pos, per_neg = [], []
        for _, g in w_train.groupby("target"):
            per_pos.append(bedroc(+g[feat].values, g.is_active.astype(int).values))
            per_neg.append(bedroc(-g[feat].values, g.is_active.astype(int).values))
        pos_m, neg_m = np.nanmean(per_pos), np.nanmean(per_neg)
        return (+1, pos_m) if pos_m >= neg_m else (-1, neg_m)

    def loto_rank_fusion(K):
        per_t = {}
        picks_by_t = {}
        for t_out in targets:
            w_tr = full[full.target != t_out]
            scored = [(f, *dir_and_panel(w_tr, f)) for f in ALL_FEATS]
            scored = sorted(scored, key=lambda x: -x[2])[:K]
            picks_by_t[t_out] = [(f, d) for f, d, _ in scored]
            g_out = full[full.target == t_out]
            if len(g_out) == 0:
                per_t[t_out] = np.nan
                continue
            rs = np.zeros(len(g_out))
            for f, d, _ in scored:
                rs += rankdata(d * g_out[f].values, method="average")
            rs /= K
            per_t[t_out] = bedroc(rs, g_out.is_active.astype(int).values)
        return per_t, picks_by_t

    # simple percentile bootstrap CI over targets
    def target_bootstrap_ci(per_t_vals, B=2000, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        vals = np.asarray([v for v in per_t_vals if np.isfinite(v)], dtype=float)
        if len(vals) < 2:
            return (np.nan, np.nan)
        boot = np.empty(B, dtype=float)
        idx_pool = np.arange(len(vals))
        for b in range(B):
            samp = rng.choice(idx_pool, size=len(vals), replace=True)
            boot[b] = float(np.mean(vals[samp]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return float(lo), float(hi)

    from collections import Counter
    K_grid = [1, 2, 3, 5, 8, 10, 15, 20, 30]
    rows = []
    for K in K_grid:
        pt, picks = loto_rank_fusion(K)
        vals = [v for v in pt.values() if np.isfinite(v)]
        mean = float(np.nanmean(vals)) if vals else np.nan
        lo, hi = target_bootstrap_ci(list(pt.values()))
        # summarise the feature set picked across folds: most-common top-K
        # concatenation
        all_picks = [f for fold_picks in picks.values() for (f, _) in fold_picks]
        top_picks = [f for f, _ in Counter(all_picks).most_common(K)]
        rows.append({
            "K": K,
            "models": ",".join(top_picks),
            "panel_bedroc": round(mean, 4),
            "ci_lo": round(lo, 4) if np.isfinite(lo) else np.nan,
            "ci_hi": round(hi, 4) if np.isfinite(hi) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------- main ----------

def main():
    print("Building tables/panel_bedroc_summary.csv ...")
    panel = build_panel_bedroc_summary()
    out1 = TABLES / "panel_bedroc_summary.csv"
    with open(out1, "w") as fh:
        fh.write("# Regenerate via: pixi run python reproduce/generate_summary_tables.py\n")
        panel.to_csv(fh, index=False)
    print(f"  wrote {len(panel)} rows -> {out1}")

    print("Building tables/rank_fusion_sweep.csv ...")
    fusion = build_rank_fusion_sweep()
    out2 = TABLES / "rank_fusion_sweep.csv"
    with open(out2, "w") as fh:
        fh.write("# LOTO Borda rank-fusion over top-K single features (NB 17).\n")
        fh.write("# Regenerate via: pixi run python reproduce/generate_summary_tables.py\n")
        fusion.to_csv(fh, index=False)
    print(f"  wrote {len(fusion)} rows -> {out2}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
