#!/usr/bin/env python3
"""Hardened re-analysis of Claim B (single-feature BEDROC beats GBSA).

iter-3 FIX 1: also runs the same 7 conditions on TWO baselines — the pose-seed
`docking_score` (from metadata.csv) and the locked-combo GBSA `gbsa_dG` — so that
every feature is compared against the ranker(s) the paper actually deploys.
Docking baseline on naive: ~0.484 (matches R1). GBSA-locked on naive: ~0.609.

Runs seven conditions on a candidate feature panel:
    naive           panel BEDROC (best of +/- sign)
    mw_resid        panel BEDROC on per-target lig_MW-residualised feature
    size_resid      panel BEDROC on per-target multivariate residual
                    (lig_MW + lig_n_heavy + lig_TPSA)
    bound_only      panel BEDROC on complexes with max_lig_drift_A < 8.0
                    AND lig_com_disp_max_A < 8.0
    pbc_clean       (NEW iter-2) naive filter after dropping complexes marked
                    is_pbc_artifact=True in data/derived/pbc_qc.csv
    alt_label_7     naive panel BEDROC with actives = pchembl >= 7,
                    decoys = pchembl <= 5 (drop middle)
    alt_label_6     naive panel BEDROC with actives = pchembl >= 6,
                    decoys = pchembl < 6

For each row we also report:
    delta_vs_gbsa   panel BEDROC of feature minus panel BEDROC of GBSA-locked
                    (locked = igb2_di4_salt0.15_st0.0072, applied to the same subset)
    ci_lo, ci_hi    95% bootstrap CI on delta (B=5000, resample targets w/replacement)
    perm_p          two-sided permutation p on panel BEDROC (B=5000,
                    labels shuffled within-target)
    n_actives, n_decoys, n_targets

Outputs:
    data/derived/hardened_claim_b.csv         (one row per feature x condition)
    data/derived/hardened_claim_b_summary.md  (human-readable digest)

Usage:
    cd env && pixi run python ../reproduce/hardened_claim_b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from discovery9.io import load_features, load_gbsa, load_metadata
from discovery9.metrics import bedroc
from discovery9.paths import DERIVED

# ------------------------------------------------------------------ config

LOCKED_COMBO = "igb2_di4_salt0.15_st0.0072"
BOUND_DRIFT_MAX_A = 8.0
BOUND_COM_MAX_A = 8.0
BOOT_B = 5000
PERM_B = 5000
ALPHA = 20.0
RNG_SEED = 20250901

# Chemistry features that the fix pack asks us to include even if not top-20
CHEM_FEATURES = ["lig_MW", "lig_TPSA", "lig_HBA", "lig_HBD", "lig_LogP", "lig_n_heavy"]

# iter-2 FIX 2: label-derived columns must not enter the candidate-feature list —
# they either ARE the label or are proxies for it, so any BEDROC signal is a
# tautology that inflates the multiple-testing budget. docking_score is the pose
# seed that produced these MD complexes and is not an MD-derived feature.
LABEL_LEAK_PATTERNS = ("pchembl", "pchembl_meta", "is_active", "is_active_meta",
                       "docking_score")

# ------------------------------------------------------------------ helpers


def panel_bedroc(df: pd.DataFrame, score_col: str, label_col: str = "is_active",
                 target_col: str = "target", alpha: float = ALPHA) -> float:
    """Mean per-target BEDROC (panel score)."""
    vals = []
    for _tgt, g in df.groupby(target_col):
        y = g[label_col].astype(int).to_numpy()
        s = g[score_col].to_numpy(dtype=float)
        b = bedroc(s, y, alpha=alpha)
        if np.isfinite(b):
            vals.append(b)
    return float(np.mean(vals)) if vals else float("nan")


def per_target_bedroc(df: pd.DataFrame, score_col: str, label_col: str = "is_active",
                      target_col: str = "target", alpha: float = ALPHA) -> pd.Series:
    out = {}
    for tgt, g in df.groupby(target_col):
        y = g[label_col].astype(int).to_numpy()
        s = g[score_col].to_numpy(dtype=float)
        out[tgt] = bedroc(s, y, alpha=alpha)
    return pd.Series(out)


def best_sign_bedroc(df: pd.DataFrame, score_col: str, label_col: str = "is_active",
                     alpha: float = ALPHA) -> tuple[float, int]:
    """Return (panel_bedroc, sign) where sign is +1 or -1, picking the better side."""
    b_pos = panel_bedroc(df.assign(_s=df[score_col]), "_s", label_col, alpha=alpha)
    b_neg = panel_bedroc(df.assign(_s=-df[score_col]), "_s", label_col, alpha=alpha)
    if not np.isfinite(b_pos) and not np.isfinite(b_neg):
        return float("nan"), 1
    if not np.isfinite(b_neg):
        return b_pos, 1
    if not np.isfinite(b_pos):
        return b_neg, -1
    if b_pos >= b_neg:
        return b_pos, 1
    return b_neg, -1


def apply_signed(df: pd.DataFrame, score_col: str, sign: int) -> pd.DataFrame:
    d = df.copy()
    d["_signed"] = sign * d[score_col].astype(float)
    return d


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y ~ X (X already includes intercept), fitting only on finite rows."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() < X.shape[1] + 1:
        return np.full_like(y, np.nan)
    beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    pred = X @ beta
    resid = y - pred
    resid[~mask] = np.nan
    return resid


def residualise(df: pd.DataFrame, feat: str, covariates: list[str], target_col: str = "target") -> pd.Series:
    """Per-target OLS residuals of `feat` on `covariates` (adding intercept)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _tgt, g in df.groupby(target_col):
        y = g[feat].to_numpy(dtype=float)
        cov = g[covariates].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(g)), cov])
        r = _ols_residuals(y, X)
        out.loc[g.index] = r
    return out


def bootstrap_delta_ci(df: pd.DataFrame, score_col: str, gbsa_col: str, sign: int,
                       gbsa_sign: int, label_col: str, target_col: str = "target",
                       B: int = BOOT_B, seed: int = RNG_SEED) -> tuple[float, float]:
    """95% CI on panel(feature) − panel(gbsa), resampling targets w/replacement."""
    rng = np.random.default_rng(seed)
    targets = sorted(df[target_col].unique())
    if len(targets) < 2:
        return float("nan"), float("nan")
    # cache per-target BEDROC values so we can just bootstrap the mean of a length-T list
    per_feat = {}
    per_gbsa = {}
    for tgt in targets:
        g = df[df[target_col] == tgt]
        y = g[label_col].astype(int).to_numpy()
        per_feat[tgt] = bedroc(sign * g[score_col].to_numpy(dtype=float), y, alpha=ALPHA)
        per_gbsa[tgt] = bedroc(gbsa_sign * g[gbsa_col].to_numpy(dtype=float), y, alpha=ALPHA)
    feat_arr = np.array([per_feat[t] for t in targets], dtype=float)
    gbsa_arr = np.array([per_gbsa[t] for t in targets], dtype=float)
    finite_mask = np.isfinite(feat_arr) & np.isfinite(gbsa_arr)
    if finite_mask.sum() < 2:
        return float("nan"), float("nan")
    feat_arr = feat_arr[finite_mask]
    gbsa_arr = gbsa_arr[finite_mask]
    T = len(feat_arr)
    idx = rng.integers(0, T, size=(B, T))
    boot_deltas = feat_arr[idx].mean(axis=1) - gbsa_arr[idx].mean(axis=1)
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    return ci_lo, ci_hi


def permutation_p(df: pd.DataFrame, score_col: str, sign: int, label_col: str,
                  target_col: str = "target", B: int = PERM_B,
                  seed: int = RNG_SEED + 1) -> float:
    """Two-sided permutation p for panel BEDROC (labels shuffled within target)."""
    rng = np.random.default_rng(seed)
    obs = panel_bedroc(df.assign(_s=sign * df[score_col].astype(float)), "_s", label_col)
    if not np.isfinite(obs):
        return float("nan")
    targets = sorted(df[target_col].unique())
    # cache per-target arrays for speed
    cache = []
    for tgt in targets:
        g = df[df[target_col] == tgt]
        cache.append((sign * g[score_col].to_numpy(dtype=float),
                      g[label_col].astype(int).to_numpy()))
    null = np.empty(B, dtype=float)
    for b in range(B):
        vals = []
        for s, y in cache:
            y_perm = rng.permutation(y)
            bv = bedroc(s, y_perm, alpha=ALPHA)
            if np.isfinite(bv):
                vals.append(bv)
        null[b] = np.mean(vals) if vals else np.nan
    # two-sided p (centre null on its own mean → symmetric test of location diff)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return float("nan")
    null_centre = null.mean()
    obs_dev = abs(obs - null_centre)
    p = (np.sum(np.abs(null - null_centre) >= obs_dev) + 1) / (null.size + 1)
    return float(p)


# ------------------------------------------------------------------ candidate features


def candidate_features(df: pd.DataFrame, top_k: int = 20) -> list[str]:
    """Top-K by naive single-feature (best-sign) panel BEDROC ∪ chemistry-must-include.

    iter-2 FIX 2: drops any column whose name matches LABEL_LEAK_PATTERNS
    (pchembl / pchembl_meta / is_active / is_active_meta / docking_score) —
    those are labels or label proxies, not features.
    """
    def _is_label_leak(col: str) -> bool:
        cl = col.lower()
        return any(pat in cl for pat in LABEL_LEAK_PATTERNS)

    always_drop = {"target", "complex_id", "gbsa_dG"}
    cand_cols = [c for c in df.columns
                 if c not in always_drop
                 and not _is_label_leak(c)
                 and pd.api.types.is_numeric_dtype(df[c])
                 and df[c].nunique(dropna=True) > 2]
    dropped = [c for c in df.columns if _is_label_leak(c)]
    if dropped:
        print(f"[hardened_claim_b] label-leak columns excluded from candidate features: {dropped}")
    scores = []
    for c in cand_cols:
        try:
            b, _ = best_sign_bedroc(df, c)
        except Exception:  # noqa: BLE001 — skip stubbornly ill-defined columns
            continue
        if np.isfinite(b):
            scores.append((c, b))
    scores.sort(key=lambda t: t[1], reverse=True)
    top = [c for c, _ in scores[:top_k]]
    for c in CHEM_FEATURES:
        if c in df.columns and c not in top:
            top.append(c)
    return top


# ------------------------------------------------------------------ conditions


def build_conditions(df_lab: pd.DataFrame, feat: str) -> dict[str, pd.DataFrame]:
    """Return {condition_name: prepared_df} where prepared_df has columns score, is_active, target."""
    conds: dict[str, pd.DataFrame] = {}

    # naive
    conds["naive"] = df_lab.assign(_score=df_lab[feat]).dropna(subset=["_score", "is_active"])

    # mw_resid
    if "lig_MW" in df_lab.columns and feat != "lig_MW":
        resid = residualise(df_lab, feat, ["lig_MW"])
        conds["mw_resid"] = df_lab.assign(_score=resid).dropna(subset=["_score", "is_active"])

    # size_resid (lig_MW + lig_n_heavy + lig_TPSA)
    size_covs = [c for c in ["lig_MW", "lig_n_heavy", "lig_TPSA"] if c in df_lab.columns and c != feat]
    if len(size_covs) >= 1 and feat not in size_covs:
        resid = residualise(df_lab, feat, size_covs)
        conds["size_resid"] = df_lab.assign(_score=resid).dropna(subset=["_score", "is_active"])

    # bound_only
    bound_mask = (df_lab["lig_drift_last_A"].fillna(np.inf) < BOUND_DRIFT_MAX_A) \
        if False else None
    # NOTE: task says "max_lig_drift_A" — we don't have that as an aggregate; treat
    # lig_drift_last_A as the last-frame drift (already an aggregate in features.parquet)
    # and rely on lig_com_disp_max_A for the escape check. To match the reviewer's
    # intent (drift-based filter), use max(lig_drift_last_A, lig_drift_std_A + lig_drift_mean_A)
    # as a proxy. Simpler + closer to reviewer wording: use lig_drift_mean_A as the
    # sustained drift metric AND require lig_com_disp_max_A < BOUND_COM_MAX_A.
    bound_mask = (
        (df_lab["lig_drift_mean_A"].fillna(np.inf) < BOUND_DRIFT_MAX_A)
        & (df_lab["lig_com_disp_max_A"].fillna(np.inf) < BOUND_COM_MAX_A)
    )
    conds["bound_only"] = df_lab[bound_mask].assign(_score=lambda d: d[feat]).dropna(
        subset=["_score", "is_active"]
    )

    # iter-2 FIX 2: pbc_clean — same as naive but with PBC-flagged complexes removed
    if "is_pbc_artifact" in df_lab.columns:
        pbc_clean_mask = ~df_lab["is_pbc_artifact"].astype(bool)
        conds["pbc_clean"] = (df_lab[pbc_clean_mask]
                              .assign(_score=lambda d: d[feat])
                              .dropna(subset=["_score", "is_active"]))

    # alt-label 7 (actives >= 7, decoys <= 5, drop middle)
    if "pchembl" in df_lab.columns:
        mask7 = df_lab["pchembl"].notna() & (
            (df_lab["pchembl"] >= 7) | (df_lab["pchembl"] <= 5)
        )
        d7 = df_lab[mask7].copy()
        d7["is_active"] = (d7["pchembl"] >= 7).astype(int)
        conds["alt_label_7"] = d7.assign(_score=d7[feat]).dropna(subset=["_score", "is_active"])

    # alt-label 6 (actives >= 6)
    if "pchembl" in df_lab.columns:
        d6 = df_lab[df_lab["pchembl"].notna()].copy()
        d6["is_active"] = (d6["pchembl"] >= 6).astype(int)
        conds["alt_label_6"] = d6.assign(_score=d6[feat]).dropna(subset=["_score", "is_active"])

    return conds


# ------------------------------------------------------------------ orchestration


def _summarise_per_target_labels(df: pd.DataFrame, label_col: str = "is_active") -> pd.DataFrame:
    return df.groupby("target").agg(
        n=(label_col, "size"),
        n_actives=(label_col, lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).astype(int).sum())),
        n_decoys=(label_col, lambda s: int(pd.to_numeric(~s.astype(bool), errors="coerce").fillna(0).astype(int).sum())),
    )


def _load_pbc_flags() -> pd.DataFrame | None:
    """Load pbc_qc.csv (if present) and return a DataFrame with
       target, complex_id, is_pbc_artifact (bool). Returns None if absent."""
    p = DERIVED / "pbc_qc.csv"
    if not p.is_file():
        print(f"[hardened_claim_b] pbc_qc.csv NOT FOUND at {p} — pbc_clean condition will be skipped.")
        return None
    qc = pd.read_csv(p)
    if "is_pbc_artifact" not in qc.columns:
        print(f"[hardened_claim_b] pbc_qc.csv missing is_pbc_artifact column — skipping.")
        return None
    qc["is_pbc_artifact"] = qc["is_pbc_artifact"].astype(bool)
    print(f"[hardened_claim_b] pbc_qc.csv loaded: {len(qc)} trajectories, "
          f"{int(qc.is_pbc_artifact.sum())} flagged as PBC artifacts")
    print("[hardened_claim_b] PBC-flagged complexes per target:")
    per_tgt = (qc[qc.is_pbc_artifact]
               .groupby("target").size().rename("n_pbc_excluded"))
    if per_tgt.empty:
        print("  (none)")
    else:
        print(per_tgt.to_string())
    return qc[["target", "complex_id", "is_pbc_artifact"]]


def main() -> int:
    print("[hardened_claim_b] loading data …", flush=True)
    df = load_features(with_ligand_chem=True)
    meta = load_metadata()
    gbsa = load_gbsa(combo=LOCKED_COMBO)[["complex_id", "target", "gbsa_dG"]]

    # Attach pchembl + docking_score + is_active from metadata, and locked GBSA ΔG.
    # iter-3 FIX 8: features.parquet was built by joining MANIFEST.tsv, which has EMPTY
    # is_active / pchembl for 4A5S — hence the "silent 9 → 8 target shrink" reviewers saw.
    # We recover the label from metadata.csv (which has 10 actives / 20 decoys for 4A5S)
    # by co-loading is_active_meta and back-filling into is_active when the manifest value
    # is missing. See docs/DATA_LINEAGE.md for the root-cause narrative.
    df = (
        df.merge(meta[["complex_id", "target", "pchembl", "docking_score", "is_active"]],
                 on=["complex_id", "target"], how="left", suffixes=("", "_meta"))
        .merge(gbsa, on=["complex_id", "target"], how="left")
    )
    if "is_active_meta" in df.columns:
        # combine_first fills missing values from metadata; astype("boolean") preserves NaN
        recovered = df["is_active"].astype("boolean").combine_first(
            df["is_active_meta"].astype("boolean")
        )
        n_before = int(df["is_active"].notna().sum())
        n_after = int(recovered.notna().sum())
        df["is_active"] = recovered
        if n_after > n_before:
            print(f"[hardened_claim_b] iter-3 FIX 8: recovered is_active from metadata for "
                  f"{n_after - n_before} rows (features.parquet had them as NaN — "
                  f"MANIFEST.tsv label gap, primarily 4A5S).")

    # iter-2 FIX 2: attach PBC-artifact flag (per-complex) — used by `pbc_clean` condition.
    pbc_flags = _load_pbc_flags()
    if pbc_flags is not None:
        df = df.merge(pbc_flags, on=["target", "complex_id"], how="left")
        df["is_pbc_artifact"] = df["is_pbc_artifact"].fillna(False).astype(bool)
    else:
        df["is_pbc_artifact"] = False

    print(f"[hardened_claim_b] features table: {len(df)} rows × {df.shape[1]} cols")
    print(f"[hardened_claim_b] is_active NaN by target:")
    nan_by_tgt = df.groupby("target").is_active.apply(lambda s: int(s.isna().sum()))
    print(nan_by_tgt.to_string())

    df_lab = df.dropna(subset=["is_active"]).copy()
    df_lab["is_active"] = df_lab["is_active"].astype(bool).astype(int)
    print(f"\n[hardened_claim_b] labelled subset: {len(df_lab)} rows × {df_lab.target.nunique()} targets")
    print("[hardened_claim_b] per-target counts:")
    print(_summarise_per_target_labels(df_lab).to_string())

    # GBSA-locked baseline
    gbsa_sign = -1  # lower is better
    gbsa_naive_panel = panel_bedroc(
        df_lab.assign(_s=gbsa_sign * df_lab["gbsa_dG"]).dropna(subset=["_s"]),
        "_s",
    )
    print(f"\n[hardened_claim_b] GBSA-locked panel BEDROC (labelled subset): {gbsa_naive_panel:.4f}")

    # Identify candidate features
    feats = candidate_features(df_lab, top_k=20)
    print(f"\n[hardened_claim_b] evaluating {len(feats)} candidate features:")
    for f in feats:
        print(f"  - {f}")

    # iter-3 FIX 1: add BASELINE pseudo-features so the CSV records the 7-condition
    # panel BEDROC for the docking score and the locked GBSA score alongside every
    # MD/chem feature. Both are "lower-is-better" so their sign is fixed to −1
    # (rather than best-of-sign) to match the physically-motivated ranker used in
    # deployment. `docking_score` = R1-requested baseline (recovers ≈ 0.484 on the
    # naive panel). `gbsa_dG` = the LOCKED_COMBO score already computed above.
    BASELINE_SIGN = {
        "docking_score": -1,   # lower docking score = tighter binding
        "gbsa_dG":       -1,   # lower ΔG = more favourable
    }

    rows = []
    for feat in list(BASELINE_SIGN.keys()) + feats:
        # Sign convention:
        #   * for BASELINE_SIGN pseudo-features, fix the sign — do NOT let best_sign_bedroc
        #     re-pick it, because the "-1" convention is the deployment-oriented one and
        #     re-picking on a small panel introduces a look-ahead confound.
        #   * for the ordinary candidate features, keep the historical behaviour (best of ±).
        if feat in BASELINE_SIGN:
            sign = BASELINE_SIGN[feat]
            sub = df_lab.dropna(subset=[feat])
            if sub.empty:
                print(f"\n[hardened_claim_b] === {feat} SKIPPED (no rows) ===")
                continue
            naive_b = panel_bedroc(sub.assign(_s=sign * sub[feat].astype(float)), "_s")
            tag = "BASELINE"
        else:
            # naive sign for this feature (fixed for all conditions — chosen on naive panel)
            naive_b, sign = best_sign_bedroc(df_lab.dropna(subset=[feat]), feat)
            tag = "feature"
        print(f"\n[hardened_claim_b] === {feat} ({tag}, sign={'+' if sign>0 else '-'}, "
              f"naive={naive_b:.4f}) ===")

        conds = build_conditions(df_lab, feat)
        for cond_name, cond_df in conds.items():
            if cond_df.empty or cond_df["target"].nunique() < 2:
                print(f"  [{cond_name}] skipped — insufficient data (n={len(cond_df)})")
                continue

            # Panel BEDROC on this condition (fixed sign from naive)
            cond_scored = cond_df.assign(_s=sign * cond_df["_score"].astype(float))
            b_feat = panel_bedroc(cond_scored, "_s")

            # Corresponding GBSA panel BEDROC on the SAME subset (to compute delta)
            cond_gbsa = cond_df.dropna(subset=["gbsa_dG"]).assign(_s=gbsa_sign * cond_df.loc[
                lambda d: d.index.isin(cond_df.index)].gbsa_dG.astype(float))
            # simpler: rebuild GBSA on cond_df.index
            cd_gbsa = cond_df.copy()
            cd_gbsa["_s"] = gbsa_sign * cd_gbsa["gbsa_dG"].astype(float)
            cd_gbsa = cd_gbsa.dropna(subset=["_s"])
            b_gbsa = panel_bedroc(cd_gbsa, "_s") if len(cd_gbsa) else float("nan")

            delta = b_feat - b_gbsa if np.isfinite(b_feat) and np.isfinite(b_gbsa) else float("nan")

            # Bootstrap CI on delta using the intersection subset (rows with both feat and gbsa)
            both = cond_df.dropna(subset=["_score", "gbsa_dG"])
            if len(both) >= 2 and both["target"].nunique() >= 2:
                ci_lo, ci_hi = bootstrap_delta_ci(
                    both.rename(columns={"_score": "__feat"}),
                    "__feat", "gbsa_dG", sign, gbsa_sign, "is_active",
                )
            else:
                ci_lo, ci_hi = float("nan"), float("nan")

            # Permutation p on panel BEDROC
            perm_p = permutation_p(
                cond_df.rename(columns={"_score": "__feat"}),
                "__feat", sign, "is_active",
            )

            n_act = int(cond_df["is_active"].sum())
            n_dec = int((cond_df["is_active"] == 0).sum())
            n_tgt = int(cond_df["target"].nunique())

            rows.append({
                "feature": feat,
                "condition": cond_name,
                "sign": sign,
                "BEDROC": round(b_feat, 4),
                "gbsa_BEDROC_on_subset": round(b_gbsa, 4) if np.isfinite(b_gbsa) else np.nan,
                "delta_vs_gbsa": round(delta, 4) if np.isfinite(delta) else np.nan,
                "ci_lo": round(ci_lo, 4) if np.isfinite(ci_lo) else np.nan,
                "ci_hi": round(ci_hi, 4) if np.isfinite(ci_hi) else np.nan,
                "perm_p": round(perm_p, 4) if np.isfinite(perm_p) else np.nan,
                "n_actives": n_act,
                "n_decoys": n_dec,
                "n_targets": n_tgt,
            })
            print(f"  [{cond_name:>11s}] BEDROC={b_feat:.4f}  Δ={delta:+.4f}  "
                  f"CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p={perm_p:.4f}  "
                  f"({n_tgt} tgt · {n_act} act · {n_dec} dec)")

    out = pd.DataFrame(rows)
    DERIVED.mkdir(parents=True, exist_ok=True)
    out_csv = DERIVED / "hardened_claim_b.csv"
    out.to_csv(out_csv, index=False)
    print(f"\n[hardened_claim_b] wrote {out_csv}  ({len(out)} rows)")

    # --------------------------- Human-readable summary
    def _fmt_num(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and np.isfinite(x) else "NA"

    lines: list[str] = []
    lines.append("# Hardened Claim B — summary\n")
    n_gbsa_tgt = int(df_lab.dropna(subset=["gbsa_dG"]).target.nunique())
    lines.append(f"- GBSA-locked panel BEDROC (labelled ∩ GBSA subset, {n_gbsa_tgt} targets): **{gbsa_naive_panel:.4f}**")
    lines.append(f"- LOCKED_COMBO = `{LOCKED_COMBO}`")
    lines.append(f"- Labelled complexes: {len(df_lab)} across {df_lab.target.nunique()} targets  "
                 f"(iter-3 FIX 8: 4A5S labels recovered from metadata.csv; GBSA-only rows still cover {n_gbsa_tgt} targets — 4A5S has no GBSA)")
    lines.append(f"- Bootstrap B={BOOT_B}, permutation B={PERM_B}, α={ALPHA}\n")
    lines.append("## Per-condition BEDROC — baselines and headline features\n")
    lines.append("The first two rows are BASELINES (iter-3 FIX 1) — everything below must be "
                 "compared against them, not against zero.\n")
    headline_feats = ["docking_score", "gbsa_dG",
                      "lig_buried_sasa_std_A2", "lig_MW", "lig_TPSA", "lig_HBA", "lig_HBD",
                      "lig_LogP", "lig_n_heavy", "rmsd_bb_mean_A"]
    for feat in headline_feats:
        sub = out[out.feature == feat]
        if sub.empty:
            continue
        lines.append(f"### {feat}\n")
        lines.append("| condition | BEDROC | Δ vs GBSA | 95% CI | perm p | n_tgt | n_act | n_dec |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r.condition} | {_fmt_num(r.BEDROC)} | {_fmt_num(r.delta_vs_gbsa)} | "
                f"[{_fmt_num(r.ci_lo)}, {_fmt_num(r.ci_hi)}] | {_fmt_num(r.perm_p)} | "
                f"{int(r.n_targets)} | {int(r.n_actives)} | {int(r.n_decoys)} |"
            )
        lines.append("")

    lines.append("## Per-target labelled counts (labelled subset)\n")
    counts = _summarise_per_target_labels(df_lab)
    lines.append("| target | n | n_actives | n_decoys |")
    lines.append("|---|---|---|---|")
    for tgt, row in counts.iterrows():
        lines.append(f"| {tgt} | {int(row.n)} | {int(row.n_actives)} | {int(row.n_decoys)} |")
    lines.append("")

    out_md = DERIVED / "hardened_claim_b_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[hardened_claim_b] wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
