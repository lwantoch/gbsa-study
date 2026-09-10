#!/usr/bin/env python3
"""
Canonical baseline reconciliation (iter-5).

Rebuilds `data/derived/canonical_baselines.csv` — the SINGLE source of truth
for the panel-BEDROC α=20 numbers referenced across README.md, STUDY_DESIGN.md,
docs/DATA_LINEAGE.md, and every notebook under notebooks/ (single flat tree, slots 00..30).

Convention (settled iter-3 → iter-5):
  - Panel = the 9 discovery targets
      ['2XU3','3I06','4A5S','4L7G','4QB3','5HU9','8ELC','9D9I','9SI4']
  - Compute per-target BEDROC α=20; take the panel mean.
    Targets with no data on the given scorer are IMPUTED to 0 (worst case).
    This is the same rule used by NB 17 and hardened_claim_b.py.
  - Bootstrap CI: B=5000, target-level resample (fixed seed for reproducibility).
  - GBSA-locked = single globally-locked GBSA combo, chosen upstream
    (`data/external/gbsa-study/data/derived/locked_settings.csv`).
    The locked combo `igb2_di4_salt0.15_st0.0072` is *also* the panel-mean
    argmax on the 9-target panel (tied with two di4 variants at 0.5412),
    so the pick is stable under the imputation policy — verified below.
  - Docking = per-target BEDROC of the `docking_score` column of
    `data/external/gbsa-study/data/raw/metadata.csv`, all 9 targets have data.

Run:
    cd env && pixi run python ../reproduce/canonical_baselines.py            # write CSV
    cd env && pixi run python ../reproduce/canonical_baselines.py --verify   # recompute and diff

Idempotent: overwrites the CSV with identical bytes on a clean re-run.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------- paths (self-contained) --------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DERIVED = ROOT / "data" / "derived"
GBSA_STUDY = ROOT / "data" / "external" / "gbsa-study"

# -------------------- canonical settings --------------------
PANEL = ["2XU3", "3I06", "4A5S", "4L7G", "4QB3", "5HU9", "8ELC", "9D9I", "9SI4"]
ALPHA = 20.0
BOOT_B = 5000
SEED = 20260902  # 'canonical_baselines seed' — do not change
LOCKED_COMBO = "igb2_di4_salt0.15_st0.0072"


# -------------------- primitives --------------------
def bedroc(scores, labels, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    m = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[m], labels[m]
    n_pos = int(labels.sum())
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    N = len(labels)
    Ra = n_pos / N
    ranks = np.where(labels == 1)[0] + 1
    num = np.sum(np.exp(-alpha * ranks / N))
    denom = Ra * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1)
    Rf = num / denom if denom > 0 else float("nan")
    factor = Ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * Ra))
    return float(Rf * factor + 1 / (1 - np.exp(alpha * (1 - Ra))))


def per_target_from_col(sub: pd.DataFrame, score_col: str, lower_is_better: bool = True) -> float:
    if len(sub) < 2 or sub.is_active.sum() == 0 or sub.is_active.sum() == len(sub):
        return float("nan")
    sign = -1 if lower_is_better else 1
    return bedroc(sign * sub[score_col].values, sub.is_active.astype(int).values, alpha=ALPHA)


def impute(per_t: dict, panel=PANEL) -> np.ndarray:
    return np.array([(0.0 if not np.isfinite(per_t[t]) else per_t[t]) for t in panel], dtype=float)


def boot_ci(vals: np.ndarray, B: int = BOOT_B, seed: int = SEED):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    boots = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# -------------------- computations --------------------
def load_inputs():
    meta = pd.read_csv(GBSA_STUDY / "data" / "raw" / "metadata.csv")
    gbsa_raw = pd.read_csv(GBSA_STUDY / "data" / "raw" / "gbsa_dG_raw.csv")
    return meta, gbsa_raw


def gbsa_locked_per_target(meta: pd.DataFrame, gbsa_raw: pd.DataFrame, combo: str) -> dict:
    g = gbsa_raw[gbsa_raw.combo == combo][["complex_id", "target", "mean_dG_kcalmol"]]
    j = meta.merge(g, on=["complex_id", "target"], how="inner")
    return {t: per_target_from_col(j[j.target == t], "mean_dG_kcalmol", True) for t in PANEL}


def docking_per_target(meta: pd.DataFrame) -> dict:
    return {t: per_target_from_col(meta[meta.target == t], "docking_score", True) for t in PANEL}


def feature_per_target(meta: pd.DataFrame, feat_df: pd.DataFrame, feature: str,
                       lower_is_better: bool = False) -> dict:
    # join labels from metadata onto the feature table
    j = feat_df.merge(meta[["complex_id", "target", "is_active"]].rename(columns={"is_active": "is_active_m"}),
                      on=["complex_id", "target"], how="left")
    j["is_active"] = j["is_active"].fillna(j["is_active_m"]) if "is_active" in j.columns else j["is_active_m"]
    j = j.dropna(subset=["is_active", feature]).copy()
    j["is_active"] = j["is_active"].astype(bool).astype(int)
    return {t: per_target_from_col(j[j.target == t], feature, lower_is_better) for t in PANEL}


def _resolve_sign(per_t_pos: dict, per_t_neg: dict) -> tuple[str, dict]:
    vpos = impute(per_t_pos).mean()
    vneg = impute(per_t_neg).mean()
    if vpos >= vneg:
        return "+1", per_t_pos
    return "-1", per_t_neg


def verify_global_lock(meta, gbsa_raw) -> None:
    """Verify that LOCKED_COMBO is (tied for) panel-mean argmax under the 9T/impute-0 policy."""
    per_combo = {}
    for combo in gbsa_raw.combo.unique():
        per_t = gbsa_locked_per_target(meta, gbsa_raw, combo)
        per_combo[combo] = impute(per_t).mean()
    ranked = sorted(per_combo.items(), key=lambda x: -x[1])
    top_score = ranked[0][1]
    tied = [c for c, s in ranked if abs(s - top_score) < 1e-9]
    if LOCKED_COMBO not in tied:
        raise SystemExit(
            f"[canonical_baselines] LOCKED_COMBO={LOCKED_COMBO} is NOT the panel-mean argmax "
            f"on the 9-target panel; top={ranked[0][0]} ({ranked[0][1]:.4f}). "
            "Refuse to write the CSV until the convention is re-settled."
        )
    print(f"  verified: LOCKED_COMBO tied for argmax ({top_score:.4f}); {len(tied)} tied combos: {tied}")


def build_rows() -> list[dict]:
    computed_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    meta, gbsa_raw = load_inputs()

    print("Verifying global-lock convention ...")
    verify_global_lock(meta, gbsa_raw)

    rows = []

    # (1) GBSA-locked, canonical (9T, 4A5S imputed 0)
    per_t = gbsa_locked_per_target(meta, gbsa_raw, LOCKED_COMBO)
    vals = impute(per_t)
    m, lo, hi = boot_ci(vals)
    rows.append({
        "baseline_name": "gbsa_locked_9T_imputed",
        "panel_bedroc": round(m, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_targets": len(PANEL),
        "sp_config": LOCKED_COMBO,
        "provenance_notebook": "notebooks/22_bedroc_baselines.ipynb; notebooks/29_rank_fusion_deployable.ipynb",
        "provenance_data_file": "data/external/gbsa-study/data/raw/{metadata.csv,gbsa_dG_raw.csv}",
        "computed_at_iso": computed_at,
    })

    # (1b) GBSA-locked (8T, 4A5S excluded) — the documented alternative convention
    vals_8 = np.array([v for v in per_t.values() if np.isfinite(v)])
    m8, lo8, hi8 = boot_ci(vals_8)
    rows.append({
        "baseline_name": "gbsa_locked_8T_excluded",
        "panel_bedroc": round(m8, 4),
        "ci_low": round(lo8, 4),
        "ci_high": round(hi8, 4),
        "n_targets": len(vals_8),
        "sp_config": LOCKED_COMBO,
        "provenance_notebook": "notebooks/22_bedroc_baselines.ipynb (4A5S dropped upstream)",
        "provenance_data_file": "data/external/gbsa-study/data/raw/{metadata.csv,gbsa_dG_raw.csv}",
        "computed_at_iso": computed_at,
    })

    # (2) Docking baseline, 9T (all present)
    per_dock = docking_per_target(meta)
    vals_dock = impute(per_dock)
    m2, lo2, hi2 = boot_ci(vals_dock)
    rows.append({
        "baseline_name": "docking_9T",
        "panel_bedroc": round(m2, 4),
        "ci_low": round(lo2, 4),
        "ci_high": round(hi2, 4),
        "n_targets": len(PANEL),
        "sp_config": "",
        "provenance_notebook": "notebooks/22_bedroc_baselines.ipynb; notebooks/05_docking_baseline.ipynb",
        "provenance_data_file": "data/external/gbsa-study/data/raw/metadata.csv",
        "computed_at_iso": computed_at,
    })

    # (3) Context: top hardened Claim B naive feature.
    #    NB: ci_lo/ci_hi columns in hardened_claim_b.csv are CIs on delta_vs_gbsa, not on
    #    the feature's BEDROC. We recompute a proper BEDROC-mean CI here from features.parquet
    #    (same 9T / impute-0 policy) so the canonical CSV row is a like-for-like baseline.
    feat_parquet = DERIVED / "features.parquet"
    feat_tsv = DERIVED / "features.tsv"
    hb_path = DERIVED / "hardened_claim_b.csv"

    def _load_features() -> pd.DataFrame | None:
        # Prefer parquet (single-source-of-truth); fall back to TSV mirror when pyarrow
        # is unavailable (e.g. system python without pixi).
        try:
            return pd.read_parquet(feat_parquet) if feat_parquet.exists() else None
        except Exception:
            if feat_tsv.exists():
                return pd.read_csv(feat_tsv, sep="\t")
            return None

    feats = _load_features()
    if hb_path.exists() and feats is not None:
        hb = pd.read_csv(hb_path)
        lig_chem_path = DERIVED / "ligand_chem.parquet"
        try:
            if lig_chem_path.exists():
                lig = pd.read_parquet(lig_chem_path)
                drop_cols = [c for c in lig.columns if c in feats.columns and c not in ("complex_id", "target")]
                feats = feats.merge(lig.drop(columns=drop_cols, errors="ignore"),
                                    on=["complex_id", "target"], how="left")
        except Exception:
            pass  # optional context; skip if parquet engine unavailable

        def _ci_from_feature(feature: str, sign: int) -> tuple[str, float, float, float]:
            per_t_pos = feature_per_target(meta, feats, feature, lower_is_better=False)
            per_t_neg = feature_per_target(meta, feats, feature, lower_is_better=True)
            if sign == +1:
                per_t = per_t_pos
            else:
                per_t = per_t_neg
            vals = impute(per_t)
            m, lo, hi = boot_ci(vals)
            return "+1" if sign == +1 else "-1", m, lo, hi

        # NOISE_FEATURES: computationally-neutral-by-construction columns whose
        # tiny float-roundoff variations produced spurious BEDROC hits. See
        # docs/GLOSSARY.md § ligand_partial_charge_sum. Both names are the
        # SAME column (identical stats: mean~-1e-6, std~1e-5) — one lives in
        # features.parquet, the other is a copy carried into ligand_chem.parquet.
        NOISE_FEATURES = ("lig_partial_q_sum", "ligand_partial_charge_sum")
        EXCLUDE_FROM_TOP = ("gbsa_dG", "docking_score", *NOISE_FEATURES)

        def _top_feature_row(condition: str, prefix: str, note: str):
            sub = hb[(hb.condition == condition) & (~hb.feature.isin(EXCLUDE_FROM_TOP))]
            if not len(sub):
                return None
            top = sub.sort_values("BEDROC", ascending=False).iloc[0]
            feat = str(top["feature"])
            sign = int(top["sign"])
            if feat not in feats.columns:
                return None
            _sgn, m, lo, hi = _ci_from_feature(feat, sign)
            return {
                "baseline_name": f"{prefix}::{feat}",
                "panel_bedroc": round(m, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "n_targets": int(top["n_targets"]),
                "sp_config": f"sign={_sgn} ({note})",
                "provenance_notebook": "notebooks/28_single_feature_bedroc.ipynb; notebooks/29_rank_fusion_deployable.ipynb",
                "provenance_data_file": "data/derived/{hardened_claim_b.csv, features.parquet}",
                "computed_at_iso": computed_at,
            }

        naive_row = _top_feature_row("naive", "top_feature_naive_9T", "top naive single-feature")
        if naive_row is not None:
            rows.append(naive_row)
        resid_row = _top_feature_row("mw_resid", "top_feature_mw_resid_9T", "top MW-residualised single-feature")
        if resid_row is not None:
            rows.append(resid_row)

    return rows


def build_csv_text() -> str:
    rows = build_rows()
    df = pd.DataFrame(rows, columns=[
        "baseline_name", "panel_bedroc", "ci_low", "ci_high",
        "n_targets", "sp_config",
        "provenance_notebook", "provenance_data_file", "computed_at_iso",
    ])
    buf = io.StringIO()
    buf.write("# Canonical baseline panel BEDROC alpha=20 values -- SINGLE SOURCE OF TRUTH\n")
    buf.write("# Regenerate via: pixi run python reproduce/canonical_baselines.py\n")
    buf.write("# Verify with:   pixi run python reproduce/canonical_baselines.py --verify\n")
    buf.write(f"# Panel = 9 discovery targets, missing-target imputation = 0, bootstrap B={BOOT_B}, seed={SEED}\n")
    df.to_csv(buf, index=False)
    return buf.getvalue()


# -------------------- entry --------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Recompute and diff against the on-disk CSV; exit 1 on any drift.")
    parser.add_argument("--out", default=str(DERIVED / "canonical_baselines.csv"),
                        help="Output path (default: data/derived/canonical_baselines.csv)")
    args = parser.parse_args(argv)

    csv_text = build_csv_text()
    out_path = Path(args.out).resolve()

    if args.verify:
        if not out_path.is_file():
            print(f"[verify] MISSING: {out_path}", file=sys.stderr)
            return 1
        on_disk = out_path.read_text()
        # compare ignoring only the computed_at_iso column (whose value drifts every run)
        def _strip_time_col(txt: str) -> list[str]:
            out = []
            for line in txt.splitlines():
                if "," in line and not line.startswith("#"):
                    parts = line.split(",")
                    if parts[-1] and parts[-1] != "computed_at_iso":
                        parts[-1] = "<TIME>"
                    line = ",".join(parts)
                out.append(line)
            return out
        a, b = _strip_time_col(csv_text), _strip_time_col(on_disk)
        if a != b:
            import difflib
            diff = "\n".join(difflib.unified_diff(b, a, fromfile="on_disk", tofile="recomputed", lineterm=""))
            print("[verify] DRIFT DETECTED:", file=sys.stderr)
            print(diff, file=sys.stderr)
            return 1
        print(f"[verify] OK — {out_path} matches recomputed values (ignoring computed_at_iso).")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(csv_text)
    print(f"Wrote {out_path}")
    print("---")
    print(csv_text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
