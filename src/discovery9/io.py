"""Loading measurement data — features + GBSA + BEDROC references.

Nothing here is edited — all fitting/aggregation happens in the notebooks. These
helpers just find and concatenate the CSVs, add provenance, and return DataFrames.

    from discovery9.io import load_features, load_gbsa, load_bedroc_matrix
    feat = load_features()                    # per-complex MD features + is_active label
    gbsa = load_gbsa(combo='igb2_di4_salt0.15_st0.0072')  # per-complex GBSA ΔG
    btc  = load_bedroc_matrix()               # per-(target, sp-config) BEDROC α=20
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .paths import DERIVED, RAW, GBSA_STUDY, complex_dir


def load_features(with_ligand_chem: bool = True) -> pd.DataFrame:
    """Wide per-complex feature table (keyed by target, complex_id)."""
    fp = pd.read_parquet(DERIVED / "features.parquet")
    if with_ligand_chem and (DERIVED / "ligand_chem.parquet").exists():
        lig = pd.read_parquet(DERIVED / "ligand_chem.parquet")
        fp = fp.merge(lig, on=["target", "complex_id"], how="left")
    return fp


def load_gbsa(combo: str = "igb2_di4_salt0.15_st0.0072") -> pd.DataFrame:
    """Per-complex GBSA ΔG at one GBSA combo from the upstream study."""
    p = GBSA_STUDY / "data" / "raw" / "gbsa_dG_raw.csv"
    df = pd.read_csv(p).rename(columns={"mean_dG_kcalmol": "gbsa_dG"})
    return df[df.combo == combo][["complex_id", "target", "gbsa_dG"]].reset_index(drop=True)


def load_gbsa_all() -> pd.DataFrame:
    """All combos × all complexes (long-form)."""
    p = GBSA_STUDY / "data" / "raw" / "gbsa_dG_raw.csv"
    return pd.read_csv(p).rename(columns={"mean_dG_kcalmol": "gbsa_dG"})


def load_metadata() -> pd.DataFrame:
    """Per-complex ligand identity + activity label + docking score from upstream study."""
    return pd.read_csv(GBSA_STUDY / "data" / "raw" / "metadata.csv")


def load_bedroc_matrix(alpha: int = 20) -> pd.DataFrame:
    """Per-(target × sp-config) BEDROC α=20 matrix from the temporal study."""
    p = GBSA_STUDY / "data" / "derived" / "study2" / "bedroc20_partial.csv"
    b = pd.read_csv(p)
    return b.pivot(index="target", columns="config", values=f"bedroc{alpha}")


def load_bedroc_all_combos() -> pd.DataFrame:
    """Per-(GBSA combo × target) BEDROC (α=8, 20, 80) — used for the parameter-preference analysis."""
    return pd.read_csv(GBSA_STUDY / "data" / "derived" / "temporal" / "bedroc_all_combos_per_target.csv")


def load_per_complex_analysis(target: str, complex_id: str) -> dict:
    """Read one complex's per-frame outputs (summary.json + timeseries.parquet + contacts_persistence.tsv + …)."""
    d = complex_dir(target, complex_id)
    out = {}
    if (d / "summary.json").exists():
        out["summary"] = json.load(open(d / "summary.json"))
    for stem in ("timeseries", "hbond_timeseries", "ifp_timeseries", "ligand_rmsf", "protein_ca_rmsf"):
        f = d / f"{stem}.parquet"
        if f.exists():
            out[stem] = pd.read_parquet(f)
    for stem in ("contacts_persistence", "hbonds_persistence"):
        f = d / f"{stem}.tsv"
        if f.exists():
            out[stem] = pd.read_csv(f, sep="\t")
    return out
