"""Load the raw measurement CSVs — one small helper, used by every notebook.

The raw data is the read-only source of truth (``data/raw/``); the derived tables in
``data/derived/`` are recomputed from it by the notebooks. ``load()`` reads a raw CSV by
its stem so notebooks never hardcode a path.

    from gbsabench.io import load
    gbsa = load("gbsa_dG_raw")      # per (complex, combo) mean ΔTOTAL, 48 combos
    meta = load("metadata")         # per complex: target, pKi, is_active, docking_score
"""
from __future__ import annotations

import pandas as pd

from .paths import RAW, DERIVED


def load(stem: str, derived: bool = False) -> pd.DataFrame:
    """Read one CSV by stem from data/raw/ (or data/derived/ if derived=True)."""
    base = DERIVED if derived else RAW
    return pd.read_csv(base / f"{stem}.csv")
