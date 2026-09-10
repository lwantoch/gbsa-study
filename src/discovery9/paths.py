"""Repository paths — resolved RELATIVE to this file, never hardcoded.

Every notebook and script imports its data/figure/table locations from here, so the
whole project works unchanged after someone clones or moves it.

    from discovery9.paths import RAW, DERIVED, FIGURES, TABLES
    df = pd.read_parquet(DERIVED / "features.parquet")

How the root is found: this file lives at <repo>/src/discovery9/paths.py, so the
repo root is three levels up. We double-check by looking for the marker file
(pyproject.toml) and, as a fallback, walk upwards until we find one.
"""
from __future__ import annotations

from pathlib import Path

_MARKER = "pyproject.toml"


def find_repo_root() -> Path:
    """Return the repository root directory."""
    here = Path(__file__).resolve()
    guess = here.parents[2]  # src/discovery9/paths.py -> src/discovery9 -> src -> repo
    if (guess / _MARKER).is_file():
        return guess
    for parent in here.parents:
        if (parent / _MARKER).is_file():
            return parent
    return guess


ROOT = find_repo_root()

# --- top-level locations (all relative to ROOT) ---
SRC = ROOT / "src"
DATA = ROOT / "data"
NOTEBOOKS = ROOT / "notebooks"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
DOCS = ROOT / "docs"
REPRODUCE = ROOT / "reproduce"

# --- data sub-locations ---
RAW = DATA / "raw"              # per-complex analysis outputs (summary.json, timeseries, ...)
DERIVED = DATA / "derived"      # aggregated tables (features.parquet, deep_research_*.csv)
EXTERNAL = DATA / "external"    # the GBSA study package (upstream source)

# --- specific external assets ---
GBSA_STUDY = EXTERNAL / "gbsa-study"  # symlink to the extracted study tree


def complex_dir(target: str, complex_id: str) -> Path:
    """Path to one complex's per-frame analysis outputs (summary.json, timeseries.parquet, ...)."""
    return RAW / target / complex_id


if __name__ == "__main__":
    for name in ("ROOT", "RAW", "DERIVED", "EXTERNAL", "FIGURES", "TABLES", "GBSA_STUDY"):
        print(f"{name:12s} = {globals()[name]}")
