"""Repository paths — resolved RELATIVE to this file, never hardcoded.

Every notebook and script imports its data/figure/table locations from here, so the
whole project works unchanged after someone clones or unzips it to any directory.

    from gbsabench.paths import RAW, DERIVED, FIGURES, TABLES
    df = pd.read_csv(RAW / "gbsa_dG_raw.csv")

How the root is found: this file lives at <repo>/src/gbsabench/paths.py, so the repo
root is three levels up. We double-check by looking for a marker file (pyproject.toml)
and, as a fallback, walk upwards until we find one. No absolute paths, no "~", no
dependence on the current working directory.
"""
from __future__ import annotations

from pathlib import Path

_MARKER = "pyproject.toml"


def find_repo_root() -> Path:
    """Return the repository root directory.

    First guess: three parents up from this file (<repo>/src/gbsabench/paths.py).
    If that does not contain the marker file, walk upwards until one does.
    """
    here = Path(__file__).resolve()
    guess = here.parents[2]  # src/gbsabench/paths.py -> src/gbsabench -> src -> repo
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
# FIGURES flattened: after iter_05 flattening, both upstream and downstream NBs write
# to a single figures/ tree with NN_slug_figK.png prefixes that no longer collide.
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
DOCS = ROOT / "docs"
REPRODUCE = ROOT / "reproduce"

# --- data sub-locations ---
# Upstream RAW/DERIVED live under data/external/gbsa-study/ (symlink to the
# gbsa-study snapshot), NOT under the downstream data/raw & data/derived which
# hold different tables. Merge iter_05.
_GBSA_STUDY = DATA / "external" / "gbsa-study"
RAW = _GBSA_STUDY / "data" / "raw"          # raw per-measurement CSVs (read-only source of truth)
DERIVED = _GBSA_STUDY / "data" / "derived"  # computed tables (recomputed by the notebooks)
EXTERNAL = DATA / "external"                # heavy reference inputs (SI PDF; trajectories fetched separately)


if __name__ == "__main__":
    # Quick self-check: `python -m gbsabench.paths` prints the resolved locations.
    for name in ("ROOT", "RAW", "DERIVED", "FIGURES", "TABLES"):
        print(f"{name:8s} = {globals()[name]}")
