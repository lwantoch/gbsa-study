"""Per-target benchmark descriptors, parsed out of the NewBench SI PDF (table S4).

Parsed rather than hand-transcribed so the numbers quoted in METHODS.md cannot drift
from the source document. `verify.py` re-runs `build()` and compares it to the shipped
`tables/benchmark_quality.csv`. Requires `pdftotext -layout` (poppler-utils).

    python -m gbsabench.benchmark_quality
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .paths import EXTERNAL, RAW, ROOT, TABLES

SI = EXTERNAL / "newbench_SI.pdf"
OUT = TABLES / "benchmark_quality.csv"

# Closed vocabulary from SI table S4. Matched off the END of the free-text
# "<target name> <family>" field, longest first, because both parts are of
# unknown width and the target name is truncated with an ellipsis in some rows.
FAMILIES = ["Electrochemical transporter", "Ligand-gated ion channel",
            "Other cytosolic protein", "Nuclear receptor", "Secreted protein",
            "Unclassified", "Transferase", "Hydrolase", "Protease", "Kinase", "Reader"]

# PDB, <name+family>, resolution, pKi range, Ki span (nM), scaffolds, NN Tanimoto,
# relaxed inactives, censored inactives, MW overlap. Anchored on the numeric tail.
ROW = re.compile(
    r"^\s*(?P<pdb>[0-9][A-Z0-9]{3})\s+(?P<rest>.+?)\s+"
    r"(?P<resolution_A>\d\.\d{2})\s+(?P<pki>\d+\.\d+-\d+\.\d+)\s+(?P<ki_span_nM>[\d.]+-[\d.]+)\s+"
    r"(?P<n_scaffolds>\d+)\s+(?P<nn_tanimoto_median>[01]\.\d{2})\s+"
    r"(?P<n_relaxed_inactive>\d+)\s+(?P<n_censored_inactive>\d+)\s+"
    r"(?P<mw_overlap>[01]\.\d{2})\s*$")

COLS = ["pdb", "target_name", "family", "in_discovery_set", "resolution_A",
        "pki_min", "pki_max", "ki_span_nM", "n_scaffolds", "nn_tanimoto_median",
        "n_relaxed_inactive", "n_censored_inactive", "mw_overlap"]


def si_text() -> str:
    return subprocess.run(["pdftotext", "-layout", str(SI), "-"],
                          capture_output=True, text=True, check=True).stdout


def build() -> pd.DataFrame:
    rows = []
    for line in si_text().splitlines():
        m = ROW.match(line)
        if not m:
            continue
        d = m.groupdict()
        rest = d.pop("rest")
        fam = next((f for f in FAMILIES if rest.endswith(f)), None)
        if fam is None:
            raise ValueError(f"unrecognised family in SI row: {rest!r}")
        d["family"], d["target_name"] = fam, rest[: -len(fam)].strip()
        lo, hi = d.pop("pki").split("-")
        d["pki_min"], d["pki_max"] = float(lo), float(hi)
        rows.append(d)
    if len(rows) != 27:
        raise ValueError(f"expected 27 SI target rows, parsed {len(rows)}")
    df = pd.DataFrame(rows)
    for c in ("resolution_A", "nn_tanimoto_median", "mw_overlap"):
        df[c] = df[c].astype(float)
    for c in ("n_scaffolds", "n_relaxed_inactive", "n_censored_inactive"):
        df[c] = df[c].astype(int)
    disc = set(pd.read_csv(RAW / "metadata.csv").target.unique())
    df["in_discovery_set"] = df.pdb.isin(disc)
    return df.sort_values(["in_discovery_set", "pdb"], ascending=[False, True])[COLS]


if __name__ == "__main__":
    df = build()
    df.to_csv(OUT, index=False)
    d = df[df.in_discovery_set]
    if len(d) != 9:
        sys.exit(f"expected 9 discovery targets in the SI table, matched {len(d)}")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} targets, {len(d)} in the discovery set")
    print(f"  discovery MW overlap  {d.mw_overlap.min():.2f}-{d.mw_overlap.max():.2f}")
    print(f"  discovery scaffolds   {d.n_scaffolds.min()}-{d.n_scaffolds.max()}")
    print(f"  discovery NN Tanimoto {d.nn_tanimoto_median.min():.2f}-{d.nn_tanimoto_median.max():.2f}")
    print(f"  discovery pKi         {d.pki_min.min():.1f}-{d.pki_max.max():.1f}")
    print(f"  discovery relaxed     {d.n_relaxed_inactive.min()}-{d.n_relaxed_inactive.max()}/20, "
          f"censored {d.n_censored_inactive.sum()} total")
    print(f"  all-27 MW overlap     {df.mw_overlap.min():.2f}-{df.mw_overlap.max():.2f}")
