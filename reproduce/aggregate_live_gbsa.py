#!/usr/bin/env python3
"""Aggregate live SLURM GBSA outputs into reviewer-firm raw CSVs.

Reads (with fallback if the live workspace is not mounted):
  * OHDS cross-check chains:
      $XCHK_ROOT/*/mmpbsa_work/FINAL_RESULTS_MMPBSA.dat
    where XCHK_ROOT defaults to
      /mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/pipeline-mps/_ohds_xchk_all/work
  * FT3 full-chain repro (docking → 30 ns MD → GBSA):
      $REPRO_ROOT/*/gbsa/mmpbsa_work/FINAL_RESULTS_MMPBSA.dat
      + $REPRO_ROOT/*/signac_statepoint.json
    where REPRO_ROOT defaults to
      /mnt/netapp1/Store_othcxlwa/pipeline-mps-workspaces/discovery9_ohds_repro/workspace

Writes (regardless of source availability):
  data/raw/ohds_xchk_chains.csv           (target, complex_id, dTOTAL, components…)
  data/raw/ft3_repro_chains.csv           (target, ligand, replica, dTOTAL, components…)

Notebook 31 reads *only these CSVs*, so once this script has run, the analysis
is reproducible from checked-in data even without access to Lustre/Store live
workspaces (reviewer-firm).
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import pandas as pd

DEFAULT_XCHK = "/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/pipeline-mps/_ohds_xchk_all/work"
DEFAULT_REPRO = "/mnt/netapp1/Store_othcxlwa/pipeline-mps-workspaces/discovery9_ohds_repro/workspace"

COMPONENTS = ["VDWAALS", "EEL", "EGB", "ESURF", "GGAS", "GSOLV", "TOTAL"]
_ROW_RE = re.compile(r"^Δ([A-Z\-]+(?:\s[A-Z]+)?)\s+([\-\+0-9\.]+)")


def parse_final_results(dat_path: Path) -> dict | None:
    out: dict = {}
    for line in dat_path.read_text().splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        name = m.group(1).strip().replace(" ", "_")
        if name in COMPONENTS:
            try:
                out[f"d{name}"] = float(m.group(2))
            except ValueError:
                pass
    return out if "dTOTAL" in out else None


def aggregate_xchk(root: Path) -> pd.DataFrame:
    if not root.exists():
        print(f"  xchk root not mounted: {root} — leaving CSV unchanged if present")
        return pd.DataFrame()
    rows = []
    for wd in sorted(root.iterdir()):
        dat = wd / "mmpbsa_work" / "FINAL_RESULTS_MMPBSA.dat"
        if not dat.exists():
            continue
        parts = wd.name.split("_", 1)
        if len(parts) != 2:
            continue
        target, cid = parts
        vals = parse_final_results(dat)
        if vals is None:
            continue
        rows.append({"target": target, "complex_id": cid, **vals})
    return pd.DataFrame(rows)


def aggregate_repro(root: Path) -> pd.DataFrame:
    if not root.exists():
        print(f"  repro root not mounted: {root} — leaving CSV unchanged if present")
        return pd.DataFrame()
    rows = []
    for jd in sorted(root.iterdir()):
        dat = jd / "gbsa" / "mmpbsa_work" / "FINAL_RESULTS_MMPBSA.dat"
        sp = jd / "signac_statepoint.json"
        if not (dat.exists() and sp.exists()):
            continue
        st = json.loads(sp.read_text())
        vals = parse_final_results(dat)
        if vals is None:
            continue
        rows.append({
            "target": st["target_pdb"],
            "ligand": st["ligand_name"],
            "replica": st["replica"],
            **vals,
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xchk-root", default=os.environ.get("XCHK_ROOT", DEFAULT_XCHK))
    p.add_argument("--repro-root", default=os.environ.get("REPRO_ROOT", DEFAULT_REPRO))
    args = p.parse_args()

    # locate raw/ inside gbsa-study repo (this script lives at reproduce/aggregate_live_gbsa.py)
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print(f"→ xchk root : {args.xchk_root}")
    print(f"→ repro root: {args.repro_root}")
    print(f"→ writing to: {raw}")

    xchk = aggregate_xchk(Path(args.xchk_root))
    if not xchk.empty:
        out = raw / "ohds_xchk_chains.csv"
        xchk.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(xchk)} rows)")

    repro = aggregate_repro(Path(args.repro_root))
    if not repro.empty:
        out = raw / "ft3_repro_chains.csv"
        repro.to_csv(out, index=False)
        print(f"  wrote {out}  ({len(repro)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
