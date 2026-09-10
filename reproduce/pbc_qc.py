#!/usr/bin/env python3
"""PBC / drift QC — flag per-trajectory artifacts.

For every complex under data/raw/complex_analyses/{TARGET}/{CID}/timeseries.parquet,
compute:
  - max frame-to-frame COM jump  (diff of lig_com_disp_A)  — flag > COM_JUMP_THRESHOLD_A
  - max lig_drift_A              — flag > DRIFT_THRESHOLD_A as likely unbound

Writes data/derived/pbc_qc.csv with columns:
  target, complex_id, max_com_jump_A, max_drift_A, is_pbc_artifact, is_probably_unbound

Usage (from the repo env):
    cd env && pixi run python ../reproduce/pbc_qc.py

Threshold rationale (iter-3 FIX 9):

* ``COM_JUMP_THRESHOLD_A = 6.0 A`` (name in-code: ``MAX_COM_JUMP_A``).
  A frame-to-frame ligand-COM jump larger than 6 Å over the 100 ps between saved
  frames means the ligand would have to translate faster than typical protein-bound
  motion. The physical scale used to set the threshold is the **TIP3P first-solvation-shell
  diameter (~5 Å)**: any single-frame displacement more than ~1.5× that is much more
  likely a periodic-image switch (the ligand crossed the box boundary and the
  wrapping algorithm placed its image on the other side) than a real translation.
  6 Å is deliberately permissive — legitimate rare pocket-hopping events do exist
  and can produce 3–5 Å jumps; the threshold catches unambiguous image flips only.

* ``DRIFT_THRESHOLD_A = 8.0 A`` (name in-code: ``MAX_DRIFT_A``).
  Typical binding-pocket depth in the discovery-9 set is 5–10 Å (measured as the
  maximum pocket-lining-atom distance to the pocket surface). A sustained drift
  larger than 8 Å over 30 ns means the ligand's COM has translated more than the
  typical pocket depth — i.e., it has left the primary pocket. 8 Å is again
  permissive: some induced-fit repositioning is legitimately in the 5–8 Å range,
  so we do not treat drift-only complexes as artifacts, only as "probably unbound"
  (the downstream `hardened_claim_b.py:bound_only` condition filters on this).

The two flags are reported separately so downstream analyses can filter on one,
the other, or both. Only ``is_pbc_artifact`` is used by the `pbc_clean` condition in
`hardened_claim_b.py`; ``is_probably_unbound`` is used by the `bound_only` condition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src importable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from discovery9.paths import RAW, DERIVED

# --- QC thresholds (see module docstring for physical rationale) --------------
COM_JUMP_THRESHOLD_A = 6.0   # frame-to-frame ligand-COM jump; > this = PBC image flip
DRIFT_THRESHOLD_A = 8.0      # max ligand drift over trajectory; > this = probably unbound

# Aliases matching the names used in the iter-3 SYNTHESIS.md fix-pack text.
MAX_COM_JUMP_A = COM_JUMP_THRESHOLD_A
MAX_DRIFT_A = DRIFT_THRESHOLD_A


def per_traj_qc(ts: pd.DataFrame) -> dict:
    out: dict = {"max_com_jump_A": np.nan, "max_drift_A": np.nan}
    if "lig_com_disp_A" in ts.columns and len(ts) > 1:
        d = np.abs(np.diff(ts["lig_com_disp_A"].to_numpy(dtype=float)))
        out["max_com_jump_A"] = float(np.nanmax(d)) if d.size else float("nan")
    if "lig_drift_A" in ts.columns:
        out["max_drift_A"] = float(np.nanmax(ts["lig_drift_A"].to_numpy(dtype=float)))
    return out


def main() -> int:
    ca = RAW / "complex_analyses"
    if not ca.is_dir():
        print(f"ERROR: {ca} not found", file=sys.stderr)
        return 2

    rows = []
    targets = sorted(p.name for p in ca.iterdir() if p.is_dir())
    for target in targets:
        tdir = ca / target
        for cdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            ts_path = cdir / "timeseries.parquet"
            if not ts_path.is_file():
                continue
            try:
                ts = pd.read_parquet(ts_path)
            except Exception as e:  # noqa: BLE001 — report and skip a broken file
                print(f"WARN failed to read {ts_path}: {e}", file=sys.stderr)
                continue
            qc = per_traj_qc(ts)
            qc["target"] = target
            qc["complex_id"] = cdir.name
            qc["is_pbc_artifact"] = bool(
                np.isfinite(qc["max_com_jump_A"]) and qc["max_com_jump_A"] > COM_JUMP_THRESHOLD_A
            )
            qc["is_probably_unbound"] = bool(
                np.isfinite(qc["max_drift_A"]) and qc["max_drift_A"] > DRIFT_THRESHOLD_A
            )
            rows.append(qc)

    if not rows:
        print("ERROR: no trajectories found", file=sys.stderr)
        return 3

    df = pd.DataFrame(rows)[
        ["target", "complex_id", "max_com_jump_A", "max_drift_A",
         "is_pbc_artifact", "is_probably_unbound"]
    ].sort_values(["target", "complex_id"]).reset_index(drop=True)

    DERIVED.mkdir(parents=True, exist_ok=True)
    out_path = DERIVED / "pbc_qc.csv"
    df.to_csv(out_path, index=False)

    n_total = len(df)
    n_pbc = int(df.is_pbc_artifact.sum())
    n_unbound = int(df.is_probably_unbound.sum())
    print(f"wrote {out_path}  ·  {n_total} trajectories")
    print(f"  PBC artifacts (COM jump > {COM_JUMP_THRESHOLD_A} A):     {n_pbc}  ({100*n_pbc/n_total:.1f}%)")
    print(f"  probably unbound (drift > {DRIFT_THRESHOLD_A} A):     {n_unbound}  ({100*n_unbound/n_total:.1f}%)")
    # per-target breakdown
    print("\nper-target QC breakdown:")
    per = df.groupby("target").agg(
        n=("complex_id", "size"),
        n_pbc=("is_pbc_artifact", "sum"),
        n_unbound=("is_probably_unbound", "sum"),
    )
    print(per.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
