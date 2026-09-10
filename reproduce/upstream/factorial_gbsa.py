#!/usr/bin/env python3
"""Full-factorial per-frame GBSA driver for ONE complex.

Efficiency contract (see plan): the expensive per-complex work — PBC-whole
trjconv + index build — is done ONCE. Then the 48 PHYSICS combinations
(igb x intdiel x saltcon x molsurf) each run gmx_MMPBSA over the SAME 10-ps
whole trajectory with ``-eo`` (per-frame CSV). Per-frame components are stored
so that all TEMPORAL factors (length, stride, equil-offset) and the IE/C2
entropies are computed POST-HOC by slicing — never re-run.

Usage:
  factorial_gbsa.py <production_dir> <out_dir> [--skip N] [--combos a,b] [--np K]

<production_dir> = workspace/<id>/production  (needs process/gromacs.{tpr,top,xtc})
"""
from __future__ import annotations
import argparse, itertools, json, os, shutil, subprocess, sys, time
from pathlib import Path

GROMACS_CUDA = "/home/otras/hcx/lwa/gromacs-cuda/.pixi/envs/default"
os.environ["PATH"] = f"{GROMACS_CUDA}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{GROMACS_CUDA}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# ---- factor grid (PHYSICS only; temporal + entropy are post-hoc) ----
IGB     = [1, 2, 5, 8]
INTDIEL = [1.0, 2.0, 4.0]
SALTCON = [0.0, 0.15]
# Nonpolar factor: surface-tension coefficient (kcal/mol/Å²). 0.0072 = standard,
# 0.0 = no nonpolar (SASA) term. Uses the working LCPO surface (molsurf=0);
# molsurf=1 (analytical molecular surface) is broken in this gmx_MMPBSA+MPI build
# (fails to emit _GMXMMPBSA_*_gb_surf.dat and aborts), so surften is the working
# nonpolar axis with the same interpretation (does the SASA term matter?).
SURFTEN = [0.0072, 0.0]

def all_combos():
    out = []
    for igb, indi, salt, sft in itertools.product(IGB, INTDIEL, SALTCON, SURFTEN):
        out.append(dict(igb=igb, intdiel=indi, saltcon=salt, surften=sft))
    return out

def combo_id(c):
    return f"igb{c['igb']}_di{c['intdiel']:g}_salt{c['saltcon']:g}_st{c['surften']:g}"

def build_index_and_whole(prod_dir: Path, out_dir: Path, skip: int):
    """Reuse gbsa.py's molecule-aware Receptor/Ligand index + PBC-whole trjconv.
    Returns (tpr, gmx_top, whole_xtc, index_ndx, n_frames)."""
    import BioSimSpace as BSS
    manifest = json.loads((prod_dir / "result.json").read_text())
    gro = Path(manifest["gro_file"]); tpr = prod_dir / "process/gromacs.tpr"
    gmx_top = prod_dir / "process/gromacs.top"; traj = prod_dir / "process/gromacs.xtc"
    for p in (gro, tpr, gmx_top, traj):
        if not p.exists(): raise FileNotFoundError(p)

    system = BSS.IO.readMolecules([str(gro), str(gmx_top)], make_whole=True)._sire_object
    mols = list(system)
    def has_unk(m):
        try: return any(str(r.name().value()) == "UNK" for r in m.residues())
        except Exception: return False
    lig_idx = next((i for i, m in enumerate(mols) if has_unk(m)), 1)
    protein_nums = {mols[i].number() for i in range(lig_idx)}
    ligand_num = mols[lig_idx].number()
    rec, lig, ac = [], [], 1
    for m in system:
        n = len(m.atoms()); s, e = ac, ac + n
        if m.number() in protein_nums: rec.extend(range(s, e))
        elif m.number() == ligand_num: lig.extend(range(s, e))
        ac = e
    if not rec or not lig:
        raise RuntimeError(f"index build failed rec={len(rec)} lig={len(lig)}")
    ndx = out_dir / "index.ndx"
    with open(ndx, "w") as fh:
        fh.write("[ Receptor ]\n")
        for i in range(0, len(rec), 15): fh.write(" ".join(map(str, rec[i:i+15])) + "\n")
        fh.write("\n[ Ligand ]\n")
        for i in range(0, len(lig), 15): fh.write(" ".join(map(str, lig[i:i+15])) + "\n")

    whole = out_dir / "traj_whole_10ps.xtc"
    subprocess.run(
        [shutil.which("gmx") or "gmx", "trjconv", "-s", str(tpr), "-f", str(traj),
         "-o", str(whole), "-pbc", "mol", "-center", "-skip", str(skip)],
        input="Protein\nSystem\n", text=True, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(out_dir))
    # frame count
    chk = subprocess.run([shutil.which("gmx") or "gmx", "check", "-f", str(whole)],
                         capture_output=True, text=True)
    nfr = 0
    for ln in (chk.stderr + chk.stdout).splitlines():
        if ln.strip().startswith(("Coords", "Step")):
            try: nfr = int(ln.split()[1]); break
            except Exception: pass
    return tpr, gmx_top, whole, ndx, nfr

def run_combo(c, tpr, gmx_top, whole, ndx, out_dir, np_mpi):
    from gbsa_pipeline.mmbsa import GBParams, GeneralParams, MMPBSAConfig, run_gmx_mmpbsa_from_gromacs
    cdir = out_dir / combo_id(c); cdir.mkdir(parents=True, exist_ok=True)
    inp = cdir / "mmpbsa.in"
    MMPBSAConfig(
        pb=None,
        gb=GBParams(igb=c["igb"], intdiel=c["intdiel"], saltcon=c["saltcon"],
                    surften=c["surften"], molsurf=0),
        general=GeneralParams(interval=1),   # use ALL frames of the 10-ps whole traj
    ).write(inp)
    exe = "/home/otras/hcx/lwa/mmbsa_gmx_mmpbsa_mpi.sh" if np_mpi > 0 else "gmx_MMPBSA"
    if np_mpi > 0:
        os.environ["GBSA_NP"] = str(np_mpi)   # the MPI wrapper reads GBSA_NP (else defaults to 32)
    t0 = time.time()
    res = run_gmx_mmpbsa_from_gromacs(
        input_file=inp, complex_structure=tpr, trajectory=whole, topology=gmx_top,
        index_file=ndx, receptor_group=0, ligand_group=1, output_dir=cdir,
        gmx_mmpbsa=exe, extra_args=["-eo", "per_frame.csv"])
    dt = time.time() - t0
    ok = res.returncode == 0 and (cdir / "per_frame.csv").exists()
    return dict(combo=combo_id(c), rc=res.returncode, ok=ok, seconds=round(dt, 1),
                eo=str(cdir / "per_frame.csv"), stderr_tail=res.stderr[-1500:] if not ok else "")

def parse_delta(eo_csv: Path):
    """Return the per-frame DELTA block as list-of-dicts (ΔBOND..ΔTOTAL incl ΔGGAS)."""
    import csv
    rows = list(csv.reader(open(eo_csv)))
    hdr = None; data = []; in_delta = False
    for r in rows:
        if not r: continue
        c0 = str(r[0])
        if c0 == "Delta Energy Terms": in_delta = True; continue
        if in_delta and c0 == "Frame #": hdr = r; continue
        if in_delta and hdr and c0 and c0[0].isdigit():
            rec = {hdr[i]: float(r[i]) for i in range(1, len(hdr))}
            rec["frame"] = int(c0); data.append(rec)
        elif in_delta and c0 and not c0[0].isdigit() and c0 != "Frame #":
            break   # next section after delta (none expected, but safe)
    return data

def store_perframe(out_dir: Path, results):
    """Collate every ok combo's DELTA per-frame block into one tidy parquet."""
    import pandas as pd
    frames = []
    for r in results:
        if not r["ok"]: continue
        d = parse_delta(Path(r["eo"]))
        if not d: continue
        df = pd.DataFrame(d); df.insert(0, "combo", r["combo"])
        frames.append(df)
    if not frames: return None
    alldf = pd.concat(frames, ignore_index=True)
    out = out_dir / "perframe_delta.parquet"
    alldf.to_parquet(out, index=False)
    return out, len(alldf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prod_dir"); ap.add_argument("out_dir")
    ap.add_argument("--skip", type=int, default=5, help="trjconv frame skip (2ps*skip); 5 -> 10 ps")
    ap.add_argument("--combos", default="", help="comma combo_ids subset (default all 48)")
    ap.add_argument("--np", type=int, default=0, help="gmx_MMPBSA MPI ranks (0=serial)")
    a = ap.parse_args()
    prod = Path(a.prod_dir); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tpr, top, whole, ndx, nfr = build_index_and_whole(prod, out, a.skip)
    prep_s = round(time.time() - t0, 1)
    print(f"[prep] whole traj {nfr} frames (skip={a.skip}), index+trjconv {prep_s}s", flush=True)

    combos = all_combos()
    if a.combos:
        want = set(a.combos.split(",")); combos = [c for c in combos if combo_id(c) in want]
    print(f"[run] {len(combos)} physics combos", flush=True)
    results = []
    for i, c in enumerate(combos, 1):
        r = run_combo(c, tpr, top, whole, ndx, out, a.np)
        results.append(r)
        print(f"  [{i}/{len(combos)}] {r['combo']:28} rc={r['rc']} ok={r['ok']} {r['seconds']}s", flush=True)
        if not r["ok"]:
            print("    STDERR:", r["stderr_tail"][-400:], flush=True)
    stored = store_perframe(out, results)
    summary = dict(prod_dir=str(prod), n_frames=nfr, skip=a.skip, prep_s=prep_s,
                   combos=results, total_s=round(time.time() - t0, 1),
                   perframe_parquet=str(stored[0]) if stored else None,
                   perframe_rows=stored[1] if stored else 0)
    (out / "factorial_summary.json").write_text(json.dumps(summary, indent=2))
    nok = sum(1 for r in results if r["ok"])
    print(f"[done] {nok}/{len(results)} combos ok, total {summary['total_s']}s"
          f"{', parquet '+str(stored[1])+' rows' if stored else ''} -> {out/'factorial_summary.json'}", flush=True)

if __name__ == "__main__":
    main()
