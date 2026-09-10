#!/usr/bin/env python3
"""GROMACS MD-parameter study configs (newbench 9-target speed study).

Two blocks:
  SPEED  = Taguchi L27 orthogonal array over 5 three-level SPEED factors
           (dt/HMR, MTS, rcoulomb, rvdw, nstlist). 27 runs = 1/9 of the full 3^5=243,
           all 5 main effects balanced & orthogonal -> clean eta^2 for speed AND ranking.
  ENSEMBLE = 5 OFAT robustness configs (sd, md-vv+MTTK, v-rescale+C-rescale,
             all-bonds, tight-LINCS) vs the baseline. NOT speed levers; kept separate.

Each config -> a dict of gromacs.mdp key overrides applied on top of the per-complex
baseline production mdp. Frames kept at 2 ps (nstxout recomputed per dt); nsteps for 30 ns
(overridable for the smoke test). Fresh Maxwell velocities from the equilibrated system.gro,
same gen-seed across configs so velocity draw does not confound the parameter effect.
"""
PROD_NS   = 20.0     # production length (user: 20 ns for the deadline BO/spectrum run)
FRAME_PS  = 10.0     # 10 ps frames (matches factorial base stride; ~1.3 TB for L27x270 @20ns)
GEN_SEED  = 12345    # fixed so all configs share the initial velocity draw

# ---- speed-factor level maps (index 0/1/2) ----
DT_FS  = [2, 3, 4]          # timestep; MOST important speed lever
MRF    = [1.0, 2.0, 3.0]    # mass-repartition-factor paired with dt (HMR)
MTS    = [1, 2, 3]          # PME/long-range every N steps (1 = mts off)
RCOUL  = [1.0, 1.1, 1.2]    # Coulomb real-space cutoff (nm)
GAP    = [0.0, 0.1, 0.2]    # Coulomb-LJ gap: rvdw = rcoulomb - gap
                            # (GROMACS Verlet law: only rcoulomb >= rvdw is allowed with PME)
NSTL   = [20, 40, 80]       # neighbor-list update frequency

def _common(dt_fs, nsteps=None):
    dt_ps = dt_fs / 1000.0
    ns = nsteps if nsteps is not None else round(PROD_NS * 1000.0 / dt_ps)
    return {
        "dt": f"{dt_ps:.4f}",
        "nsteps": str(int(ns)),
        "nstxout-compressed": str(int(round(FRAME_PS / dt_ps))),
        "continuation": "no",
        "gen-vel": "yes",
        "gen-temp": "300.0",
        "gen-seed": str(GEN_SEED),
        # let grompp recompute the PME grid from fourierspacing (box-dependent)
        "fourier-nx": "0", "fourier-ny": "0", "fourier-nz": "0",
        # gentle, uniform coupling: Parrinello-Rahman with nstpcouple=100 blows up at 4 fs
        # (CUDA illegal-address). 12 is stable at 4 fs and a multiple of every mts-factor.
        "nstcalcenergy": "12", "nstpcouple": "12", "nstcomm": "12",
        "nstenergy": "600", "nstlog": "600",
    }

def _mts(factor):
    if factor <= 1:
        return {"mts": "no"}
    # MTS requires nstcalcenergy/nstpcouple/nstcomm/nstenergy/nstlog to be multiples of
    # the factor. 12 = LCM(2,3,4) covers all our factors; 600 is a multiple of 12.
    return {"mts": "yes", "mts-levels": "2",
            "mts-level2-forces": "longrange-nonbonded",
            "mts-level2-factor": str(int(factor)),
            "nstcalcenergy": "12", "nstpcouple": "12", "nstcomm": "12",
            "nstenergy": "600", "nstlog": "600"}

def l27_rows():
    """Standard GF(3) L27: 3 independent columns + 2 derived, all orthogonal."""
    rows = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                d = (a + b + c) % 3          # rvdw column
                e = (a + 2 * b + 2 * c) % 3  # nstlist column
                rows.append((a, b, c, d, e))
    return rows

def build_speed():
    cfgs = []
    for i, (a, b, c, d, e) in enumerate(l27_rows()):
        dt_fs = DT_FS[a]
        rcoul = RCOUL[c]; rvdw = round(rcoul - GAP[d], 2)
        ov = _common(dt_fs)
        ov["mass-repartition-factor"] = f"{MRF[a]:.1f}"
        ov.update(_mts(MTS[b]))
        ov["rcoulomb"] = f"{rcoul:.2f}"
        ov["rvdw"]     = f"{rvdw:.2f}"
        ov["nstlist"]  = str(NSTL[e])
        cfgs.append(dict(
            id=f"sp{i:02d}", block="speed",
            dt_fs=dt_fs, mts=MTS[b], rcoulomb=rcoul, gap=GAP[d], rvdw=rvdw, nstlist=NSTL[e],
            pme_interval_fs=dt_fs * MTS[b],
            overrides=ov,
            note=f"dt{dt_fs} mts{MTS[b]} rc{rcoul} lj{rvdw} nl{NSTL[e]}"))
    return cfgs

def build_ensemble():
    base = lambda: _common(DT_FS[0])   # dt=2 fs baseline for all ensemble OFAT
    out = []
    # sd: Langevin is its own thermostat -> tcoupl=no, no MTS
    ov = base(); ov.update(_mts(1)); ov.update(
        {"integrator": "sd", "tcoupl": "no", "pcoupl": "C-rescale",
         "mass-repartition-factor": "1.0", "rcoulomb": "1.0", "rvdw": "1.0", "nstlist": "20"})
    out.append(dict(id="ens_sd", block="ensemble", overrides=ov,
                    note="integrator=sd (Langevin), tcoupl=no, C-rescale, mts off"))
    # md-vv velocity-Verlet + nose-hoover + C-rescale (consistent VV NPT), no MTS.
    # (MTTK barostat is incompatible with LINCS constraints -> use C-rescale, keep LINCS.)
    ov = base(); ov.update(_mts(1)); ov.update(
        {"integrator": "md-vv", "tcoupl": "nose-hoover", "pcoupl": "C-rescale",
         "mass-repartition-factor": "1.0", "rcoulomb": "1.0", "rvdw": "1.0", "nstlist": "20"})
    out.append(dict(id="ens_mdvv", block="ensemble", overrides=ov,
                    note="md-vv (velocity-Verlet) + nose-hoover + C-rescale, mts off"))
    # v-rescale + C-rescale (modern robust coupling), keep baseline speed knobs
    ov = base(); ov.update(_mts(2)); ov.update(
        {"tcoupl": "v-rescale", "pcoupl": "C-rescale", "tau-t": "1.0",
         "mass-repartition-factor": "1.0", "rcoulomb": "1.0", "rvdw": "1.0", "nstlist": "20"})
    out.append(dict(id="ens_vrescale", block="ensemble", overrides=ov,
                    note="tcoupl=v-rescale + pcoupl=C-rescale"))
    # all-bonds constraints
    ov = base(); ov.update(_mts(2)); ov.update(
        {"constraints": "all-bonds", "mass-repartition-factor": "1.0",
         "rcoulomb": "1.0", "rvdw": "1.0", "nstlist": "20"})
    out.append(dict(id="ens_allbonds", block="ensemble", overrides=ov,
                    note="constraints=all-bonds"))
    # tight LINCS
    ov = base(); ov.update(_mts(2)); ov.update(
        {"lincs-order": "6", "lincs-iter": "2", "mass-repartition-factor": "1.0",
         "rcoulomb": "1.0", "rvdw": "1.0", "nstlist": "20"})
    out.append(dict(id="ens_lincshi", block="ensemble", overrides=ov,
                    note="lincs-order=6, lincs-iter=2"))
    return out

def all_configs():
    return build_speed() + build_ensemble()

def get(cfg_id, nsteps=None):
    for c in all_configs():
        if c["id"] == cfg_id:
            if nsteps is not None:   # smoke override of length
                dt_ps = float(c["overrides"]["dt"])
                c["overrides"]["nsteps"] = str(int(nsteps))
            return c
    raise KeyError(cfg_id)

if __name__ == "__main__":
    import csv, sys
    cs = all_configs()
    w = csv.writer(sys.stdout)
    w.writerow(["id", "block", "dt_fs", "mts", "rcoulomb", "rvdw", "nstlist",
                "pme_interval_fs", "note"])
    for c in cs:
        w.writerow([c["id"], c["block"], c.get("dt_fs", 2), c.get("mts", ""),
                    c.get("rcoulomb", ""), c.get("rvdw", ""), c.get("nstlist", ""),
                    c.get("pme_interval_fs", ""), c["note"]])
    sys.stderr.write(f"\n{len(cs)} configs "
                     f"({sum(1 for c in cs if c['block']=='speed')} speed L27 + "
                     f"{sum(1 for c in cs if c['block']=='ensemble')} ensemble)\n")
    # sanity: L27 balance (each speed level appears 9x) + stability prune check
    sp = [c for c in cs if c["block"] == "speed"]
    for key, levels in [("dt_fs", DT_FS), ("mts", MTS), ("rcoulomb", RCOUL),
                        ("gap", GAP), ("nstlist", NSTL)]:
        from collections import Counter
        cnt = Counter(c[key] for c in sp)
        bal = all(cnt[l] == 9 for l in levels)
        sys.stderr.write(f"  {key}: {dict(cnt)} balanced={bal}\n")
    bad = [c["id"] for c in sp if c["pme_interval_fs"] > 12]
    sys.stderr.write(f"  PME-interval>12fs (unstable): {bad or 'none'}\n")
