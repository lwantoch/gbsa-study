#!/usr/bin/env python3
"""One-pass MD analysis for a single discovery-9 production.

Emits: <out_dir>/{target}/{complex_id}/{summary.json, timeseries.parquet,
       contacts_persistence.tsv, hbonds_persistence.tsv, ifp_persistence.tsv}

Metrics (per stride):
  - backbone RMSD (aligned on frame 0 backbone)
  - active-site backbone RMSD (aligned on AS backbone at frame 0)
  - ligand pose drift (aligned on AS backbone, RMSD of UNK heavy atoms)
  - ligand COM displacement from frame 0 (after AS alignment)
  - buried SASA of ligand (freesasa, per frame)
  - per-residue min-distance ligand-AS (contact map, persistence)
  - H-bond count (donor-acceptor), IFP-style interaction counts by type
  - ligand-protein Coulomb sum from topology partial charges
  - water bridges (# waters within 3.5 Å of both UNK and any AS polar atom)

Also static:
  - active-site definition (residues within 5 Å of UNK at frame 0)
  - whole-protein sum of partial charges, integer formal charge inferred from
    residue names (HID/HIE/HIP, ASH/ASP, GLH/GLU, LYN/LYS, CYM/CYS)
  - active-site charge (same, restricted to AS residues)

PBC handling:
  - MDAnalysis `unwrap()` on protein (make whole across box), then wrap
    other groups into the unit cell around protein COM so ligand doesn't
    fly across images. Reference frame captured after unwrap+wrap.
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import MDAnalysis as mda

# Explicit logger for PBC / analysis warnings — surfaced to stderr so batch runs
# can grep for "PBC" and audit exactly which (target, complex, frame) triggered
# a fallback path. See FIX 1 (iter-2): silent `except: pass` around unwrap() was
# hiding real trajectory pathology.
_LOG = logging.getLogger("analyze_complex")
if not _LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s analyze_complex] %(message)s"))
    _LOG.addHandler(_h)
    _LOG.setLevel(logging.INFO)
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.transformations import unwrap, wrap
from MDAnalysis.lib.distances import distance_array, self_distance_array
import freesasa

# ---------- residue / topology helpers ----------

STD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HID","HIE","HIP","ASH","GLH","LYN","CYM","CYX",
    "ACE","NME","NMA","NHE",
}
# integer formal charge per residue name at pH 7 as AMBER encodes it
RESIDUE_CHARGE = {
    "ARG": +1, "LYS": +1, "LYN": 0,
    "ASP": -1, "ASH": 0, "GLU": -1, "GLH": 0,
    "HIS": 0,  "HID": 0, "HIE": 0, "HIP": +1,
    "CYS": 0,  "CYM": -1, "CYX": 0,
    "TYR": 0,
}
TITRATABLE = {"ARG","LYS","LYN","ASP","ASH","GLU","GLH","HIS","HID","HIE","HIP","CYS","CYM","CYX","TYR"}
POLAR_ATOMS = {"N","O","S","F"}  # by element, for water-bridge detection

def parse_top_charges(top_path):
    """Parse a GROMACS .top file into per-residue (resid, resname, sum_charge).
       Uses `#include`s? We rely on the fact that system.top for these systems
       has an explicit protein moleculetype and an explicit ligand moleculetype
       inlined. We just walk the [ atoms ] blocks."""
    per_atom = []   # list of dicts {resid, resname, atomname, charge, moltype}
    current_mol = None
    in_atoms = False
    with open(top_path, "r") as fh:
        for raw in fh:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                sec = line.strip("[] ").lower()
                in_atoms = (sec == "atoms")
                continue
            if line.startswith("#"):
                # macro; if `#define` moltype token, ignore
                continue
            if not in_atoms:
                # look for moleculetype declaration on prior sections
                # actually [ moleculetype ] declares name in the following data line;
                # we track it via a rougher approach: if the line has two tokens
                # only (name + nrexcl) and previous section was moleculetype,
                # capture it. To stay simple, we just record everything and later
                # split on moleculetype boundaries.
                continue
            toks = line.split()
            # [ atoms ] rows: nr type resnr residue atom cgnr charge mass ...
            if len(toks) < 7:
                continue
            try:
                nr = int(toks[0])
                resnr = int(toks[2])
                resname = toks[3]
                atomname = toks[4]
                charge = float(toks[6])
            except ValueError:
                continue
            per_atom.append({
                "nr": nr, "resnr": resnr, "resname": resname,
                "atomname": atomname, "charge": charge,
            })
    return per_atom

def protein_formal_charge(top_atoms):
    """Sum integer per-residue formal charges over the protein moleculetype.
       Detect protein by resname in STD_AA."""
    seen = {}
    for a in top_atoms:
        if a["resname"] not in STD_AA:
            continue
        seen.setdefault((a["resnr"], a["resname"]), None)
    total = 0
    n_titr = 0
    breakdown = {}
    for (resnr, resname) in seen:
        q = RESIDUE_CHARGE.get(resname, 0)
        total += q
        if resname in TITRATABLE:
            n_titr += 1
        breakdown[resname] = breakdown.get(resname, 0) + 1
    return {
        "sum_integer_charge": total,
        "n_titratable": n_titr,
        "residue_counts": breakdown,
    }

def partial_charge_of_atoms(top_atoms, resname_filter=None, atom_pred=None):
    """Sum partial charges over atoms matching filters."""
    tot = 0.0
    for a in top_atoms:
        if resname_filter is not None and a["resname"] not in resname_filter:
            continue
        if atom_pred is not None and not atom_pred(a):
            continue
        tot += a["charge"]
    return tot

# ---------- MD analysis ----------

def build_universe(gro_path, xtc_path, top_path=None):
    """Load Universe(gro, xtc); guess bonds only for protein+ligand (skips 50k
       solvent atoms); optionally overlay per-atom partial charges from the
       .top file so HB analysis and dipole calc work without --tpr."""
    u = mda.Universe(gro_path, xtc_path)
    # bonds only for protein+ligand (unwrap doesn't need waters)
    from MDAnalysis.topology.guessers import guess_bonds
    sel = u.select_atoms("protein or resname UNK")
    bond_tuples = guess_bonds(sel, sel.positions, vdwradii=_VDW)
    # remap to global atom indices
    global_idx = sel.indices
    bonds_global = [(int(global_idx[a]), int(global_idx[b])) for a,b in bond_tuples]
    u.add_TopologyAttr("bonds", bonds_global)
    # charges: default 0, then overlay from .top for protein+ligand
    charges = np.zeros(len(u.atoms), dtype=float)
    if top_path and os.path.exists(top_path):
        top_atoms = parse_top_charges(top_path)
        # map (resid, atomname) -> charge (top uses per-molecule resid; the .gro
        # has global resids that match after the first molecule; for our protein
        # topology this usually lines up, and for UNK the ligand is standalone.)
        top_map = {}
        for a in top_atoms:
            top_map.setdefault((a["resname"], a["atomname"]), []).append(a["charge"])
        for i, atom in enumerate(u.atoms):
            key = (atom.resname, atom.name)
            vals = top_map.get(key)
            if vals:
                charges[i] = vals[0]  # first occurrence
    u.add_TopologyAttr("charges", charges)
    return u

def make_pbc_transforms(u):
    """No-op — PBC handling is done explicitly per analyzed frame via
       protein.unwrap() + lig.unwrap() + min-image shift. On-the-fly
       transforms with wrap(compound='fragments') proved unreliable when
       some atom groups (waters) lack bonds/fragments."""
    return u

def pbc_fix_frame(protein_ag, lig_ag, box, frame_idx=None, prev_lig_cog=None,
                  cog_jump_threshold_A=10.0):
    """Make protein and ligand each whole, then shift ligand into the periodic
       image nearest the protein COM. Modifies positions in place (only in RAM
       for the current frame — MDA reloads next frame from disk).

       Returns a dict with:
         - 'lig_cog'         : ligand center-of-geometry after PBC fixing
         - 'frame_pbc_flag'  : True if the per-frame ligand COG moved more than
                               `cog_jump_threshold_A` from `prev_lig_cog` (a proxy
                               for residual PBC image jumps that survived the fix).

       Only `AttributeError` from `unwrap()` is caught (very old MDAnalysis
       versions without unwrap support) — other exceptions propagate.
    """
    for name, ag in (("protein", protein_ag), ("ligand", lig_ag)):
        try:
            ag.unwrap(compound="fragments", reference="cog", inplace=True)
        except AttributeError as e:
            # Legacy MDAnalysis without .unwrap() — log once per frame, do NOT
            # silently swallow: caller can see something is off.
            _LOG.warning(
                "frame=%s %s.unwrap() unavailable (%s: %s); falling back to raw positions",
                frame_idx, name, type(e).__name__, e,
            )
        # NOTE: NoDataError, ValueError etc. now propagate — that is by design.

    # min-image shift ligand toward protein
    if box is not None and len(lig_ag) and len(protein_ag):
        L = np.asarray(box[:3])
        d = lig_ag.center_of_mass() - protein_ag.center_of_mass()
        shift = L * np.round(d / L)
        lig_ag.positions -= shift

    lig_cog = lig_ag.positions.mean(axis=0) if len(lig_ag) else None
    frame_pbc_flag = False
    if prev_lig_cog is not None and lig_cog is not None:
        jump = float(np.linalg.norm(lig_cog - prev_lig_cog))
        if jump > cog_jump_threshold_A:
            frame_pbc_flag = True
            _LOG.warning(
                "frame=%s ligand COG jumped %.2f A vs previous frame (> %.1f A threshold); "
                "flagging as probable PBC artifact",
                frame_idx, jump, cog_jump_threshold_A,
            )
    return {"lig_cog": lig_cog, "frame_pbc_flag": frame_pbc_flag}

def active_site_residues(u, ligand_sel="resname UNK", cutoff=5.0):
    """Residues with any heavy atom within cutoff Å of ligand at frame 0."""
    u.trajectory[0]
    lig = u.select_atoms(ligand_sel)
    prot = u.select_atoms("protein and not name H*")
    d = distance_array(lig.positions, prot.positions, box=u.dimensions)
    close_atoms = prot[(d < cutoff).any(axis=0)]
    resids = sorted(set((r.segid, r.resid) for r in close_atoms.residues))
    return resids

def resid_selection_string(resids):
    parts = []
    by_seg = {}
    for seg, rid in resids:
        by_seg.setdefault(seg, []).append(rid)
    for seg, rids in by_seg.items():
        rid_str = " ".join(str(r) for r in sorted(set(rids)))
        parts.append(f"(segid {seg} and resid {rid_str})")
    return " or ".join(parts)

def sasa_of_ligand(prot_positions, prot_radii, lig_positions, lig_radii):
    """Buried SASA of ligand ≈ SASA(lig alone) - SASA(lig in complex, ligand only).
       Uses freesasa.calcCoord (C fast path, no Python-side addAtom loop)."""
    n_prot = len(prot_positions); n_lig = len(lig_positions)
    coord_j = np.vstack([prot_positions, lig_positions]).astype(np.float64).ravel().tolist()
    radii_j = np.concatenate([prot_radii, lig_radii]).astype(np.float64).tolist()
    result_j = freesasa.calcCoord(coord_j, radii_j)
    sasa_lig_in_complex = sum(result_j.atomArea(i) for i in range(n_prot, n_prot + n_lig))
    coord_l = lig_positions.astype(np.float64).ravel().tolist()
    radii_l = lig_radii.astype(np.float64).tolist()
    result_l = freesasa.calcCoord(coord_l, radii_l)
    return result_l.totalArea() - sasa_lig_in_complex

_VDW = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"S":1.80,"P":1.80,"F":1.47,"CL":1.75,"BR":1.85,"I":1.98}
def _vdw_radius(atom):
    el = getattr(atom, "element", None) or atom.name[0]
    return _VDW.get(el.upper()[:2], _VDW.get(el.upper()[:1], 1.7))

def _freesasa_structure_from(coords, radii):
    """Build a freesasa.Structure from raw coords+radii (skip PDB roundtrip)."""
    s = freesasa.Structure()
    for i, ((x,y,z), r) in enumerate(zip(coords, radii)):
        s.addAtom(atomName=f"X{i%1000:03d}", residueName="XXX",
                  residueNumber=str(i%9999), chainLabel="A",
                  x=float(x), y=float(y), z=float(z), radius=float(r))
    return s

def coulomb_sum(u, lig_ag, as_ag, top_charge_map, cutoff=10.0):
    """Sum q_i q_j / r_ij (kcal/mol · Å-equivalent, unnormalized) between
       ligand atoms and active-site protein atoms within cutoff. Returns unit
       is arbitrary — we care about relative values across complexes."""
    # map (resid,atomname) -> charge from topology
    lig_q = np.array([top_charge_map.get((a.resid, a.name), 0.0) for a in lig_ag])
    as_q  = np.array([top_charge_map.get((a.resid, a.name), 0.0) for a in as_ag])
    d = distance_array(lig_ag.positions, as_ag.positions, box=u.dimensions)
    mask = (d > 0) & (d < cutoff)
    q_prod = np.outer(lig_q, as_q)
    contrib = np.where(mask, q_prod / np.maximum(d, 1e-6), 0.0)
    return float(contrib.sum())

def hbond_analysis(u, lig_sel="resname UNK", as_sel_str=None, stride=1):
    """MDAnalysis HydrogenBondAnalysis between ligand and active site.

    We pass explicit donor/acceptor selections built from element rules so we
    don't need partial charges (which aren't available from a .gro topology).
    Donors: N, O, S atoms that have at least one H bonded (approximated with
      a name-based rule: any polar heavy atom whose name matches N*/O*/S* is
      allowed as donor, MDA filters by bond connectivity internally).
    Acceptors: any N/O/S (Ss are weak but allowed for consistency).
    """
    lig_ag = u.select_atoms(lig_sel)
    as_ag  = u.select_atoms(as_sel_str)
    def _sel(ag):
        idx = " ".join(str(i+1) for i in ag.indices)  # 1-based
        return f"index {idx}"
    donors_sel = f"({_sel(lig_ag)} or {_sel(as_ag)}) and (name N* or name O* or name S*)"
    accept_sel = f"({_sel(lig_ag)} or {_sel(as_ag)}) and (name N* or name O* or name S*)"
    hydrogens_sel = f"({_sel(lig_ag)} or {_sel(as_ag)}) and (name H* or name *H)"
    hba = HydrogenBondAnalysis(
        universe=u,
        donors_sel=donors_sel,
        acceptors_sel=accept_sel,
        hydrogens_sel=hydrogens_sel,
        d_a_cutoff=3.5,
        d_h_a_angle_cutoff=150.0,
        update_selections=False,
    )
    hba.run(step=stride, verbose=False)
    return hba

def ifp_per_frame(u, lig_ag, as_ag, stride=1):
    """Compute simple interaction-fingerprint counts per frame.

    Categories:
      - HB (donor->acceptor, distance <3.5 Å, cheap heuristic: N/O within cutoff)
      - hydrophobic (C-C contact within 4.5 Å between apolar carbons)
      - ionic (opposite-sign charged sidechain within 5 Å)
      - aromatic proxy (ring-atom C-C within 5.5 Å) — coarse, but stable
    Returns a DataFrame with per-frame counts and a per-residue persistence dict.
    """
    lig_pol_mask = np.array([a.name[0] in ("N","O","S","F") for a in lig_ag])
    lig_c_mask = np.array([a.name[0] == "C" for a in lig_ag])

    per_frame_rows = []
    residue_contacts = {}  # (resid,resname) -> array of bool per selected frame
    frames_selected = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0:
            continue
        frames_selected.append(ts.frame * ts.dt)  # time in ps
        L = lig_ag.positions
        P = as_ag.positions
        d = distance_array(L, P, box=u.dimensions)
        # per-residue min distance
        by_res = {}
        for j, atom in enumerate(as_ag):
            k = (atom.resid, atom.resname)
            v = d[:, j].min()
            if k not in by_res or v < by_res[k]:
                by_res[k] = v
        # accumulate persistence at 4.5 Å
        for k, v in by_res.items():
            residue_contacts.setdefault(k, []).append(v < 4.5)
        # counts
        pol_mask = np.array([a.name[0] in ("N","O","S","F") for a in as_ag])
        c_mask   = np.array([a.name[0] == "C" for a in as_ag])
        hb = int(((d < 3.5) & lig_pol_mask[:,None] & pol_mask[None,:]).sum())
        hp = int(((d < 4.5) & lig_c_mask[:,None] & c_mask[None,:]).sum())
        per_frame_rows.append({"time_ps": ts.frame * ts.dt, "hb_contacts": hb, "hp_contacts": hp,
                               "min_res_dist": float(min(by_res.values()))})
    df = pd.DataFrame(per_frame_rows)
    # pad residue persistence arrays to same length
    n = len(frames_selected)
    persistence = {}
    for k, arr in residue_contacts.items():
        if len(arr) < n:
            arr = arr + [False] * (n - len(arr))
        persistence[k] = float(np.mean(arr))
    return df, persistence

def water_bridges(u, lig_ag, as_ag, stride=1):
    """Count waters within 3.5 Å of both a ligand polar atom and any AS polar atom."""
    waters = u.select_atoms("resname HOH or resname WAT or resname SOL or resname TIP3")
    if len(waters) == 0:
        return []
    wat_o = waters.select_atoms("name O* or type OW")
    lig_pol = lig_ag.select_atoms("name N* or name O* or name S* or name F*")
    as_pol  = as_ag.select_atoms("name N* or name O* or name S*")
    rows = []
    for i, ts in enumerate(u.trajectory):
        if i % stride != 0:
            continue
        d_lig = distance_array(wat_o.positions, lig_pol.positions, box=u.dimensions).min(axis=1)
        d_prot = distance_array(wat_o.positions, as_pol.positions, box=u.dimensions).min(axis=1)
        n_bridge = int(((d_lig < 3.5) & (d_prot < 3.5)).sum())
        rows.append({"time_ps": ts.frame * ts.dt, "n_water_bridges": n_bridge})
    return rows

# ---------- main ----------

def analyze(gro, xtc, top, out_dir, stride=50, ifp_stride=50, hb_stride=10):
    """stride = frame stride for time-series. dt=2ps, 15001 frames → stride=50 gives 300 pts."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    u = build_universe(gro, xtc, top_path=top)
    make_pbc_transforms(u)

    protein = u.select_atoms("protein")
    lig = u.select_atoms("resname UNK")
    if len(protein) == 0 or len(lig) == 0:
        raise RuntimeError(f"empty protein ({len(protein)}) or ligand ({len(lig)})")

    # active site
    as_resids = active_site_residues(u, cutoff=5.0)
    as_sel_str = resid_selection_string(as_resids)
    as_ag = u.select_atoms(as_sel_str)
    as_bb = u.select_atoms(f"({as_sel_str}) and backbone")
    protein_bb = protein.select_atoms("backbone")

    # topology charges
    top_atoms = parse_top_charges(top)
    top_charge_map = {(a["resnr"], a["atomname"]): a["charge"] for a in top_atoms}
    prot_charge_info = protein_formal_charge(top_atoms)
    prot_partial_q = partial_charge_of_atoms(top_atoms, resname_filter=STD_AA)

    as_resnames = {r.resname for r in as_ag.residues}
    as_res_keys = {(r.resid, r.resname) for r in as_ag.residues}
    as_partial_q = sum(a["charge"] for a in top_atoms if (a["resnr"], a["resname"]) in as_res_keys)
    as_formal = sum(RESIDUE_CHARGE.get(rn, 0) for _, rn in as_res_keys)
    as_titr = [rn for _, rn in as_res_keys if rn in TITRATABLE]
    lig_partial_q = sum(a["charge"] for a in top_atoms if a["resname"] == "UNK")

    # reference frame (frame 0, PBC-fixed)
    u.trajectory[0]
    _pbc0 = pbc_fix_frame(protein, lig, u.dimensions, frame_idx=0, prev_lig_cog=None)
    prev_lig_cog = _pbc0["lig_cog"]
    frame_pbc_flags = []  # per analyzed frame (parallel to `times`)
    ref_bb_pos = protein_bb.positions.copy()
    ref_as_bb_pos = as_bb.positions.copy()
    ref_lig_pos = lig.positions.copy()
    lig_heavy = lig.select_atoms("not name H*")
    ref_lig_heavy = lig_heavy.positions.copy()
    protein_ca = protein.select_atoms("name CA")
    ref_ca_pos = protein_ca.positions.copy()

    # topology-derived ligand charges (aligned with lig_heavy for dipole calc)
    lig_atom_charges = np.array([top_charge_map.get((a.resid, a.name), 0.0) for a in lig])
    lig_heavy_charges = np.array([top_charge_map.get((a.resid, a.name), 0.0) for a in lig_heavy])

    # basic charged-group selections for salt-bridge heuristic
    prot_neg = u.select_atoms("(resname ASP GLU) and (name OD1 OD2 OE1 OE2)")
    prot_pos = u.select_atoms("(resname LYS ARG HIP) and (name NZ NE NH1 NH2 NE2 ND1)")
    lig_neg_ix = np.where(lig_atom_charges < -0.4)[0]
    lig_pos_ix = np.where(lig_atom_charges > +0.4)[0]

    # time series accumulators
    times = []
    rmsd_bb = []
    rmsd_as_bb = []
    lig_drift = []
    lig_com_disp = []
    lig_sasa = []
    coulomb = []
    protein_rg = []
    lig_internal_rmsd = []
    lig_rg = []
    lig_asph = []
    lig_dipole_mag = []
    salt_bridges_lp = []      # ligand-protein charged contact count
    vdw_contacts = []          # heavy-heavy < 4 Å between UNK and protein
    # RMSF accumulation over ligand heavy atoms (after alignment)
    n_lh = len(lig_heavy)
    sum_pos = np.zeros((n_lh, 3))
    sum_pos2 = np.zeros((n_lh, 3))
    n_used = 0
    # RMSF accumulation over protein Cα
    n_ca = len(protein_ca)
    sum_ca = np.zeros((n_ca, 3))
    sum_ca2 = np.zeros((n_ca, 3))
    # ligand principal axis for rotational autocorrelation
    lig_axis_ts = []          # list of unit-vectors (first principal axis of ligand heavy atoms)
    # ligand aligned positions for clustering (only if size permits)
    lig_aligned_frames = []

    # precomputed selections/radii (used every frame — expensive to rebuild)
    protein_heavy = protein.select_atoms("not name H*")
    as_heavy = as_ag.select_atoms("not name H*")
    prot_heavy_radii = np.array([_vdw_radius(a) for a in protein_heavy])
    lig_radii = np.array([_vdw_radius(a) for a in lig])

    # IFP per-residue book-keeping (collect per-frame contact bits in main loop)
    as_res_keys_sorted = sorted({(a.resid, a.resname) for a in as_heavy})
    _res_atom_idx = {}
    for k in as_res_keys_sorted:
        resid_k, _ = k
        _res_atom_idx[k] = np.array([i for i,a in enumerate(as_heavy) if a.resid == resid_k], dtype=int)
    ifp_bits_ts = []           # per-frame bool array (n_res,)
    ifp_hb_ts = []
    ifp_hp_ts = []
    ifp_minres_ts = []

    for ts in u.trajectory[::stride]:
        t_ps = ts.frame * ts.dt
        # explicit per-frame PBC fix (with sanity check vs previous frame's COG)
        _pbc_info = pbc_fix_frame(protein, lig, u.dimensions,
                                  frame_idx=ts.frame, prev_lig_cog=prev_lig_cog)
        prev_lig_cog = _pbc_info["lig_cog"]
        frame_pbc_flags.append(bool(_pbc_info["frame_pbc_flag"]))
        # global backbone RMSD (align on whole protein backbone)
        rmsd_bb.append(float(rms.rmsd(protein_bb.positions, ref_bb_pos, superposition=True)))
        # active-site backbone RMSD (align on AS bb)
        rmsd_as_bb.append(float(rms.rmsd(as_bb.positions, ref_as_bb_pos, superposition=True)))
        # ligand drift after aligning on AS bb
        # rotate current frame onto ref via AS bb, apply to ligand
        mobile_center = as_bb.center_of_mass()
        ref_center = ref_as_bb_pos.mean(axis=0)
        R, _ = align.rotation_matrix(as_bb.positions - mobile_center,
                                     ref_as_bb_pos - ref_center)
        lig_aligned = (lig.positions - mobile_center) @ R.T + ref_center
        lig_drift.append(float(np.sqrt(((lig_aligned - ref_lig_pos)**2).sum(axis=1).mean())))
        # ligand COM displacement (after alignment)
        com_aligned = lig_aligned.mean(axis=0)
        ref_lig_com_val = ref_lig_pos.mean(axis=0)
        lig_com_disp.append(float(np.linalg.norm(com_aligned - ref_lig_com_val)))
        # buried SASA (protein heavy + ligand vs ligand alone)
        try:
            lig_sasa.append(float(sasa_of_ligand(
                protein_heavy.positions, prot_heavy_radii,
                lig.positions, lig_radii)))
        except Exception as _sasa_e:
            if not lig_sasa:  # log the first failure only
                with open(os.path.join(out_dir, "sasa_error.txt"), "w") as fh:
                    fh.write(f"{type(_sasa_e).__name__}: {_sasa_e}")
            lig_sasa.append(float("nan"))
        # Coulomb (unnormalized)
        coulomb.append(coulomb_sum(u, lig, as_ag, top_charge_map))
        # RMSF accumulation (using aligned ligand)
        lig_heavy_aligned = (lig_heavy.positions - mobile_center) @ R.T + ref_center
        sum_pos += lig_heavy_aligned
        sum_pos2 += lig_heavy_aligned**2

        # protein Cα RMSF accumulation (align on protein backbone)
        Rca, _ = align.rotation_matrix(protein_bb.positions - protein_bb.center_of_mass(),
                                       ref_bb_pos - ref_bb_pos.mean(axis=0))
        ca_aligned = (protein_ca.positions - protein_bb.center_of_mass()) @ Rca.T + ref_bb_pos.mean(axis=0)
        sum_ca += ca_aligned
        sum_ca2 += ca_aligned**2

        # protein Rg (whole backbone)
        protein_rg.append(float(protein_bb.radius_of_gyration()))

        # ligand internal RMSD (fit UNK to itself, ignore protein)
        lig_internal_rmsd.append(float(rms.rmsd(lig_heavy.positions, ref_lig_heavy, superposition=True)))

        # ligand Rg + shape (asphericity from gyration tensor eigenvalues)
        lig_rg.append(float(lig_heavy.radius_of_gyration()))
        _lc = lig_heavy.positions - lig_heavy.center_of_mass()
        _gt = (_lc.T @ _lc) / len(lig_heavy)
        _ev = np.sort(np.linalg.eigvalsh(_gt))
        _lam = _ev.sum()
        # asphericity b = lam3 - 0.5*(lam1+lam2); normalize by lam sum to make comparable
        lig_asph.append(float((_ev[2] - 0.5*(_ev[0]+_ev[1])) / _lam) if _lam > 0 else 0.0)

        # ligand dipole magnitude (sum q_i * r_i, e·Å units)
        pos_lig = lig.positions - lig.center_of_mass()
        dip = (lig_atom_charges[:,None] * pos_lig).sum(axis=0)
        lig_dipole_mag.append(float(np.linalg.norm(dip)))

        # salt bridge ligand-protein: pairs of oppositely-charged atoms <4 Å
        sb = 0
        if len(lig_pos_ix) and len(prot_neg) > 0:
            d = distance_array(lig.positions[lig_pos_ix], prot_neg.positions, box=u.dimensions)
            sb += int((d < 4.0).any(axis=1).sum())
        if len(lig_neg_ix) and len(prot_pos) > 0:
            d = distance_array(lig.positions[lig_neg_ix], prot_pos.positions, box=u.dimensions)
            sb += int((d < 4.0).any(axis=1).sum())
        salt_bridges_lp.append(sb)

        # vdW contact count (heavy-heavy < 4 Å) — using AS heavy atoms
        d_lh_as = distance_array(lig_heavy.positions, as_heavy.positions, box=u.dimensions)
        vdw_contacts.append(int((d_lh_as < 4.0).sum()))

        # IFP: per-residue min-distance from ligand + hb/hp counts (reuse distances)
        d_full = distance_array(lig.positions, as_heavy.positions, box=u.dimensions)
        by_res_min = np.array([d_full[:, _res_atom_idx[k]].min() for k in as_res_keys_sorted])
        ifp_bits_ts.append(by_res_min < 4.5)
        ifp_minres_ts.append(float(by_res_min.min()))
        lig_pol_mask = np.array([a.name[0] in ("N","O","S","F") for a in lig])
        as_pol_mask = np.array([a.name[0] in ("N","O","S","F") for a in as_heavy])
        lig_c_mask = np.array([a.name[0] == "C" for a in lig])
        as_c_mask = np.array([a.name[0] == "C" for a in as_heavy])
        ifp_hb_ts.append(int(((d_full < 3.5) & lig_pol_mask[:,None] & as_pol_mask[None,:]).sum()))
        ifp_hp_ts.append(int(((d_full < 4.5) & lig_c_mask[:,None] & as_c_mask[None,:]).sum()))

        # store ligand aligned positions for clustering (cheap: n_lh × 3 × ~150 frames)
        lig_aligned_frames.append(lig_heavy_aligned.copy())

        # ligand principal axis (first eigenvector of gyration tensor) for autocorrelation
        _evec = np.linalg.eigh(_gt)[1][:, 2]
        lig_axis_ts.append(_evec / np.linalg.norm(_evec))

        n_used += 1
        times.append(t_ps)

    mean_pos = sum_pos / n_used
    var_pos = sum_pos2 / n_used - mean_pos**2
    rmsf = np.sqrt(np.maximum(var_pos, 0).sum(axis=1))
    mean_ca = sum_ca / n_used
    var_ca = sum_ca2 / n_used - mean_ca**2
    ca_rmsf = np.sqrt(np.maximum(var_ca, 0).sum(axis=1))
    # active-site Cα mask
    as_resid_set = set(rid for _, rid in as_resids)
    as_ca_mask = np.array([atom.resid in as_resid_set for atom in protein_ca])
    as_ca_rmsf_mean = float(ca_rmsf[as_ca_mask].mean()) if as_ca_mask.any() else float("nan")
    as_ca_rmsf_max = float(ca_rmsf[as_ca_mask].max()) if as_ca_mask.any() else float("nan")

    # ligand binding-mode count (RMSD clustering at three cutoffs) on aligned frames
    def _n_clusters(frames, cutoff):
        if not frames:
            return 0
        centroids = [frames[0]]
        for f in frames[1:]:
            best = min(float(np.sqrt(((f - c)**2).sum(axis=1).mean())) for c in centroids)
            if best > cutoff:
                centroids.append(f)
        return len(centroids)
    n_clusters_1p0 = _n_clusters(lig_aligned_frames, 1.0)
    n_clusters_2p0 = _n_clusters(lig_aligned_frames, 2.0)

    # ligand orientation autocorrelation: <cos²θ(t)> between axis(0) and axis(t)
    if len(lig_axis_ts) > 1:
        ax0 = lig_axis_ts[0]
        cos2 = np.array([float((v @ ax0)**2) for v in lig_axis_ts])
        orient_autocorr_mean = float(cos2.mean())
        orient_autocorr_last = float(cos2[-1])
    else:
        orient_autocorr_mean = float("nan"); orient_autocorr_last = float("nan")

    ts_df = pd.DataFrame({
        "time_ps": times,
        "rmsd_bb_A": rmsd_bb,
        "rmsd_as_bb_A": rmsd_as_bb,
        "lig_drift_A": lig_drift,
        "lig_com_disp_A": lig_com_disp,
        "lig_buried_sasa_A2": lig_sasa,
        "coulomb_arb": coulomb,
        "protein_rg_A": protein_rg,
        "lig_internal_rmsd_A": lig_internal_rmsd,
        "lig_rg_A": lig_rg,
        "lig_asphericity": lig_asph,
        "lig_dipole_eA": lig_dipole_mag,
        "salt_bridges_lp": salt_bridges_lp,
        "vdw_contacts": vdw_contacts,
        # iter-2 FIX 1: per-frame PBC sanity flag (True when ligand COG jumped
        # >10 A vs the previous frame after unwrap — probable image artifact).
        "frame_pbc_flag": frame_pbc_flags,
    })
    ts_df.to_parquet(os.path.join(out_dir, "timeseries.parquet"))

    # H-bond analysis (native MDA)
    try:
        hba = hbond_analysis(u, as_sel_str=as_sel_str, stride=hb_stride)
        counts = np.asarray(hba.count_by_time())        # 1-D per analyzed frame
        times_hb = np.asarray(hba.times)                # matches counts length
        pd.DataFrame({"time_ps": times_hb, "n_hb": counts}).to_parquet(
            os.path.join(out_dir, "hbond_timeseries.parquet"))
        types = np.asarray(hba.count_by_type())         # MDA 2.10: 3 columns [d, a, count]
        cols = ["donor_ix","hydrogen_ix","acceptor_ix","count"][:types.shape[1]] \
                if types.ndim == 2 else ["value"]
        pd.DataFrame(types, columns=cols) \
            .to_csv(os.path.join(out_dir, "hbonds_persistence.tsv"), sep="\t", index=False)
        hb_mean = float(counts.mean()) if len(counts) else 0.0
        hb_std = float(counts.std()) if len(counts) else 0.0
        hb_persistence_frac = float((counts > 0).mean()) if len(counts) else 0.0
    except Exception as e:
        hb_mean = float("nan"); hb_std = float("nan"); hb_persistence_frac = float("nan")
        with open(os.path.join(out_dir, "hbond_error.txt"), "w") as fh:
            fh.write(f"{type(e).__name__}: {e}")

    # IFP from in-loop bits: per-residue persistence + tanimoto vs frame 0
    ifp_bits_mat = np.array(ifp_bits_ts)   # (n_frames, n_res)
    contact_persist = {as_res_keys_sorted[i]: float(ifp_bits_mat[:, i].mean())
                       for i in range(len(as_res_keys_sorted))}
    with open(os.path.join(out_dir, "contacts_persistence.tsv"), "w") as fh:
        fh.write("resid\tresname\tpersistence\n")
        for (resid, resname), p in sorted(contact_persist.items()):
            fh.write(f"{resid}\t{resname}\t{p:.4f}\n")
    ifp_df = pd.DataFrame({
        "time_ps": times,
        "hb_contacts": ifp_hb_ts,
        "hp_contacts": ifp_hp_ts,
        "min_res_dist": ifp_minres_ts,
    })
    ifp_df.to_parquet(os.path.join(out_dir, "ifp_timeseries.parquet"))
    # Tanimoto vs frame 0
    ref_bits = ifp_bits_mat[0]
    tan_arr = np.array([
        (np.logical_and(b, ref_bits).sum() / max(1, int(np.logical_or(b, ref_bits).sum())))
        for b in ifp_bits_mat
    ])
    tan_median = float(np.median(tan_arr))
    tan_last = float(tan_arr[-1]) if len(tan_arr) else float("nan")
    tan_entropy = -float(np.sum([p*np.log(p) for p in np.unique(tan_arr) if p > 0])) if len(tan_arr) else float("nan")
    wb_mean = float("nan")   # water bridges skipped for perf

    summary = {
        "n_frames_used_for_ts": int(len(times)),
        "traj_dt_ps": float(u.trajectory.dt),
        "traj_n_frames": int(u.trajectory.n_frames),
        "n_protein_residues": int(len(protein.residues)),
        "n_ligand_atoms": int(len(lig)),
        "n_lig_heavy": int(n_lh),
        "active_site_resids": [{"segid": s, "resid": r} for s,r in as_resids],
        "n_active_site_residues": int(len(as_resids)),
        "protein_partial_charge": float(prot_partial_q),
        "protein_formal_charge": int(prot_charge_info["sum_integer_charge"]),
        "protein_n_titratable": int(prot_charge_info["n_titratable"]),
        "active_site_partial_charge": float(as_partial_q),
        "active_site_formal_charge": int(as_formal),
        "active_site_titratable_names": as_titr,
        "active_site_resnames": sorted(as_resnames),
        "ligand_partial_charge_sum": float(lig_partial_q),
        # ts summaries
        "rmsd_bb_mean_A": float(np.mean(rmsd_bb)),
        "rmsd_bb_std_A": float(np.std(rmsd_bb)),
        "rmsd_bb_last_A": float(rmsd_bb[-1]),
        "rmsd_as_bb_mean_A": float(np.mean(rmsd_as_bb)),
        "rmsd_as_bb_std_A": float(np.std(rmsd_as_bb)),
        "lig_drift_mean_A": float(np.mean(lig_drift)),
        "lig_drift_std_A": float(np.std(lig_drift)),
        "lig_drift_last_A": float(lig_drift[-1]),
        "lig_com_disp_mean_A": float(np.mean(lig_com_disp)),
        "lig_com_disp_max_A": float(np.max(lig_com_disp)),
        "lig_com_disp_last_A": float(lig_com_disp[-1]),
        "lig_escape_frac": float(np.mean(np.array(lig_com_disp) > 3.0)),
        "lig_buried_sasa_mean_A2": float(np.nanmean(lig_sasa)),
        "lig_buried_sasa_std_A2": float(np.nanstd(lig_sasa)),
        "coulomb_mean_arb": float(np.mean(coulomb)),
        "coulomb_std_arb": float(np.std(coulomb)),
        "lig_rmsf_mean_A": float(np.mean(rmsf)),
        "lig_rmsf_max_A": float(np.max(rmsf)),
        "n_hb_mean": hb_mean,
        "n_hb_std": hb_std,
        "hb_persistence_frac": hb_persistence_frac,
        "n_water_bridges_mean": wb_mean,
        "ifp_tanimoto_median_vs_ref": tan_median,
        "ifp_tanimoto_last_vs_ref": tan_last,
        "ifp_tanimoto_entropy": tan_entropy,
        # extras
        "protein_rg_mean_A": float(np.mean(protein_rg)),
        "protein_rg_std_A": float(np.std(protein_rg)),
        "lig_internal_rmsd_mean_A": float(np.mean(lig_internal_rmsd)),
        "lig_internal_rmsd_std_A": float(np.std(lig_internal_rmsd)),
        "lig_internal_rmsd_last_A": float(lig_internal_rmsd[-1]),
        "lig_rg_mean_A": float(np.mean(lig_rg)),
        "lig_asphericity_mean": float(np.mean(lig_asph)),
        "lig_dipole_mean_eA": float(np.mean(lig_dipole_mag)),
        "lig_dipole_std_eA": float(np.std(lig_dipole_mag)),
        "salt_bridges_lp_mean": float(np.mean(salt_bridges_lp)),
        "salt_bridges_lp_persistence": float(np.mean(np.array(salt_bridges_lp) > 0)),
        "vdw_contacts_mean": float(np.mean(vdw_contacts)),
        "vdw_contacts_std": float(np.std(vdw_contacts)),
        "protein_ca_rmsf_mean_A": float(ca_rmsf.mean()),
        "protein_ca_rmsf_max_A": float(ca_rmsf.max()),
        "as_ca_rmsf_mean_A": as_ca_rmsf_mean,
        "as_ca_rmsf_max_A": as_ca_rmsf_max,
        "lig_binding_modes_1A": int(n_clusters_1p0),
        "lig_binding_modes_2A": int(n_clusters_2p0),
        "lig_orient_autocorr_mean": orient_autocorr_mean,
        "lig_orient_autocorr_last": orient_autocorr_last,
        # iter-2 FIX 1: PBC audit
        "n_frames_pbc_flagged": int(sum(frame_pbc_flags)),
        "frac_frames_pbc_flagged": float(np.mean(frame_pbc_flags)) if frame_pbc_flags else 0.0,
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # per-residue Cα RMSF + AS flag
    rmsf_df = pd.DataFrame({
        "resid": [a.resid for a in protein_ca],
        "resname": [a.resname for a in protein_ca],
        "ca_rmsf_A": ca_rmsf,
        "is_active_site": as_ca_mask.astype(int),
    })
    rmsf_df.to_parquet(os.path.join(out_dir, "protein_ca_rmsf.parquet"))

    # per-atom ligand RMSF
    lig_rmsf_df = pd.DataFrame({
        "atomname": [a.name for a in lig_heavy],
        "element": [a.name[0] for a in lig_heavy],
        "rmsf_A": rmsf,
    })
    lig_rmsf_df.to_parquet(os.path.join(out_dir, "ligand_rmsf.parquet"))

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gro", required=True)
    ap.add_argument("--xtc", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ifp-stride", type=int, default=50)
    ap.add_argument("--hb-stride", type=int, default=10)
    args = ap.parse_args()
    summary = analyze(args.gro, args.xtc, args.top, args.out,
                      stride=args.stride, ifp_stride=args.ifp_stride, hb_stride=args.hb_stride)
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v, (list,dict))}, indent=2))

if __name__ == "__main__":
    main()
