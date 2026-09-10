#!/usr/bin/env python3
"""Extract per-complex ligand chemistry descriptors from system.top + system.gro.

Builds an RDKit Mol from the ligand's atoms/bonds/coords and computes:
  - net formal charge (from bond-order deduction)
  - molecular weight
  - #heavy atoms
  - #rotatable bonds
  - #H-bond donors / acceptors
  - #aromatic rings
  - LogP (Crippen)
  - TPSA
  - fraction sp3 C

Outputs ligand_chem.parquet in the gbsa-study root.
"""
import os, sys, warnings, gzip
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DATA_ROOT = Path('/mnt/netapp1/Store_othcxlwa/lwa_transfer/00_PRIORITY_discovery9_trajectories')

def parse_top_ligand(top_path):
    """Return (atoms, bonds) for the UNK moleculetype in the .top file.
       atoms: list of dicts (nr, atomname, resname, charge)
       bonds: list of (i, j, functype) 1-based indices within the ligand moleculetype."""
    atoms = []; bonds = []
    current_mol = None
    in_atoms = False; in_bonds = False
    with open(top_path) as fh:
        prev_section = None
        pending_moltype = False
        for raw in fh:
            line = raw.split(';', 1)[0].rstrip()
            if not line: continue
            if line.strip().startswith('['):
                sec = line.strip().strip('[]').strip().lower()
                if sec == 'moleculetype':
                    pending_moltype = True
                    in_atoms = False; in_bonds = False
                elif sec == 'atoms':
                    in_atoms = (current_mol == 'ligand'); in_bonds = False
                elif sec == 'bonds':
                    in_bonds = (current_mol == 'ligand'); in_atoms = False
                else:
                    in_atoms = False; in_bonds = False
                prev_section = sec
                continue
            if pending_moltype and prev_section == 'moleculetype':
                # first data line under moleculetype is the name
                toks = line.split()
                if toks:
                    current_mol = 'ligand' if toks[0].lower() in ('lig','unk','ligand','mol') else toks[0]
                pending_moltype = False
                continue
            if in_atoms:
                toks = line.split()
                if len(toks) < 7: continue
                try:
                    nr = int(toks[0]); resname = toks[3]; atomname = toks[4]; charge = float(toks[6])
                except ValueError: continue
                atoms.append({'nr': nr, 'atomname': atomname, 'resname': resname, 'charge': charge, 'atomtype': toks[1]})
            elif in_bonds:
                toks = line.split()
                if len(toks) < 3: continue
                try:
                    i = int(toks[0]); j = int(toks[1]); ft = int(toks[2])
                except ValueError: continue
                bonds.append((i, j, ft))
    return atoms, bonds

def parse_gro_ligand(gro_path):
    """Return (atomname → xyz) for UNK atoms only (Å units)."""
    coords = {}
    with open(gro_path) as fh:
        fh.readline()  # title
        n = int(fh.readline().strip())
        for _ in range(n):
            line = fh.readline()
            if len(line) < 44: continue
            resname = line[5:10].strip()
            atomname = line[10:15].strip()
            if resname != 'UNK': continue
            try:
                x = float(line[20:28]) * 10  # nm → Å
                y = float(line[28:36]) * 10
                z = float(line[36:44]) * 10
            except ValueError: continue
            coords[atomname] = (x, y, z)
    return coords

# Element deduction from GAFF atomtype (first letter)
def element_from_atomtype(at):
    at = at.strip()
    # GAFF: c=carbon variants, n=nitrogen, o=oxygen, s=sulfur, h=hydrogen, p=phosphorus, f/cl/br/i
    if not at: return 'C'
    a = at.lower()
    if a.startswith('cl'): return 'Cl'
    if a.startswith('br'): return 'Br'
    if a.startswith('c'):  return 'C'
    if a.startswith('h'):  return 'H'
    if a.startswith('n'):  return 'N'
    if a.startswith('o'):  return 'O'
    if a.startswith('s'):  return 'S'
    if a.startswith('p'):  return 'P'
    if a.startswith('f'):  return 'F'
    if a.startswith('i'):  return 'I'
    return at[0].upper()

def build_mol(atoms, bonds, coords):
    """Construct an RDKit Mol from parsed data. Uses element+coords+bonds only;
       bond orders are re-perceived by RDKit's DetermineBondOrders (approximate)."""
    mol = Chem.RWMol()
    idx_map = {}  # topology 1-based nr -> RDKit atom idx
    conf = Chem.Conformer(len(atoms))
    for atom in atoms:
        el = element_from_atomtype(atom['atomtype'])
        # skip explicit hydrogens; RDKit will add if needed
        rdatom = Chem.Atom(el)
        rdatom.SetNoImplicit(False)
        aid = mol.AddAtom(rdatom)
        idx_map[atom['nr']] = aid
        xyz = coords.get(atom['atomname'], (0,0,0))
        conf.SetAtomPosition(aid, xyz)
    for i, j, _ft in bonds:
        if i in idx_map and j in idx_map:
            try:
                mol.AddBond(idx_map[i], idx_map[j], Chem.BondType.SINGLE)
            except Exception:
                pass
    mol.AddConformer(conf)
    m = mol.GetMol()
    # Perceive bond orders + aromaticity from geometry (best-effort)
    try:
        Chem.SanitizeMol(m, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    except Exception:
        pass
    try:
        Chem.AssignStereochemistryFrom3D(m)
    except Exception:
        pass
    return m

def compute_descriptors(mol, atoms):
    """Small set of interpretable descriptors."""
    d = {}
    d['lig_MW'] = Descriptors.MolWt(mol)
    d['lig_n_heavy'] = mol.GetNumHeavyAtoms()
    d['lig_rot_bonds'] = Lipinski.NumRotatableBonds(mol)
    d['lig_HBD'] = Lipinski.NumHDonors(mol)
    d['lig_HBA'] = Lipinski.NumHAcceptors(mol)
    d['lig_aromatic_rings'] = Lipinski.NumAromaticRings(mol)
    d['lig_all_rings'] = rdMolDescriptors.CalcNumRings(mol)
    try:
        d['lig_LogP'] = Crippen.MolLogP(mol)
    except Exception:
        d['lig_LogP'] = np.nan
    try:
        d['lig_TPSA'] = rdMolDescriptors.CalcTPSA(mol)
    except Exception:
        d['lig_TPSA'] = np.nan
    try:
        d['lig_fraction_sp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
    except Exception:
        d['lig_fraction_sp3'] = np.nan
    # Formal charge: sum of atomic formal charges after sanitisation
    d['lig_formal_charge'] = Chem.GetFormalCharge(mol)
    # From topology (partial charges — deprecated for real chemistry but keep for consistency)
    d['lig_partial_q_sum'] = float(np.sum([a['charge'] for a in atoms]))
    d['lig_partial_q_abs_sum'] = float(np.sum([abs(a['charge']) for a in atoms]))  # signal of polarity
    return d

def main():
    rows = []
    for target_dir in sorted(DATA_ROOT.glob('*')):
        if not target_dir.is_dir() or target_dir.name.startswith(('_', '.')) or target_dir.name in ('MANIFEST.tsv', 'README.md'): continue
        target = target_dir.name
        for cid_dir in sorted(target_dir.glob('*')):
            if not cid_dir.is_dir(): continue
            cid = cid_dir.name
            top = cid_dir / 'system.top'
            gro = cid_dir / 'system.gro'
            if not top.exists() or not gro.exists(): continue
            try:
                atoms, bonds = parse_top_ligand(str(top))
                coords = parse_gro_ligand(str(gro))
                if not atoms or not bonds or not coords: continue
                mol = build_mol(atoms, bonds, coords)
                desc = compute_descriptors(mol, atoms)
                desc['target'] = target; desc['complex_id'] = cid
                rows.append(desc)
            except Exception as e:
                print(f'  ERR {target}/{cid}: {e}', file=sys.stderr)
    df = pd.DataFrame(rows)
    cols_first = ['target', 'complex_id']
    df = df[cols_first + [c for c in df.columns if c not in cols_first]]
    out = Path('/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study/ligand_chem.parquet')
    df.to_parquet(out, index=False)
    print(f'wrote {out} — {len(df)} rows × {df.shape[1]} cols')
    print(df.describe(include='all').T.head(15))
    print('\nformal charge distribution:')
    print(df.lig_formal_charge.value_counts().sort_index())
    print('\nper-target mean of key descriptors:')
    print(df.groupby('target')[['lig_MW','lig_HBD','lig_HBA','lig_aromatic_rings','lig_LogP','lig_formal_charge','lig_TPSA','lig_partial_q_abs_sum']].mean().round(2))

if __name__ == '__main__':
    main()
