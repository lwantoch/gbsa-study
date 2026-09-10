# Glossary

Two sections: (1) **Terminology** — one-line entries for vocabulary that shows up across the
READMEs and NBs; (2) **Column glossary** — every column in `features.parquet` and the
per-complex JSONs, mapped to plain English + physical meaning.

## Terminology

| Term | One-line meaning |
|------|------------------|
| **LOTO** | Leave-One-Target-Out cross-validation — train on 8 of 9 discovery targets, test on the held-out one, repeat 9× (or 8×, per convention). |
| **BEDROC α=20** | Truchon–Bayly Boltzmann-Enhanced Discrimination of ROC; α=20 weights the top ~8% of the ranked list, so the number reflects early enrichment, not whole-list ordering. Range [0, 1], higher is better. |
| **sp-config** | Single-point MM-GBSA parameter combination — one of 48 in the study (`igb × intdiel × saltcon × surften`). Sometimes written `sp00`, `sp08`, `sp25` in the Study-2 GROMACS-config context (a different sweep). |
| **Taguchi L27** | Orthogonal-array screening design that samples 27 configurations from a nominally 3-level 5-factor factorial, so main effects are estimable without running the full 243. |
| **P1 / P2 / P3** | Three ML task framings: **P1** = per-target combo selection (predict which GBSA combo wins on a target from its MD fingerprint); **P2** = fingerprint → is_active on that target; **P3** = full factorial (target × combo × MD) meta-model. |
| **MW-residualised** | A feature after per-target OLS regression on molecular weight; residuals are used to check that a signal isn't just a molecular-size confound. |
| **Panel BEDROC** | Mean of per-target BEDROC α=20 across the panel; targets where a scorer has no data are imputed at 0 (worst case). Distinguish from the 8-target-subset value, which averages only where the scorer has data. |
| **Bootstrap** | Target-level resample with replacement; canonical regime B=5000, seed 20260902. Some historical NBs use B=1000, seed 20250901. |
| **BH-FDR** | Benjamini–Hochberg false-discovery-rate correction for multiple testing; q<0.10 is the threshold used in this study. |
| **Hardening** | The seven-condition robustness sweep in NB 28: naive / MW-residualised / size-residualised / bound-only / PBC-clean / `alt_label_7` / `alt_label_6` (the last two are the CSV column literals in `hardened_claim_b.csv`). A "hardened" Claim B is one that survives all seven. |
| **Canonical baseline** | The single-source-of-truth BEDROC values in `data/derived/canonical_baselines.csv`, produced by `reproduce/canonical_baselines.py`. When README and a NB disagree, canonical wins. |
| **Panel** | Shorthand for "the 9 discovery targets averaged"; distinguish from the 27-target NewBench cohort (9 discovery + 18 validation). |
| **Envelope** | Loose bound on a design space — e.g. "conservative envelope" for the BO winner MD config means the winner picked cautious settings well inside safe LINCS / MTS / cutoff limits. |
| **GBSA-locked** | The single MM-GBSA combo `igb2_di4_salt0.15_st0.0072` chosen as the panel recipe; the "beat me" baseline everywhere. |
| **Discovery-9 vs NewBench-27** | Discovery-9 = the 9 targets with MD + GBSA already computed (the panel used for every headline number). NewBench-27 = 9 discovery + 18 validation (pre-registered, not computed yet). |

## Column glossary

Every column in `features.parquet` and the per-complex JSONs, mapped to plain English +
physical meaning.

## Ligand kinematics (from `data/raw/complex_analyses/**/summary.json`)

| Column | Plain-English meaning |
|--------|-----------------------|
| `lig_drift_mean_A / _std_A / _last_A` | Ligand pose drift after aligning the active-site backbone (Å) — how much the ligand moves inside its pocket. |
| `lig_com_disp_mean_A / _max_A / _last_A` | Distance of the ligand centre-of-mass from its initial position (Å). |
| `lig_escape_frac` | Fraction of frames with `lig_com_disp > 3 Å` — how often the ligand walked off. |
| `lig_internal_rmsd_mean_A / _std_A / _last_A` | Ligand self-RMSD (Å) — conformational stability of the ligand ignoring the pocket. |
| `lig_rmsf_mean_A / _max_A` | Ligand per-atom RMSF (Å) after alignment. |
| `lig_binding_modes_1A / _2A` | Number of distinct ligand poses at 1.0 / 2.0 Å RMSD-cluster cutoff. |
| `lig_orient_autocorr_mean / _last` | ⟨cos²θ⟩ between the ligand's principal axis at t and at 0. 1 = no rotation, 1/3 = fully random. |

## Ligand geometry / electrostatics

| Column | Meaning |
|--------|---------|
| `lig_rg_mean_A` | Ligand radius of gyration (Å). |
| `lig_asphericity_mean` | Ligand shape descriptor (0 = spherical, 1 = rod). |
| `lig_dipole_mean_eA / _std_eA` | Ligand dipole magnitude in e·Å from partial charges + geometry. |
| `ligand_partial_charge_sum` / `lig_partial_q_sum` | Sum of ligand partial charges — should be ≈ 0 for all systems (neutral by construction, so this column is numerical noise; **do not use for correlation**). Same column, two names: `ligand_partial_charge_sum` lives in `features.parquet`, `lig_partial_q_sum` in `ligand_chem.parquet`. Both are excluded from the naive top-feature ranking in `reproduce/canonical_baselines.py`. |

## Protein / active-site stability

| Column | Meaning |
|--------|---------|
| `rmsd_bb_mean_A / _std_A / _last_A` | Protein backbone RMSD vs frame 0 (Å). |
| `rmsd_as_bb_mean_A / _std_A` | Active-site backbone RMSD (Å) — usually lower than global BB when the ligand is bound. |
| `protein_rg_mean_A / _std_A` | Protein radius of gyration (Å). |
| `protein_ca_rmsf_mean_A / _max_A` | Global Cα RMSF (Å). |
| `as_ca_rmsf_mean_A / _max_A` | Cα RMSF restricted to active-site residues. |

## Interactions

| Column | Meaning |
|--------|---------|
| `vdw_contacts_mean / _std` | Ligand-heavy ↔ active-site-heavy pairs at distance < 4 Å per frame (count). |
| `n_hb_mean / _std` | H-bond count between ligand and active-site atoms (MDA HBA, 3.5 Å / 150°). |
| `hb_persistence_frac` | Fraction of analysed frames with at least one H-bond. |
| `salt_bridges_lp_mean / _persistence` | Salt-bridge contacts between oppositely-charged atoms (< 4 Å). |
| `coulomb_mean_arb / _std_arb` | Σ q_i · q_j / r_ij between ligand and active-site atoms (arbitrary units, comparable within-target only). |
| `ifp_tanimoto_median_vs_ref / _last / _entropy` | Per-residue-contact-vector Tanimoto vs frame 0 — how conserved the interaction fingerprint stays. |
| `lig_buried_sasa_mean_A2 / _std_A2` | Buried surface area of the ligand in the pocket (Å²). `lig_buried_sasa_std_A2` was the earlier Claim B candidate; **retracted** after 4A5S recovery (see NB 28). |

## Static topology / charges

| Column | Meaning |
|--------|---------|
| `protein_formal_charge` | Integer net charge of the protein at simulated protonation (per-target constant). |
| `protein_partial_charge` | Sum of protein partial charges from `.top` (should ≈ integer). |
| `active_site_formal_charge` | Integer net charge of the residues within 5 Å of the ligand at frame 0. |
| `active_site_partial_charge` | Same, but sum of partial charges. |
| `protein_n_titratable` | Count of ASH / GLH / HID / HIE / HIP / LYN / CYM residues in the protein — how many titratable positions there are. |
| `n_active_site_residues` | Count of residues within 5 Å of the ligand at frame 0. |
| `active_site_titratable_names__n` | Count of titratable residues in the active site (from the summary JSON). |

## Ligand chemistry (from `data/derived/ligand_chem.parquet`)

Computed from `system.top + system.gro` via RDKit — real chemistry, not the noisy topology
partial-charge sum.

| Column | Meaning |
|--------|---------|
| `lig_MW` | Molecular weight (Da). |
| `lig_n_heavy` | Heavy-atom count. |
| `lig_rot_bonds` | Rotatable bonds. |
| `lig_HBD / lig_HBA` | H-bond donor / acceptor count. |
| `lig_all_rings` | All rings; `lig_aromatic_rings` typically 0 in this dataset. |
| `lig_LogP` | Crippen logP. |
| `lig_TPSA` | Topological polar surface area. |
| `lig_fraction_sp3` | Fraction of sp3 carbons. |
| `lig_partial_q_abs_sum` | Sum of absolute partial charges — proxy for polarity distribution intensity. |

## Provenance / meta

| Column | Meaning |
|--------|---------|
| `target` | 4-character PDB code, e.g. `4QB3`. |
| `complex_id` | 32-hex-character hash of the (protein, ligand, force-field) tuple. Primary key together with `target`. |
| `is_active` | ChEMBL activity label (True / False) from the upstream study's `metadata.csv`. |
| `pchembl` | ChEMBL pKi / pIC50 — activity strength. |
| `docking_score` | Docking score used to place the ligand initially (kcal/mol; more negative = better). |
| `xtc_MB` | Size of the trajectory file, MB. |
| `n_frames_used_for_ts` | Number of frames sampled for the time-series metrics (default: 301 out of 15 001). |
| `traj_dt_ps / traj_n_frames` | Trajectory time-step (2 ps) and total frames (15 001). |
| `elapsed_seconds` | Wall time to compute one complex's features (perf diagnostic). |
