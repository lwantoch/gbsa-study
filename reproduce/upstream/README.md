# Reproduce — how the raw CSVs were produced

These are the exact submission scripts and configs that ran on the cluster. They are shipped for
provenance and re-execution on an equivalent SLURM + GROMACS 2026.0 CUDA + gmx_MMPBSA environment; the
study's figures/tables do NOT need them (the shipped `data/raw/` CSVs are sufficient — see the top-level
README "Reproduce it").

| File | Produces |
|------|----------|
| `md_configs.py` | the Taguchi L27 speed-parameter array (dt × MTS × rcoulomb × gap × nstlist) |
| `gen_variant_mdp.py` | per-config GROMACS MDP files from `md_configs.py` |
| `run_variant.sbatch` / `submit_full.sh` | the L27 × 270-complex MD productions → `md_productions_raw.csv` |
| `submit_smoke.sh` | the controlled single-GPU L40S smoke (identical 50 000 steps) → `md_speed_smoke_L40S.csv` |
| `run_full.sbatch` | full production driver |
| `factorial_gbsa.py` | scores a production trajectory across the 48 MM-GBSA combos → `gbsa_dG_raw.csv` |

Engine scope: GROMACS 2026.0 CUDA only (version not varied). See `../PIPELINE.md`.
