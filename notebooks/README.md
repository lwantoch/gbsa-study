# gbsa-study — notebooks index

Notebooks follow the STUDY_DESIGN experiment structure: **Phase A** ran on OHDS,
**Phase B** runs on CESGA FT3. Reading order matches the two-digit prefix within each
subdirectory.

Every notebook is self-contained: it imports paths from `gbsabench.paths` (repo-relative,
subdir-robust), loads raw data from `data/raw/` or `data/external/`, and exports figures
to `figures/{NN}_{slug}_figK.png`. Live-SLURM aggregation lives ONLY in
`../reproduce/aggregate_live_gbsa.py`; notebooks never touch cluster workspaces directly.

Kernel: **`gbsa-study`** (env at `../env/`, or plain `pip install -e ..`).

---

## `00_orientation/` — start here

| # | Notebook | Purpose |
|---|---|---|
| 00 | [`00_data_catalog`](00_orientation/00_data_catalog.ipynb) | Single entry point — where every data artefact lives |
| 01 | [`01_scope_and_map`](00_orientation/01_scope_and_map.ipynb) | NewBench-27 scope grid + research-question map |
| 02 | [`02_dataset_overview`](00_orientation/02_dataset_overview.ipynb) | Ligand / target counts + activity balance |
| 03 | [`03_coverage`](00_orientation/03_coverage.ipynb) | Coverage of the (target × ligand) matrix |

## `A1_full_quality_pipeline/` — the expensive reference (Experiment A1)

| # | Notebook | Purpose |
|---|---|---|
| 10 | [`10_docking_baseline`](A1_full_quality_pipeline/10_docking_baseline.ipynb) | Docking baseline BEDROC (cheap alternative to any GBSA-ranker) |
| 11 | [`11_gbsa_vs_docking`](A1_full_quality_pipeline/11_gbsa_vs_docking.ipynb) | Full-quality GBSA vs docking per-target (Kendall τ, BEDROC α=20) |
| 12 | [`12_bedroc_baselines`](A1_full_quality_pipeline/12_bedroc_baselines.ipynb) | Canonical 3-bar reference (docking / GBSA / MD-composite) |

## `A2_gbsa_factorial/` — can we cheapen the GBSA physics? (Experiment A2)

| # | Notebook | Purpose |
|---|---|---|
| 20 | [`20_physics_importance`](A2_gbsa_factorial/20_physics_importance.ipynb) | Variance decomposition (η²) over the 48 GBSA combos |
| 21 | [`21_doe_analysis`](A2_gbsa_factorial/21_doe_analysis.ipynb) | NIST-sequence DOE cross-check of both factorials |

## `A3_per_ligand_features/` — can features replace GBSA? (Experiment A3)

| # | Notebook | Purpose |
|---|---|---|
| 30 | [`30_per_complex_deep_dive`](A3_per_ligand_features/30_per_complex_deep_dive.ipynb) | Full 8-panel per-complex deep dive |
| 31 | [`31_per_target_aggregation`](A3_per_ligand_features/31_per_target_aggregation.ipynb) | Per-target aggregation of complex features |
| 32 | [`32_feature_distributions`](A3_per_ligand_features/32_feature_distributions.ipynb) | Split-violin actives vs measured non-binders per feature |
| 33 | [`33_kinematic_phase_space`](A3_per_ligand_features/33_kinematic_phase_space.ipynb) | Drift vs COM-displacement quadrant map |
| 34 | [`34_time_series_overlay`](A3_per_ligand_features/34_time_series_overlay.ipynb) | 30-ligand-drift trajectory overlays per target |
| 35 | [`35_active_site_fingerprint`](A3_per_ligand_features/35_active_site_fingerprint.ipynb) | Residue × complex contact-persistence heatmap |
| 36 | [`36_feature_correlation`](A3_per_ligand_features/36_feature_correlation.ipynb) | Within-target Pearson-r block heatmap |
| 37 | [`37_actives_vs_non_binders_separation`](A3_per_ligand_features/37_actives_vs_non_binders_separation.ipynb) | Actives vs measured non-binders separation per feature |
| 38 | [`38_ml_combo_selection`](A3_per_ligand_features/38_ml_combo_selection.ipynb) | **Q1** — LOTO combo-selection (best-of-48 per target) |
| 39 | [`39_full_factorial_ml`](A3_per_ligand_features/39_full_factorial_ml.ipynb) | **Q1** — full-factorial ML meta-model |
| 40 | [`40_gbsa_param_vs_md`](A3_per_ligand_features/40_gbsa_param_vs_md.ipynb) | Spearman ρ MD signatures vs GBSA-parameter axes (BH-FDR) |
| 41 | [`41_ligand_chem_gbsa_surrogate`](A3_per_ligand_features/41_ligand_chem_gbsa_surrogate.ipynb) | **Q1** — per-complex GBSA-ΔG surrogate (3 honesty tiers) |
| 42 | [`42_single_feature_bedroc`](A3_per_ligand_features/42_single_feature_bedroc.ipynb) | **Q2** — single-feature ranker sweep + hardening |
| 43 | [`43_rank_fusion_deployable`](A3_per_ligand_features/43_rank_fusion_deployable.ipynb) | **Q2** — LOTO K-fusion sweep for deployable ranker |
| 44 | [`44_family_stratification`](A3_per_ligand_features/44_family_stratification.ipynb) | Per-family panel BEDROC with bootstrap CI |
| 45 | [`45_target_footprint_diagnostic`](A3_per_ligand_features/45_target_footprint_diagnostic.ipynb) | Univariate signal-vs-noise diagnostic (Q1 / Q2 / Q3) |

## `A4_bayesopt_mdp/` — can we cheapen the MD? (Experiment A4)

| # | Notebook | Purpose |
|---|---|---|
| 50 | [`50_md_config_provenance`](A4_bayesopt_mdp/50_md_config_provenance.ipynb) | Which BayesOpt trial produced which MDP (provenance) |
| 51 | [`51_study2_gromacs_screening`](A4_bayesopt_mdp/51_study2_gromacs_screening.ipynb) | L27 GROMACS-config screen + ranking robustness |
| 52 | [`52_timestep_stability`](A4_bayesopt_mdp/52_timestep_stability.ipynb) | Timestep stability + LINCS blow-up fail-rate |
| 53 | [`53_timestep_throughput`](A4_bayesopt_mdp/53_timestep_throughput.ipynb) | Nominal vs real MD throughput (ns/day) |
| 54 | [`54_temporal_convergence`](A4_bayesopt_mdp/54_temporal_convergence.ipynb) | Minimum trajectory length to keep BEDROC stable |
| 55 | [`55_bo_md_report`](A4_bayesopt_mdp/55_bo_md_report.ipynb) | BayesOpt run report (200 v1 + 34 v2 trials, top-5 winners) |

## `B1_reproduce_baseline/` — cross-server reproduction on FT3 (Experiment B1)

| # | Notebook | Purpose |
|---|---|---|
| 60 | [`60_ohds_ft3_reproduction`](B1_reproduce_baseline/60_ohds_ft3_reproduction.ipynb) | **LIVING** — tri-source repro (OHDS ref / our xchk / our 3-replica) |

## `B2_bo_verify_production/` — cohort-scale verification (Experiment B2)

Top-5 BO winners deployed at 3600-chain scale on FT3. The final analysis notebook (70+)
lands here as chains complete.

## `aux/` — supporting analyses and tutorial

| # | Notebook | Purpose |
|---|---|---|
| 90 | [`90_selection_correction`](aux/90_selection_correction.ipynb) | Winner's-curse max-T permutation correction |
| 91 | [`91_deep_research_verdict`](aux/91_deep_research_verdict.ipynb) | Deep-research verdict (kept for provenance, superseded) |
| 92 | [`92_ml_walkthrough_tutorial`](aux/92_ml_walkthrough_tutorial.ipynb) | Pedagogical walkthrough of ML methods |

## `bo_ancillary/` — reviewer notes on the BayesOpt run

Iteration diffs + reviewer notes that companion `55_bo_md_report`. Not executable.

---

## Reproducibility

Every notebook reads only from `data/raw/` and `data/external/` — no live SLURM workspaces,
no hardcoded absolute paths. Refresh the live-source CSVs (only from FT3):

```bash
python ../reproduce/aggregate_live_gbsa.py           # snapshots live SLURM output → data/raw/*.csv
```

Then re-execute a notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/{SUBDIR}/{NN}_*.ipynb
```
