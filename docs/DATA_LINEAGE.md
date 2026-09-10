# Data lineage

Which raw file produces which derived table, and which notebook consumes it.

## Study-scope master list

| Path | Content |
|------|---------|
| `data/external/gbsa-study/data/raw/newbench_targets.csv` | **NewBench-27 master list** — 27 rows keyed on `pdb`. Columns: `pdb`, `family`, `chembl_id`, `uniprot`, `crystal_ligand`, `crystal_ligand_rscc`, `resolution_A`, `weakest_Ki_nM`, `split` (`discovery` \| `validation`). Discovery-9 (MD + GBSA computed) = `2XU3`, `3I06`, `4A5S`, `4L7G`, `4QB3`, `5HU9`, `8ELC`, `9D9I`, `9SI4`. The other 18 rows are the pre-registered validation cohort (MD/GBSA not computed yet). Consumed by `notebooks/01_scope_and_map.ipynb`. |

## Upstream — from `data/external/gbsa-study/` (symlink)

The symlink `data/external/gbsa-study` → `data/external/gbsa-study-original/` (snapshot of the
upstream MM-GBSA study used for this analysis). Prior to iter 3 a duplicate copy also lived at
the repo root (`gbsa-study-updated/`); it was moved inside `data/external/` and the root-level
duplicate deleted for a clean top-level.


| Path | Content |
|------|---------|
| `data/raw/gbsa_dG_raw.csv` | Per-(complex, GBSA combo) MM-GBSA ΔG. 270 × 48 = 12 960 rows (partial coverage). |
| `data/raw/metadata.csv` | Per-complex ligand ID, ChEMBL ref, `is_active`, `pchembl`, `docking_score`. |
| `data/derived/study2/bedroc20_partial.csv` | Per-(target, sp-config) BEDROC α=20 — reference target metric. |
| `data/derived/temporal/bedroc_all_combos_per_target.csv` | Per-(GBSA combo, target) BEDROC α=8/20/80 (48 combos × 9 targets = 432 rows). |
| `data/derived/temporal/temporal_bedroc_vs_time.csv` | Per-target BEDROC at truncated trajectory lengths (0.05–30 ns). |
| `data/derived/temporal/per_target_settling.csv` | Per-target minimum ns for BEDROC to stabilise. |

## Ours — under `data/raw/complex_analyses/`

One folder per (target, complex_id). Each contains:

| File | Content | Producer |
|------|---------|----------|
| `summary.json` | ~60 scalar features per complex (see GLOSSARY). | `reproduce/analyze_complex.py` |
| `timeseries.parquet` | 301-frame time series of all per-frame metrics. | ↓ |
| `hbond_timeseries.parquet` | Per-frame H-bond count (MDA HBA). | ↓ |
| `hbonds_persistence.tsv` | Per (donor, acceptor) H-bond persistence. | ↓ |
| `contacts_persistence.tsv` | Per-residue min-distance-based contact persistence. | ↓ |
| `ligand_rmsf.parquet` | Per-atom ligand RMSF. | ↓ |
| `protein_ca_rmsf.parquet` | Per-residue Cα RMSF + `is_active_site` flag. | ↓ |

Trajectory input (not in this repo): `system.gro`, `system.top`, `process/gromacs.xtc` from the
original discovery-9 productions on STORE.

## BayesOpt provenance — under `data/external/bayesopt/`

The MD `.mdp` used for every one of the 270 discovery-9 productions is the top-1 config from a
two-stage Optuna Bayesian-optimisation study over a 17-parameter GROMACS MDP + FF-tier search
space. The full provenance chain is vendored so a cold reader can audit every step without
leaving this repo.

| Path | Content | Provenance | Consumed by |
|------|---------|------------|-------------|
| `data/external/bayesopt/bayes_opt.db` | SQLite Optuna study DB, ~928 KB. Three studies: `smoke_5hu9_ligand01_v1` (1 trial), `md_prod_v1` (200 COMPLETE), `md_prod_v2` (33 COMPLETE + 142 FAIL — v2 re-opened the envelope, so grompp/mdrun failures on empirically-uncertain corners are the signal). Winner in `md_prod_v1` at trial 96, fitness 20.073. | Copied verbatim (2026-09-02) from `pipeline-mps/bayes_opt.db` on the source workspace. Read-only in this repo. | `notebooks/02_md_config_provenance.ipynb` |
| `data/external/bayesopt/bo_winner_v1.json` | Extracted top-1 config from `md_prod_v1` — the `.mdp` + FF triple used for the 270-set. 1.5 KB. `provenance` block records the study name, trial number, fitness, extraction UTC, and the (target, ligand) pair the fitness was measured on (5HU9 / ligand01). | Extracted by `pipeline-mps/src/pipeline_mps/bayes_opt/verify_top.py` from `bayes_opt.db`, then handed to the 270-set production launcher. | `notebooks/02_md_config_provenance.ipynb` cell 4. |
| `data/external/bayesopt/code/` | The 17-parameter search-space definition — `space_v2.py` (current), `space.py` (v1 that produced the winner), `objective.py` (fitness function), `optimize.py` / `optimize_v2.py` (Optuna study drivers), `verify_top.py` (winner extractor). ~59 KB total. | Source-tree copy from `pipeline-mps/src/pipeline_mps/bayes_opt/`. Read-only in this repo. | Read by NB 00 cell 3 to document the axes. |
| `data/external/bayesopt/report/BO_report.ipynb` + `.html` + `reviewer_notes/` | The reviewed BayesOpt report — 10 iterations × 5 reviewer rounds — that signed off on the winner and the search-space design. | Copied verbatim from `pipeline-mps/notebooks/`. Read-only. | Reader-reference; not auto-executed. |

**Verify chain.** Cell 5 of `notebooks/02_md_config_provenance.ipynb` opens `bayes_opt.db` in
read-only mode via `sqlite3` stdlib, enumerates the three studies, counts trials by state, and
asserts the JSON-recorded winner fitness (20.073) matches the DB best for `md_prod_v1`
(assertion passes: trial 96, fitness 20.073379).

**Three-tier MD data.** *Tier 1* is the 270-set analysed here. *Tier 2* is the BO search history
(`bayes_opt.db`, ~234 completed trials × ~20 ps). *Tier 3* is a planned winner-replica workspace
(8 × 30 × 3 replicas × 20 ns) at
`/mnt/netapp1/Store_othcxlwa/pipeline-mps-workspaces/discovery9_winner/` — not copied here yet;
out of scope for headline numbers.

## Aggregated tables — under `data/derived/`

| File | How it's produced | Consumed by |
|------|--------------------|-------------|
| `features.parquet` | `reproduce/aggregate_results.py` — flattens all 270 `summary.json`s into a wide table, joined to `metadata.csv`. | NB 03, 14–25, 28, 29 |
| `features.tsv` | TSV mirror of the parquet. | Manual inspection. |
| `ligand_chem.parquet` | `reproduce/ligand_chem_from_topology.py` — builds an RDKit Mol from `system.top`+`system.gro` per complex, computes 10 descriptors. | NB 27–29 |
| `deep_research_summary.csv` | `reproduce/deep_research.py` — panel BEDROC of every P1/P2/P3 model. | NB 25 |
| `deep_research_p1_per_target.csv` | ↓ per-target BEDROC of the P1 (target-fingerprint → best config) models. | NB 23 |
| `deep_research_p2_per_target.csv` | ↓ per-target of the P2 (MD → is_active) models + baselines. | NB 24, 25 |
| `deep_research_p3_per_target.csv` | ↓ per-target of the P3 (full-factorial ML) picks. | NB 24 |
| `gbsa_param_vs_md_correlations.csv` | `reproduce/gbsa_param_vs_md.py` — Spearman ρ (MD feature × GBSA parameter axis) with permutation p. | NB 26 |
| `gbsa_param_vs_md_v2_correlations.csv` | `reproduce/gbsa_param_vs_md_v2.py` — with ligand-chem features and BH-FDR-adjusted q. | NB 26 |
| `gbsa_param_vs_md_v2_{rho,q}_matrix.csv` | Full matrices for the NB 26 heatmap. | NB 26 |
| `gbsa_prediction_r_by_target.csv` | `gbsa_param_vs_md_v2.py` — per-target Pearson r for MD → GBSA regression (LOTO). | NB 27 |
| `md_surrogate_bedroc.csv` | Per-target BEDROC of the MD-surrogate score. | NB 27 |
| `single_feature_bedroc.csv` | Every candidate feature ranked by its panel BEDROC as a single-feature scorer. | NB 28, 29 |
| `hardened_claim_b.csv` | `reproduce/hardened_claim_b.py` — panel BEDROC of every candidate MD/chem feature under 7 conditions: `naive`, `mw_resid`, `size_resid`, `bound_only`, `pbc_clean`, `alt_label_7`, `alt_label_6`. Bootstrap CI (B=5000, target-level) + within-target permutation p. Excludes label-derived columns (`pchembl*`, `is_active*`, `docking_score`). | NB 28 |
| `hardened_claim_b_summary.md` | Human-readable rendering of `hardened_claim_b.csv` for the top features. | Reader reference. |
| `deep_research_wide.csv` | `reproduce/deep_research_wide.py` — wider LOTO ML sweep with in-fold `Pipeline(SimpleImputer + [StandardScaler] + estimator)`. Models: HistGradientBoosting, RandomForest, ExtraTrees, Ridge (tuned α), ElasticNet, SVM-RBF (tuned C/γ), LogReg-L1/L2. Bootstrap CI + permutation p on panel BEDROC. XGBoost/LGBM logged as unavailable in this pixi env. | NB 23 |
| `deep_research_wide_summary.md` | Human-readable rendering of `deep_research_wide.csv`. | Reader reference. |
| `pbc_qc.csv` | `reproduce/pbc_qc.py` — per-trajectory QC: `max_com_jump_A`, `max_drift_A`, `is_pbc_artifact` (COM jump > 6 Å), `is_probably_unbound` (drift > 8 Å). 270 rows. | NB 28 (via `hardened_claim_b.py` filter) |
| `canonical_baselines.csv` | `reproduce/canonical_baselines.py` — **single source of truth** for panel-BEDROC α=20 baselines (GBSA-locked, docking, plus top hardened Claim B context features). Recomputed from raw with a fixed 9-target panel and 4A5S-imputed-0 policy; 95% bootstrap CI over targets, B=5000. Run `--verify` to diff against on-disk values. | README, STUDY_DESIGN, docs, and every notebook that cites a baseline number. |

## The 4A5S label gap and its fix

`MANIFEST.tsv` on the STORE trajectory tree has EMPTY `is_active` and `pchembl` columns for all
30 rows of target **4A5S** (upstream ingestion issue — labels were never written into the
manifest for that target's protease-panel batch). Before the fix, every notebook that started
from `features.parquet` silently dropped 4A5S with `dropna(subset=['is_active'])`, shrinking
the panel from 9 targets to 8.

**Fix.** Both `reproduce/aggregate_results.py` and `reproduce/hardened_claim_b.py` now co-load
`gbsa-study/data/raw/metadata.csv`, which has the correct labels (10 actives / 20 decoys for
4A5S). When a `MANIFEST.tsv` row has NaN `is_active` we fall back to the metadata label.
Verified 2026-09-01:

    metadata 4A5S:              is_active   False = 20   True = 10
    features 4A5S (pre-fix):    is_active   None  = 30
    features 4A5S (post-fix):   is_active   False = 20   True = 10

Downstream consequences worth flagging:

- `hardened_claim_b.csv` now has `n_targets = 9` for the naive / mw_resid / size_resid /
  bound_only / pbc_clean conditions (rows keyed on features/complex_id only). For rows keyed
  on the GBSA score (the `gbsa_dG` baseline row, and the "Δ vs GBSA" intersection in every
  feature row) `n_targets` stays at 8 — 4A5S has no GBSA data upstream.
- Panel BEDROC of `lig_buried_sasa_std_A2` on the naive panel drops from **0.674 (8 tgt)** to
  **0.603 (9 tgt)** because 4A5S ranks poorly on that feature. It's still the naive top
  single-feature on the 9-target panel (see `data/derived/canonical_baselines.csv`). So the
  fix *weakens* Claim B further — the retraction stands. Two sum-of-partial-charge columns
  (`lig_partial_q_sum` in `ligand_chem.parquet`, `ligand_partial_charge_sum` in
  `features.parquet`) are numerically zero by construction and excluded from the naive ranking
  in `reproduce/canonical_baselines.py` — before that exclusion they produced a spurious BEDROC
  of 0.629 from floating-point roundoff.
- The docking baseline on the canonical 9-target panel is **0.516** [0.30, 0.73]; on the
  8-target GBSA subset it is **0.415**. Neither matches R1's 0.484 exactly because R1 used a
  slightly different label-mapping convention; the ordering (docking < GBSA-locked) holds under
  all conventions. Canonical GBSA-locked = **0.541** [0.36, 0.71] on the 9-target panel (4A5S
  imputed at 0), or 0.609 [0.48, 0.75] on the 8-target subset where GBSA has data. All
  canonical baseline values live in `data/derived/canonical_baselines.csv` (regenerated by
  `reproduce/canonical_baselines.py`).

## Upstream-origin notebooks — now flat under `notebooks/`

After the flatten, upstream and downstream NBs live in one flat tree with slots `00..30`. The
upstream-origin NBs (02, 04-13) still consume the upstream `gbsa-study` data tree directly and
use `src/gbsabench/` (whose `paths.py` still points at `data/external/gbsa-study/data/{raw,derived}/`
for RAW/DERIVED but now writes figures to a single flat `figures/` root — no more
`figures/upstream/` subtree). Downstream code (`src/discovery9/`) and upstream code
(`src/gbsabench/`) don't import each other; the two data roots stay parallel so an upstream
re-run doesn't perturb downstream outputs.

| NB | Consumes (from `data/external/gbsa-study/data/`) | Produces (under `figures/`) |
|-----|--------------------------------------------------|-----------------------------|
| `02_md_config_provenance`    | `data/external/bayesopt/bayes_opt.db`, `bo_winner_v1.json`, `code/space_v2.py` | `02_md_config_provenance_fig{1,2}.png` (fitness-vs-trial; top-10 param dominance) |
| `04_coverage`                | `raw/gbsa_dG_raw.csv`, `raw/metadata.csv`                | coverage barplot                              |
| `05_docking_baseline`        | `raw/metadata.csv`                                       | `05_docking_baseline_fig1.png`                |
| `06_gbsa_vs_docking`         | `derived/study2/bedroc20_partial.csv`, `raw/metadata.csv` | `06_gbsa_vs_docking_fig1.png`                 |
| `07_physics_importance`      | `derived/temporal/bedroc_all_combos_per_target.csv`      | `07_physics_importance_fig{1,2}.png`          |
| `08_doe_analysis`            | both factorial CSVs                                      | `08_doe_analysis_s{1,2}_*.{png,pdf}` (flat)   |
| `09_study2_gromacs_screening`| `raw/md_variants_*.csv`, `raw/md_productions_raw.csv`    | `09_study2_gromacs_screening_*.{png,pdf}` (flat, 9 plots) |
| `10_timestep_stability`      | `raw/md_variants_gbsa_scores_raw.csv`, `raw/md_variants_manifest_raw.csv` | stability figure                     |
| `11_timestep_throughput`     | `raw/md_speed_smoke_L40S.csv`, `raw/md_variants_prod_perf_raw.csv` | throughput figure                        |
| `12_temporal_convergence`    | `derived/temporal/temporal_bedroc_vs_time.csv`, `derived/temporal/per_target_settling.csv` | `12_temporal_convergence_*.{png,pdf}` (flat) |
| `13_selection_correction`    | `derived/temporal/bedroc_all_combos_per_target.csv`      | `13_selection_correction_fig1.png`            |

### Upstream reproduce scripts — `reproduce/upstream/`

| Script | Purpose |
|--------|---------|
| `bedroc_doe.py`             | Recompute per-(target, GBSA-combo) BEDROC α=8/20/80 (writes `derived/temporal/bedroc_all_combos_per_target.csv`). |
| `bedroc_param_scatter.py`   | Per-combo BEDROC scatter used by NB 06. |
| `factorial_gbsa.py`         | The 4×3×2×2 GBSA factorial recipe generator. |
| `gen_variant_mdp.py`        | Emit GROMACS `.mdp` files for the Taguchi L27 GROMACS-config screen. |
| `md_configs.py`             | Config catalogue for study 2 (variant labels + parameter tuples). |
| `temporal_convergence.py`   | Truncated-trajectory BEDROC sweep for NB 12. |
| `run_full.sbatch`, `run_variant.sbatch`, `submit_full.sh`, `submit_smoke.sh` | SLURM launchers. |

### Upstream package — `src/gbsabench/`

- `paths.py` — anchors on the upstream data root at `data/external/gbsa-study/data/`. All
  upstream-origin NBs `from gbsabench.paths import RAW, DERIVED, FIGURES`. After the flatten,
  `FIGURES` = `figures/` (flat), same as `discovery9.paths.FIGURES`.
- `io.py`, `metrics.py`, `style.py`, `benchmark_quality.py` — the upstream helpers.

### Figures land under `figures/` (flat)

After the flatten, the upstream `figures/upstream/{doe,study2,temporal}/` subtree was removed;
every figure now lives at `figures/{NN}_{slug}_...{png,pdf}` prefixed by the owning NB slot.
`figures/SI/` (per-target drift PDFs) is a separate tree, unaffected.

## Notebook → figure lineage

Every notebook exports its figures to `figures/{NN}_{slug}_figK.png`. The final cell of each
notebook holds the export step. Regenerating a figure: re-run the notebook that owns it.

NB 01 (`01_scope_and_map`) exports `figures/01_scope_and_map_fig{1,2}.png` — the scope grid and
the 27 × 3 status heatmap.
