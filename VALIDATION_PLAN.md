# Validation plan

Short, human-facing statement of what the discovery-9 pilot validated, what it didn't, and what
a next compute slot would need to run. It's the target of the "discovery stage" banner
cross-references in NB 04, 05, 06, 07 and 13.

## Scope

The full cohort is **NewBench-27**: 9 discovery targets (MD + MM-GBSA already computed) and 18
validation targets (pre-registered, MD/GBSA **not computed yet**; held out as the external
cohort). Master list: `data/external/gbsa-study/data/raw/newbench_targets.csv` (`split ==
'discovery'` vs `split == 'validation'`). Every headline number in this repo is on the 9
discovery targets and is pilot-level. Bootstrap CIs are wide enough that no downstream user
should treat point estimates as production settings.

## What was validated

- **Claim A — scoped.** None of the ML combo-selection pipelines (Ridge, RandomForest,
  ExtraTrees, HistGradientBoosting, ElasticNet, SVM-RBF, LogReg-L1/L2) beat GBSA-locked at
  panel level on the two families with n ≥ 2 (proteases n=4, kinases n=2). See NB 23 + 25 + 30.
  Minimum detectable effect at 80% power:
    - Ridge / RandomForest — paired σΔ = 0.093 → MDES = +0.11 BEDROC (Truchon-Bayly
      early-enrichment metric).
    - SVM-RBF — paired σΔ = 0.317 → MDES = +0.37 BEDROC.
- **Claim B — retracted.** After 4A5S recovery, the naive top single MD feature is
  `lig_buried_sasa_std_A2` at panel BEDROC 0.603 [0.39, 0.78] on the canonical 9-target panel
  (was 0.674 on the older 8-target subset that silently dropped 4A5S). Both fall inside the
  GBSA-locked bootstrap CI on n=9. The feature collapses under MW-residualisation (MW regressed
  out per target), PBC-cleaning, and bound-only filtering — multiple hardened conditions drop
  it below 0.51. See NB 28 + 29 + `data/derived/hardened_claim_b.csv`.

## What remains open

- **18 validation-target MD + GBSA — not computed yet.** The full NewBench-27 external cohort
  needs 18 × 30 production trajectories at the BayesOpt-winner `.mdp`
  (`data/external/bayesopt/bo_winner_v1.json`) plus per-complex MM-GBSA rescoring under the
  locked combo `igb2_di4_salt0.15_st0.0072`. Not scheduled — waiting on compute.
- **Tier-3 confirmatory production — not copied yet.** The planned 8 × 30 × 3 replicas from
  the BayesOpt winner live at
  `/mnt/netapp1/Store_othcxlwa/pipeline-mps-workspaces/discovery9_winner/` and aren't in this
  repo. This tier would let us tighten per-target CIs by resampling replicas rather than
  targets.
- **Family n-lift.** Kinase family reaches n=10 only after computing the 8 validation kinases;
  protease reaches n=5 after adding 7D5B; the four singleton families in discovery
  (bromodomain, chaperone, other, plus every unique validation family) stay singletons or n ≤ 2
  until validation compute lands.
- **DUD-E / LIT-PCBA (optional).** If the 18-target validation cohort turns out under-powered
  for some family, DUD-E MK14 (for 8ELC) and LIT-PCBA JNK-family (kinase ceiling sanity check)
  are the two cheapest external cross-checks. Both optional and only worth running once the
  internal validation is done.

## Reader guidance

1. Start with `README.md` and `STUDY_DESIGN.md` for headline numbers and cohort.
2. Read `notebooks/00_data_catalog.ipynb` for the grep-able index of every artifact in the repo,
   then `notebooks/01_scope_and_map.ipynb` for the 27-target scope grid and research-question
   map.
3. Follow the 4-act narrative from `notebooks/03_dataset_overview.ipynb` through
   `notebooks/30_family_stratification.ipynb` in slot order. Act boundaries are marked in the
   notebook titles; short bridges sit at 06→07, 13→14 and 21→22.

## Governance

This pilot is **not confirmatory**. All numbers are pilot-level with wide bootstrap CIs
(target-level resample, n=9). No number here should be quoted as a production setting without
the pre-registered 18-target validation completing first. Bootstrap regime for canonical
baselines is B=5000, seed 20260902; NB 30 and `deep_research_wide.py` use B=1000 for compute
speed (documented per-notebook).
