# Nomenclature register

Single source of truth for how each concept is *named* in this repo. Every entry has
**one canonical form** — every doc, notebook, and narrative comment should use that form.
"Acceptable aliases" lists the spellings you'll find in the wild; "reason to prefer" says
why the canonical is canonical.

This is a **naming reference**, not a scientific glossary. For plain-English definitions,
see `docs/GLOSSARY.md`. When the two disagree on wording, this file wins for naming; the
glossary wins for definitions.

Scope: **code identifiers** (column names in `features.parquet`, kwargs in `reproduce/*.py`,
keys in on-disk artifacts) are immutable. This register only governs how those identifiers
are *referred to* in prose and markdown cells. If you need to talk about column `alt_label_7`
in prose, spell it as the CSV column literal `alt_label_7` — don't rewrite the underscore.

---

## 1. Panels and cohorts

| Concept | Canonical | Acceptable aliases (in the wild) | Reason to prefer canonical |
|---|---|---|---|
| The 9-target computed panel | **Discovery-9** (nine hyphen nine, capitalised D) | `discovery-9`, `discovery9`, `discovery 9`, `newbench-9`, `9-target discovery panel`, `9-target discovery set` | Matches the repo name (`gbsa-study`), the src package (`src/discovery9/`), and the "Discovery-9 vs NewBench-27" glossary entry. `newbench-9` is a historical alias from the "Discovery stage" banner in NB 04/05/06/07/13 and should NOT survive — it collides with "NewBench-27 = Discovery-9 + Validation-18". |
| The full study cohort | **NewBench-27** (capital N, capital B, hyphen, 27) | `newbench-27`, `NewBench27`, `newbench27`, `27-target NewBench cohort`, `n=27` | Same form the upstream master list uses (`newbench_targets.csv`); every other spelling drifts. |
| The 18 pre-registered but uncomputed targets | **Validation-18** | `validation cohort`, `validation set`, `held-out 18`, `18 validation targets`, `n=18 validation`, `validation-18` (lower-v) | Mirrors "Discovery-9" — reads as a matched pair. |
| Panel BEDROC averaged across all 9 discovery targets, with 4A5S imputed at 0 where a scorer has no data | **9-target panel** (BEDROC value is called the "9T panel value" or "canonical 9-target value") | `9T`, `9T panel`, `9-target discovery panel`, `full 9-target panel`, `panel of 9`, `n=9 panel`, `9-target scale` | Full "9-target panel" reads unambiguously. `9T` is OK shorthand inside narrow table columns — never in prose. |
| The 8-of-9 subset where GBSA-locked has data (excludes 4A5S) | **8-target subset** | `8T`, `8T subset`, `8-target subset`, `8-target-subset`, `8T-subset`, `8-target subpanel`, `naive 8-target panel` | Same rule: full name in prose, `8T` only in narrow columns. "Subpanel" is unnecessary jargon. |
| The word "panel" alone | **Panel** always means the 9-target panel (with 4A5S imputed at 0). **Subset** always means the 8-target subset (no imputation). | — | Glossary fixes this too; every NB that violates it earns a fix. |

## 2. Baselines

| Concept | Canonical | Acceptable aliases | Reason to prefer canonical |
|---|---|---|---|
| The single MM-GBSA combo `igb2_di4_salt0.15_st0.0072` used as the beat-me baseline | **GBSA-locked** (write out the combo literal `igb2_di4_salt0.15_st0.0072` on first mention per doc, then "GBSA-locked" everywhere else) | `GBSA-locked combo`, `locked GBSA combo`, `locked GBSA`, `canonical GBSA`, `canonical GBSA-locked`, `GBSA @ locked combo`, `GBSA baseline`, `panel-locked combo`, `panel recipe` | What the glossary and `canonical_baselines.csv` both call it. Avoid "canonical GBSA" — say "canonical baseline value" for the CSV, "GBSA-locked" for the ranker. |
| Docking-only ranker (upstream pose-seed score) | **Docking baseline** (write the column name `docking_score` if you mean the column) | `docking score` (as a ranker name), `docking-only baseline`, `docking baseline` | Straightforward; already consistent. |
| Best single-feature ranker on the naive 9-target panel | **naive top single-feature** (name the feature literal, e.g. `lig_buried_sasa_std_A2`) | `top single-feature`, `naive single-feature`, `single-feature ranker`, `best MD feature` | The "naive" qualifier reminds the reader this is BEFORE hardening. Without it the number looks stronger than it is. |
| The CSV that pins every headline value | **`data/derived/canonical_baselines.csv`** ("the canonical baselines table"); each individual value is a **canonical baseline value** | `single source of truth for baselines`, `canonical CSV`, `canonical panel values` | Already the naming used everywhere — this entry just fixes it. |

## 3. Feature and MD-parameter names

| Concept | Canonical | Acceptable aliases | Reason to prefer canonical |
|---|---|---|---|
| A single MM-GBSA parameter tuple (one of 48: `igb × intdiel × saltcon × surften`) | **sp-config** (or spell out **GBSA combo** where the reader might confuse it with the Study-2 GROMACS-config sweep) | `sp_config`, `sp-configuration`, `GBSA combo`, `GBSA-combo`, `GBSA parameter combo`, `combo` | `sp-config` is the glossary spelling. Genuine ambiguity: in Study-2 GROMACS-config, `sp00 / sp08 / sp25` are labels for GROMACS `.mdp` files, not GBSA combos. Never write bare `sp08` outside Study-2 without saying so. |
| The winning GROMACS `.mdp` used for the 270 discovery productions | **BayesOpt winner** or **`bo_winner_v1.json`** (the file); the underlying Optuna study is **`md_prod_v1` trial 96** | `winner config`, `BO winner`, `top-1 BO config`, `BayesOpt v1 winner`, `md_prod_v1 winner` | Two levels: the *file* (`bo_winner_v1.json`) and the *trial* (`md_prod_v1` trial 96). Both are load-bearing — quote both on first mention per doc. |
| The frozen force-field triple used for every trial and every production | **FF-Tier-1** = `ff14SB + gaff2 + tip3p + vdw_modifier=Potential_Shift` | `FF Tier 1`, `FF-tier-1`, `Tier-1 FF`, `FF tier 1 freeze`, `frozen FF`, `AMBER-ff14SB / GAFF2 / TIP3P` | Matches the Optuna study config comment; hyphenated, capital T. |
| The 17-parameter GROMACS-2025 MDP + FF-tier search space | **17-parameter search space** (source: `data/external/bayesopt/code/space_v2.py`) | `17-param search`, `17-axis search`, `GROMACS-MDP + FF-tier space` | Exact count is load-bearing — don't round to "~17" or "seventeen-ish". |
| MD parameter literals inside prose | Use the GROMACS keyword as-is: `integrator`, `dt`, `tcoupl`, `pcoupl`, `constraints`, `LINCS order`, `HMR`, `MTS`. | `time step` (for `dt`), `LINCS-order`, `heavy-hydrogen mass repartition` | Match the `.mdp` keyword. In numeric context, `dt=2 fs` is canonical. |

## 4. Methodology terms

| Concept | Canonical | Acceptable aliases | Reason to prefer canonical |
|---|---|---|---|
| The negative result about ML combo-selection pipelines | **Claim A** (capital C, capital A) — always paired with a modifier: "retained, scoped" | `claim A`, `claim a`, `Claim-A` | Bare `Claim A` on later reference is fine; first mention per doc should give the modifier. |
| The retracted claim about a single MD feature beating GBSA | **Claim B** (paired with "retracted") | `claim B`, `Claim-B` | Same rule. Both claims are proper nouns here. |
| The seven-condition robustness sweep in NB 28 | **hardening** (the sweep is "the hardened comparison" or "the seven hardened conditions"); a claim that survives is **hardened** | `hardening pass`, `robustness sweep`, `hardening audit` | Glossary term. The seven conditions are literal CSV column values: `naive`, `mw_resid`, `size_resid`, `bound_only`, `pbc_clean`, `alt_label_7`, `alt_label_6` — always spell those with the underscores and lowercase from the CSV. |
| Per-target OLS regression on molecular weight | **MW-residualised** (British `-ised`) | `MW-residualized` (US spelling), `MW residualisation`, `MW-residualisation`, `MW-residual` | Repo prose settles on `-ised` in README / VALIDATION_PLAN / NB 28; the glossary term used `-ized`. Genuine inconsistency; canonical is `-ised` to match the majority. |
| Per-target OLS regression on MW + heavy atoms + TPSA | **size-residualised** | `size-residualized`, `size residualisation` | Same rule. |
| Target-level resample of BEDROC values | **bootstrap** (target-level, B=5000, seed 20260902 for canonical results; B=1000, seed 20250901 for NB 30 and `deep_research_wide.py`) | `bootstrapping`, `target-level bootstrap`, `bootstrap CI` | Seed and B are load-bearing — quote them where reproducibility matters. |
| Benjamini–Hochberg false-discovery-rate correction | **BH-FDR** (uppercase, hyphen; q<0.10 threshold) | `BH FDR`, `Benjamini-Hochberg`, `BH correction` | Standard abbreviation; spell out on first mention if the reader is unfamiliar. |
| Leave-one-target-out cross-validation | **LOTO** (all caps) — the K-fusion variant is `LOTO K=N` | `Leave-One-Target-Out`, `leave-one-target-out`, `LOO-CV`, `LOO cross-validation` | `LOTO` is the glossary term. First mention per doc, spell out `LOTO (leave-one-target-out)`. Never write `LOO` — this repo does leave-one-*target*-out, not leave-one-*complex*-out; confusing them is a scientific error. |
| Panel-composition variants used in the hardening pass | CSV column literals: **`alt_label_7`**, **`alt_label_6`** | `alt-label-7`, `alt-label-6`, `alt label 7` | Match the on-disk `hardened_claim_b.csv` column values. The glossary bullet uses hyphens (`alt-label-7`) — out of register, gets a fix. |
| Max-T selection-correction test | **max-T permutation test** (5000 within-target permutations) | `maxT test`, `max-T sign-flip`, `max-T label-permutation test` | The `maxt_selection_corrected_p_signflip` function name is load-bearing; prose spelling is `max-T`. |

## 5. ML pipelines

| Concept | Canonical | Acceptable aliases | Reason to prefer canonical |
|---|---|---|---|
| The full model panel from the wider sweep (`reproduce/deep_research_wide.py`) | **Ridge** (tuned α), **RandomForest**, **ExtraTrees**, **HistGradientBoosting**, **ElasticNet**, **SVM-RBF** (tuned C/γ), **LogReg-L1** and **LogReg-L2** — quote them in this order on first list per doc | `Random Forest` (space), `RF`, `HGBT`, `HGB`, `SVM RBF`, `SVM_RBF`, `Elastic Net`, `Ridge Regression`, `LR L1`, `LR L2`, `LR-L1/L2`, `Logistic-L1`, `LogReg L1/L2` | Camel-case one-word class names match scikit-learn attribute names. **`HGBT` is common shorthand for `HistGradientBoosting`, OK in tables**, never for first mention. |
| The three P-protocols | **P1**, **P2**, **P3** — always spell out on first mention: **P1** (per-target combo selection), **P2** (fingerprint → is_active), **P3** (full-factorial meta-model) | `Protocol P1`, `task P1`, `protocol-P1` | Glossary defines them; the first-mention rule fixes reader confusion. NB 23 introduces them well; NB 25/24/30 refer to them by number alone. |
| The narrow four-model sweep in `reproduce/deep_research.py` | **narrow sweep** (models: RidgeCV, RandomForest, HGBT, mean-baseline / constant classifier / logistic) | `narrow ML sweep`, `4-model sweep`, `deep-research narrow` | Distinguishes from the "wide sweep" (7-model) — the two answer different questions on different panels (NB 25's reconciliation note explains). |
| The wider seven-model sweep in `reproduce/deep_research_wide.py` | **wide sweep** | `wide ML sweep`, `wider sweep`, `7-model sweep`, `wide-sweep P1` | Same rule. |

## 6. File and artifact names

Refer to on-disk artifacts by their exact filename in backticks — no paraphrases.

| Concept | Canonical | Notes |
|---|---|---|
| Baseline reconciliation table | **`data/derived/canonical_baselines.csv`** — produced by `reproduce/canonical_baselines.py` | Called "the canonical baselines table" in prose. |
| Hardened Claim B sweep | **`data/derived/hardened_claim_b.csv`** — produced by `reproduce/hardened_claim_b.py`; human summary in `hardened_claim_b_summary.md` | "hardened conditions" refers to its seven `naive / mw_resid / size_resid / bound_only / pbc_clean / alt_label_7 / alt_label_6` rows. |
| Wide-sweep ML results | **`data/derived/deep_research_wide.csv`** — produced by `reproduce/deep_research_wide.py`; human summary in `deep_research_wide_summary.md` | |
| Narrow-sweep ML results | **`data/derived/deep_research_summary.csv`** + `deep_research_p1_per_target.csv` + `deep_research_p2_per_target.csv` + `deep_research_p3_per_target.csv` — produced by `reproduce/deep_research.py` | |
| Per-complex MD feature table | **`data/derived/features.parquet`** (TSV mirror: `features.tsv`) — produced by `reproduce/aggregate_results.py` | |
| Per-complex ligand chemistry from RDKit | **`data/derived/ligand_chem.parquet`** — produced by `reproduce/ligand_chem_from_topology.py` | |
| Upstream MM-GBSA ΔG matrix | **`data/raw/gbsa_dG_raw.csv`** | Lives under the upstream `gbsa-study` tree accessed via the `data/external/gbsa-study/` symlink. |
| Per-trajectory PBC / drift QC | **`data/derived/pbc_qc.csv`** — produced by `reproduce/pbc_qc.py` | Consumed by `hardened_claim_b.py` for the `pbc_clean` condition. |
| Per-(target, GBSA-combo) BEDROC | **`data/derived/temporal/bedroc_all_combos_per_target.csv`** (48 combos × 9 targets) | Upstream artifact. |
| Per-(target, sp-config) BEDROC | **`data/derived/study2/bedroc20_partial.csv`** | The sp-config here is the Study-2 sense (GROMACS-config), not the GBSA combo. |
| Single-feature ranker sweep | **`data/derived/single_feature_bedroc.csv`** | Consumed by NB 28 / 29. |
| BayesOpt Optuna DB | **`data/external/bayesopt/bayes_opt.db`** — three studies: `smoke_5hu9_ligand01_v1`, `md_prod_v1` (200 COMPLETE), `md_prod_v2` (33 COMPLETE + 142 FAIL) | Read-only in this repo. |
| BayesOpt winner config | **`data/external/bayesopt/bo_winner_v1.json`** — extracted top-1 from `md_prod_v1` trial 96 (fitness 20.073, target 5HU9 / ligand01) | Read-only in this repo. |

## 7. Study identifiers

| Concept | Canonical | Acceptable aliases | Reason to prefer canonical |
|---|---|---|---|
| The upstream MM-GBSA physics factorial (`igb × intdiel × saltcon × surften`) | **Study 1** (or **GBSA physics factorial**) | `study 1`, `Study-1`, `Study I`, `GBSA sensitivity study` | NB 08 and NB 09 use "Study 1" and "Study 2" — stick with those. |
| The Taguchi L27 GROMACS-config screen | **Study 2** (or **GROMACS-config screen** / **Taguchi L27**) | `study 2`, `Study-2`, `Study II`, `GROMACS screening study`, `GROMACS-DOE` | Same rule. Don't abbreviate to `S1` / `S2` in prose. |
| The Optuna Bayesian-optimisation of the MD `.mdp` | **BayesOpt** (proper noun, capitalised) — refers to the vendored artifacts under `data/external/bayesopt/`; underlying algorithm is Optuna TPE | `Bayes-opt`, `bayes opt`, `BO` (only after first spelled-out use), `Optuna`, `Optuna BO`, `Bayesian-optimisation study`, `two-stage BayesOpt` | The lowercase directory (`bayesopt/`) is the on-disk layout; the concept is spelled `BayesOpt`. `BO` is OK as an in-context abbreviation once "BayesOpt" has been introduced. |
| The two BayesOpt studies | **`md_prod_v1`** (200 COMPLETE, produced the winner) and **`md_prod_v2`** (33 COMPLETE + 142 FAIL, envelope expansion) — always in backticks; the smoke study is **`smoke_5hu9_ligand01_v1`** | `md_prod v1`, `BayesOpt v1`, `Stage 1`, `Stage 2`, `BO v2`, `BayesOpt round 2` | Literal Optuna `study_name` values — quoting them exactly is what makes the SQLite queries in NB 02 reproducible. |
| The three MD data tiers | **Tier 1** (the 270-set analysed here), **Tier 2** (BO search history, ~234 trials × ~20 ps), **Tier 3** (planned winner-replica, 8 × 30 × 3 × 20 ns, not in repo yet) | `tier 1`, `tier-1`, `Tier-1`, `three-tier` | Capital T, space, arabic numeral. `FF-Tier-1` in § 3 uses hyphens as a compound modifier; the standalone MD-data tier is `Tier 1`. |

## 8. Time markers

| Concept | Canonical | Acceptable aliases | Reason to prefer canonical |
|---|---|---|---|
| A revision cycle of the whole study | **iter-N** (lowercase `iter`, hyphen, arabic numeral: `iter-3`, `iter-4`, `iter-5`, `iter-6`, `iter-7`) | `iter N`, `iter_N`, `iteration N`, `Iter-N`, `iter3` (no separator) | Matches `reviews/iter_NN/` directory naming (underscore is a filesystem convention) and what every NB uses in prose. |
| A specific reviewer / fix reference | **iter-N FIX K** (arabic K; used in prose to point at a specific patch) — e.g. `iter-3 FIX 8` recovered the 4A5S labels | `iter 3 FIX 8`, `Iter-3 FIX 8`, `iter-3 fix 8` | Uppercase FIX; arabic K; no punctuation between `iter-N` and `FIX K`. |
| The review-directory names | **`reviews/iter_NN/`** (underscore, two-digit zero-padded) | `reviews/iter-NN/` | Filesystem convention — don't "correct" the underscore to match the prose form. |
| An absolute date in narrative prose | **YYYY-MM-DD** ISO format (e.g. `2026-09-02`) | `Sep 2, 2026`, `02/09/2026` | Sorts and greps cleanly. |

## 9. Cross-references to notebooks and internal sections

| Concept | Canonical | Notes |
|---|---|---|
| A pointer to another notebook | **`NN_slug`** in backticks (e.g. `28_single_feature_bedroc`) — full slug on first mention; **`NB NN`** as shorthand once the slot is unambiguous | Don't write `NB NN` where the number doesn't correspond to the current flat-tree slot — see the flatten renumbering in `reviews/iter_07/R6_nomenclature.md`. |
| An internal section within a notebook | **`§ N`** (section symbol, space, arabic — preserved from the pre-flatten monolith and matches the notebook's own `##` heading) | These numbers are 1..20 and do NOT match the notebook slot numbers 00..30. E.g. NB 27 = `27_ligand_chem_gbsa_surrogate` internally has section headings numbered `§ 17`. Stable historical convention — don't renumber. |
| A pointer to a figure | **`figures/NN_slug_figK.png`** (or `.pdf`) — the `NN` MUST match the owning notebook's slot | Any figure path whose `NN` doesn't match the notebook that emits it is a stale reference from the flatten and should be fixed. |
