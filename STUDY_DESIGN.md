# Purpose of the study

The main objective of the study is to determine **how cheap MM-GBSA rescoring of MD trajectories
can be made while still reproducing a usable early-enrichment ranker (BEDROC α = 20)** on the
NewBench-27 discovery cohort, and **whether the winning cheap configuration can be predicted
for a target that was not seen during calibration**. Because BEDROC is a *ranking* metric it
does not require accurate absolute ΔG values per complex — it only requires the correct
ordering of actives ahead of measured non-binders. That opens a compute Pareto frontier: MD may
be shortened, integrated at a larger timestep, and rescored on fewer frames with a coarser
implicit-solvent recipe, each of which degrades individual ΔG accuracy but may leave the panel
ranking intact. This study locates that frontier for the reference benchmark of
[@ohds_newbench15], tests whether it generalises across targets under leave-one-target-out
cross-validation (a cheap-configuration *selector* trained on the training targets and applied
to a held-out target), selects the cheapest configuration whose panel BEDROC stays inside the
expensive reference's 95 % CI, and confirms the choice at cohort scale on an independent
compute site (CESGA FT3).

# Objective and success metric

The study will trade compute cost against ranking quality and reproducibility, and report a
single winning configuration per compute axis. The main success metrics are:

- **Panel BEDROC α = 20** [@truchon_bayly_2007] under LOTO (leave-one-target-out) with 95 %
  bootstrap CI over targets (`B = 5000`). Success on the speed-vs-accuracy axis is defined as
  a cheaper configuration whose panel BEDROC point estimate is at least as high as the
  expensive reference and whose CI overlaps the reference CI.
- **Compute cost per complex**, reported as (i) MD wallclock in GPU-hours per 30-ns
  trajectory, (ii) GBSA rescoring wallclock in CPU-seconds per 3001-frame trajectory, and
  (iii) end-to-end pipeline throughput in chains per GPU per day at cohort deployment scale.
  Success is a Pareto-dominant configuration: no other configuration reaches the same BEDROC
  at lower total cost.
- **Per-complex reproducibility of ΔG values under compute reduction**. The point of the
  study is *not* to reproduce ΔG accurately — it is to reproduce BEDROC accurately. But
  because ΔG variance is what drives BEDROC variance, the per-complex inter-replica ΔG
  standard deviation must be measured directly (from ≥ 3 replicas per complex, see
  Experiment B2) rather than assumed. A cheaper configuration is only accepted when its
  measured ΔG std stays below the ranking-critical noise floor derived from the reference.

Because raw BEDROC is not comparable across scorers with different labelled-target coverage,
we use two conventions consistently: the *canonical* 9-target panel imputes 0 for targets a
given scorer has no data for; the *subset* value restricts the comparison to targets both
scorers cover. Both numbers are computed in `reproduce/canonical_baselines.py`.

# Expected outcomes and known limitations

We expect that (i) the initial full-quality pipeline on OHDS establishes an expensive baseline
whose BEDROC anchors every downstream comparison; (ii) the 48-combo GBSA factorial identifies
a small number of parameters that actually drive ranking, allowing the rescore grid to be
pruned; (iii) the per-ligand MD trajectory feature bundle either provides a *replacement*
ranker (features beat GBSA) or a *predictor* of per-target GBSA quality (features pick the
best combo per target); (iv) Bayesian optimisation of the MDP finds a substantially cheaper
production configuration whose BEDROC stays inside the expensive reference CI; and (v) the
cheaper configuration reproduces on an independent compute site with measured per-complex ΔG
variance small enough that ranking survives.

Known limitations. The discovery cohort has n = 9 targets, so per-target bootstrap CIs are
wide and MDES for a panel-level BEDROC lift is only ≈ +0.11 for tight predictors and ≈ +0.37
for high-variance ones. Any speed-vs-accuracy configuration whose BEDROC point estimate is
inside the reference CI cannot be shown to be *worse* than the reference within this
cohort; the pre-registered NewBench-27 validation-18 subset is required for a confirmatory
panel-level claim and is out of scope here. The OHDS reference ships `n = 1` per (complex,
combo), so the reference itself has unmeasured MD-sampling variance; our own replica campaign
(Experiment B2) is what pins the noise floor.

# Study overview

## Input data

The cohort is NewBench-27, split into a 9-target discovery subset already fully analysed and
an 18-target validation subset that is pre-registered but not yet computed. Every headline
number in this study is on the discovery subset:

- **Discovery-9**: `2XU3`, `3I06`, `4A5S`, `4L7G`, `4QB3`, `5HU9`, `8ELC`, `9D9I`, `9SI4`.
  Each target has ≈ 30 candidate ligands (~ 10 actives + 20 measured non-binders drawn from
  the same ChEMBL series; not decoys, not DUD-E-style synthetic ligands). Actives are
  `pchembl ≥ 5` in the primary labelling.
- **Validation-18**: pre-registered in `data/raw/reference/ohds_newbench_targets.csv` under
  `split == 'validation'`. Not run here.

## Input preparation

### Protein preparation

Receptor structures for the discovery-9 subset are prepared with the in-house **FRUTON**
framework (Framework for Reconstruction, UniProt alignment, and Topology-Oriented protein
Normalization), which converts cropped or otherwise non-trivial UNIPROT/PDB inputs into
simulation-ready assemblies through an explicit, state-driven chain that treats the
UNIPROT sequence and the PDB coordinate deposition as a single joint reference. Raw structures are
parsed into a FASTA sequence and aligned to their canonical UniProt entry to establish which
residue numbering the coordinate file actually represents. Insertion codes and repeated
coordinate copies of the same chain are then normalised, and a representative-unit selector
walks the chain graph so that duplicate equivalent chains from oligomeric assemblies are
removed while ligands and cofactors bound to the retained chains are preserved (a chain-ID
based ownership rule, not distance-based). The representative unit is then split into its
protein, ligand, water, and metal components, and each is processed downstream according to
its own logic — waters and crystallographic metals in particular are optionally forwarded
into the solvation box, while cofactors are handed off to a small-molecule parameterisation
step. Gap detection on the representative protein identifies missing internal residues; where
present, MODELLER-based comparative modelling reconstructs the missing spans so that a
complete-model variant can be produced alongside the gapped variant, and the gap policy per
target is recorded in the state JSON. Titratable-residue protonation states at physiological
pH are assigned by empirical pKa prediction and applied via GROMACS `pdb2gmx`. Where no
suitable crystallographic template is available, the FRUTON fallback path substitutes an
AlphaFold2 or ESMFold sequence-based model into the same protonation-and-solvation branch.
FRUTON's design principle is that every important transformation leaves visible evidence on
disk: intermediate files, alignments, and per-target JSON/XLSX state are preserved, so that
each preparation decision is inspectable rather than implicit. The
`receptor.pdb` / `receptor_WAT.pdb` files delivered to the discovery-9 subset are the
FRUTON output for each target and are consumed unchanged by the downstream MD stage.

### Ligand preparation

Ligands are received as neutralised, protonated SDFs. Docking is performed with AutoDock
Vina using each target's pre-defined box centre and size from the receptor metadata; the
top-scoring pose per (target, ligand) seeds the MD stage.

### MD preparation

MD preparation is driven by the `gbsa-pipeline` action chain
(`prepare_inputs → dock → export_pose → setup_system → md_sd → md_cg → md_nvt_res →
md_npt_res → md_npt → md_production → gbsa`) orchestrated by `grubicy` on `signac`-managed
per-complex workspaces, so that each stage's inputs, outputs, and parent dependency are
declared on disk. Each equilibration action is one GROMACS `mdrun` call executed through
BioSimSpace, which handles topology assembly, restraint placement, and coordinate exchange
between stages and checkpoints the intermediate `.gro` / `.top` pair used by the next stage.
Every complex uses the same equilibration protocol: steepest-descent minimisation is
followed by conjugate-gradient minimisation to relax steric clashes introduced by docking;
the system is then heated to the target temperature under NVT for 50 ps with backbone
position restraints; density is brought to atmospheric pressure by 100 ps of NPT
equilibration under isotropic Parrinello–Rahman pressure coupling, again with backbone
restraints; finally the restraints are released and the system runs a further 100 ps of
unrestrained NPT before the production trajectory begins. Solvation is performed in a
periodic cubic box with a 1.0 nm buffer between solute and box edge, using TIP3P water and
0.15 M NaCl to neutralise the system. Long-range electrostatics are handled by Particle Mesh
Ewald with a 1.0 nm real-space cutoff, van der Waals interactions are truncated at 1.0 nm,
and hydrogen bonds are constrained via LINCS to permit the production timestep chosen by
the BO winner of Experiment A4. The AMBER ff14SB protein force field, the GAFF2 small-
molecule force field, and the TIP3P water model are fixed for the entire campaign so that
no cross-force-field variance enters the BEDROC comparisons; the only production knob left
open is the MDP itself (integrator, timestep, thermostat/barostat, LINCS order, HMR/MTS
toggles), which Experiment A4 tunes and Experiment B2 verifies at cohort scale.

### GBSA rescoring

The reviewer-locked GBSA recipe used for the reference is `igb2_di4_salt0.15_st0.0072` —
Onufriev–Bashford–Case implicit solvent (igb = 2) [@onufriev_bashford_case_2004], internal
dielectric 4, 0.15 M salt, surface-tension coefficient 0.0072 kcal / mol / Å²
[@sitkoff_sharp_honig_1994]. Rescoring uses `gmx_MMPBSA`
[@valdes_tresanco_gmx_mmpbsa_2021] with the AmberTools sander backend on 3001 frames per
trajectory (one snapshot every 10 ps of the 30-ns production). The wider 48-combo sweep of
Experiment A2 explores whether a cheaper combo (fewer physics terms, lower `saltcon`, etc.)
preserves ranking at lower per-frame cost.

## Proposed experiments

The plan runs in **two phases** separated by a compute-site switch. Phase A ran on the OHDS
compute environment and produced the expensive reference plus the cheaper-configuration
candidates. Phase B moves to CESGA FT3 and independently reproduces the reference, then
deploys the top BO cheaper candidates at production scale with enough replicas to measure
per-complex ΔG variance directly.

### Phase A — establish the expensive reference and locate cheaper candidates (on OHDS)

- **Experiment A1** — ***Run the full-quality pipeline once, end-to-end (the expensive reference)***

  The first experiment runs the complete pipeline — Vina docking → 30-ns GROMACS MD →
  gmx_MMPBSA rescoring with the reviewer-locked combo — on the full discovery-9 cohort
  (9 × 30 = 270 complexes). The purpose is to establish the *expensive reference* that
  every downstream cheaper configuration is compared against: its panel BEDROC α = 20, its
  per-target BEDROC values with bootstrap CIs, and its per-complex ΔG values. It also
  caches the 270 trajectories that Experiments A2 and A3 depend on.

  Success is defined as ≥ 95 % completed chains and a panel BEDROC point estimate at or
  above the docking baseline (docking is the trivial cheap alternative to any GBSA-based
  ranker; if GBSA at full quality does not beat docking, the study has no headroom to
  trade accuracy for speed and stops here).

- **Experiment A2** — ***GBSA full-factorial parameter sweep (can we cheapen the physics?)***

  With the 270 trajectories from Experiment A1 in hand, we sweep the GBSA parameter space:
  `igb ∈ {1, 2, 5, 8}` × `intdiel ∈ {1, 2, 4}` × `saltcon ∈ {0.15, 0.20}` × `surften
  ∈ {0.0, 0.0072}` = 48 combos per (target, ligand). This produces one ΔG per (target,
  ligand, combo), yielding ~ 9553 usable rows.

  The purpose is to test whether the reviewer-locked combo is *the* configuration required
  to reach the reference BEDROC, or whether a cheaper choice reaches the same BEDROC at
  lower per-frame cost. One-way variance decomposition (η²) of ranking metrics (Kendall τ,
  panel BEDROC α = 20) against each combo factor identifies which axes matter for ranking;
  factors whose variance contribution is inside noise can be *dropped* rather than swept in
  production, cheapening the rescore. A DOE analysis using the NIST-sequence factorial
  toolkit provides a cross-check. Any combo whose panel BEDROC point estimate is inside the
  reviewer-locked CI at lower per-frame cost is a candidate cheaper configuration.

- **Experiment A3** — ***Per-ligand MD trajectory feature analysis (can features replace GBSA?)***

  For every one of the 270 trajectories cached in Experiment A1 we extract a common bundle
  of ≈ 60 per-complex MD-stability features (kinematic drift, COM displacement, RMSF of
  protein CA and ligand heavy atoms, active-site H-bond persistence, interaction-fingerprint
  Tanimoto over time, ligand buried SASA statistics, PBC-image jump flags) plus 10
  ligand-chemistry descriptors (MW, n_heavy, HBD/HBA counts, ring counts,
  sum-of-partial-charges, logP proxies). MDAnalysis handles the trajectory reads; RDKit
  handles the ligand chemistry.

  The purpose is to test the ultimate cheap alternative to GBSA rescoring: MD-derived
  features cost nothing beyond MDAnalysis on the trajectory that MD has already produced.
  Two sub-questions. **Q1**: can a leave-one-target-out (LOTO) regressor pick the best-of-48
  GBSA combo per target from an MD fingerprint alone (a combo *selector*, cheaper than
  running the full sweep)? **Q2**: does any single MD feature (or a small rank fusion) beat
  the reviewer-locked GBSA as a standalone panel ranker (a full *replacement* for GBSA)?
  Success on Q2 is a per-target BEDROC lift larger than the paired standard deviation at
  80 % power AND survival of all four hardening conditions (MW-residualised,
  size-residualised, bound-only, PBC-clean). Anything that survives naive-only is not
  deployable. The feature cache is written to `data/raw/complex_analyses/{TARGET}/{ID}/` and
  consumed read-only by later experiments.

- **Experiment A4** — ***Bayesian optimisation of the production MDP (can we cheapen the MD?)***

  Since MD dominates the pipeline cost, the biggest speed lever is the MDP: larger timestep
  (with HMR / MTS / LINCS-iter/order), fewer neighbour-list rebuilds, cheaper thermostat /
  barostat. But cheaper MD may destabilise the trajectory or move it away from the
  ranking-relevant conformational ensemble. This experiment finds the cheapest MDP whose
  trajectory is still stable and whose downstream GBSA ΔG is not degraded.

  Optuna Bayesian optimisation is run over a 17-parameter GROMACS-2025 MDP + FF search
  space (`data/external/bayesopt/code/space_v2.py`) with a TPE (Tree-structured Parzen
  Estimator) sampler [@bergstra_tpe_2011]. The fitness function is `ns/day × stability`
  maximised on a pre-equilibrated seed topology; `ns/day` is the speed factor and
  `stability` (an energy-drift-based score) is the accuracy factor, so the fitness *is* the
  speed-vs-accuracy tradeoff evaluated at trial length. FF Tier 1 (`ff14SB + gaff2 + tip3p
  + vdw_modifier=Potential_Shift`) is frozen because trials replay the seed topology, so
  swapping FF mid-flight is not physically valid.

  Two studies are run: `md_prod_v1` (200 trials, base envelope) and `md_prod_v2` (34
  trials, extended envelope enabling HMR and MTS). The top-1 winner is the deployed MDP;
  the top-5 winners are all forwarded to Phase B so that fitness measured at 20-ps trial
  scale can be verified at 20-ns production, since cheap MDPs that pass the trial scale can
  still fail at production length (drift accumulates). Every trial is stored in the Optuna
  SQLite database (`data/external/bayesopt/bayes_opt.db`) with parameter vector, fitness,
  and wallclock.

### Phase B — independent reproduction and cohort-scale verification (on CESGA FT3)

- **Experiment B1** — ***Reproduce the expensive reference on FT3***

  Two independent reproduction runs on FT3, in sequence, so that the setup and MD
  contributions to any per-complex disagreement with OHDS are separable.

  *B1a — setup-only cross-check.* We run our `gmx_MMPBSA` (AmberTools sander backend,
  reviewer-locked combo) on OHDS' own MD trajectories for a representative subset of
  complexes (~ 60–65 chains covering 2XU3, 3I06, and a 4L7G spot check). If our number
  equals OHDS's number to within the sub-kcal MD-sampling noise floor, our GBSA rescoring
  is 1:1 identical to the reference; any subsequent full-pipeline disagreement is
  attributable to the MD side, not to setup. The success criterion is `|Δ|_mean` well
  below 0.5 kcal / mol across the panel.

  *B1b — full-pipeline reproduction.* With the setup verified, we run the complete
  pipeline (Vina docking → 30-ns MD → MM-GBSA) on a small end-to-end reproduction cohort:
  8 discovery targets × ligand01 × 3 replicas = 24 chains at the reviewer-locked combo.
  For each (target, ligand) the median, standard deviation, and range across the three
  replicas are reported, together with the delta of the median relative to the OHDS
  reference. The measured inter-replica std sets the noise floor for the cohort-scale
  replication in Experiment B2 and is the ranking-critical variance bound for the
  cheaper configuration acceptance in Experiment A4.

- **Experiment B2** — ***Verify the top-5 BayesOpt winners at cohort scale on FT3***

  The final experiment deploys the top-5 BO-winner MDPs from Experiment A4 on the full
  discovery cohort: 8 targets × 30 ligands × 3 replicas × 5 winners = 3600 chains. The
  purpose is threefold. (i) Confirm that the cheap-MDP fitness measured at 20-ps trial
  scale generalises to 20-ns production — cheap MDPs sometimes pass at trial scale and
  fail at production length. (ii) Give every discovery complex ≥ 15 GBSA measurements (3
  replicas × 5 MDP variants) so that per-complex MD-sampling variance is measurable
  directly from our own data rather than assumed from an `n = 1` OHDS reference; this is
  the noise floor that the cheaper configurations of Experiments A2–A4 must stay below to
  be accepted. (iii) Recompute panel BEDROC per BO winner and compare against the
  expensive reference of Experiment A1 — winners whose BEDROC point estimate stays inside
  the reference CI *at their cheaper MDP cost* are the deployable configurations.

  The compute path is `gpupack` MPS-packed A100 GPUs. The per-GPU concurrency K is set
  from the gpupack calibration derived by the companion GPU-MPS-Performance study
  [@gromacs_mps_study] — i.e. the N*, footprint constants, and δ_HOM measured under the
  gromacs-mps calibration protocol for the AMBER-ff14SB / TIP3P soluble-system size class
  that our discovery-9 chains fall into. K is therefore not re-derived here and is not a
  free parameter of this study; it is inherited from Study A so that the two studies stay
  aligned. The deployment is chained through a watcher on the short partition
  (`bo_waves_watcher.sbatch`) that respects the QoS `MaxSubmit = 50` limit; each wave
  splits into two 90-task array batches. Per (target, ligand) the median, std, and range
  across the 15 replicas are reported and cross-referenced against Experiment B1a
  (setup Δ) and B1b (baseline_8t Δ) in a tri-source living notebook that re-reads the
  aggregated CSVs on every execution.

## Proposed timeline

The following timeline is preliminary and excludes cluster queueing delays. Phase A ran on
the OHDS compute environment before the FT3 switch; Phase B is the current campaign.

- **Experiment A1** (initial 270-chain full-quality pipeline) — approximately 5–7 days
  end-to-end including retries;
- **Experiment A2** (48-combo GBSA sweep on the 270 trajectories) — approximately 3–4
  days;
- **Experiment A3** (per-ligand feature extraction + Q1 / Q2 analyses) — approximately
  1–2 days for extraction + 3–4 days for the ranking analyses with bootstrap CIs and
  hardening;
- **Experiment A4** (BayesOpt of production MDP, 200 + 34 trials) — approximately 3 days;
- **Experiment B1a** (setup cross-check, ~65 chains) — approximately 1 day
  (~11 min per chain, batched);
- **Experiment B1b** (baseline_8t reproduction, 24 chains) — approximately 2 days;
- **Experiment B2** (5-wave BO-verify production, 3600 chains) — approximately 10 days
  under normal queue conditions;
- final analysis and preparation of the reviewer-facing report — approximately 2–3 days.

Total time depends strongly on queue availability and whether the per-complex MD-sampling
std flagged in B1b triggers extended replication (n ≥ 5) for individual complexes in B2.

## Main reading

1. Truchon J-F, Bayly CI. Evaluating Virtual Screening Methods: Good and Bad Metrics for
   the "Early Recognition" Problem. Journal of Chemical Information and Modeling. 2007;
   47(2):488–508. https://doi.org/10.1021/ci600426e
2. Onufriev A, Bashford D, Case DA. Exploring protein native states and large-scale
   conformational changes with a modified generalized Born model. Proteins. 2004;
   55(2):383–394. https://doi.org/10.1002/prot.20033
3. Valdés-Tresanco MS, Valdés-Tresanco ME, Valiente PA, Moreno E. gmx_MMPBSA: A New Tool to
   Perform End-State Free Energy Calculations with GROMACS. Journal of Chemical Theory and
   Computation. 2021; 17(10):6281–6291. https://doi.org/10.1021/acs.jctc.1c00645
4. Maier JA, et al. ff14SB: Improving the Accuracy of Protein Side Chain and Backbone
   Parameters from ff99SB. Journal of Chemical Theory and Computation. 2015;
   11(8):3696–3713. https://doi.org/10.1021/acs.jctc.5b00255
5. Abraham MJ, et al. GROMACS: High performance molecular simulations through multi-level
   parallelism from laptops to supercomputers. SoftwareX. 2015; 1–2:19–25.
   https://doi.org/10.1016/j.softx.2015.06.001
6. Bergstra J, Bardenet R, Bengio Y, Kégl B. Algorithms for Hyper-Parameter Optimization.
   Advances in Neural Information Processing Systems. 2011; 24:2546–2554.

# References

.bib not supported in Markdown
