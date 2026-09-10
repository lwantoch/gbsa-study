# RUN ON FT3 — pull the AWS data package

**1 TB destination budget.** Transfer = STEP 1 + STEP 2 only (**≈266 GB**):
discovery-9 trajectories + all project code/results. STEP 3 (the ~9 TB raw MD bulk)
is **SKIPPED** — it can't fit and isn't needed (GBSA already computed; raw MD stays
on AWS as provenance). ~730 GB headroom remains if you later want to add a named
raw subset.

AWS→FT3 is firewalled; FT3→AWS works, so **FT3 pulls**. No Claude bridge needed.
Keep the AWS box reachable for the whole pull. All rsyncs are resumable
(`--partial --append-verify`) — re-run any step to continue/verify.
**Environments/toolchains are NOT included** (rebuild on target).

AWS host: `othcxlwa@hpc-compute.dataspace.cesga.es`
Source  : `/home/otras/hcx/lwa/TRANSFER/`
Pick a destination once:  `export DEST=$STORE/lwa_transfer`

The package is organised **by project** (`projects/NN_<name>/` = `code/` + `raw_md/`
+ `inputs/`). Nothing is duplicated; symlinks dereference on pull (`-aHL`).

---

## STEP 1 — discovery-9 trajectories  (≈237 GB, do this first)

The 270 × 30 ns productions you need for the stability analysis.
```bash
mkdir -p "$DEST"
rsync -aHL --copy-unsafe-links --info=progress2 --partial --append-verify \
  othcxlwa@hpc-compute.dataspace.cesga.es:/home/otras/hcx/lwa/TRANSFER/00_PRIORITY_discovery9_trajectories/ \
  "$DEST/00_PRIORITY_discovery9_trajectories/"
```

## STEP 2 — all project CODE + inputs + docs  (≈50 GB, no trajectories)

Every project's source, notebooks, figures, results, reviewer packages, inputs —
excludes the heavy `raw_md/` and all envs.
```bash
rsync -aHL --copy-unsafe-links --info=progress2 --partial --append-verify \
  --exclude='raw_md/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.ipynb_checkpoints/' \
  --exclude='.pixi/' --exclude='.venv/' --exclude='.uv_cache/' --exclude='*.lock' \
  --exclude='.stabenv/' --exclude='.nbenv/' --exclude='.boenv/' \
  othcxlwa@hpc-compute.dataspace.cesga.es:/home/otras/hcx/lwa/TRANSFER/ \
  "$DEST/"
```

## STEP 3 — fill the rest of the 1 TB budget  (≈621 GB)

Whole raw for the three smaller projects — fits alongside steps 1-2 under 1 TB
(cumulative ≈887 GB). Each is resumable.
```bash
for proj in 03_gbsa-factorial 04_dekois-screen 05_mps-gpu-benchmark; do
  rsync -aHL --copy-unsafe-links --info=progress2 --partial --append-verify \
    --exclude='.pixi/' --exclude='.venv/' --exclude='__pycache__/' \
    othcxlwa@hpc-compute.dataspace.cesga.es:/home/otras/hcx/lwa/TRANSFER/projects/$proj/raw_md/ \
    "$DEST/projects/$proj/raw_md/"
done
```

## STEP 4 — mmbsa200 computed results only  (0.6 GB, no trajectories)

The affinity-benchmark GBSA outputs (`*.h5`) so mmbsa200 stays reproducible without
its 3.3 TB of trajectories.
```bash
rsync -aHL --copy-unsafe-links --info=progress2 --partial \
  --include='*/' --include='*.h5' --exclude='*' \
  othcxlwa@hpc-compute.dataspace.cesga.es:/home/otras/hcx/lwa/TRANSFER/projects/02_affinity-benchmark-mmbsa200/raw_md/ \
  "$DEST/projects/02_affinity-benchmark-mmbsa200/raw_md/"
```

## NOT transferred (exceeds 1 TB; provenance only)
- `02_.../raw_md/` mmbsa200 trajectories — 3.3 TB (results captured by STEP 4)
- `01_.../raw_md/newbench15_gbsa` MD-variant runs — ~5.2 TB (beyond the discovery-9
  productions in STEP 1; results live in `01_.../code/gbsa-study`, in STEP 2)

---

## Project map (what you get)

| Project | code | raw_md | approx |
|---|---|---|---:|
| `projects/01_newbench-gbsa-study` | gbsa-study, newbench_27_review | newbench15_gbsa, fruton_prepared, previews | ~5.4 TB |
| `projects/02_affinity-benchmark-mmbsa200` | affinity-benchmark | mmbsa200_enrich_*, mmbsa200_core75_*, transfer | ~3.3 TB |
| `projects/03_gbsa-factorial` | gbsa-factorial | factorial, gbsa_prod, pb_rescore, quarantines | ~244 GB |
| `projects/04_dekois-screen` | — | dekois_*, DEKOIS-TESTSET, decoys_1uou | ~352 GB |
| `projects/05_mps-gpu-benchmark` | mps-study, gromacs-mps, gpupack-records, figs | mps_*, test_*, bench_*, grod_test, gpupack-partitions | ~30 GB |
| `projects/06_idis-movie` | — | blender_movie, nci | ~5 GB |
| `projects/99_misc` | — | 1jcn*, jcn_perfect, loose root files | ~3 GB |

`projects/01_.../trajectories_discovery9_PULL_FIRST` → the STEP-1 priority bucket
(same 270 trajectories; curated index into project 01's raw_md, no data duplication).
