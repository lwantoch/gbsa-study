# TRANSFER — data package (organised by project)

Reorganised 2026-08-31. A **segregated, per-project map** of the work on the AWS
hpc-compute box. Entries are symlinks into their live `$HOME` locations (nothing was
moved or duplicated); they dereference on transfer (`rsync -aHL`).
**Environments/toolchains are NOT included** — rebuild on target.

## How to transfer
See **`RUN_ON_FT3.md`** (run on FT3; it pulls). Three steps: (1) discovery-9
trajectories, (2) all project code+inputs, (3) the ~9 TB raw MD bulk.

## Layout

```
00_PRIORITY_discovery9_trajectories/   270 × 30ns productions (9 targets), ~237 GB — PULL FIRST
projects/
  01_newbench-gbsa-study/       code/ + raw_md/ + reviewer_package/   ~5.4 TB
  02_affinity-benchmark-mmbsa200/ code/ + raw_md/ + inputs/           ~3.3 TB
  03_gbsa-factorial/            code/ + raw_md/                       ~244 GB
  04_dekois-screen/             raw_md/                               ~352 GB
  05_mps-gpu-benchmark/         code/ + raw_md/                       ~30 GB
  06_idis-movie/                blender_movie/ + nci/                 ~5 GB
  99_misc/                      1jcn*, jcn_perfect, loose root files  ~3 GB
```

Each project: `code/` = source/notebooks/figures/results; `raw_md/` = the MD/GBSA
signac workspaces (trajectories + per-frame results); `inputs/` = prepared PDBs, boxes.

## For a reviewer
Start in each project's `code/`. `raw_md/` is provenance / re-analysis only.
`01_.../reviewer_package/` and `02_.../code/affinity-benchmark` have their own
READMEs + `verify.py`.
