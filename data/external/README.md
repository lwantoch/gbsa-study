# data/external — heavy inputs not shipped in git

The small reference tables that the notebooks actually consume are copied into
`data/raw/reference/` (prefixed `ohds_*`) and tracked in the repo. Everything under
`data/external/` proper is heavy or provenance-only and lives outside git.

## What lives here (all gitignored)

| Path | Contents | How to obtain |
|------|----------|---------------|
| `newbench15/` | Full OHDS `gbsa-study` code snapshot (its own embedded git repo, ~44 MB) | Ask the OHDS group for the tarball, or clone their internal fork. Only needed if you want the original OHDS notebooks alongside ours. |
| `newbench15/trajectories/` | 30 ns MD trajectories per (target × ligand), the OHDS run of the reviewer combo (~100 GB) | Symlink to your local `Store_othcxlwa/gbsa-study-data/newbench15_trajectories/` mirror. Only needed to re-run `_ohds_xchk_all` (our gmx_MMPBSA cross-check on OHDS trajectories). |
| `study2_full/` | Study 2 L27 GROMACS parameter screen (~2.4 GB) | Symlink to `Store_othcxlwa/gbsa-study-data/study2_full/`. Needed only for `09_study2_gromacs_screening.ipynb` if you want to re-aggregate raw per-run CSVs. |
| `study2_pilot/` | Study 2 pilot (780 MB) | Symlink to `Store_othcxlwa/gbsa-study-data/study2_pilot/`. Optional. |
| `bayesopt/` | Optuna trial DB + BO winner MDPs (~2.8 MB) | Tracked in git — needed by `32_bo_md_report.ipynb`. |

## The CSVs the notebooks actually read

Notebooks import via `gbsabench.paths.RAW / "reference" / …` — the copies in
`data/raw/reference/` are the source of truth in this repo:

```
data/raw/reference/
  ohds_metadata.csv                     — complex_id ↔ target, ligand_file, pchembl, is_active, docking_score
  ohds_gbsa_dG_raw.csv                  — 48 combos × 30 ligs × 8 targets = 9553 rows OHDS reference dG
  ohds_newbench_targets.csv             — 8 target PDB IDs with residue selections
  ohds_md_productions_raw.csv           — per-run MD wallclock (OHDS)
  ohds_md_variants_manifest_raw.csv     — Study 2 L27 config manifest
  ohds_md_variants_prod_perf_raw.csv    — Study 2 per-variant throughput
  ohds_md_variants_gbsa_scores_raw.csv  — Study 2 per-variant GBSA scores
  ohds_md_speed_smoke_L40S.csv          — L40S smoke test throughput
```

If you want to regenerate any of these from the OHDS embedded repo, fetch it first
(see the table above), then rerun `reproduce/refresh_ohds_reference.py` (writes back
into `data/raw/reference/`).
