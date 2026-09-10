# Software stack

This document names the exact software versions and external packages used to produce the raw
data that ships in `data/raw/`. Notebooks re-analyse those CSVs and require only the plain
Python stack listed in `pyproject.toml`.

## Analysis (notebooks + helpers in `src/gbsabench/`)

Managed via `pyproject.toml` or the pixi environment in `env/`. Any Python ≥3.10 install with
the dependencies in `pyproject.toml` reproduces every figure and table in this repo from a
fresh clone.

```
python -m venv .venv && source .venv/bin/activate
pip install -e .        # analysis stack (numpy/scipy/pandas/polars/matplotlib/seaborn/sklearn/jupyter)
pip install -e .[reproduce]   # optional: also install MDAnalysis, freesasa, rdkit
```

Pixi users:

```
cd env && pixi install
pixi run jupyter lab
```

## Data-generating pipeline (external to this repo)

The raw CSVs in `data/raw/` come from MD + MM-GBSA runs on the CESGA FT3 cluster. The
run harness lives in a separate repository and is not vendored here; its version and
configuration are named below so a reviewer can trace every number to source.

| Component | Version / commit | Purpose |
|-----------|------------------|---------|
| `pipeline-mps` (fork of `gbsa-grub`) | branch `main`, commit sha in `docs/DATA_LINEAGE.md` | Orchestrates docking → MD → MM-GBSA per (target, ligand, replica) |
| `gpupack` (MPS packing layer) | `GrHeCo-Xen/gromacs-mps` branch `calibration_data` | Packs K concurrent chains per A100 via CUDA MPS |
| GROMACS | `2025.4-conda_forge` (AVX2_256) | MD engine |
| `gmx_MMPBSA` | 1.6.x with AmberTools sander backend | GBSA free-energy calculation |
| AutoDock Vina | 1.2.x | Docking (via `pipeline-mps.dock_stage`) |
| BioSimSpace (BSS) | 2024.x | MD orchestration + energy parsing |
| HyperQueue | 0.24.x | Concurrent pipeline dispatch under one SLURM job |

The reviewer-locked GBSA combo throughout this study is
`igb2_di4_salt0.15_st0.0072` (Onufriev-Bashford-Case implicit-solvent model, internal dielectric 4,
0.15 M salt, 3-fs snapshot spacing). This name is decoded in `docs/GLOSSARY.md`.

## Cluster / hardware

| Item | Value |
|------|-------|
| Cluster | CESGA FT3 |
| GPU | NVIDIA A100-PCIE-40GB |
| CPU:GPU ratio | 32:1 (used with `OMP_NUM_THREADS = 32 / K`) |
| MPS packing K | 4 (verified stable in the K-sweep 2026-09-09; K≥6 timed out) |

## Where each raw CSV came from

See `docs/DATA_LINEAGE.md` for the per-file provenance (job IDs, workspace paths, aggregation
script). Every CSV under `data/raw/` is regenerable by running the SLURM sbatches referenced
there and re-running `reproduce/aggregate_live_gbsa.py`.
