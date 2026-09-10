#!/bin/bash
# Rebuild TRANSFER/ as a clean PROJECT tree (symlinks only; nothing moved/copied).
# Idempotent: wipes and rebuilds TRANSFER/projects/ each run.
set -uo pipefail
H=/home/otras/hcx/lwa
T=$H/TRANSFER
P=$T/projects
ln_(){ [ -e "$H/$2" ] && ln -sfn "$H/$2" "$1/$(basename "$2")"; }   # link $H/$2 into dir $1

rm -rf "$P"; mkdir -p "$P"

# ---- 01 newbench-gbsa-study ------------------------------------------------
d=$P/01_newbench-gbsa-study; mkdir -p "$d/code" "$d/raw_md" "$d/reviewer_package"
ln_ "$d/code" projects/gbsa-study
ln_ "$d/code" projects/newbench_27_review
ln_ "$d/reviewer_package" newbench_27_review
ln_ "$d/raw_md" newbench15_gbsa
ln_ "$d/raw_md" fruton_prepared
ln_ "$d/raw_md" newbench_27_review_PREVIEW
ln_ "$d/raw_md" newbench_smoke_8X92
# curated discovery-9 index (pointers, no data dup) lives at TRANSFER/00_PRIORITY_*
ln -sfn ../../00_PRIORITY_discovery9_trajectories "$d/trajectories_discovery9_PULL_FIRST"

# ---- 02 affinity-benchmark-mmbsa200 ---------------------------------------
d=$P/02_affinity-benchmark-mmbsa200; mkdir -p "$d/code" "$d/raw_md" "$d/inputs"
ln_ "$d/code" projects/affinity-benchmark
[ -e "$H/projects/docking_boxes.csv" ] && ln -sfn "$H/projects/docking_boxes.csv" "$d/inputs/docking_boxes.csv"
ln_ "$d/inputs" ft3_staging
ln_ "$d/inputs" ft3_pull_staging
for x in $H/mmbsa200_enrich_* $H/mmbsa200_core75_* $H/mmbsa200_transfer; do
  [ -e "$x" ] && ln -sfn "$x" "$d/raw_md/$(basename "$x")"
done

# ---- 03 gbsa-factorial -----------------------------------------------------
d=$P/03_gbsa-factorial; mkdir -p "$d/code" "$d/raw_md"
ln_ "$d/code" projects/gbsa-factorial
for x in factorial gbsa_prod pb_rescore gbsa_invalid_quarantine numbering_mismatch_quarantine; do ln_ "$d/raw_md" "$x"; done

# ---- 04 dekois-screen ------------------------------------------------------
d=$P/04_dekois-screen; mkdir -p "$d/raw_md"
for x in $H/dekois_* $H/DEKOIS-TESTSET $H/decoys_1uou; do
  [ -e "$x" ] && ln -sfn "$x" "$d/raw_md/$(basename "$x")"
done

# ---- 05 mps-gpu-benchmark --------------------------------------------------
d=$P/05_mps-gpu-benchmark; mkdir -p "$d/code" "$d/raw_md"
for x in mps-study gromacs-mps gpupack-mps-records mps_longrun_figs mps_test_figs study2_wallclock_report; do ln_ "$d/code" "projects/$x"; done
for x in mps_bench mps_longrun mps_test bench_fastest grod_test gpupack-partitions; do ln_ "$d/raw_md" "$x"; done
for x in $H/test_*; do [ -e "$x" ] && ln -sfn "$x" "$d/raw_md/$(basename "$x")"; done

# ---- 06 idis-movie ---------------------------------------------------------
d=$P/06_idis-movie; mkdir -p "$d"
ln_ "$d" blender_movie
ln_ "$d" nci

# ---- 99 misc ---------------------------------------------------------------
d=$P/99_misc; mkdir -p "$d" "$d/projects_root_loose"
for x in 1jcn 1jcn_clean 1JCN_perfect jcn_perfect 1jcn_gbsa_logs; do ln_ "$d" "$x"; done
for x in $H/projects/*.png $H/projects/README.md $H/projects/progress \
         $H/projects/gbsa-study.zip $H/projects/gbsa-study.backup.*; do
  [ -e "$x" ] && ln -sfn "$x" "$d/projects_root_loose/$(basename "$x")"
done

echo "=== project tree built ==="
for pd in "$P"/*/; do
  echo "--- $(basename "$pd") ---"
  find "$pd" -maxdepth 2 -mindepth 1 | sed "s#$P/##" | sort | sed 's/^/    /'
done