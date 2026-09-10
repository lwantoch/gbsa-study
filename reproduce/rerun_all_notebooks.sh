#!/usr/bin/env bash
# Rerun every notebook end-to-end. Regenerates all figures + all data/derived CSVs.
# Prereqs: pixi env at env/ installed; gbsa-study installed editable (pip install -e .).
# Usage:   bash reproduce/rerun_all_notebooks.sh [--fast] [--pattern NN]
#   --fast     skip slow notebooks (08, 09, 24 heavy ML/DoE)
#   --pattern  only run notebooks matching prefix NN (e.g., "08")
set -uo pipefail

FAST=0
PATTERN=""
while [ $# -gt 0 ]; do
    case "$1" in
        --fast) FAST=1;;
        --pattern) PATTERN="$2"; shift;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
    shift
done

REPO=/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study
PY=$REPO/env/.pixi/envs/default/bin/python
LOG=$REPO/reproduce/rerun_all.log

: > "$LOG"
cd "$REPO"

# Refresh live snapshots first (aggregates finished SLURM chains into data/raw/)
$PY reproduce/aggregate_live_gbsa.py >> "$LOG" 2>&1 || true

FAIL=0; OK=0
for nb in notebooks/[0-9]*.ipynb; do
    name=$(basename "$nb" .ipynb)
    [ -n "$PATTERN" ] && [[ "$name" != $PATTERN* ]] && continue
    if [ $FAST -eq 1 ] && [[ "$name" =~ ^(08|09|24)_ ]]; then
        echo "SKIP  $name (--fast)"; continue
    fi
    start=$(date +%s)
    if timeout 600 "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.kernel_name=gbsa-study \
        --ExecutePreprocessor.timeout=540 \
        "$nb" >> "$LOG" 2>&1; then
        printf "OK    %-45s  %3ds\n" "$name" $(( $(date +%s) - start ))
        OK=$((OK+1))
    else
        printf "FAIL  %-45s  %3ds\n" "$name" $(( $(date +%s) - start ))
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "=== summary: $OK ok, $FAIL failed  (details in $LOG) ==="
[ $FAIL -eq 0 ]
