#!/bin/bash
#SBATCH -J disc9_analyze
#SBATCH -p short
#SBATCH -t 05:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -o /mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study/logs/%x.%A_%a.out
#SBATCH -e /mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study/logs/%x.%A_%a.err

# Usage:
#   sbatch --export=ALL,TARGET=4QB3,CID=04d89e15... run_one.sh
# or  as array indexed into MANIFEST.tsv row order.

set -euo pipefail

DATA_ROOT=/mnt/netapp1/Store_othcxlwa/lwa_transfer/00_PRIORITY_discovery9_trajectories
WS=/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study
export PIXI_CACHE_DIR=$WS/.pixi-cache
export MDA_TIMESTEP_WORKERS=${SLURM_CPUS_PER_TASK:-1}

MANIFEST=$DATA_ROOT/MANIFEST.tsv

# Resolve target / cid
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # rows 2..N in the manifest (1-indexed skipping header) ==> array index 0..N-1
  row=$((SLURM_ARRAY_TASK_ID + 2))
  TARGET=$(awk -v r=$row -F'\t' 'NR==r{print $1}' "$MANIFEST")
  CID=$(awk    -v r=$row -F'\t' 'NR==r{print $2}' "$MANIFEST")
fi

if [[ -z "${TARGET:-}" || -z "${CID:-}" ]]; then
  echo "ERR: need TARGET/CID (env or via array index into MANIFEST)"; exit 2
fi

BASE=$DATA_ROOT/$TARGET/$CID
OUT=$WS/results/$TARGET/$CID
mkdir -p "$OUT"

# Skip if already done
if [[ -f "$OUT/summary.json" ]]; then
  echo "SKIP already done: $TARGET/$CID"; exit 0
fi

echo "== analyzing $TARGET/$CID =="
echo "GRO: $BASE/system.gro"
echo "XTC: $BASE/process/gromacs.xtc"
echo "TOP: $BASE/system.top"
echo "OUT: $OUT"
echo "CPUS: ${SLURM_CPUS_PER_TASK:-?}"
echo "NODE: $(hostname)"
date

cd $WS/env
pixi run python "$WS/scripts/analyze_complex.py" \
  --gro "$BASE/system.gro" \
  --xtc "$BASE/process/gromacs.xtc" \
  --top "$BASE/system.top" \
  --out "$OUT" \
  --stride "${STRIDE:-50}" \
  --ifp-stride "${IFP_STRIDE:-50}" \
  --hb-stride "${HB_STRIDE:-10}"

echo "== done $TARGET/$CID =="
date
