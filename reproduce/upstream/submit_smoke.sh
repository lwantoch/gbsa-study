#!/bin/bash
# Compatibility + speed smoke: 2 complexes x all 32 configs, short run (SMOKE_NSTEPS steps).
# Validates every config grompp+mdrun and records ns/day. GPU partition (abundant).
set -euo pipefail
HERE=/home/otras/hcx/lwa/projects/gbsa-factorial/scripts/md_variants
OUTROOT=/home/otras/hcx/lwa/newbench15_gbsa/md_variants/smoke
mkdir -p "$OUTROOT"
: > "$OUTROOT/perf.csv"
echo "config,cid,status,ns_day,steps,wall_s" > "$OUTROOT/perf.csv"

# smoke complexes (4QB3, have equilibrated inputs)
COMPLEXES=(cc7b6263adb98dd621cdb08fde955a6f 6b43e44fb3ada6de6c351a68d32eaa65)
# all config ids
mapfile -t CFGS < <(python3 "$HERE/md_configs.py" 2>/dev/null | tail -n +2 | cut -d, -f1)
echo "configs: ${#CFGS[@]}  complexes: ${#COMPLEXES[@]}"

LIST=$OUTROOT/tasklist.txt; : > "$LIST"
for c in "${COMPLEXES[@]}"; do for cfg in "${CFGS[@]}"; do echo "$c $cfg" >> "$LIST"; done; done
N=$(wc -l < "$LIST")
echo "tasks: $N"

sbatch --array=1-${N}%32 \
  --export=ALL,FAC_LIST="$LIST",OUTROOT="$OUTROOT",SMOKE_NSTEPS=50000 \
  "$HERE/run_variant.sbatch"
