#!/bin/bash
# Launch the full L27 (27 configs) x 270 complexes = 7290 MD-only productions (20 ns).
# One array per config (270 tasks; under MaxArraySize=1001). GBSA scored later.
set -euo pipefail
HERE=/home/otras/hcx/lwa/projects/gbsa-factorial/scripts/md_variants
OUTROOT=/home/otras/hcx/lwa/newbench15_gbsa/md_variants/full20ns
mkdir -p "$OUTROOT"
[ -f "$OUTROOT/perf.csv" ] || echo "config,cid,status,ns_day,wall_s" > "$OUTROOT/perf.csv"

# 270 complex ids (9 targets) that have equilibrated inputs
CIDS=$OUTROOT/cids.txt
python3 - > "$CIDS" <<'PY'
import csv,os
RUN="/home/otras/hcx/lwa/newbench15_gbsa"
for r in csv.DictReader(open("/home/otras/hcx/lwa/projects/gbsa-factorial/data/metadata.csv")):
    cid=r["complex_id"]
    if os.path.exists(f"{RUN}/workspace/{cid}/production/system.gro") and \
       os.path.exists(f"{RUN}/workspace/{cid}/production/result.json"):
        print(cid)
PY
N=$(wc -l < "$CIDS"); echo "complexes: $N"

# L27 speed config ids
mapfile -t CFGS < <(python3 "$HERE/md_configs.py" 2>/dev/null | tail -n +2 | awk -F, '$2=="speed"{print $1}')
echo "configs: ${#CFGS[@]}"

for cfg in "${CFGS[@]}"; do
  jid=$(sbatch --parsable --array=1-${N} \
    --export=ALL,CONFIG="$cfg",CIDS="$CIDS",OUTROOT="$OUTROOT" \
    "$HERE/run_full.sbatch")
  echo "  submitted $cfg -> job $jid (270 tasks)"
done
echo "ALL SUBMITTED: ${#CFGS[@]} configs x $N complexes = $(( ${#CFGS[@]} * N )) productions"
