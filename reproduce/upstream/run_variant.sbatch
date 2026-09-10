#!/bin/bash
#SBATCH -J md_variant
#SBATCH -p gpu-l40s-mixed-mps
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 12
#SBATCH -t 08:00:00
#SBATCH -o /home/otras/hcx/lwa/projects/gbsa-factorial/logs/mdvar_%A_%a.out
# Re-run ONE (complex,config) GROMACS production variant from the equilibrated system.
# grompp a variant mdp onto workspace/<cid>/production/system.{gro,top}, mdrun on GPU,
# record ns/day. env: FAC_LIST (tasklist "cid config"), OUTROOT, SMOKE_NSTEPS (optional).
# Uses the CUDA GROMACS 2026.0 build that the original productions used (the pixi env's
# gmx is an OpenCL 2025.4 build that cannot offload bonded -> matches production engine).
set -uo pipefail
ENV=/home/otras/hcx/lwa/gromacs-cuda/.pixi/envs/default
export PATH=$ENV/bin:$PATH
GMX=$ENV/bin/gmx
GEN=/home/otras/hcx/lwa/projects/gbsa-factorial/scripts/md_variants/gen_variant_mdp.py
RUN=/home/otras/hcx/lwa/newbench15_gbsa
OUTROOT=${OUTROOT:?set OUTROOT}
LIST=${FAC_LIST:?set FAC_LIST}
RESULTS=$OUTROOT/perf.csv

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -z "$line" ] && exit 0
cid=$(echo "$line" | awk '{print $1}'); cfg=$(echo "$line" | awk '{print $2}')
src=$RUN/workspace/$cid/production
gro=$src/system.gro; top=$src/system.top
base=$src/process/gromacs.mdp
out=$OUTROOT/$cfg/$cid; mkdir -p "$out"
export TMPDIR=/tmp/mdvar_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$TMPDIR"
cd "$TMPDIR"

if [ ! -f "$gro" ] || [ ! -f "$top" ] || [ ! -f "$base" ]; then
  echo "MISSING inputs for $cid"; flock "$RESULTS.lock" -c \
    "echo '$cfg,$cid,MISSING_INPUT,,,' >> '$RESULTS'"; exit 0
fi

echo "=== VARIANT $cfg / $cid on $(hostname) $(date -u) ==="
python3 "$GEN" "$cfg" "$base" "$out/variant.mdp" ${SMOKE_NSTEPS:+$SMOKE_NSTEPS} \
  || { flock "$RESULTS.lock" -c "echo '$cfg,$cid,GEN_FAIL,,,' >> '$RESULTS'"; exit 0; }

$GMX grompp -f "$out/variant.mdp" -c "$gro" -p "$top" -o "$TMPDIR/prod.tpr" \
  -po "$out/grompp.out.mdp" -maxwarn 15 > "$out/grompp.log" 2>&1
if [ ! -f "$TMPDIR/prod.tpr" ]; then
  echo "GROMPP FAIL $cfg/$cid"; tail -5 "$out/grompp.log"
  flock "$RESULTS.lock" -c "echo '$cfg,$cid,GROMPP_FAIL,,,' >> '$RESULTS'"; exit 0
fi

t0=$SECONDS
# Match the original productions: let GROMACS auto-assign GPU tasks (nb+PME on GPU,
# update/bonded on CPU where required). Forcing offload breaks configs (mts/sd/md-vv) and
# OpenCL. Auto gives each config its best valid assignment.
$GMX mdrun -s "$TMPDIR/prod.tpr" -deffnm "$TMPDIR/prod" \
  -ntmpi 1 -ntomp ${SLURM_CPUS_PER_TASK:-12} -pin on -cpt 60 > "$out/mdrun.log" 2>&1
rc=$?
wall=$((SECONDS - t0))
cp "$TMPDIR/prod.log" "$out/prod.log" 2>/dev/null
nsday=$(grep -A2 "^ *Performance:" "$TMPDIR/prod.log" 2>/dev/null | grep -oE "[0-9.]+" | head -1)
steps=$(grep -oE "nsteps *= *[0-9]+" "$out/variant.mdp" | grep -oE "[0-9]+" | head -1)

if [ $rc -ne 0 ] || [ -z "$nsday" ]; then
  echo "MDRUN FAIL rc=$rc $cfg/$cid"; tail -8 "$out/mdrun.log"
  flock "$RESULTS.lock" -c "echo '$cfg,$cid,MDRUN_FAIL,,$steps,$wall' >> '$RESULTS'"
else
  echo "OK $cfg/$cid  ns/day=$nsday  wall=${wall}s"
  # keep full trajectory only for real (non-smoke) runs
  [ -z "${SMOKE_NSTEPS:-}" ] && cp "$TMPDIR/prod.xtc" "$out/prod.xtc" 2>/dev/null
  flock "$RESULTS.lock" -c "echo '$cfg,$cid,OK,$nsday,$steps,$wall' >> '$RESULTS'"
fi
rm -rf "$TMPDIR"
