#!/bin/bash
#SBATCH -J md_prod
#SBATCH -p gpu-l40s-mixed-mps,gpu-l40s-mixed-spot,gpu-l40s-mixed
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH --requeue
#SBATCH -o /home/otras/hcx/lwa/projects/gbsa-factorial/logs/prod_%A_%a.out
# MD-ONLY production variant (GBSA scored later). grompp a config's variant mdp onto the
# equilibrated system.{gro,top} and run 20 ns on GPU; SAVE the trajectory for post-hoc GBSA.
# env: CONFIG (config id), CIDS (file of complex ids), OUTROOT.
set -uo pipefail
ENV=/home/otras/hcx/lwa/gromacs-cuda/.pixi/envs/default
export PATH=$ENV/bin:$PATH
GMX=$ENV/bin/gmx
GEN=/home/otras/hcx/lwa/projects/gbsa-factorial/scripts/md_variants/gen_variant_mdp.py
RUN=/home/otras/hcx/lwa/newbench15_gbsa
cfg=${CONFIG:?set CONFIG}; CIDS=${CIDS:?set CIDS}; OUTROOT=${OUTROOT:?set OUTROOT}
RESULTS=$OUTROOT/perf.csv

cid=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CIDS"); [ -z "$cid" ] && exit 0
src=$RUN/workspace/$cid/production
gro=$src/system.gro; top=$src/system.top; base=$src/process/gromacs.mdp
out=$OUTROOT/$cfg/$cid; mkdir -p "$out"
[ -f "$out/prod.xtc" ] && { echo "$cfg/$cid already done"; exit 0; }
export TMPDIR=/tmp/prod_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p "$TMPDIR"; cd "$TMPDIR"

if [ ! -f "$gro" ] || [ ! -f "$top" ] || [ ! -f "$base" ]; then
  flock "$RESULTS.lock" -c "echo '$cfg,$cid,MISSING_INPUT,,' >> '$RESULTS'"; exit 0; fi

echo "=== PROD $cfg / $cid on $(hostname) $(date -u) ==="
python3 "$GEN" "$cfg" "$base" "$out/variant.mdp" >/dev/null 2>&1 \
  || { flock "$RESULTS.lock" -c "echo '$cfg,$cid,GEN_FAIL,,' >> '$RESULTS'"; exit 0; }
$GMX grompp -f "$out/variant.mdp" -c "$gro" -p "$top" -o "$TMPDIR/prod.tpr" \
  -po "$out/grompp.out.mdp" -maxwarn 20 > "$out/grompp.log" 2>&1
[ -f "$TMPDIR/prod.tpr" ] || { flock "$RESULTS.lock" -c "echo '$cfg,$cid,GROMPP_FAIL,,' >> '$RESULTS'"; exit 0; }

t0=$SECONDS
$GMX mdrun -s "$TMPDIR/prod.tpr" -deffnm "$TMPDIR/prod" \
  -ntmpi 1 -ntomp ${SLURM_CPUS_PER_TASK:-12} -pin on -cpt 60 > "$out/mdrun.log" 2>&1
rc=$?; wall=$((SECONDS - t0))
cp "$TMPDIR/prod.log" "$out/prod.log" 2>/dev/null
nsday=$(grep -A2 "^ *Performance:" "$TMPDIR/prod.log" 2>/dev/null | grep -oE "[0-9.]+" | head -1)

if [ $rc -eq 0 ] && [ -f "$TMPDIR/prod.xtc" ] && [ -n "$nsday" ]; then
  cp "$TMPDIR/prod.xtc" "$out/prod.xtc"; cp "$TMPDIR/prod.tpr" "$out/prod.tpr"
  echo "OK $cfg/$cid ns/day=$nsday wall=${wall}s"
  flock "$RESULTS.lock" -c "echo '$cfg,$cid,OK,$nsday,$wall' >> '$RESULTS'"
else
  echo "MDRUN FAIL rc=$rc $cfg/$cid"; tail -8 "$out/mdrun.log"
  flock "$RESULTS.lock" -c "echo '$cfg,$cid,MDRUN_FAIL,,$wall' >> '$RESULTS'"
fi
rm -rf "$TMPDIR"
