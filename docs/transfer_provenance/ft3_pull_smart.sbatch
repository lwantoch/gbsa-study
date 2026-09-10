#!/bin/bash
# ===== SMART FT3 PULL (raw_md bulk) — submit on FT3:  sbatch ft3_pull_smart.sbatch =====
# Parallel + resumable pull of every projects/*/raw_md/* item from AWS into FT3.
# AWS->FT3 is firewalled; FT3->AWS works, so this runs ON FT3 and PULLS.
# Environments/toolchains are NOT transferred.
#
# EDIT --partition / --time / DEST after `sinfo`.
#
#SBATCH --job-name=lwa_pull
#SBATCH --partition=__PARTITION__          # <-- EDIT (a serial/transfer/short partition)
#SBATCH --time=24:00:00                     # <-- EDIT to the partition max walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                   # = number of parallel rsync streams
#SBATCH --output=lwa_pull_%j.log
#
# Requeue-safe: if it times out, just `sbatch` again — rsync resumes.

set -uo pipefail
AWS=othcxlwa@hpc-compute.dataspace.cesga.es
SRC=/home/otras/hcx/lwa/TRANSFER
DEST="${DEST:-$STORE/lwa_transfer}"         # <-- EDIT or export DEST=...
STREAMS="${SLURM_CPUS_PER_TASK:-8}"
NEWBENCH="projects/01_newbench-gbsa-study/raw_md/newbench15_gbsa"   # the 5.4TB monolith

SSH="ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 -o BatchMode=yes"
RSYNC_OPTS="-aHL --copy-unsafe-links --partial --append-verify --human-readable \
  --exclude=__pycache__/ --exclude=*.pyc --exclude=.ipynb_checkpoints/ \
  --exclude=.pixi/ --exclude=.venv/ --exclude=.uv_cache/ --exclude=.uvcache/ \
  --exclude=.stabenv/ --exclude=.nbenv/ --exclude=.boenv/ \
  --exclude=*.lock --exclude=*.pid --exclude=tmp_*/"

mkdir -p "$DEST"
echo "=== $(date -u +%FT%TZ) raw_md pull start :: streams=$STREAMS dest=$DEST ==="

# 1) Worklist: every projects/*/raw_md/* item (path relative to $SRC), full parent
#    path preserved on the dest. The newbench15_gbsa monolith is expanded to one
#    item per workspace/<hash> so the streams actually parallelise.
WORK="$DEST/.pull_worklist.txt"; : > "$WORK"
$SSH "$AWS" "cd $SRC && for p in projects/*/raw_md/*; do echo \"\$p\"; done" | \
while read -r rel; do
  [ -z "$rel" ] && continue
  if [ "$rel" = "$NEWBENCH" ]; then
    $SSH "$AWS" "cd $SRC/$NEWBENCH && ls -1 workspace 2>/dev/null" | \
      while read -r ws; do [ -n "$ws" ] && echo "$NEWBENCH/workspace/$ws"; done >> "$WORK"
    echo "$NEWBENCH/__toplevel_stubs__" >> "$WORK"     # the ~2.3k tiny 5K stubs, one cheap item
  else
    echo "$rel" >> "$WORK"
  fi
done
n=$(wc -l < "$WORK"); echo "work items: $n"

# 2) Pull each item in parallel; full parent path preserved (no flattening).
export AWS SRC DEST SSH RSYNC_OPTS NEWBENCH
pull_one() {
  local rel="$1" parent
  if [ "$rel" = "$NEWBENCH/__toplevel_stubs__" ]; then
    mkdir -p "$DEST/$NEWBENCH"
    echo ">> [stubs] $(date -u +%T)"
    rsync $RSYNC_OPTS --exclude=workspace/ -e "$SSH" \
      "$AWS:$SRC/$NEWBENCH/" "$DEST/$NEWBENCH/" \
      && echo "<< [stubs] OK" || echo "!! [stubs] FAILED"
    return
  fi
  parent="$(dirname "$rel")"; mkdir -p "$DEST/$parent"
  echo ">> [$rel] $(date -u +%T)"
  rsync $RSYNC_OPTS -e "$SSH" "$AWS:$SRC/$rel" "$DEST/$parent/" \
    && echo "<< [$rel] OK" || echo "!! [$rel] FAILED (resume on re-run)"
}
export -f pull_one
cat "$WORK" | xargs -P "$STREAMS" -I{} bash -c 'pull_one "$@"' _ {}

echo "=== $(date -u +%FT%TZ) raw_md pull done -> $DEST ==="
echo "Re-run to resume/verify. Project code+inputs come from STEP 2 in RUN_ON_FT3.md."
