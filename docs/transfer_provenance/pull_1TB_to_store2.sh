#!/bin/bash
# ===== RUN THIS ON FT3 =====  one-shot 1 TB pull into $STORE2.
# AWS->FT3 is firewalled; FT3->AWS works, so FT3 pulls. Resumable: re-run to continue.
# Usage on FT3:   bash pull_1TB_to_store2.sh          (dest = $STORE2/lwa_transfer)
#                 DEST=/path bash pull_1TB_to_store2.sh
set -uo pipefail
AWS=othcxlwa@hpc-compute.dataspace.cesga.es
SRC=/home/otras/hcx/lwa/TRANSFER
DEST="${DEST:-$STORE2/lwa_transfer}"

# rsync uses ssh by default here (FT3->AWS works with defaults). Use a function +
# array so the exclude flags never get word-split (that was the earlier bug).
EXC=(--exclude=.pixi/ --exclude=.venv/ --exclude=.uv_cache/ --exclude=.uvcache/ \
     --exclude=.stabenv/ --exclude=.nbenv/ --exclude=.boenv/ --exclude=__pycache__/ \
     --exclude='*.pyc' --exclude=.ipynb_checkpoints/ --exclude='*.lock' --exclude='*.pid')
R(){ rsync -aHL --copy-unsafe-links --info=progress2 --partial --append-verify "${EXC[@]}" "$@"; }

mkdir -p "$DEST"
echo "=== pull -> $DEST  (target ~887 GB / 1 TB budget) ==="

echo "### STEP 1: discovery-9 trajectories (237 GB)"
mkdir -p "$DEST/00_PRIORITY_discovery9_trajectories"
R "$AWS:$SRC/00_PRIORITY_discovery9_trajectories/" "$DEST/00_PRIORITY_discovery9_trajectories/"

echo "### STEP 2: all project code + results (29 GB)"
R --exclude='raw_md/' "$AWS:$SRC/" "$DEST/"

echo "### STEP 3: factorial + dekois + mps raw, whole (621 GB)"
for proj in 03_gbsa-factorial 04_dekois-screen 05_mps-gpu-benchmark; do
  mkdir -p "$DEST/projects/$proj/raw_md"
  R "$AWS:$SRC/projects/$proj/raw_md/" "$DEST/projects/$proj/raw_md/"
done

echo "### STEP 4: mmbsa200 computed results only, *.h5 (0.6 GB)"
mkdir -p "$DEST/projects/02_affinity-benchmark-mmbsa200/raw_md"
rsync -aHL --copy-unsafe-links --info=progress2 --partial \
  --include='*/' --include='*.h5' --exclude='*' \
  "$AWS:$SRC/projects/02_affinity-benchmark-mmbsa200/raw_md/" \
  "$DEST/projects/02_affinity-benchmark-mmbsa200/raw_md/"

echo "=== done -> $DEST ==="
du -sh "$DEST" 2>/dev/null
echo "NOT pulled (provenance, exceeds budget): mmbsa200 raw 3.1 TB, newbench variant runs ~5.2 TB"
