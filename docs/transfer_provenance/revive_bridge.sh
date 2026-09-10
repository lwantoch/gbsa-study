#!/bin/bash
# ===== RUN ON FT3 (once) =====  revive the AWS<->FT3 mailbox bridge.
# Restarts the file-sync loop (message flow + heartbeat) and installs a cron
# watchdog so it self-heals across tmux death / node reboot. Idempotent.
#
#   bash revive_bridge.sh          # revive using existing sync script if present
#   FORCE=1 bash revive_bridge.sh  # (re)write the canonical sync script first
#
# Direction: AWS->FT3 is firewalled, so FT3 does all rsync (pull inbox, push
# outbox+heartbeat). Nothing here runs remote commands — it only moves 3 files.
set -uo pipefail

AWS=othcxlwa@hpc-compute.dataspace.cesga.es
REMOTE=/home/otras/hcx/lwa           # AWS-side mailbox dir
LOCAL="$HOME"                        # FT3-side mailbox dir
SESSION=mbox
INTERVAL="${INTERVAL:-20}"           # seconds between sync cycles
SYNC="$LOCAL/ft3_mailbox_sync.sh"
SSH_OPTS="-o ConnectTimeout=15 -o ServerAliveInterval=30 -o BatchMode=yes"

# 1) Ensure the sync loop script exists (write canonical one if missing/FORCE).
if [ ! -f "$SYNC" ] || [ "${FORCE:-0}" = 1 ]; then
  echo "[revive] writing canonical $SYNC"
  cat > "$SYNC" <<SYNCEOF
#!/bin/bash
# AWS<->FT3 mailbox sync loop. Usage: ft3_mailbox_sync.sh loop [interval] | once
set -uo pipefail
AWS=$AWS
REMOTE=$REMOTE
LOCAL=$LOCAL
SSH="ssh $SSH_OPTS"
one_cycle() {
  # pull AWS outbox -> FT3 (messages FROM aws)
  rsync -qt -e "\$SSH" "\$AWS:\$REMOTE/mailbox_from_aws.md" "\$LOCAL/mailbox_from_aws.md" 2>/dev/null || true
  # push FT3 outbox -> AWS (messages FROM ft3), if present
  [ -f "\$LOCAL/mailbox_from_ft3.md" ] && \
    rsync -qt -e "\$SSH" "\$LOCAL/mailbox_from_ft3.md" "\$AWS:\$REMOTE/mailbox_from_ft3.md" 2>/dev/null || true
  # heartbeat -> AWS (liveness signal)
  echo "\$(date -u +%FT%TZ) \$(hostname)" > "\$LOCAL/mailbox_ft3_heartbeat.txt"
  rsync -qt -e "\$SSH" "\$LOCAL/mailbox_ft3_heartbeat.txt" "\$AWS:\$REMOTE/mailbox_ft3_heartbeat.txt" 2>/dev/null || true
}
case "\${1:-loop}" in
  once) one_cycle ;;
  loop) while true; do one_cycle; sleep "\${2:-20}"; done ;;
esac
SYNCEOF
  chmod +x "$SYNC"
fi
chmod +x "$SYNC"

# 2) Probe the tunnel before committing (login-node-only reachability caveat).
if ! ssh $SSH_OPTS "$AWS" 'echo up' >/dev/null 2>&1; then
  echo "[revive] WARN: cannot reach AWS from this host right now."
  echo "         Run this on an FT3 *login* node (compute nodes have no outbound SSH)."
fi

# 3) (Re)start the tmux session running the loop.
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" "cd '$LOCAL' && '$SYNC' loop $INTERVAL"
  echo "[revive] tmux session '$SESSION' started (loop every ${INTERVAL}s)"
else
  echo "[revive] tmux not found — starting nohup background loop instead"
  nohup "$SYNC" loop "$INTERVAL" >"$LOCAL/mbox.log" 2>&1 &
  echo "[revive] pid $!  (log: $LOCAL/mbox.log)"
fi

# 4) Install cron watchdog (self-heal across tmux death / reboot).
WATCH="tmux has-session -t $SESSION 2>/dev/null || tmux new-session -d -s $SESSION 'cd $LOCAL && $SYNC loop $INTERVAL'"
CRON="*/5 * * * * $WATCH"
( crontab -l 2>/dev/null | grep -vF "$SYNC loop"; echo "$CRON" ) | crontab - \
  && echo "[revive] cron watchdog installed (*/5 min)" \
  || echo "[revive] WARN: could not install crontab (add manually): $CRON"

# 5) Immediate cycle so AWS sees a heartbeat and the queued task lands now.
echo "[revive] running one immediate sync cycle..."
"$SYNC" once && echo "[revive] OK — heartbeat pushed; inbox pulled to $LOCAL/mailbox_from_aws.md"

cat <<DONE

=== bridge revived ===
  session : $SESSION (tmux)   interval: ${INTERVAL}s   watchdog: cron */5
  inbox   : $LOCAL/mailbox_from_aws.md   (read AWS's queued 1 TB pull task here)
  outbox  : $LOCAL/mailbox_from_ft3.md   (write replies here; auto-pushed to AWS)
Next: the queued task asks to run  pull_1TB_to_store2.sh  -> \$STORE2/lwa_transfer.
To execute it now:
  rsync -e ssh $AWS:$REMOTE/TRANSFER/pull_1TB_to_store2.sh . && DEST=\$STORE2/lwa_transfer bash pull_1TB_to_store2.sh
DONE
