#!/bin/bash
# Watchdog for the long Qwen3.6-27B DFlash training run.
# If torchrun/train_dflash dies UNEXPECTEDLY (not a clean finish), relaunch the
# training script, which --resume's from the last checkpoint. Stops on clean
# completion (an epoch_6_step_* checkpoint) and guards against crash-loops.
set -u
ROOT_DIR=/personal/SpecForge
OUT=$ROOT_DIR/outputs/qwen3.6-27b-dflash-nemotron
LAUNCH=$ROOT_DIR/examples/run_qwen3.6_27b_dflash_nemotron_v2.sh
TRAIN_LOG=$OUT/train.log
WLOG=$OUT/watchdog.log
NUM_EPOCHS=6           # clean-finish sentinel: epoch_${NUM_EPOCHS}_step_*
CHECK_EVERY=120        # seconds between liveness checks
MAX_RESTARTS=8         # give up after this many restarts
RESTART_WINDOW=1800    # if >3 restarts within this window, treat as crash-loop

log(){ echo "$(date -u +%FT%TZ) [watchdog] $*" >> "$WLOG"; }

restart_times=()
log "watchdog started (pid $$)"
while true; do
  if pgrep -f "scripts/train_dflash.py" >/dev/null; then
    sleep "$CHECK_EVERY"; continue
  fi
  # process not running — clean finish?
  if ls -d "$OUT"/epoch_${NUM_EPOCHS}_step_* >/dev/null 2>&1; then
    log "clean completion detected (epoch_${NUM_EPOCHS}_step_* exists). exiting."
    exit 0
  fi
  # crash-loop guard
  now=$(date +%s)
  restart_times+=("$now")
  recent=0
  for t in "${restart_times[@]}"; do
    if [ $((now - t)) -le "$RESTART_WINDOW" ]; then recent=$((recent+1)); fi
  done
  if [ "${#restart_times[@]}" -gt "$MAX_RESTARTS" ] || [ "$recent" -gt 3 ]; then
    log "TOO MANY RESTARTS (total=${#restart_times[@]}, recent=$recent). giving up — investigate manually."
    exit 1
  fi
  log "training not running and not finished — relaunching (restart #${#restart_times[@]})"
  sleep 20  # let any lingering GPU allocations clear
  echo "===== WATCHDOG RELAUNCH $(date -u +%FT%TZ) =====" >> "$TRAIN_LOG"
  nohup bash "$LAUNCH" 8 flex_attention >> "$TRAIN_LOG" 2>&1 &
  sleep 90  # give it time to come up before the next liveness check
done
