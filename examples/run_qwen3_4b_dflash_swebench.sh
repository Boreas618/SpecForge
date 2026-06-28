#!/bin/bash

# Train DFlash on agentic SWE-bench trajectories for Qwen3-4B.
#
# This is the training script (Phase B) for the agentic DFlash pipeline.
# It mirrors examples/run_qwen3_4b_dflash_online.sh (same draft recipe:
# block_size=16, gamma=7.0, chat-template=qwen) but consumes the pretokenized
# agentic JSONL produced in Phase A instead of regen chat data.
#
# Unlike the Kimi-K2.7 MoE recipe, Qwen3-4B is DENSE and small: a single node
# runs data-parallel training where each rank holds a full SGLang target
# replica (tp=1) — no expert/data-parallel-attention/deepep plumbing, and no
# embedding-/lm-head-/mask-token-id overrides (standard model layout).
#
# PREREQUISITE: run examples/run_qwen3_4b_dflash_swebench_rollout.sh first to
# generate the pretokenized train/eval JSONL under $DATA_DIR.
#
# USAGE (single node, the common case):
#   1. Run once: `bash examples/run_qwen3_4b_dflash_swebench.sh setup`
#   2. Run once: `bash examples/run_qwen3_4b_dflash_swebench.sh prepare`
#   3. In tmux:  `bash examples/run_qwen3_4b_dflash_swebench.sh watchdog`
#
# MULTI-NODE (optional): set NNODES>1 and, on each node, export NODE_RANK and
# MASTER_ADDR=<rank0-host> before invoking `watchdog` (same rdzv id on all
# nodes), exactly like the Kimi recipe.
#
# KEY ENV (all overridable):
#   NNODES=1  NUM_GPUS=8  MASTER_ADDR=127.0.0.1  MASTER_PORT=29500
#   BATCH_SIZE=1  LEARNING_RATE=6e-4  MEM_FRACTION=0.5  MAX_LENGTH=16384
#   BLOCK_SIZE=16  NUM_ANCHORS=512  LOSS_DECAY_GAMMA=7.0  REPORT_TO=tensorboard
#   WANDB_API_KEY=<...>  (for REPORT_TO=wandb)
#   HF_TOKEN=<...>       (for prepare)

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ---- knobs ----------------------------------------------------------------
NNODES=${NNODES:-1}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
BATCH_SIZE=${BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-6e-4}
NUM_EPOCHS=${NUM_EPOCHS:-6}
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
EVAL_INTERVAL=${EVAL_INTERVAL:-2000}
# Fraction of GPU memory the co-located SGLang target reserves; the rest is for
# FSDP draft training. Lower if you see OOM during prefill of long trajectories.
MEM_FRACTION=${MEM_FRACTION:-0.5}
MAX_RESTARTS=${MAX_RESTARTS:-0}

REPORT_TO=${REPORT_TO:-tensorboard}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DATA_DIR=${DATA_DIR:-$ROOT_DIR/cache/dataset/swebench-agentic-qwen3-4b}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-4b-dflash-swebench}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-4b-dflash.json}
MAX_LENGTH=${MAX_LENGTH:-16384}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-qwen}
BLOCK_SIZE=${BLOCK_SIZE:-16}
NUM_ANCHORS=${NUM_ANCHORS:-512}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flex_attention}
RDZV_ID=${RDZV_ID:-qwen3-4b-dflash-swebench}

TRAIN_JSONL=${TRAIN_JSONL:-$DATA_DIR/swebench_agentic_train.jsonl}
EVAL_JSONL=${EVAL_JSONL:-$DATA_DIR/swebench_agentic_eval.jsonl}

export HF_HOME=${HF_HOME:-/cluster-storage/models}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# --------------------------------------------------------------------------- #

cmd_setup() {
  cd "$ROOT_DIR"
  pip install --no-deps -e .
  pip install accelerate tensorboard
  python3 - <<'PY'
import importlib, sys
miss=[m for m in ["torch","sglang","transformers","datasets","accelerate","specforge"]
      if (importlib.util.find_spec(m) is None)]
if miss: print("PREFLIGHT FAILED, missing:", miss); sys.exit(1)
import torch, sglang
print(f"PREFLIGHT OK  torch={torch.__version__}  sglang={sglang.__version__}")
PY
}

cmd_prepare() {
  mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

  if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    log "prepare[1/2]: downloading $TARGET_MODEL (~8 GB, idempotent)"
    [ -n "${HF_TOKEN:-}" ] || log "  NOTE: HF_TOKEN unset (Qwen3-4B is public, usually fine)."
    python3 - "$TARGET_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  ->", snapshot_download(repo_id=sys.argv[1]))
PY
  else
    log "prepare[1/2]: skipping model download (SKIP_MODEL_DOWNLOAD=1)"
  fi

  if [ -f "$TRAIN_JSONL" ]; then
    log "prepare[2/2]: pretokenized data present ($TRAIN_JSONL), ready to train."
  else
    log "prepare[2/2]: pretokenized data NOT found at $TRAIN_JSONL"
    log "  Generate it first with: bash examples/run_qwen3_4b_dflash_swebench_rollout.sh rollout"
  fi

  log "prepare: done."
  log "  single node:  bash $0 watchdog"
}

cmd_train() {
  NODE_RANK=${NODE_RANK:-0}
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }
  [ -f "$TRAIN_JSONL" ] || { echo "ERROR: train data not found: $TRAIN_JSONL"; echo "Run rollout first: bash examples/run_qwen3_4b_dflash_swebench_rollout.sh rollout"; exit 1; }

  local world=$((NNODES * NUM_GPUS))
  [ "$REPORT_TO" = "wandb" ] && export WANDB_API_KEY=${WANDB_API_KEY:?set WANDB_API_KEY for wandb reporting}
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
  export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
  export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
  export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
  # Only needed for the multi-node case; harmless on a single node.
  if [ "$NNODES" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
  fi

  if ! python3 -c "import specforge, accelerate, sglang" 2>/dev/null; then
    echo "ERROR: training deps missing on this node. Run: $0 setup" >&2; exit 1
  fi

  local tracker=(--report-to "$REPORT_TO")
  [ "$REPORT_TO" = "wandb" ] && tracker+=(--wandb-project specforge-dflash --wandb-name qwen3-4b-dflash-swebench)

  local eval_args=()
  [ -f "$EVAL_JSONL" ] && eval_args=(--eval-data-path "$EVAL_JSONL")

  mkdir -p "$OUTPUT_DIR"
  log "train: node_rank=$NODE_RANK/$NNODES world=$world master=$MASTER_ADDR:$MASTER_PORT \
batch=$BATCH_SIZE lr=$LEARNING_RATE max_len=$MAX_LENGTH (data=pretokenized swebench, dense data-parallel)"

  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --rdzv-backend c10d --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
    --rdzv-id "$RDZV_ID" --max-restarts "$MAX_RESTARTS" \
    "$ROOT_DIR/scripts/train_dflash.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend sglang --trust-remote-code \
    --sglang-mem-fraction-static "$MEM_FRACTION" \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_JSONL" \
    "${eval_args[@]}" \
    --data-format pretokenized \
    --output-dir "$OUTPUT_DIR" --cache-dir "$ROOT_DIR/cache" \
    --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
    --warmup-ratio 0.04 --max-grad-norm 1.0 --max-length "$MAX_LENGTH" \
    --chat-template "$CHAT_TEMPLATE" --attention-backend "$ATTENTION_BACKEND" \
    --block-size "$BLOCK_SIZE" --num-anchors "$NUM_ANCHORS" --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --dataloader-num-workers 0 --log-interval "$LOG_INTERVAL" --save-interval "$SAVE_INTERVAL" --eval-interval "$EVAL_INTERVAL" \
    --dist-timeout 60 "${tracker[@]}" --resume
}

cmd_watchdog() {
  NODE_RANK=${NODE_RANK:-0}
  local wlog="$OUTPUT_DIR/watchdog.rank${NODE_RANK}.log"
  local tlog="$OUTPUT_DIR/train.rank${NODE_RANK}.log"
  mkdir -p "$OUTPUT_DIR"
  local -a starts=(); local max=8 window=1800
  echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] started" >> "$wlog"
  while true; do
    if pgrep -f "scripts/train_dflash.py" >/dev/null; then sleep 120; continue; fi
    if ls -d "$OUTPUT_DIR"/epoch_${NUM_EPOCHS}_step_* >/dev/null 2>&1; then
      echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] clean finish, exit" >> "$wlog"; exit 0
    fi
    local now; now=$(date +%s); starts+=("$now"); local recent=0
    for t in "${starts[@]}"; do [ $((now - t)) -le "$window" ] && recent=$((recent+1)); done
    if [ "${#starts[@]}" -gt "$max" ] || [ "$recent" -gt 3 ]; then
      echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] crash-loop, giving up" >> "$wlog"; exit 1
    fi
    echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] relaunch #${#starts[@]}" >> "$wlog"
    sleep 20
    echo "===== WATCHDOG RELAUNCH $(date -u +%FT%TZ) =====" >> "$tlog"
    bash "$SCRIPT_PATH" train >> "$tlog" 2>&1 &
    sleep 120
  done
}

# --------------------------------------------------------------------------- #

case "${1:-}" in
  setup)    cmd_setup ;;
  prepare)  cmd_prepare ;;
  train)    cmd_train ;;
  watchdog) cmd_watchdog ;;
  *)
    echo "Usage: bash $0 {setup|prepare|train|watchdog}"
    echo ""
    echo "  setup    Install SpecForge + training deps"
    echo "  prepare  Download target model + check for pretokenized data"
    echo "  train    Run torchrun training (called by watchdog)"
    echo "  watchdog Supervisor that (re)launches train --resume (run in tmux)"
    echo ""
    echo "Prerequisite: pretokenized JSONL from examples/run_qwen3_4b_dflash_swebench_rollout.sh"
    if [ -n "${1:-}" ]; then echo "ERROR: unknown command '${1}'"; exit 1; fi
    ;;
esac
