#!/bin/bash
# Faithful DeepSeek-V4-Flash-DSpark drafter training across TWO GB300 nodes,
# following the two-node DFlash practice (examples/run_kimi_k2.7_code_dflash.sh):
# genuine data-parallelism via sglang DP-attention.
#
# TOPOLOGY (Approach 1, decided in EXECUTION_DOC_2NODE.md §2/§9):
#   WORLD = NNODES*NUM_GPUS = 2*4 = 8 (a GB300 node has 4 GPUs). One logical
#   sglang target engine spans all 8 ranks with DP-attention (attn_tp=1 -> each
#   rank forwards its OWN data shard) + expert-parallel MoE over deepep. The draft
#   is FSDP FULL_SHARD over all 8 ranks; the DSpark pooled-global-mean objective
#   (core/dspark.py) is already correct for this genuine-DP case. Effective global
#   batch = WORLD*BATCH_SIZE, held to the recipe's 512 by ACC = 512/(WORLD*BATCH_SIZE).
#
# IMPORTANT — /scratch is NODE-LOCAL here (not a shared FS). So the repo, the
# ~274GB FP8 teacher, the dataset jsonl, and the tokenized cache must exist on
# BOTH nodes, and checkpoints (written by global rank 0 only) must be synced to
# rank 1 for a clean resume. Two ways to satisfy this:
#   (a) node->node ssh available: run `prepare` on rank 0, then `sync` (rsync to
#       rank 1) — one download.  (b) no node->node ssh: run `prepare` on EACH node.
#
# WORKFLOW (from your LOCAL machine, ssh into EACH node; use tmux):
#   1. On EACH node:            bash examples/run_dsv4_flash_dspark_2node.sh setup
#   2. Data plane (pick a/b):   rank0: ...prepare   then (a) ...sync   OR (b) run
#                               ...prepare on rank 1 too.
#   3. Smoke test first:        SMOKE=1 on BOTH nodes (see below) — validates the
#                               DP-attn+deepep+dsv4+FP8 gate before the long run.
#   4. Launch (both nodes):     rank0: NODE_RANK=0 MASTER_ADDR=10.41.203.7 ...watchdog
#                               rank1: NODE_RANK=1 MASTER_ADDR=10.41.203.7 ...watchdog
#
# KEY ENV (all overridable): NNODES NUM_GPUS MASTER_ADDR MASTER_PORT BATCH_SIZE
#   MEM_FRAC REPORT_TO WANDB_API_KEY HF_TOKEN RANK1_HOST SMOKE

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ---- topology knobs -------------------------------------------------------
NNODES=${NNODES:-2}
NUM_GPUS=${NUM_GPUS:-4}                 # per node — a GB300 node is physically 4 GPUs
WORLD=$((NNODES * NUM_GPUS))            # = 8; tp = ep = dp = WORLD (DP-attention)
MASTER_ADDR=${MASTER_ADDR:-10.41.203.7} # rank-0 routable IP (rendezvous)
MASTER_PORT=${MASTER_PORT:-29500}
RANK1_HOST=${RANK1_HOST:-10.41.203.9}   # for `sync` / `ckptsync` (needs ssh)

# ---- recipe (identical to single-node run_dsv4_flash_dspark.sh) -----------
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}             # per-rank micro-batch; start 1 (see D9/§7)
GLOBAL_BATCH=${GLOBAL_BATCH:-512}
# ACC auto-derived to hold global batch 512 across WORLD data-parallel ranks.
ACC_STEPS=${ACC_STEPS:-$(( GLOBAL_BATCH / (WORLD * BATCH_SIZE) ))}
[ "$ACC_STEPS" -lt 1 ] && ACC_STEPS=1
LEARNING_RATE=${LEARNING_RATE:-6e-4}
# Warmup 5x faster than the DeepSpec recipe (0.04 -> 0.008): warmup_steps =
# warmup_ratio*total_steps, so LR reaches peak 6e-4 in ~1/5 the steps (~210 vs
# ~1050 for a ~26k-step 10-epoch run), then cosine-decays over the rest unchanged.
WARMUP_RATIO=${WARMUP_RATIO:-0.008}
MAX_LEN=${MAX_LEN:-4096}
NUM_ANCHORS=${NUM_ANCHORS:-512}
MEM_FRAC=${MEM_FRAC:-0.5}               # teacher ~34GB/rank at tp8 -> ample room
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-10}
MAX_RESTARTS=${MAX_RESTARTS:-0}         # coordinated whole-job relaunch via watchdog
RDZV_ID=${RDZV_ID:-dsv4-flash-dspark-2node}

# ---- paths / models -------------------------------------------------------
export HF_HOME=${HF_HOME:-/scratch/hf_cache}
export HF_TOKEN=${HF_TOKEN:-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}
TARGET_MODEL=${TARGET_MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/deepseek-v4-flash-dspark.json}
DATA_DIR=$ROOT_DIR/cache/dataset
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/perfectblend_trainsplit.jsonl}
EVAL_DATA=${EVAL_DATA:-$DATA_DIR/perfectblend_eval.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/dsv4-flash-dspark-2node}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-deepseek-v3}

# ---- tracking -------------------------------------------------------------
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-specforge-dspark}
WANDB_NAME=${WANDB_NAME:-dsv4-flash-dspark-2node}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

cmd_setup() {
  cd "$ROOT_DIR"
  pip install --no-deps -e . >/dev/null 2>&1 || true
  pip install accelerate yunchang wandb >/dev/null
  python3 - <<'PY'
import importlib, sys
miss=[m for m in ["torch","sglang","transformers","datasets","accelerate",
                  "yunchang","deep_ep","specforge"]
      if importlib.util.find_spec(m) is None]
if miss: print("PREFLIGHT FAILED, missing:", miss); sys.exit(1)
import torch, sglang
print(f"PREFLIGHT OK  torch={torch.__version__}  sglang={sglang.__version__}  gpus={torch.cuda.device_count()}")
PY
}

cmd_prepare() {
  # Cold start (the env reset wiped models/data). Idempotent. Run on rank 0, then
  # `sync` to rank 1; or run on each node if node->node ssh is unavailable.
  mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
  if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    log "prepare[1/3]: downloading $TARGET_MODEL (~274GB, idempotent)"
    python3 - "$TARGET_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  ->", snapshot_download(repo_id=sys.argv[1]))
PY
  fi
  if [ -f "$TRAIN_DATA" ] && [ -f "$EVAL_DATA" ]; then
    log "prepare[2/3]: train/eval jsonl present, skipping"
  else
    log "prepare[2/3]: building perfectblend, then carving train/eval split"
    # prepare_data.py writes $DATA_DIR/perfectblend_train.jsonl (~1.42M rows).
    [ -f "$DATA_DIR/perfectblend_train.jsonl" ] || \
      python3 "$ROOT_DIR/scripts/prepare_data.py" --dataset perfectblend
    # Deterministic shuffle + carve 3,000 held-out eval (matches EXECUTION_DOC §4f).
    python3 - "$DATA_DIR/perfectblend_train.jsonl" "$TRAIN_DATA" "$EVAL_DATA" "${EVAL_SIZE:-3000}" <<'PY'
import sys, random
src, train_out, eval_out, n_eval = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
lines = open(src).readlines()
random.Random(42).shuffle(lines)
open(eval_out, "w").writelines(lines[:n_eval])
open(train_out, "w").writelines(lines[n_eval:])
print(f"  carved {len(lines)} -> train {len(lines)-n_eval} / eval {n_eval}")
PY
  fi
  log "prepare[3/3]: warming tokenized cache (avoids a cross-node re-tokenize race)"
  python3 - "$TRAIN_DATA" "$MAX_LEN" "$CHAT_TEMPLATE" "$TARGET_MODEL" "$ROOT_DIR/cache" <<'PY'
import hashlib, os, sys
from transformers import AutoTokenizer
from datasets import load_dataset
from specforge.data import build_eagle3_dataset
train, max_len, tmpl, model, cache = sys.argv[1:6]
key = hashlib.md5(f"{train}-{int(max_len)}-{tmpl}-{model}".encode()).hexdigest()
tok = AutoTokenizer.from_pretrained(model)
ds = load_dataset("json", data_files=train)["train"]
build_eagle3_dataset(dataset=ds, tokenizer=tok, chat_template=tmpl, max_length=int(max_len),
                     cache_dir=os.path.join(cache, "processed_dataset"), cache_key=key,
                     num_proc=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 64)))
print("  tokenized cache ready:", key)
PY
  log "prepare: done. Next: `$0 sync` (if node->node ssh) or run prepare on rank 1."
}

cmd_sync() {
  # rank0 -> rank1 one-time bulk sync (needs passwordless ssh to $RANK1_HOST).
  # Pushes repo + HF cache (teacher) + dataset + tokenized cache. ~274GB over IB/eth.
  : "${RANK1_HOST:?set RANK1_HOST}"
  log "sync: repo -> $RANK1_HOST"
  rsync -a --delete --exclude outputs --exclude '.git' "$ROOT_DIR/" "$RANK1_HOST:$ROOT_DIR/"
  log "sync: HF cache (teacher ~274GB) -> $RANK1_HOST  (this is the long pole)"
  rsync -a "$HF_HOME/" "$RANK1_HOST:$HF_HOME/"
  log "sync: dataset + tokenized cache -> $RANK1_HOST"
  rsync -a "$ROOT_DIR/cache/" "$RANK1_HOST:$ROOT_DIR/cache/"
  log "sync: done."
}

cmd_ckptsync() {
  # Detached loop: mirror rank0's checkpoints -> rank1 so either node can --resume
  # (only global rank 0 writes them; /scratch is node-local). Skips dirs <3min old.
  : "${RANK1_HOST:?set RANK1_HOST}"
  while true; do
    for d in "$OUTPUT_DIR"/epoch_*_step_*; do
      [ -d "$d" ] || continue
      [ $(( $(date +%s) - $(stat -c %Y "$d") )) -lt 180 ] && continue
      rsync -a "$d" "$RANK1_HOST:$OUTPUT_DIR/" 2>/dev/null || true
    done
    sleep 120
  done
}

cmd_train() {
  : "${NODE_RANK:?set NODE_RANK=0 (master) or 1}"
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }
  [ "$REPORT_TO" = "wandb" ] && export WANDB_API_KEY=${WANDB_API_KEY:-}

  export PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}
  export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
  export WANDB_DIR=${WANDB_DIR:-$HF_HOME}
  # Leave sglang MoE runner on AUTO (the deepseek_v4 hook resolves it for fp4/fp8);
  # do NOT force it. Keep wo_a bf16 on Blackwell + serialize the big target load
  # (host-RAM guard); offload the fp32 master+moments to CPU (FULL_SHARD is
  # mandatory for the 19.85B MoE draft — see EXECUTION_DOC §4i).
  unset SPECFORGE_SGLANG_MOE_RUNNER_BACKEND || true
  export SGLANG_OPT_FP8_WO_A_GEMM=0
  export SGLANG_EAGER_INPUT_NO_COPY=1
  export SPECFORGE_SGLANG_SERIAL_LOAD=${SPECFORGE_SGLANG_SERIAL_LOAD:-1}
  export SPECFORGE_OFFLOAD_MASTER=${SPECFORGE_OFFLOAD_MASTER:-1}
  export SPECFORGE_FSDP_STRATEGY=${SPECFORGE_FSDP_STRATEGY:-full_shard}
  export SPECFORGE_DRAFT_ATTN=flex
  # DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on the deepep path
  # (it conflicts with sglang pynccl cuMem/NVLS -> ncclCommInitRank invalid usage;
  # this reverses the single-node setting — watch objective transient allocs).
  # Rendezvous/bootstrap over routable Ethernet; NCCL/deepep data plane over the
  # mlx5 IB HCAs (auto-detected). Set the iface holding $MASTER_ADDR (10.41.x).
  # Rendezvous/bootstrap iface = the routable Ethernet that carries this node's
  # 10.41.203.x address. VERIFIED enP22p3s0np0 (rank 0 = 10.41.203.7; only iface
  # with a 10.41.203.x addr — cilium_host/lxc* carry the pod net 10.88.x and must
  # NOT be used). Pinned explicitly; override NCCL_SOCKET_IFNAME per node if its
  # fabric iface name differs. The NCCL/deepep DATA plane uses the mlx5 IB HCAs
  # (auto-detected); this is only the TCP bootstrap/control iface.
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enP22p3s0np0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}

  if ! python3 -c "import specforge, yunchang, accelerate, sglang, deep_ep" 2>/dev/null; then
    echo "ERROR: deps missing on this node. Run: $0 setup" >&2; exit 1
  fi

  local EXTRA=()
  [ "${RESUME:-1}" = "1" ] && EXTRA+=(--resume)
  if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA+=(--max-steps "${SMOKE_STEPS:-4}")
    SAVE_INTERVAL=${SMOKE_SAVE:-2}
    WANDB_NAME="${WANDB_NAME}-smoke"
    log "SMOKE MODE: --max-steps ${SMOKE_STEPS:-4}, save every ${SAVE_INTERVAL}"
  fi
  local tracker=(--report-to "$REPORT_TO")
  [ "$REPORT_TO" = "wandb" ] && tracker+=(--wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME")

  mkdir -p "$OUTPUT_DIR"
  log "train: node_rank=$NODE_RANK/$NNODES world=$WORLD master=$MASTER_ADDR:$MASTER_PORT \
bs=$BATCH_SIZE acc=$ACC_STEPS (eff. global batch=$((WORLD * BATCH_SIZE * ACC_STEPS))) \
iface=$NCCL_SOCKET_IFNAME"

  # STATIC rendezvous (not c10d): node-rank 0 deterministically hosts the TCPStore
  # at MASTER_ADDR:MASTER_PORT. We do NOT use --rdzv-backend c10d here because its
  # host election calls _matches_machine_hostname(MASTER_ADDR), which returns False
  # in this container (hostname s68xhcc4 != fabric IP 10.41.203.7) -> rank 0 wrongly
  # acts as a client to a store nobody hosts -> RendezvousConnectionError. Static +
  # --node-rank sidesteps the hostname match entirely. MAX_RESTARTS=0 + the watchdog
  # still give coordinated whole-job relaunch.
  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --max-restarts "$MAX_RESTARTS" \
    "$ROOT_DIR/scripts/train_dspark_v4.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend sglang --tp-size "$WORLD" \
    --sglang-dp-size "$WORLD" --sglang-ep-size "$WORLD" \
    --sglang-enable-dp-attention --sglang-moe-a2a-backend deepep \
    --sglang-attention-backend dsv4 \
    --sglang-mem-fraction-static "$MEM_FRAC" --sglang-context-length 8192 \
    --embedding-key embed.weight --lm-head-key head.weight \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
    --output-dir "$OUTPUT_DIR" --cache-dir "$ROOT_DIR/cache" \
    --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --accumulation-steps "$ACC_STEPS" \
    --learning-rate "$LEARNING_RATE" --warmup-ratio "$WARMUP_RATIO" --max-grad-norm 1.0 --seed 42 \
    --max-length "$MAX_LEN" --chat-template "$CHAT_TEMPLATE" \
    --num-anchors "$NUM_ANCHORS" --loss-decay-gamma 4.0 \
    --ce-loss-alpha 0.1 --l1-loss-alpha 0.9 --confidence-head-alpha 1.0 \
    --log-interval "$LOG_INTERVAL" --save-interval "$SAVE_INTERVAL" \
    --dataloader-num-workers 4 --build-dataset-num-proc "$SPECFORGE_DATA_NUM_PROC" \
    --dist-timeout 60 "${tracker[@]}" "${EXTRA[@]}"
}

cmd_watchdog() {
  # Per-node supervisor: relaunch `train` (which --resume's) on unexpected death;
  # stop on clean finish (epoch_${NUM_EPOCHS}_step_*); crash-loop guard.
  : "${NODE_RANK:?set NODE_RANK for watchdog}"
  local wlog="$OUTPUT_DIR/watchdog.rank${NODE_RANK}.log"
  local tlog="$OUTPUT_DIR/train.rank${NODE_RANK}.log"
  mkdir -p "$OUTPUT_DIR"
  local -a starts=(); local max=8 window=1800
  echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] started" >> "$wlog"
  while true; do
    if pgrep -f "scripts/train_dspark_v4.py" >/dev/null; then sleep 120; continue; fi
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

case "${1:-}" in
  setup)     cmd_setup ;;
  prepare)   cmd_prepare ;;
  sync)      cmd_sync ;;
  ckptsync)  cmd_ckptsync ;;
  train)     cmd_train ;;
  watchdog)  cmd_watchdog ;;
  *) sed -n '2,45p' "$SCRIPT_PATH"; echo; echo "ERROR: unknown command '${1:-}'"; exit 1 ;;
esac
