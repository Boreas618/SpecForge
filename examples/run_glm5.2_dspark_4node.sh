#!/bin/bash
# GLM-5.2 dense DSpark drafter training across FOUR GB300 nodes (16 GPUs), genuine
# data-parallelism via sglang DP-attention (Approach 1, GLM52_EXECUTION_DOC.md §5).
#
# TOPOLOGY: WORLD = NNODES*NUM_GPUS = 4*4 = 16. One logical sglang engine spans all
#   16 ranks with DP-attention (attn_tp=1 -> each rank forwards its OWN data shard)
#   + expert-parallel MoE over deepep. Target = zai-org/GLM-5.2-FP8 (~756GB, ~47GB
#   /rank at tp16, glm_moe_dsa -> sglang "dsa" attention backend, aux+final hidden
#   captured via set_dflash_layers_to_capture). Draft = dense 5-layer DSpark
#   (~3.8B), FSDP SHARD_GRAD_OP over 16 ranks (resident params; no CPU offload).
#   Hidden-state teacher = ONLINE prefill every epoch (no decode, no offline cache;
#   R-DATA-3). Effective global batch = WORLD*BATCH_SIZE, held to 512 via
#   ACC = 512/(WORLD*BATCH_SIZE). The DSpark pooled-global-mean objective
#   (core/dspark.py) is already correct for genuine DP.
#
# IMPORTANT — /scratch is NODE-LOCAL. The repo, the ~756GB FP8 target, the mixed
#   dataset jsonl, and the tokenized cache must exist on ALL FOUR nodes; checkpoints
#   (written by global rank 0 only) must reach the other nodes for a clean resume.
#
# WORKFLOW (from your LOCAL machine, into EACH node; tmux):
#   1. On EACH node:        bash examples/run_glm5.2_dspark_4node.sh setup
#   2. On EACH node:        bash examples/run_glm5.2_dspark_4node.sh prepare
#                           (or run prepare on rank 0 then `sync` if node->node ssh)
#   3. Smoke first:         SMOKE=1 on ALL nodes (validates DP-attn+deepep+dsa+FP8)
#   4. Launch (all nodes):  NODE_RANK=0 MASTER_ADDR=10.41.203.21 ...watchdog  (rank 0)
#                           NODE_RANK=1 MASTER_ADDR=10.41.203.21 ...watchdog  (rank 1)
#                           NODE_RANK=2 ...  NODE_RANK=3 ...
#
# KEY ENV (overridable): NNODES NUM_GPUS MASTER_ADDR MASTER_PORT NODE_RANK BATCH_SIZE
#   MEM_FRAC REPORT_TO WANDB_API_KEY HF_TOKEN SMOKE EVAL_DATASETS_DIR RANK_HOSTS

set -eo pipefail
SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ---- topology knobs -------------------------------------------------------
NNODES=${NNODES:-4}
NUM_GPUS=${NUM_GPUS:-4}                    # per node — a GB300 node is 4 GPUs
WORLD=$((NNODES * NUM_GPUS))               # = 16
# APPROACH 2 (DEFAULT): one per-NODE tp engine (intra-node NVLink target, no
#   DeepEP) + cross-node data-parallel over the draft's FSDP (NCCL, TCP if no IB).
#   -> NNODES unique streams. This is the validated topology in this container
#   (no /dev/infiniband; DeepEP forces IBGDA which hangs cross-container here).
# APPROACH 1 (opt-in): genuine DP via sglang DP-attention + DeepEP across all WORLD
#   ranks -> ~16 unique data streams. REQUIRES cross-node InfiniBand verbs
#   (/dev/infiniband); DeepEP/NVSHMEM cannot init otherwise (IBGDA/IBRC fail).
#   Set APPROACH=1 only after the devbox is relaunched with RDMA.
APPROACH=${APPROACH:-2}
if [ "$APPROACH" = "2" ]; then
  DATA_STREAMS=$NNODES
  # Per-NODE batch: the tp=4 target prefills all 4 samples in one cooperative
  # pass, then TP-batch scatter trains the draft on 1 distinct sample per rank
  # (16 unique streams; draft compute deduplicated). ACC recomputes below.
  BATCH_SIZE=${BATCH_SIZE:-4}
else
  DATA_STREAMS=$WORLD
fi
MASTER_ADDR=${MASTER_ADDR:-10.41.203.21}   # rank-0 routable IP (rendezvous)
MASTER_PORT=${MASTER_PORT:-29500}
# Other nodes (for optional rank0->node checkpoint/data sync). rank order 0..3.
RANK_HOSTS=${RANK_HOSTS:-"10.41.203.23 10.41.202.251 10.41.203.9"}

# ---- recipe (GLM52_EXECUTION_DOC.md) --------------------------------------
NUM_EPOCHS=${NUM_EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-1}                # per-rank micro-batch
GLOBAL_BATCH=${GLOBAL_BATCH:-512}
ACC_STEPS=${ACC_STEPS:-$(( GLOBAL_BATCH / (DATA_STREAMS * BATCH_SIZE) ))}   # A1: 512/16=32 ; A2: 512/4=128
LEARNING_RATE=${LEARNING_RATE:-6e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_LEN=${MAX_LEN:-4096}
BLOCK_SIZE=${BLOCK_SIZE:-7}                # config-owned; passed for the loss-token filter
NUM_ANCHORS=${NUM_ANCHORS:-512}          # DeepSpec canonical (all their dspark configs use 512).
                                         # Was 1024 (RedHat); 512 halves supervision density/step and
                                         # is the likely reason DeepSpec is stable at unscaled 6e-4.
MEM_FRAC=${MEM_FRAC:-0.6}                  # target ~47GB/rank at tp16 -> ample room
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-10}
MAX_RESTARTS=${MAX_RESTARTS:-0}
SGLANG_ATTN_BACKEND=${SGLANG_ATTN_BACKEND:-dsa}   # glm_moe_dsa -> sglang "dsa" backend

# ---- paths / models -------------------------------------------------------
export HF_HOME=${HF_HOME:-/scratch/hf_cache}
# Provide HF_TOKEN via your shell env (do NOT hardcode secrets in the repo):
#   export HF_TOKEN=hf_...    (only needed by `prepare` to download the target)
export HF_TOKEN=${HF_TOKEN:-}
TARGET_MODEL=${TARGET_MODEL:-zai-org/GLM-5.2-FP8}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/glm-5.2-dspark.json}
DATA_DIR=${DATA_DIR:-$HF_HOME/glm52_data}
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/glm52_dspark_train.jsonl}
EVAL_DATA=${EVAL_DATA:-$DATA_DIR/glm52_dspark_eval.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/glm5.2-dspark-4node}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-glm-5.2}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-1500000}
# DeepSpec accept-length benchmark jsonl dir -> in-loop gsm8k accept-length eval
# (built by `prepare`). Set empty to disable the periodic eval.
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-$HF_HOME/glm52_evalds}

# ---- tracking -------------------------------------------------------------
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-specforge-dspark}
WANDB_NAME=${WANDB_NAME:-glm5.2-dspark-4node}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

cmd_setup() {
  cd "$ROOT_DIR"
  pip install --no-deps -e . >/dev/null 2>&1 || true
  pip install accelerate yunchang wandb >/dev/null
  python3 - <<'PY'
import importlib.util, sys
miss=[m for m in ["torch","sglang","transformers","datasets","accelerate",
                  "yunchang","deep_ep","specforge"]
      if importlib.util.find_spec(m) is None]
if miss: print("PREFLIGHT FAILED, missing:", miss); sys.exit(1)
import torch, sglang
print(f"PREFLIGHT OK  torch={torch.__version__}  sglang={sglang.__version__}  gpus={torch.cuda.device_count()}")
PY
}

cmd_prepare() {
  mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
  if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    log "prepare[1/3]: downloading $TARGET_MODEL (~756GB, idempotent)"
    python3 - "$TARGET_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  ->", snapshot_download(repo_id=sys.argv[1], max_workers=16))
PY
  fi
  if [ -f "$TRAIN_DATA" ] && [ -f "$EVAL_DATA" ]; then
    log "prepare[2/3]: train/eval jsonl present, skipping"
  else
    log "prepare[2/3]: building mixed GLM-5.2 corpus (mgoin + codealpaca -> ${TOTAL_SAMPLES})"
    python3 "$ROOT_DIR/scripts/prepare_glm52_dspark_data.py" \
      --output-dir "$DATA_DIR" --total-samples "$TOTAL_SAMPLES" --eval-size "${EVAL_SIZE:-2000}"
  fi
  # gsm8k accept-length benchmark (DeepSpec-exact: openai/gsm8k main/test, each
  # question + the DeepSpec reasoning suffix). Small; built per node (node-local FS).
  if [ -n "$EVAL_DATASETS_DIR" ] && [ ! -f "$EVAL_DATASETS_DIR/gsm8k.jsonl" ]; then
    log "prepare[2b/3]: building DeepSpec gsm8k.jsonl -> $EVAL_DATASETS_DIR"
    mkdir -p "$EVAL_DATASETS_DIR"
    python3 - "$EVAL_DATASETS_DIR/gsm8k.jsonl" <<'PY'
import sys, json
from datasets import load_dataset
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ds = load_dataset("openai/gsm8k", "main", split="test")
with open(sys.argv[1], "w") as f:
    for row in ds:
        f.write(json.dumps({"turns": [f"{row['question']}{SUFFIX}"]}) + "\n")
print("  gsm8k rows:", sum(1 for _ in open(sys.argv[1])))
PY
  fi
  log "prepare[3/3]: warming tokenized cache"
  python3 - "$TRAIN_DATA" "$MAX_LEN" "$CHAT_TEMPLATE" "$TARGET_MODEL" "$ROOT_DIR/cache" <<'PY'
import hashlib, os, sys
from transformers import AutoTokenizer
from datasets import load_dataset
from specforge.data import build_eagle3_dataset
train, max_len, tmpl, model, cache = sys.argv[1:6]
key = hashlib.md5(f"{train}-{int(max_len)}-{tmpl}-{model}".encode()).hexdigest()
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ds = load_dataset("json", data_files=train)["train"]
build_eagle3_dataset(dataset=ds, tokenizer=tok, chat_template=tmpl, max_length=int(max_len),
                     cache_dir=os.path.join(cache, "processed_dataset"), cache_key=key,
                     num_proc=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 64)))
print("  tokenized cache ready:", key)
PY
  log "prepare: done."
}

cmd_ckptsync() {
  # Detached loop: mirror rank0's checkpoints -> all other nodes (only global rank 0
  # writes them; /scratch is node-local). Needs passwordless ssh to $RANK_HOSTS.
  while true; do
    for h in $RANK_HOSTS; do
      for d in "$OUTPUT_DIR"/epoch_*_step_*; do
        [ -d "$d" ] || continue
        [ $(( $(date +%s) - $(stat -c %Y "$d") )) -lt 180 ] && continue
        rsync -a "$d" "$h:$OUTPUT_DIR/" 2>/dev/null || true
      done
    done
    sleep 120
  done
}

cmd_train() {
  : "${NODE_RANK:?set NODE_RANK=0 (master) or 1/2/3}"
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }
  [ "$REPORT_TO" = "wandb" ] && export WANDB_API_KEY=${WANDB_API_KEY:-}

  export PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}
  export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
  export WANDB_DIR=${WANDB_DIR:-$HF_HOME}
  # Settled training behavior (FSDP shard_grad_op, compile OFF, sanitizer ON,
  # chunked objective, TP-batch scatter) is now baked into train_dspark.py /
  # core/dspark.py defaults — no longer env-toggled. --lr-scale and
  # --resume-lr-rewarm-steps are real CLI args passed in the torchrun call below.
  # Kill allocator fragmentation (the OOM report showed 12.8 GB reserved-but-
  # unallocated). Safe here: the in-process sglang engine runs with
  # disable_cuda_graph=True.
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  # Bound host-RAM staging on the big FP8 target load (16 ranks loading in parallel).
  export SPECFORGE_SGLANG_SERIAL_LOAD=${SPECFORGE_SGLANG_SERIAL_LOAD:-1}
  # deepep path (APPROACH=1): do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # (conflicts with sglang pynccl cuMem/NVLS -> ncclCommInitRank invalid usage).
  # Rendezvous/bootstrap iface = the routable Ethernet carrying this node's
  # 10.41.20x.x address (data plane = mlx5 IB HCAs, auto-detected). Override per node.
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
  [ -n "$EVAL_DATASETS_DIR" ] && EXTRA+=(--eval-datasets-dir "$EVAL_DATASETS_DIR")
  local tracker=(--report-to "$REPORT_TO")
  [ "$REPORT_TO" = "wandb" ] && tracker+=(--wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME")

  # Target parallelism per APPROACH (topology note above).
  local tp_size
  local -a PAR_FLAGS
  if [ "$APPROACH" = "2" ]; then
    tp_size=$NUM_GPUS                              # one tp engine per node (intra-node NVLink target)
    PAR_FLAGS=()                                   # no DP-attention, no DeepEP -> no cross-node IB needed
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}   # no IB verbs -> NCCL over TCP for cross-node draft FSDP
    MEM_FRAC=${MEM_FRAC_A2:-0.78}                  # target ~189GB/rank at tp=NUM_GPUS -> high fraction
  else
    tp_size=$WORLD
    PAR_FLAGS=(--sglang-dp-size "$WORLD" --sglang-ep-size "$WORLD"
               --sglang-enable-dp-attention --sglang-moe-a2a-backend deepep)
  fi

  mkdir -p "$OUTPUT_DIR"
  log "train: APPROACH=$APPROACH node_rank=$NODE_RANK/$NNODES world=$WORLD tp=$tp_size \
streams=$DATA_STREAMS master=$MASTER_ADDR:$MASTER_PORT bs=$BATCH_SIZE acc=$ACC_STEPS \
(eff. global batch=$((DATA_STREAMS * BATCH_SIZE * ACC_STEPS))) attn=$SGLANG_ATTN_BACKEND iface=$NCCL_SOCKET_IFNAME"

  # STATIC rendezvous (node-rank 0 hosts the TCPStore at MASTER_ADDR:MASTER_PORT).
  # c10d host election calls _matches_machine_hostname(MASTER_ADDR) which is False
  # in-container (hostname != fabric IP); static + --node-rank sidesteps it.
  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --max-restarts "$MAX_RESTARTS" \
    "$ROOT_DIR/scripts/train_dspark.py" \
    --target-model-path "$TARGET_MODEL" --trust-remote-code \
    --target-model-backend sglang --tp-size "$tp_size" \
    "${PAR_FLAGS[@]}" \
    --sglang-attention-backend "$SGLANG_ATTN_BACKEND" \
    --sglang-mem-fraction-static "$MEM_FRAC" --sglang-context-length 8192 \
    --draft-config-path "$DRAFT_CONFIG" --block-size "$BLOCK_SIZE" \
    --attention-backend flex_attention \
    --markov-rank 256 --enable-confidence-head --confidence-head-with-markov \
    --train-data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
    --output-dir "$OUTPUT_DIR" --cache-dir "$ROOT_DIR/cache" \
    --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --accumulation-steps "$ACC_STEPS" \
    --learning-rate "$LEARNING_RATE" --warmup-ratio "$WARMUP_RATIO" --max-grad-norm 1.0 --seed 42 \
    --lr-scale "${LR_SCALE:-1.0}" --resume-lr-rewarm-steps "${RESUME_LR_REWARM_STEPS:-64}" \
    --max-length "$MAX_LEN" --chat-template "$CHAT_TEMPLATE" \
    --num-anchors "$NUM_ANCHORS" --loss-decay-gamma 4.0 \
    --ce-loss-alpha 0.1 --l1-loss-alpha 0.9 --confidence-head-alpha 1.0 \
    --log-interval "$LOG_INTERVAL" --save-interval "$SAVE_INTERVAL" \
    --evals-per-epoch "${EVALS_PER_EPOCH:-10}" \
    --dataloader-num-workers 0 --build-dataset-num-proc "$SPECFORGE_DATA_NUM_PROC" \
    "${tracker[@]}" "${EXTRA[@]}"
}

cmd_watchdog() {
  : "${NODE_RANK:?set NODE_RANK for watchdog}"
  local wlog="$OUTPUT_DIR/watchdog.rank${NODE_RANK}.log"
  local tlog="$OUTPUT_DIR/train.rank${NODE_RANK}.log"
  mkdir -p "$OUTPUT_DIR"
  local -a starts=(); local max=8 window=1800
  echo "$(date -u +%FT%TZ) [wd r$NODE_RANK] started" >> "$wlog"
  while true; do
    if pgrep -f "scripts/train_dspark.py" >/dev/null; then sleep 120; continue; fi
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

cmd_eval() {
  # Multi-node DeepSpec accept-length eval (dp=NNODES x tp=NUM_GPUS): each node is
  # one tp-sharded target replica, prompts sharded across the 4 replicas, metric
  # sums all-reduced over the dp group -> ~NNODES x throughput vs single node.
  : "${NODE_RANK:?set NODE_RANK=0/1/2/3}"
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }
  export PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enP22p3s0np0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}     # cross-node NCCL over TCP (metric all-reduce only)
  export SPECFORGE_SGLANG_SERIAL_LOAD=${SPECFORGE_SGLANG_SERIAL_LOAD:-1}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  unset SPECFORGE_DRAFT_FLEX_BACKEND SPECFORGE_SGLANG_MOE_RUNNER_BACKEND || true

  local EVAL_CKPT=${EVAL_CKPT:-$OUTPUT_DIR/BEST_epoch2_step101500_gsm8k4.887}
  local EVAL_TASKS=${EVAL_TASKS:-"gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca arena-hard-v2"}
  local EVAL_LIMIT=${EVAL_LIMIT:-0}           # 0 -> DeepSpec per-task upstream caps (gsm8k 500, aime25 30, ...)
  local EVAL_MAX_NEW=${EVAL_MAX_NEW:-2048}    # DeepSpec protocol; thinking-ON needs room for the reasoning chain
  local EVAL_TEMP=${EVAL_TEMP:-1.0}           # DeepSpec default: stochastic rejection-sampling verify
  # Thinking-ON by default (GLM-5.2's deployment mode + our training target).
  # =0 for a thinking-OFF (direct-answer) sweep. MUST match the deployment mode
  # you report; the two give very different accept lengths.
  export SPECFORGE_EVAL_ENABLE_THINKING=${SPECFORGE_EVAL_ENABLE_THINKING:-1}
  local EVAL_TAG=${EVAL_TAG:-$(basename "$EVAL_CKPT")_think${SPECFORGE_EVAL_ENABLE_THINKING}}
  mkdir -p "$OUTPUT_DIR/evals"

  if ! python3 -c "import specforge, sglang" 2>/dev/null; then
    echo "ERROR: deps missing on this node. Run: $0 setup" >&2; exit 1
  fi
  log "eval: node_rank=$NODE_RANK/$NNODES dp=$NNODES tp=$NUM_GPUS ckpt=$EVAL_CKPT \
limit=$EVAL_LIMIT max_new=$EVAL_MAX_NEW temp=$EVAL_TEMP tasks=[$EVAL_TASKS]"

  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --max-restarts 0 \
    "$ROOT_DIR/scripts/eval_dspark_deepspec.py" \
    --target-model-path "$TARGET_MODEL" --target-model-backend sglang --trust-remote-code \
    --tp-size "$NUM_GPUS" \
    --sglang-attention-backend "$SGLANG_ATTN_BACKEND" \
    --sglang-mem-fraction-static "${MEM_FRAC_A2:-0.78}" --sglang-context-length 8192 \
    --draft-checkpoint "$EVAL_CKPT" --draft-attention-backend sdpa \
    --eval-datasets-dir "$EVAL_DATASETS_DIR" \
    --tasks $EVAL_TASKS \
    --limit-per-task "$EVAL_LIMIT" --max-new-tokens "$EVAL_MAX_NEW" \
    --temperature "$EVAL_TEMP" --seed 980406 \
    --output-json "$OUTPUT_DIR/evals/eval9_${EVAL_TAG}.json" \
    --dist-timeout 30
}

case "${1:-}" in
  setup)     cmd_setup ;;
  prepare)   cmd_prepare ;;
  ckptsync)  cmd_ckptsync ;;
  train)     cmd_train ;;
  watchdog)  cmd_watchdog ;;
  eval)      cmd_eval ;;
  *) sed -n '2,45p' "$SCRIPT_PATH"; echo; echo "ERROR: unknown command '${1:-}'"; exit 1 ;;
esac
