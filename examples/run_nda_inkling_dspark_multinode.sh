#!/bin/bash
# NDA/Inkling dense DSpark drafter training across N GB300 nodes (4 GPUs/node),
# N in 1..16 — a full NVL72 rack is 16 trays x 4 = 64 ranks in ONE NVLink domain.
#
# TOPOLOGY (Approach 2, scales with NNODES): one per-NODE tp=4 sglang engine
#   (947B NVFP4 target, ~138GB weights/rank) prefills that node's batch; the
#   draft trains FSDP SHARD_GRAD_OP over ALL WORLD ranks. Cross-tray FSDP
#   traffic rides the NVL72 Multi-Node NVLink fabric (MNNVL/IMEX; NCCL >= 2.28
#   auto-detects it) — TCP is only the torchrun rendezvous channel.
#   DATA_STREAMS = NNODES engines; TP-batch scatter gives each rank a distinct
#   1/tp slice of its node batch => WORLD unique training streams.
#   Effective global batch = DATA_STREAMS*BATCH_SIZE*ACC, held at GLOBAL_BATCH.
#
# TARGET REQUIREMENTS (asserted by the model constructor, inkling.py:766-770):
#   SGLANG_ENABLE_UNIFIED_RADIX_TREE=1, radix cache ON, hybrid SWA memory ON,
#   mamba_radix_cache_strategy=extra_buffer. Attention backend MUST be fa4.
#
# IMPORTANT — /scratch is NODE-LOCAL. The three repos (nda_sgl fork, SpecForge,
#   configs), the 552GB NVFP4 checkpoint, the train/eval jsonl, and the
#   tokenized cache must exist on EVERY node; checkpoints (written by global
#   rank 0 only) must reach other nodes before any cross-node resume.
#
# WORKFLOW (from your LOCAL machine, into EACH node; tmux):
#   0. Transport repos + checkpoint + regen data per node (owner pipeline).
#   1. On EACH node:   bash examples/run_nda_inkling_dspark_multinode.sh setup
#   2. On rank 0:      bash ... prepare       (builds train/eval jsonl + gsm8k
#                      + warms the tokenized cache; rerun per node, or copy
#                      $DATA_DIR + cache/ to the other nodes)
#   3. Probe first:    bash ... probe         (single node; capture parity)
#   4. Smoke:          SMOKE=1 NODE_RANK=k MASTER_ADDR=<rank0-ip> ... train (all nodes)
#   5. Launch:         NODE_RANK=k MASTER_ADDR=<rank0-ip> ... train        (all nodes)
#
# KEY ENV: NNODES NUM_GPUS MASTER_ADDR MASTER_PORT NODE_RANK BATCH_SIZE MEM_FRAC
#   GLOBAL_BATCH REPORT_TO WANDB_API_KEY SMOKE EVAL_DATASETS_DIR RANK_HOSTS
#   TARGET_MODEL REGEN_GLOB NDA_SGL_FORK

set -eo pipefail
SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# ---- topology knobs -------------------------------------------------------
NNODES=${NNODES:-4}                        # scale 1..16 (full NVL72 rack = 16)
NUM_GPUS=${NUM_GPUS:-4}                    # per node — a GB300 tray is 4 GPUs
WORLD=$((NNODES * NUM_GPUS))
DATA_STREAMS=$NNODES                       # one tp engine (= one stream) per node
MASTER_ADDR=${MASTER_ADDR:-}               # rank-0 routable IP (rendezvous)
MASTER_PORT=${MASTER_PORT:-29500}
# Other nodes (for optional rank0->node checkpoint sync), space-separated IPs
# in rank order 1..NNODES-1.
RANK_HOSTS=${RANK_HOSTS:-}

# ---- the sglang FORK (nda_sgl) must be imported, not any pip sglang --------
# Named NDA_SGL_FORK because sglang's environ shim warns on ANY exported SGL_*
# variable; the legacy SGL_FORK is still honored as a fallback.
SGL_FORK=${NDA_SGL_FORK:-${SGL_FORK:-/scratch/yi/nda_sgl}}   # checkout of Boreas618/nda_sgl

# ---- recipe (EXECUTION_DOC.md Step-3 locked plan) ---------------------------
NUM_EPOCHS=${NUM_EPOCHS:-10}               # on paper; stop-at-convergence
BATCH_SIZE=${BATCH_SIZE:-4}                # per-NODE batch (tp engine prefills all;
                                           # TP-batch scatter trains 1/rank). Probe 8
                                           # at smoke if memory allows (owner: fast+correct).
GLOBAL_BATCH=${GLOBAL_BATCH:-512}
if (( GLOBAL_BATCH % (DATA_STREAMS * BATCH_SIZE) != 0 )); then
  echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH not divisible by DATA_STREAMS*BATCH_SIZE=$((DATA_STREAMS*BATCH_SIZE))" >&2
  exit 1
fi
ACC_STEPS=${ACC_STEPS:-$(( GLOBAL_BATCH / (DATA_STREAMS * BATCH_SIZE) ))}  # 4n/b4:32  16n/b4:8  16n/b8:4
LEARNING_RATE=${LEARNING_RATE:-6e-4}
# 0.5 (was 1.0): the documented GLM edge-of-stability fallback, now observed
# on Inkling too — first 4-node run collapsed at optimizer step ~100-150
# (mid-warmup, LR ~3e-4): windowed mean_acc rose to 0.053 by micro-step 3k,
# then fell to ~0.006 and flatlined while loss kept descending (degenerate
# easy-token fit). train_dspark's own comment documents 0.5 as the intended
# default; the launcher was overriding it to 1.0.
LR_SCALE=${LR_SCALE:-1.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_LEN=${MAX_LEN:-4096}                   # owner decision Q-F1
BLOCK_SIZE=${BLOCK_SIZE:-7}
NUM_ANCHORS=${NUM_ANCHORS:-512}
# 4.0 pairs with the historical block 7; block 15 uses 28/3 so the block-end
# weight exp(-14/(28/3)) = exp(-1.5) matches the block-7/gamma-4 normalization.
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-4.0}
FIXED_LR=${FIXED_LR:-0}                     # >0: constant LR, cosine disabled
MEM_FRAC=${MEM_FRAC:-0.65}                 # ~138GB/rank weights at tp4; calibrate at smoke
# 250 (was 500): the engine-wedge recoveries resume from the last checkpoint;
# at ~0.5-1 s/micro-step a save every 250 steps caps the loss at ~4 min of
# work (save itself takes ~1-2 s, negligible).
SAVE_INTERVAL=${SAVE_INTERVAL:-250}
LOG_INTERVAL=${LOG_INTERVAL:-10}
MAX_RESTARTS=${MAX_RESTARTS:-0}
EVALS_PER_EPOCH=${EVALS_PER_EPOCH:-4}      # owner: 4 evals/epoch, 32 prompts
EVAL_LIMIT=${EVAL_LIMIT:-32}

# ---- Inkling engine flags (serve parity; constructor-asserted) --------------
SGLANG_ATTN_BACKEND=${SGLANG_ATTN_BACKEND:-fa4}          # HARD requirement
SGLANG_PAGE_SIZE=${SGLANG_PAGE_SIZE:-128}
SGLANG_MAMBA_STRATEGY=${SGLANG_MAMBA_STRATEGY:-extra_buffer}
SGLANG_MAX_MAMBA_CACHE=${SGLANG_MAX_MAMBA_CACHE:-64}     # training batch <= 8
SGLANG_SWA_FULL_RATIO=${SGLANG_SWA_FULL_RATIO:-0.2}
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-modelopt_fp4}
SGLANG_FP4_GEMM=${SGLANG_FP4_GEMM:-flashinfer_trtllm}
# trtllm_routed is the ONLY correct backend for this checkpoint: its MoE w13
# weights are stored INTERLEAVED (inference_moe_w13_interleaved) and only the
# trtllm routed path consumes that layout. flashinfer_cutlass reads them flat
# -> a scrambled but self-consistent forward: agreement probe (folded head on
# captured final hiddens vs recorded corpus tokens, shift +1) = 0.8448 under
# trtllm_routed vs 0.0033 under cutlass. The cutlass detour (39067c1) was a
# wedge-hunt mitigation; the wedge's true cause was asymmetric resume (fixed
# in f9f0b3a), and cutlass silently poisoned the L1/tau/agree training
# signals instead — the windowed-acc collapse at ~2-3k steps.
SGLANG_MOE_RUNNER=${SGLANG_MOE_RUNNER:-flashinfer_trtllm_routed}
CTX_LEN=${CTX_LEN:-$((MAX_LEN + 512))}

# ---- paths / models ---------------------------------------------------------
TARGET_MODEL=${TARGET_MODEL:-/scratch/yi/model-share-v1-nvfp4}
# Override with configs/nda-inkling-dspark-gqa16.json for the quality-oriented
# GQA16 draft (64 query heads / 16 KV heads, matching Qwen3's 4:1 ratio);
# the existing MHA draft remains the default for checkpoint compatibility.
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/nda-inkling-dspark.json}
DATA_DIR=${DATA_DIR:-/scratch/yi/nda_data}
REGEN_GLOB=${REGEN_GLOB:-'/scratch/yi/regen_out/regen.shard*of*.jsonl'}  # success files only
TRAIN_DATA=${TRAIN_DATA:-$DATA_DIR/nda_inkling_dspark_train.jsonl}
EVAL_DATA=${EVAL_DATA:-$DATA_DIR/nda_inkling_dspark_eval.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/nda-inkling-dspark}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-nda-inkling-thinking}
EVAL_DATASETS_DIR=${EVAL_DATASETS_DIR:-$DATA_DIR/evalds}
MASK_TOKEN_ID=${MASK_TOKEN_ID:-200064}     # padded-vocab slot (tokenizer-unreachable)

# ---- tracking ---------------------------------------------------------------
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-specforge-dspark}
WANDB_NAME=${WANDB_NAME:-nda-inkling-dspark-${NNODES}node}

# ---- continual phase (CONTINUAL=1) ------------------------------------------
# Adapt the converged epoch-3 pretrain draft to the new_weights_09/_099 regen
# corpus. Recipe: fresh cosine at 1/3 peak LR (converged drafts are
# edge-of-stability sensitive at the full 6e-4 — GLM collapsed twice from
# converged states), 3 planned epochs with stop-at-convergence, everything
# serve-parity unchanged. Init weights arrive as $OUTPUT_DIR/epoch_0_step_0
# (weights-only, no training_state.pt -> fresh optimizer/schedule) fetched
# from the HF preview repo by scripts/continual_prep_node.sh; the
# world-consistent resume guard then enforces identical init on every node.
if [ "${CONTINUAL:-0}" = "1" ]; then
  # NEW target weights (tml-model-share @ nvfp4 revision) — the new_weights_*
  # corpus was regenerated by these weights; training/capture must verify
  # against the same target. Same architecture/vocab/muP as v1 (asserted by
  # the constructor); agreement probe re-gated after the switch.
  TARGET_MODEL=${TARGET_MODEL_OVERRIDE:-/scratch/yi/tml-model-share-nvfp4}
  LEARNING_RATE=2e-4
  NUM_EPOCHS=3
  DATA_DIR=/scratch/yi/nda_data_continual
  TRAIN_DATA=$DATA_DIR/nda_inkling_dspark_train.jsonl
  EVAL_DATA=$DATA_DIR/nda_inkling_dspark_eval.jsonl
  EVAL_DATASETS_DIR=$DATA_DIR/evalds
  OUTPUT_DIR=$ROOT_DIR/outputs/nda-inkling-dspark-continual-v1
  WANDB_NAME="nda-inkling-dspark-${NNODES}node-continual-v1"
fi

log() { echo "[$(date -u +%FT%TZ)] $*"; }

env_common() {
  # The FORK must win the import race over any pip sglang.
  export PYTHONPATH="$SGL_FORK/python:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  # NCCL flight recorder: a wedged collective is otherwise SILENT (log just
  # freezes). On a stall every rank can dump its recent collectives —
  # which op, which peers missing — to ~/.cache/torch/comm_lib_trace_rank_<r>
  # (torch 2.11 path; TORCH_NCCL_TRACE_BUFFER_SIZE is honored but deprecated).
  export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}
  export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
  # Cross-communicator launch ordering (NCCL >= 2.26): the engine's per-node
  # TP communicator interleaves with the FSDP world communicator on every
  # micro-step; enforce a device-wide launch order so mixed-communicator
  # kernels cannot deadlock. Unknown-var-safe on older NCCL.
  export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
  # Owner hint (2026-07-14): the fork's shared-experts-on-alt-stream overlap
  # (default ON) races the trtllm routed FP4 MoE kernel under training's
  # prefill cadence — the twice-captured all-4-ranks spin inside
  # trtllm_fp4_block_scale_routed_moe. Disable overlap for training; the
  # serving-parity MoE runner (flashinfer_trtllm_routed) stays unchanged.
  export SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP=${SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP:-0}
  # sglang.__file__ = <fork>/python/sglang/__init__.py -> two dirnames up is
  # <fork>/python (NOT <fork>): compare against that.
  python3 - "$SGL_FORK" <<'PY'
import sys, sglang, os
fork_python = os.path.realpath(os.path.join(sys.argv[1], "python"))
got = os.path.realpath(os.path.dirname(os.path.dirname(sglang.__file__)))
assert got == fork_python, (
    f"sglang imports from {got}, expected {fork_python} — fix PYTHONPATH/NDA_SGL_FORK"
)
print(f"PYTHONPATH OK: sglang <- {got}")
PY
  # Inkling model-constructor asserts (inkling.py:766-770).
  export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
  export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
  export SPECFORGE_SGLANG_SERIAL_LOAD=${SPECFORGE_SGLANG_SERIAL_LOAD:-1}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  # SConv prefix state cannot ride the plain-RadixCache KV session -> exact
  # full re-prefill for the accept-length eval (in-loop and standalone).
  export SPECFORGE_EVAL_KV_REUSE=0
  # Rendezvous/bootstrap iface = the routable Ethernet of this tray. The heavy
  # cross-tray traffic (draft FSDP reduce-scatter/all-gather) rides the NVL72
  # MNNVL NVLink fabric, which NCCL auto-detects (IMEX must be up:
  # /dev/nvidia-caps-imex-channels/channel0). NCCL_IB_DISABLE only disables the
  # IB transport and does NOT affect NVLink/MNNVL.
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enP22p3s0np0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
  export WANDB_DIR=${WANDB_DIR:-$DATA_DIR}
}

sglang_flags() {
  # Engine flags shared by train / probe / eval (Inkling serve parity).
  echo "--sglang-attention-backend $SGLANG_ATTN_BACKEND \
        --sglang-mem-fraction-static $MEM_FRAC --sglang-context-length $CTX_LEN \
        --sglang-page-size $SGLANG_PAGE_SIZE \
        --sglang-mamba-radix-cache-strategy $SGLANG_MAMBA_STRATEGY \
        --sglang-max-mamba-cache-size $SGLANG_MAX_MAMBA_CACHE \
        --sglang-swa-full-tokens-ratio $SGLANG_SWA_FULL_RATIO \
        --sglang-quantization $SGLANG_QUANTIZATION \
        --sglang-fp4-gemm-runner-backend $SGLANG_FP4_GEMM \
        --sglang-moe-runner-backend $SGLANG_MOE_RUNNER"
}

cmd_setup() {
  cd "$ROOT_DIR"
  pip install --no-deps -e . >/dev/null 2>&1 || true
  # yunchang: imported at specforge module load (llama3_eagle) even though the
  # DSpark path never uses it — required for ANY specforge import.
  pip install accelerate yunchang wandb >/dev/null
  # helion: imported by the fork's inkling MoE kernels (tml/kernels/inkling_moe).
  # Without it the model registry SILENTLY skips inkling.py and arch resolution
  # fails much later with "InklingForConditionalGeneration is not a registered
  # model". --no-deps so pip cannot touch the pinned torch/triton stack.
  pip install --no-deps helion >/dev/null
  env_common
  python3 - <<'PY'
import importlib.util, sys
miss=[m for m in ["torch","sglang","transformers","datasets","accelerate","yunchang","specforge","helion"]
      if importlib.util.find_spec(m) is None]
if miss: print("PREFLIGHT FAILED, missing:", miss); sys.exit(1)
import torch, sglang
# The registry scan swallows per-module ImportErrors; import inkling directly
# so a broken model module fails HERE, not as a late arch-resolution error.
import sglang.srt.models.inkling
print(f"PREFLIGHT OK  torch={torch.__version__}  gpus={torch.cuda.device_count()}")
PY
}

cmd_prepare() {
  env_common
  mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
  if [ -f "$TRAIN_DATA" ] && [ -f "$EVAL_DATA" ]; then
    log "prepare[1/3]: train/eval jsonl present, skipping"
  else
    log "prepare[1/3]: assembling corpus from regen outputs: $REGEN_GLOB"
    python3 "$ROOT_DIR/scripts/prepare_nda_inkling_dspark_data.py" \
      --inputs $REGEN_GLOB --output-dir "$DATA_DIR" --eval-size "${EVAL_SIZE:-2000}"
  fi
  if [ ! -f "$EVAL_DATASETS_DIR/gsm8k.jsonl" ]; then
    log "prepare[2/3]: building DeepSpec gsm8k.jsonl -> $EVAL_DATASETS_DIR"
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
  if [ ! -f "$EVAL_DATASETS_DIR/aime25.jsonl" ]; then
    log "prepare[2b/3]: building DeepSpec aime25.jsonl -> $EVAL_DATASETS_DIR"
    python3 - "$EVAL_DATASETS_DIR/aime25.jsonl" <<'PY'
import sys, json
from datasets import load_dataset
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
rows = []
# Primary source, with a fallback mirror (both carry the 30 AIME-2025 problems).
try:
    ds = load_dataset("math-ai/aime25", split="test")
    rows = [r["problem"] for r in ds]
except Exception:
    for cfg in ("AIME2025-I", "AIME2025-II"):
        ds = load_dataset("opencompass/AIME2025", cfg, split="test")
        rows += [r["question"] for r in ds]
assert len(rows) >= 30, f"expected >=30 AIME-2025 problems, got {len(rows)}"
with open(sys.argv[1], "w") as f:
    for p in rows:
        f.write(json.dumps({"turns": [f"{p}{SUFFIX}"]}) + "\n")
print("  aime25 rows:", len(rows))
PY
  fi
  log "prepare[3/3]: warming tokenized cache (max_len=$MAX_LEN, template=$CHAT_TEMPLATE)"
  python3 - "$TRAIN_DATA" "$MAX_LEN" "$CHAT_TEMPLATE" "$TARGET_MODEL" "$ROOT_DIR/cache" <<'PY'
import hashlib, os, sys
from transformers import AutoTokenizer
from datasets import load_dataset
from specforge.data import build_eagle3_dataset
from specforge.data.template import packaged_chat_template_hash
from specforge.utils import file_content_hash
train, max_len, tmpl, model, cache = sys.argv[1:6]
th = packaged_chat_template_hash(tmpl) or "none"
# MUST mirror train_dspark.py's cache_params_string exactly (incl. the data
# content hash) or training misses the cache prepare just warmed.
key = hashlib.md5(f"{train}-{int(max_len)}-{tmpl}-{th}-{model}-maskv2-glm-thinkhybrid-{file_content_hash(train)}".encode()).hexdigest()
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
ds = load_dataset("json", data_files=train)["train"]
build_eagle3_dataset(dataset=ds, tokenizer=tok, chat_template=tmpl, max_length=int(max_len),
                     cache_dir=os.path.join(cache, "processed_dataset"), cache_key=key,
                     num_proc=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 64)))
print("  tokenized cache ready:", key)
PY
  log "prepare: done."
}

cmd_probe() {
  # Single-node capture-plumbing parity probe (EXECUTION_DOC P9). Run BEFORE the
  # first smoke; requires this node's 4 GPUs free.
  env_common
  torchrun --nnodes 1 --nproc-per-node "$NUM_GPUS" --master-port "$MASTER_PORT" \
    "$ROOT_DIR/scripts/probe_nda_inkling_capture.py" \
    --target-model-path "$TARGET_MODEL" --tp-size "$NUM_GPUS" \
    --chat-template "$CHAT_TEMPLATE" \
    --embedding-key model.llm.embed.weight --lm-head-key model.llm.unembed.weight \
    $(sglang_flags)
}

cmd_train() {
  : "${NODE_RANK:?set NODE_RANK=0 (master) or 1..$((NNODES-1))}"
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }
  [ "$REPORT_TO" = "wandb" ] && export WANDB_API_KEY=${WANDB_API_KEY:-}
  env_common

  local EXTRA=()
  [ "${RESUME:-1}" = "1" ] && EXTRA+=(--resume)
  if [ "${SMOKE:-0}" = "1" ]; then
    EXTRA+=(--max-steps "${SMOKE_STEPS:-4}")
    SAVE_INTERVAL=${SMOKE_SAVE:-2}
    WANDB_NAME="${WANDB_NAME}-smoke"
    log "SMOKE MODE: --max-steps ${SMOKE_STEPS:-4}, save every ${SAVE_INTERVAL}"
  fi
  # EVALS_PER_EPOCH=0 must fully disable in-loop eval: --eval-interval has an
  # argparse default (1000) that survives when the per-epoch block is skipped.
  [ -n "$EVAL_DATASETS_DIR" ] && [ "${EVALS_PER_EPOCH}" != "0" ] && EXTRA+=(--eval-datasets-dir "$EVAL_DATASETS_DIR")
  local tracker=(--report-to "$REPORT_TO")
  [ "$REPORT_TO" = "wandb" ] && tracker+=(--wandb-project "$WANDB_PROJECT" --wandb-name "$WANDB_NAME")

  mkdir -p "$OUTPUT_DIR"
  log "train: nodes=$NNODES node_rank=$NODE_RANK world=$WORLD tp=$NUM_GPUS \
streams=$DATA_STREAMS master=$MASTER_ADDR:$MASTER_PORT bs=$BATCH_SIZE acc=$ACC_STEPS \
(global batch=$((DATA_STREAMS * BATCH_SIZE * ACC_STEPS))) max_len=$MAX_LEN attn=$SGLANG_ATTN_BACKEND"

  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --max-restarts "$MAX_RESTARTS" \
    "$ROOT_DIR/scripts/train_dspark.py" \
    --target-model-path "$TARGET_MODEL" --trust-remote-code \
    --target-model-backend sglang --tp-size "$NUM_GPUS" \
    --embedding-key model.llm.embed.weight --lm-head-key model.llm.unembed.weight \
    $(sglang_flags) \
    --draft-config-path "$DRAFT_CONFIG" --block-size "$BLOCK_SIZE" \
    --attention-backend flex_attention \
    --mask-token-id "$MASK_TOKEN_ID" \
    --markov-rank 256 --enable-confidence-head --confidence-head-with-markov \
    --train-data-path "$TRAIN_DATA" --eval-data-path "$EVAL_DATA" \
    --output-dir "$OUTPUT_DIR" --cache-dir "$ROOT_DIR/cache" \
    --num-epochs "$NUM_EPOCHS" --batch-size "$BATCH_SIZE" --accumulation-steps "$ACC_STEPS" \
    --learning-rate "$LEARNING_RATE" --warmup-ratio "$WARMUP_RATIO" --max-grad-norm 1.0 --seed 42 \
    --lr-scale "$LR_SCALE" --resume-lr-rewarm-steps "${RESUME_LR_REWARM_STEPS:-64}" \
    --fixed-lr "$FIXED_LR" \
    --max-length "$MAX_LEN" --chat-template "$CHAT_TEMPLATE" \
    --num-anchors "$NUM_ANCHORS" --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --ce-loss-alpha 0.1 --l1-loss-alpha 0.9 --confidence-head-alpha 1.0 \
    --log-interval "$LOG_INTERVAL" --save-interval "$SAVE_INTERVAL" \
    --evals-per-epoch "$EVALS_PER_EPOCH" --eval-limit-per-task "$EVAL_LIMIT" \
    --eval-max-new-tokens "${EVAL_MAX_NEW_INLOOP:-1024}" \
    --dataloader-num-workers 0 --build-dataset-num-proc "$SPECFORGE_DATA_NUM_PROC" \
    "${tracker[@]}" "${EXTRA[@]}"
}

cmd_eval() {
  # Multi-node DeepSpec accept-length eval: each node is one tp=4 target
  # replica (dp=NNODES), prompts sharded across replicas, sums all-reduced.
  # KV-reuse is OFF (env_common) -> exact O(n^2) re-prefill verify.
  : "${NODE_RANK:?set NODE_RANK=0..$((NNODES-1))}"
  [ -n "$MASTER_ADDR" ] || { echo "ERROR: MASTER_ADDR unset"; exit 1; }
  env_common
  local EVAL_CKPT=${EVAL_CKPT:?set EVAL_CKPT=<draft checkpoint dir>}
  local EVAL_TASKS=${EVAL_TASKS:-"gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca arena-hard-v2"}
  local EVAL_LIMIT_SWEEP=${EVAL_LIMIT_SWEEP:-0}   # 0 -> DeepSpec upstream caps
  local EVAL_MAX_NEW=${EVAL_MAX_NEW:-2048}        # DeepSpec protocol (Q-H1)
  local EVAL_TEMP=${EVAL_TEMP:-1.0}               # DeepSpec stochastic (Q-H2)
  local EVAL_TAG=${EVAL_TAG:-$(basename "$EVAL_CKPT")_temp${EVAL_TEMP}}
  mkdir -p "$OUTPUT_DIR/evals"

  log "eval: node_rank=$NODE_RANK/$NNODES dp=$NNODES tp=$NUM_GPUS ckpt=$EVAL_CKPT \
limit=$EVAL_LIMIT_SWEEP max_new=$EVAL_MAX_NEW temp=$EVAL_TEMP tasks=[$EVAL_TASKS]"

  torchrun \
    --nnodes "$NNODES" --nproc-per-node "$NUM_GPUS" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --max-restarts 0 \
    "$ROOT_DIR/scripts/eval_dspark_deepspec.py" \
    --target-model-path "$TARGET_MODEL" --target-model-backend sglang --trust-remote-code \
    --tp-size "$NUM_GPUS" \
    $(sglang_flags) \
    --draft-checkpoint "$EVAL_CKPT" --draft-attention-backend sdpa \
    --embedding-key model.llm.embed.weight --lm-head-key model.llm.unembed.weight \
    --chat-template "$CHAT_TEMPLATE" \
    --stop-token-ids 200006 200000 200002 200003 \
    --eval-datasets-dir "$EVAL_DATASETS_DIR" \
    --tasks $EVAL_TASKS \
    --limit-per-task "$EVAL_LIMIT_SWEEP" --max-new-tokens "$EVAL_MAX_NEW" \
    --temperature "$EVAL_TEMP" --seed 980406 \
    --output-json "$OUTPUT_DIR/evals/eval9_${EVAL_TAG}.json" \
    --dist-timeout 30
}

case "${1:-}" in
  setup)     cmd_setup ;;
  prepare)   cmd_prepare ;;
  probe)     cmd_probe ;;
  train)     cmd_train ;;
  eval)      cmd_eval ;;
  *) sed -n '2,40p' "$SCRIPT_PATH"; echo; echo "ERROR: unknown command '${1:-}'"; exit 1 ;;
esac
