#!/bin/bash
# Offline teacher-forced probe of a DSpark-V4 drafter checkpoint against the FP8
# DeepSeek-V4-Flash target on the HELD-OUT eval split (perfectblend_eval.jsonl).
# This is the project's canonical "goal probe" (EXECUTION_DOC §Eval split): reports
# train/accuracy, per-block-slot acceptance, ce/l1 loss, draft-vs-teacher agreement,
# and the teacher-vs-data top1/top5 ceiling. Mirrors the training forward exactly,
# so numbers are directly comparable to the wandb train/* curves.
#
# Usage:  bash scripts/run_dspark_v4_probe.sh <checkpoint_dir> [num_batches] [batch_size]
set -euo pipefail

ROOT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )
CKPT=${1:?usage: run_dspark_v4_probe.sh <checkpoint_dir> [num_batches] [batch_size]}
NUM_BATCHES=${2:-64}
BATCH_SIZE=${3:-4}

TP_SIZE=${TP_SIZE:-4}
TARGET_MODEL=${TARGET_MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}
EVAL_DATA=${EVAL_DATA:-$ROOT_DIR/cache/dataset/perfectblend_eval.jsonl}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/deepseek-v4-flash-dspark.json}
MEM_FRAC=${MEM_FRAC:-0.4}
NUM_ANCHORS=${NUM_ANCHORS:-512}

export HF_HOME=${HF_HOME:-/scratch/hf_cache}
export HF_TOKEN=${HF_TOKEN:-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}
export PYTHONPATH=$ROOT_DIR:${PYTHONPATH:-}
# Host-RAM guards (same as training): serialize the big FP8 target weight-load, keep
# wo_a bf16 on Blackwell, no-copy eager inputs.
export SPECFORGE_SGLANG_SERIAL_LOAD=${SPECFORGE_SGLANG_SERIAL_LOAD:-1}
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_EAGER_INPUT_NO_COPY=1

echo "[probe] ckpt=$CKPT tp=$TP_SIZE batches=$NUM_BATCHES bs=$BATCH_SIZE mem_frac=$MEM_FRAC"
torchrun --standalone --nproc_per_node "$TP_SIZE" \
    "$ROOT_DIR/scripts/eval_dspark_v4_probe.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend sglang --tp-size "$TP_SIZE" \
    --draft-checkpoint "$CKPT" \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$EVAL_DATA" \
    --chat-template deepseek-v3 \
    --max-length 4096 \
    --batch-size "$BATCH_SIZE" --num-batches "$NUM_BATCHES" \
    --num-anchors "$NUM_ANCHORS" \
    --embedding-key embed.weight --lm-head-key head.weight \
    --cache-dir "$ROOT_DIR/cache" \
    --sglang-attention-backend dsv4 \
    --sglang-mem-fraction-static "$MEM_FRAC" --sglang-context-length 8192
