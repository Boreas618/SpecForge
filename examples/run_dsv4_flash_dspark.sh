#!/bin/bash
# Faithful DeepSeek-V4-Flash-DSpark drafter training (1:1 with the released model).
#
# Trains the exact mtp.* draft architecture of deepseek-ai/DeepSeek-V4-Flash-DSpark:
# 3 DeepSeek-V4 decoder blocks (shared-KV MLA + hyper-connections + 256-expert
# top-k noaux_tc MoE with 1 shared expert), learned hc_head, Markov(rank 256) +
# confidence heads; block_size 5, noise/mask token 128799, target aux layers
# [40,41,42]. All architecture comes from configs/deepseek-v4-flash-dspark.json.
# Target = DeepSeek-V4-Flash-FP8 served via sglang tp8. Checkpoints are written in
# the released mtp.* layout (model.dspark_mtp.safetensors) for direct vLLM load.
set -x
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
export SPECFORGE_SGLANG_MOE_RUNNER_BACKEND=${SPECFORGE_SGLANG_MOE_RUNNER_BACKEND:-triton}
export SGLANG_EAGER_INPUT_NO_COPY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The faithful draft is a ~19.7B 256-expert MoE; keep the fp32 master + AdamW
# moments on CPU so the sharded draft fits alongside the resident tp8 target.
export SPECFORGE_OFFLOAD_MASTER=${SPECFORGE_OFFLOAD_MASTER:-1}

TRAIN_DATA=${TRAIN_DATA:-$ROOT_DIR/cache/dataset/perfectblend.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/dsv4-flash-dspark}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/deepseek-v4-flash-dspark.json}
MAX_STEPS=${MAX_STEPS:-}
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
ACC_STEPS=${ACC_STEPS:-32}
SAVE_INTERVAL=${SAVE_INTERVAL:-320}
LOG_INTERVAL=${LOG_INTERVAL:-8}
MAX_LEN=${MAX_LEN:-4096}
NUM_ANCHORS=${NUM_ANCHORS:-512}
MEM_FRAC=${MEM_FRAC:-0.8}

EXTRA=""; [ -n "$MAX_STEPS" ] && EXTRA="--max-steps $MAX_STEPS"

torchrun --standalone --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_dspark_v4.py \
    --target-model-path sgl-project/DeepSeek-V4-Flash-FP8 \
    --target-model-backend sglang --tp-size 8 \
    --embedding-key embed.weight --lm-head-key head.weight \
    --draft-config-path $DRAFT_CONFIG \
    --train-data-path $TRAIN_DATA --output-dir $OUTPUT_DIR \
    --num-epochs $NUM_EPOCHS --batch-size $BATCH_SIZE --accumulation-steps $ACC_STEPS \
    --learning-rate 6e-4 --warmup-ratio 0.04 --max-grad-norm 1.0 \
    --max-length $MAX_LEN --chat-template deepseek-v3 \
    --num-anchors $NUM_ANCHORS --loss-decay-gamma 4.0 \
    --ce-loss-alpha 0.1 --l1-loss-alpha 0.9 --confidence-head-alpha 1.0 \
    --log-interval $LOG_INTERVAL --save-interval $SAVE_INTERVAL \
    --report-to ${REPORT_TO:-none} --wandb-project ${WANDB_PROJECT:-specforge-dspark} \
    --wandb-name ${WANDB_NAME:-dsv4-flash-dspark} \
    --dataloader-num-workers 0 --build-dataset-num-proc $SPECFORGE_DATA_NUM_PROC \
    --cache-dir $ROOT_DIR/cache \
    --sglang-mem-fraction-static $MEM_FRAC --sglang-context-length 8192 \
    $EXTRA
