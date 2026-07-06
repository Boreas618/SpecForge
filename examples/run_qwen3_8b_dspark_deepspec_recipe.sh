#!/bin/bash
# DSpark training for Qwen3-8B following the DeepSpec recipe (dspark_qwen3_8b block7).
#
# Recipe alignment vs config/dspark/dspark_qwen3_8b.py in deepseek-ai/DeepSpec:
#   block_size=7, num_draft_layers=5, target_layer_ids=[1,9,17,25,33],
#   mask_token_id=151669, num_anchors=512, markov_rank=256 (vanilla),
#   confidence head (with markov) alpha=1.0, loss_decay_gamma=4.0,
#   ce=0.1 / l1=0.9, lr=6e-4, warmup_ratio=0.04, max_grad_norm=1.0,
#   global batch 512 = 8 GPUs x local bs 4 x accumulation 16, max_length 4096.
# Documented divergences: raw open-perfectblend (no target regeneration),
# online HF target (numerically equivalent to DeepSpec's offline cache),
# epochs limited by pod walltime instead of 10.

set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

TRAIN_DATA=${TRAIN_DATA:-$ROOT_DIR/cache/dataset/perfectblend_raw_900k.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/qwen3-8b-dspark-deepspec-recipe}
MAX_STEPS=${MAX_STEPS:-}          # micro-steps; empty = run to num-epochs end
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
ACC_STEPS=${ACC_STEPS:-16}        # 8 x 4 x 16 = 512 global batch
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
LOG_INTERVAL=${LOG_INTERVAL:-16}  # one log per optimizer step

EXTRA_ARGS=""
if [ -n "$MAX_STEPS" ]; then
  EXTRA_ARGS="--max-steps $MAX_STEPS"
fi

torchrun \
    --standalone \
    --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_dspark.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dspark.json \
    --train-data-path $TRAIN_DATA \
    --output-dir $OUTPUT_DIR \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACC_STEPS \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template qwen \
    --attention-backend flex_attention \
    --target-model-backend hf \
    --block-size 7 \
    --num-anchors 512 \
    --loss-decay-gamma 4.0 \
    --mask-token-id 151669 \
    --markov-rank 256 \
    --markov-head-type vanilla \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --ce-loss-alpha 0.1 \
    --l1-loss-alpha 0.9 \
    --confidence-head-alpha 1.0 \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --report-to ${REPORT_TO:-tensorboard} \
    --wandb-project ${WANDB_PROJECT:-specforge-dspark} \
    --wandb-name ${WANDB_NAME:-qwen3-8b-dspark-deepspec-recipe} \
    --build-dataset-num-proc $SPECFORGE_DATA_NUM_PROC \
    --dataloader-num-workers 4 \
    --cache-dir $ROOT_DIR/cache \
    $EXTRA_ARGS
