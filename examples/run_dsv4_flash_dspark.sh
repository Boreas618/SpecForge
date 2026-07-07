#!/bin/bash
# Faithful DeepSeek-V4-Flash-DSpark drafter training (1:1 with the released model),
# tuned for a single 4x NVIDIA GB300 (284GB) node.
#
# Trains the exact mtp.* draft architecture of deepseek-ai/DeepSeek-V4-Flash-DSpark:
# 3 DeepSeek-V4 decoder blocks (shared-KV MLA + hyper-connections + 256-expert
# top-k noaux_tc MoE with 1 shared expert), learned hc_head, Markov(rank 256) +
# confidence heads; block_size 5, noise/mask token 128799, target aux layers
# [40,41,42]. All architecture comes from configs/deepseek-v4-flash-dspark.json.
#
# PRECISION (matched to the release): the drafter is trained in bf16 (DeepSpec
# recipe precision=bf16) and checkpointed in the released mtp.* layout as bf16
# "pre-quantization" weights. The released -DSpark file is FP4 experts + FP8
# linears; produce that from our bf16 checkpoint downstream via the released
# inference/convert.py --expert-dtype fp4.
#
# TEACHER = sgl-project/DeepSeek-V4-Flash-FP8, the SGLang-blessed FP8 re-pack of
# deepseek-ai/DeepSeek-V4-Flash. Its MoE experts are a *lossless* FP4->FP8 upcast
# of the deployed FP4 experts (identical weight values; the released convert.py's
# cast_e2m1fn_to_e4m3fn), and its linears are the same FP8 -> numerically the same
# teacher at >= the deployed precision. We use it (not the native FP4+FP8 repo)
# because sglang 0.5.14's training-path fused-MoE cannot consume DeepSeek's
# fp4-packed (I8) experts (it resolves quant_method=Fp8MoEMethod and asserts a
# hidden-size mismatch on the half-width packed weights) -- which is precisely why
# this FP8 re-pack exists. Served via sglang tp=4 with the auto dsv4 attention
# backend + fp8 KV cache.
set -x
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# ---- Environment ----
export HF_HOME=${HF_HOME:-/scratch/hf_cache}
export HF_TOKEN=${HF_TOKEN:-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}
export SPECFORGE_DATA_NUM_PROC=${SPECFORGE_DATA_NUM_PROC:-64}
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
# Leave sglang MoE runner backend on AUTO. sglang's deepseek_v4 hook resolves it
# to flashinfer_trtllm_routed for the FP4(nvfp4)+FP8 checkpoint; forcing 'triton'
# would block that resolution. (Do NOT set SPECFORGE_SGLANG_MOE_RUNNER_BACKEND.)
unset SPECFORGE_SGLANG_MOE_RUNNER_BACKEND
export SGLANG_EAGER_INPUT_NO_COPY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The FP8 re-pack stores attn wo_a in bf16 (the released convert.py dequantizes it),
# but on Blackwell (sm100+) sglang's default SGLANG_OPT_FP8_WO_A_GEMM=1 makes the
# wo_a param fp8 -> "Downcasting not allowed bf16->fp8" at load. Keep wo_a bf16
# (faithful; matches the checkpoint) by disabling the fp8-wo_a optimization.
export SGLANG_OPT_FP8_WO_A_GEMM=0
# Bound the host-RAM spike when the tp=4 FP8 target loads: serialize each rank's
# safetensors load (1 shard at a time) + drop the page cache after. Without this the
# 274GB target buffers many 6GB shards in parallel -> ~940GB transient host peak that
# trips earlyoom. (Code-default 1; pinned here for clarity. See EXECUTION_DOC 4h.)
export SPECFORGE_SGLANG_SERIAL_LOAD=${SPECFORGE_SGLANG_SERIAL_LOAD:-1}
# Offload the fp32 master + AdamW moments to CPU. Under FULL_SHARD the optimizer
# state spans the ~5B-param/rank *shard* -> ~60GB/rank; on CPU that is ~240GB host
# RAM total (safe on the 955GB box) and frees ~60GB/rank of GPU for the larger
# BATCH_SIZE=8 micro-batch. The CPU AdamW step (~25-30s) is paid once per ACC(=64)
# micro-steps -> ~0.4s/step amortized (negligible); GPU AdamW would be ~0.1s but its
# 60GB/rank blocks BS8. NOTE: NO_SHARD/DDP (DeepSpec's own choice) would hold the full
# 19.85B params/rank -> ~238GB/rank optimizer x4 ranks = ~952GB > box RAM -> NOT viable
# here; FULL_SHARD is mandatory (see trainer comment + EXECUTION_DOC 4i/5).
export SPECFORGE_OFFLOAD_MASTER=${SPECFORGE_OFFLOAD_MASTER:-1}

# ---- Paths ----
# Teacher = SGLang-blessed FP8 re-pack (lossless upcast of the deployed FP4+FP8
# target; see header). Both embed.weight and head.weight are bf16 (unquantized).
TARGET_MODEL=${TARGET_MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}
TRAIN_DATA=${TRAIN_DATA:-$ROOT_DIR/cache/dataset/perfectblend_trainsplit.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/dsv4-flash-dspark}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/deepseek-v4-flash-dspark.json}

# ---- Parallelism (4x GB300) ----
NPROC=${NPROC:-4}          # was 8; this node has 4 GPUs
TP_SIZE=${TP_SIZE:-4}      # sglang target tp == world; draft FSDP full-shard over all ranks (dp=1)

# ---- Schedule / batch (DeepSpec DSpark recipe: 10 epochs, global batch 512) ----
NUM_EPOCHS=${NUM_EPOCHS:-10}
# Per-rank micro-batch. dp=1 so global = BS*ACC; a larger BS does NOT change the
# effective batch (ACC compensates) but amortizes the fixed ~2s/step FSDP all-gather.
# BS4 is COMPUTE-BOUND here: measured ~1.16s/step at 95-100% GPU util on all 4 ranks,
# so it is the throughput optimum for this dp=1/tp=4 topology -- a larger batch cannot
# help (global batch is fixed at 512, and the FSDP all-gather a bigger BS would amortize
# is already hidden by prefetch at 100% util). BS8 does NOT OOM but hits the objective's
# vocab-softmax MEMORY WALL: at n_blocks up to num_anchors(512) the loss materializes
# [B, nb, bs, 129280] fp32 tensors (~10GB each), whose ~30-50GB transient peak triggers
# synchronous allocator stalls (0-1% GPU util, 15-28s/step; py-spy -> _dspark_objective).
# Chunking the objective would remove the wall but buys ~nothing (GPUs already saturated),
# so it is not implemented. The only real ~2x lever is tp2/dp2 (topology change; see doc).
BATCH_SIZE=${BATCH_SIZE:-4}
GLOBAL_BATCH=${GLOBAL_BATCH:-512}
# ACC auto-derived to hold global batch 512 (dp=1). floor(512/BS), min 1.
ACC_STEPS=${ACC_STEPS:-$(( GLOBAL_BATCH / BATCH_SIZE ))}
[ "$ACC_STEPS" -lt 1 ] && ACC_STEPS=1
MAX_STEPS=${MAX_STEPS:-}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-10}
MAX_LEN=${MAX_LEN:-4096}
NUM_ANCHORS=${NUM_ANCHORS:-512}
# sglang target static fraction. FP8 target weights are ~68.5GB/rank; 0.4*284=113GB
# holds them + the prefill KV pool and leaves ~166GB/rank for the GC-off FSDP draft at
# BS4 (proven in profile_real4). For BS8+objective-chunking use MEM_FRAC=0.32 (frees
# ~19GB more for the larger activations). Measured at load: 0.32 -> 92GB/rank target.
MEM_FRAC=${MEM_FRAC:-0.4}

# ---- Tracking ----
REPORT_TO=${REPORT_TO:-wandb}
WANDB_PROJECT=${WANDB_PROJECT:-specforge-dspark}
WANDB_NAME=${WANDB_NAME:-dsv4-flash-dspark}

EXTRA=""; [ -n "$MAX_STEPS" ] && EXTRA="--max-steps $MAX_STEPS"
# RESUME=1 -> --resume: auto-loads the latest epoch_*_step_* checkpoint in OUTPUT_DIR
# (weights + optimizer/scheduler/step from training_state.pt) and fast-forwards the
# dataloader. Safe to leave on for a long run so a restart continues instead of
# retraining from scratch. No-op if OUTPUT_DIR has no checkpoint.
[ "${RESUME:-0}" = "1" ] && EXTRA="$EXTRA --resume"

torchrun --standalone --nproc_per_node $NPROC \
    $ROOT_DIR/scripts/train_dspark_v4.py \
    --target-model-path $TARGET_MODEL \
    --target-model-backend sglang --tp-size $TP_SIZE \
    --embedding-key embed.weight --lm-head-key head.weight \
    --draft-config-path $DRAFT_CONFIG \
    --train-data-path $TRAIN_DATA --output-dir $OUTPUT_DIR \
    --num-epochs $NUM_EPOCHS --batch-size $BATCH_SIZE --accumulation-steps $ACC_STEPS \
    --learning-rate 6e-4 --warmup-ratio 0.04 --max-grad-norm 1.0 --seed 42 \
    --max-length $MAX_LEN --chat-template deepseek-v3 \
    --num-anchors $NUM_ANCHORS --loss-decay-gamma 4.0 \
    --ce-loss-alpha 0.1 --l1-loss-alpha 0.9 --confidence-head-alpha 1.0 \
    --log-interval $LOG_INTERVAL --save-interval $SAVE_INTERVAL \
    --report-to $REPORT_TO --wandb-project $WANDB_PROJECT --wandb-name $WANDB_NAME \
    --dataloader-num-workers 4 --build-dataset-num-proc $SPECFORGE_DATA_NUM_PROC \
    --cache-dir $ROOT_DIR/cache \
    --sglang-attention-backend dsv4 \
    --sglang-mem-fraction-static $MEM_FRAC --sglang-context-length 8192 \
    $EXTRA
