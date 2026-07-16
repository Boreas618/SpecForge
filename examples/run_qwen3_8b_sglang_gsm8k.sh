#!/usr/bin/env bash
# Controlled SGLang evaluation for a Qwen3-8B DFlash or DConv checkpoint.
# Usage: run_qwen3_8b_sglang_gsm8k.sh DFLASH|DCONV /path/to/draft/checkpoint

set -euo pipefail

ALGORITHM="${1:?usage: $0 DFLASH|DCONV /path/to/draft/checkpoint}"
DRAFT_MODEL="${2:?usage: $0 DFLASH|DCONV /path/to/draft/checkpoint}"
ALGORITHM="${ALGORITHM^^}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DFLASH_REPO="${DFLASH_REPO:-/personal/yi/DConv/dflash}"
SGLANG_REPO="${SGLANG_REPO:-/personal/yi/DConv/sglang}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-30000}"
OUTPUT_DIR="${OUTPUT_DIR:-/personal/yi/DConv/eval_results}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"

if [[ ! -f "${DFLASH_REPO}/dflash/benchmark.py" ]]; then
  echo "Missing ${DFLASH_REPO}/dflash/benchmark.py; clone https://github.com/z-lab/dflash there." >&2
  exit 2
fi
if [[ ! -d "${DRAFT_MODEL}" ]]; then
  echo "Draft checkpoint does not exist: ${DRAFT_MODEL}" >&2
  exit 2
fi

case "${ALGORITHM}" in
  DFLASH)
    SPECIFIC_ARGS=(--speculative-dflash-block-size 16)
    ;;
  DCONV)
    SPECIFIC_ARGS=(--speculative-dconv-block-size 15)
    ;;
  *)
    echo "Algorithm must be DFLASH or DCONV, got ${ALGORITHM}." >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFIX="${OUTPUT_DIR}/${ALGORITHM,,}_qwen3_8b_gsm8k_32_${STAMP}"

export PYTHONPATH="${SGLANG_REPO}/python${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

setsid python -m sglang.launch_server \
  --model-path "${TARGET_MODEL}" \
  --speculative-algorithm "${ALGORITHM}" \
  --speculative-draft-model-path "${DRAFT_MODEL}" \
  --speculative-num-draft-tokens 16 \
  "${SPECIFIC_ARGS[@]}" \
  --tp-size 1 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --speculative-draft-attention-backend "${ATTENTION_BACKEND}" \
  --disable-overlap-schedule \
  --disable-cuda-graph \
  --max-running-requests 1 \
  --random-seed 0 \
  --port "${PORT}" \
  >"${PREFIX}.server.log" 2>&1 &
SERVER_PID=$!
trap 'kill -- -"${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health_generate" >/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "SGLang server exited during startup; see ${PREFIX}.server.log" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${PORT}/health_generate" >/dev/null

(
  cd "${DFLASH_REPO}"
  python -m dflash.benchmark \
    --backend sglang \
    --model "${TARGET_MODEL}" \
    --dataset gsm8k \
    --num-prompts 32 \
    --concurrency 1 \
    --max-new-tokens 2048 \
    --temperature 0 \
    --top-k 1 \
    --top-p 1 \
    --base-url "http://127.0.0.1:${PORT}"
) 2>&1 | tee "${PREFIX}.benchmark.log"
