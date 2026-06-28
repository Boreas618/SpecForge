#!/bin/bash

# Generate agentic SWE-bench trajectories for DFlash training.
#
# This script handles Phase A of the staged agentic-DFlash pipeline:
# launch the SGLang target server, run Harbor agent rollouts on SWE-bench
# tasks in Docker sandboxes, and collect pretokenized train/eval JSONL
# that the companion training script consumes.
#
# The target model is frozen during DFlash training (only the draft updates),
# so the agentic trajectory distribution is stationary — generate once here,
# then train separately with examples/run_kimi_k2.7_code_dflash_swebench.sh.
#
# STEPS (run in order on the GPU node that serves the target):
#   1. `tasks`   : materialize SWE-bench Verified Harbor tasks (needs uv+docker).
#   2. `serve`   : launch the SGLang OpenAI server for the target (tmux #1).
#   3. `rollout` : run the agent rollouts -> pretokenized JSONL (tmux #2,
#                  in HARBOR's uv env). Waits for `serve` to be healthy.
#
# KEY ENV (all overridable):
#   TARGET_MODEL=moonshotai/Kimi-K2.7-Code
#   SERVE_PORT=30000  SERVE_TP=8  SERVE_MEM_FRACTION=0.9
#   SWEBENCH_LIMIT=200  SWEBENCH_TASKS_DIR=<harbor>/datasets/swebench-verified
#   ROLLOUT_AGENT=codex  (codex|terminus-2)
#   ROLLOUT_N_CONCURRENT=8  ROLLOUT_TEMPERATURE=0.7  ROLLOUT_MAX_TURNS=<n>
#   ROLLOUT_REASONING_EFFORT=high  (codex only)
#   DATA_DIR=<SpecForge>/cache/dataset/swebench-agentic
#   MAX_LENGTH=16384  (window size for long trajectories)
#
# AGENT NOTES:
#   - codex (default): the Codex CLI drives the target via OPENAI_BASE_URL and
#     emits an ATIF trajectory that we RE-TOKENIZE with the target tokenizer
#     ($TARGET_MODEL) to get (input_ids, loss_mask). The SGLang server does NOT
#     need to return token ids for this path. Caveat: the Codex CLI defaults to
#     OpenAI's Responses API -- if your server only serves /v1/chat/completions,
#     configure a Codex model_provider with wire_api="chat".
#   - terminus-2: streams exact prompt/completion token ids from the server
#     (requires the SGLang OpenAI endpoint to return token ids).
#
# OUTPUT:
#   $DATA_DIR/swebench_agentic_train.jsonl   (pretokenized, for train_dflash.py)
#   $DATA_DIR/swebench_agentic_eval.jsonl

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/$(basename "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
ROOT_DIR=$(dirname "$SCRIPT_DIR")                          # SpecForge repo root
WORKSPACE_DIR=$(dirname "$ROOT_DIR")                       # Spec/ (holds SpecForge/, harbor/, sglang/)
HARBOR_DIR=${HARBOR_DIR:-$WORKSPACE_DIR/harbor}

# ---- target model ----------------------------------------------------------
TARGET_MODEL=${TARGET_MODEL:-moonshotai/Kimi-K2.7-Code}
export HF_HOME=${HF_HOME:-/cluster-storage/models}

# ---- serving knobs ---------------------------------------------------------
SERVED_NAME=${SERVED_NAME:-kimi-k2.7-code}
SERVE_ADDR=${SERVE_ADDR:-127.0.0.1}
SERVE_PORT=${SERVE_PORT:-30000}
SERVE_TP=${SERVE_TP:-8}
SERVE_MEM_FRACTION=${SERVE_MEM_FRACTION:-0.9}
SERVE_CONTEXT_LENGTH=${SERVE_CONTEXT_LENGTH:-262144}
SERVE_EXTRA_ARGS=${SERVE_EXTRA_ARGS:-}

# ---- task / rollout knobs --------------------------------------------------
SWEBENCH_TASKS_DIR=${SWEBENCH_TASKS_DIR:-$HARBOR_DIR/datasets/swebench-verified}
SWEBENCH_LIMIT=${SWEBENCH_LIMIT:-}
ROLLOUT_AGENT=${ROLLOUT_AGENT:-codex}
ROLLOUT_N_CONCURRENT=${ROLLOUT_N_CONCURRENT:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_MAX_TURNS=${ROLLOUT_MAX_TURNS:-}
ROLLOUT_PARSER=${ROLLOUT_PARSER:-xml}
ROLLOUT_EVAL_RATIO=${ROLLOUT_EVAL_RATIO:-0.02}
ROLLOUT_MODEL_NAME=${ROLLOUT_MODEL_NAME:-openai/$SERVED_NAME}
# codex-only knobs
ROLLOUT_REASONING_EFFORT=${ROLLOUT_REASONING_EFFORT:-high}
CODEX_OPENAI_API_KEY=${CODEX_OPENAI_API_KEY:-sk-local-sglang}

# ---- output ----------------------------------------------------------------
DATA_DIR=${DATA_DIR:-$ROOT_DIR/cache/dataset/swebench-agentic}
MAX_LENGTH=${MAX_LENGTH:-16384}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

# --------------------------------------------------------------------------- #

cmd_tasks() {
  command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found (needed for the harbor swebench adapter)"; exit 1; }
  local adapter="$HARBOR_DIR/adapters/swebench"
  [ -d "$adapter" ] || { echo "ERROR: harbor swebench adapter not found at $adapter"; exit 1; }
  local limit_args=()
  [ -n "$SWEBENCH_LIMIT" ] && limit_args=(--limit "$SWEBENCH_LIMIT")
  log "tasks: generating SWE-bench tasks -> $SWEBENCH_TASKS_DIR ${limit_args[*]:-(all)}"
  ( cd "$adapter" && uv run swebench --task-dir "$SWEBENCH_TASKS_DIR" "${limit_args[@]}" )
  log "tasks: done."
}

cmd_serve() {
  if ! python3 -c "import sglang" 2>/dev/null; then
    echo "ERROR: sglang not importable. Install sglang first." >&2; exit 1
  fi
  log "serve: launching SGLang OpenAI server for $TARGET_MODEL on 0.0.0.0:$SERVE_PORT \
(served-model-name=$SERVED_NAME tp=$SERVE_TP)"
  # shellcheck disable=SC2086
  exec python3 -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --trust-remote-code \
    --served-model-name "$SERVED_NAME" \
    --tool-call-parser kimi_k2 \
    --tp-size "$SERVE_TP" \
    --ep-size "$SERVE_TP" \
    --dp-size "$SERVE_TP" \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --attention-backend flashinfer \
    --mem-fraction-static "$SERVE_MEM_FRACTION" \
    --context-length "$SERVE_CONTEXT_LENGTH" \
    --host 0.0.0.0 --port "$SERVE_PORT" \
    $SERVE_EXTRA_ARGS
}

cmd_rollout() {
  mkdir -p "$DATA_DIR"
  local api_base="http://$SERVE_ADDR:$SERVE_PORT/v1"

  log "rollout: waiting for SGLang server at $api_base ..."
  local tries=0
  until curl -sf "http://$SERVE_ADDR:$SERVE_PORT/health" >/dev/null 2>&1 \
        || curl -sf "http://$SERVE_ADDR:$SERVE_PORT/get_model_info" >/dev/null 2>&1; do
    tries=$((tries+1))
    [ "$tries" -ge "${ROLLOUT_WAIT_TRIES:-180}" ] && { echo "ERROR: server not healthy after waiting"; exit 1; }
    sleep 10
  done
  log "rollout: server healthy. Starting rollouts."

  local driver="$ROOT_DIR/scripts/run_swebench_rollouts.py"
  local rollout_py=(${ROLLOUT_PYTHON:-python3})
  local extra=()
  [ -n "$SWEBENCH_LIMIT" ] && extra+=(--limit "$SWEBENCH_LIMIT")
  [ -n "$ROLLOUT_MAX_TURNS" ] && extra+=(--max-turns "$ROLLOUT_MAX_TURNS")

  # Agent-specific args: codex re-tokenizes with the target tokenizer, while
  # terminus-2 needs the action parser + sampling temperature.
  local agent_args=(--agent "$ROLLOUT_AGENT")
  if [ "$ROLLOUT_AGENT" = "codex" ]; then
    agent_args+=(--target-model "$TARGET_MODEL"
                 --reasoning-effort "$ROLLOUT_REASONING_EFFORT"
                 --openai-api-key "$CODEX_OPENAI_API_KEY")
  else
    agent_args+=(--parser "$ROLLOUT_PARSER"
                 --temperature "$ROLLOUT_TEMPERATURE")
  fi

  "${rollout_py[@]}" "$driver" \
    --tasks-dir "$SWEBENCH_TASKS_DIR" \
    --api-base "$api_base" \
    --model-name "$ROLLOUT_MODEL_NAME" \
    "${agent_args[@]}" \
    --n-concurrent "$ROLLOUT_N_CONCURRENT" \
    --max-length "$MAX_LENGTH" \
    --min-loss-tokens "$((2 * 8))" \
    --eval-ratio "$ROLLOUT_EVAL_RATIO" \
    --output-dir "$DATA_DIR" \
    "${extra[@]}"

  log "rollout: done."
  log "  train: $DATA_DIR/swebench_agentic_train.jsonl"
  log "  eval:  $DATA_DIR/swebench_agentic_eval.jsonl"
  log ""
  log "Next: train with examples/run_kimi_k2.7_code_dflash_swebench.sh"
}

# --------------------------------------------------------------------------- #

case "${1:-}" in
  tasks)    cmd_tasks ;;
  serve)    cmd_serve ;;
  rollout)  cmd_rollout ;;
  *)
    echo "Usage: bash $0 {tasks|serve|rollout}"
    echo ""
    echo "  tasks    Materialize SWE-bench Harbor tasks (needs uv+docker)"
    echo "  serve    Launch SGLang target server (tmux session #1)"
    echo "  rollout  Run agent rollouts -> pretokenized JSONL (tmux session #2)"
    echo ""
    echo "Run 'tasks' once, then 'serve' and 'rollout' in separate terminals."
    if [ -n "${1:-}" ]; then echo "ERROR: unknown command '${1}'"; exit 1; fi
    ;;
esac
