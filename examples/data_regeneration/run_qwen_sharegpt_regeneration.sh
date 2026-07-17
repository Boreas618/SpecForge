#!/usr/bin/env bash
# Regenerate ShareGPT conversations through `specforge data regen`.
#
# Model and sampling choices live in the recipe files under
# examples/data_regeneration/recipes/; this script only selects a recipe,
# points it at the input dataset, and supplies runtime endpoints. The output
# is a finalized, content-addressed DatasetArtifact; validation (including
# the reasoning contract) runs as part of the artifact lifecycle.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PROFILE="${MODEL_PROFILE:-qwen3-8b}"
case "${MODEL_PROFILE}" in
    qwen3-8b)
        DEFAULT_RECIPE="examples/data_regeneration/recipes/qwen3-8b-sharegpt-non-reasoning.yaml"
        DEFAULT_ARTIFACT_DIR="${ROOT_DIR}/cache/dataset/sharegpt-regen-qwen3-8b-non-reasoning"
        ;;
    qwen3.6-27b)
        DEFAULT_RECIPE="examples/data_regeneration/recipes/qwen3.6-27b-sharegpt-reasoning.yaml"
        DEFAULT_ARTIFACT_DIR="${ROOT_DIR}/cache/dataset/sharegpt-regen-qwen3.6-27b-reasoning"
        ;;
    *)
        echo "Unsupported MODEL_PROFILE: ${MODEL_PROFILE}" >&2
        echo "Expected qwen3-8b or qwen3.6-27b." >&2
        exit 1
        ;;
esac

PYTHON="${PYTHON:-python}"
RECIPE="${RECIPE:-${DEFAULT_RECIPE}}"
INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/cache/dataset/sharegpt_train.jsonl}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${DEFAULT_ARTIFACT_DIR}}"
SERVER_ADDRESSES="${SERVER_ADDRESSES:-localhost:30000}"

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Input dataset does not exist: ${INPUT_FILE}" >&2
    exit 1
fi
if [[ -e "${ARTIFACT_DIR}/manifest.json" ]]; then
    echo "Refusing to reuse a finalized artifact: ${ARTIFACT_DIR}" >&2
    echo "Choose a fresh ARTIFACT_DIR or remove the old artifact explicitly." >&2
    exit 1
fi

read -r -a server_list <<< "${SERVER_ADDRESSES}"
endpoint_args=()
for address in "${server_list[@]}"; do
    if [[ "${address}" == http*://* ]]; then
        endpoint_args+=("--endpoint" "teacher=${address}")
    else
        endpoint_args+=("--endpoint" "teacher=http://${address}")
    fi
done

"${PYTHON}" -m specforge data regen run --config "${RECIPE}" \
    "sources.sharegpt.config.path=${INPUT_FILE}" \
    "output.uri=${ARTIFACT_DIR}" \
    "${endpoint_args[@]}"

"${PYTHON}" -m specforge data regen validate --artifact "${ARTIFACT_DIR}"
"${PYTHON}" -m specforge data regen finalize --artifact "${ARTIFACT_DIR}"

INSPECT_JSON="$(mktemp)"
trap 'rm -f "${INSPECT_JSON}"' EXIT
"${PYTHON}" -m specforge data regen inspect --artifact "${ARTIFACT_DIR}" --json \
    > "${INSPECT_JSON}"

INPUT_ROWS=$(awk 'END {print NR}' "${INPUT_FILE}")

"${PYTHON}" - "${INPUT_ROWS}" "${INSPECT_JSON}" <<'PY'
import json
import sys

input_rows = int(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as handle:
    manifest = json.load(handle)
status = manifest["counts"]["status_counts"]
success_rows = status.get("success", 0)
error_rows = status.get("unresolved_error", 0)
skipped_rows = status.get("policy_reject", 0) + status.get("terminal_reject", 0)
completed_rows = success_rows + error_rows + skipped_rows

print(f"input rows: {input_rows}")
print(f"success rows: {success_rows}")
print(f"error rows: {error_rows}")
print(f"skipped rows: {skipped_rows}")
if completed_rows != input_rows:
    raise SystemExit(
        "regeneration did not account for every input row: "
        f"completed={completed_rows}, input={input_rows}"
    )
success_fraction = success_rows / completed_rows if completed_rows else 0.0
print(f"success fraction: {success_fraction:.2%}")
PY

echo "Finalized artifact: ${ARTIFACT_DIR}"
echo "Train against it with data.dataset_artifact: ${ARTIFACT_DIR}/manifest.json"
