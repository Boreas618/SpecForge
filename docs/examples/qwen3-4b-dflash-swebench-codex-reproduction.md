# Qwen3-4B DFlash SWE-bench Codex Rollout Reproduction

This document records the exact Phase A reproduction that generated the current
Qwen3-4B agentic SWE-bench DFlash JSONL files, plus the validation used to check
the tokenizer, chat-template rendering, and `loss_mask` construction.

It complements
[`qwen3-4b-dflash-swebench-codex-rollout.md`](./qwen3-4b-dflash-swebench-codex-rollout.md),
which is the reusable runbook. This file is the concrete reproduction log and
audit trail for the run completed on 2026-06-28.

## Environment

- Workspace root: `/persistent`
- SpecForge checkout: `/persistent/SpecForge`
- Harbor checkout: `/persistent/harbor`
- Environment file: `/persistent/.env`
- Daytona API key source: `DAYTONA_API_KEY` from `/persistent/.env`
- Target model: `Qwen/Qwen3-4B`
- Served model name: `qwen3-4b`
- SGLang local endpoint during rollout: `http://127.0.0.1:30000/v1`
- Tunnel endpoint during rollout: `https://parade-stopping-roll-seeker.trycloudflare.com/v1`
- Harbor jobs dir for the 20-task batch:
  `/persistent/SpecForge/cache/harbor_jobs/swebench-20260628-201553`
- Output dataset dir:
  `/persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b`

The quick Cloudflare tunnel and SGLang server were stopped after validation.
The output JSONL and Harbor job artifacts remain on disk.

## Code State Required

The reproduction depends on these local changes:

1. `SpecForge/scripts/run_swebench_rollouts.py`
   - Adds `--environment` and `--env-kwargs`.
   - Uses `EnvironmentType(args.environment)` instead of hardcoded Docker.
   - Passes `CODEX_WIRE_API` into the Codex agent environment.
   - Uses `await Job.create(config)` for the current Harbor API.
   - Re-renders Codex ATIF trajectories with the target tokenizer/chat template
     and constructs `loss_mask` from assistant-generated spans.
   - Handles current Transformers/Qwen tokenizer return shapes by rendering the
     chat template to text and then encoding with `add_special_tokens=False`.

2. `/persistent/harbor/src/harbor/agents/installed/codex.py`
   - Writes `openai_base_url = "${OPENAI_BASE_URL}"` for the current
     Responses API path.
   - Keeps an env-gated `CODEX_WIRE_API=chat` provider block only for older
     Codex CLI versions that still accept `wire_api = "chat"`.

3. `SpecForge/examples/run_qwen3_4b_dflash_swebench_rollout.sh`
   - Adds `ROLLOUT_ENVIRONMENT`, `ROLLOUT_API_BASE`, and `ROLLOUT_ENV_KWARGS`.
   - Keeps the local SGLang health check on `127.0.0.1` while passing the tunnel
     URL to Codex/Harbor through `--api-base`.

4. `SpecForge/scripts/validate_swebench_agentic_jsonl.py`
   - Recomputes windows from saved `trajectory.json` files and compares them to
     the emitted train/eval JSONL token and mask windows.

## Reproduction Commands

Run from a shell with `/persistent/.env` available.

### 1. Install Harbor with Daytona support

```shell
cd /persistent/harbor
set -a; . /persistent/.env; set +a
uv sync --extra daytona
uv run python -c "import harbor, daytona; print('ok')"
```

Observed result:

```text
ok
```

### 2. Materialize 20 SWE-bench Verified tasks

```shell
cd /persistent/harbor/adapters/swebench
set -a; . /persistent/.env; set +a
uv run swebench --task-dir /persistent/harbor/datasets/swebench-verified --limit 20
```

Observed result: 20 task directories were written under
`/persistent/harbor/datasets/swebench-verified`, with `Failures: 0`.

### 3. Serve Qwen3-4B locally

```shell
cd /persistent/SpecForge
set -a; . /persistent/.env; set +a
export HF_HOME=${HF_HOME:-/persistent/hf-cache}
export TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
export SERVED_NAME=${SERVED_NAME:-qwen3-4b}
export SERVE_TP=${SERVE_TP:-1}
export SERVE_TOOL_CALL_PARSER=${SERVE_TOOL_CALL_PARSER:-qwen25}
bash examples/run_qwen3_4b_dflash_swebench_rollout.sh serve
```

Observed result: SGLang loaded `Qwen/Qwen3-4B`, served it as `qwen3-4b`, and
reported readiness on `0.0.0.0:30000`.

### 4. Expose SGLang through a Cloudflare quick tunnel

`cloudflared` was installed locally at `/persistent/bin/cloudflared`:

```shell
mkdir -p /persistent/bin
curl -L --fail --retry 3 \
  -o /persistent/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /persistent/bin/cloudflared
/persistent/bin/cloudflared --version
```

Then the tunnel was started:

```shell
/persistent/bin/cloudflared tunnel --url http://127.0.0.1:30000
```

Observed tunnel URL:

```text
https://parade-stopping-roll-seeker.trycloudflare.com
```

The tunneled model endpoint was checked with:

```shell
curl -sf https://parade-stopping-roll-seeker.trycloudflare.com/v1/models
```

Observed response included:

```json
{"id":"qwen3-4b","object":"model","owned_by":"sglang"}
```

### 5. Smoke rollout, 1 task

```shell
cd /persistent/SpecForge
set -a; . /persistent/.env; set +a
export HF_HOME=${HF_HOME:-/persistent/hf-cache}
export TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
export SERVED_NAME=${SERVED_NAME:-qwen3-4b}
export ROLLOUT_AGENT=codex
export ROLLOUT_ENVIRONMENT=daytona
export ROLLOUT_API_BASE=https://parade-stopping-roll-seeker.trycloudflare.com/v1
export ROLLOUT_PYTHON='uv run --project /persistent/harbor --with transformers python'
export SWEBENCH_LIMIT=1
export ROLLOUT_N_CONCURRENT=1
export CODEX_WIRE_API=responses
bash examples/run_qwen3_4b_dflash_swebench_rollout.sh rollout
```

Observed summary:

```text
Rollout summary: trials=1 with_tokens=1 missing_tokens=0 resolved=0 segments=1 windows=1
Wrote 1 train rows -> /persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b/swebench_agentic_train.jsonl
```

This proved the critical path before scaling:

- Daytona built the SWE-bench task image.
- Codex installed and ran inside the Daytona sandbox.
- Codex reached SGLang through the public tunnel.
- Harbor synced `agent/trajectory.json` back to the local jobs dir.
- The driver re-tokenized the trajectory and emitted a non-empty supervised row.

### 6. Small batch rollout, 20 tasks

```shell
cd /persistent/SpecForge
set -a; . /persistent/.env; set +a
export HF_HOME=${HF_HOME:-/persistent/hf-cache}
export TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
export SERVED_NAME=${SERVED_NAME:-qwen3-4b}
export ROLLOUT_AGENT=codex
export ROLLOUT_ENVIRONMENT=daytona
export ROLLOUT_API_BASE=https://parade-stopping-roll-seeker.trycloudflare.com/v1
export ROLLOUT_PYTHON='uv run --project /persistent/harbor --with transformers python'
export SWEBENCH_LIMIT=20
export ROLLOUT_N_CONCURRENT=4
export MAX_LENGTH=16384
export CODEX_WIRE_API=responses
bash examples/run_qwen3_4b_dflash_swebench_rollout.sh rollout
```

Observed summary:

```text
Discovered 20 SWE-bench task(s) under /persistent/harbor/datasets/swebench-verified
Running rollouts: agent=codex model=openai/qwen3-4b api_base=https://parade-stopping-roll-seeker.trycloudflare.com/v1 n_concurrent=4
Loading target tokenizer for re-tokenization: Qwen/Qwen3-4B
  [warn] swe-bench/swebench-verified__astropy__astropy-13398: no Codex trajectory.json found (agent may have failed before producing a session).
Rollout summary: trials=20 with_tokens=19 missing_tokens=1 resolved=0 segments=19 windows=31
Wrote 30 train rows -> /persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b/swebench_agentic_train.jsonl
Wrote 1 eval rows -> /persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b/swebench_agentic_eval.jsonl
```

The completed job summary is:

```text
/persistent/SpecForge/cache/harbor_jobs/swebench-20260628-201553/result.json
n_total_trials: 20
n_completed_trials: 20
n_errored_trials: 6
```

The six errored trials were Daytona/Codex execution errors, but 19 trials still
produced usable `trajectory.json` files. The missing-token count is 1 because
`astropy__astropy-13398` failed before producing a Codex trajectory.

## Output Files

The rollout produced:

```text
/persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b/swebench_agentic_train.jsonl
/persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b/swebench_agentic_eval.jsonl
```

Validated output shape:

```text
swebench_agentic_train.jsonl rows=30 instances=19 min_len=2366 max_len=16384 min_loss=590 max_loss=16384
swebench_agentic_eval.jsonl  rows=1  instances=1  min_len=16384 max_len=16384 min_loss=14601 max_loss=14601
```

## Tokenization, Chat Template, and Loss Mask Construction

### Source Data

The Codex CLI does not persist raw model token IDs in ATIF. It persists a text
trajectory at `agent/trajectory.json`. Therefore this pipeline cannot prove
byte-for-byte equality with the exact token stream used inside SGLang at
inference time. Instead, it constructs a self-consistent supervised dataset by
re-rendering the saved Codex text trajectory through the target model tokenizer
and chat template.

That distinction matters:

- **Proven:** the emitted JSONL is exactly consistent with re-rendering saved
  Codex trajectories using `Qwen/Qwen3-4B`'s tokenizer/chat template in the
  current environment.
- **Not provable from Codex ATIF:** exact equality with raw SGLang internal
  prompt/completion token IDs, because Codex ATIF did not save those IDs.

For exact inference token IDs, a different agent path must record server-side
`prompt_token_ids` and `completion_token_ids` directly. The existing Terminus-2
path in `run_swebench_rollouts.py` is structured for that kind of exact-token
collection.

### Algorithm

For each saved `trajectory.json`:

1. Convert ATIF steps into OpenAI-style chat messages:
   - `source == "system"` -> `role="system"`
   - `source == "user"` -> `role="user"`
   - `source == "agent"` -> `role="assistant"`
   - Codex tool observations -> `role="tool"`
   - copied-context steps are skipped to avoid training on duplicated context

2. For every assistant message at index `k`:
   - Render `messages[:k]` with `add_generation_prompt=True`.
   - Render `messages[:k+1]` with `add_generation_prompt=False`.
   - Tokenize both rendered strings with the target tokenizer using
     `add_special_tokens=False`.
   - Find their common prefix length.
   - Mark the suffix that appears only in `messages[:k+1]` as supervised
     assistant output.

3. Render the full message list with `add_generation_prompt=False` and build:
   - `input_ids`: full rendered trajectory tokens
   - `loss_mask`: `1` for assistant-generated spans, `0` for system/user/tool
     context

4. Window long trajectories to `MAX_LENGTH=16384`, dropping windows with fewer
   than `2 * BLOCK_SIZE = 32` supervised tokens.

### Why Render Text Then Encode

Current Transformers/Qwen tokenizers can return a `BatchEncoding` from
`apply_chat_template(..., tokenize=True)`. Earlier code treated the returned
object as a simple token list, which can silently produce the wrong result.

The fixed path uses:

```python
rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=add_generation_prompt,
    tools=tools,
)
ids = tokenizer.encode(rendered, add_special_tokens=False)
```

This avoids return-shape ambiguity and makes the exact rendered chat string the
source of truth.

## Validation Commands

### Static and unit checks

```shell
cd /persistent/SpecForge
python3 -m py_compile \
  scripts/run_swebench_rollouts.py \
  scripts/validate_swebench_agentic_jsonl.py \
  /persistent/harbor/src/harbor/agents/installed/codex.py
python3 scripts/run_swebench_rollouts.py --self-test
bash -n examples/run_qwen3_4b_dflash_swebench_rollout.sh
```

Observed result:

```text
self-test: PASS
```

The self-test covers:

- prompt/completion reconstruction for exact-token rollouts
- seam tolerance and segment splitting
- window filtering
- Codex ATIF message reconstruction
- tool-call and tool-result handling
- assistant-only masking
- skipping copied context
- trial URI parsing

### JSONL shape validation

```shell
python3 - <<'PY'
import json, pathlib
base = pathlib.Path('/persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b')
for p in [base/'swebench_agentic_train.jsonl', base/'swebench_agentic_eval.jsonl']:
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    for row in rows:
        assert isinstance(row['input_ids'], list)
        assert isinstance(row['loss_mask'], list)
        assert len(row['input_ids']) == len(row['loss_mask'])
        assert sum(row['loss_mask']) > 0
        assert all(m in (0, 1) for m in row['loss_mask'])
    print(p.name, len(rows), 'rows ok')
PY
```

Observed result:

```text
swebench_agentic_train.jsonl 30 rows ok
swebench_agentic_eval.jsonl 1 rows ok
```

### Full trajectory-to-JSONL revalidation

This is the strongest validation. It re-reads every saved `trajectory.json`,
reconstructs messages, re-renders the Qwen chat template, rebuilds
`input_ids`/`loss_mask`, re-windows the data, and compares the resulting token
and mask windows to the emitted train/eval JSONL.

```shell
export HF_HOME=${HF_HOME:-/persistent/hf-cache}
uv run --project /persistent/harbor --with transformers python \
  /persistent/SpecForge/scripts/validate_swebench_agentic_jsonl.py \
  --jobs-dir /persistent/SpecForge/cache/harbor_jobs/swebench-20260628-201553 \
  --train /persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b/swebench_agentic_train.jsonl \
  --eval /persistent/SpecForge/cache/dataset/swebench-agentic-qwen3-4b/swebench_agentic_eval.jsonl \
  --target-model Qwen/Qwen3-4B \
  --max-length 16384 \
  --min-loss-tokens 32
```

Observed result:

```text
validation: PASS
emitted rows: train=30 eval=1 total=31 tokens=300644 loss_tokens=232591
trajectories: total=19 with_assistant=19 with_loss=19 windows=31
```

This proves that the current JSONL rows are exactly reproducible from the saved
Codex trajectories under the current `Qwen/Qwen3-4B` tokenizer/chat-template
implementation.

## Residual Risks and How To Tighten Further

The current Codex path is correct for **text-trajectory retokenization**. It is
not exact-token telemetry from SGLang. Residual risks are:

- Codex ATIF stores text, not SGLang `prompt_token_ids` / `completion_token_ids`.
- If Codex or SGLang applies hidden request fields that are not represented in
  ATIF text, retokenization cannot recover them.
- If `Qwen/Qwen3-4B` updates its tokenizer or chat template upstream and the
  local cache changes, future revalidation may differ. Pin a model revision for
  fully immutable reproduction.
- Minimal synthetic tool schemas are used when rendering trajectories with tool
  calls. The completed 20-task batch still revalidates because its emitted rows
  and fresh re-render agree under the same schema logic.

For stricter guarantees, add a Codex/SGLang telemetry mode that stores raw
`prompt_token_ids` and `completion_token_ids` for each model call, then build
`loss_mask` from those IDs directly. That would upgrade the guarantee from
"perfectly self-consistent retokenization of Codex text trajectories" to
"exact inference-token supervision."

## Final Status

The requested Phase A output exists and validates:

- 20 SWE-bench tasks were attempted through Daytona.
- 19 saved Codex trajectories produced supervised token windows.
- 31 pretokenized windows were emitted.
- Train/eval JSONL rows have aligned `input_ids` and `loss_mask` lists.
- Every emitted row has nonzero supervised tokens.
- A fresh re-tokenization of saved trajectories exactly reproduces all emitted
  token/mask windows.

