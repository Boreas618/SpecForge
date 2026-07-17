# Data Regeneration Infrastructure Plan

> **Status:** R0–R3 implemented; R4 is next. `specforge/data/regen` is the
> generic runtime extracted from the `_temp/regen` prototype (see
> `docs/adr/0001-data-regeneration-boundary.md` and
> `docs/audits/data-regeneration-publication.md`). `_temp/regen` is retained as a
> non-package migration input until its remaining material (Inkling template,
> campaign docs) clears provenance review.
>
> **Decision:** SpecForge will own a first-class **data regeneration runtime** under
> `specforge/data/regen/`. It is the canonical producer of any first-party dataset
> whose text messages, reasoning traces, tool interactions, or text candidates are
> created by a model. Its payload is exclusively decoded text plus structured
> trajectory metadata—never hidden states, logits, embeddings, or other model tensors.
> A run is not a loose JSONL file: it becomes usable by downstream consumers only after it is
> finalized as a content-addressed, validated `DatasetArtifact`.

This is the remaining active track document; the domain, online-disaggregation,
and eval planning docs were retired when the unified-runtime refactor (#678)
consolidated training behind one typed run configuration.

---



## 1. Why this is necessary SpecForge infrastructure

Speculative-model quality depends on more than having syntactically valid chat rows. The regenerated trajectory must match the frozen target's serving behavior: later turns must condition on regenerated history, reasoning and tool blocks must round-trip through the serving renderer, and SpecForge must supervise exactly the tokens that the target sampled. A silent template, tokenizer, or tool-JSON rewrite can train the draft on transitions that never occur in deployment.

SpecForge already owns the downstream pieces that make this a framework concern: chat templates, loss masks, tokenizer and target identity, text-dataset preparation, training configuration, and checkpoint provenance. Leaving regeneration as an external script creates an unverifiable gap immediately before all of those pieces.

There are currently two incomplete paths:


| Path                                                                        | What it gets right                                                                                                                                                                                              | Why it is not the product boundary                                                                                                                                                                   |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/regenerate_train_data.py` + `scripts/validate_regenerated_data.py` | Public ShareGPT flow, several ordinary chat models, multiple endpoints, reasoning on/off                                                                                                                        | Weak provenance and resume contracts; no shard manifest or exact merge; validation is structural; model and operation logic live in one script                                                       |
| `_temp/regen`                                                               | Stable row contracts, on-policy multi-turn replay, deterministic request seeds, exact resume, pinned contiguous shards, worker locks, agentic raw-token transport, exact rebuild, renderer/loss-mask validation | Defaults and contracts are tied to particular datasets, Inkling/Kimi codecs, local paths, and a single operational campaign; preparation, scheduling, transport, policy, and storage are interleaved |
| `scripts/prepare_data.py`                                                   | Broad public dataset ingestion and conversion into SpecForge's chat shape                                                                                                                                       | Each dataset is another CLI branch; source loading, schema normalization, sampling, and output are not reusable regeneration components                                                              |


The merge is therefore an **extraction and generalization**, not a directory copy. The prototype supplies the hard-won invariants and golden behavior. SpecForge supplies the public contracts, registries, CLI, artifact lifecycle, training integration, and long-term compatibility policy.

Once Phase R3 lands, the supported path is:

```text
source revision + regeneration recipe + frozen generator revision
                              |
                              v
                 text/trajectory regeneration
                              |
                              v
                 validated DatasetArtifact
                 (messages + reasoning/tool
                  trajectories + provenance)
                              |
                              v
               text-dataset consumers such as
               tokenization, training, and eval
```

Direct JSONL remains valid for original or user-authored datasets. The artifact requirement applies to **SpecForge-produced model-derived data**; it does not force every external dataset through a model-generation pipeline.

## 2. Product definition and scope

The feature is called **data regeneration** even though it supports more than replacing answers. A regeneration recipe is a reproducible, row-oriented workflow that reads one or more versioned sources, applies model-backed and deterministic operations, validates the result, and publishes an immutable dataset artifact.

The same kernel must support these workloads without adding a new top-level script:


| Workload                             | Operation expressed by the recipe                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Target-aligned replay                | Preserve system/user turns, replace every assistant turn, and condition later turns on the new history                         |
| Prompt completion                    | Generate an assistant turn for prompt-only rows                                                                                |
| Reasoning trajectory capture         | Store the visible answer and complete structured reasoning for every generated turn; derive narrower views without mutating the captured artifact |
| Agentic trajectory regeneration      | Preserve/replay or execute tool results, require tool-call shape where configured, and prove exact render rebuild              |
| Multi-candidate curation             | Generate `n` candidates with one or several models, score/filter them, and retain a declared selection or all variants         |
| Multi-corpus assembly                | Normalize, namespace, filter, sample, and mix several pinned sources in one recipe                                             |
| Dataset refresh                      | Re-run the same recipe against an explicitly changed source/model revision and produce a new artifact with a diffable manifest |
| Cross-model critique and selection   | Generate, critique, revise, rank, or retain text candidates while preserving the complete stage history                     |


The first executable release remains intentionally narrower: JSONL/Hugging Face chat sources, linear workflows, chat and raw-token SGLang-compatible generation, local artifacts, and the replay/prompt-completion operations. The contracts are shaped for the broader table from the start, so later breadth adds adapters rather than breaking the artifact schema.

## 3. Architectural decisions



### 3.1 Orthogonal axes, not model × dataset branches

The pipeline composes named resources. No core module may branch on a dataset name,
model family, or private profile.


| Axis           | Contract                         | Examples                                                                                    |
| -------------- | -------------------------------- | ------------------------------------------------------------------------------------------- |
| Source         | `SourceAdapter`                  | local JSONL, Hugging Face revision, iterable/plugin source                                  |
| Row schema     | `RecordAdapter`                  | ShareGPT, OpenAI messages, prompt/completion, typed tool trajectory                         |
| Selection      | `SelectionPolicy`                | all, stable indices, seeded sample, filter, weighted source mix                             |
| Operation      | `Operation`                      | replay assistants, complete prompt, derive reasoning, generate candidates, select candidate |
| Generation     | `GenerationBackend`              | OpenAI-compatible chat, SGLang raw token, in-process target, test fake                      |
| Serving format | `ConversationCodec`              | tokenizer chat template, split reasoning, typed Inkling blocks, Kimi grammar                |
| Tools          | `ToolPolicy` / `ToolEnvironment` | reject, preserve-and-shape-match, deterministic replay, live sandbox                        |
| Quality        | `Validator`                      | schema, preservation, control-token leak, exact rebuild, loss-mask parity, thresholds       |
| Storage        | `ArtifactStore`                  | local filesystem first; object storage later                                                |
| Execution      | `Executor`                       | local async, static shard worker, leased distributed worker                                 |


Each registered component declares capabilities such as `chat`, `reasoning_split`, `raw_token_ids`, `tools`, `request_seed`, `logprobs`, and `multiple_candidates`. Planning computes the capabilities required by every workflow stage and fails before scanning or generating data if the selected components cannot satisfy them. For example, exact tool call rebuild requires a raw-token backend and a codec that implements `rebuild_ids`; it must never silently fall back to parsed chat.

### 3.2 A small workflow, not a general DAG engine

A recipe owns mappings of named `sources` and `generators`, followed by an ordered list of named `stages`. A stage consumes record envelopes and emits zero or more envelopes. This is enough for multi-source and multi-model workflows:

```text
source A --+
           +--> normalize/mix --> teacher-A candidates --+
source B --+                                           judge-B --> artifact
                               teacher-C candidates --+
```

Version 1 executes stages linearly and permits fan-out through explicit variants. It does not implement arbitrary cycles, joins, or a general workflow language. If a later use case genuinely needs a DAG, the named-stage provenance already provides a migration path.

### 3.3 Text and reasoning trajectories only

This boundary is absolute. Regeneration reads and writes JSON-serializable semantic
records: message roles and text, `reasoning_content`, tool calls and results, finish
metadata, candidate relationships, and provenance. It does not request, capture,
validate, store, transport, or cache hidden states, logits, embeddings, KV caches, or
other model tensors.

Any current path that combines textual regeneration with model-feature capture must be
split. Feature capture remains outside this plan and is neither a
dependency nor an extension point of `specforge/data/regen`.

A raw-token backend may use sampled token IDs transiently to parse a response and prove
that the stored text/tool trajectory re-renders exactly. Those token IDs are serialization
evidence, not model hidden states; the finalized dataset payload remains semantic text and
structured trajectory data.

### 3.4 Validation is a publication boundary

Generation success is not artifact success. Shards are **attempt outputs** until all
required validators pass, coverage and reject-rate policies are satisfied, and final
checksums are written. `specforge train` consumes a finalized manifest, never a worker's
partial success file.

Validators are composable findings producers, not one growing `if model_family == ...`
function. A recipe selects a validation profile, which expands to registered validators
and thresholds. Model codecs may contribute required validators, but cannot disable the
baseline structural, provenance, coverage, or source-preservation gates.

### 3.5 Reproducible control, honest stochasticity

SpecForge guarantees that fixed input bytes/revisions and a fixed recipe produce the
same plan, eligible-row set, stable row identities, stage request seeds, shard ownership,
merge order, and provenance. It does **not** claim that an arbitrary inference engine is
numerically deterministic across GPU types, kernels, or server versions. The manifest
records enough engine/model/tokenizer/sampling identity to explain that boundary.

## 4. Target architecture

```text
                         Recipe (typed, canonical JSON digest)
                                      |
                                      v
Planner: resolve registries -> negotiate capabilities -> pin revisions/fingerprints
                                      |
                                      v
SourceAdapter -> RecordAdapter -> SelectionPolicy -> RecordEnvelope
                                      |
                         ordered Operation stages
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
              GenerationRequest             deterministic transform
                         |
              GenerationBackend <----> ConversationCodec
                         |
              GenerationResult + raw-evidence digest
                         |
              per-stage / per-record validators
                         |
                         v
              immutable shard + attempt journal
                         |
             finalize: coverage + shard compatibility + gates
                         |
                         v
                    DatasetArtifact
        (manifest + text/trajectory data + reports)
                         |
                         v
        text preparation / training / eval consumers
```

The planner emits an immutable `run-plan.json` before workers start. Workers receive
only a plan digest, shard/task assignment, endpoints, and runtime credentials. Every
worker verifies the source and plan identities before appending an attempt.

## 5. Canonical contracts and invariants



### 5.1 `RecordEnvelope`

The envelope separates framework metadata from the payload schema:

```python
RecordEnvelope(
    key=RecordKey(source="blend", source_id="train:123", variant="base"),
    input_position=123,
    payload=<typed conversation / prompt / preference record>,
    source_fingerprint="sha256:...",
    stage_history=(...),
)
```

Required rules:

- `RecordKey` is type-preserving and namespaced by source; integer `1` and string
`"1"` are not the same key.
- Source IDs are stable source identifiers when available and deterministic positional
IDs otherwise. A stage never silently invents a random row ID.
- The original normalized payload or its canonical digest remains available for
preservation validators. Generated output never overwrites the only copy of the
source evidence.
- Payloads use typed messages/content parts. Unknown source fields may be retained in
an extension map; they are not casually copied into model requests.
- Each fan-out candidate receives a deterministic variant key. Retries keep the same
key; deliberate rerolls use a new declared variant.



### 5.2 Generation request/result

`GenerationRequest` contains only normalized serving inputs: stage/record identity,
messages or input token IDs, tools, sampling, stop policy, and derived request seed.
`GenerationResult` retains the structured assistant output, finish metadata, usage,
backend/model/codec identities, and optional raw token IDs or bytes.

Capture is lossless at the semantic trajectory level. If the model emits reasoning,
visible text, tool calls, or multiple typed blocks, the backend/codec must preserve all
of them in `GenerationResult`; it may not silently drop a block because a downstream
training view does not use it. A recipe may explicitly disable thinking before generation,
which means no reasoning was emitted. That is different from discarding emitted reasoning.

The raw stream may be large or sensitive. Exact-rebuild codecs compare it in memory and
store a digest, token count, and mismatch diagnostic by default. Full raw streams are an
explicit debug artifact with separate retention controls; they are not copied into every
training row.

Every structured result must answer this question: **can the stored record reproduce the
serving sequence that was sampled?** Depending on the codec, the proof is one of:

1. token-identical `rebuild_ids` (required for tool-call/raw-token paths);
2. byte/token parity through the same packaged renderer;
3. an explicit weaker `structured_chat` guarantee, allowed only for recipes that do not
  require exact rebuild and recorded in the manifest.



### 5.3 Recipe and provenance identity

The recipe digest is SHA-256 over canonical, secret-free JSON after defaults and registry
versions are resolved. It includes:

- every source identifier, revision/config/split, adapter, and source fingerprint;
- selection/filter/mix rules and seeds;
- ordered stage definitions and component versions;
- model, tokenizer, chat-template/codec, and server compatibility identity;
- all sampling, reasoning, stop, tool, retry, and truncation policies;
- validation profiles and thresholds;
- output schema, sharding strategy, and artifact format version.

Endpoint credentials, tokens, and raw environment values are never part of the recipe or
logs. A redacted endpoint pool label may be recorded; the actual endpoints are worker
runtime configuration.

Per-turn seeds are derived from `(recipe seed, record key, stage id, variant id, generation ordinal)` with a versioned hash strategy. Scheduling, retry order, and endpoint
choice therefore cannot change a request seed.

### 5.4 Failure model

Attempts use stable categories so retry and quality policy are independent:


| Category              | Examples                                                      | Default disposition                     |
| --------------------- | ------------------------------------------------------------- | --------------------------------------- |
| `source_invalid`      | bad role sequence, malformed tool binding                     | terminal reject                         |
| `capability_error`    | tool stage on chat-only backend                               | fail plan/run, never row-level skip     |
| `transport_retryable` | timeout, 429, transient 5xx                                   | bounded retry, then unresolved error    |
| `transport_terminal`  | authentication, invalid request                               | fail fast or unresolved error by policy |
| `generation_invalid`  | length truncation, empty required field, leaked control token | terminal reject or declared reroll      |
| `exact_rebuild`       | sampled token stream differs from stored re-render            | terminal reject                         |
| `policy_reject`       | tool-call shape changed, judge threshold failed               | expected reject, counted separately     |
| `validation_failed`   | preservation or loss-mask mismatch                            | quarantine artifact; never publish      |
| `internal_error`      | invariant violation, serialization bug                        | fail shard and preserve diagnostic      |


Success, expected reject, unresolved error, and resolved retry are separate counters. An
old error line does not make a later success ambiguous: finalization chooses the latest
valid attempt under a versioned, explicit resolution policy and reports the history.

### 5.5 Shards, resume, and finalization

The prototype's strongest operational guarantees become framework invariants:

- Static workers pin the plan digest, source fingerprint, row range/hash partition,
generation-seed strategy, and shard index in an atomic sidecar before work begins.
- A success or error attempt records its exact input position/task identity. Resume
refuses missing or incompatible sidecars, duplicate keys, foreign positions, or a
changed recipe.
- Success and attempt journals are independently locked; two workers cannot own the same
local shard output.
- Shards are immutable after completion. Retry writes a new attempt segment instead of
editing published bytes in place.
- Finalization requires the expected shard set, one compatible plan/source identity,
disjoint ownership, accounted-for coverage, and all required validation reports.
- Final data is ordered deterministically by source order, input position, stage variant,
and key—not by request completion time. Plain `cat` is permitted only when the manifest
proves that shard layout already has that order.
- All files in the final manifest have byte counts, row counts, and SHA-256 digests.



### 5.6 `DatasetArtifact`

The local v1 artifact layout is deliberately boring and inspectable:

```text
artifact/
├── manifest.json                 # final, content-addressed product contract
├── state.json                    # mutable work state; excluded from the artifact payload/digest
├── recipe.json                   # resolved secret-free recipe
├── run-plan.json                 # source identities + task/shard plan
├── data/
│   ├── part-00000.jsonl
│   └── part-00001.jsonl
├── rejects/
│   └── part-*.jsonl              # safe row identity + category + diagnostic
├── attempts/
│   └── part-*.jsonl              # optional operational history/retention policy
└── reports/
    ├── validation.json
    └── summary.json
```

`manifest.json` contains schema/version, artifact digest, recipe digest, source and
generator identities, codec/exactness level, counts by status/category/source/stage,
validation results, shard list and digests, creation tool version, and parent artifact(s)
for derived/mixed outputs. It contains no dataset rows, secrets, private endpoint values,
or unrestricted exception dumps. The artifact digest is SHA-256 over the canonical
manifest payload with its self-digest field omitted; that payload contains the sorted
path and digest of every immutable payload file (`state.json` is work metadata). It is
not path-derived or self-referential.

An artifact has an explicit state:

```text
PLANNED -> RUNNING -> COMPLETE_UNVALIDATED -> VALIDATED -> FINALIZED
                       |                         |
                       +------> QUARANTINED <----+
```

Only `FINALIZED` artifacts are accepted by normal first-party dataset consumers.

## 6. Package and ownership layout

```text
specforge/data/regen/
├── contracts.py              # RecordKey/Envelope, request/result, finding, task contracts
├── recipe.py                 # Pydantic recipe schema + canonical secret-free digest
├── registry.py               # typed registries and capability negotiation
├── planner.py                # pin sources, select rows, create run-plan/tasks
├── pipeline.py               # ordered stage execution; no model/dataset branches
├── artifact.py               # manifests, atomic lifecycle, local ArtifactStore
├── finalize.py               # coverage, attempt resolution, deterministic merge
├── errors.py                 # stable failure taxonomy and safe diagnostics
├── sources/
│   ├── jsonl.py
│   └── huggingface.py
├── records/
│   ├── sharegpt.py
│   └── openai_messages.py
├── operations/
│   ├── replay.py
│   ├── complete.py
│   └── candidates.py
├── backends/
│   ├── openai_chat.py
│   ├── sglang_raw.py
│   └── fake.py
├── codecs/
│   ├── chat_template.py
│   ├── split_reasoning.py
│   ├── inkling.py
│   └── kimi_k3.py
├── validators/
│   ├── baseline.py
│   ├── preservation.py
│   ├── exact_rebuild.py
│   ├── render_parity.py
│   └── loss_mask.py
└── cli.py

tests/test_data/test_regen/
examples/data_regeneration/recipes/
docs/concepts/data-regeneration.md
```

`specforge.data` currently imports preprocessing code eagerly. Phase R1 makes its public
imports lazy so planning, manifest inspection, merge, and CPU validation do not import
Torch, distributed training, or model backends. Backend and tokenizer dependencies remain
lazy/optional at their adapter boundary.

Model-specific server launch flags do not belong in this package. They remain deployment
recipes under `examples/data_regeneration/` (or private operator configuration), while the
resolved model/codec compatibility identity is recorded by the artifact.

## 7. User-facing recipe and CLI

The recipe is separate from the training config because it has a different lifecycle and
may produce artifacts used by many training runs. Training refers to the finalized artifact
URI/digest.

This target example deliberately uses multiple sources and models:

```yaml
version: 1
seed: 20260717

sources:
  blend:
    source: {type: huggingface, id: org/blend, revision: 7e91c2a, split: train}
    record_adapter: sharegpt
    select: {sample: 800000, seed: 11}
  tools:
    source: {type: jsonl, path: /datasets/tools.jsonl}
    record_adapter: openai_messages
    select: {sample: 200000, seed: 12}

generators:
  teacher:
    backend: openai_chat
    model: org/teacher-model
    revision: 83b5f10
    tokenizer: org/teacher-model
    codec: auto
    sampling: {temperature: 0.7, top_p: 0.95, max_tokens: 8192}
  judge:
    backend: openai_chat
    model: org/judge-model
    revision: 29aa419
    codec: chat_template
    sampling: {temperature: 0, max_tokens: 1024}

workflow:
  - id: regenerate
    operation: replay_assistants
    generator: teacher
    candidates: 2
    tool_policy: preserve_shape
  - id: choose
    operation: select_candidate
    generator: judge

validation:
  profiles: [baseline, source_preservation, codec_required, specforge_loss_mask]
  max_unresolved_error_rate: 0.005
  max_policy_reject_rate: 0.05

output:
  uri: /artifacts/blend-tools-v1
  format: jsonl
  shards: 32
```

The planner resolves `codec: auto` to a named, versioned codec and writes that choice into
`recipe.json`; `auto` never remains ambiguous in a running job.

The CLI extends the existing `specforge` entry point:

```bash
# Resolve adapters/capabilities, pin source identity, and show cost/row estimates.
specforge data regen plan --config regen.yaml

# Convenience local runner; creates the plan if needed.
specforge data regen run --config regen.yaml

# Independently launched static workers use the same immutable plan.
specforge data regen worker --plan /artifacts/blend-tools-v1/run-plan.json \
  --shard-index 7

# Retry only retry-eligible unresolved attempts without changing request identity.
specforge data regen retry --artifact /artifacts/blend-tools-v1 \
  --categories transport_retryable

# Run/refresh validators, then atomically publish the final manifest.
specforge data regen validate --artifact /artifacts/blend-tools-v1
specforge data regen finalize --artifact /artifacts/blend-tools-v1

# Human and machine-readable provenance/debug views.
specforge data regen inspect --artifact /artifacts/blend-tools-v1 --json
```

Training consumes the artifact, not a guessed data file:

```yaml
data:
  dataset_artifact: /artifacts/blend-tools-v1/manifest.json
  max_length: 4096
```

The loader verifies the manifest state and file digests, then includes the artifact digest,
codec/template identity, and processed-data cache key in run/checkpoint provenance. A temporary
`data.allow_unverified_regenerated_jsonl` escape hatch exists only during migration and is noisy
in logs and checkpoints.

## 8. What moves from the prototype


| Current file                                 | Destination / treatment                                                                                                                                                       |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_temp/regen/contracts.py`                   | Split generic IDs, canonical JSON, source preservation, deterministic seeds, and shard identities into core contracts; move dataset normalization into record/source adapters |
| `_temp/regen/prepare.py`                     | Replace source `choices=` branching with `SourceAdapter` + `RecordAdapter` + `SelectionPolicy`; preserve seeded sampling and atomic writes                                    |
| `_temp/regen/regenerate.py`                  | Split `GenerationBackend`, replay `Operation`, executor, retry policy, shard planner, attempt journal, and artifact writer; no monolithic CLI module remains                  |
| `_temp/regen/inkling.py`                     | Move raw-token generation/rebuild behavior behind the generic raw backend and Inkling codec; retain the token-identical gate                                                  |
| `_temp/regen/kimi_k3.py`                     | Move grammar and reasoning split into a Kimi codec profile; remove model-family branching from the executor                                                                   |
| `_temp/regen/specforge_bridge.py`            | Delete after validators can import the public template/parser APIs normally; packaging must not load training-only dependencies for validation                                |
| `_temp/regen/inkling_chat_template.jinja`    | Package only after license/publication review; otherwise load it through an external/private codec plugin while the public codec contract remains                             |
| `_temp/regen/validate.py`                    | Split baseline artifact validators from codec/SpecForge validators; preserve coverage, shard, special-token, render, and mask gates                                           |
| `_temp/regen/tests/`                         | Port as golden contract tests, replacing private/local paths with synthetic or approved public fixtures                                                                       |
| `_temp/regen/*.sh`, README campaign commands | Convert safe settings into example recipes and operator docs; keep model serving and cluster addresses outside the library                                                    |
| `_temp/regen/AGENTIC.md`                     | Distill the generic raw-token/exact-rebuild design into `docs/concepts/data-regeneration.md`; retain campaign-specific/private details only where approved                    |

Any hidden-state or tensor-generation behavior found in a current regeneration entry point is
intentionally **not migrated**. It remains in, or is moved to, a separately named feature-preparation
command. The new regeneration recipe schema has no hidden-state fields, the backend result contract
has no tensor payload, and compatibility wrappers reject rather than reinterpret such options.


The existing public scripts converge too:

- `scripts/regenerate_train_data.py` and `scripts/validate_regenerated_data.py` are
**removed outright** (decision revised from the originally planned one-release
compatibility wrappers): the recipe CLI is the only supported path, and their
reasoning contracts moved into recipe sampling policies, codec control tokens,
and the artifact validation profiles.
- `scripts/prepare_data.py` adopts the same `SourceAdapter`/`RecordAdapter` registry incrementally
so ordinary preparation and model-backed regeneration do not maintain competing schema
normalizers. Pure preparation remains usable without invoking a generator.
- `examples/data_regeneration/*.sh` move model choices into recipe files and invoke the CLI.
- No old path is deleted until its golden input produces contract-equivalent rows and the new
artifact has equivalent or stronger accounting; that parity was demonstrated before removal.

Restricted model paths, dataset locations, server addresses, dataset contents, credentials, and
campaign dates are not migrated into the public package or fixtures.

## 9. Implementation phases



### R0 — Contract freeze and publication audit · size S · CPU · status: implemented

- **Goal:** preserve the prototype's valuable behavior before refactoring and separate public
mechanisms from campaign-specific material.
- **Implementation:**
  - Inventory every field written by chat, Kimi, and agentic paths; document which are source,
  attempt, generation, shard, and validation metadata.
  - Create approved synthetic/public golden fixtures for multi-turn reasoning, truncation,
  control-token leakage, tool-call shape mismatch, non-canonical tool JSON, retry, and shard
  resume.
  - Record legacy output/sidecar/report snapshots with volatile timestamps/endpoints redacted.
  - Run license, secret, and private-path review over templates, docs, fixtures, and defaults.
  - Write a short ADR fixing the decisions in this plan: `specforge/data/regen`, JSONL artifact
  v1, named linear stages, and manifest-gated publication.
- **Tests / gates:** existing `_temp/regen/tests` and public script tests pass before extraction;
golden fixture checksums are stable; repository secret scan is clean.
- **Done when:** every behavior to preserve has a public-safe fixture and every item that cannot
be published has an explicit adapter/plugin boundary instead of an accidental hard dependency.



### R1 — Generic kernel and artifact lifecycle · size L · CPU · status: implemented

- **Goal:** land a dependency-light pipeline with no model/dataset conditionals and make
finalization a real correctness boundary.
- **Implementation:**
  - Add the package layout from §6: contracts, recipe, registries, planner, linear pipeline,
  local artifact store, failure taxonomy, finalizer, and fake backend.
  - Implement local JSONL + generic OpenAI-message/ShareGPT adapters, stable all/index/sample
  selection, replay/prompt-completion operations, and baseline validators.
  - Generalize the prototype's canonical JSON, type-preserving IDs, content fingerprints,
  deterministic per-stage seeds, contiguous sharding, locks, exact resume, atomic sidecars,
  attempt resolution, and coverage checks.
  - Make planning and inspection bounded-memory. Large source scans write a compact plan/index
  database or sorted task file rather than retaining every normalized row in Python memory.
  - Make `specforge.data` imports lazy and keep HTTP/tokenizer/model imports inside adapters.
- **Tests / gates:** registry collision/capability tests; recipe digest injectivity; same bytes at
different paths have the documented identity semantics; 1-shard and 5-shard plans have exactly
the same union; crash/retry/duplicate/foreign-position fault injection; deterministic finalize
output; no-Torch import smoke.
- **Done when:** a fake backend can regenerate, interrupt, resume, validate, and finalize a
multi-source synthetic artifact, and adding a fake source/codec/backend touches no pipeline
branch.



### R2 — Adapter migration and vertical-slice parity · size L · CPU + live engine · status: implemented (Inkling exact-raw slice remains an external codec plugin pending publication approval)

- **Goal:** replace both existing regeneration paths with registered components while preserving
their strongest guarantees.
- **Implementation:**
  - Add the OpenAI-compatible chat backend, raw-token SGLang backend, ordinary chat-template and
  split-reasoning codecs, and codec-required validator hooks.
  - Port four vertical slices where publication permits: public ShareGPT/Qwen reasoning-off/on,
  PerfectBlend/Inkling split reasoning, Kimi client-side grammar splitting, and Inkling agentic
  raw-token exact rebuild.
  - Extract tool behavior into `ToolPolicy`: `reject`, `preserve_shape`, and deterministic recorded
  result replay. Keep live tool execution out of this phase.
  - Package renderer/template identity and replace `specforge_bridge.py` import surgery with a
  supported, lightweight parser/template API.
  - Port smoke tests so live servers are opt-in integration gates, never ordinary unit-test
  requirements.
- **Tests / gates:** golden contract equivalence with R0; on-policy later-turn request equality;
raw-token non-canonical JSON rejection; Kimi grammar rejection; special-token leak defense;
real renderer ↔ codec parity; SpecForge loss-mask parity; fake-server retry and response-shape
tests.
- **Done when:** each approved legacy profile is config, registered adapters, and validators—not
an executor branch—and `_temp/regen` contains no unique executable behavior.



### R3 — Composition, CLI, and required dataset boundary · size L · CPU · status: implemented

- **Goal:** make regeneration a visible SpecForge feature and make its artifact the canonical
text/trajectory boundary for model-derived datasets.
- **Implementation:**
  - Add `specforge data regen {plan,run,worker,retry,validate,finalize,inspect}` and typed YAML/JSON
  recipes with dotted overrides consistent with the training CLI.
  - Support named multiple sources, deterministic source mixing, named multiple generators,
  explicit variants, and linear generate/filter/select stages. Record per-stage provenance in
  every derived row and in the artifact graph.
  - Add `DataConfig.dataset_artifact`; verify final state and file digests before preprocessing.
  Feed the artifact digest and semantic preprocessing fields into content-addressed text
  tokenization and loss-mask cache keys.
  - Add the artifact/recipe/model/codec digests to training run metadata and checkpoints.
  - Remove the old scripts once recipe parity is demonstrated (revised from the originally
  planned one-release wrappers) and convert examples to recipes. Remove `_temp/regen` only
  after parity and documentation gates pass.
- **Tests / gates:** CLI subprocess tests; multi-source identity collision test; two-generator
candidate/select test with fakes; finalized artifact accepted by train-data preparation;
unfinalized/tampered artifact rejected; artifact byte change invalidates processed cache;
text-dataset loader smoke preserves the manifest digest in its run metadata.
- **Done when:** a documented user can plan through text-dataset preparation from one recipe without importing
campaign code, every first-party regeneration example produces a finalized artifact, and the
normal data-consumer path refuses partial or tampered regeneration artifacts.



### R4 — Distributed scale and operational hardening · size L · infra · status: implemented (local/lease/object-store seams; cloud blob adapter and 1M-row soak remain deployment work)

- **Goal:** run many sources/models and billion-row-scale plans across independent workers without
  weakening identity, resume, or publication guarantees.
- **Implementation:**
  - Add hash partitioning and a regeneration-owned row-task lease/heartbeat/reconcile interface
  alongside static contiguous shards. Do not depend on feature-capture runtime contracts.
  - Add endpoint-pool health, rate/concurrency budgets, circuit breakers, fair scheduling by
  generator, bounded in-flight bytes/tokens, and per-category retry budgets.
  - Add an `ArtifactStore` backend for object storage with conditional/atomic manifest publish,
  multipart shard writes, and checksum verification on read.
  - Add restart reconciliation, orphan attempt cleanup, compacted attempt indexes, metrics, and
  redacted structured logs.
  - Benchmark planning memory, generation throughput, finalization throughput, and validator cost
  separately; expensive token/mask validators support deterministic samples plus a required
  full-scan baseline.
- **Tests / gates:** kill/restart workers during request/write/finalize; lease expiry and duplicate
completion races; static-versus-leased coverage equivalence; object-store publish race; changed
source bytes or recipe digest rejected on every worker; 1M-row soak without O(rows) resident
memory in workers/finalizer.
- **Done when:** worker count and endpoint placement change scheduling only, never ownership,
request identity, accounting, or final artifact order, and a failed publisher cannot expose a
manifest that references missing/corrupt data.



### R5 — Operation and trajectory breadth · size L · CPU + engine/sandboxes · status: implemented (deterministic/sandbox-declared environments; live network sandboxes remain deployment work)

- **Goal:** prove the abstraction is a general SpecForge data feature rather than a chat replay
utility.
- **Implementation:**
  - Add sandboxed `ToolEnvironment` execution with declared tool versions, recorded request/result
  evidence, side-effect policy, timeouts, and deterministic fake environments for tests.
  - Add critique, revise, branch, merge-by-selection, and text candidate-pair operations without
  flattening their stage relationships into assistant message strings.
  - Add richer multi-agent and tool-result trajectories while preserving typed authorship,
  call/result bindings, and complete reasoning history.
  - Stabilize a third-party adapter API and optional entry-point discovery only after the internal
  contracts have survived R2/R3. Plugin versions and code identity become recipe inputs.
- **Tests / gates:** deterministic fake tool loop; forbidden side-effect test; critique/revision
stage provenance; candidate-pair artifact consumed by a purpose-built reader; external adapter
capability/version negotiation.
- **Done when:** at least one tool-executed multi-agent workflow and one candidate/critique
workflow use the unchanged planner, executor, artifact lifecycle, and finalizer.



## 10. Dependencies and delivery order

```text
R0 contract/audit
        |
        v
R1 kernel/artifact
        |
        v
R2 adapter parity
        |
        v
R3 CLI/composition/dataset boundary
        |                         \
        v                          v
R4 distributed hardening      R5 trajectory/operation breadth
```

R0–R2 can proceed independently of the inference/feature runtime and algorithm breadth. R3 touches
the typed CLI/config and text-dataset preparation boundary, so it coordinates with the domain
track's run-surface work. No regeneration phase depends on feature capture or feature transport.

## 11. Cross-phase quality gates

Every phase preserves these gates:

1. **Source preservation:** fields declared immutable by the operation have canonical digest
   equality between source and output.
2. **On-policy history:** each later generation request contains prior regenerated assistant/tool
   history, never the replaced source assistant output.
3. **Lossless trajectory capture:** every emitted reasoning, text, and tool block survives in the
   canonical result; a narrower training view is a separate derived artifact.
4. **Render exactness:** codecs that claim exact rebuild prove token equality; weaker codecs state
   and test their exactness level.
5. **Loss-mask parity:** finalized training rows render through the packaged serving-compatible
   template and SpecForge parser with the expected supervised spans.
6. **Coverage/accounting:** every planned task is exactly one of success, policy reject, terminal
   source reject, or unresolved error under the final resolution policy; thresholds are enforced.
7. **Topology independence:** local, statically sharded, and eventually leased execution produce
   the same task set, request identities, and deterministic final order.
8. **Tamper detection:** changing source, recipe, sidecar, shard, report, or final data bytes fails
   resume/finalization/consumption at the earliest boundary.
9. **Import isolation:** plan/inspect/finalize and their unit tests run without GPUs or loading
   model/training dependencies.
10. **Safe diagnostics:** fixtures/logs/reports contain no secrets, full private paths, unbounded
    source text, or raw server exception bodies.



## 12. Security and operational rules

- Credentials are provided through environment/secret injection to workers and are never accepted
as recipe values that could be serialized.
- Source and model revisions must be pinned for a final artifact. Mutable aliases may be used only
during exploration; planning resolves and records the immutable revision.
- Remote-code datasets/models are disabled by default and require an explicit trust policy recorded
in the plan.
- Tool execution is disabled until R5 and then runs in a declared sandbox; replaying recorded tool
results is not presented as live execution.
- Error messages are categorized and bounded. Full prompts/responses and raw token streams require
an explicit debug retention policy and a protected artifact location.
- Artifact finalization is atomic. A directory existing at the desired path does not imply that it
is complete; only a verified final manifest does.
- Server launchers must validate model/tokenizer/codec compatibility through a probe. Endpoint
placement, cluster topology, and private engine flags remain operator concerns.



## 13. Non-goals

- This is not a universal ETL/data-lake system. Deterministic non-model transforms may compose with
regeneration, but arbitrary SQL/stream processing is out of scope.
- It never captures or emits hidden states, logits, embeddings, KV caches, or tensor features.
  Existing code that mixes those concerns with regeneration must be split rather than migrated
  into this package.
- It does not manage the lifecycle of inference servers in the core library. Examples may launch a
server, and future orchestration may call an external provider, but generation uses a backend
contract.
- It does not guarantee identical sampled text across engine builds or hardware. It guarantees
request identity and records the execution identity.
- It does not silently coerce lossy model output into a trainable row. Truncation, malformed control
grammar, missing reasoning when required, and exact-rebuild mismatch are rejects.
- Version 1 does not support arbitrary workflow DAGs or live tool side effects.
- Model/dataset-specific quality policy is not smuggled into core defaults; it lives in named,
versioned validation profiles and recipes.



## 14. Track completion criteria

This track is complete when all of the following are true:

- `_temp/regen` has been removed because every approved generic behavior and test lives under
`specforge/data/regen`, not because the prototype was abandoned.
- Adding a new source schema, model serving format, generation backend, or validator is one
registered component plus tests; the planner/executor/finalizer need no family branch.
- One recipe can name multiple datasets and multiple models and produces stage/variant provenance
that is unambiguous at row and artifact level.
- Interrupted and differently sharded executions preserve task coverage, request seeds, exact
resume semantics, and deterministic final ordering.
- Agentic raw-token rows either prove exact rebuild and SpecForge loss-mask parity or are rejected.
- Canonical rows retain every emitted text, reasoning, and tool block; any field-reduced training
  view is a separately identified child artifact with the complete parent preserved.
- A finalized artifact is content-addressed, inspectable, threshold-gated, and traceable through
text preprocessing, training runs, evaluation, and checkpoints.
- The ordinary SpecForge path rejects partial, incompatible, or tampered first-party regeneration
outputs before training begins.
- Public code, examples, manifests, logs, and fixtures contain no campaign-specific secrets or
restricted data.
