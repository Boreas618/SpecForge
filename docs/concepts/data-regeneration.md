# Data regeneration

SpecForge data regeneration produces immutable datasets of decoded text and
structured trajectories. It captures complete message content, reasoning
traces, tool calls/results, candidate relationships, and provenance. It never
captures hidden states, logits, embeddings, KV caches, or tensor features.

## Lifecycle

1. A typed recipe names pinned sources, generators, codecs, ordered stages,
   validators, and output sharding.
2. `plan` fingerprints sources, resolves registered capabilities, and writes an
   immutable task/shard plan without contacting a model.
3. `run` or independently launched `worker` processes static shards. Each
   attempt is an atomic immutable segment, so a crash before commit is safely
   repeated with the same request seed.
4. Attempt resolution creates deterministic data/reject/history parts.
5. Validators enforce source preservation, trajectory structure, coverage, and
   recipe thresholds.
6. `finalize` atomically publishes a content-addressed `manifest.json`.

Only a verified `FINALIZED` manifest is a normal SpecForge dataset input.

```bash
specforge data regen plan --config recipe.yaml
specforge data regen run --config recipe.yaml \
  --endpoint teacher=https://inference.example.invalid
specforge data regen validate --artifact ./artifact
specforge data regen finalize --artifact ./artifact
specforge data regen inspect --artifact ./artifact --json
```

Credentials are referenced by environment-variable name at worker runtime:

```bash
specforge data regen worker --plan ./artifact/run-plan.json --shard-index 0 \
  --endpoint teacher=https://inference.example.invalid \
  --api-key-env teacher=INFERENCE_API_KEY
```

Neither endpoint URLs nor credential values enter recipes, attempts, reports,
or manifests.

## Complete trajectories and derived views

Every emitted reasoning, visible text, and tool block remains in the canonical
record. A training consumer may derive a narrower view, but that view is a new
identified child dataset; it does not mutate or replace the captured parent.

Ordinary replay replaces each assistant at its original position. Later turns
therefore condition on regenerated earlier turns. For recorded tool replay,
the generated call name/order must match the source shape; source-authored tool
results are rebound to the regenerated call IDs. A mismatch is a policy reject.

Raw-token backends may carry sampled token IDs transiently. An exact codec must
parse the semantic assistant message, rebuild it with the declared renderer,
and prove token equality. Finalized records retain only the semantic trajectory
and evidence digest/count, never the raw stream or any model tensor.

## Validation profiles

A recipe names registered validation profiles; codec-required validators can
extend, but never disable, the baseline gates. `baseline` enforces text-only
trajectories, conversation structure, tool-result binding, and source
preservation. `specforge_loss_mask` additionally renders every finalized row
through the packaged serving-compatible chat template and the SpecForge
parser, and requires exactly one unambiguous supervised span per assistant
turn:

```yaml
validation:
  profiles: [baseline, specforge_loss_mask]
  config:
    specforge_loss_mask:
      chat_template: qwen
      tokenizer: Qwen/Qwen3-8B
      max_length: 8192
```

Rows that over-supervise (for example leaked assistant-header text inside a
source-authored message), render with no supervised tokens, or no longer fit
`max_length` are findings that quarantine the artifact. Renderer and
tokenizer imports are deferred until the first validated row, so planning and
inspection stay dependency-light.

## Tool execution, critique, and preference pairs

A stage with `tool_policy: execute` runs a generate → execute → observe loop
against a registered `ToolEnvironment` named in the stage config. The
environment declares its version, its tools, and its side-effect class; a
stage whose `side_effect_policy` is `forbid` (the default) refuses any
environment that is not side-effect free, at plan time. Every executed call
is appended as a tool message bound to its generated call id, and the stage
history records the environment identity plus argument/result digests.
Replaying recorded tool results (`tool_policy: replay`) is never presented
as live execution.

Candidate workflows compose from the same stages: `replay_assistants` with
`candidates: N` fans out variants, `critique` attaches a typed critique to
`payload["critiques"]`, `revise` replaces the final assistant turn while
preserving the replaced text in `payload["revisions"]`, and
`candidate_pair` joins two variants into a preference record — the chosen
trajectory stays the canonical conversation and the rejected trajectory
survives in `payload["rejected_conversations"]`. Preference artifacts are
consumed through `DatasetArtifact.iter_preference_records()`, which refuses
to coerce plain conversation rows.

Third-party components load only when a recipe names them under `plugins:`;
each plugin is an installed `specforge.data.regen` entry point whose
distribution name and version are recorded in the plan's registry snapshot,
so external code identity is part of provenance.

## Scale and operations

Shard ownership is pinned by the immutable plan under one of two strategies:
static contiguous ranges (default) or stable key-hash partitions
(`output.partitioning: hash`). Execution topology only changes scheduling —
local sequential, independently launched static workers, or leased workers
that claim shards from a shared lease store:

```bash
specforge data regen worker --plan ./artifact/run-plan.json --leased \
  --worker-id host-a --endpoint teacher=http://localhost:30000
```

A lease carries a fence: a worker that stalls past its lease can never
complete or extend a shard that another worker reclaimed, and the reclaiming
worker resumes the committed attempt journal exactly. After a crash or
restart, `specforge data regen reconcile --artifact <dir>` removes
uncommitted temporary files, returns expired leases to the pool, and reports
per-shard progress; committed attempt segments are immutable evidence and are
never rewritten (the finalizer's SQLite index is the compacted view).

Each generator's runtime endpoints form a pool with health tracking,
consecutive-failure circuit breakers, cool-down probes, and optional bounded
in-flight budgets. Endpoint placement never enters request identity, so pool
membership and failover change scheduling only. Structured operational logs
are redacted (no endpoint URLs, credentials, or private paths).

A finalized artifact can be published to object storage through the
`BlobStore` seam: payload files upload first and the content-addressed
manifest is written last with a conditional put, so a failed or racing
publisher can never expose a manifest that references missing or corrupt
data; readers verify every file digest on download and refuse manifest-less
prefixes.

Expensive validators (for example `specforge_loss_mask`) may declare a
deterministic sample (`sample_modulus`); the structural `baseline` profile
always scans every row. `benchmarks/bench_regen_scale.py` measures planning,
generation, resolution, and validation throughput and peak memory
separately.

## Extension points

Sources, record adapters, operations, backends, codecs, and validators are
independent registries with explicit versions and capabilities. Core planning,
execution, and finalization contain no dataset-name or model-family branches.
Private serving templates can remain external: an exact raw-token codec accepts
runtime renderer/parser/rebuilder callables while the recipe pins their public
format identity.
