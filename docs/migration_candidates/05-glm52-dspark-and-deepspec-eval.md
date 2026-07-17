# PR report: GLM-5.2 DSpark training and DeepSpec acceptance evaluation

**Suggested feature-series title:** `feat(dspark): add GLM-5.2 training and target-verified acceptance evaluation`

**Source material:** `e5755a7` through `114b1f2`; the thinking-mask correction
in `1dac1b9` and DeepSpec-parity work in `5636b6f` are mandatory parts of the
final behavior.

**Migration status:** feature-level redesign. The source implementation uses a
removed in-process `scripts/train_dspark.py`; current `main` exposes DSpark
through `DSparkTrainStrategy` and the disaggregated server-capture path.

## PR description

### Background

The source branch added a dense five-layer, Qwen-shaped DSpark drafter for a
large GLM-5.2 target. It included target-specific conversation rendering,
thinking-aware loss masks, online hidden-state teaching, a memory-bounded DSpark
objective, and a true propose/verify/accept evaluator modeled on DeepSpec.

Current `main` already has a newer DSpark model and loss implementation:

- `specforge/modeling/draft/dspark.py::DSparkDraftModel` owns Markov and
  confidence heads;
- `specforge/core/dflash.py::OnlineDSparkModel` owns the block-parallel objective;
- `specforge/training/strategies/base.py::DSparkTrainStrategy` owns the training
  contract;
- `specforge/inference/adapters/server_capture.py` transports both auxiliary and
  target-final hidden states;
- `examples/disagg/run_disagg_dspark.py` is the supported launcher.

The migration must extend those seams. Restoring the old trainer would create a
second checkpoint, optimizer, FSDP, and evaluation lifecycle.

### Motivation

This candidate provides two capabilities that `main` does not currently have:

1. a validated GLM-5.2 data/configuration path for current DSpark training;
2. an end-to-end acceptance-length evaluator that runs the draft proposal and
   target verification loop, rather than inferring acceptance from teacher-forced
   training accuracy.

The data work also fixes a severe silent-failure mode. GLM can render a
thinking-off response as an empty thinking block followed by an answer, while a
thinking-on response contains generated reasoning inside that block. Matching a
literal empty block as the assistant header can zero-mask every thinking-on row.
The final source behavior supervises generated reasoning, the closing thinking
token, and the answer, while excluding template scaffolding.

### Design and implementation

Deliver this candidate as stacked PRs with four explicit seams.

#### Data and masking seam

Register a `glm-5.2` chat-template profile whose stable assistant prefix is the
open assistant/thinking scaffold. Add a GLM assistant-pattern implementation that
optionally skips the immediate close token used for thinking-off rows, then
captures through the next user delimiter. Rendering mode must be caller-controlled;
the parser must not silently force thinking off.

Dataset preparation must report total rows, rows with non-zero supervision, and
retention percentage. A configurable minimum-retention gate should fail before
training when a template change unexpectedly removes a large fraction of the
corpus. Candidate 3's template-content and source-content fingerprints must be
part of the processed cache key.

#### Model and training seam

Keep the draft Qwen-shaped and express all DSpark settings in `dflash_config`.
Use the existing server-capture schema to deliver selected target-layer states and
the final target hidden state. `DSparkTrainStrategy` and `OnlineDSparkModel` remain
the only training implementation.

For large vocabularies, add an optional objective chunk size to
`OnlineDSparkModel`. Compute vocabulary projections and CE/L1/confidence terms in
anchor-block chunks, use activation checkpointing where it lowers peak memory,
and accumulate numerators/denominators before one final division. The chunked and
unchunked paths must be gradient-equivalent. Current global gradient clipping and
checkpoint management remain unchanged.

#### Evaluation seam

Add a target-verified speculative-decoding evaluator separate from the existing
`Evaluator`. The existing evaluator aggregates training-strategy metrics;
the new evaluator owns an actual loop:

1. prefill the prompt through the target and obtain auxiliary/final hidden states;
2. have DSpark propose one block, applying its Markov head autoregressively;
3. verify the block with the target;
4. perform DeepSpec rejection sampling using the exact draft and target
   distributions;
5. commit the accepted prefix plus residual/bonus token and repeat.

The correctness implementation can re-prefill first. KV/radix reuse should be a
capability-gated optimization layered on top of the same evaluator, never a
requirement for correct metrics.

#### Product and recipe seam

Extend typed configuration to recognize `strategy="dspark"` and add validated
DSpark options rather than relying on environment-only switches. Keep the example
topology in `examples/disagg`; it must use current producer/consumer roles and
must not encode fixed hosts, credentials, or output paths.

## Implementation plan (code walkthrough)

### Stack A: GLM rendering, masks, and corpus gates

1. Extend `specforge/data/parse.py` with a GLM assistant-pattern builder. If a
   dedicated `GLMParser` is needed for tokenizer kwargs, make the thinking mode
   an explicit input rather than a hardcoded default.
2. Register the parser in `specforge/data/preprocessing.py` and the template in
   `specforge/data/template.py`.
3. Add `scripts/prepare_glm52_dspark_data.py` only for schema normalization and
   deterministic train/eval splitting. Dataset locations, mix fractions, limits,
   and seed must be CLI arguments.
4. Add a reusable retention summary/gate after tokenization. Log counts before
   filtering and fail when retention falls below the configured threshold.
5. Use the shared cache-key API from candidate 3, including a template schema
   version so mask changes cannot hit stale tensors.

CPU tests should cover thinking-on, thinking-off, multi-turn, final-turn, and
truncated conversations. Assert exact supervised token spans, including that an
immediate scaffold close is excluded and a generated close is included.

### Stack B: current DSpark configuration and capture

1. Add `configs/glm-5.2-dspark.json` in the current format. Put
   `projector_type`, Markov, confidence, mask-token, and capture-layer settings
   under `dflash_config`.
2. Extend the typed config in `specforge/config/schema.py` so DSpark is a valid
   strategy and its CE/L1/confidence weights, anchor count, block size, and
   objective chunk size are validated.
3. Reuse `specforge/inference/adapters/server_capture.py`'s DSpark schema. Add a
   capture-contract test proving selected auxiliary layers and
   `target_last_hidden_states` survive batching and padding.
4. Update `examples/disagg/_dflash_family_disagg.py` only where model-agnostic
   builder support is missing. Do not add a parallel GLM trainer.

The CPU gate loads the config through the draft registry and runs a small
forward/backward pass. The server gate captures a tiny substitute target before
using the real target on the intended hardware.

### Stack C: memory-bounded objective and DeepSpec parity

1. Add `objective_chunk_blocks` to `OnlineDSparkModel` with `0` meaning the
   existing unchunked path.
2. Refactor the objective into a pure per-chunk helper. Flatten vocabulary GEMMs
   to two dimensions before projection, then reshape outputs back to block form.
3. Sum CE, L1, confidence, accuracy, and probabilistic-acceptance numerators and
   denominators across chunks. Divide once after all chunks so chunk boundaries
   cannot change weighting.
4. Preserve the current anchor sampler's first-target-valid and contiguous-prefix
   semantics. Add a reference test against the intended DeepSpec algorithm.
5. Add gradient-equivalence tests for every trainable component and several chunk
   sizes. Require close equality in float32 and documented tolerance in bf16.

Do not port the source's duplicate global-clip logic; current `BF16Optimizer` and
training backend already own global norm reduction.

### Stack D: target-verified evaluator

1. Add `specforge/eval/deepspec.py` with sampling primitives, proposal/verification
   records, and a `DeepSpecEvaluator` API. Keep RNG ownership explicit and seed
   per sample.
2. Define a narrow target-verifier protocol instead of importing SGLang internals
   into the evaluator. The first implementation may call full-prefix capture;
   an optional session API can expose incremental verification later.
3. Return raw sums/counts for accepted lengths and per-position proposals, then
   reduce them across DP ranks before computing means.
4. Add stop-token handling from the draft config and guarantee evaluator cleanup
   restores model train/eval modes even after failure.
5. Provide a standalone CLI wrapper under `scripts/` and an in-loop callback for
   `TrainerController`; both must call the same evaluator.

Test rejection sampling with hand-constructed distributions, greedy equivalence
at temperature zero, residual sampling after rejection, full-accept bonus tokens,
stop-token truncation, deterministic seeding, and distributed aggregation. Then
compare cached and non-cached verification on a small target.

### Stack E: examples and GPU gates

1. Add a sanitized GLM example using current disaggregated producer/consumer
   commands and environment-provided model/data paths.
2. Document two topology profiles only when both are tested; otherwise publish the
   validated profile and keep experimental modes out of the default script.
3. Add staged gates: tiny model CPU, small GPU model end-to-end, real target smoke,
   then a short convergence and acceptance-length run.
4. Record peak memory, tokens/second, finite-gradient status, checkpoint
   save/resume, and target-verified acceptance length.

## Risks and non-goals

- A simulated acceptance metric from teacher-forced accuracy is not a replacement
  for the target-verified evaluator.
- KV reuse is an optimization and must not delay the correctness path.
- Source defaults that were toggled during debugging are not migrated; only the
  final validated recipe is documented.
- This series does not add DeepSeek-V4 draft architectures or private-target
  serving flags.

## Acceptance criteria

- Thinking-on and thinking-off data retain the intended supervised spans, with a
  pre-training retention gate.
- GLM DSpark trains through the current strategy/server-capture path and resumes
  through `CheckpointManager`.
- Chunked and unchunked objectives are gradient-equivalent within the documented
  dtype tolerance.
- The evaluator implements target-verified rejection sampling and reports stable
  per-position and mean accepted-length metrics.
