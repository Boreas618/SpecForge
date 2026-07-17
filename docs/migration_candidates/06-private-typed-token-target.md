# PR report: private typed-token target integration

> **Restricted review required.** The source branch labels this integration NDA.
> Before opening or publishing a PR, confirm which model names, templates,
> configuration values, and test artifacts may leave the private environment.
> This report intentionally omits credentials, repository URLs, cluster addresses,
> and production topology details.

**Suggested internal feature-series title:** `feat(dspark): support a packaged typed-token target profile`

**Source material:** `7490759`, `0bcd37f`, and the final corrections through
`40be4f8`

**Migration status:** restricted selective port. The source range mixes reusable
data/model behavior with a legacy trainer, private serving-fork APIs, cluster
watchdogs, transient backend toggles, and committed credentials.

## PR description

### Background

The private target differs from ordinary Hugging Face causal-language models in
several independent ways:

- conversations use typed message headers and can emit multiple model blocks in
  one assistant turn;
- the tokenizer does not carry the training Jinja template, so the exact template
  must be packaged and installed by SpecForge;
- tool-result rendering depends on the message `name` field;
- sampling vocabulary and physical checkpoint vocabulary differ;
- target logits apply a muP width scale outside the bare LM-head matrix;
- the model type may require a private config registration or a guarded local
  config fallback;
- SGLang construction requires additional quantization, cache, and hybrid-state
  settings;
- the serving fork's request-extension API can differ from the pinned public
  SGLang API.

The source branch solved these inside the legacy DFlash target model and a large
cluster launcher. Current SpecForge has better extension points: packaged data
profiles, `TargetCapturePolicy`, `SGLangCaptureBackend`, typed configuration, the
DSpark strategy, and the server-capture adapter.

### Motivation

The integration should make target-specific assumptions explicit and testable
without weakening normal targets. Silent behavior is especially risky here:
dropped tool names change rendered tokens, an unscaled head changes softmax
distributions, a truncated physical vocabulary can reject valid mask IDs, and an
ignored serving flag can build an incompatible cache layout.

The migration also provides an opportunity to separate reusable primitives from
restricted assets. Generic parser and target-head improvements can be public;
the actual template, model profile, and launcher can remain in an approved private
package or plugin.

### Design and implementation

#### Reusable public primitives

1. Extend chat-template metadata to support multiple assistant terminators instead
   of hardcoding a target-name branch in `GeneralParser`.
2. Preserve `name` in `ThinkingParser` sanitization so tool authors survive
   rendering.
3. Add a packaged-template installer with content fingerprints and package-data
   support.
4. Land candidates 1–3 for physical vocabulary sizing, GQA validation, and
   template/data cache invalidation.
5. Add an explicit frozen-head output scale. Prefer a small wrapper that divides
   logits at forward time over mutating tied embedding weights.
6. Make optional SGLang engine settings typed and pass them to `ServerArgs`
   strictly; an explicitly supplied unsupported setting must fail.
7. Isolate request-extension API differences inside
   `SGLangCaptureBackend`, selected through feature detection or a version adapter.

#### Restricted target profile

The approved private layer supplies the exact Jinja template, typed-header
terminators, model/draft configuration, embedding/head checkpoint keys, logit
scale, stop-token set, capture-layer IDs, and serving-engine settings. It should
reuse `examples/disagg/run_disagg_dspark.py` or a thin private wrapper, not copy
the training loop.

#### Capture and evaluation

Reuse DSpark's server-capture schema for auxiliary and final target hidden states.
If the target needs capture behavior beyond DFlash, register a small target
capture policy rather than branching in the generic SGLang engine. Target-verified
evaluation should use candidate 5's verifier, with KV reuse disabled unless the
private cache state has an explicitly validated session implementation.

## Implementation plan (code walkthrough)

### Stack A: generic typed-turn parsing

1. Extend `specforge/data/template.py::ChatTemplate` with an optional immutable
   `assistant_terminators` list. Validate that it is non-empty when supplied.
2. Update `GeneralParser.set_assistant_pattern` to construct one escaped alternation
   from that list. The capture begins after the model-generation header and ends
   at the next user, tool, or system header or end of sequence.
3. Add `name` to `ThinkingParser.standard_keys` without changing sanitization for
   ordinary messages.
4. Add parser tests with multiple model blocks, intervening tool results, a
   mid-conversation system message, and a final assistant turn. Assert terminator
   tokens are excluded from supervision.

This public API should be target-neutral. The exact typed tokens belong in the
restricted template profile.

### Stack B: packaged templates and cache identity

1. Add a package-data directory under `specforge/data/templates/` and a registry
   mapping approved profile names to resources.
2. Implement `install_packaged_chat_template(tokenizer, profile)` using
   `importlib.resources`, not paths relative to a source checkout.
3. Expose a template-content SHA-256 fingerprint and include it in candidate 3's
   processed-dataset cache payload.
4. Update packaging metadata so wheels and source distributions include the Jinja
   files.
5. Add tests that install from an actual built package/resource context and prove
   a template edit changes the cache key.

If the template cannot be published, place the resource and registry entry in a
private plugin while keeping the installer interface public.

### Stack C: config loading, physical vocabulary, and logit scale

1. Reuse candidate 1's physical-vocabulary resolver in
   `TargetEmbeddingsAndHead`.
2. Prefer normal `AutoConfig` registration from the private model package. If a
   raw JSON fallback is still needed, permit it only for an existing local
   `config.json`, expose it as an explicit option, and validate required scalar
   fields. Do not silently catch arbitrary remote or parsing failures.
3. Add a frozen LM-head wrapper that computes `linear(hidden) / logit_scale`.
   This preserves tied embedding weights and makes the scale visible in module
   metadata. Reject non-positive scales.
4. Test that wrapper logits and probabilities match a reference target at both
   greedy and non-zero temperatures. Test tied and untied heads separately.

The target profile supplies the actual scale and keys. The generic loader must
not infer a private model type by name.

### Stack D: current DSpark config and GQA16

1. Translate the final source config into the current layout described in
   candidate 2. Do not include the intermediate GQA8 config.
2. Load it through `AutoConfig` and the draft registry, then validate Q/K/V
   shapes, Markov/confidence heads, mask-token range, stop tokens, and capture
   layers.
3. Store restricted config values outside the public tree unless publication is
   approved. A public fixture should use tiny synthetic dimensions and placeholder
   token IDs.

### Stack E: strict SGLang target construction

1. Extend `SGLangBackendArgs` and/or the typed model configuration with only the
   optional engine fields the current pinned and private SGLang versions can
   validate.
2. Pass explicitly set fields through `SGLangTargetEngine` to
   `SGLangCaptureBackend.build`. Current `ServerArgs` construction already fails
   on unknown kwargs; preserve that fail-loud behavior.
3. Add one request-range compatibility helper inside
   `specforge/inference/target_engine/sglang_backend/capture.py`. Use the newer
   range setter when available and the pinned public fields otherwise. Unit-test
   both with fake request objects.
4. Put any private model-runner hooks behind a target capture policy or private
   adapter. Do not fork the entire generic capture backend.

### Stack F: data preparation and launch

1. Build a restricted preparation command that accepts input globs, validates
   success rows, deduplicates identity/content with explicit counters, installs
   the packaged template, and enforces a minimum loss-mask retention rate.
2. Reuse candidate 4 for deterministic sampling and candidate 3 for cache keys.
3. Keep the launch wrapper declarative: model path, private SGLang package path,
   output/data paths, topology, and credentials come from configuration or the
   environment. No default secret values or fixed network addresses are allowed.
4. Use the current DSpark producer/consumer and `CheckpointManager`. Watchdog,
   checkpoint sync, and diagnostics are separate operational tools.

### Stack G: validation gates

Run gates in increasing privilege/cost order:

- CPU template render and exact loss-mask spans;
- CPU config/head tests with a synthetic checkpoint;
- meta-device draft construction and GQA16 shape checks;
- private-fork capture parity probe on one node;
- two-step DSpark consumer smoke with finite CE/L1/confidence losses;
- checkpoint save/resume through the current manager;
- target-verified acceptance evaluation with the approved stop-token set;
- multi-node smoke only after all single-node gates pass.

Every private gate should emit a redacted summary suitable for attaching to a PR.

## Security and publication requirements

- Rotate and remove every credential present in the source branch before using it
  as a remote or patch source.
- Add a secret scan covering the restricted files and generated reports.
- Do not publish private model/dataset locations, access tokens, cluster IPs, or
  proprietary template contents without explicit approval.
- Keep logs from dumping full `ServerArgs` values when they may contain private
  paths or endpoints.

## Scope and non-goals

- Intermediate MoE backend and learning-rate toggles are not part of this port.
- Plain radix-cache reuse is disabled unless hybrid-state correctness is proven.
- The feature does not make a private SGLang fork a dependency of ordinary
  SpecForge installations.

## Acceptance criteria

- Typed conversations render and mask exactly like the approved serving path,
  including tool names and multi-block assistant turns.
- Frozen-head logits match the target distribution, with physical vocabulary and
  scaling handled explicitly.
- The current DSpark server-capture/training/checkpoint path runs without a
  duplicate trainer.
- Public artifacts contain no restricted values or credentials.
