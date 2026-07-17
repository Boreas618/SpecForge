# Data regeneration publication audit

This audit records the R0 extraction boundary for `_temp/regen`. It is an input
to implementation, not permission to publish campaign material wholesale.

## Field inventory

| Class | Public artifact fields | Runtime-only or excluded fields |
| --- | --- | --- |
| Source | namespaced/type-preserving ID, input position, adapter/version, content fingerprint, immutable revision, normalized text payload digest | local mount path, cache path, dataset credentials |
| Attempt | plan/shard/task identity, attempt number, stable status/category, bounded redacted diagnostic, semantic output envelopes | endpoint URL, authorization values, raw provider body, full exception dump |
| Generation | decoded assistant content, complete `reasoning_content`, typed tool calls, finish reason, bounded usage, model/backend/codec identity, request seed/digest | hidden states, logits, embeddings, KV caches, tensors, credentials |
| Raw exactness evidence | transient sampled token IDs; finalized digest/count and mismatch position/length | full token streams unless a separately protected debug policy is added |
| Shard | plan/recipe/task digest, ownership range, shard index, attempt-segment digest/count | worker hostname, private cluster topology |
| Validation | validator/version, counts, bounded finding examples, rates, thresholds, pass/fail gates | unrestricted source rows or server responses |

## Behavior disposition

| Prototype area | Disposition |
| --- | --- |
| Canonical JSON, typed IDs, deterministic seeds, contiguous ranges | Migrated into dependency-light public contracts |
| Immutable sidecars, locks, exact resume, attempt accounting | Migrated into the generic local executor/finalizer |
| Ordinary chat replay and split reasoning | Migrated behind registered operation/backend/codec contracts |
| Agentic raw-token equality | Migrated as a generic exact-rebuild codec; the renderer/parser is an injected, versioned adapter |
| Inkling template and local SGLang checkout path | Not copied; remains an external adapter boundary pending separate license/publication approval |
| Kimi split grammar | Migrated as a strict codec with synthetic public tests; no server address or checkpoint path is retained |
| Campaign shell commands, dates, restricted paths and endpoints | Excluded |
| Any tensor/feature capture mixed into regeneration | Excluded and must remain a separately named feature-preparation path |

## Review findings

- The nested prototype contains campaign-specific launch material and absolute
  model/tokenizer paths. It must not be packaged or used as a runtime import.
- `inkling_chat_template.jinja` has not been copied into the public package.
  Exact raw-token support accepts a renderer/parser/rebuilder supplied at
  runtime and pins its declared format identity in the recipe.
- Public fixtures are synthetic and contain no model outputs, private dataset
  rows, server addresses, credentials, or private filesystem paths.
- Local JSONL artifact provenance is path-independent: finalized metadata
  records the content fingerprint, not the operator's mount path.
- Diagnostics retain stable categories and bounded sanitized messages. Provider
  HTTP bodies are deliberately discarded.

## Publication gate

The public implementation and fixtures may be distributed under SpecForge's
repository license. The nested `_temp/regen` checkout, its template, campaign
README, and launch scripts remain non-package migration inputs until their own
provenance is approved. They are not dependencies of `specforge.data.regen`.
