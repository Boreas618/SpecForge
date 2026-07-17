# ADR 0001: Data regeneration is a text-trajectory subsystem

- **Status:** Accepted
- **Date:** 2026-07-17
- **Roadmap:** [`../roadmap/data-regeneration.md`](../roadmap/data-regeneration.md)

## Decision

SpecForge data regeneration lives under `specforge/data/regen`. Its canonical
payload is JSON-serializable text and structured trajectory data: message roles,
visible content, `reasoning_content`, tool calls/results, candidate relationships,
finish metadata, and provenance.

The subsystem never requests, captures, stores, transports, validates, or caches
hidden states, logits, embeddings, KV caches, or other tensor features. Any legacy
entry point that combines those concerns must be split. Raw token IDs may exist only
transiently to parse a sampled response or prove exact textual reconstruction; they
are not part of a finalized dataset payload.

The first artifact format is versioned JSONL plus a canonical JSON manifest. A
regeneration job is usable only after validation and atomic finalization. Recipes
contain named sources and generators followed by ordered, linear stages. Fan-out is
represented by explicit variants; arbitrary workflow cycles are not supported.

## Consequences

- `_temp/regen` is extracted by behavior, not copied as a model/dataset-specific
  package.
- Source, record schema, operation, generation backend, serving codec, tool policy,
  validator, storage, and executor are independent registered axes.
- Complete emitted reasoning/text/tool blocks are retained in the canonical artifact.
  Field-reduced training views are separately identified child artifacts.
- Tensor feature preparation and online feature transport remain outside this package
  and are not accepted as recipe options.
- Recipe, source, shard, and artifact identities are content-derived and secret-free.

