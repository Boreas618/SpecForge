# PR report: deterministic sampling and deduplication for regenerated JSONL

**Suggested PR title:** `feat(data): add reproducible sampler for regenerated JSONL corpora`

**Source material:** `0155ee1`, with the source-preservation correction from
`71b1b41`

**Migration status:** extract the final standalone utility. The source commit also
modifies a target-specific preparation script that does not exist on `main`.

## PR description

### Background

Regeneration jobs can append multiple attempts for the same source ID. They can
also emit files without a `source` field, while other shards identify rows by a
`(source, id)` pair. Naively sampling lines can select multiple rerolls of one
prompt, and unconditionally stamping a source can overwrite valid provenance
already present in richer rows.

The source utility established three useful semantics:

1. keep the first occurrence of each non-null ID;
2. sample uniformly from the distinct-ID pool using an explicit seed;
3. preserve input order in the output and only add `source` when it is missing.

### Motivation

Continual-training and mixed-corpus recipes need a sample that can be rebuilt
independently on each node with identical bytes. The policy must be explicit so
rerolls and malformed rows do not produce topology-dependent corpora. A standalone
script also lets dataset assembly remain separate from model-specific launchers.

### Design and implementation

Add `scripts/sample_regen_jsonl.py` as a deterministic JSONL transform. Its CLI
accepts input/output paths, sample size, seed, optional source label, duplicate-ID
policy, and malformed-row policy.

Default behavior should match the final source implementation:

- ignore blank lines;
- report malformed JSON and skip it only when the user explicitly requests
  permissive behavior; otherwise fail with a line number;
- for non-null IDs, keep the first row and count later occurrences;
- treat rows without an ID as distinct rows;
- sample indices with `random.Random(seed)` from the deduplicated pool;
- sort selected indices before writing so source order is preserved;
- add `source` only when `--source` is provided and the row lacks a non-empty
  source;
- write UTF-8 JSON with stable separators and a trailing newline.

Emit a machine-readable summary containing rows read, blank/malformed rows,
duplicate IDs dropped, distinct pool size, sampled count, seed, and output path.
Write through a temporary file followed by `os.replace` so an interrupted run
does not leave a partial output that appears complete.

## Implementation plan (code walkthrough)

### 1. Add a reusable transformation function

Keep CLI parsing thin. Implement a function such as:

```python
sample_regenerated_jsonl(
    input_path,
    output_path,
    *,
    sample_size,
    seed,
    source=None,
    malformed="error",
) -> SamplingSummary
```

Use `pathlib.Path` and a frozen dataclass for the summary. This lets unit tests
exercise behavior without spawning a subprocess and lets future preparation
commands call the same implementation.

### 2. Define identity and mutation rules explicitly

Maintain a `seen_ids` set only for non-null IDs. Preserve the first row object for
each ID. Before writing, shallow-copy a selected row if a source must be added so
the in-memory input representation is not unexpectedly mutated.

The utility should not content-deduplicate conversations; identity-level and
content-level deduplication answer different questions. A target-specific corpus
builder can add canonical conversation hashing after validation gates.

### 3. Make output reproducible

Use one local RNG instance and never depend on the global random state. Sort the
sampled indices before output. Serialize with fixed JSON options. The same input
bytes and arguments must produce the same output bytes across repeated runs.

Include an optional `--require-num` flag if recipes need an exact row count. By
default, sampling fewer rows than requested should be reported clearly rather
than hidden; whether it is fatal must be explicit.

### 4. Integrate with regeneration tooling

Document the utility next to `scripts/regenerate_train_data.py` and
`scripts/validate_regenerated_data.py`. Do not change the validator's existing
policy that duplicate IDs are allowed: validation reports what the generator
produced, while this sampler intentionally chooses keep-first semantics for a
derived training corpus.

An example should show validation first, then deterministic sampling, then
content-addressed cache preparation from candidate 3.

### 5. Add CPU tests

Create `tests/test_scripts/test_sample_regen_jsonl.py` covering:

- duplicate IDs keep the first completion;
- null/missing IDs remain independently sampleable;
- existing source values survive `--source`;
- missing source values are stamped;
- the same seed produces byte-identical output;
- different seeds select different rows for a sufficiently large fixture;
- output order follows input order rather than RNG draw order;
- malformed JSON fails by default and is counted in permissive mode;
- requesting more rows than available follows the documented exact/non-exact
  policy;
- interrupted writes cannot replace a previously complete output.

### 6. Operational validation

Run the script twice against a representative regeneration shard on two nodes and
compare SHA-256 digests. Record the summary JSON in the training-run metadata so
the sampled corpus can be reproduced later.

## Scope and non-goals

- The script does not download datasets or launch regeneration servers.
- It does not choose a domain-specific source label.
- It does not merge shards or decide first-source precedence; callers control
  input order and can concatenate validated samples explicitly.
- It does not load or expose credentials.

## Acceptance criteria

- Sampling is deterministic and byte-reproducible for fixed input and arguments.
- Duplicate-ID and source-preservation behavior is covered by CPU tests.
- The tool can be used independently of any private target or cluster launcher.
