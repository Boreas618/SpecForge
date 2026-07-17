# PR report: content-addressed processed-dataset cache keys

**Suggested PR title:** `fix(data): invalidate tokenized caches when input bytes change`

**Source material:** `1f236bc`

**Migration status:** behavioral port across active callers. The source helper
applies cleanly, but its two call sites belong to a removed trainer and launcher.

## PR description

### Background

The processed-dataset cache is keyed by strings assembled in training scripts.
Those strings generally include the dataset path, maximum length, chat-template
name, and target/tokenizer path. They do not include the bytes of a local JSONL
file. Rebuilding a corpus in place therefore reuses tokenized tensors from the
previous contents.

This is particularly dangerous for regenerated corpora, which are often extended
or replaced at a stable path. The cache hit looks healthy and training proceeds,
but it silently sees stale samples and stale loss masks.

### Motivation

Cache identity should describe the artifact being cached, not just where the
artifact was found. A streamed file digest makes in-place corpus changes visible
without loading large datasets into memory. Centralizing key construction also
eliminates subtly different ad hoc strings across EAGLE3, P-EAGLE, DFlash,
Domino, and hidden-state preparation.

### Design and implementation

Introduce a small cache-key module under `specforge/data` rather than adding more
string concatenation to individual scripts. It should provide:

- `stream_file_sha256(path)`: hash a file incrementally in fixed-size blocks;
- `dataset_source_fingerprint(source)`: return a content fingerprint for local
  files and a stable identifier for non-file dataset references;
- `processed_dataset_cache_key(**fields)`: serialize a versioned field mapping as
  canonical JSON and hash that payload.

The canonical payload must include the local file digest plus every preprocessing
input that changes token IDs or masks: maximum length, tokenizer/target identity,
chat-template name, template-content fingerprint where available, preformatted
mode, train-only-last-turn mode, VLM mode, and an explicit mask/cache schema
version.

Use SHA-256 for source and key digests. A short prefix is acceptable for a file
name, but logs and metadata should retain the full digest so collisions can be
diagnosed. Hashing must be read-only and streamed; no corpus is copied into the
cache directory.

## Implementation plan (code walkthrough)

### 1. Add the shared cache-key API

Create `specforge/data/cache.py` with pure, dependency-light helpers. Use
`json.dumps(..., sort_keys=True, separators=(",", ":"))` for an injective
payload rather than delimiter-joined strings. Include a constant such as
`PROCESSED_DATASET_CACHE_VERSION = 2` so future mask changes can invalidate all
old entries intentionally.

For local files:

- resolve the path for logging but hash the file bytes, not path metadata;
- read in bounded chunks (for example 4 MiB);
- raise a contextual error if the file changes or disappears while hashing.

For Hugging Face dataset identifiers, preserve the source identifier and add a
revision/config field when the caller has one. Do not pretend a remote name is a
content hash.

### 2. Migrate active callers

Replace the local cache-string construction in:

- `scripts/train_eagle3.py::build_dataloaders`;
- `scripts/train_peagle.py::build_dataloaders`;
- `scripts/train_dflash.py::build_dataloader`;
- `scripts/train_domino.py::build_dataloader`;
- `scripts/prepare_hidden_states.py` before `build_eagle3_dataset`;
- any preparation/warm-cache command added by the GLM or private-target reports.

The vocabulary-mapping cache must incorporate the processed-dataset key plus its
own vocabulary dimensions. It must not reuse the old path-only base string.

Train and eval inputs need separate keys. Where a script accepts an HF dataset ID
instead of a local file, pass the source type explicitly into the helper.

### 3. Make warm and train paths identical

If an example script warms the cache before launch, have it call the same Python
helper as the trainer. Do not duplicate the canonical field list in shell or an
inline Python snippet. A public function that returns both the key and its
canonical payload makes mismatches easy to log and test.

### 4. Log cache provenance

At cache build/load time, log:

- the short cache key;
- source path or dataset ID;
- source digest prefix;
- cache schema version;
- whether the operation was a hit or build.

Do not log dataset contents or credentials embedded in remote URLs.

### 5. Add regression tests

Add `tests/test_data/test_cache_key.py` and extend
`tests/test_scripts/test_train_eagle3_cache_key.py`:

- identical bytes at different paths yield the same source fingerprint;
- changing bytes at the same path changes the key;
- changing each semantic field changes the key;
- dictionary insertion order does not change the key;
- delimiter-like field values cannot collide;
- large files are read in multiple bounded chunks;
- train and warm paths compute the same key;
- vocabulary-mapping keys change when either the data digest or vocabulary size
  changes.

### 6. Rollout and compatibility

This intentionally causes a one-time cache miss because the key schema changes.
Do not delete old cache entries automatically; operators can remove them after a
successful run. Document the new key version in the release note and include the
payload fields in debug logs.

## Scope and non-goals

- The PR does not prove that two nodes have identical files; distributed
  fingerprint agreement can be a follow-up or part of candidate 7.
- It does not content-address model checkpoints.
- It does not change Hugging Face Dataset's own cache semantics.

## Acceptance criteria

- Replacing a JSONL file at the same path cannot hit its previous processed cache.
- Every active local-data training/preparation path uses the shared key builder.
- Warm-cache and training invocations produce byte-identical keys for the same
  inputs.
