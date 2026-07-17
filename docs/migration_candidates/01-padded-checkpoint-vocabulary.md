# PR report: distinguish checkpoint and generatable vocabulary sizes

**Suggested PR title:** `fix(target): model checkpoint and generatable vocabulary geometry explicitly`

**Source material:** `f43c983`

**Investigation date:** 2026-07-17

**Migration status:** selective semantic port. The source change is necessary for
an SGLang-registered Inkling config, but it is a no-op for the other surveyed
model families and is not sufficient by itself to enable Inkling in SpecForge.

## Impact verdict

The source commit changes allocation only when the selected config object exposes
`padded_vocab_size` with a value larger than `vocab_size`. Of the public and local
checkpoints surveyed, that condition occurs only in the SGLang representation of
Thinking Machines' Inkling:

- the SGLang config reports 200,058 logical rows and 201,024 padded checkpoint
  rows;
- the published BF16 and NVFP4 embedding and unembedding tensors both contain
  201,024 rows;
- the source patch therefore adds 966 rows to each frozen matrix and changes an
  immediate shape mismatch into a successful load.

Qwen 3/3.5/3.6, DeepSeek V3/V4, GLM-5.2, Kimi K3, and the additional models in
the survey already report a `vocab_size` equal to their checkpoint row count.
For them, the proposed resolver returns exactly the value used today: module
shapes, weights, logits, memory use, and checkpoints are unchanged.

The source diff also changes direct `pad_token_id` access to
`getattr(..., "pad_token_id", None)`. Registered Transformers configs in the
survey already expose that attribute, often with value `None`, so this is also a
no-op for them. It prevents an unrelated `AttributeError` for a raw or minimal
config object that omits the field.

The Inkling result comes with an important qualification. Its 201,024-row
matrices are a physical storage domain, while only the first 200,058 rows are
generatable. The source patch fixes allocation but does not preserve that output
boundary. A safe port must model both sizes instead of replacing one ambiguous
`vocab_size` with another.

## Investigation scope and method

The audit covered the source diff, every current `TargetEmbeddingsAndHead` call
site, the separate EAGLE3 `TargetHead`, registered `AutoConfig` objects where
available, raw model configs, weight indexes, and safetensors headers.

Tensor shapes were read from safetensors metadata without downloading public
weight bodies, following Hugging Face's documented
[HTTP range metadata method](https://huggingface.co/docs/safetensors/en/metadata_parsing).
Public model revisions are pinned in the links below so later config changes do
not silently rewrite the conclusion. Kimi K3 was inspected from the complete
local checkpoint on this machine because its Hub repository was not accessible
without authentication during the audit.

“No-op” in the tables means the vocabulary-size resolver does not change that
checkpoint. It does not imply that every model-specific config class, tensor key,
quantization, or target backend is already supported by SpecForge.

## Model compatibility matrix

| Model family and audited checkpoint | Config view used by the loader | Physical embedding/head rows | Direct effect of `f43c983` | Result |
|---|---:|---:|---|---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json), [Qwen3-Next-80B-A3B](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7/config.json), [Qwen3-Coder-480B-A35B](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/9d90cf8fca1bf7b7acca42d3fc9ae694a2194069/config.json) | `vocab_size=151936` | 151,936 | none | Safe no-op |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json) | `text_config.vocab_size=248320` | 248,320, tied | none | Safe no-op |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json) | `text_config.vocab_size=248320` | 248,320 | none | Safe no-op |
| [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json) | `text_config.vocab_size=248320` | 248,320 | none | Safe no-op |
| [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) | `vocab_size=129280` | 129,280 | none | Safe no-op |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) | `vocab_size=129280` | 129,280 | none | Safe no-op; keys still need adaptation |
| [GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/config.json) | `vocab_size=154880` | 154,880 | none | Safe no-op |
| Kimi K3 local snapshot `79a424f` | `text_config.vocab_size=163840` | 163,840 | none | Safe no-op; keys and remote code are separate concerns |
| [Inkling BF16](https://huggingface.co/thinkingmachines/Inkling/blob/86b4d430ab871652a707666b89203a866888c5e5/config.json) and [NVFP4](https://huggingface.co/thinkingmachines/Inkling-NVFP4/blob/d11961f515e883e37796edb9dd6ec1bf0e0e8212/config.json) | raw/native: `vocab_size=201024`, `unpadded_vocab_size=200058` | 201,024 | none in raw/native config | Allocation already correct once the config can load |
| Inkling through the audited SGLang config class | `vocab_size=200058`, `padded_vocab_size=201024` | 201,024 | **+966 rows per matrix** | Directly fixed by the source patch |
| [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json) | `vocab_size=201088` | 201,088 | none | Safe no-op |
| [Phi-4](https://huggingface.co/microsoft/phi-4/blob/2db69c1c3e91a05d2c64a3185acfbaf36f744e25/config.json) | `vocab_size=100352` | 100,352 | none | Safe no-op |
| [LongCat-Flash](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat-FP8/blob/a373e08a1c4897ab12cbac2a540e497502d55118/config.json), [Ling-flash-2.0](https://huggingface.co/inclusionAI/Ling-flash-2.0/blob/18ca64a019b553be57bab50af3207fb2f3675edc/config.json), [LFM2.5-1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9/config.json) | 131,072 / 157,184 / 65,536 | same as config | none | Safe no-op |

### Qwen 3, 3.5, and 3.6

Three Qwen3 layouts were checked: dense Qwen3-8B, Qwen3-Next-80B-A3B, and
Qwen3-Coder-480B-A35B. All use 151,936 config rows and 151,936-row physical
embedding/head tensors. Dense and MoE layout differences do not affect this PR.

Qwen3.5 and Qwen3.6 use a multimodal wrapper with the language geometry under
`text_config`; the current target utility already selects that object. The
audited 4B, 35B-A3B, and 27B checkpoints all use 248,320 rows with no
`padded_vocab_size` or `unpadded_vocab_size` attribute in their registered
Transformers configs.

Qwen3.5-4B is the useful tied-weight regression case. Its checkpoint contains
the embedding tensor but no independent LM-head tensor, and
`tie_word_embeddings=true`. The proposed resolver must preserve the existing
post-load aliasing behavior. Qwen3.5-35B-A3B and Qwen3.6-27B are untied and have
both 248,320-row matrices.

The Qwen3.5/3.6 embedding key is
`model.language_model.embed_tokens.weight`, not the default
`model.embed_tokens.weight`. Existing recipes already pass the alternate key;
the vocabulary PR neither fixes nor regresses this requirement.

### DeepSeek V3 and V4

DeepSeek-V3 exposes 129,280 rows in its config and uses 129,280-row primary
embedding and head tensors. Its checkpoint also contains an MTP-layer embedding;
that tensor is not the frozen base embedding requested by the current loaders
and should not participate in vocabulary resolution.

DeepSeek-V4-Flash also uses 129,280 config and checkpoint rows, so the proposed
change is a no-op. The native release names the relevant tensors `embed.weight`
and `head.weight` instead of SpecForge's defaults. Explicit CLI keys or a
model-key registry are still required; padded allocation does not make V4 load
automatically.

### GLM-5.2

The cached and published GLM-5.2-FP8 config reports 154,880 rows, and both
`model.embed_tokens.weight` and `lm_head.weight` have shape
`[154880, 6144]`. GLM reserves tokenizer-unreachable rows inside this declared
space, but its registered config does not rewrite `vocab_size` to a smaller
sampling domain. The source resolver therefore returns 154,880 before and after
the PR.

This distinction matters for terminology: a model can contain reserved or
tokenizer-unreachable rows without exposing `padded_vocab_size`. Such a model is
not directly affected by this commit.

### Kimi K3

The local Kimi K3 checkpoint at `/scratch/models/Kimi-K3` was inspected at Hub
cache ref `79a424f302e6ed63cd82211b0e99918cdb6de8df`. Its remote config class loads a
nested text config with `vocab_size=163840`, `hidden_size=7168`, no padded or
unpadded size, and untied weights. Both local tensors are exactly
`[163840, 7168]`.

The source PR is therefore a strict no-op for this snapshot. Kimi still needs
its explicit keys:

- `language_model.model.embed_tokens.weight`;
- `language_model.lm_head.weight`.

It also requires `trust_remote_code` for the local `KimiK3Config`. Neither
requirement is changed by the vocabulary resolver.

### Thinking Machines Inkling

Inkling is the one directly affected architecture, but only under one config
representation. The published BF16 and NVFP4 releases contain separate
`model.llm.embed.weight` and `model.llm.unembed.weight` tensors, each shaped
`[201024, 6144]`. Their raw text config declares:

- `vocab_size=201024` for physical module construction;
- `unpadded_vocab_size=200058` for the output/sampling domain;
- no `padded_vocab_size` field.

The audited SGLang config normalizes the same JSON differently. It retains
201,024 as `padded_vocab_size` and rewrites `vocab_size` to 200,058. That is the
exact runtime path fixed by `f43c983`: the current SpecForge utility allocates
`[200058, 6144]`, then fails while copying a `[201024, 6144]` checkpoint tensor.

There are four distinct loading outcomes:

| Config path | Values visible to SpecForge | Effect of the source patch |
|---|---|---|
| Repo-pinned Transformers 5.8.1 | `inkling_mm_model` is unknown | Fails before allocation; no benefit |
| Current Transformers main implementation | physical 201,024, unpadded 200,058 | No allocation change; already physical |
| SGLang config registered before `AutoConfig` | logical 200,058, padded 201,024 | Fixes both embedding and head allocation |
| Raw-JSON fallback proposed in candidate 6 | `vocab_size=201024` | No allocation change; already physical |

The public Inkling support was contributed to Transformers on 2026-07-15, after
SpecForge's pinned version. Even the newer Transformers 5.12.1 installed on this
machine did not yet recognize the published model type during the audit. This PR
must not be described as standalone Inkling enablement.

The released Transformers implementation also provides the reference output
semantics: it allocates the full 201,024-row head, applies the model's muP logit
scale, and then slices logits to 200,058 before loss or generation. The source
SpecForge patch allocates the full head but returns all 201,024 logits. That can
put padded mask rows into softmax, argmax, distribution-distillation, or
speculative-draft outputs. Correct loading and correct output semantics are
separate requirements.

### Additional representative models

The survey also read public config and tensor metadata for gpt-oss-20b, Phi-4,
LongCat-Flash-Chat-FP8, Ling-flash-2.0, and LFM2.5-1.2B-Instruct. Each physical
row count equals the registered `vocab_size`, and none of their inspected config
objects exposes a distinct padded or unpadded size. They are unchanged by this
PR.

Llama 3.3, Llama 4, and Gemma 3 were not included in the tensor-header evidence
because their official Hub checkpoints required authentication in the audit
environment. The generic test matrix should still include synthetic config
fixtures for their ordinary `vocab_size` behavior, but the report does not claim
unverified checkpoint shapes.

## Impact on current SpecForge call sites

| Path | Uses `TargetEmbeddingsAndHead`? | Impact |
|---|---|---|
| `scripts/train_dflash.py` | yes | SGLang Inkling starts loading; frozen projection width changes from 200,058 to 201,024 |
| `scripts/train_domino.py` | yes | Same allocation and output-width effect as DFlash |
| `examples/disagg/_dflash_family_disagg.py` | yes | DFlash, Domino, and DSpark consumers inherit the change |
| `examples/disagg/run_disagg_dspark.py` | indirectly | Highest correctness risk: Markov bias, draft softmax, target softmax, and confidence labels must use the same explicit output domain |
| `scripts/train_eagle3.py` and `specforge/cli.py` offline path | no | Uses `TargetHead`; the source commit does not fix its physical-head allocation |
| Full HF/SGLang target engines | no | Their own model loaders and logits processors remain unchanged |

The target embedding and head are frozen and removed by the DFlash-family
`checkpoint_state_filter`. The change does not enlarge draft checkpoints or add
optimizer state. It only changes live frozen-target allocation and any logits
derived from it.

## Resource and performance impact

For every surveyed model except SGLang-registered Inkling, the exact delta is
zero.

For Inkling, the change from 200,058 to 201,024 adds 966 rows to two untied
`6144`-wide matrices:

- 11,870,208 additional frozen parameters;
- 22.64 MiB additional steady BF16 storage per unsharded replica;
- 45.28 MiB additional allocation during the current FP32 constructor phase;
- 5,935,104 additional multiply-accumulates per projected token;
- 966 additional logits per projected token, a 0.483% width increase.

The complete 201,024-row embedding plus head occupy about 4.60 GiB in BF16.
That large baseline already exists conceptually; the PR's incremental memory is
small. The current loader nevertheless constructs both matrices in FP32 before
moving them to the requested dtype and device, so peak host and per-rank device
memory should be measured separately. This PR does not solve that pre-existing
loading inefficiency.

Because the modules are frozen, the extra rows create no gradients and no Adam
moments. FSDP may shard the steady state later, but the current scripts construct
and move the full target components before the wrapper is sharded, so startup
peak remains per process.

## Correctness and compatibility risks

1. **Physical and generatable sizes are not interchangeable.** Inkling needs
   physical rows for weight loading and its padded mask embedding, but its
   reference logits are restricted to the first 200,058 rows.
2. **Successful loading can hide a distribution mismatch.** Full-width softmax
   changes normalization even when a padded token is not argmax. DSpark's L1 and
   confidence targets are especially sensitive to this.
3. **The source commit does not cover `TargetHead`.** EAGLE3 can still fail on a
   physical head whose rows exceed the config's logical `vocab_size`.
4. **The source commit does not load unknown configs.** Inkling needs a supported
   Transformers version, prior SGLang registration, or an audited raw-config
   fallback.
5. **The source commit does not resolve tensor keys.** DeepSeek V4, Kimi K3, and
   Inkling all use non-default checkpoint keys.
6. **Family names are not sufficient evidence.** Fine-tunes that add tokens,
   converted checkpoints, or repacks can change tensor rows independently of the
   upstream model. Load-time tensor validation remains necessary.
7. **Cropping or padding weights is unsafe.** A mismatched checkpoint must fail
   before training; it must never be silently truncated or extended.

## PR description

### Background

`TargetEmbeddingsAndHead` loads only the frozen input embedding and output head
used by DFlash-family objectives. It currently allocates both modules from
`config.vocab_size`, or from `config.text_config.vocab_size` for a multimodal
target, before copying checkpoint tensors into them.

Most open checkpoints use one number for module rows, logits, and sampling. A
few models expose multiple vocabulary domains. Inkling is the concrete case:
its checkpoint stores 201,024 physical rows, only 200,058 are generatable, and a
registered serving config may present the two numbers as `vocab_size` and
`padded_vocab_size`. Allocating from the logical value produces a shape mismatch;
using only the physical value for every downstream operation exposes padded
tokens to training and sampling.

### Motivation

The source patch fixes the first real Inkling checkpoint load, but the broader
survey shows that a model-name conditional or a global redefinition of
`vocab_size` would be the wrong abstraction. Mainstream Qwen, DeepSeek, GLM,
Kimi, gpt-oss, Phi, LongCat, Ling, and LFM checkpoints already agree with their
configs and must remain byte-for-byte unchanged.

SpecForge needs a small vocabulary-geometry contract at the frozen-target
boundary: one size for checkpoint storage and embedding lookup, and one size for
loss/generation logits. Making both explicit fixes the affected model, preserves
all no-op families, and gives shape errors enough context to diagnose config or
conversion problems before a distributed run starts.

### Design and implementation

Resolve a `VocabularyGeometry` from the selected text config:

1. `checkpoint_size` prefers a positive `padded_vocab_size`; otherwise it uses
   `vocab_size`.
2. `output_size` prefers a positive `unpadded_vocab_size`; otherwise, when
   `padded_vocab_size` exists and `vocab_size <= checkpoint_size`, it uses
   `vocab_size`; ordinary configs use `checkpoint_size` for both.
3. Require `0 < output_size <= checkpoint_size` and reject booleans, non-integral
   values, and an out-of-range `pad_token_id`.
4. Allocate and load `nn.Embedding` and the stored `nn.Linear` weight at
   `checkpoint_size`.
5. Preserve `config.vocab_size`; expose both resolved sizes on the target
   component and make the output-domain policy explicit at every projection
   consumer.

For public/native Inkling this resolves `(checkpoint=201024, output=200058)`.
For the audited SGLang config it resolves the same pair from
`padded_vocab_size=201024` and `vocab_size=200058`. For every other surveyed
model it resolves equal sizes.

The PR must validate embedding and head tensor shapes against
`checkpoint_size`. It must not infer correctness from the config alone and must
not silently crop or pad checkpoint weights. Weight tying remains a storage
property: tied models load one physical matrix and share it only after validation.

Output trimming must happen before softmax, cross-entropy, argmax, or speculative
token selection. If carrying `output_size` through DFlash/DSpark is judged too
large for this foundational PR, then the loader may land as latent support, but
Inkling execution must remain explicitly gated until candidate 6 supplies the
output-domain and muP-scaling integration.

## Implementation plan (code walkthrough)

### 1. Add an explicit vocabulary-geometry resolver

In `specforge/modeling/target/target_utils.py`, add a frozen internal value type
or validated tuple such as:

```python
VocabularyGeometry(checkpoint_size: int, output_size: int)
```

Add `_resolve_vocab_geometry(config)` and keep `text_config` selection inside the
resolver so hidden size, padding ID, and both vocabulary sizes always come from
the same language config. Include the original `vocab_size`,
`padded_vocab_size`, and `unpadded_vocab_size` values in validation errors.

Do not special-case any model type or hard-code any row count.

### 2. Allocate physical modules and validate both tensors

Update `TargetEmbeddingsAndHead.__init__` to allocate `embed_tokens` and
`lm_head` from `geometry.checkpoint_size`. Store the geometry on the component
without modifying the source config.

In `_load_file_content`, validate both embedding and head shapes before `copy_`.
The current code explicitly checks only the head; embedding failures fall through
to a generic `copy_` error. Each error should report:

- checkpoint key and file;
- actual and expected tensor shapes;
- hidden size;
- raw logical, padded, and unpadded config fields;
- resolved checkpoint and output sizes.

Preserve the existing tied-weight path, but validate the sole tensor before
replacing `lm_head.weight` with the embedding parameter.

### 3. Carry the output vocabulary through DFlash-family consumers

Expose `checkpoint_vocab_size` and `output_vocab_size` on
`TargetEmbeddingsAndHead`. Add one projection helper or pass `output_vocab_size`
to the wrappers so logits are sliced before any probability or token operation.

Audit these call sites:

- `specforge/core/dflash.py::OnlineDFlashModel`;
- `OnlineDominoModel` in the same module;
- `OnlineDSparkModel._aligned_target_logits` and every direct
  `F.linear(..., self.lm_head.weight)` path;
- `scripts/train_dflash.py`, `scripts/train_domino.py`, and
  `examples/disagg/_dflash_family_disagg.py` construction.

DSpark cannot use one ambiguous config field for both domains when a padded mask
ID is outside the output vocabulary. Its input/Markov contract must distinguish
embedding rows from output rows, or Inkling DSpark must stay gated for candidate
6. Add shape assertions before adding Markov bias or comparing draft and target
distributions.

### 4. Make the EAGLE3 decision explicit

`specforge/modeling/target/target_head.py::TargetHead` independently allocates an
LM head from `text_config.vocab_size`. Reuse the geometry resolver so its stored
weight has the checkpoint row count and its forward path returns only the output
domain.

If EAGLE3 intentionally requires a different policy, document and test that
policy. Leaving `TargetHead` silently inconsistent with the DFlash-family loader
is not an acceptable final state.

### 5. Add CPU regression tests for config dialects

Create `tests/test_modeling/test_target/test_target_utils.py` with tiny tensors
covering:

- ordinary config: `vocab_size == checkpoint_size == output_size`;
- public/native Inkling dialect: `vocab_size > unpadded_vocab_size`;
- SGLang Inkling dialect: `padded_vocab_size > vocab_size`;
- multimodal wrapper: geometry comes from `text_config`;
- tied and untied heads;
- `pad_token_id=None` and an invalid padding ID;
- missing, zero, boolean, non-integral, and inverted size fields;
- exact embedding and head shape errors;
- a padded input ID that embeds successfully but cannot appear in returned
  generatable logits.

Use tiny safetensors fixtures; CI must not download any production model.

### 6. Add model-level compatibility gates

Run config-and-header smoke tests outside the hermetic unit suite for the pinned
revisions in the matrix. The gate need only inspect metadata and load the two
small target components where weights are locally available.

Required assertions are:

- Qwen3, Qwen3.5, Qwen3.6, DeepSeek-V3, DeepSeek-V4, GLM-5.2, Kimi K3, and the
  extra surveyed models resolve equal sizes and retain their prior shapes;
- both public Inkling checkpoint variants resolve 201,024 physical and 200,058
  output rows regardless of config representation;
- the chosen padded mask row can be embedded;
- Inkling projection, loss, and argmax see only 200,058 output rows;
- the full stored head retains 201,024 rows and loads without mutation;
- no target component is added to a saved draft checkpoint.

### 7. Measure startup and projection overhead

For the affected Inkling path, record host peak during FP32 construction, device
peak before and after FSDP wrapping, and projection time with the physical head.
The expected steady-state allocation delta is 22.64 MiB BF16 per unsharded
replica. A materially larger delta indicates an accidental duplicate tensor or
lost weight tying.

## Scope and non-goals

- No vocabulary remapping or mutation of tokenizer IDs.
- No silent checkpoint cropping, padding, or random initialization.
- No automatic tensor-key guessing in this PR.
- No Kimi or Inkling remote-code/config fallback; that belongs with the
  model-integration work.
- No muP scaling change. Inkling's `logits_mup_width_multiplier` is a separate
  correctness requirement in candidate 6.
- No claim that GGUF, AWQ, FP4-packed heads, or arbitrary converted checkpoints
  are supported merely because their configs expose a size.

## Acceptance criteria

- All surveyed equal-size models resolve exactly their existing module and
  output shapes.
- SGLang-registered and native/raw Inkling configs resolve the same physical and
  output geometry.
- The 201,024-row Inkling embedding and head load without mutation.
- Padded input rows are addressable, while loss and generation operate on the
  200,058-row output domain.
- DFlash, Domino, DSpark, and EAGLE3 have an explicit and tested policy for both
  sizes.
- Invalid tensor/config combinations fail before distributed training with an
  actionable error.
- Frozen target rows create no optimizer state and do not enlarge draft
  checkpoints.

## Primary evidence

- [Hugging Face safetensors metadata parsing](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json), [Qwen3.5-35B-A3B config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json), and [Qwen3.6-27B config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json)
- [DeepSeek-V3 config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) and [DeepSeek-V4-Flash config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json)
- [GLM-5.2-FP8 config](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/ba978f7d347eaf65d22f1a86833408afdb953541/config.json)
- [Thinking Machines Inkling config](https://huggingface.co/thinkingmachines/Inkling/blob/86b4d430ab871652a707666b89203a866888c5e5/config.json)
- [Transformers Inkling config implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/inkling/configuration_inkling.py) and [logit truncation implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/inkling/modeling_inkling.py)
