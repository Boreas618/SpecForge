# PR report: validate GQA geometry and migrate the final GQA16 draft config

**Suggested PR title:** `fix(dflash): validate grouped-query attention geometry`

**Source material:** `b7f5c8f`, `40be4f8`

**Migration status:** selective port. The validation hunk applies to the current
attention implementation, while the source tests and target config use obsolete
paths and the old DSpark config layout.

## PR description

### Background

`Qwen3DFlashAttention` derives `num_key_value_groups` with integer division:

```python
num_attention_heads // num_key_value_heads
```

Without validation, zero, negative, or non-divisible head counts fail later in a
reshape or attention kernel with little indication that the configuration is
invalid. The source work added an explicit guard while evolving a target-specific
draft from multi-head attention through GQA8 to its final GQA16 shape.

The current DSpark implementation is newer than the source branch and stores
DSpark settings inside `dflash_config`. Therefore the generic geometry check is
directly reusable, but the final config must be translated rather than copied.

### Motivation

GQA changes the K/V projection width and the number of query groups sharing each
K/V head. A bad ratio can instantiate parameters successfully and fail only on
the first real batch. Validating at model construction makes configuration errors
deterministic, cheap, and testable on CPU. A checked final GQA16 config also gives
the private-target migration a stable quality-oriented endpoint without replaying
the superseded GQA8 intermediate state.

### Design and implementation

Validate the attention geometry in `Qwen3DFlashAttention.__init__` before any
projection is allocated:

- `num_attention_heads > 0`;
- `num_key_value_heads > 0`;
- `num_attention_heads % num_key_value_heads == 0`;
- explicit `head_dim > 0`;
- when `head_dim` is derived rather than supplied, `hidden_size` must be divisible
  by `num_attention_heads`.

After validation, compute `num_key_value_groups` once and keep the existing
projection formulas:

- Q rows: `num_attention_heads * head_dim`;
- K/V rows: `num_key_value_heads * head_dim`;
- output input width: `num_attention_heads * head_dim`.

The final target config uses 64 query heads and 16 K/V heads, so the group ratio
is four. Its DSpark fields must live in `dflash_config` and include
`projector_type="dspark"`, matching the current `DSparkDraftModel` contract.

## Implementation plan (code walkthrough)

### 1. Add constructor validation

Update `specforge/modeling/draft/dflash.py::Qwen3DFlashAttention.__init__` to
normalize the relevant config values to integers, validate them, and then create
the projections. The exception should include all four dimensions so an invalid
configuration can be diagnosed without a debugger.

Keep the validation in the attention class rather than only in a CLI or config
loader. Models can be instantiated through `AutoModel`, the draft registry, tests,
or downstream code that bypasses SpecForge's CLI.

### 2. Add focused model tests

Create `tests/test_modeling/test_draft/test_dflash.py` or extend the closest
draft-model test module with cases for:

- MHA (`num_attention_heads == num_key_value_heads`);
- valid GQA8 and GQA16 ratios;
- non-divisible query/KV counts;
- zero query or K/V heads;
- explicit `head_dim` where `hidden_size` is not equal to
  `num_attention_heads * head_dim` but the projection geometry is still valid;
- a forward/backward pass proving K and V gradients are populated for GQA16.

Use small hidden dimensions while preserving the 64:16 ratio in the regression
case.

### 3. Translate the final config

If publication of the target-specific config is approved, add the final GQA16
configuration under `configs/`. Translate the legacy fields as follows:

```json
{
  "dflash_config": {
    "projector_type": "dspark",
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "confidence_head_alpha": 1.0,
    "enable_confidence_head": true,
    "confidence_head_with_markov": true,
    "mask_token_id": "<target value>",
    "target_layer_ids": ["<approved layer ids>"]
  }
}
```

Keep architecture dimensions such as `num_attention_heads`,
`num_key_value_heads`, and `head_dim` at the top level because they belong to the
Qwen draft backbone. Do not add the GQA8 source config or its rename history.

### 4. Exercise the current registry path

Add a test that loads the translated config through `AutoConfig`, resolves
`DSparkDraftModel` through the draft registry, and asserts:

- `num_key_value_groups == 4`;
- Q/K/V projection shapes match the declared geometry;
- DSpark's Markov and confidence heads are enabled from `dflash_config`;
- the state-dict key layout matches the current direct DSpark architecture.

### 5. GPU validation

Run a short current `examples/disagg/run_disagg_dspark.py` consumer smoke with
server-captured features. The gate is a finite loss and non-zero K/V projection
gradients for at least two optimizer steps. This validates the current strategy
and FSDP path rather than the removed standalone trainer.

## Scope and non-goals

- This PR does not choose a quality/performance ratio for arbitrary models; it
  validates the ratio declared by a config.
- It does not migrate target-specific capture, templates, or launch flags.
- It does not support non-integral query-to-K/V grouping.

## Acceptance criteria

- Invalid head geometry fails during model construction with a precise message.
- MHA and valid GQA configurations retain existing numerical behavior.
- The final GQA16 config, if included, loads through the current DSpark registry
  and completes a forward/backward smoke.
