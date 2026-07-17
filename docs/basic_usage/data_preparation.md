# 📝 Data Preparation

## 📍 Overview

Data is an important aspect of speculative decoding as the quality of the dataset directly affects the acceptance rate of the draft model. In this section, we will introduce how to prepare the dataset for both online and offline training.

## ☁️ Pre-supported Datasets

We have provided a script to prepare some sample datasets out of the box, these datasets include:
1. [ultrachat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (200k)
2. [sharegpt](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered) (120k)
3. [perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) (1.4M)
4. and others (we continuously add support for more datasets)

You can run the script below to prepare the corresponding dataset.

```bash
# ultrachat
python scripts/prepare_data.py --dataset ultrachat

# sharegpt
python scripts/prepare_data.py --dataset sharegpt
```

You can view the full list of pre-supported datasets using `python scripts/prepare_data.py --help`. The datasets are processed and saved as `jsonl` files in the `cache/dataset/<dataset_name>` directory of the project path by default.


## ↩️ Regenerate Datasets

> **Scope:** Dataset regeneration captures serialized conversation semantics only:
> message roles and text, visible assistant content, structured `reasoning_content`,
> tool calls/results, and trajectory provenance. It does not request, capture, store,
> or transport model hidden states, logits, embeddings, KV caches, or other tensor
> features. Any hidden-state preparation used by an offline training workflow is a
> separately invoked workflow outside regeneration. The canonical regenerated
> dataset retains every emitted text, reasoning, and tool block; field-reduced
> training views are separate derived datasets.

When training speculative decoding draft models for a specific target model, instead of using the original dataset, we can regenerate the assistant message text and reasoning trajectories using the target model to better align the draft model with the target model's output distribution. Later turns are conditioned on the newly regenerated textual history. This can improve the acceptance rate of the draft model and the overall performance of speculative decoding. According to the [EAGLE1 paper](https://arxiv.org/pdf/2401.15077), the EAGLE method is not very sensitive to dataset quality, which means performance can still be good with the original dataset. For optimal production alignment, however, regenerating the textual dataset with the target model is recommended.

We can follow the following steps to regenerate the dataset. In the example below, we will use `meta-llama/Llama-3.1-8B-Instruct` as an example, you can replace it with your own target model.

1. Start the SGLang server for the target model.

```shell
python3 -m sglang.launch_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --cuda-graph-bs 1 2 4 8 16 32 64 128 \
    --dtype bfloat16 \
    --mem-frac=0.8 \
    --port 30000
```

2. Author a regeneration recipe (see `examples/data_regeneration/recipes/`
   for complete templates) and run the artifact lifecycle:

```shell
specforge data regen run --config recipe.yaml \
    --endpoint teacher=http://localhost:30000
specforge data regen validate --artifact ./artifacts/sharegpt-regen
specforge data regen finalize --artifact ./artifacts/sharegpt-regen
```

The finalized artifact directory is the training input
(`data.dataset_artifact: <artifact>/manifest.json`). For reasoning models,
set the generator's `sampling.reasoning: required` to capture
`reasoning_content` for every regenerated turn, or `reasoning: disabled`
(with `chat_template_kwargs.enable_thinking: false`) for thinking-off
regeneration; both contracts are enforced during generation and validation.

The legacy `scripts/regenerate_train_data.py` entry point remains available
for one release as a deprecated compatibility wrapper. It accepts the
historical flags, executes through the same pipeline, produces the same
finalized artifact under `<output>.regen-artifact/`, and additionally derives
the historical `<output>.jsonl` / `_error.jsonl` / `_skipped.jsonl` files:

```shell
python scripts/regenerate_train_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --concurrency 128 \
    --max-tokens 98304 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl
```

With the wrapper, `--reasoning save` stores `reasoning_content` and
`--reasoning disable` disables thinking, as before.

### Multi-turn structured reasoning

A regenerated multi-turn reasoning conversation stores several assistant
targets in one row. When training uses last-turn-only loss masking, only the
final assistant target is supervised. Convert the validated conversation-level
output into one generation-event row per assistant turn so every reasoning
target is trained with the visible history available at its serving boundary:

```bash
python scripts/explode_generation_events.py \
    --input-file-path ./cache/dataset/sharegpt_train_regen_reasoning.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen_reasoning_exploded.jsonl
```

Each event ends at its current assistant target and preserves that turn's
`reasoning_content` and visible `content`. Historical assistant messages keep
only visible `content`; their earlier `reasoning_content` is removed from the event
context. Train the exploded output with the entry point's
`train_only_last_turn` option enabled.

This explosion is an optional derived training view. It does not alter the regeneration
contract: the validated conversation-level artifact retains every captured assistant
message and its complete reasoning trajectory.

The converter accepts only successful rows with non-empty IDs, message content,
and assistant `reasoning_content`. Invalid input is written to a skipped JSONL;
if any turn is invalid, the entire source conversation is skipped. Output files
must be fresh and distinct from the input.

For maximum performance, we recommend to scale the number of GPUs to regenerate the dataset in data parallel mode. To do this, you can simply add more server addresses to the `--server-address` argument, e.g. `--server-address localhost:30000 localhost:30001 localhost:30002 localhost:30003`.

### Qwen ShareGPT recipes

The Qwen model and sampling choices live in recipe files under
`examples/data_regeneration/recipes/`; the entry scripts select a recipe,
point it at the input dataset, and drive the
`specforge data regen run → validate → finalize` lifecycle with complete-row
accounting. They expect one or more SGLang servers to already be running; they
do not launch or stop the servers themselves.

For Qwen3-8B non-reasoning regeneration, start a target server in one terminal:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --host 0.0.0.0 \
    --port 30000
```

After `curl --fail http://127.0.0.1:30000/health` succeeds, run the recipe from
the SpecForge repository root:

```bash
MODEL_PROFILE=qwen3-8b \
INPUT_FILE=./cache/dataset/sharegpt_train.jsonl \
ARTIFACT_DIR=./cache/dataset/sharegpt-regen-qwen3-8b-non-reasoning \
SERVER_ADDRESSES="localhost:30000" \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

This profile (`recipes/qwen3-8b-sharegpt-non-reasoning.yaml`) sets temperature
to zero, disables thinking through `chat_template_kwargs.enable_thinking=false`,
and rejects non-empty structured reasoning and leaked think markers in
successful rows.

For Qwen3.6-27B structured-reasoning regeneration, start SGLang with the Qwen3
reasoning parser so the OpenAI-compatible response includes
`reasoning_content`:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-27B \
    --tp-size 1 \
    --dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 30000
```

After `curl --fail http://127.0.0.1:30000/health` succeeds, run:

```bash
MODEL_PROFILE=qwen3.6-27b \
INPUT_FILE=./cache/dataset/sharegpt_train.jsonl \
ARTIFACT_DIR=./cache/dataset/sharegpt-regen-qwen3.6-27b-reasoning \
SERVER_ADDRESSES="localhost:30000" \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

The default maximum completion lengths are 4096 tokens for Qwen3-8B and 32768
tokens for Qwen3.6-27B; the server context length must accommodate both the
rendered prompt and the recipe's `max_tokens`. Adjust these by editing the
recipe (or passing a dotted override such as
`generators.teacher.sampling.max_tokens=8192` to `specforge data regen run`).

`INPUT_FILE` may point to another dataset without changing the recipe as long
as it is JSONL in the conversation format documented below, with a non-empty
string `id` and alternating `user`/`assistant` messages. Always choose a fresh
`ARTIFACT_DIR`: the entry script deliberately refuses to reuse a finalized
artifact.

The run produces a finalized artifact directory containing the regenerated
data shards, a rejects ledger with stable failure categories, the attempt
history, validation reports, and a content-addressed `manifest.json`. The
accounting step passes only when
`success + error + skipped == input`; it prints the success fraction without a
fixed minimum threshold. Row structure, the reasoning contract, and raw
`<think>` marker defense are enforced by the recipe's validation profile
before the manifest can be published.

To distribute regeneration across independently launched servers, provide all
addresses as a space-separated list, for example:

```bash
SERVER_ADDRESSES="localhost:30000 localhost:30010" \
bash examples/data_regeneration/run_qwen_sharegpt_regeneration.sh
```

## 🤩 Prepare your own dataset

Besides the provided datasets, you can also prepare your own dataset. We support two formats:

#### Option 1: Conversation Format

You should prepare the dataset in jsonl format and the schema should look like this:

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ],
}
```

#### Option 2: Pre-formatted Text Format

If you already have conversations formatted with a specific chat template, you can use the pre-formatted text directly:

```json
{
    "id": "xxxx",
    "text": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n"
}
```

This format is useful when you have pre-formatted prompts that were used during training of the target model and have raw generations from the target model.

To use pre-formatted datasets, add the `--is-preformatted` flag to your training command. Note that the `--chat-template` parameter is still needed and should match the template used in your pre-formatted text, as it is used to identify user/assistant tokens to determine the assistant spans and generate the corresponding loss mask.

```bash
# Online training with pre-formatted data
torchrun --standalone --nproc_per_node 8 \
    scripts/train_eagle3.py \
    --is-preformatted \
    --train-data-path ./your_preformatted_dataset.jsonl \
    # ... other arguments
```

For offline training, you can also use `--is-preformatted` in the separate hidden-state
preparation step shown below. This command consumes an already prepared or regenerated
textual JSONL dataset; it is not part of dataset regeneration.

```bash
# Generate hidden states from pre-formatted data
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --data-path ./your_preformatted_dataset.jsonl \
    --output-path ./cache/hidden_states \
    --chat-template llama3 \
    --is-preformatted \
    --max-length 2048
```

Regeneration itself ends once the validated textual `jsonl` artifact is ready. You can then
use that artifact for online training or pass it to the separate offline preparation flow.
See the Training guide for more details.


## ➕ Handling Multiple Datasets

If you have multiple datasets, you can just merge them into the one jsonl file. For example, you can do something like this

```bash
cat dataset1.jsonl dataset2.jsonl > merged_dataset.jsonl
```
