# Regenerating `mlabonne/open-perfectblend` with a target model (thinking-ON, SGLang)

How to produce an on-policy, thinking-ON regeneration of
[`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend)
(~1.42 M ShareGPT conversations) for a **specific target model**, using SGLang as the
generation server and the tooling already in this repo — i.e. how to reproduce
`mgoin/open-perfectblend-glm5.2-regen` for a new target. Companion background:
[GLM52_EXECUTION_DOC.md](./GLM52_EXECUTION_DOC.md) (how the regen data is consumed at
train time).

**What "regeneration" means here** (deepspec-style, per the mgoin dataset card and
`scripts/regenerate_train_data.py`):

* Human/system turns are **preserved verbatim** from the source dataset.
* Every assistant turn is **replaced** by the target model's own completion, generated
  **conditioned on the already-regenerated prior turns** (so multi-turn rows are
  on-policy at every turn, not just the first).
* With thinking ON, the stored assistant `content` is the model's **raw completion**:
  `{reasoning}</think>{answer}` — no opening `<think>`, because the chat template puts
  the opening `<think>` in the *generation prompt*, so it is never part of the model's
  output. This inline form is deliberate: the target's own chat template re-splits it at
  render time (see §1), and SpecForge's GLM pipeline only round-trips this form.
* This is response **regeneration** (decode). It is unrelated to the *hidden-state
  teacher* (a train-time prefill) — do not conflate the two (GLM52_EXECUTION_DOC R-DATA-1).

Throughout, `$MODEL` is the target (worked example: `zai-org/GLM-5.2-FP8`).

---

## 1. Decide the reasoning storage format FIRST

There are two ways to store a thinking-ON assistant turn, and the correct choice is
dictated by **the target tokenizer's `chat_template.jinja`**, not by preference:

| format | example row content | produced by | round-trips when |
|---|---|---|---|
| **A. inline** (mgoin-style, **recommended**) | `content = "{reasoning}</think>{answer}"` | SGLang **without** `--reasoning-parser` (raw completion) | the template's assistant branch splits `content` on `</think>` |
| B. split field | `content = "{answer}"`, `reasoning_content = "{reasoning}"` | SGLang **with** `--reasoning-parser` + `--reasoning save` | the template has a `reasoning_content is string` branch |

Before anything else, open the target's `chat_template.jinja` and check the assistant
branch for three properties:

1. **Which storage format it re-ingests.** GLM-5.2 accepts both — from its template:

   ```jinja
   {%- if m.reasoning_content is string %}
       {%- set reasoning_content = m.reasoning_content %}
   {%- elif '</think>' in content %}
       {%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}
       {%- set content = content.split('</think>')[-1] %}
   {%- endif %}
   ```

2. **The always-generated scaffold prefix** (what the template emits before model
   tokens — GLM: `<|assistant|><think>`). This becomes the SpecForge
   `assistant_header` for the loss mask (§5, V4).

3. **Which turns keep their reasoning on re-render.** GLM strips reasoning from every
   assistant turn at/before the last user turn (renders `<think></think>{answer}`) and
   keeps it only after the last user turn. Your loss-mask pattern must handle **both**
   renderings (this exact miss silently zero-masked 49.4 % of the GLM corpus — commit
   `1dac1b9`).

**Why inline wins for this repo:** SpecForge's non-`ThinkingParser` parsers sanitize
messages down to `{role, content, tool_calls}` (`specforge/data/parse.py:24,47`) — a
separate `reasoning_content` field is **silently dropped** by the GLM pipeline, and
`scripts/prepare_glm52_dspark_data.py` likewise keeps only role/content. If you must
use format B (e.g. the template has no `</think>`-split branch), re-inline it in a
post-pass before feeding SpecForge:

```python
if msg.get("reasoning_content"):
    msg["content"] = msg["reasoning_content"] + "</think>" + (msg["content"] or "")
    del msg["reasoning_content"]
```

(using whatever close-tag your target's template splits on).

---

## 2. Build the input jsonl

`scripts/prepare_data.py` already knows open-perfectblend and emits exactly the schema
`regenerate_train_data.py` consumes (`{"id", "conversations": [{"role","content"}]}`,
roles normalized from `{from,value}`):

```bash
python scripts/prepare_data.py \
    --dataset perfectblend \
    --output-path ./cache/dataset
# -> a jsonl of {"id": ..., "conversations": [{"role": "user"|"assistant"|"system", "content": ...}]}
```

Notes:

* Keep the `id` column — it is the anchor for the coverage check in §5 (output order
  is **not** input order; see V1).
* Keep **all turns**, not just the first user turn: the regen script walks the whole
  conversation and regenerates each assistant turn in place. Rows starting with an
  assistant turn or containing roles outside `{system, user, assistant}` are routed to
  the error file by the script.
* Shard the jsonl (`split -n l/N`) if you will run several regen workers/servers in
  parallel with separate output files — simpler resume semantics than one giant file.

---

## 3. Launch SGLang

Worked example — GLM-5.2-FP8 on one 4-GPU node:

```bash
python3 -m sglang.launch_server \
    --model-path zai-org/GLM-5.2-FP8 \
    --tp-size 4 \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs 256 \
    --host 0.0.0.0 --port 30000 \
    --trust-remote-code
```

The two flags that decide your dataset format:

* **Format A (inline, recommended): do NOT pass `--reasoning-parser`.** Without a
  parser, `/v1/chat/completions` returns the raw completion in `message.content` —
  i.e. `{reasoning}</think>{answer}` — which is exactly what you store.
* **Format B (split): pass `--reasoning-parser <name>`** (e.g. `glm45`, `qwen3`,
  `deepseek-r1` — whatever SGLang registers for the target) **and** run the regen
  script with `--reasoning save`. With a parser active, `message.content` contains
  only the post-`</think>` answer; running `--reasoning none` against a
  parser-enabled server **silently discards all reasoning** — the single easiest way
  to ruin a thinking-ON regen. Conversely `--reasoning save` against a
  parser-less server stores `reasoning_content: null`.

Scale-out: launch one server per node/GPU-group and pass all of them via
`--server-address` — the regen script round-robins with per-server concurrency
(`--concurrency` is per server).

---

## 4. Run the regeneration

`scripts/regenerate_train_data.py` is the harness (OpenAI client → SGLang, threaded,
resumable). GLM-5.2 example, format A:

```bash
python scripts/regenerate_train_data.py \
    --model zai-org/GLM-5.2-FP8 \
    --server-address 10.0.0.1:30000 10.0.0.2:30000 \
    --concurrency 128 \
    --input-file-path  ./cache/dataset/perfectblend.jsonl \
    --output-file-path ./cache/dataset/perfectblend_regen.jsonl \
    --temperature 1.0 --top-p 0.95 \
    --max-tokens 8192 \
    --reasoning none \
    --resume
```

Output rows keep all original fields (`id`, `source`, …), with `conversations`
replaced by the regenerated turns plus `status: "success"`. Failures go to
`<output>_error.jsonl` with an `error` field.

### 4.1 Getting thinking ON — per-model matrix

`--reasoning` in the script controls thinking as follows (`build_query_kwargs`):
`none` sends nothing (template default applies), `disable` sends
`chat_template_kwargs={"enable_thinking": false}`, `save` only changes what is
*stored* (§3). So:

| target template behavior | how to get thinking-ON |
|---|---|
| thinking is the **default** (GLM-5.2, Qwen3-style: `enable_thinking` unset ⇒ True, generation prompt ends with an open `<think>`) | `--reasoning none` — nothing to send |
| thinking is **opt-in** via a template kwarg | patch the script (below) and pass `--reasoning enable` |
| thinking controlled by another kwarg / a system prompt (model-specific) | put the right `chat_template_kwargs` in `extra_body`, same patch point |

The patch is two lines in `build_query_kwargs` (mirrors the existing `disable` branch):

```python
if args.reasoning == "enable":
    extra_body["chat_template_kwargs"] = {"enable_thinking": True}
```

(and add `"enable"` to the `--reasoning` choices). Verify with a one-off request that
the rendered prompt actually ends with the open think tag — SGLang logs the prompt at
`--log-level debug`, or check the completion starts mid-reasoning rather than with an
answer.

GLM-5.2 extra: the template also takes a `reasoning_effort` kwarg (`high` → "High",
anything else → "Max"; default Max). Leave it at default unless your deployment pins
one — it changes the `<|system|>Reasoning Effort: …` header, i.e. the distribution
you are distilling.

### 4.2 Sampling parameters

On-policy means *matching the deployment distribution*, so take temperature/top-p from
the target's `generation_config.json` (GLM-5.2 thinking mode: `temperature 1.0`,
`top_p 0.95`), not from habit. Two constraints in the script:

* It asserts `0.0 ≤ temperature ≤ 1.0` — fine for GLM (1.0 is the boundary), relax the
  assert if your target recommends >1.0.
* `--max-tokens` is per assistant turn. The default 4096 **truncates long reasoning
  chains** (a turn that dies mid-reasoning has no `</think>` at all — V2 catches
  this). mgoin used 8 k/turn; start there.

### 4.3 Recommended harness patch: store `finish_reason`

The script does not record `finish_reason`, so length-truncated turns are only
detectable heuristically. Before a 1.4 M-row run, add one line in `call_sglang`:

```python
resp_msg = {
    "role": "assistant",
    "content": response_text,
    "finish_reason": resp.choices[0].finish_reason,   # add
}
```

This turns V2's truncation check from a heuristic into an exact count. Strip the field
(or extend the prepare script's normalizer) before training — SpecForge's sanitizer
drops unknown keys anyway.

### 4.4 Resume semantics (caveat)

`--resume` counts lines in output + error files and **skips that many lines from the
head of the input**. Results are written in *completion* order, and in-flight requests
at kill time are lost — so after a crash the skipped-prefix assumption can be slightly
wrong (a few head rows unprocessed, a few later rows already written). Treat resume as
approximately correct during the run, and reconcile **by `id`** at the end (V1): rerun
exactly the missing ids into a patch file and concatenate. Dedup by `id` as well (a
resumed run can double-process a few rows).

---

## 5. Verifying the regenerated dataset

Layered, cheapest first. V1–V3 need no GPU; V4 needs the SpecForge repo + tokenizer;
V5 needs a serving target; V6 is the end-to-end gate. Run V1–V4 on the **full**
output, V5 on a sample.

> The V2–V4 snippets below were run (2026-07-11) against a 500-row slice of the real
> `mgoin/open-perfectblend-glm5.2-regen` + the GLM-5.2-FP8 tokenizer as a reference
> baseline: V2 = 96.4 % ok / 3.6 % `multiple_close_tags` / 0 everything else; V3 =
> pass (incl. a synthetic multi-turn history-strip case); V4 = 100 % retention, spans
> start at the reasoning and end `…</think>{answer}`. Expect your numbers to look
> like that, not like perfection.

### V1 — Coverage & integrity (id reconciliation)

Because output order ≠ input order and resume is approximate:

```python
import json

def ids(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line)["id"])
    return out

inp = set(ids("perfectblend.jsonl"))
ok = ids("perfectblend_regen.jsonl")
err = ids("perfectblend_regen_error.jsonl")
ok_set, err_set = set(ok), set(err)

print(f"input {len(inp):,}  success {len(ok):,}  error {len(err):,}")
print(f"missing (rerun these): {len(inp - ok_set - err_set):,}")
print(f"duplicated success rows (dedup these): {len(ok) - len(ok_set):,}")
print(f"unexpected ids (not in input): {len((ok_set | err_set) - inp):,}")
```

Gate: missing = 0 after patch-reruns, duplicates deduped, error rate < ~0.5 % (and
eyeball the error file — it should be malformed rows, not server errors).

### V2 — Format & shape (the think-tag contract)

Checks every assistant turn for the format-A contract: exactly one `</think>`, no
opening `<think>` (it was prompt scaffold), a non-empty answer, no truncation, no
leaked template tokens. Adjust `SPECIALS`/close-tag for a non-GLM target.

```python
import json, collections

CLOSE = "</think>"
SPECIALS = ["<|user|>", "<|assistant|>", "<|system|>", "<|observation|>",
            "[gMASK]", "<sop>", "<|endoftext|>"]
stats = collections.Counter()
examples = {}

with open("perfectblend_regen.jsonl") as f:
    for line in f:
        row = json.loads(line)
        for m in row["conversations"]:
            if m["role"] != "assistant":
                continue
            stats["asst_turns"] += 1
            c = m["content"] or ""
            n = c.count(CLOSE)
            if n == 0:
                # thinking-ON but no close tag ⇒ truncated mid-reasoning
                # (exact if you stored finish_reason: check == "length")
                key = "no_close_tag(truncated?)"
            elif n > 1:
                key = "multiple_close_tags"
            elif not c.split(CLOSE, 1)[1].strip():
                key = "empty_answer_after_think"
            elif "<think>" in c:
                key = "unexpected_open_tag"
            elif any(s in c for s in SPECIALS):
                key = "leaked_special_token"
            else:
                key = "ok"
            stats[key] += 1
            if key != "ok" and key not in examples:
                examples[key] = (row.get("id"), c[:160])

total = stats.pop("asst_turns")
for k, v in stats.most_common():
    print(f"{k:32s} {v:>9,}  ({100*v/total:.2f}%)")
for k, (i, snip) in examples.items():
    print(f"\n[{k}] id={i}\n{snip!r}")
```

Gates (tune to taste): `ok` ≥ 95 %; `no_close_tag` ≤ ~1 % (raise `--max-tokens` or
drop those rows); `leaked_special_token` = 0 (nonzero means a server-side
template/stop-token misconfiguration — fix and regenerate, don't paper over);
`empty_answer_after_think` small (drop rows).

`multiple_close_tags` deserves a real decision, not a shrug. Calibration: the actual
mgoin GLM-5.2 regen measures **~3.6 %** on a 500-row slice — thinking models emit
extra bare `</think>` separators in tool/Python-interpreter-style answers (multi-round
"reason → answer → reason → code" completions), not tag-quoting. The catch: GLM's
jinja takes `split('</think>')[0]` as reasoning and `split('</think>')[-1]` as answer,
so **every middle segment is silently dropped at render time** — the trained sequence
is `<think>{first}</think>{last}` and the stored middle content never reaches the
model. Options: (a) keep them and knowingly train on the jinja-lossy view (what the
GLM-5.2 run does), (b) drop the rows, (c) truncate content to the first
`{reasoning}</think>{answer}` round. Whatever you pick, V3 below asserts the actual
render behavior so the choice is explicit.

### V3 — Chat-template round-trip (the template *is* the reasoning parser)

Verifies that the target's jinja re-ingests your stored format the way training will
see it: final-turn reasoning preserved, history reasoning stripped, answer intact.

```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2-FP8")

def check(row):
    conv = row["conversations"]
    txt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    last_asst = max(i for i, m in enumerate(conv) if m["role"] == "assistant")
    for i, m in enumerate(conv):
        if m["role"] != "assistant" or "</think>" not in (m["content"] or ""):
            continue
        # Mirror the jinja split semantics exactly: FIRST segment = reasoning,
        # LAST segment = answer; middle segments (multi-close turns, V2) are
        # dropped by the template and must NOT appear in the render.
        parts = m["content"].split("</think>")
        reasoning, answer, middles = parts[0], parts[-1], parts[1:-1]
        if i == last_asst:
            # after the last user turn: reasoning must survive the re-render
            assert f"<think>{reasoning}</think>" in txt, "final-turn reasoning lost"
        else:
            # history turn: template strips reasoning to an empty block
            assert reasoning not in txt, "history reasoning unexpectedly kept"
        assert answer.strip() in txt, "answer lost in re-render"
        for mid in middles:
            assert mid not in txt, "template kept a middle segment (semantics changed?)"

n = 0
with open("perfectblend_regen.jsonl") as f:
    for line in f:
        check(json.loads(line)); n += 1
        if n >= 2000: break
print(f"round-trip OK on {n} rows")
```

If the final-turn assertion fails, your storage format and the template disagree —
back to §1 before touching SpecForge.

### V4 — SpecForge loss-mask retention (the `1dac1b9` check)

The costliest historical failure mode: the mask pattern matches only one of the two
assistant renderings and silently zero-masks half the corpus. Validate **before**
tokenizing 1.4 M rows — render, mask, decode the supervised span, measure retention:

```python
import json, torch
from transformers import AutoTokenizer
from specforge.data.parse import GLMParser          # match your template's parser_type
from specforge.data.template import TEMPLATE_REGISTRY

tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2-FP8")
parser = GLMParser(tok, TEMPLATE_REGISTRY.get("glm-5.2"))

kept = 0; n = 0; shown = 0
with open("perfectblend_regen.jsonl") as f:
    for line in f:
        conv = json.loads(line)["conversations"]
        input_ids, loss_mask = parser.parse(conv, max_length=4096)
        n += 1
        if loss_mask.sum() > 0:
            kept += 1
        if shown < 3 and loss_mask.sum() > 0:
            span = tok.decode(input_ids[loss_mask.bool()])
            print("supervised span starts:", repr(span[:120]), "…ends:", repr(span[-80:]), "\n")
            shown += 1
        if n >= 5000: break
print(f"retention: {kept}/{n} = {100*kept/n:.1f}%")
```

Gates:

* **Retention ≥ 99 %.** ~50 % means the thinking-ON rendering isn't matched by the
  assistant pattern (the exact `1dac1b9` bug).
* The decoded span for a thinking-ON turn must **start at the first reasoning token**
  (not after `</think>`) and **include the generated `</think>` + answer**; for a
  thinking-OFF/history turn it must exclude the scaffold `</think>`.
* Keep the runtime guards when training: `SPECFORGE_MIN_RETENTION` (abort < 90 %
  survival of the loss-mask filter) and **bump the processed-dataset cache key**
  whenever template/mask logic changes — the cache is keyed by template name and will
  happily serve stale masks.

For a new target you will first register the template: `assistant_header` = the
scaffold prefix from §1 point 2, plus a pattern that supervises
`{reasoning}{close}{answer}` and plain `{answer}` alike (see
`assistant_pattern_type="glm"` in `specforge/data/parse.py:156-176` as the model).

### V5 — On-policy check (is it really the target's distribution?)

The point of regeneration is that the tokens are high-probability under the target.
Cheap statistical check: teacher-forced NLL of the assistant tokens under the target,
**compared against the original (off-policy) open-perfectblend answers** on the same
prompts. Use the SGLang native API with prompt logprobs:

```python
import requests, json
from transformers import AutoTokenizer

HOST = "10.0.0.1:30000"
tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2-FP8")

def mean_nll(conv):
    txt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    r = requests.post(f"http://{HOST}/generate", json={
        "text": txt,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
        "return_logprob": True, "logprob_start_len": 0,
    }).json()
    lps = [t[0] for t in r["meta_info"]["input_token_logprobs"] if t[0] is not None]
    # crude but adequate: NLL over the whole rendered row; prompts are identical
    # between the two corpora, so the delta is driven by the assistant tokens.
    return -sum(lps) / len(lps)
```

Protocol: sample ~200 rows, compute `mean_nll` for (a) the regenerated row and (b) the
same row with the **original** answers substituted back in. Expect a large, consistent
gap — regenerated well under the original (rule of thumb: regen ≈ sampling entropy of
the target at your temperature; original typically several× higher). No gap ⇒ you did
not generate from the model you think you did (wrong server, wrong template, thinking
silently off, or a reasoning parser ate the reasoning).

Also worthwhile: 10 prompts regenerated twice at `temperature 0` must match exactly
(server determinism / no cross-request contamination).

### V6 — End-to-end gate

Before committing the multi-week run: normalize/tokenize the corpus
(`scripts/prepare_glm52_dspark_data.py`-style), smoke-train the drafter a few hundred
steps, and run the accept-length eval (`scripts/eval_dspark_deepspec.py`) **in the
same thinking mode as the data** (the GLM-5.2 pipeline is thinking-ON only). Accept length
climbing off the random-draft floor is the only verification that closes the loop;
everything above just makes sure this step can't fail for a data reason.

---

## 6. Pitfall register

| # | pitfall | symptom | guard |
|---|---|---|---|
| P1 | `--reasoning-parser` on the server + `--reasoning none` in the script | dataset has answers but **no reasoning at all** | §3; V2 shows ~0 % `</think>` |
| P2 | mask pattern matches only one thinking rendering | ~50 % of samples zero-masked and filtered | V4 retention; `SPECFORGE_MIN_RETENTION`; cache-key bump |
| P3 | `--max-tokens` too small for reasoning chains | turns with no `</think>` (died mid-reasoning) | V2 `no_close_tag`; store `finish_reason` (§4.3); 8 k/turn |
| P4 | resume after crash skips/dupes rows | count mismatch, duplicate ids | V1 id reconciliation + patch rerun |
| P5 | separate `reasoning_content` fed to SpecForge GLM pipeline | reasoning silently stripped by the message sanitizer | §1 re-inline post-pass |
| P6 | sampling params ≠ deployment (`generation_config.json`) | data subtly off-policy; V5 gap smaller than expected | §4.2 |
| P7 | training render vs deployment prompt mismatch (e.g. a thinking-mode system header present at eval but absent at train because the training render passed a different `enable_thinking`) | small accept-length loss at prompt starts | render training byte-identical to deployment (the GLM-5.2 pipeline passes `enable_thinking=True` in both `GLMParser` and the eval encoder) |
| P8 | stale processed-dataset cache after any template fix | fixed code, old masks | version the cache key (see `train_dspark.py` `maskv2-*`) |
