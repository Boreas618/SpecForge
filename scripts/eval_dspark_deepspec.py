#!/usr/bin/env python3
# coding=utf-8
"""DeepSpec-style mean-accepted-length evaluator for a dense GLM-5.2 DSpark drafter.

This is the *correct* speculative-decoding acceptance evaluator (it replaces the
probe-based ``eval_dspark_probe.py``, which only measured teacher-forced
draft-vs-data / draft-vs-teacher agreement on fixed batches and never actually
ran the propose -> verify -> accept loop).

What it measures
----------------
For each eval prompt it runs speculative decoding with the trained DSpark
drafter against the *real* target model and reports, per dataset:

  * ``mean_accepted_length`` = acceptance_length_sum / num_proposals (the DeepSpec
    ``acceptance_length`` metric; each verify step contributes accepted + 1 for
    the target's bonus token, or just accepted when a stop token terminates the
    step inside the accepted drafts).
  * ``accept_rate@k``       = accepted_at_pos[k] / proposals_at_pos[k], where a
    proposal counts at k iff its effective proposal length > k and is accepted at
    k iff its accepted-draft length > k (position 0 .. block_size-1).
  * ``verify_rate``         = acceptance_length_sum / (proposal_length_sum +
    num_proposals).

plus a pooled ``overall`` mean over all datasets.

Algorithm (DeepSpec, faithful rejection-sampling verify)
--------------------------------------------------------
This is a 100% faithful mirror of DeepSpec's ``generate_decoding_sample`` +
``verify_draft_tokens`` (deepspec/eval/base_evaluator.py) and its sampling math
(deepspec/utils/sampling.py). DeepSpec's rejection-sampling accept test reduces
to greedy argmax-match at temperature 0, so the result is bit-exact vs
DeepSpec-greedy at ``--temperature 0`` and remains faithful at any temperature.

1. Prefill the prompt through the target -> aux context hidden states (for the
   draft) + the target's final hidden state (-> target lm_head -> first token via
   ``sample_from_probs(logits_to_probs(...))``).
2. Repeat:
   (a) PROPOSE: the draft proposes ``block_size`` tokens. Slot 0 embeds the last
       accepted token, slots 1..block_size-1 embed ``mask_token_id``. Run the
       draft backbone once (conditioned on the target aux context), then apply
       the Markov head *autoregressively* within the block: each slot's logits
       get the low-rank bigram bias from the previously drafted token. Returns
       the ``block_size`` drafted tokens AND the per-position draft distribution
       ``draft_probs`` used to propose them.
   (b) VERIFY: run the target over ``[accepted_token, draft_0 .. draft_{B-1}]``
       and read its per-position distribution ``target_probs =
       logits_to_probs(target_lm_head(last_hidden_states))``.
   (c) ACCEPT (rejection sampling): accept_prob[j] = min(1, target_probs[j,tok] /
       draft_probs[j,tok]); accept a leading prefix via ``rand < accept_prob``
       (cumprod). The next token is a residual sample on the first rejected
       position, or the target's bonus token on full accept. Truncate on a stop
       token, advance, and reuse the verify prefill's hidden states as the next
       block's draft context.

Why re-prefill?  (cost / correctness trade-off)
-----------------------------------------------
The 753B ``zai-org/GLM-5.2-FP8`` target CANNOT fit on one GPU, so it is driven
through SpecForge's tp-sharded sglang backend (``SGLangDFlashTargetModel``).
That backend only exposes a *prefill* path (``generate_dflash_data`` /
``_extend``) -- there is **no** KV-cache incremental-decode API. So each verify
step RE-PREFILLS the whole growing sequence and reads ``last_hidden_states`` (->
target lm_head -> logits) at the block positions, plus the aux context for the
next draft block. Under greedy this is numerically identical to DeepSpec's
KV-cache verify path (same prefix -> same per-position logits); it is O(n^2) in
sequence length but exactly correct. The standalone CLI defaults to DeepSpec's
gsm8k protocol (``--max-new-tokens 2048``, ``--limit-per-task 500``); shrink both
for a quick sweep or for the in-training periodic eval.

TODO: a faster path would keep the target KV cache across spec steps (needs an
incremental-decode entry point on the sglang backend) instead of re-prefilling.

Datasets
--------
Reads prebuilt ``<task>.jsonl`` files (one JSON object per line with a ``turns``
list; only the first turn is used) from ``--eval-datasets-dir``. Build them with
the DeepSpec converter:

    python eval_datasets/convert_eval_datasets_to_jsonl.py openai/gsm8k \\
        --output-path <eval-datasets-dir>/gsm8k.jsonl

(see ``deepspec/eval_datasets/convert_eval_datasets_to_jsonl.py`` for all 9
tasks + their formatting). Missing dataset files are skipped with a warning.

APIs
----
* ``run_deepspec_eval(target_model, draft_model, target_lm_head,
  target_embed_tokens, tokenizer, tasks, eval_datasets_dir, ...) -> dict`` --
  in-process entry point so the trainer can call it periodically, reusing the
  already-loaded sglang target + draft (no reload). Side-effect-free: it
  restores the draft's train/eval mode and attention backend on exit.
* ``main()`` -- standalone CLI that builds the target (sglang), draft
  (``DSparkDraftModel.from_pretrained``), target embed/lm_head
  (``TargetEmbeddingsAndHead``) and tokenizer itself, then calls
  ``run_deepspec_eval``.

Distributed
-----------
Launch with ``torchrun``. The target is one tp-sharded replica per data-parallel
group (``init_distributed(tp_size=...)``). For the 753B target use
``tp_size == world_size`` (one replica, ``dp_size=1``): every rank runs the same
prompts in lockstep (the sglang forward is collective within the TP group).
When ``dp_size>1`` prompts are sharded across replicas and metric sums are
all-reduced over the data-parallel group.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from specforge.args import SGLangBackendArgs
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    init_distributed,
)
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.utils import get_local_device, print_on_rank0

# The DeepSpec 9-task suite with their upstream per-task sample caps. When
# ``--limit-per-task`` is set it further caps each of these.
DEFAULT_TASKS: Tuple[Tuple[str, int], ...] = (
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25", 30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
)


# ---------------------------------------------------------------------------
# Sampling helpers -- verbatim ports of DeepSpec's deepspec/utils/sampling.py
# (+ seed_all) so the propose/verify/accept math is a bit-exact mirror. At
# temperature 0 every one of these reduces to the deterministic argmax path.
# ---------------------------------------------------------------------------
def logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """DeepSpec ``logits_to_probs``: one-hot argmax at temp<1e-5, else softmax."""
    if temperature < 1e-5:
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, torch.argmax(logits, dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """DeepSpec ``sample_from_probs``: multinomial over probs [B, S, V] -> [B, S]."""
    bsz, seq_len, vocab_size = probs.shape
    flat = probs.reshape(-1, vocab_size)
    return torch.multinomial(flat, num_samples=1).reshape(bsz, seq_len)


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """DeepSpec ``sample_tokens``: argmax at temp<1e-5, else multinomial. -> [B, S]."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    flat_logits = logits.reshape(-1, vocab_size) / temperature
    probs = torch.softmax(flat_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).reshape(bsz, seq_len)


def gather_token_probs(probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """DeepSpec ``gather_token_probs``: probs[..., V], ids[...] -> [...]."""
    return probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def sample_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """DeepSpec ``sample_residual``: sample the clamped (target-draft) residual. -> [B]."""
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    if torch.any(residual_mass <= 1e-8):
        residual = torch.where(residual_mass <= 1e-8, target_probs, residual)
        residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = residual / residual_mass.clamp_min(1e-8)
    return sample_from_probs(residual.unsqueeze(1)).squeeze(1)


def seed_all(seed: int) -> None:
    """DeepSpec ``seed_all``: seed the CPU + all-CUDA torch RNGs (per-sample)."""
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _draft_block_tokens(
    draft_model: DSparkDraftModel,
    base_logits: torch.Tensor,
    *,
    seed_token: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autoregressive within-block draft sampling (DeepSpec ``sample_block_tokens``).

    ``base_logits`` [1, block_size, vocab] are ``target_lm_head`` applied to the
    (mask-noise) draft backbone hidden states -- computed once, in parallel, for
    the whole block. The Markov head then adds, per slot j, the low-rank bigram
    bias conditioned on the *previously drafted* token (slot 0's prev token is
    the accepted ``seed_token``); slot j is sampled from the corrected logits and
    then feeds slot j+1's bias. Mirrors ``VanillaMarkov.sample_block_tokens`` in
    DeepSpec, using SpecForge's ``VanillaMarkov.compute_step_bias(token_ids)``
    (no hidden_states arg -- identical for the vanilla head).

    Returns ``(tokens [1, block_size], corrected_logits [1, block_size, vocab])``;
    the caller forms ``draft_probs = logits_to_probs(corrected_logits, temp)`` so
    the drafted token is exactly a sample from that distribution.
    """
    block_size = base_logits.shape[1]
    markov = getattr(draft_model, "markov_head", None)
    if markov is None:
        # No bigram correction: slots are independent, so sample in parallel and
        # the corrected logits are just the base logits.
        return sample_tokens(base_logits, temperature), base_logits

    sampled: List[torch.Tensor] = []
    corrected: List[torch.Tensor] = []
    prev_token = seed_token.long()  # [1]
    for step_idx in range(block_size):
        step_logits = base_logits[:, step_idx, :]  # [1, vocab]
        bias = markov.compute_step_bias(prev_token)  # [1, vocab]
        step_logits = step_logits + bias.to(step_logits.dtype)
        corrected.append(step_logits.unsqueeze(1))
        next_token = sample_tokens(step_logits.unsqueeze(1), temperature).squeeze(1)
        sampled.append(next_token)
        prev_token = next_token
    return torch.stack(sampled, dim=1), torch.cat(corrected, dim=1)


def _contains_stop(
    token_ids: torch.Tensor, stop_token_ids: Optional[Sequence[int]]
) -> bool:
    if not stop_token_ids:
        return False
    stop = torch.tensor(list(stop_token_ids), device=token_ids.device)
    return bool(torch.isin(token_ids, stop).any().item())


def _resolve_stop_token_ids(tokenizer) -> Optional[List[int]]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return None
    if isinstance(eos, int):
        return [int(eos)]
    out: List[int] = []
    for token_id in eos:
        token_id = int(token_id)
        if token_id not in out:
            out.append(token_id)
    return out or None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def _load_task_prompts(
    task: str,
    eval_datasets_dir: str,
    limit: Optional[int],
    seed: int,
) -> Optional[List[str]]:
    """Load first-turn prompts for ``task``; None if the jsonl is missing."""
    path = os.path.join(eval_datasets_dir, f"{task}.jsonl")
    if not os.path.exists(path):
        return None

    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = row.get("turns")
            if not turns:
                continue
            prompts.append(turns[0])

    # Deterministic shuffle-then-truncate (matches DeepSpec BaseEvaluator), so
    # every rank selects the same subset.
    if limit is not None and len(prompts) > limit:
        rng = random.Random(seed)
        rng.shuffle(prompts)
        prompts = prompts[:limit]
    return prompts


def _encode_prompt(
    tokenizer,
    turn: str,
    device: torch.device,
    max_prompt_len: int,
) -> Optional[torch.Tensor]:
    """Format one user turn with the chat template (enable_thinking=False)."""
    messages = [{"role": "user", "content": turn}]
    try:
        enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
    except TypeError:
        # Older/other tokenizers may not accept enable_thinking.
        enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    # apply_chat_template may return a bare tensor OR a BatchEncoding/dict
    # (GLM-5.2's tokenizer returns the latter); normalize to a [1, S] tensor.
    input_ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[1] > max_prompt_len:
        return None
    return input_ids.to(device)


# ---------------------------------------------------------------------------
# Core speculative-decoding loop (single prompt, re-prefill verify)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _spec_decode_sample(
    *,
    target_model,
    draft_model: DSparkDraftModel,
    target_lm_head: torch.nn.Module,
    target_embed_tokens: torch.nn.Module,
    input_ids: torch.Tensor,  # [1, num_input] on device
    block_size: int,
    mask_token_id: int,
    max_new_tokens: int,
    temperature: float,
    stop_token_ids: Optional[Sequence[int]],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, object]:
    """DeepSpec speculative decoding for one prompt via re-prefill verification.

    A faithful mirror of DeepSpec's ``generate_decoding_sample`` +
    ``verify_draft_tokens`` (rejection-sampling accept), which reduces to greedy
    argmax-match at temperature 0. Returns, per proposal:

      * ``acceptance_lengths``     -- accepted + bonus (or ``accepted`` when the
        step is terminated by a stop token inside the accepted drafts),
      * ``proposal_lengths``       -- the effective proposal length (``block_size``
        or ``eos_pos + 1`` on EOS-terminate),
      * ``accepted_draft_lengths`` -- the number of accepted drafted tokens,

    plus token counts.
    """
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens

    ones = torch.ones_like(input_ids)

    # ---- Prefill the prompt ----
    out = target_model.generate_dflash_data(input_ids, ones, ones)
    if out.last_hidden_states is None:
        raise RuntimeError(
            "target backend did not surface last_hidden_states; DeepSpec eval "
            "needs the target final hidden state to compute verification logits."
        )
    # context = aux multi-layer feature for positions 0..num_input-1 (draft ctx).
    context = out.hidden_states.to(device=device, dtype=dtype)  # [1, num_input, k*h]
    last_hidden = out.last_hidden_states.to(device=device, dtype=dtype)

    # First token = sample_from_probs(logits_to_probs(target_logits)) at last pos.
    first_logits = target_lm_head(last_hidden[:, -1:, :])  # [1, 1, vocab]
    first_token = sample_from_probs(
        logits_to_probs(first_logits, temperature)
    )  # [1, 1]

    cur_ids = torch.cat([input_ids, first_token], dim=1)  # [1, num_input+1]
    start = num_input  # absolute position index of the current accepted (seed) token

    acceptance_lengths: List[int] = []
    proposal_lengths: List[int] = []
    accepted_draft_lengths: List[int] = []

    if _contains_stop(first_token, stop_token_ids):
        return {
            "acceptance_lengths": acceptance_lengths,
            "proposal_lengths": proposal_lengths,
            "accepted_draft_lengths": accepted_draft_lengths,
            "num_input": num_input,
            "num_output": 1,
        }

    while start < max_length:
        # ---- (a) PROPOSE: draft block_size tokens + their per-position probs ----
        draft_input_ids = torch.full(
            (1, block_size), int(mask_token_id), dtype=torch.long, device=device
        )
        draft_input_ids[:, 0] = cur_ids[:, start]  # slot 0 = last accepted token
        noise_embedding = target_embed_tokens(draft_input_ids).to(dtype)  # [1, B, h]
        # position_ids cover [context (0..start-1) | block (start..start+B-1)].
        position_ids = torch.arange(start + block_size, device=device).unsqueeze(0)
        block_hidden = draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=context,  # [1, start, k*h]
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )  # [1, block_size, h]
        base_logits = target_lm_head(block_hidden)  # [1, block_size, vocab]
        draft_tokens, draft_logits = _draft_block_tokens(
            draft_model,
            base_logits,
            seed_token=draft_input_ids[:, 0],
            temperature=temperature,
        )  # [1, block_size], [1, block_size, vocab]
        draft_probs = logits_to_probs(draft_logits, temperature)  # [1, B, vocab]

        # verify_input_ids: slot 0 = current accepted token, slots 1.. = drafts.
        verify_input_ids = torch.cat(
            [cur_ids[:, start : start + 1], draft_tokens], dim=1
        )  # [1, 1+B]

        # ---- (b) VERIFY: re-prefill the whole growing sequence + drafts ----
        # There is no KV-cache incremental decode on the sglang backend, so we
        # re-prefill; under greedy this is numerically identical to DeepSpec's
        # KV-cache verify path (same prefix -> same logits at every position).
        verify_ids = torch.cat([cur_ids, draft_tokens], dim=1)  # [1, start+1+B]
        v_ones = torch.ones_like(verify_ids)
        vout = target_model.generate_dflash_data(verify_ids, v_ones, v_ones)
        if vout.last_hidden_states is None:
            raise RuntimeError("target backend did not surface last_hidden_states.")
        v_last = vout.last_hidden_states.to(device=device, dtype=dtype)
        # target logits over [accepted_token, draft_0..draft_{B-1}] == DeepSpec's
        # target_output.logits [1, 1+B, V] (positions start..start+B of the
        # re-prefill mirror the KV-cache forward over verify_input_ids).
        target_logits = target_lm_head(
            v_last[:, start : start + block_size + 1, :]
        )  # [1, 1+B, vocab]
        target_probs = logits_to_probs(target_logits, temperature)  # [1, 1+B, vocab]

        # ---- (c) ACCEPT: DeepSpec rejection sampling ----
        proposed = verify_input_ids[:, 1:]  # [1, B]
        sel_t = gather_token_probs(target_probs[:, :-1, :], proposed)  # [1, B]
        sel_d = gather_token_probs(draft_probs, proposed).clamp_min(1e-8)  # [1, B]
        accept_prob = torch.clamp(sel_t / sel_d, max=1.0)  # [1, B]
        accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
        accept_prefix = accept_mask.cumprod(dim=1)
        accepted = int(accept_prefix.sum(dim=1)[0].item())  # in [0, block_size]

        # ---- EOS: truncate accepted to the first stop token (inclusive) ----
        effective_proposal_length = block_size
        terminated_by_stop = False
        if stop_token_ids and accepted > 0:
            accepted_slice = verify_input_ids[0, 1 : accepted + 1]
            stop_tensor = torch.tensor(
                list(stop_token_ids),
                device=accepted_slice.device,
                dtype=accepted_slice.dtype,
            )
            eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]
            if eos_hits.numel() > 0:
                eos_pos = int(eos_hits[0].item())
                accepted = eos_pos + 1
                effective_proposal_length = eos_pos + 1
                terminated_by_stop = True

        # ---- next token: residual (partial accept) or bonus (full accept). Always
        # sampled (like DeepSpec) so the RNG stream matches even when terminated. ----
        if accepted < block_size:
            next_token = sample_residual(
                target_probs[:, accepted, :], draft_probs[:, accepted, :]
            )  # [1]
        else:
            next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)  # [1]

        proposal_lengths.append(effective_proposal_length)
        accepted_draft_lengths.append(accepted)
        accepted_tokens = draft_tokens[:, :accepted]  # [1, accepted]

        # ---- COMMIT ----
        if terminated_by_stop:
            # A stop token is inside the accepted drafts: commit them, drop the
            # bonus, and stop (DeepSpec's terminated_by_stop_token branch).
            cur_ids = torch.cat([cur_ids, accepted_tokens], dim=1)
            acceptance_lengths.append(accepted)
            break

        cur_ids = torch.cat([cur_ids, accepted_tokens, next_token.view(1, 1)], dim=1)
        acceptance_lengths.append(accepted + 1)
        new_start = start + accepted + 1
        # Reuse the verify prefill's aux hidden as the next block's draft context
        # (positions 0..new_start-1 are all committed tokens, so their hidden
        # states are computed against the correct prefix).
        context = vout.hidden_states[:, :new_start, :].to(device=device, dtype=dtype)
        start = new_start

        # Stop if the newly committed tokens (accepted drafts + next_token) hit EOS.
        new_tokens = torch.cat([accepted_tokens, next_token.view(1, 1)], dim=1)
        if _contains_stop(new_tokens, stop_token_ids):
            break

    return {
        "acceptance_lengths": acceptance_lengths,
        "proposal_lengths": proposal_lengths,
        "accepted_draft_lengths": accepted_draft_lengths,
        "num_input": num_input,
        "num_output": cur_ids.shape[1] - num_input,
    }


# ---------------------------------------------------------------------------
# Distributed reduction
# ---------------------------------------------------------------------------
def _reduce_int_list(
    values: Sequence[int],
    group,
    device: torch.device,
) -> List[int]:
    """SUM-all-reduce a list of ints over the data-parallel (replica) group.

    Every rank of a TP group holds identical local sums (they ran the same
    prompts), and each dp_group contains exactly one rank per replica, so a SUM
    over dp_group yields the correct global total on every rank without double
    counting the tp replicas.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return list(values)
    tensor = torch.tensor(list(values), dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor.tolist()


def _normalize_tasks(
    tasks: Sequence[Union[str, Tuple[str, int]]],
    limit_per_task: Optional[int],
) -> List[Tuple[str, Optional[int]]]:
    normalized: List[Tuple[str, Optional[int]]] = []
    for task in tasks:
        if isinstance(task, str):
            name, task_limit = task, limit_per_task
        else:
            name, task_limit = task[0], task[1]
            if limit_per_task is not None:
                task_limit = (
                    limit_per_task
                    if task_limit is None
                    else min(int(task_limit), int(limit_per_task))
                )
        normalized.append((name, task_limit))
    return normalized


# ---------------------------------------------------------------------------
# Public entry point (reuse already-loaded target + draft)
# ---------------------------------------------------------------------------
def run_deepspec_eval(
    target_model,
    draft_model: DSparkDraftModel,
    target_lm_head: torch.nn.Module,
    target_embed_tokens: torch.nn.Module,
    tokenizer,
    tasks: Optional[Sequence[Union[str, Tuple[str, int]]]],
    eval_datasets_dir: str,
    *,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    max_new_tokens: int = 512,
    limit_per_task: Optional[int] = 64,
    temperature: float = 0.0,
    max_prompt_len: int = 2048,
    seed: int = 980406,
    stop_token_ids: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    output_json: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the DeepSpec acceptance eval in-process and return a metrics dict.

    The trainer can call this periodically, passing the already-built sglang
    target, the (unwrapped) DSparkDraftModel, and the target embed/lm_head from
    :class:`TargetEmbeddingsAndHead`. This function does not init/destroy
    distributed and restores the draft's train/eval mode + attention backend on
    exit, so it is safe to interleave with training.
    """
    device = device or get_local_device()

    if block_size is None:
        block_size = int(draft_model.block_size)
    if mask_token_id is None:
        mask_token_id = getattr(draft_model, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError(
            "mask_token_id could not be resolved from the draft config; pass it "
            "explicitly (run_deepspec_eval(..., mask_token_id=...))."
        )

    if stop_token_ids is None:
        stop_token_ids = _resolve_stop_token_ids(tokenizer)

    # Ensure the sglang target captures the draft's context layers.
    target_model.set_capture_layers(list(draft_model.target_layer_ids))

    # The fresh single-block draft forward passes attention_mask=None, which
    # flex_attention cannot consume -> force sdpa for the eval, then restore.
    prev_attn = getattr(draft_model.config, "_attn_implementation", None)
    prev_training = draft_model.training
    if prev_attn not in ("sdpa", "eager"):
        draft_model.config._attn_implementation = "sdpa"
    draft_model.eval()

    dp_group = get_dp_group()
    if dist.is_available() and dist.is_initialized():
        if dp_group is not None:
            replica_id = dist.get_rank(dp_group)
            num_replicas = dist.get_world_size(dp_group)
        else:
            replica_id = dist.get_rank()
            num_replicas = dist.get_world_size()
    else:
        replica_id, num_replicas = 0, 1

    dtype = next(draft_model.parameters()).dtype

    # Default to the single gsm8k task (e.g. the trainer's periodic eval passes
    # tasks=None); any of the DeepSpec 9 may be requested explicitly.
    if not tasks:
        tasks = ["gsm8k"]
    normalized_tasks = _normalize_tasks(tasks, limit_per_task)

    per_dataset: Dict[str, Dict[str, object]] = {}
    overall_accept_sum = 0
    overall_proposal_count = 0
    overall_sample_count = 0

    try:
        for task_name, task_limit in normalized_tasks:
            prompts = _load_task_prompts(task_name, eval_datasets_dir, task_limit, seed)
            if prompts is None:
                print_on_rank0(
                    f"[deepspec-eval] skipping '{task_name}': "
                    f"{os.path.join(eval_datasets_dir, task_name + '.jsonl')} not found"
                )
                continue

            # Mirror DeepSpec BaseEvaluator.run_dataset: reseed torch RNG once per
            # dataset before iterating (per-sample seed_all(seed+idx) follows).
            seed_all(int(seed))

            local_sample_count = 0
            local_proposal_count = 0
            local_acceptance_length_sum = 0
            local_proposal_length_sum = 0
            local_proposals_at_pos = [0] * block_size
            local_accepted_at_pos = [0] * block_size

            for global_idx in range(replica_id, len(prompts), num_replicas):
                # Deterministic per-sample seed (DeepSpec seed_all(seed+idx));
                # matters only for temperature>0 sampling.
                seed_all(int(seed) + global_idx)
                input_ids = _encode_prompt(
                    tokenizer, prompts[global_idx], device, max_prompt_len
                )
                if input_ids is None:
                    # Deterministic across all TP ranks of this replica, so they
                    # all skip together and stay in collective lockstep.
                    continue

                stats = _spec_decode_sample(
                    target_model=target_model,
                    draft_model=draft_model,
                    target_lm_head=target_lm_head,
                    target_embed_tokens=target_embed_tokens,
                    input_ids=input_ids,
                    block_size=block_size,
                    mask_token_id=int(mask_token_id),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    stop_token_ids=stop_token_ids,
                    device=device,
                    dtype=dtype,
                )
                local_sample_count += 1
                # DeepSpec allreduce_response_metrics: acceptance_length_sum,
                # proposal_length_sum, and per-position tallies keyed on the
                # *effective* proposal length (block_size or eos_pos+1).
                for accept_len, prop_len, accepted in zip(
                    stats["acceptance_lengths"],
                    stats["proposal_lengths"],
                    stats["accepted_draft_lengths"],
                ):
                    local_proposal_count += 1
                    local_acceptance_length_sum += int(accept_len)
                    local_proposal_length_sum += int(prop_len)
                    for pos in range(block_size):
                        if prop_len > pos:
                            local_proposals_at_pos[pos] += 1
                        if accepted > pos:
                            local_accepted_at_pos[pos] += 1

            # Reduce integer SUMS (never ratios) over the data-parallel group.
            reduced = _reduce_int_list(
                [
                    local_sample_count,
                    local_proposal_count,
                    local_acceptance_length_sum,
                    local_proposal_length_sum,
                ]
                + local_proposals_at_pos
                + local_accepted_at_pos,
                dp_group,
                device,
            )
            sample_count = reduced[0]
            proposal_count = reduced[1]
            acceptance_length_sum = reduced[2]
            proposal_length_sum = reduced[3]
            proposals_at_pos = reduced[4 : 4 + block_size]
            accepted_at_pos = reduced[4 + block_size : 4 + 2 * block_size]

            if proposal_count > 0:
                # DeepSpec build_metrics_row.
                mean_accepted_length = acceptance_length_sum / proposal_count
                draft_tokens_per_proposal = proposal_length_sum / proposal_count
                verify_rate = acceptance_length_sum / (
                    proposal_length_sum + proposal_count
                )
                accept_rate_at_pos = [
                    (
                        (accepted_at_pos[pos] / proposals_at_pos[pos])
                        if proposals_at_pos[pos] > 0
                        else None
                    )
                    for pos in range(block_size)
                ]
            else:
                mean_accepted_length = 0.0
                draft_tokens_per_proposal = 0.0
                verify_rate = 0.0
                accept_rate_at_pos = [None] * block_size

            per_dataset[task_name] = {
                "num_samples": sample_count,
                "num_proposals": proposal_count,
                "mean_accepted_length": mean_accepted_length,
                "draft_tokens_per_proposal": draft_tokens_per_proposal,
                "verify_rate": verify_rate,
                "accept_rate_at_pos": accept_rate_at_pos,
            }
            overall_accept_sum += acceptance_length_sum
            overall_proposal_count += proposal_count
            overall_sample_count += sample_count

            if verbose:
                _print_dataset_row(task_name, per_dataset[task_name])
    finally:
        # Restore draft state so training can resume unaffected.
        if prev_attn is not None:
            draft_model.config._attn_implementation = prev_attn
        draft_model.train(prev_training)

    overall_mean = (
        overall_accept_sum / overall_proposal_count
        if overall_proposal_count > 0
        else 0.0
    )
    result = {
        "config": {
            "block_size": block_size,
            "mask_token_id": int(mask_token_id),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "limit_per_task": limit_per_task,
            "max_prompt_len": max_prompt_len,
            "seed": seed,
            "tasks": [name for name, _ in normalized_tasks],
        },
        "per_dataset": per_dataset,
        "overall": {
            "mean_accepted_length": overall_mean,
            "num_proposals": overall_proposal_count,
            "num_samples": overall_sample_count,
        },
    }

    if verbose:
        _print_summary(result)

    _is_rank0 = (
        not (dist.is_available() and dist.is_initialized())
    ) or dist.get_rank() == 0
    if output_json and _is_rank0:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print_on_rank0(f"[deepspec-eval] wrote metrics to {output_json}")

    return result


# ---------------------------------------------------------------------------
# Pretty printing (rank 0)
# ---------------------------------------------------------------------------
def _print_dataset_row(task_name: str, metrics: Dict[str, object]) -> None:
    print_on_rank0(
        f"[deepspec-eval] {task_name:<16s} "
        f"n={metrics['num_samples']:<5d} "
        f"#prop={metrics['num_proposals']:<7d} "
        f"mean_accept_len={metrics['mean_accepted_length']:.3f} "
        f"verify_rate={metrics['verify_rate']:.4f}"
    )


def _print_summary(result: Dict[str, object]) -> None:
    per_dataset = result["per_dataset"]
    overall = result["overall"]
    lines = []
    lines.append("=" * 78)
    lines.append(
        "DeepSpec speculative-decoding eval  "
        "(mean accepted length, rejection-sampling verify)"
    )
    lines.append("-" * 78)
    lines.append(
        f"{'dataset':<16s} {'n':>5s} {'#prop':>8s} "
        f"{'mean_accept':>12s} {'verify_rate':>12s}"
    )
    for name, metrics in per_dataset.items():
        lines.append(
            f"{name:<16s} {metrics['num_samples']:>5d} "
            f"{metrics['num_proposals']:>8d} "
            f"{metrics['mean_accepted_length']:>12.3f} "
            f"{metrics['verify_rate']:>12.4f}"
        )
    lines.append("-" * 78)
    lines.append(
        f"{'OVERALL':<16s} {overall['num_samples']:>5d} "
        f"{overall['num_proposals']:>8d} "
        f"{overall['mean_accepted_length']:>12.3f}"
    )
    lines.append("-" * 78)
    lines.append("accept_rate@k (per dataset):")
    for name, metrics in per_dataset.items():
        rates = metrics["accept_rate_at_pos"]
        rendered = ", ".join(
            f"{rate:.3f}" if rate is not None else "-" for rate in rates
        )
        lines.append(f"  {name:<16s} [{rendered}]")
    lines.append("=" * 78)
    print_on_rank0("\n".join(lines))


# ---------------------------------------------------------------------------
# Standalone build + CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSpec-style mean-accepted-length eval for a DSpark drafter."
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--target-model-path",
        type=str,
        default="zai-org/GLM-5.2-FP8",
        help="Target model (must be tp-sharded through the sglang backend).",
    )
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="sglang",
        choices=["sglang", "hf"],
        help="Target backend. The 753B FP8 target requires 'sglang' (tp-sharded).",
    )
    model_group.add_argument(
        "--draft-checkpoint",
        type=str,
        required=True,
        help="Path to the trained DSpark draft checkpoint (a directory with "
        "config.json + weights + dspark.py/dflash.py).",
    )
    model_group.add_argument(
        "--draft-attention-backend",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention backend for the draft's per-block eval forward "
        "(flex_attention is unusable here since attention_mask=None).",
    )
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Override the draft's mask_token_id (default: read from its config).",
    )
    model_group.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Override the draft's block_size (default: read from its config).",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Target embedding weight key (default: model.embed_tokens.weight).",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="Target lm_head weight key (default: lm_head.weight).",
    )
    model_group.add_argument("--trust-remote-code", action="store_true", default=True)

    eval_group = parser.add_argument_group("eval")
    eval_group.add_argument(
        "--eval-datasets-dir",
        type=str,
        required=True,
        help="Directory of prebuilt <task>.jsonl files (see module docstring / "
        "deepspec/eval_datasets/convert_eval_datasets_to_jsonl.py).",
    )
    eval_group.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["gsm8k"],
        help="Task names to evaluate (default: gsm8k). Any of the DeepSpec 9 "
        f"are accepted: {', '.join(name for name, _ in DEFAULT_TASKS)}.",
    )
    eval_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max generated tokens per prompt (DeepSpec gsm8k protocol: 2048). "
        "Verification re-prefills, so this is O(n^2); shrink for a quick sweep.",
    )
    eval_group.add_argument(
        "--limit-per-task",
        type=int,
        default=500,
        help="Max prompts per dataset (subsampled deterministically; DeepSpec "
        "gsm8k protocol: 500). Set <=0 to use each task's full upstream cap.",
    )
    eval_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. The verify is DeepSpec's exact "
        "rejection-sampling test at ANY temperature; 0.0 (default) is greedy = "
        "deterministic and bit-exact vs DeepSpec-greedy, while DeepSpec's "
        "stochastic protocol uses 1.0.",
    )
    eval_group.add_argument(
        "--max-prompt-len",
        type=int,
        default=2048,
        help="Skip prompts longer than this many tokens.",
    )
    eval_group.add_argument("--seed", type=int, default=980406)

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write the metrics dict to this JSON path (rank 0).",
    )

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel size for the target model. For the 753B target "
        "set this to the world size (one replica).",
    )

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_everything(args: argparse.Namespace, device: torch.device):
    """Build target (sglang), draft, target embed/lm_head, tokenizer."""
    print_on_rank0(f"[deepspec-eval] loading draft checkpoint: {args.draft_checkpoint}")
    draft_model = DSparkDraftModel.from_pretrained(
        args.draft_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.draft_attention_backend,
    )
    draft_model = draft_model.to(device=device, dtype=torch.bfloat16).eval()
    print_on_rank0(
        f"[deepspec-eval] draft: block_size={draft_model.block_size}, "
        f"target_layer_ids={draft_model.target_layer_ids}, "
        f"mask_token_id={getattr(draft_model, 'mask_token_id', None)}, "
        f"markov_head={draft_model.markov_head is not None}"
    )

    target_kwargs = {}
    if args.target_model_backend == "sglang":
        target_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    print_on_rank0(
        f"[deepspec-eval] loading target ({args.target_model_backend}): "
        f"{args.target_model_path}"
    )
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device.type if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_kwargs,
    )
    target_model.set_capture_layers(list(draft_model.target_layer_ids))

    print_on_rank0("[deepspec-eval] loading target embeddings + lm_head")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device.type,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )

    return target_model, draft_model, target_components, tokenizer


def main() -> None:
    args = parse_args()

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    device = get_local_device()

    target_model, draft_model, target_components, tokenizer = build_everything(
        args, device
    )

    limit_per_task = (
        args.limit_per_task if args.limit_per_task and args.limit_per_task > 0 else None
    )

    run_deepspec_eval(
        target_model=target_model,
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        tokenizer=tokenizer,
        tasks=list(args.tasks),
        eval_datasets_dir=args.eval_datasets_dir,
        block_size=args.block_size,
        mask_token_id=args.mask_token_id,
        max_new_tokens=args.max_new_tokens,
        limit_per_task=limit_per_task,
        temperature=args.temperature,
        max_prompt_len=args.max_prompt_len,
        seed=args.seed,
        device=device,
        output_json=args.output_json,
        verbose=True,
    )

    destroy_distributed()


if __name__ == "__main__":
    main()
