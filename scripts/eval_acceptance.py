#!/usr/bin/env python3
# coding=utf-8
"""Evaluate DFlash draft acceptance length on agentic prompts.

Phase C of the staged agentic-DFlash pipeline: quantify the core hypothesis
that a SWE-bench-trained draft accepts more tokens per step on agentic
workloads than a Nemotron-trained one.

For each prompt it runs ``DFlashDraftModel.spec_generate`` (block-wise
speculative decoding against the target) and reads the per-step acceptance
lengths the draft records on ``self._last_acceptance_lengths``. The mean
acceptance length (accepted draft tokens + 1, per verification step) is the
key metric and is ~proportional to the decode speedup.

Prompts are derived from a pretokenized agentic eval JSONL (as produced by
``run_swebench_rollouts.py``): for each row the context preceding its first
supervised token is used as the speculation prompt.

Pass one or more ``--draft-paths`` to compare drafts on identical prompts:

    python scripts/eval_acceptance.py \
        --target-model-path <target-or-proxy> \
        --eval-data-path cache/dataset/swebench-agentic/swebench_agentic_eval.jsonl \
        --draft-paths outputs/...-nemotron/epoch_6_step_X \
                      outputs/...-swebench/epoch_6_step_Y \
        --num-prompts 100 --max-new-tokens 256

Note: ``spec_generate`` drives the target via the HuggingFace CausalLM
interface (``target.model.embed_tokens`` / ``target.lm_head`` /
``output_hidden_states``); use a target whose HF layout matches, or a tractable
proxy. For the full Kimi-K2.7-Code target, measure acceptance via SGLang
speculative decoding instead (production path).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure DFlash acceptance length on agentic prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target-model-path", type=str, required=True)
    p.add_argument(
        "--draft-paths",
        nargs="+",
        required=True,
        help="One or more draft checkpoint dirs to compare on identical prompts.",
    )
    p.add_argument("--eval-data-path", type=str, required=True,
                   help="Pretokenized agentic JSONL (input_ids + loss_mask rows).")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-len", type=int, default=8192,
                   help="Truncate (tail) the derived prompt to this many tokens.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--mask-token-id", type=int, default=None,
                   help="Override draft mask token id (else from draft config).")
    p.add_argument("--stop-token-ids", type=int, nargs="+", default=None,
                   help="Stop token ids (default: tokenizer eos if available).")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_prompts(
    eval_data_path: str, num_prompts: int, max_prompt_len: int
) -> List[List[int]]:
    """Derive speculation prompts = context before each row's first loss token."""
    prompts: List[List[int]] = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            input_ids = row["input_ids"]
            loss_mask = row["loss_mask"]
            first_loss = next((i for i, m in enumerate(loss_mask) if m), None)
            if first_loss is None or first_loss == 0:
                continue
            prompt = input_ids[:first_loss]
            if len(prompt) > max_prompt_len:
                prompt = prompt[-max_prompt_len:]  # keep recent context
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= num_prompts:
                break
    return prompts


@torch.inference_mode()
def evaluate_draft(
    draft_path: str,
    target,
    prompts: List[List[int]],
    args: argparse.Namespace,
    stop_token_ids: Optional[List[int]],
    dtype: torch.dtype,
) -> dict:
    from specforge.modeling.draft.dflash import DFlashDraftModel

    draft = DFlashDraftModel.from_pretrained(draft_path, torch_dtype=dtype)
    draft = draft.to(args.device).eval()
    if args.mask_token_id is not None:
        draft.mask_token_id = args.mask_token_id
    if draft.mask_token_id is None:
        raise ValueError(
            f"draft at {draft_path} has no mask_token_id; pass --mask-token-id."
        )

    all_acc: List[int] = []
    n_prompts = 0
    n_steps = 0
    n_generated = 0
    t0 = time.time()
    for prompt in prompts:
        input_ids = torch.tensor(prompt, dtype=torch.long, device=args.device)[None, :]
        out = draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
        )
        acc = list(getattr(draft, "_last_acceptance_lengths", []) or [])
        if acc:
            all_acc.extend(acc)
            n_steps += len(acc)
            n_generated += sum(acc)
        n_prompts += 1
    elapsed = time.time() - t0

    mean_acc = (sum(all_acc) / len(all_acc)) if all_acc else 0.0
    return {
        "draft_path": draft_path,
        "prompts": n_prompts,
        "verify_steps": n_steps,
        "tokens_generated": n_generated,
        "mean_acceptance_length": mean_acc,
        "tokens_per_sec": (n_generated / elapsed) if elapsed > 0 else 0.0,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_prompts(args.eval_data_path, args.num_prompts, args.max_prompt_len)
    if not prompts:
        print("No prompts derived from eval data; check the JSONL.")
        return 1
    print(f"Loaded {len(prompts)} agentic speculation prompts.")

    stop_token_ids = args.stop_token_ids
    if stop_token_ids is None:
        try:
            tok = AutoTokenizer.from_pretrained(
                args.target_model_path, trust_remote_code=args.trust_remote_code
            )
            if tok.eos_token_id is not None:
                stop_token_ids = [tok.eos_token_id]
        except Exception as e:  # noqa: BLE001
            print(f"  (could not load tokenizer for eos default: {e})")

    print(f"Loading target model: {args.target_model_path}")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        torch_dtype=dtype,
        output_hidden_states=True,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()

    results = []
    for draft_path in args.draft_paths:
        print(f"\nEvaluating draft: {draft_path}")
        res = evaluate_draft(draft_path, target, prompts, args, stop_token_ids, dtype)
        results.append(res)
        print(
            f"  mean_acceptance_length={res['mean_acceptance_length']:.3f} "
            f"over {res['verify_steps']} steps "
            f"({res['tokens_generated']} tok, {res['tokens_per_sec']:.1f} tok/s)"
        )

    print("\n=== Acceptance-length comparison (agentic prompts) ===")
    print(f"{'draft':<60} {'mean_accept':>12} {'tok/s':>10}")
    for res in results:
        name = res["draft_path"]
        if len(name) > 58:
            name = "..." + name[-55:]
        print(
            f"{name:<60} {res['mean_acceptance_length']:>12.3f} "
            f"{res['tokens_per_sec']:>10.1f}"
        )
    if len(results) >= 2:
        base = results[0]["mean_acceptance_length"] or 1e-9
        for res in results[1:]:
            rel = res["mean_acceptance_length"] / base
            print(
                f"  {res['draft_path']} vs {results[0]['draft_path']}: "
                f"{rel:.2f}x acceptance length"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
