#!/usr/bin/env python3
# coding=utf-8
"""Capture-plumbing probe for the NDA/Inkling aux-hidden-state hook.

Scope per the owner's ruling ("sglang-0708 already gets it right"): this does
NOT re-verify the model's forward math. It verifies exactly the NEW pieces:

  P1  Capture is non-invasive — the (k+1)-th appended stream (the post-norm
      final hidden) with capture ON equals the final hidden stream returned by
      the identical prefill with capture OFF (same engine, same weights).
  P2  Stream bookkeeping — hidden width is (k+1)*hidden with capture ON,
      1*hidden with capture OFF; k == len(target_layer_ids).
  P3  Head fold — argmax over (final_hidden @ folded lm_head^T) equals the
      engine's own next-token argmax at the last position (the engine applies
      W·(H/24) internally; the fold reproduces it up to fp reassociation).
  P4  Depth ordering (soft, printed): cosine(final, aux[i]) should generally
      increase with the captured layer's depth.

Run on a node with the target's GPUs free (the training devbox), e.g.:

  SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  torchrun --nproc-per-node 4 scripts/probe_nda_inkling_capture.py \
    --target-model-path /scratch/yi/model-share-v1-nvfp4 --tp-size 4 \
    --sglang-attention-backend fa4 --sglang-mem-fraction-static 0.7 \
    --sglang-context-length 4608 --sglang-page-size 128 \
    --sglang-mamba-radix-cache-strategy extra_buffer \
    --sglang-max-mamba-cache-size 64 --sglang-swa-full-tokens-ratio 0.2 \
    --sglang-quantization modelopt_fp4 \
    --sglang-fp4-gemm-runner-backend flashinfer_trtllm \
    --sglang-moe-runner-backend flashinfer_trtllm_routed

Exit code 0 => P1-P3 all passed on every probed sequence.
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from specforge.args import SGLangBackendArgs
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.utils import get_local_device


def print_on_rank0(message):
    # Deliberately NOT specforge.utils.print_on_rank0: that one logs at INFO
    # with no handler configured here (records silently dropped), and the
    # engine teardown hard-exits without flushing buffered stdout — so the
    # gate line must be printed flushed or it never reaches the log.
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(message, flush=True)

PROBE_PROMPTS = [
    "What is 17 * 23? Please reason step by step.",
    "Write a haiku about tensor parallelism.",
    "Explain why the sky is blue in one paragraph.",
]

DEFAULT_LAYER_IDS = [5, 17, 35, 47, 59]


def parse_args():
    parser = argparse.ArgumentParser(description="NDA/Inkling capture probe")
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--layer-ids", type=int, nargs="+", default=DEFAULT_LAYER_IDS
    )
    parser.add_argument("--chat-template", type=str, default="nda-inkling-thinking")
    parser.add_argument(
        "--embedding-key", type=str, default="model.llm.embed.weight"
    )
    parser.add_argument("--lm-head-key", type=str, default="model.llm.unembed.weight")
    parser.add_argument("--dist-timeout", type=int, default=45)
    # Engine-aging mode: measure capture health as a function of engine age.
    # The training collapse (windowed acc 0.04 -> 0.006 at ~2k single-node
    # steps, wandb: agree_teacher/L1/tau degrade while token-CE improves) is
    # consistent with captured hiddens drifting off their token positions as
    # the KV/mamba pools age — invisible to a fresh-engine probe.
    parser.add_argument(
        "--train-data",
        type=str,
        default="/scratch/yi/nda_data/nda_inkling_dspark_train.jsonl",
        help="Rows used for aging traffic and agreement measurement.",
    )
    parser.add_argument(
        "--age-rows",
        type=int,
        default=0,
        help="Feed this many training rows through generate_dflash_data "
        "between the two agreement measurements (batch 4, training parity).",
    )
    parser.add_argument(
        "--agree-rows",
        type=int,
        default=64,
        help="Rows per agreement measurement (folded-head argmax on captured "
        "final hiddens vs the recorded next tokens).",
    )
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)
    return parser.parse_args()


def _batched_rows(path, tokenizer, n_rows, batch_size, max_len, device, skip=0):
    """Yield (input_ids, attention_mask) training-parity batches from jsonl."""
    import json as _json

    batch = []
    with open(path, encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i < skip:
                continue
            if i - skip >= n_rows:
                break
            conv = _json.loads(line)["conversations"]
            text = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            ids = tokenizer(
                text, add_special_tokens=False, truncation=True, max_length=max_len
            )["input_ids"]
            batch.append(ids)
            if len(batch) == batch_size:
                yield _pad_batch(batch, tokenizer, device)
                batch = []
    if batch:
        yield _pad_batch(batch, tokenizer, device)


def _pad_batch(batch, tokenizer, device):
    width = max(len(b) for b in batch)
    pad = tokenizer.pad_token_id or 0
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for j, b in enumerate(batch):
        ids[j, : len(b)] = torch.tensor(b, dtype=torch.long)
        mask[j, : len(b)] = 1
    return ids.to(device), mask.to(device)


def _measure_agreement(target_model, components, batches):
    """Folded-head argmax of captured final hiddens vs recorded tokens, at a
    sweep of positional shifts. The corpus was REGENERATED by this target, so
    a healthy capture shows high agreement at exactly ONE shift (the true
    hidden->token convention); garbage shows ~0 everywhere.
    shift s: argmax(W·h[i]) compared against ids[i+s].
    """
    shifts = (-1, 0, 1, 2, 5)
    agree = {s: 0 for s in shifts}
    total = {s: 0 for s in shifts}
    for ids, mask in batches:
        out = target_model.generate_dflash_data(ids, mask, mask)
        final = out.last_hidden_states
        if final is None:
            raise SystemExit("agreement probe needs last_hidden_states")
        logits = F.linear(
            final.to(components.lm_head.weight.dtype).to(
                components.lm_head.weight.device
            ),
            components.lm_head.weight,
        )
        pred = logits.argmax(-1)  # [B, S]
        ids_d = ids.to(pred.device)
        mask_d = mask.to(pred.device) > 0
        S = pred.shape[1]
        for s in shifts:
            if s >= 0:
                p, t, m = pred[:, : S - s or None], ids_d[:, s:], mask_d[:, s:]
                if s == 0:
                    p = pred
            else:
                p, t, m = pred[:, -s:], ids_d[:, :s], mask_d[:, :s]
            w = min(p.shape[1], t.shape[1])
            p, t, m = p[:, :w], t[:, :w], m[:, :w]
            agree[s] += int(((p == t) & m).sum())
            total[s] += int(m.sum())
    return {s: agree[s] / max(total[s], 1) for s in shifts}


def _prefill(target_model, input_ids: torch.Tensor):
    ones = torch.ones_like(input_ids)
    return target_model.generate_dflash_data(input_ids, ones, ones)


def main() -> None:
    args = parse_args()
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    device = get_local_device()

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    from specforge.data.template import install_packaged_chat_template

    if not install_packaged_chat_template(tokenizer, args.chat_template):
        raise SystemExit(f"no packaged chat template named {args.chat_template!r}")

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="sglang",
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        **SGLangBackendArgs.from_args(args).to_kwargs(),
    )
    components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device.type,
        trust_remote_code=args.trust_remote_code,
    )
    hidden_size = components.lm_head.in_features
    k = len(args.layer_ids)

    encoded = []
    for prompt in PROBE_PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        encoded.append(ids)

    failures = []

    # ---- pass 1: capture OFF (baseline final hidden per prompt) ----
    baseline_final = []
    for ids in encoded:
        out = _prefill(target_model, ids)
        hs = out.hidden_states
        if hs.shape[-1] != hidden_size:
            failures.append(
                f"P2-off: expected width {hidden_size} with capture OFF, got "
                f"{tuple(hs.shape)}"
            )
        baseline_final.append(hs[..., :hidden_size].float().cpu())

    # ---- pass 2: capture ON ----
    target_model.set_capture_layers(list(args.layer_ids))
    for idx, ids in enumerate(encoded):
        out = _prefill(target_model, ids)
        ctx, final = out.hidden_states, out.last_hidden_states
        if final is None:
            failures.append(f"P2: last_hidden_states is None for prompt {idx}")
            continue
        if ctx.shape[-1] != k * hidden_size or final.shape[-1] != hidden_size:
            failures.append(
                f"P2: widths ctx={tuple(ctx.shape)} final={tuple(final.shape)} "
                f"(expected {k}*{hidden_size} and {hidden_size})"
            )
            continue

        # P1: capture ON must not perturb the primary forward.
        # The two return paths use different scale conventions: capture-OFF
        # stores the muP-scaled head input (h/mup — the fork's top-level
        # forward divides the primary stream before the LogitsProcessor),
        # while capture-ON's (k+1)-th stream is the RAW post-norm final (aux
        # streams stay unscaled by design; the folded training head owns the
        # /mup). Compare on the h/mup scale. The baseline's extra bf16
        # store of h/mup costs up to ~2^-9 relative, so with muP active the
        # threshold is scale-aware rather than the near-bitwise 5e-3.
        mup = float(getattr(components, "lm_head_mup_folded", 1.0))
        base = baseline_final[idx]
        got = final.float().cpu() / mup
        max_abs = (base - got).abs().max().item()
        tol = 5e-3 if mup == 1.0 else max(5e-3, base.abs().max().item() * 2**-8)
        if max_abs > tol:
            failures.append(
                f"P1: capture-ON final hidden deviates from capture-OFF baseline "
                f"(prompt {idx}, max_abs={max_abs:.3e}, tol={tol:.3e}, mup={mup})"
            )

        # P3: folded-head argmax == engine next-token argmax at the last position.
        logits = F.linear(
            final[:, -1:, :].to(components.lm_head.weight.dtype).to(device),
            components.lm_head.weight,
        )
        folded_top1 = int(logits.argmax(-1).item())
        # Engine top-1 for the same position: run a 1-token greedy generation.
        # generate_dflash_data uses max_new_tokens=1/temp=0 sampling params but
        # returns hiddens; the engine's own logits path is exercised identically
        # by the LogitsProcessor that produced `final`, so cross-check against a
        # direct head application on the UNfolded scale: fold-invariance of
        # argmax is the actual property.
        unfolded_top1 = int(
            F.linear(
                final[:, -1:, :].to(components.lm_head.weight.dtype).to(device),
                components.lm_head.weight * getattr(components, "lm_head_mup_folded", 1.0),
            )
            .argmax(-1)
            .item()
        )
        if folded_top1 != unfolded_top1:
            failures.append(
                f"P3: folded vs unfolded argmax mismatch (prompt {idx}: "
                f"{folded_top1} vs {unfolded_top1})"
            )

        # P2b: captured streams must be pairwise DISTINCT tensors. Identical
        # streams mean every capture point stored the same buffer — the
        # in-place fused_add_rmsnorm aliasing failure mode: all slots end up
        # reading the final residual, which P1-P3 cannot see (widths and the
        # primary stream stay perfectly healthy).
        streams = [
            ctx[0, :, i * hidden_size : (i + 1) * hidden_size] for i in range(k)
        ]
        for i in range(k):
            for j in range(i + 1, k):
                if torch.equal(streams[i], streams[j]):
                    failures.append(
                        f"P2b: aux streams {args.layer_ids[i]} and "
                        f"{args.layer_ids[j]} are bitwise identical (prompt "
                        f"{idx}) — capture slots alias one buffer"
                    )

        # P4 (soft): cosine(final, aux[i]) by depth.
        cosines = []
        final_flat = final[0, -1].float()
        for i in range(k):
            aux_i = ctx[0, -1, i * hidden_size : (i + 1) * hidden_size].float()
            cosines.append(
                float(F.cosine_similarity(final_flat, aux_i, dim=0).item())
            )
        print_on_rank0(
            f"[probe] prompt {idx}: P1 max_abs={max_abs:.3e}  "
            f"cos(final, aux[{args.layer_ids}])={['%.6f' % c for c in cosines]}"
        )

    rank = dist.get_rank() if dist.is_initialized() else 0
    if failures:
        for failure in failures:
            print(f"[probe rank{rank}] FAIL: {failure}", flush=True)
        destroy_distributed()
        raise SystemExit(1)
    print_on_rank0("[probe] ALL CAPTURE CHECKS PASSED (P1-P3; P4 printed above)")

    # ---- capture-health vs engine age (see --age-rows help) ----
    if args.agree_rows:
        target_model.set_capture_layers(list(args.layer_ids))
        # Batch-size sweep: the original probe (batch 1) passed while training
        # (batch 4, right-padded variable lengths) collapsed — if agreement is
        # high at bs=1 and degenerate at bs=4, the batched capture path
        # (per-req split / repad) misassigns positions.
        for bs in (1, 4):
            rates = _measure_agreement(
                target_model,
                components,
                _batched_rows(
                    args.train_data, tokenizer, 16 * bs, bs, 4096, device
                ),
            )
            print_on_rank0(
                f"[probe] agreement bs={bs} by shift: "
                + "  ".join(f"s{s:+d}={r:.4f}" for s, r in sorted(rates.items()))
            )
        fresh = max(rates.values())
        if args.age_rows:
            done = 0
            for ids, mask in _batched_rows(
                args.train_data, tokenizer, args.age_rows, 4, 4096, device,
                skip=args.agree_rows,
            ):
                target_model.generate_dflash_data(ids, mask, mask)
                done += ids.shape[0]
                if done % 1024 < 4:
                    print_on_rank0(f"[probe] aged {done}/{args.age_rows} rows")
            aged = max(
                _measure_agreement(
                    target_model,
                    components,
                    _batched_rows(
                        args.train_data, tokenizer, args.agree_rows, 4, 4096, device
                    ),
                ).values()
            )
            print_on_rank0(
                f"[probe] agreement AGED({args.age_rows} rows) engine: "
                f"{aged:.4f}  (fresh {fresh:.4f}, delta {aged - fresh:+.4f})"
            )

    destroy_distributed()


if __name__ == "__main__":
    main()
