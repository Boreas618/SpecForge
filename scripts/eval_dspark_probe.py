"""Diagnostic probe for DSpark runs: teacher-data agreement, draft eval, feature stats.

Answers, for a given (target, draft checkpoint, data cache):
  1. teacher-vs-data: top-1/top-5 agreement of the TARGET's next-token argmax with
     the data tokens on loss-masked positions (upper bound / "ceiling" for draft
     accuracy, since the L1 term distills the draft toward the target).
  2. thinking-token check: the target's top-3 preferred tokens at the FIRST
     supervised position of each assistant span.
  3. draft-vs-data acc + l1 (re-measured on fixed batches) + per-position acc.
  4. context-feature stats: per-captured-layer norms and adjacent-layer cosine
     similarity (detects a degenerate/washed-out feature convention).
"""

import argparse
import hashlib
import os
from collections import Counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs
from specforge.core.dspark import OnlineDSparkModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.utils import get_local_device, print_on_rank0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", type=str, required=True)
    p.add_argument("--target-model-backend", type=str, default="hf")
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--draft-checkpoint", type=str, required=True)
    p.add_argument("--train-data-path", type=str, required=True)
    p.add_argument("--chat-template", type=str, required=True)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=24)
    p.add_argument("--mask-token-id", type=int, required=True)
    p.add_argument("--embedding-key", type=str, default=None)
    p.add_argument("--lm-head-key", type=str, default=None)
    p.add_argument("--attention-backend", type=str, default="sdpa")
    p.add_argument("--cache-dir", type=str, default="/personal/SpecForge/cache")
    SGLangBackendArgs.add_args(p.add_argument_group("sglang"))
    p.add_argument("--dist-timeout", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    device = get_local_device()

    # --- draft from checkpoint ---
    cfg = AutoConfig.from_pretrained(args.draft_checkpoint)
    cfg._attn_implementation = args.attention_backend
    draft = DSparkDraftModel(cfg)
    from safetensors.torch import load_file

    state = load_file(os.path.join(args.draft_checkpoint, "model.safetensors"))
    missing, unexpected = draft.load_state_dict(state, strict=False)
    print_on_rank0(f"draft loaded; missing={missing} unexpected={unexpected}")
    draft = draft.to(device=device, dtype=torch.bfloat16).eval()

    # --- target ---
    tkw = {}
    if args.target_model_backend == "sglang":
        tkw = SGLangBackendArgs.from_args(args).to_kwargs()
    target = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device.type if args.target_model_backend == "hf" else None,
        trust_remote_code=True,
        **tkw,
    )
    target.set_capture_layers(draft.target_layer_ids)

    heads = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device.type,
        trust_remote_code=True,
    )

    wrapper = OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=heads.lm_head,
        target_embed_tokens=heads.embed_tokens,
        mask_token_id=args.mask_token_id,
        block_size=draft.block_size,
        attention_backend=args.attention_backend,
        num_anchors=512,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    ).eval()

    tok = AutoTokenizer.from_pretrained(args.target_model_path)

    # --- data (same cache as training) ---
    key = hashlib.md5(
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}".encode()
    ).hexdigest()
    ds = load_dataset("json", data_files=args.train_data_path)["train"]
    ds = build_eagle3_dataset(
        dataset=ds,
        tokenizer=tok,
        chat_template=args.chat_template,
        max_length=args.max_length,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=key,
        num_proc=32,
    )
    dl = prepare_dp_dataloaders(
        ds, args.batch_size, num_workers=0, shuffle=True, process_group=get_dp_group()
    )

    n_top1 = n_top5 = n_tok = 0
    draft_acc_sum = draft_l1_sum = draft_ce_sum = 0.0
    extras_sum = {}
    acc_pp_sum = None
    cnt_pp_sum = None
    span_start_top1 = Counter()
    hdim = cfg.hidden_size
    k = len(draft.target_layer_ids)
    cos_adj = torch.zeros(k - 1, dtype=torch.float64)
    cos_n = 0
    chunk_norms = torch.zeros(k, dtype=torch.float64)

    it = iter(dl)
    for bi in range(args.num_batches):
        data = next(it)
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        loss_mask = data["loss_mask"].to(device)
        with torch.no_grad():
            tout = target.generate_dflash_data(input_ids, attention_mask, loss_mask)
            hs = tout.hidden_states.to(device)
            lhs = tout.last_hidden_states
            assert lhs is not None, "no last_hidden_states from backend"
            lhs = lhs.to(device)

            # 1) teacher-vs-data
            logits_t = F.linear(lhs[:, :-1], heads.lm_head.weight).float()
            labels = input_ids[:, 1:]
            m = loss_mask[:, 1:] > 0.5
            if m.any():
                top5 = logits_t.topk(5, dim=-1).indices
                eq1 = (top5[..., 0] == labels) & m
                eq5 = (top5 == labels.unsqueeze(-1)).any(-1) & m
                n_top1 += eq1.sum().item()
                n_top5 += eq5.sum().item()
                n_tok += m.sum().item()

            # 2) span starts: first supervised position per row
            for r in range(input_ids.shape[0]):
                idx = (loss_mask[r] > 0.5).nonzero()
                if len(idx) == 0:
                    continue
                t0 = idx[0, 0].item()
                if t0 == 0:
                    continue
                tt = logits_t[r, t0 - 1].topk(3).indices.tolist()
                span_start_top1[tok.decode([tt[0]])] += 1

            # 3) draft eval (training forward, no grad)
            (loss, acc, loss_pp, acc_pp, cnt_pp, comps) = wrapper(
                input_ids=input_ids,
                hidden_states=hs,
                loss_mask=loss_mask,
                last_hidden_states=lhs,
            )
            draft_acc_sum += acc.item()
            draft_l1_sum += comps["l1_loss"].item()
            draft_ce_sum += comps["ce_loss"].item()
            for extra in ("agree_teacher", "teacher_top1_prob", "draft_top1_prob"):
                if extra in comps:
                    extras_sum[extra] = extras_sum.get(extra, 0.0) + comps[extra].item()
            acc_pp_sum = acc_pp.double() * cnt_pp.double() + (
                acc_pp_sum if acc_pp_sum is not None else 0
            )
            cnt_pp_sum = cnt_pp.double() + (cnt_pp_sum if cnt_pp_sum is not None else 0)

            # 4) feature stats on masked tokens (sample up to 2k)
            flat_m = (loss_mask > 0.5).reshape(-1)
            feats = hs.reshape(-1, k * hdim)[flat_m][:2000].float()
            if feats.numel():
                ch = feats.view(-1, k, hdim)
                chunk_norms += ch.norm(dim=-1).mean(0).double().cpu()
                cn = F.normalize(ch, dim=-1)
                for j in range(k - 1):
                    cos_adj[j] += (cn[:, j] * cn[:, j + 1]).sum(-1).mean().double().cpu()
                cos_n += 1
        if dist.get_rank() == 0 and (bi + 1) % 8 == 0:
            print(f"batch {bi+1}/{args.num_batches}", flush=True)

    if dist.get_rank() == 0:
        nb = args.num_batches
        print("=" * 70)
        print(f"PROBE RESULTS target={args.target_model_path}")
        print(f"  masked tokens evaluated: {n_tok}")
        print(f"  TEACHER-vs-DATA  top1: {n_top1/max(n_tok,1):.4f}  top5: {n_top5/max(n_tok,1):.4f}")
        print(f"  DRAFT-vs-DATA    acc:  {draft_acc_sum/nb:.4f}")
        print(f"  DRAFT-vs-TEACHER l1:   {draft_l1_sum/nb:.4f}  (TV={draft_l1_sum/nb/2:.4f})")
        print(f"  DRAFT ce:              {draft_ce_sum/nb:.4f}")
        for kk, vv in extras_sum.items():
            print(f"  {kk}: {vv/nb:.4f}")
        accpp = (acc_pp_sum / cnt_pp_sum.clamp(min=1)).tolist()
        print(f"  draft acc per block-slot: {[round(x,4) for x in accpp]}")
        print(f"  span-start teacher top1 tokens: {span_start_top1.most_common(8)}")
        print(f"  ctx-feature mean norms per layer: {[round(x,1) for x in (chunk_norms/nb).tolist()]}")
        print(f"  adjacent-layer feature cosine:    {[round(x,3) for x in (cos_adj/max(cos_n,1)).tolist()]}")
        print("=" * 70)

    destroy_distributed()


if __name__ == "__main__":
    main()
