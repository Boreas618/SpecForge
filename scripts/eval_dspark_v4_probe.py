#!/usr/bin/env python3
"""Measure train/accuracy of a DSpark-V4 drafter checkpoint against a DeepSeek-V4
target — used to set the reproduction GOAL from the released drafter, and to
compare our trained drafter against it on the SAME data + objective.

Reports (averaged over --num-batches drawn from the training data cache):
  * DRAFT-vs-DATA acc            — the training "accuracy" metric (all block slots).
  * DRAFT acc per block slot     — position-wise acceptance proxy.
  * DRAFT ce / l1 loss           — training loss components.
  * TEACHER-vs-DATA top1/top5    — the target's own next-token agreement (accuracy
                                   ceiling; the L1 term distills the draft toward it).
  * agree_teacher / top1 probs   — draft-vs-teacher argmax agreement + confidences.

Mirrors the online training forward exactly (OnlineDSparkV4Model over sglang-captured
target hidden states), so the numbers are directly comparable to train/accuracy.
"""
import argparse
import hashlib
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer

from datasets import load_dataset
from safetensors.torch import load_file
from specforge.args import SGLangBackendArgs
from specforge.core.dspark_v4 import OnlineDSparkV4Model
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dspark_v4 import DSparkV4DraftModel, build_dspark_v4_config
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.utils import get_local_device, print_on_rank0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target-model-path", required=True)
    p.add_argument("--target-model-backend", default="sglang")
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--draft-checkpoint", required=True, help="dir with model.safetensors + config.json")
    p.add_argument("--draft-config-path", default=None)
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--chat-template", default="deepseek-v3")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-batches", type=int, default=32)
    p.add_argument("--num-anchors", type=int, default=512)
    p.add_argument("--embedding-key", default="embed.weight")
    p.add_argument("--lm-head-key", default="head.weight")
    p.add_argument("--cache-dir", default="/scratch/SpecForge/cache")
    p.add_argument("--seed", type=int, default=1234)
    SGLangBackendArgs.add_args(p.add_argument_group("sglang"))
    p.add_argument("--dist-timeout", type=int, default=45)
    return p.parse_args()


def main():
    args = parse_args()
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    device = get_local_device()

    cfg_path = args.draft_config_path or os.path.join(args.draft_checkpoint, "config.json")
    cfg = build_dspark_v4_config(json.load(open(cfg_path)))
    # Build the ~40GB draft DIRECTLY on this rank's GPU (each rank holds the full
    # unsharded draft). Building on CPU first (default) would materialize ~40GB x
    # n_ranks of host RAM and, together with the target's CPU weight staging, trip
    # earlyoom (SIGTERM). Weights also stream straight to GPU.
    with torch.device(device):
        draft = DSparkV4DraftModel(cfg)
    draft = draft.to(dtype=torch.bfloat16).eval()
    state = load_file(os.path.join(args.draft_checkpoint, "model.safetensors"), device=str(device))
    missing, unexpected = draft.load_state_dict(state, strict=False)
    del state
    missing = [m for m in missing if not m.startswith("rotary_emb.")]
    print_on_rank0(f"draft loaded; extra-missing={missing} unexpected={unexpected}")

    tkw = SGLangBackendArgs.from_args(args).to_kwargs() if args.target_model_backend == "sglang" else {}
    target = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device.type if args.target_model_backend == "hf" else None,
        **tkw,
    )
    target.set_capture_layers(draft.target_layer_ids)

    heads = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path, embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key, device=device.type,
    )

    wrapper = OnlineDSparkV4Model(
        draft_model=draft, target_lm_head=heads.lm_head,
        target_embed_tokens=heads.embed_tokens, block_size=draft.block_size,
        mask_token_id=draft.mask_token_id, attention_backend="eager",
        num_anchors=args.num_anchors, loss_decay_gamma=4.0,
        ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0,
    ).eval()

    tok = AutoTokenizer.from_pretrained(args.target_model_path)
    key = hashlib.md5(
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}".encode()
    ).hexdigest()
    ds = load_dataset("json", data_files=args.train_data_path)["train"]
    ds = build_eagle3_dataset(
        dataset=ds, tokenizer=tok, chat_template=args.chat_template,
        max_length=args.max_length, cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=key, num_proc=32,
    )
    # num_proc=1: the target (CUDA) is already initialized, so a forked worker pool
    # would hit "Cannot re-initialize CUDA in forked subprocess". Eval set is small.
    ds = ds.filter(lambda x: x["loss_mask"].sum() >= 2 * draft.block_size, num_proc=1)
    dl = prepare_dp_dataloaders(ds, args.batch_size, num_workers=0, shuffle=True, process_group=get_dp_group())

    n_top1 = n_top5 = n_tok = 0
    acc_sum = ce_sum = l1_sum = 0.0
    extras = {}
    acc_pp_sum = cnt_pp_sum = None
    nb = 0
    it = iter(dl)
    for bi in range(args.num_batches):
        try:
            data = next(it)
        except StopIteration:
            break
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        loss_mask = data["loss_mask"].to(device)
        with torch.no_grad():
            tout = target.generate_dflash_data(input_ids, attention_mask, loss_mask)
            hs = tout.hidden_states.to(device)
            lhs = tout.last_hidden_states
            assert lhs is not None, "backend did not surface last_hidden_states"
            lhs = lhs.to(device)

            # teacher-vs-data ceiling
            logits_t = F.linear(lhs[:, :-1], heads.lm_head.weight).float()
            labels = input_ids[:, 1:]
            m = loss_mask[:, 1:] > 0.5
            if m.any():
                top5 = logits_t.topk(5, -1).indices
                n_top1 += ((top5[..., 0] == labels) & m).sum().item()
                n_top5 += ((top5 == labels.unsqueeze(-1)).any(-1) & m).sum().item()
                n_tok += m.sum().item()

            loss, acc, loss_pp, acc_pp, cnt_pp, comps = wrapper(
                input_ids=input_ids, hidden_states=hs, loss_mask=loss_mask,
                last_hidden_states=lhs,
            )
        acc_sum += acc.item(); ce_sum += comps["ce_loss"].item(); l1_sum += comps["l1_loss"].item()
        for e in ("agree_teacher", "teacher_top1_prob", "draft_top1_prob"):
            if e in comps:
                extras[e] = extras.get(e, 0.0) + comps[e].item()
        acc_pp_sum = acc_pp.double() * cnt_pp.double() + (acc_pp_sum if acc_pp_sum is not None else 0)
        cnt_pp_sum = cnt_pp.double() + (cnt_pp_sum if cnt_pp_sum is not None else 0)
        nb += 1
        if dist.get_rank() == 0:
            print(f"batch {bi+1}/{args.num_batches} acc={acc.item():.4f} ce={comps['ce_loss'].item():.4f} l1={comps['l1_loss'].item():.4f}", flush=True)

    if dist.get_rank() == 0 and nb:
        accpp = (acc_pp_sum / cnt_pp_sum.clamp(min=1)).tolist()
        print("=" * 72)
        print(f"DSPARK-V4 PROBE  draft={args.draft_checkpoint}")
        print(f"  target={args.target_model_path}  batches={nb} bs={args.batch_size} anchors={args.num_anchors}")
        print(f"  DRAFT-vs-DATA  train/accuracy : {acc_sum/nb:.4f}")
        print(f"  DRAFT ce_loss                 : {ce_sum/nb:.4f}")
        print(f"  DRAFT l1_loss  (TV={l1_sum/nb/2:.4f}) : {l1_sum/nb:.4f}")
        print(f"  DRAFT acc per block-slot      : {[round(x,4) for x in accpp]}")
        for k, v in extras.items():
            print(f"  {k:28s}: {v/nb:.4f}")
        print(f"  TEACHER-vs-DATA top1/top5     : {n_top1/max(n_tok,1):.4f} / {n_top5/max(n_tok,1):.4f}  (ceiling)")
        print("=" * 72)
    destroy_distributed()


if __name__ == "__main__":
    main()
