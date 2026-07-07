#!/usr/bin/env python3
"""Isolate the DSpark-V4 draft fwd/bwd bottleneck on a single GPU (no sglang, no FSDP).
Feeds synthetic target hidden states so we can iterate fast on the ~28s backward."""
import json, os, time
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

from specforge.modeling.draft.dspark_v4 import DSparkV4DraftModel, build_dspark_v4_config
from specforge.core.dspark_v4 import OnlineDSparkV4Model

dev = "cuda:0"
cfg = build_dspark_v4_config(json.load(open("/scratch/SpecForge/configs/deepseek-v4-flash-dspark.json")))
if os.environ.get("SPECFORGE_NO_GC") == "1":
    cfg.gradient_checkpointing = False
_impl = os.environ.get("SPECFORGE_EXPERTS_IMPL")
if _impl:
    cfg._experts_implementation = _impl
    print(f"experts_implementation = {_impl}")
with torch.device(dev):
    draft = DSparkV4DraftModel(cfg)
draft = draft.to(torch.bfloat16).train()
H, V = cfg.hidden_size, cfg.vocab_size
lm_head = nn.Linear(H, V, bias=False).to(dev, torch.bfloat16).requires_grad_(False)
embed = nn.Embedding(V, H).to(dev, torch.bfloat16).requires_grad_(False)
wrap = OnlineDSparkV4Model(
    draft_model=draft, target_lm_head=lm_head, target_embed_tokens=embed,
    mask_token_id=draft.mask_token_id, block_size=draft.block_size,
    attention_backend="eager", num_anchors=int(os.environ.get("NUM_ANCHORS", 512)),
    loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0,
).train()

B = int(os.environ.get("BS", 4)); S = int(os.environ.get("S", 4096))
torch.manual_seed(0)
input_ids = torch.randint(0, V, (B, S), device=dev)
K = len(draft.target_layer_ids)
hidden = torch.randn(B, S, K * H, device=dev, dtype=torch.bfloat16)
last = torch.randn(B, S, H, device=dev, dtype=torch.bfloat16)
loss_mask = torch.zeros(B, S, device=dev); loss_mask[:, S // 4:] = 1.0  # supervised tail

def stepfn():
    out = wrap(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, last_hidden_states=last)
    loss = out[0]
    loss.backward()
    for p in draft.parameters():
        p.grad = None
    return loss

import os as _os
print("DRAFT_ATTN =", _os.environ.get("SPECFORGE_DRAFT_ATTN", "flex"))
with torch.no_grad():
    _l = wrap(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, last_hidden_states=last)[0]
    print(f"LOSS (eval, seed-fixed) = {_l.item():.6f}")

print("params:", sum(p.numel() for p in draft.parameters()) / 1e9, "B", "| GC:", cfg.gradient_checkpointing)
for _ in range(2):
    stepfn()
torch.cuda.synchronize()

# time fwd / bwd separately
out = wrap(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, last_hidden_states=last)
loss = out[0]; torch.cuda.synchronize()
t = time.time()
out = wrap(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, last_hidden_states=last)
loss = out[0]; torch.cuda.synchronize(); fwd = time.time() - t
t = time.time(); loss.backward(); torch.cuda.synchronize(); bwd = time.time() - t
for p in draft.parameters(): p.grad = None
print(f"FWD {fwd:.2f}s  BWD {bwd:.2f}s")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    stepfn(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
