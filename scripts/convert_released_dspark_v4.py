#!/usr/bin/env python3
"""Dequantize the released deepseek-ai/DeepSeek-V4-Flash-DSpark mtp.* drafter
(FP4 experts + FP8 linears) into a bf16 SpecForge DSparkV4DraftModel checkpoint.

This is the inverse of ``dspark_v4_state_dict_to_checkpoint`` plus dequantization,
so we can load the CONVERGED released drafter into our exact training model and
measure its train/accuracy as the reproduction goal.

Precision handling (verified against the released safetensors + inference/convert.py):
  * FP8 linears: F8_E4M3 weight + F8_E8M0 (ue8m0) scale, block 128x128 ->
    w = w_fp8.float().reshape(Ob,bo,Ib,bi) * scale.float()[:,None,:,None].
  * FP4 experts: I8-packed float4_e2m1fn_x2 + F8_E8M0 scale, block 32 on the
    (unpacked) input dim -> unpack x2, * per-32 scale. w1(gate)/w3(up) are fused
    into gate_up_proj[e]=cat([w1,w3],0); w2 -> down_proj[e].
  * Norms / hyper-connections / gate / attn_sink / markov / confidence: as-is.
"""
import argparse
import glob
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequant_fp8_block(w, scale):
    """F8_E4M3 [O,I] + F8_E8M0 [Ob,Ib] blockwise -> bf16 [O,I]."""
    O, I = w.shape
    Ob, Ib = scale.shape
    bo, bi = O // Ob, I // Ib
    wf = w.float().reshape(Ob, bo, Ib, bi)
    wf = wf * scale.float()[:, None, :, None]
    return wf.reshape(O, I).bfloat16()


# e2m1 fp4 value table (index = 4-bit code), from released inference/convert.py.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequant_fp4_expert(w_i8, scale):
    """I8 float4_e2m1fn_x2-packed [O, I/2] + F8_E8M0 [O, I/32] -> bf16 [O, I].

    Manual nibble unpack (low then high) via FP4_TABLE — matches the released
    convert.py exactly. (Native .float() on float4_e2m1fn_x2 is unimplemented on CPU.)
    """
    xu = w_i8.view(torch.uint8)
    low = xu & 0x0F
    high = (xu >> 4) & 0x0F
    w = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)  # [O, I]
    O, I = w.shape
    Sb = scale.shape[1]
    bi = I // Sb  # == 32
    wf = w.reshape(O, Sb, bi) * scale.float()[:, :, None]
    return wf.reshape(O, I).bfloat16()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--released-dir", required=True, help="DSpark snapshot dir with mtp shards + index")
    ap.add_argument("--draft-config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from specforge.modeling.draft.dspark_v4 import DSparkV4DraftModel, build_dspark_v4_config

    cfg_dict = json.load(open(args.draft_config))
    cfg = build_dspark_v4_config(cfg_dict)
    num_layers = int(cfg.num_hidden_layers)
    last = num_layers - 1
    moe_inter = int(cfg.moe_intermediate_size)

    # Reference shapes (meta build — no allocation).
    with torch.device("meta"):
        ref = {k: tuple(v.shape) for k, v in DSparkV4DraftModel(cfg).state_dict().items()}

    idx = json.load(open(os.path.join(args.released_dir, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def get(name):
        if name not in idx:
            return None
        fn = idx[name]
        if fn not in handles:
            handles[fn] = safe_open(os.path.join(args.released_dir, fn), framework="pt")
        return handles[fn].get_tensor(name)

    def has(name):
        return name in idx

    def fp8(name):
        return dequant_fp8_block(get(name), get(name.replace(".weight", ".scale")))

    out = {}

    # ---- stage-0-only head: main_proj / main_norm ----
    out["main_proj.weight"] = fp8("mtp.0.main_proj.weight")
    out["main_norm.weight"] = get("mtp.0.main_norm.weight").bfloat16()

    # ---- last-stage head stack ----
    out["norm.weight"] = get(f"mtp.{last}.norm.weight").bfloat16()
    out["hc_head.hc_fn"] = get(f"mtp.{last}.hc_head_fn").float()
    out["hc_head.hc_base"] = get(f"mtp.{last}.hc_head_base").float()
    out["hc_head.hc_scale"] = get(f"mtp.{last}.hc_head_scale").float()
    out["markov_head.markov_w1.weight"] = get(f"mtp.{last}.markov_head.markov_w1.weight").bfloat16()
    out["markov_head.markov_w2.weight"] = get(f"mtp.{last}.markov_head.markov_w2.weight").bfloat16()
    out["confidence_head.proj.weight"] = get(f"mtp.{last}.confidence_head.proj.weight").bfloat16()

    attn_fp8 = {
        "wq_a": "q_a_proj", "wq_b": "q_b_proj", "wkv": "kv_proj",
        "wo_a": "o_a_proj", "wo_b": "o_b_proj",
    }
    for s in range(num_layers):
        p = f"layers.{s}."
        m = f"mtp.{s}."
        # attention (fp8 linears)
        for src, dst in attn_fp8.items():
            out[p + f"self_attn.{dst}.weight"] = fp8(m + f"attn.{src}.weight")
        out[p + "self_attn.q_a_norm.weight"] = get(m + "attn.q_norm.weight").bfloat16()
        out[p + "self_attn.kv_norm.weight"] = get(m + "attn.kv_norm.weight").bfloat16()
        out[p + "self_attn.sinks"] = get(m + "attn.attn_sink").float()
        out[p + "input_layernorm.weight"] = get(m + "attn_norm.weight").bfloat16()
        out[p + "post_attention_layernorm.weight"] = get(m + "ffn_norm.weight").bfloat16()
        # hyper-connections (f32, as-is)
        for hc_src, hc_dst in [("hc_attn", "attn_hc"), ("hc_ffn", "ffn_hc")]:
            for suf in ("fn", "base", "scale"):
                out[p + f"{hc_dst}.{suf}"] = get(m + f"{hc_src}_{suf}").float()
        # MoE gate + shared experts
        out[p + "mlp.gate.weight"] = get(m + "ffn.gate.weight").bfloat16()
        out[p + "mlp.gate.e_score_correction_bias"] = get(m + "ffn.gate.bias").float()
        out[p + "mlp.shared_experts.gate_proj.weight"] = fp8(m + "ffn.shared_experts.w1.weight")
        out[p + "mlp.shared_experts.up_proj.weight"] = fp8(m + "ffn.shared_experts.w3.weight")
        out[p + "mlp.shared_experts.down_proj.weight"] = fp8(m + "ffn.shared_experts.w2.weight")
        # routed experts: fp4 -> fuse
        n_exp = int(cfg.n_routed_experts)
        gate_up = torch.empty(n_exp, 2 * moe_inter, cfg.hidden_size, dtype=torch.bfloat16)
        down = torch.empty(n_exp, cfg.hidden_size, moe_inter, dtype=torch.bfloat16)
        for e in range(n_exp):
            w1 = dequant_fp4_expert(get(m + f"ffn.experts.{e}.w1.weight"), get(m + f"ffn.experts.{e}.w1.scale"))
            w3 = dequant_fp4_expert(get(m + f"ffn.experts.{e}.w3.weight"), get(m + f"ffn.experts.{e}.w3.scale"))
            w2 = dequant_fp4_expert(get(m + f"ffn.experts.{e}.w2.weight"), get(m + f"ffn.experts.{e}.w2.scale"))
            gate_up[e, :moe_inter] = w1
            gate_up[e, moe_inter:] = w3
            down[e] = w2
        out[p + "mlp.experts.gate_up_proj"] = gate_up
        out[p + "mlp.experts.down_proj"] = down
        print(f"stage {s}: converted attn+moe ({n_exp} experts)", flush=True)

    # ---- validate shapes vs reference ----
    bad = []
    for k, v in out.items():
        if k in ref and tuple(v.shape) != ref[k]:
            bad.append((k, tuple(v.shape), ref[k]))
    if bad:
        for k, gots, exps in bad[:20]:
            print(f"SHAPE MISMATCH {k}: got {gots} expected {exps}")
        raise SystemExit(f"{len(bad)} shape mismatches")
    missing_ref = [k for k in ref if k not in out and not k.startswith("rotary_emb.")]
    print(f"tensors converted: {len(out)}; ref (non-rotary) not produced: {missing_ref}")

    os.makedirs(args.out_dir, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(args.out_dir, "model.safetensors"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"saved bf16 released drafter -> {args.out_dir}/model.safetensors")


if __name__ == "__main__":
    main()
