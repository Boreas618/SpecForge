# coding=utf-8
"""Faithful DeepSeek-V4-Flash-DSpark draft model (1:1 with the released drafter).

This is a from-scratch-trainable, fully differentiable reconstruction of the
``mtp.*`` draft that ships inside ``deepseek-ai/DeepSeek-V4-Flash-DSpark`` and is
served by vLLM's ``DSparkDeepseekV4ForCausalLM``. Every module, parameter shape,
and weight name matches the released checkpoint, so a trained checkpoint saved
via :func:`dspark_v4_state_dict_to_checkpoint` drops straight into vLLM.

Architecture (verified against the released ``config.json`` + safetensors index):
  * 3 ``mtp`` blocks, each a full DeepSeek-V4 decoder layer reused verbatim from
    ``transformers`` (MLA shared-KV attention + Manifold-Constrained Hyper-
    Connections + 256-expert top-k ``noaux_tc`` MoE with 1 shared expert). The
    released draft blocks are all ``sliding_attention`` (no CSA/HCA compressor)
    and standard (non-hash) MoE — confirmed by the absence of ``compressor.*`` /
    ``tid2eid`` and the presence of ``ffn.gate.bias`` in the checkpoint.
  * ``main_proj`` / ``main_norm``: fuse the concatenated target aux hidden states
    (layers ``dspark_target_layer_ids = [40, 41, 42]``) into ``main_x``.
  * learned ``hc_head`` (:class:`DeepseekV4HyperHead`) collapsing the hc streams,
    then a final ``norm`` — NOT a mean-pool.
  * Markov head (low-rank bigram bias) + confidence head (accept-rate predictor).

Conditioning (the load-bearing faithfulness point): at inference each draft layer
derives its sliding-window context KV from the SAME fixed ``main_x`` via that
layer's own ``kv`` projection (``precompute_and_store_context_kv``), and the draft
block queries attend to that context plus their own block. That is exactly the
DFlash dual-source-KV pattern — context K/V re-derived from ``main_x`` at every
layer, NOT the target features run through the decoder stack. We reproduce it here
with :class:`DSparkV4Attention` (context K/V from ``main_x``, query/noise K/V from
the block stream), so the trained network matches the served forward.
"""

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Experts,
    DeepseekV4HashRouter,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
    DeepseekV4TopKRouter,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from torch.nn.attention.flex_attention import BlockMask  # noqa: E402

from specforge.modeling.draft.dspark import VanillaMarkov

# DeepSeek-V4 draft attention has one shared KV head. We expand that KV head as a
# zero-stride view before flex_attention instead of using enable_gqa=True because
# PyTorch 2.9's flex GQA specialization miscompiles this 64q/1kv, head_dim=512 case
# on SM100. This keeps block-sparse flex attention without materializing dense scores.
_FLEX_KERNEL_OPTIONS = {
    "BLOCK_M": 32, "BLOCK_N": 32,
    "BLOCK_M1": 32, "BLOCK_N1": 32,
    "BLOCK_M2": 32, "BLOCK_N2": 32,
}
# dynamic=True: the draft query length Q and context S vary per batch (data-dependent
# num_anchors); without it torch.compile re-traces flex on every new shape (GPU stalls
# in inductor each step). Dynamic shapes compile once and reuse.
_flex_attention_compiled = torch.compile(flex_attention, dynamic=True)


class AcceptRateHead(nn.Module):
    """Per-position accept-rate predictor (single bias-free linear).

    Matches the released ``mtp.*.confidence_head.proj.weight`` (no bias). Fused
    with the Markov prev-token embedding when ``confidence_head_with_markov``.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


class DSparkV4Attention(DeepseekV4Attention):
    """DeepSeek-V4 shared-KV MLA in DSpark dual-source mode (sliding attention).

    Reuses the released attention parameters verbatim (``q_a_proj``/``q_a_norm``/
    ``q_b_proj``/``q_b_norm``/``kv_proj``/``kv_norm``/``o_a_proj``/``o_b_proj``/
    ``sinks``) and the exact eager math (per-head sink, partial interleaved RoPE,
    conjugate-RoPE on the output, grouped output projection). The only change vs
    :class:`DeepseekV4Attention` is dual-source KV:

      * context K/V come from ``context_hidden`` (``main_x``, shared by all layers,
        NOT layernorm'd) — mirrors ``precompute_and_store_context_kv``;
      * query + noise K/V come from the block stream ``hidden_states``.

    K == V (shared-KV MQA); the block/context split is handled by ``attention_mask``.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, Q, H]  input_layernorm'd block stream
        context_hidden: torch.Tensor,     # [B, S, H]  main_x (context features)
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        cos_ctx: torch.Tensor,
        sin_ctx: torch.Tensor,
        attention_mask,                   # BlockMask (flex) or additive [B,1,Q,S+Q] (eager)
    ) -> torch.Tensor:
        B, Q, _ = hidden_states.shape
        S = context_hidden.shape[1]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(B, Q, -1, self.head_dim).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)

        # Context K/V from main_x (single shared KV head), RoPE at context positions.
        kv_ctx = self.kv_norm(self.kv_proj(context_hidden))
        kv_ctx = kv_ctx.view(B, S, -1, self.head_dim).transpose(1, 2)
        kv_ctx = apply_rotary_pos_emb(kv_ctx, cos_ctx, sin_ctx)
        # Query/noise K/V from the block stream, RoPE at draft positions.
        kv_noise = self.kv_norm(self.kv_proj(hidden_states))
        kv_noise = kv_noise.view(B, Q, -1, self.head_dim).transpose(1, 2)
        kv_noise = apply_rotary_pos_emb(kv_noise, cos_q, sin_q)
        kv = torch.cat([kv_ctx, kv_noise], dim=2)  # [B, 1, S+Q, D], K == V

        if isinstance(attention_mask, BlockMask):
            # Block-sparse flex attention: never materializes the [B,64,Q,S+Q] score
            # matrix, so cost tracks the ~sliding_window keys actually attended, not S.
            # K==V, with the single shared KV head expanded as a zero-stride view to
            # avoid the broken flex GQA specialization on this stack.
            kv_flex = kv.expand(-1, q.shape[1], -1, -1)
            attn_out, lse = _flex_attention_compiled(
                q,
                kv_flex,
                kv_flex,
                block_mask=attention_mask,
                scale=self.scaling,
                return_lse=True,
                kernel_options=_FLEX_KERNEL_OPTIONS,
            )  # attn_out [B, num_heads, Q, D], lse [B, num_heads, Q]
            # Per-head attention sink: eager appends a valueless sink logit as an extra
            # softmax column, i.e. a post-hoc rescale by sigmoid(lse - sink) (exact;
            # verified fp32 max-diff 6e-7). Keep lse attached so grads reach self.sinks.
            corr = torch.sigmoid(lse - self.sinks.float().view(1, -1, 1)).to(attn_out.dtype)
            attn_output = (attn_out * corr.unsqueeze(-1)).transpose(1, 2)  # [B, Q, num_heads, D]
        else:
            attn_output, _ = eager_attention_forward(
                self,
                q,
                kv,
                kv,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )  # [B, Q, num_heads, D]

        # K==V picked up RoPE on its trailing slice; undo it on the output rope
        # slice with the conjugate rotation (-sin) at the query (draft) positions.
        attn_output = apply_rotary_pos_emb(
            attn_output.transpose(1, 2), cos_q, -sin_q
        ).transpose(1, 2)

        grouped = attn_output.reshape(B, Q, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped)


class DSparkV4Block(nn.Module):
    """One DSpark ``mtp`` block: DeepSeek-V4 decoder layer with dual-source MLA.

    Identical residual math to :class:`DeepseekV4DecoderLayer` (two mHC sites, one
    around attention, one around the MoE), only the attention is dual-source. The
    submodule names (``self_attn`` / ``mlp`` / ``input_layernorm`` /
    ``post_attention_layernorm`` / ``attn_hc`` / ``ffn_hc``) map 1:1 to the
    released ``mtp.{i}.*`` weights.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DSparkV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, Q, hc_mult, H]
        context_hidden: torch.Tensor,     # [B, S, H]
        cos_q: torch.Tensor,
        sin_q: torch.Tensor,
        cos_ctx: torch.Tensor,
        sin_ctx: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        # mHC around attention: collapse hc streams -> attn -> post/comb recombine.
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(
            self.input_layernorm(collapsed),
            context_hidden,
            cos_q,
            sin_q,
            cos_ctx,
            sin_ctx,
            attention_mask,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )
        # mHC around the MoE.
        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=None)
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


class DSparkV4DraftModel(nn.Module):
    """DeepSeek-V4-Flash-DSpark draft network (faithful 1:1 reconstruction)."""

    _no_split_modules = ["DSparkV4Block"]

    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_size
        self.hidden_size = H
        self.hc_mult = int(config.hc_mult)
        self.num_layers = int(config.num_hidden_layers)
        self.block_size = int(config.dspark_block_size)
        self.target_layer_ids = list(config.dspark_target_layer_ids)
        self.mask_token_id = int(config.dspark_noise_token_id)
        self.sliding_window = getattr(config, "sliding_window", None)
        self.markov_rank = int(getattr(config, "dspark_markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )

        # main_proj / main_norm: fuse concat(target aux hidden) -> main_x.
        self.main_proj = nn.Linear(len(self.target_layer_ids) * H, H, bias=False)
        self.main_norm = DeepseekV4RMSNorm(H, eps=config.rms_norm_eps)

        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DSparkV4Block(config, i) for i in range(self.num_layers)]
        )
        # Optional torch.compile of each decoder block. The block is launch-bound
        # (hyper-connection Sinkhorn iters + norms + mHC glue as many tiny kernels
        # around the flex attention + grouped_mm MoE); fusing them cuts the per-step
        # overhead. dynamic=True since Q/S vary per batch. Env SPECFORGE_COMPILE_DRAFT=1.
        if os.environ.get("SPECFORGE_COMPILE_DRAFT") == "1":
            for i in range(len(self.layers)):
                self.layers[i] = torch.compile(self.layers[i], dynamic=True)

        # Head stack: learned hc_head collapse + final norm (pre-lm_head).
        self.hc_head = DeepseekV4HyperHead(config)
        self.norm = DeepseekV4RMSNorm(H, eps=config.rms_norm_eps)

        # DSpark heads.
        self.markov_head: Optional[nn.Module] = (
            VanillaMarkov(vocab_size=config.vocab_size, markov_rank=self.markov_rank)
            if self.markov_rank > 0
            else None
        )
        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            in_dim = H + (self.markov_rank if self.confidence_head_with_markov else 0)
            self.confidence_head = AcceptRateHead(in_dim)

        self.gradient_checkpointing = bool(getattr(config, "gradient_checkpointing", True))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self, std: Optional[float] = None) -> None:
        """From-scratch init matching ``DeepseekV4PreTrainedModel._init_weights``.

        The reused V4 modules allocate several ``torch.empty`` parameters (hyper-
        connection ``fn``/``base``/``scale``, attention ``sinks``, expert / router
        weights) that ``from_pretrained`` would normally fill; uninitialized they
        are garbage. Init everything with the released model's conventions.
        """
        std = self.config.initializer_range if std is None else std
        for module in self.modules():
            if isinstance(module, DeepseekV4RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if getattr(module, "e_score_correction_bias", None) is not None:
                    nn.init.zeros_(module.e_score_correction_bias)
                if getattr(module, "tid2eid", None) is not None:
                    nn.init.zeros_(module.tid2eid)
            elif isinstance(module, DeepseekV4Experts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
                nn.init.normal_(module.down_proj, mean=0.0, std=std)
            elif isinstance(module, DeepseekV4Attention):
                nn.init.zeros_(module.sinks)
            elif isinstance(module, DeepseekV4HyperConnection):
                nn.init.normal_(module.fn, mean=0.0, std=std)
                nn.init.zeros_(module.base)
                nn.init.ones_(module.scale)
            elif isinstance(module, DeepseekV4HyperHead):
                nn.init.normal_(module.hc_fn, mean=0.0, std=std)
                nn.init.zeros_(module.hc_base)
                nn.init.ones_(module.hc_scale)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def combine_hidden_states(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """main_x = main_norm(main_proj(concat target aux hidden states))."""
        return self.main_norm(self.main_proj(target_hidden))

    def forward(
        self,
        *,
        target_hidden: torch.Tensor,        # [B, S, len(target_layer_ids)*H]
        noise_embedding: torch.Tensor,      # [B, N*block_size, H]
        context_position_ids: torch.Tensor,  # [B, S]
        draft_position_ids: torch.Tensor,    # [B, N*block_size]
        attention_mask: torch.Tensor,        # additive [B, 1, N*block_size, S+N*block_size]
    ) -> torch.Tensor:
        main_x = self.combine_hidden_states(target_hidden)  # [B, S, H]

        cos_q, sin_q = self.rotary_emb(
            noise_embedding, position_ids=draft_position_ids, layer_type="main"
        )
        cos_ctx, sin_ctx = self.rotary_emb(
            main_x, position_ids=context_position_ids, layer_type="main"
        )

        # Expand block embeddings into hc_mult parallel streams.
        hidden = (
            noise_embedding.unsqueeze(2)
            .expand(-1, -1, self.hc_mult, -1)
            .contiguous()
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden,
                    main_x,
                    cos_q,
                    sin_q,
                    cos_ctx,
                    sin_ctx,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                hidden = layer(
                    hidden, main_x, cos_q, sin_q, cos_ctx, sin_ctx, attention_mask
                )

        hidden = self.hc_head(hidden)  # learned hc-stream collapse [B, Q, H]
        hidden = self.norm(hidden)
        return hidden


def build_dspark_v4_config(base: dict) -> DeepseekV4Config:
    """Build the 3-block draft :class:`DeepseekV4Config` from a config dict.

    Forces the draft to all-``sliding_attention`` blocks and all-``moe`` (non-hash)
    routing — the released draft layout — regardless of the target's per-layer
    schedule, and carries the DSpark head fields (``dspark_*`` / markov /
    confidence) through as plain attributes.
    """
    cfg_kwargs = dict(base)
    num_layers = int(cfg_kwargs.get("num_hidden_layers", 3))
    cfg_kwargs["num_hidden_layers"] = num_layers
    cfg_kwargs["layer_types"] = ["sliding_attention"] * num_layers
    cfg_kwargs["mlp_layer_types"] = ["moe"] * num_layers
    # Drop the target-schedule field so it can't override our explicit layer_types.
    cfg_kwargs.pop("compress_ratios", None)
    config = DeepseekV4Config(**cfg_kwargs)
    # Carry DSpark head fields as attributes (kept off the strict dataclass).
    for key in (
        "dspark_target_layer_ids",
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_markov_rank",
        "markov_head_type",
        "enable_confidence_head",
        "confidence_head_with_markov",
        "gradient_checkpointing",
    ):
        if key in base:
            setattr(config, key, base[key])
    # Fused grouped-GEMM experts. The transformers reference DeepseekV4Experts.forward
    # is a Python loop over the 256 routed experts (torch.where/.nonzero() CPU syncs +
    # index_add per expert); under gradient checkpointing its backward recomputes and
    # backprops through 256*num_layers sync-bound tiny ops -> the draft backward is ~10x
    # the forward (the training-throughput bottleneck). "grouped_mm" dispatches to one
    # torch._grouped_mm over expert-sorted tokens (numerically equivalent to the loop:
    # cosine 0.99999) for a ~10x faster backward. Overridable via config key.
    config._experts_implementation = base.get("experts_implementation", "grouped_mm")
    return config


def dspark_v4_state_dict_to_checkpoint(
    state_dict: dict, config: DeepseekV4Config
) -> dict:
    """Remap this model's ``state_dict`` to the released ``mtp.*`` checkpoint layout.

    Produces exactly the key names in ``deepseek-ai/DeepSeek-V4-Flash-DSpark``
    (verified against its safetensors index), so a trained checkpoint loads via
    vLLM's ``DSparkDeepseekV4ForCausalLM.load_weights``. Fused expert weights are
    split back into per-expert ``w1``/``w3`` (gate/up) and ``w2`` (down); the
    head-stack (main_proj/main_norm on stage 0; norm/hc_head/markov/confidence on
    the last stage) is placed to match the release. Weights are emitted in bf16
    (the pre-quantization form) — expert fp4/fp8 quantization is a downstream
    deploy step.
    """
    num_layers = int(config.num_hidden_layers)
    last = num_layers - 1
    moe_inter = int(config.moe_intermediate_size)

    # Per-block submodule name -> checkpoint suffix under mtp.{stage}.
    attn_map = {
        "self_attn.q_a_proj.weight": "attn.wq_a.weight",
        "self_attn.q_a_norm.weight": "attn.q_norm.weight",
        "self_attn.q_b_proj.weight": "attn.wq_b.weight",
        "self_attn.kv_proj.weight": "attn.wkv.weight",
        "self_attn.kv_norm.weight": "attn.kv_norm.weight",
        "self_attn.o_a_proj.weight": "attn.wo_a.weight",
        "self_attn.o_b_proj.weight": "attn.wo_b.weight",
        "self_attn.sinks": "attn.attn_sink",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.scale": "hc_ffn_scale",
        "mlp.gate.weight": "ffn.gate.weight",
        "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
        "mlp.shared_experts.gate_proj.weight": "ffn.shared_experts.w1.weight",
        "mlp.shared_experts.up_proj.weight": "ffn.shared_experts.w3.weight",
        "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
    }

    out: dict = {}
    for name, tensor in state_dict.items():
        tensor = tensor.detach().to(torch.bfloat16).contiguous()

        if name.startswith("main_proj.") or name.startswith("main_norm."):
            out[f"mtp.0.{name}"] = tensor
            continue
        if name == "norm.weight":
            out[f"mtp.{last}.norm.weight"] = tensor
            continue
        if name.startswith("hc_head."):
            suffix = name[len("hc_head.") :]  # hc_fn / hc_base / hc_scale
            remap = {"hc_fn": "hc_head_fn", "hc_base": "hc_head_base", "hc_scale": "hc_head_scale"}
            out[f"mtp.{last}.{remap[suffix]}"] = tensor
            continue
        if name.startswith("markov_head."):
            out[f"mtp.{last}.{name}"] = tensor
            continue
        if name.startswith("confidence_head."):
            out[f"mtp.{last}.{name}"] = tensor
            continue

        if name.startswith("layers."):
            rest = name[len("layers.") :]
            stage, sub = rest.split(".", 1)
            # Fused experts -> per-expert w1/w3 (gate/up) and w2 (down).
            if sub == "mlp.experts.gate_up_proj":
                for e in range(tensor.shape[0]):
                    out[f"mtp.{stage}.ffn.experts.{e}.w1.weight"] = tensor[e, :moe_inter].contiguous()
                    out[f"mtp.{stage}.ffn.experts.{e}.w3.weight"] = tensor[e, moe_inter:].contiguous()
                continue
            if sub == "mlp.experts.down_proj":
                for e in range(tensor.shape[0]):
                    out[f"mtp.{stage}.ffn.experts.{e}.w2.weight"] = tensor[e].contiguous()
                continue
            if sub in attn_map:
                out[f"mtp.{stage}.{attn_map[sub]}"] = tensor
                continue
            # Fallback: keep the sub-path verbatim under the stage prefix.
            out[f"mtp.{stage}.{sub}"] = tensor
            continue

        # Anything else (unexpected) is passed through untouched.
        out[name] = tensor
    return out
