# coding=utf-8
"""DSpark draft with a Qwen3-MoE backbone (dense attention + standard top-k MoE).

Design point: match the released DeepSeek-V4-Flash-DSpark drafter's *scale/capacity*
(~19.85B total, ~1B active) while dropping the V4-specific awkwardness (MLA, mHC,
hash routing). This reuses SpecForge's *verified* DFlash backbone verbatim —
dual-source KV injection (the actual DSpark conditioning), fc/hidden_norm target
fuse, block-diffusion mask — and only swaps the per-layer FFN from the dense
``Qwen3MLP`` to the standard ``Qwen3MoeSparseMoeBlock`` (learned top-k routing).
Markov + confidence heads and the training objective are unchanged, so the
existing (audited) ``OnlineDSparkModel`` trains it with no changes.

Only genuine deviation vs a dense DSpark draft: MoE load-balancing aux loss is not
added (the block returns only hidden states); routing is trained through the main
objective. Fine for this scale-matching run.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspark import AcceptRatePredictor, build_markov_head


class DSparkMoEDraftModel(DFlashDraftModel):
    """DFlash backbone (dense attn + dual-source KV) with Qwen3-MoE FFNs + DSpark heads."""

    def __init__(self, config) -> None:
        # Build the DFlash backbone with a *tiny* dense MLP first (it is replaced
        # by the MoE block below); avoids allocating full-size dense FFNs.
        orig_intermediate = config.intermediate_size
        config.intermediate_size = 1
        super().__init__(config)
        config.intermediate_size = orig_intermediate

        # Swap each layer's FFN for a standard Qwen3-MoE sparse block.
        for layer in self.layers:
            layer.mlp = Qwen3MoeSparseMoeBlock(config)

        # DSpark heads (identical to DSparkDraftModel).
        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", True)
        )
        self.markov_head = build_markov_head(config)
        self.confidence_head: Optional[nn.Module] = None
        if getattr(config, "enable_confidence_head", False):
            conf_in = config.hidden_size + (
                self.markov_rank if self.confidence_head_with_markov else 0
            )
            self.confidence_head = AcceptRatePredictor(conf_in)

        self.post_init()  # initialize attention/fc/norm/head modules
        self._init_moe_params(std=float(getattr(config, "initializer_range", 0.02)))

        # Select the experts kernel AFTER post_init: the model-level HF validation
        # only allows "eager" for a custom subclass, but the experts read
        # config._experts_implementation live at forward time. "grouped_mm" is a
        # fused, differentiable grouped-GEMM kernel — ~60x faster than the eager
        # Python expert loop (essential to train 128 experts at speed).
        self.config._experts_implementation = getattr(
            config, "dspark_experts_impl", "grouped_mm"
        )

    def _init_moe_params(self, std: float = 0.02) -> None:
        """post_init()/_init_weights only touch nn.Linear/nn.Embedding modules; the
        Qwen3-MoE experts are bare 3D nn.Parameters and the router is left at zero,
        so without this the experts output 0 and receive no gradient. Initialize the
        fused expert weights and the router gate explicitly."""
        for layer in self.layers:
            moe = layer.mlp
            with torch.no_grad():
                for pname in ("gate_up_proj", "down_proj"):
                    p = getattr(moe.experts, pname, None)
                    if p is not None:
                        p.normal_(0.0, std)
                for gp in moe.gate.parameters():
                    gp.normal_(0.0, std)

    @property
    def num_active_params(self) -> int:
        """Params touched per token = non-expert + top_k/num_experts of expert params."""
        ne = int(self.config.num_experts)
        k = int(self.config.num_experts_per_tok)
        exp = nonexp = 0
        for n, p in self.named_parameters():
            if ".experts." in n:
                exp += p.numel()
            else:
                nonexp += p.numel()
        return int(nonexp + exp * k / ne)
