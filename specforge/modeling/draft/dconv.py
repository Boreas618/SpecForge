# coding=utf-8
"""DConv draft model: DFlash with causal short-convolution residuals."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    Qwen3Config,
    eager_attention_forward,
)
from typing_extensions import Unpack

from .dflash import (
    DFlashDraftModel,
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    apply_rotary_pos_emb,
    extract_context_feature,
    sample,
)
from .dspark import AcceptRatePredictor
from .registry import register_draft


class ShortConv(nn.Module):
    """Zero-initialized depthwise causal convolution in residual form.

    ``num_blocks`` isolates independently packed training streams. The
    convolution is only used on the draft stream; injected target features do
    not pass through it.
    """

    def __init__(self, channels: int, width: int):
        super().__init__()
        if width <= 0:
            raise ValueError(f"ShortConv width must be positive, got {width}")
        self.width = int(width)
        self.kernel = nn.Parameter(torch.zeros(channels, 1, self.width))

    def forward(self, x: torch.Tensor, num_blocks: int = 1) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if num_blocks <= 0 or seq_len % num_blocks != 0:
            raise ValueError(
                f"sequence length {seq_len} must be divisible by num_blocks "
                f"({num_blocks})"
            )
        block_len = seq_len // num_blocks
        hidden = x.reshape(batch_size * num_blocks, block_len, channels)
        hidden = hidden.transpose(1, 2)
        hidden = F.pad(hidden, (self.width - 1, 0))
        hidden = F.conv1d(hidden, self.kernel, groups=channels)
        return x + hidden.transpose(1, 2).reshape(batch_size, seq_len, channels)


class Qwen3DConvAttention(Qwen3DFlashAttention):
    """DFlash attention with causal short convolutions on stream K and V."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        kv_channels = config.num_key_value_heads * self.head_dim
        dconv_config = getattr(config, "dflash_config", {}) or {}
        conv_width = int(dconv_config.get("conv_width", 4))
        self.phi_k = ShortConv(kv_channels, conv_width)
        self.phi_v = ShortConv(kv_channels, conv_width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_blocks: int = 1,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_len = hidden_states.shape[:-1]
        context_len = target_hidden.shape[1]

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, query_len, -1, self.head_dim)
        query_states = self.q_norm(query_states).transpose(1, 2)

        context_key_states = self.k_proj(target_hidden)
        stream_key_states = self.phi_k(self.k_proj(hidden_states), num_blocks)
        context_value_states = self.v_proj(target_hidden)
        stream_value_states = self.phi_v(self.v_proj(hidden_states), num_blocks)

        key_states = torch.cat([context_key_states, stream_key_states], dim=1).view(
            batch_size, context_len + query_len, -1, self.head_dim
        )
        value_states = torch.cat(
            [context_value_states, stream_value_states], dim=1
        ).view(batch_size, context_len + query_len, -1, self.head_dim)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]
        kwargs.pop("is_causal", None)
        attention_output, attention_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attention_output = attention_output.reshape(batch_size, query_len, -1)
        return self.o_proj(attention_output), attention_weights


class Qwen3DConvDecoderLayer(Qwen3DFlashDecoderLayer):
    """DFlash decoder layer with short convs before both residual adds."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3DConvAttention(config=config, layer_idx=layer_idx)
        dconv_config = getattr(config, "dflash_config", {}) or {}
        conv_width = int(dconv_config.get("conv_width", 4))
        self.phi_o = ShortConv(config.hidden_size, conv_width)
        self.phi_m = ShortConv(config.hidden_size, conv_width)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_blocks: int = 1,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            num_blocks=num_blocks,
            **kwargs,
        )[0]
        hidden_states = residual + self.phi_o(hidden_states, num_blocks)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.phi_m(hidden_states, num_blocks)


def create_dconv_inference_mask(
    kv_len: int,
    stream_len: int,
    causal: bool,
    device: torch.device,
) -> torch.Tensor:
    """Make a dense inference mask for context plus one DConv stream."""
    mask = torch.ones(stream_len, kv_len, dtype=torch.bool, device=device)
    if causal:
        stream_start = kv_len - stream_len
        rows = torch.arange(stream_len, device=device)
        cols = torch.arange(stream_len, device=device)
        mask[:, stream_start:] = cols[None, :] <= rows[:, None]
    return mask[None, None]


@register_draft
class DConvDraftModel(DFlashDraftModel):
    """DFlash backbone with four causal short-convolution sites per layer."""

    expected_projector_type = "dconv"
    _no_split_modules = ["Qwen3DConvDecoderLayer"]

    def __init__(self, config) -> None:
        dconv_config = getattr(config, "dflash_config", None) or {}
        projector_type = dconv_config.get("projector_type")
        if projector_type is None:
            dconv_config["projector_type"] = self.expected_projector_type
        elif projector_type != self.expected_projector_type:
            raise ValueError(
                "DConvDraftModel requires dflash_config.projector_type='dconv'."
            )
        dconv_config.setdefault("conv_width", 4)
        dconv_config.setdefault("window_len", 4)
        config.dflash_config = dconv_config

        super().__init__(config)
        # Replace one parent layer at a time. Building a second full ModuleList
        # before dropping DFlash's layers would briefly double host memory for
        # the 5-layer, width-4096 Qwen3-8B drafter.
        for layer_idx in range(config.num_hidden_layers):
            self.layers[layer_idx] = Qwen3DConvDecoderLayer(config, layer_idx)
        self.conv_width = int(dconv_config["conv_width"])
        self.window_len = int(dconv_config["window_len"])
        if self.window_len < self.conv_width:
            raise ValueError(
                "window_len must be at least conv_width so the first emitting "
                "position has no padded convolution taps"
            )
        self.reread_threshold = float(dconv_config.get("reread_threshold", 1.0))
        if not 0.0 <= self.reread_threshold <= 1.0:
            raise ValueError("reread_threshold must be in [0, 1]")
        self.post_init()

    @property
    def num_draft_tokens(self) -> int:
        return self.block_size

    @property
    def stream_len(self) -> int:
        return self.window_len + self.block_size - 1

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        self.confidence_head = None
        if dflash_config.get("enable_confidence_head", False):
            self.confidence_head = AcceptRatePredictor(input_dim=config.hidden_size)

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del prev_token_ids
        if self.confidence_head is None:
            return None
        return self.confidence_head(hidden_states).float()

    @torch.inference_mode()
    def draft_block(
        self,
        window_ids: torch.Tensor,
        target_hidden_new: torch.Tensor,
        past_key_values: DynamicCache,
        t_anchor: int,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        temperature: float = 0.0,
    ) -> dict:
        """Draft in parallel, then optionally rescore the proposal causally."""
        batch_size = window_ids.shape[0]
        device = window_ids.device
        gamma = self.block_size
        stream_positions = torch.arange(
            t_anchor - self.window_len + 1,
            t_anchor + gamma,
            device=device,
        )

        def run(
            slot_ids: Optional[torch.Tensor],
            new_hidden: torch.Tensor,
            *,
            causal: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if slot_ids is None:
                slots = torch.full(
                    (batch_size, gamma - 1),
                    self.mask_token_id,
                    dtype=torch.long,
                    device=device,
                )
            else:
                slots = slot_ids
            stream_ids = torch.cat([window_ids, slots], dim=1)
            new_context_len = new_hidden.shape[1]
            context_positions = torch.arange(
                t_anchor - new_context_len, t_anchor, device=device
            )
            position_ids = torch.cat([context_positions, stream_positions]).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
            kv_len = (
                past_key_values.get_seq_length() + new_context_len + self.stream_len
            )
            hidden_states = self(
                position_ids=position_ids,
                attention_mask=create_dconv_inference_mask(
                    kv_len, self.stream_len, causal, device
                ),
                noise_embedding=embed_tokens(stream_ids),
                target_hidden=new_hidden,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values.crop(t_anchor)
            emitting_states = hidden_states[:, self.window_len - 1 :]
            return lm_head(emitting_states), emitting_states

        def probabilities(logits: torch.Tensor) -> torch.Tensor:
            scaled = logits / temperature if temperature > 1e-5 else logits
            return F.softmax(scaled.float(), dim=-1)

        logits_pass1, hidden_pass1 = run(None, target_hidden_new, causal=False)
        proposal = sample(logits_pass1, temperature)
        probabilities_pass1 = probabilities(logits_pass1)

        confidence_logits = self.predict_confidence(hidden_pass1)
        run_pass2 = torch.ones(batch_size, dtype=torch.bool, device=device)
        expected_accept_len = None
        if confidence_logits is not None and self.reread_threshold < 1.0:
            expected_accept_len = (
                torch.sigmoid(confidence_logits).cumprod(dim=1).sum(dim=1)
            )
            run_pass2 = expected_accept_len < self.reread_threshold * gamma

        if bool(run_pass2.any()):
            logits_pass2, _ = run(
                proposal[:, : gamma - 1],
                target_hidden_new[:, :0],
                causal=True,
            )
            rescored_tokens = sample(logits_pass2, temperature)
            probabilities_pass2 = probabilities(logits_pass2)
            tokens = torch.where(run_pass2[:, None], rescored_tokens, proposal)
            token_probabilities = torch.where(
                run_pass2[:, None, None],
                probabilities_pass2,
                probabilities_pass1,
            )
        else:
            tokens = proposal
            token_probabilities = probabilities_pass1

        return {
            "tokens": tokens,
            "q": token_probabilities,
            "pass2_ran": run_pass2,
            "expected_accept_len": expected_accept_len,
            "d1": proposal,
        }

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ) -> torch.LongTensor:
        """Reference lossless greedy/sampling loop for the two-pass drafter."""
        self.eval()
        num_input_tokens = input_ids.shape[1]
        if num_input_tokens <= self.window_len:
            raise ValueError("prompt must be longer than window_len")
        max_length = num_input_tokens + max_new_tokens
        gamma = self.block_size

        output_ids = torch.full(
            (1, max_length + gamma + 1),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        target_cache = DynamicCache()
        draft_cache = DynamicCache()
        target_output = target(
            input_ids,
            position_ids=torch.arange(num_input_tokens, device=target.device)[None],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens] = sample(target_output.logits, temperature)
        target_hidden = extract_context_feature(
            target_output.hidden_states, self.target_layer_ids
        )

        anchor = num_input_tokens
        while anchor < max_length:
            draft_result = self.draft_block(
                output_ids[:, anchor - self.window_len + 1 : anchor + 1],
                target_hidden,
                draft_cache,
                anchor,
                target.model.embed_tokens,
                target.lm_head,
                temperature,
            )
            block_ids = torch.cat(
                [output_ids[:, anchor : anchor + 1], draft_result["tokens"]],
                dim=1,
            )
            target_output = target(
                block_ids,
                position_ids=torch.arange(
                    anchor, anchor + gamma + 1, device=target.device
                )[None],
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = sample(target_output.logits, temperature)
            accepted = int(
                (block_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, anchor : anchor + accepted + 1] = block_ids[:, : accepted + 1]
            output_ids[:, anchor + accepted + 1] = posterior[:, accepted]
            anchor += accepted + 1
            target_cache.crop(anchor)
            target_hidden = extract_context_feature(
                target_output.hidden_states, self.target_layer_ids
            )[:, : accepted + 1]
            if stop_token_ids and any(
                stop_id in output_ids[0, num_input_tokens:]
                for stop_id in stop_token_ids
            ):
                break

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids:
            stops = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_indices = torch.isin(output_ids[0, num_input_tokens:], stops).nonzero(
                as_tuple=True
            )[0]
            if stop_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]
        return output_ids


__all__ = [
    "DConvDraftModel",
    "Qwen3DConvAttention",
    "Qwen3DConvDecoderLayer",
    "ShortConv",
    "create_dconv_inference_mask",
]
