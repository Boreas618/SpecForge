# coding=utf-8
"""Packed online training wrapper and attention masks for DConv."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from specforge.modeling.draft.dconv import DConvDraftModel

from .dflash import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDFlashModel,
    create_block_mask,
)


def create_dconv_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    causal_block_mask: torch.Tensor,
    S: int,
    stream_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Create the dense packed DConv mask.

    Every stream sees target context strictly before its anchor and only its
    own stream. Pass-1 streams are bidirectional; pass-2 streams are causal.
    """
    batch_size, num_blocks = anchor_positions.shape
    query_len = num_blocks * stream_len
    kv_len = S + query_len

    query_indices = torch.arange(query_len, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(kv_len, device=device).view(1, 1, 1, -1)
    query_block_ids = query_indices // stream_len

    anchors = anchor_positions.view(batch_size, 1, num_blocks, 1)
    anchors = anchors.repeat_interleave(stream_len, dim=2)
    context_visible = (kv_indices < S) & (kv_indices < anchors)

    is_stream = kv_indices >= S
    kv_block_ids = (kv_indices - S) // stream_len
    same_block = is_stream & (query_block_ids == kv_block_ids)
    causal = causal_block_mask.view(batch_size, 1, num_blocks, 1)
    causal = causal.repeat_interleave(stream_len, dim=2)
    stream_visible = same_block & (
        (~causal) | ((kv_indices - S) % stream_len <= query_indices % stream_len)
    )

    valid = block_keep_mask.view(batch_size, 1, num_blocks, 1)
    valid = valid.repeat_interleave(stream_len, dim=2)
    return (context_visible | stream_visible) & valid


def create_dconv_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    causal_block_mask: torch.Tensor,
    S: int,
    stream_len: int,
    device: torch.device,
):
    """Create the FlexAttention packed DConv mask."""
    if not FLEX_ATTENTION_AVAILABLE:
        raise ValueError("flex_attention is not available; use sdpa/eager")

    batch_size, num_blocks = anchor_positions.shape
    query_len = num_blocks * stream_len
    kv_len = S + query_len

    def dconv_mask_mod(batch_idx, head_idx, query_idx, kv_idx):
        del head_idx
        query_block_id = query_idx // stream_len
        safe_block_id = query_block_id.clamp(max=num_blocks - 1)
        anchor = anchor_positions[batch_idx, safe_block_id]

        context_visible = (kv_idx < S) & (kv_idx < anchor)
        is_stream = kv_idx >= S
        kv_block_id = (kv_idx - S) // stream_len
        same_block = is_stream & (query_block_id == kv_block_id)
        within_stream_causal = (kv_idx - S) % stream_len <= query_idx % stream_len
        stream_visible = same_block & (
            (~causal_block_mask[batch_idx, safe_block_id]) | within_stream_causal
        )

        in_bounds = query_block_id < num_blocks
        valid = block_keep_mask[batch_idx, safe_block_id]
        return (context_visible | stream_visible) & valid & in_bounds

    return create_block_mask(
        dconv_mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=query_len,
        KV_LEN=kv_len,
        device=device,
    )


class OnlineDConvModel(OnlineDFlashModel):
    """Train both DConv passes in one packed multi-anchor forward.

    Each anchor is independently assigned a masked, bidirectional pass-1
    stream or a gold-prefix, causal pass-2 stream. Both forms use shifted
    next-token labels and the standard DFlash position-decayed CE objective.
    Optional target-distribution and confidence losses remain available for a
    later DSpark-style disaggregated training stage.
    """

    def __init__(
        self,
        draft_model: DConvDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = 7.0,
        rho_reread: float = 0.5,
        alpha_ce: float = 1.0,
        alpha_tv: float = 0.0,
        alpha_conf: float = 0.0,
    ):
        if not 0.0 <= rho_reread <= 1.0:
            raise ValueError(f"rho_reread must be in [0, 1], got {rho_reread}")
        if min(alpha_ce, alpha_tv, alpha_conf) < 0.0:
            raise ValueError("DConv loss weights must be non-negative")
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=draft_model.block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        self.window_len = draft_model.window_len
        self.stream_len = draft_model.stream_len
        self.rho_reread = float(rho_reread)
        self.alpha_ce = float(alpha_ce)
        self.alpha_tv = float(alpha_tv)
        self.alpha_conf = float(alpha_conf)

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eligible = loss_mask.clone()
        eligible[:, : self.window_len - 1] = 0
        if self.block_size <= seq_len:
            eligible[:, seq_len - self.block_size :] = 0
        return super()._sample_anchor_positions(seq_len, eligible, device)

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(
            -(self.window_len - 1),
            self.block_size,
            device=anchor_positions.device,
        ).view(1, 1, -1)
        return (anchor_positions.unsqueeze(-1) + offsets).view(
            anchor_positions.shape[0], -1
        )

    def _build_stream_ids(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        reread_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        offsets = torch.arange(
            -(self.window_len - 1),
            self.block_size,
            device=input_ids.device,
        ).view(1, 1, -1)
        indices = (anchor_positions.unsqueeze(-1) + offsets).clamp(
            min=0, max=seq_len - 1
        )
        gathered_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            indices,
        )
        is_proposal_slot = (offsets > 0).expand_as(gathered_ids)
        reveal = (~is_proposal_slot) | reread_mask.unsqueeze(-1)
        reveal = reveal & block_keep_mask.unsqueeze(-1)
        return torch.where(
            reveal,
            gathered_ids,
            torch.full_like(gathered_ids, self.mask_token_id),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager"
            )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        anchors, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        num_blocks = anchors.shape[1]
        reread_mask = (
            torch.rand(batch_size, num_blocks, device=device) < self.rho_reread
        ) & block_keep_mask

        stream_ids = self._build_stream_ids(
            input_ids, anchors, block_keep_mask, reread_mask
        )
        noise_embedding = self.embed_tokens(
            stream_ids.view(batch_size, num_blocks * self.stream_len)
        )
        context_position_ids = torch.arange(seq_len, device=device)[None].expand(
            batch_size, -1
        )
        position_ids = torch.cat(
            [context_position_ids, self._create_position_ids(anchors)], dim=1
        )
        mask_kwargs = dict(
            anchor_positions=anchors,
            block_keep_mask=block_keep_mask,
            causal_block_mask=reread_mask,
            S=seq_len,
            stream_len=self.stream_len,
            device=device,
        )
        if self.attention_backend == "flex_attention":
            attention_mask = create_dconv_block_mask(**mask_kwargs)
        else:
            attention_mask = create_dconv_sdpa_mask(**mask_kwargs)

        output_hidden = self.draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=attention_mask,
            num_blocks=num_blocks,
        )
        emitting_hidden = output_hidden.view(
            batch_size, num_blocks, self.stream_len, -1
        )[:, :, self.window_len - 1 :]
        logits = self.lm_head(emitting_hidden)

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchors.unsqueeze(-1) + label_offsets
        valid_labels = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        eval_weight = (block_keep_mask.unsqueeze(-1) & valid_labels).float()
        eval_weight = eval_weight * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        loss_weight = eval_weight
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(self.block_size, device=device).view(1, 1, -1)
            loss_weight = loss_weight * torch.exp(
                -positions.float() / float(self.loss_decay_gamma)
            )

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_loss_weight = loss_weight.reshape(-1)
        denominator = flat_loss_weight.sum() + 1e-6
        ce_loss = (
            F.cross_entropy(flat_logits, flat_targets, reduction="none")
            * flat_loss_weight
        ).sum() / denominator
        loss = self.alpha_ce * ce_loss
        valid_block_count = block_keep_mask.sum().clamp_min(1)
        metrics = {
            "ce_loss": ce_loss.detach(),
            "reread_frac": (reread_mask.sum().float() / valid_block_count).detach(),
        }

        if target_last_hidden_states is not None and (
            self.alpha_tv > 0 or self.alpha_conf > 0
        ):
            aligned_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(-1, num_blocks, -1, -1),
                2,
                aligned_indices.unsqueeze(-1).expand(
                    -1, -1, -1, target_last_hidden_states.size(-1)
                ),
            )
            draft_probabilities = torch.softmax(logits.float(), dim=-1)
            target_probabilities = torch.softmax(
                self.lm_head(aligned_target_hidden).float(), dim=-1
            )
            l1_distance = (draft_probabilities - target_probabilities).abs().sum(dim=-1)
            if self.alpha_tv > 0:
                tv_loss = (l1_distance * loss_weight).sum() / denominator
                loss = loss + self.alpha_tv * tv_loss
                metrics["tv_loss"] = tv_loss.detach()
            confidence_logits = self.draft_model.predict_confidence(emitting_hidden)
            if confidence_logits is not None and self.alpha_conf > 0:
                acceptance_target = (1.0 - 0.5 * l1_distance).clamp(0.0, 1.0).detach()
                confidence_loss = (
                    F.binary_cross_entropy_with_logits(
                        confidence_logits,
                        acceptance_target,
                        reduction="none",
                    )
                    * loss_weight
                ).sum() / denominator
                loss = loss + self.alpha_conf * confidence_loss
                metrics["conf_loss"] = confidence_loss.detach()

        with torch.no_grad():
            predictions = flat_logits.argmax(dim=-1)
            flat_eval_mask = eval_weight.reshape(-1) > 0.5
            hits = (predictions == flat_targets) & flat_eval_mask
            accuracy_denom = eval_weight.sum()
            accuracy = hits.sum().float() / (accuracy_denom + 1e-6)
            flat_reread = (
                reread_mask.unsqueeze(-1).expand_as(eval_weight.bool()).reshape(-1)
            )
            metrics["accuracy_denom"] = accuracy_denom
            metrics["acc_pass1"] = (hits & ~flat_reread).sum().float() / (
                (flat_eval_mask & ~flat_reread).sum() + 1e-6
            )
            metrics["acc_pass2"] = (hits & flat_reread).sum().float() / (
                (flat_eval_mask & flat_reread).sum() + 1e-6
            )
        return loss, accuracy, metrics


__all__ = [
    "OnlineDConvModel",
    "create_dconv_block_mask",
    "create_dconv_sdpa_mask",
]
