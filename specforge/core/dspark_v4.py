# coding=utf-8
"""Online training wrapper for the faithful DeepSeek-V4-Flash-DSpark draft.

Reuses :class:`OnlineDSparkModel`'s anchor sampling, MASK-token noise stream,
label/eval-mask construction, and the shared DSpark objective (``_dspark_objective``:
CE + L1 distribution distillation + confidence BCE with DeepSpec's pooled global
mean). The only differences from the dense DSpark path are the backbone
(:class:`DSparkV4DraftModel` with dual-source MLA + hyper-connections + hash-free
256-expert MoE) and the attention mask it consumes:

The mask reproduces the served DSpark conditioning — each draft block's queries
attend (1) their own block bidirectionally and (2) the target context strictly
before the block's anchor, within the model's ``sliding_window``. Context K/V are
derived per layer from ``main_x`` inside the draft (see :mod:`dspark_v4`), so this
is the training-time equivalent of ``precompute_and_store_context_kv`` + the
query-block forward.
"""

import os
from typing import Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask

from specforge.core.dspark import OnlineDSparkModel
from specforge.modeling.draft.dspark_v4 import DSparkV4DraftModel

# Optional override for the DSpark-V4 draft attention path. By default we honor the
# `attention_backend` passed by the trainer; set SPECFORGE_DRAFT_ATTN=flex to opt
# into block-sparse flex attention for experiments.
_DRAFT_ATTN_OVERRIDE = os.environ.get("SPECFORGE_DRAFT_ATTN")


class OnlineDSparkV4Model(OnlineDSparkModel):
    """DSpark online wrapper with the faithful DeepSeek-V4 backbone."""

    def _build_dual_source_mask(
        self,
        anchor_positions: torch.Tensor,   # [B, N]
        block_keep_mask: torch.Tensor,    # [B, N]
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Additive attention bias ``[B, 1, N*bs, S + N*bs]`` for the draft.

        Query ``q`` in block ``b`` (draft absolute position ``p = anchor_b + k``)
        attends:
          * context key ``j < S`` iff ``j < anchor_b`` (strictly before the anchor,
            matching the served draft where the anchor's own target hidden is not
            yet available) and, when a finite ``sliding_window`` is set, within
            ``(p - sliding_window, ...)`` — the sliding-window branch's local span;
          * noise key in the same block ``b`` (bidirectional within the block).
        Invalid (padded) blocks attend nothing; the per-head attention sink keeps
        those all-masked rows finite, and they are excluded from the loss anyway.
        """
        B, N = anchor_positions.shape
        bs = self.block_size
        Q = N * bs
        sliding_window = getattr(self.draft_model, "sliding_window", None)

        qidx = torch.arange(Q, device=device)
        q_block = (qidx // bs).unsqueeze(0).expand(B, -1)           # [B, Q]
        k_in_block = (qidx % bs).unsqueeze(0)                       # [1, Q]
        anchor_q = torch.gather(anchor_positions, 1, q_block)       # [B, Q]
        keep_q = torch.gather(block_keep_mask, 1, q_block)          # [B, Q]
        p_abs = anchor_q + k_in_block                              # [B, Q]

        # Context: j < anchor (+ sliding window), block kept.
        ctx_j = torch.arange(seq_len, device=device).view(1, 1, -1)  # [1, 1, S]
        attend_ctx = ctx_j < anchor_q.unsqueeze(-1)                  # [B, Q, S]
        if sliding_window is not None and sliding_window > 0:
            attend_ctx = attend_ctx & (ctx_j > (p_abs.unsqueeze(-1) - int(sliding_window)))
        attend_ctx = attend_ctx & keep_q.unsqueeze(-1)

        # Noise: same block (bidirectional), block kept.
        same_block = q_block.unsqueeze(-1) == q_block.unsqueeze(1)  # [B, Q, Q]
        attend_noise = same_block & keep_q.unsqueeze(-1)

        attend = torch.cat([attend_ctx, attend_noise], dim=-1)      # [B, Q, S+Q]
        bias = torch.zeros(B, 1, Q, seq_len + Q, dtype=dtype, device=device)
        bias.masked_fill_(~attend.unsqueeze(1), torch.finfo(dtype).min)
        return bias

    def _build_dual_source_block_mask(
        self,
        anchor_positions: torch.Tensor,   # [B, N]
        block_keep_mask: torch.Tensor,    # [B, N]
        seq_len: int,
        device: torch.device,
    ):
        """Block-sparse flex ``BlockMask`` [B, 1, N*bs, S+N*bs] — same semantics as the
        additive :meth:`_build_dual_source_mask`, but flex skips the fully-masked key
        blocks instead of materializing the dense score matrix."""
        B, N = anchor_positions.shape
        bs = self.block_size
        Q = N * bs
        S = seq_len
        sw = getattr(self.draft_model, "sliding_window", None)
        sw = int(sw) if sw else 0

        def mask_mod(b, h, q_idx, kv_idx):
            qb = q_idx // bs
            safe_qb = qb.clamp(max=N - 1)
            anchor = anchor_positions[b, safe_qb]
            keep = block_keep_mask[b, safe_qb]
            p_abs = anchor + (q_idx - qb * bs)
            ctx = (kv_idx < S) & (kv_idx < anchor)
            if sw > 0:
                ctx = ctx & (kv_idx > (p_abs - sw))
            draft = (kv_idx >= S) & (qb == (kv_idx - S) // bs)
            return (ctx | draft) & keep & (qb < N)

        return create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=Q, KV_LEN=S + Q,
            device=device, BLOCK_SIZE=(32, 32),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """DSpark-V4 training forward.

        ``hidden_states`` is the concatenated target aux feature
        ``[B, S, len(target_layer_ids)*hidden]`` (the draft's ``main_proj`` /
        ``main_norm`` fuse it into ``main_x``). ``last_hidden_states`` is the
        target's final hidden ``[B, S, hidden]`` (L1 / confidence objectives).
        Returns the same 6-tuple as :class:`OnlineDSparkModel`.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        attention_backend = (_DRAFT_ATTN_OVERRIDE or self.attention_backend).lower()
        if attention_backend in ("flex", "flex_attention"):
            attn_mask = self._build_dual_source_block_mask(
                anchor_positions, block_keep_mask, seq_len, device
            )
        else:
            attn_mask = self._build_dual_source_mask(
                anchor_positions, block_keep_mask, seq_len, noise_embedding.dtype, device
            )

        draft_hidden = self.draft_model(
            target_hidden=hidden_states,
            noise_embedding=noise_embedding,
            context_position_ids=context_position_ids,
            draft_position_ids=draft_position_ids,
            attention_mask=attn_mask,
        )

        return self._dspark_objective(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            draft_hidden=draft_hidden,
            last_hidden_states=last_hidden_states,
        )
