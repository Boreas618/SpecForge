# coding=utf-8
"""DSpark online training wrapper: DFlash backbone + Markov / L1 / confidence losses.

Ported from TorchSpec PR #129 (``torchspec/models/dspark.py``). Reuses SpecForge's
:class:`OnlineDFlashModel` anchor sampling, block-causal mask construction, and
MASK-token noise stream verbatim (via ``super()``), then layers on the DSpark
training objective:

  - Markov-biased draft logits (teacher-forced previous token).
  - Cross-entropy against the ground-truth next tokens (hard labels).
  - L1 distribution distillation: ``|softmax(draft) - softmax(target)|`` where the
    target distribution is the frozen LM head applied to the *target's* final
    hidden state at the aligned position (requires ``last_hidden_states``).
  - Confidence head BCE against the empirical per-token accept rate.

Combined: ``ce_alpha*ce + l1_alpha*l1 + confidence_alpha*confidence``.

Loss formulation adapted from DeepSeek's DeepSpec (``deepspec/modeling/dspark/loss.py``,
MIT), including its pooled global-mean reduction: local numerators over a
cross-rank all-reduced denominator, scaled by world_size to cancel FSDP's mean
gradient reduction.

Key SpecForge differences vs TorchSpec (see port notes in the PR):
  - SpecForge's :class:`OnlineDFlashModel.forward` returns ``(loss, accuracy)``;
    DSpark needs the per-component losses, so this forward returns a 6-tuple
    ``(loss, accuracy, loss_per_position, acc_per_position, count_per_position,
    loss_components)``. ``train_dspark.py`` consumes the extra elements.
  - The target ``lm_head`` is a frozen ``nn.Linear`` module on the wrapper
    (``self.lm_head``); the L1 path uses ``self.lm_head.weight`` for ``F.linear``.
  - The fused multi-layer context feature (``hidden_states``) is produced upstream
    by ``generate_dflash_data`` and fed straight to the draft as ``target_hidden``.
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _finite_report(name: str, t: Optional[torch.Tensor]) -> str:
    """One-line finiteness summary of a tensor (for non-finite loss debugging)."""
    if t is None:
        return f"{name}=None"
    td = t.detach()
    finite = torch.isfinite(td)
    n_bad = int((~finite).sum())
    if n_bad == 0:
        return f"{name}=finite"
    has_inf = bool(torch.isinf(td).any())
    has_nan = bool(torch.isnan(td).any())
    absmax = float(td[finite].abs().max()) if bool(finite.any()) else float("nan")
    return (
        f"{name}={n_bad}/{td.numel()} nonfinite "
        f"(inf={has_inf} nan={has_nan} finite_absmax={absmax:.3e})"
    )

from specforge.core.dflash import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDFlashModel,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dspark import DSparkDraftModel


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper (DFlash backbone + Markov/L1/confidence heads)."""

    def __init__(
        self,
        draft_model: DSparkDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 7,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = 4.0,
        ce_loss_alpha: float = 0.1,
        l1_loss_alpha: float = 0.9,
        confidence_head_alpha: float = 1.0,
    ):
        # Reuse DFlash anchor/mask/noise machinery. loss_type="dflash" is only a
        # placeholder to satisfy the parent validator — DSpark overrides forward()
        # entirely and never dispatches on loss_type.
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DeepSpec-exact anchor sampling (``deepspec/modeling/dspark/common.py``
        ``sample_anchor_positions``), overriding the inherited DFlash sampler.

        Differences vs the DFlash sampler this replaces:
          - candidates p in [0, seq_len-2] must satisfy loss_mask[p] AND
            loss_mask[p+1] (DeepSpec's first-target-valid rule) — every kept
            block has >= 1 supervised slot; the DFlash rule (loss_mask[p] only)
            wastes anchors on zero-supervision blocks at turn boundaries;
          - anchors within block_size of the sequence end are allowed (their
            blocks truncate via the eval mask) instead of excluded;
          - keep count per sample = min(valid_count, num_anchors) with no
            off-by-one (the DFlash sampler capped at batch_max_valid - 1).

        One disclosed deviation: DeepSpec always pads the anchor tensor to
        num_anchors columns; we cap the width at the batch max keep count.
        Dummy columns are fully masked either way (identical supervision and
        loss); padding them out only burns draft-MLP compute on short samples.
        """
        bsz = loss_mask.shape[0]
        num_candidates = max(seq_len - 1, 0)
        if num_candidates == 0:
            raise ValueError("seq_len < 2: no anchor candidates; preprocess the data.")
        valid = (loss_mask[:, :num_candidates] > 0.5) & (
            loss_mask[:, 1 : num_candidates + 1] > 0.5
        )
        valid_counts = valid.sum(dim=1)
        max_n = int(min(self.num_anchors, int(valid_counts.max().item())))
        if max_n <= 0:
            raise ValueError(
                "no valid anchors in batch (need loss_mask[p] & loss_mask[p+1]); "
                "preprocess the data."
            )
        indices = (
            torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.full_like(indices, seq_len + 1)
        )
        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(
            valid, random_vals, torch.full_like(random_vals, 2.0)
        )
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values
        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
            valid_counts.unsqueeze(1).clamp(max=max_n)
        )
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
        return anchors, keep_mask

    def _decay_weights(self, device: torch.device) -> torch.Tensor:
        """exp(-k/gamma) over within-block position k (DeepSpec convention).

        Every slot 0..B-1 is a real prediction in DSpark (unlike DFlash, where
        slot 0 is the masked anchor), so slot 0 (the first predicted token) gets
        weight 1.0 and later slots decay.
        """
        k = torch.arange(self.block_size, device=device).view(1, 1, -1)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            return torch.exp(-k.float() / self.loss_decay_gamma)
        return torch.ones_like(k, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """DSpark training forward.

        ``hidden_states`` is the fused multi-layer context feature
        ``[B, S, len(target_layer_ids)*hidden]`` (the draft model applies its
        ``fc``/``hidden_norm`` internally). ``last_hidden_states`` is the target
        model's final hidden state ``[B, S, hidden]`` (needed only for the L1 /
        confidence objectives).

        Returns ``(loss, accuracy, loss_per_position, acc_per_position,
        count_per_position, loss_components)``. ``loss`` is the combined
        ce+l1+confidence objective; ``loss_components`` is a dict of detached
        per-rank local-mean scalars (ce_loss / l1_loss / confidence_loss) for
        logging.
        """
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Non-finite inputs (captured target hidden states) poison the whole
        # objective: fc(inf) -> inf logits -> CE = +inf. Catch it at the source so
        # a bad FP8-target capture is not misdiagnosed as an objective/draft bug.
        if os.environ.get("SPECFORGE_DEBUG_NONFINITE", "1") == "1":
            for _nm, _t in (
                ("hidden_states(context)", hidden_states),
                ("last_hidden_states(target_final)", last_hidden_states),
            ):
                if _t is not None and not bool(torch.isfinite(_t).all()):
                    _r = dist.get_rank() if dist.is_initialized() else 0
                    print(
                        f"[NONFINITE-INPUT rank{_r}] {_finite_report(_nm, _t)} "
                        f"shape={tuple(_t.shape)}",
                        flush=True,
                    )

        # Opt-in robustness (default OFF): under genuine DP the gradient all-reduce
        # propagates one rank's non-finite sample to all ranks, so a single bad
        # capture over 1.5M samples can NaN the whole run. When enabled, drop
        # non-finite tokens from supervision (loss_mask -> 0) AND zero their hidden
        # so the inf cannot leak into good tokens through the draft's attention.
        if os.environ.get("SPECFORGE_SANITIZE_NONFINITE", "0") == "1":
            ctx_finite = torch.isfinite(hidden_states).all(dim=-1)  # [B, S]
            tgt_finite = (
                torch.isfinite(last_hidden_states).all(dim=-1)
                if last_hidden_states is not None
                else ctx_finite
            )
            bad_tok = ~(ctx_finite & tgt_finite)
            if bool(bad_tok.any()):
                _r = dist.get_rank() if dist.is_initialized() else 0
                print(
                    f"[SANITIZE rank{_r}] excluding {int(bad_tok.sum())}/{bad_tok.numel()} "
                    f"non-finite tokens from supervision",
                    flush=True,
                )
                loss_mask = loss_mask * (~bad_tok).to(loss_mask.dtype)
                _z = dict(nan=0.0, posinf=0.0, neginf=0.0)
                hidden_states = torch.nan_to_num(hidden_states, **_z)
                if last_hidden_states is not None:
                    last_hidden_states = torch.nan_to_num(last_hidden_states, **_z)

        # ---- DFlash backbone (identical construction to OnlineDFlashModel.forward) ----
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
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        draft_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        return self._dspark_objective(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            draft_hidden=draft_hidden,
            last_hidden_states=last_hidden_states,
        )

    def _dspark_objective(
        self,
        *,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        draft_hidden: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict,
    ]:
        """Shared DSpark objective over the drafted block hidden states.

        Given ``draft_hidden`` ``[B, n_blocks*block_size, hidden]`` (produced by
        any DSpark backbone — the DFlash/Qwen3 dense draft or the faithful
        DeepSeek-V4 draft), applies the frozen target ``lm_head`` + Markov head,
        and computes the combined ce + L1 + confidence objective with DeepSpec's
        pooled global-mean reduction. Backbone-agnostic: the anchor sampling,
        noise stream, and attention masks are the caller's responsibility; from
        the block hidden states onward the objective is identical across
        backbones, so both wrappers reuse this method (single source of truth).
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        n_blocks = anchor_positions.shape[1]
        hidden_4d = draft_hidden.view(bsz, n_blocks, self.block_size, -1)

        # ---- Labels + eval mask (DSpark / DeepSpec convention) ----
        # Slot j predicts the token at anchor+j+1 (the real anchor token seeds
        # slot 0). All block_size slots are supervised — there is no masked anchor
        # slot, unlike SpecForge DFlash which drops slot 0.
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, nb, bs]
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )  # [B, nb, bs]

        # eval mask = contiguous supervised prefix per block (DeepSpec
        # build_eval_mask): block kept, label in-bounds, target token supervised,
        # then cumprod so a gap truncates the rest of the block.
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_label_indices
        )
        eval_bool = (
            block_keep_mask.unsqueeze(-1) & valid_label_mask & (target_loss_mask > 0.5)
        )
        eval_bool = eval_bool.to(torch.int32).cumprod(dim=-1).bool()
        eval_mask = eval_bool.float()  # [B, nb, bs]

        decay_weight_mask = eval_mask * self._decay_weights(device)
        local_den = decay_weight_mask.sum()

        # prev token for slot j is the ground-truth token immediately before the
        # one slot j predicts: slot 0's prev is the real anchor token, slot j's is
        # target_ids[j-1]. Matches DeepSpec prev_token_ids.
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)  # [B, nb]
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )

        need_target = (self.l1_loss_alpha > 0) or (
            self.draft_model.confidence_head is not None
            and self.confidence_head_alpha > 0
        )
        aligned_hidden_4d = None
        if need_target:
            if last_hidden_states is None:
                raise ValueError(
                    "DSpark L1/confidence losses require target last_hidden_states; "
                    "ensure the target model surfaces its final hidden state."
                )
            # target distribution for the token at label_indices = target LM head
            # applied to the target hidden one position earlier (anchor+j).
            tgt_idx = (safe_label_indices - 1).clamp(min=0)  # [B, nb, bs]
            hdim = last_hidden_states.size(-1)
            gather_idx = tgt_idx.reshape(bsz, -1, 1).expand(-1, -1, hdim)
            aligned_hidden_4d = torch.gather(last_hidden_states, 1, gather_idx).view(
                bsz, n_blocks, self.block_size, hdim
            )

        # Numerators + metrics. The chunked path (default) processes the block dim
        # in slices with recompute-in-backward so the [B, nb, bs, V] float tensors
        # (V=155k) never materialize at full nb — the unchunked path peaks at
        # ~30 GB at nb=1024 and OOMs next to the sglang pool. Set
        # SPECFORGE_OBJECTIVE_CHUNK_BLOCKS=0 to force the legacy path.
        chunk_blocks = int(os.environ.get("SPECFORGE_OBJECTIVE_CHUNK_BLOCKS", "128"))
        numerators_fn = (
            self._chunked_numerators if chunk_blocks > 0 else self._full_numerators
        )
        (
            ce_num,
            l1_num,
            conf_num,
            probe_extras,
            correct_sum,
            ce_pp,
            acc_pp_correct,
            count_per_position,
        ) = numerators_fn(
            hidden_4d=hidden_4d,
            prev_token_ids=prev_token_ids,
            target_ids=target_ids,
            decay_weight_mask=decay_weight_mask,
            eval_mask=eval_mask,
            aligned_hidden_4d=aligned_hidden_4d,
            chunk_blocks=chunk_blocks,
        )
        self._probe_extras = probe_extras

        # ---- Pooled global loss (DeepSpec _build_loss) ----
        # Local numerators over a cross-rank-summed denominator, x world_size to
        # cancel FSDP's mean gradient reduction -> a true token-pooled global mean
        # rather than a mean-of-per-rank-means.
        # NOTE: uses the global training group size; correct for plain DP / ZeRO-2
        # (single shard group). With a multi-dim mesh (e.g. HSDP/USP) the FSDP
        # shard group differs from world_size and this would need the shard group.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_den = local_den.detach().clone()
        if world_size > 1:
            dist.all_reduce(global_den, op=dist.ReduceOp.SUM)
        global_den = global_den + 1e-6
        loss = (
            self.ce_loss_alpha * ce_num / global_den
            + self.l1_loss_alpha * l1_num / global_den
            + self.confidence_head_alpha * conf_num / global_den
        ) * world_size

        # Fires only when the loss is already broken -> no cost on the healthy
        # path. Pinpoints which numerator went non-finite and whether the draft
        # hidden / target hidden are the source (vs a degenerate denominator).
        if not bool(torch.isfinite(loss)):
            _r = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[NONFINITE-LOSS rank{_r}] loss={float(loss):.3e} "
                f"ce_num={float(ce_num):.3e} l1_num={float(l1_num):.3e} "
                f"conf_num={float(conf_num):.3e} "
                f"local_den={float(local_den):.3e} global_den={float(global_den):.3e} | "
                f"{_finite_report('draft_hidden', draft_hidden)} | "
                f"{_finite_report('last_hidden', last_hidden_states)}",
                flush=True,
            )

        # Per-component loss values (per-rank local means) for logging only — lets
        # you watch L1 fall while the greedy-CE proxy plateaus.
        local_den_eps = local_den + 1e-6
        loss_components = {
            "ce_loss": (ce_num / local_den_eps).detach(),
            "l1_loss": (l1_num / local_den_eps).detach(),
            "confidence_loss": (conf_num / local_den_eps).detach(),
        }
        loss_components.update(getattr(self, "_probe_extras", {}))
        self._probe_extras = {}

        # ---- Metrics (cross-entropy based; all block_size slots are productive) ----
        with torch.no_grad():
            accuracy = correct_sum.float() / eval_mask.sum().clamp(min=1e-6)
            count_pp = count_per_position.clamp(min=1.0)
            loss_per_position = ce_pp / count_pp
            acc_per_position = acc_pp_correct / count_pp

        return (
            loss,
            accuracy,
            loss_per_position,
            acc_per_position,
            count_per_position,
            loss_components,
        )

    def _chunk_terms(
        self,
        dh: torch.Tensor,  # [B, cb, bs, h] draft hidden slice (grad)
        prev_ids: torch.Tensor,  # [B, cb, bs]
        tids: torch.Tensor,  # [B, cb, bs]
        w: torch.Tensor,  # [B, cb, bs] decay*eval weights
        ev: torch.Tensor,  # [B, cb, bs] eval mask
        ah: Optional[torch.Tensor],  # [B, cb, bs, h] aligned target hidden or None
    ) -> Tuple[torch.Tensor, ...]:
        """Objective terms for one slice of the block dim.

        Ops match the legacy full-materialization path 1:1 (only summed over a
        slice instead of all blocks), so Σ over chunks reproduces the legacy
        numerators up to fp reassociation.
        """
        markov = self.draft_model.markov_head
        conf_head = self.draft_model.confidence_head
        use_conf = conf_head is not None and self.confidence_head_alpha > 0
        hdim = dh.size(-1)

        # NOTE: all vocab-sized linears are flattened to 2D first. On sliced 4D
        # inputs (chunk views) cuBLAS picks a degenerate batched-GEMM kernel
        # (M=block_size per batch) that is ~35x slower than the flat GEMM.
        base = F.linear(dh.reshape(-1, hdim), self.lm_head.weight).view(
            *dh.shape[:-1], -1
        )
        lg = base
        if markov is not None:
            # == markov.apply_block_logits(base, token_ids=prev_ids), flattened.
            bias = markov.project_bias(
                markov.get_prev_embeddings(prev_ids.reshape(-1))
            ).view_as(base)
            lg = base + bias
        vocab_size = lg.size(-1)
        ce = F.cross_entropy(
            lg.reshape(-1, vocab_size), tids.reshape(-1), reduction="none"
        ).view(tids.shape)
        ce_num = (ce * w).sum()

        zero = base.new_zeros((), dtype=torch.float32)
        l1_num, conf_num = zero, zero
        accept_rate = None
        p = q = t_arg = None
        if ah is not None:
            with torch.no_grad():
                tl = (
                    F.linear(ah.reshape(-1, hdim), self.lm_head.weight)
                    .view_as(lg)
                    .float()
                )
                q = torch.softmax(tl, dim=-1)
                t_arg = tl.argmax(-1)
            p = torch.softmax(lg.float(), dim=-1)
            l1_tok = (p - q).abs().sum(dim=-1)
            if self.l1_loss_alpha > 0:
                l1_num = (l1_tok * w).sum()
            accept_rate = (1.0 - 0.5 * l1_tok).clamp(0.0, 1.0)
        if use_conf:
            feats = dh
            if self.draft_model.confidence_head_with_markov:
                prev_emb = markov.get_prev_embeddings(prev_ids).to(dh.dtype)
                feats = torch.cat([dh, prev_emb], dim=-1)
            conf_pred = conf_head(feats).float()
            conf_num = (
                F.binary_cross_entropy_with_logits(
                    conf_pred, accept_rate.detach(), reduction="none"
                )
                * w
            ).sum()

        with torch.no_grad():
            pred = lg.argmax(-1)
            correct = ((pred == tids) & (ev > 0.5)).float()
            correct_sum = correct.sum()
            ce_pp = (ce.detach() * ev).sum(dim=(0, 1))  # [bs]
            acc_pp = correct.sum(dim=(0, 1))  # [bs]
            if ah is not None:
                agree = ((pred == t_arg).float() * ev).sum()
                ttop = (q.max(-1).values * ev).sum()
                dtop = (p.max(-1).values * ev).sum()
                # DeepSpec tau_probabilistic: expected accepted drafts per block
                # (+1 bonus token), over blocks with >= 1 supervised slot.
                vw = (ev.max(dim=-1).values > 0.5).float()  # [B, cb]
                var = accept_rate.detach() * ev
                tau_num = ((var.cumprod(dim=-1).sum(dim=-1) + 1.0) * vw).sum()
                tau_den = vw.sum()
            else:
                agree = ttop = dtop = tau_num = tau_den = zero
        return (
            ce_num, l1_num, conf_num, correct_sum, ce_pp, acc_pp,
            agree, ttop, dtop, tau_num, tau_den,
        )

    def _chunked_numerators(
        self,
        *,
        hidden_4d: torch.Tensor,
        prev_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        decay_weight_mask: torch.Tensor,
        eval_mask: torch.Tensor,
        aligned_hidden_4d: Optional[torch.Tensor],
        chunk_blocks: int,
    ):
        """Chunked objective: slice the block dim, recompute in backward.

        Peak memory ~= one chunk's [B, cb, bs, V] tensors (~2-3 GB at cb=128)
        instead of the full-nb ~30 GB stack; checkpointing saves only the chunk
        INPUTS ([B, cb, bs, h] views — negligible) so backward memory drops the
        same way. Recompute cost is one extra chunk forward (~ms; the run is
        target-prefill-bound).
        """
        import torch.utils.checkpoint as _ckpt

        n_blocks = hidden_4d.shape[1]
        use_ckpt = torch.is_grad_enabled() and hidden_4d.requires_grad
        acc = None
        for s in range(0, n_blocks, chunk_blocks):
            e = min(s + chunk_blocks, n_blocks)
            chunk_args = (
                hidden_4d[:, s:e],
                prev_token_ids[:, s:e],
                target_ids[:, s:e],
                decay_weight_mask[:, s:e],
                eval_mask[:, s:e],
                aligned_hidden_4d[:, s:e] if aligned_hidden_4d is not None else None,
            )
            if use_ckpt:
                out = _ckpt.checkpoint(
                    self._chunk_terms, *chunk_args, use_reentrant=False
                )
            else:
                out = self._chunk_terms(*chunk_args)
            acc = out if acc is None else tuple(a + o for a, o in zip(acc, out))

        (
            ce_num, l1_num, conf_num, correct_sum, ce_pp, acc_pp,
            agree, ttop, dtop, tau_num, tau_den,
        ) = acc
        probe_extras = {}
        if aligned_hidden_4d is not None:
            with torch.no_grad():
                _den = eval_mask.sum().clamp(min=1.0)
                probe_extras = {
                    "agree_teacher": agree / _den,
                    "teacher_top1_prob": ttop / _den,
                    "draft_top1_prob": dtop / _den,
                    "tau_probabilistic": tau_num / tau_den.clamp(min=1.0),
                }
        count_per_position = eval_mask.sum(dim=(0, 1))
        return (
            ce_num,
            l1_num,
            conf_num,
            probe_extras,
            correct_sum,
            ce_pp,
            acc_pp,
            count_per_position,
        )

    def _full_numerators(
        self,
        *,
        hidden_4d: torch.Tensor,
        prev_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        decay_weight_mask: torch.Tensor,
        eval_mask: torch.Tensor,
        aligned_hidden_4d: Optional[torch.Tensor],
        chunk_blocks: int,  # unused; signature parity
    ):
        """Legacy full-materialization path (SPECFORGE_OBJECTIVE_CHUNK_BLOCKS=0).

        Materializes the full [B, nb, bs, V] logits/probs stack — needs ~30 GB
        transient at nb=1024/V=155k. Kept as a fallback and as the reference for
        the chunked path's equivalence test.
        """
        del chunk_blocks  # signature parity with _chunked_numerators
        bsz, n_blocks = hidden_4d.shape[:2]
        base_logits_4d = self.lm_head(
            hidden_4d.reshape(bsz, n_blocks * self.block_size, -1)
        ).view(bsz, n_blocks, self.block_size, -1)
        vocab_size = base_logits_4d.size(-1)

        logits_4d = base_logits_4d
        if self.draft_model.markov_head is not None:
            logits_4d = self.draft_model.markov_head.apply_block_logits(
                base_logits_4d, token_ids=prev_token_ids
            )

        flat_logits = logits_4d.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        ce_per_token = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        ).view(bsz, n_blocks, self.block_size)
        ce_num = (ce_per_token * decay_weight_mask).sum()

        zero = base_logits_4d.new_zeros((), dtype=torch.float32)
        l1_num, conf_num = zero, zero
        accept_rate = None
        probe_extras = {}
        if aligned_hidden_4d is not None:
            aligned_target_logits = F.linear(
                aligned_hidden_4d, self.lm_head.weight
            )
            draft_probs = torch.softmax(logits_4d.float(), dim=-1)
            target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
            l1_per_token = (draft_probs - target_probs).abs().sum(dim=-1)
            accept_rate = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
            with torch.no_grad():
                _den = eval_mask.sum().clamp(min=1.0)
                _t_arg = aligned_target_logits.argmax(-1)
                _vw = (eval_mask.max(dim=-1).values > 0.5).float()
                _var = accept_rate.detach() * eval_mask
                probe_extras = {
                    "agree_teacher": (
                        ((logits_4d.argmax(-1) == _t_arg).float() * eval_mask).sum()
                        / _den
                    ),
                    "teacher_top1_prob": (
                        (target_probs.max(-1).values * eval_mask).sum() / _den
                    ),
                    "draft_top1_prob": (
                        (draft_probs.max(-1).values * eval_mask).sum() / _den
                    ),
                    "tau_probabilistic": (
                        ((_var.cumprod(dim=-1).sum(dim=-1) + 1.0) * _vw).sum()
                        / _vw.sum().clamp(min=1.0)
                    ),
                }
            if self.l1_loss_alpha > 0:
                l1_num = (l1_per_token * decay_weight_mask).sum()

        if (
            self.draft_model.confidence_head is not None
            and self.confidence_head_alpha > 0
        ):
            if self.draft_model.confidence_head_with_markov:
                prev_emb = self.draft_model.markov_head.get_prev_embeddings(
                    prev_token_ids
                ).to(hidden_4d.dtype)
                conf_features = torch.cat([hidden_4d, prev_emb], dim=-1)
            else:
                conf_features = hidden_4d
            confidence_pred = self.draft_model.confidence_head(conf_features).float()
            conf_num = (
                F.binary_cross_entropy_with_logits(
                    confidence_pred, accept_rate.detach(), reduction="none"
                )
                * decay_weight_mask
            ).sum()

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            flat_binary = eval_mask.reshape(-1)
            correct = (pred_ids == flat_targets) & (flat_binary > 0.5)
            correct_sum = correct.sum()
            correct_3d = correct.view(bsz, n_blocks, self.block_size).float()
            ce_pp = (ce_per_token.detach() * eval_mask).sum(dim=(0, 1))
            acc_pp = correct_3d.sum(dim=(0, 1))
            count_per_position = eval_mask.sum(dim=(0, 1))

        return (
            ce_num,
            l1_num,
            conf_num,
            probe_extras,
            correct_sum,
            ce_pp,
            acc_pp,
            count_per_position,
        )
