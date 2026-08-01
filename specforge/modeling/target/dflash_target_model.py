import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.dp_attn import (
    prepare_mlp_sync_batch_raw,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_group

from .sglang_backend import SGLangRunner


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor  # [batch, seq_len, n_capture*hidden_size]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]
    # Target model's FINAL hidden state [batch, seq_len, hidden_size]. Optional:
    # DFlash never reads it, but DSpark's L1 distribution-distillation and
    # confidence-head losses need it (the frozen target LM head is applied to it
    # to form the soft next-token distribution). None when the backend does not
    # surface it (then DSpark must run CE-only).
    last_hidden_states: Optional[torch.Tensor] = None
    # When a radix-cache session is active (start_cache_session), hidden_states/
    # last_hidden_states cover only positions [prefix_len, seq_len) — the prefix
    # was served from KV cache and not recomputed. None outside sessions.
    prefix_len: Optional[int] = None
    # Engine-computed next-token logits for each request's LAST position
    # [batch, vocab]. Diagnostic passthrough: lets probes compare the engine's
    # own head against an externally loaded frozen head on the same hidden.
    next_token_logits: Optional[torch.Tensor] = None


class DFlashTargetModel(ABC):
    """
    Abstract base class for DFlash target model backend.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model_runner: SGLangRunner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetModel":
        tp_size = dist.get_world_size(get_tp_group())
        # Drop kwargs this sglang version's ServerArgs does not know (the
        # SGLangBackendArgs field set tracks a moving target).
        import dataclasses as _dc

        _valid = {f.name for f in _dc.fields(ServerArgs)}
        _dropped = sorted(k for k in kwargs if k not in _valid)
        # Silently dropping an EXPLICITLY-SET engine flag is how a required
        # backend knob (e.g. Inkling's mamba/page/quantization set) would
        # quietly not take effect — fail loud on those; dropping unset (None)
        # fields remains fine (they were never requested).
        _dropped_set = [k for k in _dropped if kwargs.get(k) is not None]
        if _dropped_set:
            raise ValueError(
                f"[SGLangDFlashTargetModel] this sglang's ServerArgs does not "
                f"accept explicitly-set engine args: {_dropped_set} — refusing "
                f"to silently drop them (values: "
                f"{ {k: kwargs[k] for k in _dropped_set} })"
            )
        if _dropped:
            print(f"[SGLangDFlashTargetModel] dropping unset ServerArgs: {_dropped}")
        kwargs = {k: v for k, v in kwargs.items() if k in _valid}
        # Optional escape hatch for ServerArgs fields SpecForge does not plumb
        # through its CLI (e.g. moe_runner_backend=deep_gemm for blockwise-FP8
        # MoE models on Hopper, where the triton fused path rejects the layout).
        import os as _os

        _moe_backend = _os.environ.get("SPECFORGE_SGLANG_MOE_RUNNER_BACKEND")
        if _moe_backend and "moe_runner_backend" in _valid:
            kwargs["moe_runner_backend"] = _moe_backend
        # Bound host-RAM at load. The default safetensors loader runs 8 threads/rank
        # (DEFAULT_NUM_THREADS) and every tp rank loads in parallel, so a large target
        # (e.g. 274GB DeepSeek-V4-Flash-FP8 at tp4) buffers many 6GB shards at once ->
        # a transient ~940GB host-RAM peak that trips OOM killers (earlyoom) even
        # though weights are mmap'd. Serialize the per-rank load (one shard at a time)
        # and drop the page cache after load so the peak stays bounded (~1 shard/rank).
        # Override with SPECFORGE_SGLANG_SERIAL_LOAD=0.
        if _os.environ.get("SPECFORGE_SGLANG_SERIAL_LOAD", "1") == "1":
            if "model_loader_extra_config" in _valid:
                kwargs.setdefault(
                    "model_loader_extra_config",
                    {"enable_multithread_load": False, "num_threads": 1},
                )
            if "weight_loader_drop_cache_after_load" in _valid:
                kwargs.setdefault("weight_loader_drop_cache_after_load", True)
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enable_return_hidden_states=True,  # Critical for DFlash
            disable_cuda_graph=True,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)

        # The Scheduler normally seeds the module-level MoE config (runner /
        # a2a backend globals) from server_args; constructing ModelRunner
        # directly skips that, leaving MOE_RUNNER_BACKEND stuck on AUTO.
        try:
            from sglang.srt.layers.moe import initialize_moe_config

            initialize_moe_config(server_args)
        except ImportError:
            pass

        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        # Newer sglang moved KV/req pool creation out of ModelRunner.__init__
        # into alloc_memory_pool() + init_attention_backends() + init_cuda_graphs()
        # (normally invoked by the Scheduler in that order). init_cuda_graphs also
        # builds the always-required eager_runner; with disable_cuda_graph=True the
        # actual graph capture is skipped.
        if (
            getattr(model_runner, "req_to_token_pool", None) is None
            and hasattr(model_runner, "alloc_memory_pool")
        ):
            model_runner.alloc_memory_pool()
            if hasattr(model_runner, "init_attention_backends"):
                model_runner.init_attention_backends()
            if hasattr(model_runner, "init_cuda_graphs"):
                model_runner.init_cuda_graphs()
        return cls(model_runner)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)
        model = self.model_runner.model
        # Prefer the DFlash capture hook: it applies a consistent +1 offset so the
        # captured aux stream is the OUTPUT of each requested layer id (== the HF
        # backend's hidden_states[idx+1] and the DSpark training semantics). The
        # eagle3 hook only special-cases layer_ids[0]==1, so it would MIS-capture a
        # spread that starts at a value != 1; the dflash hook is correct for any
        # spread (e.g. GLM-5.2 aux [1, 19, 38, 57, 76] -> capture layers [2,20,39,58,77]
        # = outputs of [1,19,38,57,76]). Fall back to eagle3 on older sglang.
        if hasattr(model, "set_dflash_layers_to_capture"):
            model.set_dflash_layers_to_capture(layer_ids)
        elif hasattr(model, "set_eagle3_layers_to_capture"):
            model.set_eagle3_layers_to_capture(layer_ids)
        inner = getattr(model, "model", None)
        # Ensure the target's FINAL post-norm hidden is surfaced for DSpark's L1 /
        # confidence losses. sglang's deepseek_v2/GlmMoeDsa path CONCATENATES the k
        # captured aux layers into output.hidden_states (width k*hidden) and does NOT
        # separately return the final hidden. Wrap the inner model's forward to append
        # the post-norm final hidden as the last aux entry, so the concat becomes
        # (k+1)*hidden and _extend splits off last_hidden_states (the (k+1)*hidden
        # branch). Idempotent; no-op if the forward already returns a non-tuple.
        self._ensure_final_hidden_appended(inner)

    def _ensure_final_hidden_appended(self, inner) -> None:
        if inner is None or getattr(inner, "_dspark_final_appended", False):
            return
        orig_forward = inner.forward

        def _forward_with_final(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            # Under aux capture the inner model returns (final_hidden, aux_list);
            # append the post-norm final as the last aux entry. Leave the no-capture
            # single-tensor return untouched.
            if isinstance(out, tuple) and len(out) == 2:
                hidden, aux = out
                if isinstance(aux, list) and hidden is not None:
                    return hidden, [*aux, hidden]
            return out

        inner.forward = _forward_with_final
        inner._dspark_final_appended = True

    def start_cache_session(self):
        """Enable persistent radix-cache prefix reuse across generate calls.

        For the speculative-decoding eval: each verify step re-submits the whole
        growing sequence, but with a persistent RadixCache the matched prefix is
        served from KV cache and only the new suffix (~block_size+1 tokens) is
        computed — DeepSpec's incremental-verify cost (their DynamicCache +
        crop()) on the tp-sharded sglang target, instead of an O(n^2) full
        re-prefill per step. Only committed tokens are inserted into the tree
        (``cache_commit_len``), so the computed suffix provably covers every
        position the verifier needs. Batch size must be 1 while a session is
        active. Call :meth:`end_cache_session` to release the cached KV.
        """
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        self._session_tree = RadixCache(cache_params)

    def end_cache_session(self):
        self._session_tree = None
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

    @torch.no_grad
    def _extend(self, reqs, cache_commit_len=None):
        session_tree = getattr(self, "_session_tree", None)
        if session_tree is not None:
            tree_cache = session_tree
        else:
            cache_params = CacheInitParams(
                disable=False,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
                page_size=self.model_runner.server_args.page_size,
            )
            tree_cache = RadixCache(cache_params)

        for req in reqs:
            # DP-attention expects one logprob-alignment token per request when
            # return_logprob=False. Start at the final prompt token so long
            # prompts do not look like full-prompt logprob requests.
            if not req.return_logprob:
                req.logprob_start_len = max(len(req.origin_input_ids) - 1, 0)
            req.init_next_round_input(tree_cache)
            # Admit the full request in one shot (what PrefillAdder does for
            # unchunked prefill). Newer sglang replaced fill_len/extend_input_len
            # with Req.extend_range; get_fill_ids() truncates by extend_range.end
            # (mirrors schedule_policy.py's unchunked admission).
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()

        if require_mlp_sync(self.model_runner.server_args):
            # sglang >=0.5.10: prepare_mlp_sync_batch_raw moved from a Scheduler
            # staticmethod to a free function in scheduler_components.dp_attn, and
            # the signature changed (dropped spec_algorithm / speculative_num_draft_
            # tokens; added the now-required attn_cp_size; attn_tp_size comes from
            # get_attention_tp_size() = tp_size // dp_size, i.e. 1 under full
            # DP-attention). This branch is only taken when DP-attention is enabled
            # (require_mlp_sync -> dp_size>1); the old call crashed against 0.5.14.
            prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=get_attention_tp_size(),
                attn_cp_size=getattr(self.model_runner, "attn_cp_size", 1),
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

        # Newer sglang: ForwardBatch.init_new consumes the ScheduleBatch
        # directly (no ModelWorkerBatch), and reads capture_hidden_mode from it.
        # prepare_for_extend stages tokens in pinned CPU memory; materialize the
        # device input_ids the way the scheduler does before building the FB.
        if (
            getattr(batch, "input_ids", None) is None
            and getattr(batch, "prefill_input_ids_cpu", None) is not None
        ):
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        output = self.model_runner.forward(forward_batch)
        if hasattr(output, "logits_output"):
            output = output.logits_output

        # Hidden states are returned only for COMPUTED tokens: with an empty
        # tree (no session) extend_range.length == len(origin_input_ids) and
        # this is identical to the old full-length split; with a session cache
        # the matched prefix is served from KV and the output covers the suffix.
        input_lens = [req.extend_range.length for req in reqs]
        prefix_lens = [len(req.prefix_indices) for req in reqs]
        # context = the captured (aux) mid-layer concat used by DFlash; final = the
        # post-norm last-layer hidden, surfaced for DSpark's L1 / confidence losses
        # (None if the runner only returned a single hidden stream).
        final_list = None
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            context_list = torch.split(output.aux_hidden_states, input_lens, dim=0)
            if hasattr(output, "hidden_states") and output.hidden_states is not None:
                final_list = torch.split(output.hidden_states, input_lens, dim=0)
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            # Newer sglang stores the aux concat INTO .hidden_states under
            # CaptureHiddenMode.FULL (no separate aux field). Our model patch
            # appends the final post-norm hidden as the last capture entry, so
            # the combined width is (k+1)*hidden — split context vs final here.
            hs = output.hidden_states
            k = len(self.capture_layer_ids or [])
            hidden_size = getattr(
                getattr(self.model_runner.model_config, "hf_config", None),
                "hidden_size",
                None,
            ) or getattr(self.model_runner.model_config, "hidden_size", None)
            if k and hidden_size and hs.shape[-1] == (k + 1) * hidden_size:
                context_list = torch.split(hs[..., : k * hidden_size], input_lens, dim=0)
                final_list = torch.split(hs[..., k * hidden_size :], input_lens, dim=0)
            else:
                context_list = torch.split(hs, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        if session_tree is not None and os.environ.get("SPECFORGE_KV_DEBUG") == "1":
            print(f"[KV] fill={reqs[0].extend_range.end} prefix={len(reqs[0].prefix_indices)} extend={reqs[0].extend_range.length} commit={cache_commit_len}", flush=True)
        if session_tree is not None:
            # Persist the KV of the COMMITTED prefix into the session tree so the
            # next call's prefix match serves it from cache. Only committed
            # tokens are inserted (cache_commit_len): the uncommitted tail (the
            # drafted block under verification) is freed, so a future match can
            # never extend past the committed length — the computed suffix is
            # guaranteed to cover every position the verifier reads.
            for req in reqs:
                fill_len = req.extend_range.end
                commit = (
                    fill_len
                    if cache_commit_len is None
                    else max(0, min(int(cache_commit_len), fill_len))
                )
                req.kv_committed_len = commit
                kv_row = self.model_runner.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :fill_len
                ]
                # The paged allocator frees WHOLE pages (unique(idx // page)):
                # cache_finished_req already freed [page_aligned(commit) : commit],
                # which covers the entire page containing `commit`. Freeing from
                # an unaligned `commit` would double-free that boundary page (the
                # same page later gets handed to two sequences). Start our tail
                # at the first page boundary strictly after commit-1.
                page = int(self.model_runner.server_args.page_size or 1)
                tail_start = ((commit - 1) // page + 1) * page if commit > 0 else 0
                tail = kv_row[tail_start:].to(dtype=torch.int64, copy=True)
                # Inserts kv[:commit] (freeing the already-cached duplicate
                # prefix), frees the page-unaligned remainder below `commit`,
                # and releases the prefix lock from init_next_round_input.
                tree_cache.cache_finished_req(req)
                if tail.numel():
                    self.model_runner.token_to_kv_pool_allocator.free(tail)
            # Req slots are transient (the tree holds copies of the kv indices).
            self.model_runner.req_to_token_pool.clear()
        else:
            self.model_runner.req_to_token_pool.clear()
            self.model_runner.token_to_kv_pool_allocator.clear()

        return context_list, final_list, prefix_lens

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        cache_commit_len: Optional[int] = None,
    ) -> DFlashTargetOutput:
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            from array import array as _array

            # Strip right padding (the collator pads right): prefilling pad
            # tokens through the target wastes prefill compute at batch > 1 and
            # produces garbage hiddens no one reads (loss_mask is 0 there and
            # the draft's block mask only attends context <= anchor < true_len).
            true_len = max(int(curr_attn.view(-1).sum().item()), 1)
            req = Req(
                rid=str(idx),
                origin_input_text="",
                # Newer sglang types origin_input_ids as array("q") and
                # concatenates it with array output_ids in _refresh_fill_ids.
                origin_input_ids=_array("q", curr_ids.view(-1)[:true_len].tolist()),
                sampling_params=sampling_params,
            )
            # fill_ids / extend_input_len are set via req.init_next_round_input()
            # inside _extend (the official Req prefill protocol in newer sglang).
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)

        context_list, final_list, prefix_lens = self._extend(
            reqs, cache_commit_len=cache_commit_len
        )

        if getattr(self, "_session_tree", None) is not None:
            # Session mode (spec-decode eval): batch of 1, return the computed
            # suffix directly with its offset — no re-padding to full length.
            assert len(context_list) == 1, "cache sessions require batch size 1"
            return DFlashTargetOutput(
                hidden_states=context_list[0].unsqueeze(0),
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                last_hidden_states=(
                    final_list[0].unsqueeze(0) if final_list is not None else None
                ),
                prefix_len=int(prefix_lens[0]),
            )

        # Stack back to batch, re-padding each row to the batch seq length
        # (requests were prefetched pad-stripped; zeros at pad positions are
        # never read — loss_mask is 0 there and blocks only attend <= anchor).
        seq_len = input_ids.size(1)

        def _repad(h: torch.Tensor) -> torch.Tensor:
            if h.size(0) == seq_len:
                return h
            pad = h.new_zeros(seq_len - h.size(0), h.size(1))
            return torch.cat([h, pad], dim=0)

        hidden_states = torch.stack([_repad(h) for h in context_list], dim=0)
        last_hidden_states = None
        if final_list is not None:
            last_hidden_states = torch.stack([_repad(h) for h in final_list], dim=0)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            last_hidden_states=last_hidden_states,
        )


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetModel":

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        if device:
            target_model = target_model.to(device)

        return cls(target_model)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output;
        # hidden_states[-1] = final (post-norm) hidden, i.e. the LM-head input.
        offset = 1
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                selected.append(outputs.hidden_states[idx + offset])
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1]

        # Final hidden state for DSpark's L1 / confidence losses (DFlash ignores it).
        last_hidden_states = outputs.hidden_states[-1]

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            last_hidden_states=last_hidden_states,
        )


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetModel:
    if backend == "sglang":
        return SGLangDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
