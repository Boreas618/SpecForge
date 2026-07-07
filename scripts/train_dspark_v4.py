#!/usr/bin/env python3
# coding=utf-8
"""Faithful DeepSeek-V4-Flash-DSpark drafter training (1:1 with the release).

Trains :class:`DSparkV4DraftModel` — a from-scratch-trainable reconstruction of
the ``mtp.*`` draft inside ``deepseek-ai/DeepSeek-V4-Flash-DSpark`` (3 DeepSeek-V4
decoder blocks: shared-KV MLA + Manifold-Constrained Hyper-Connections + 256-expert
top-k MoE; learned ``hc_head``; Markov + confidence heads) against a
DeepSeek-V4-Flash target. The objective is DSpark's CE + L1 distribution
distillation + confidence BCE (shared with :class:`OnlineDSparkModel`), and the
conditioning matches the served ``precompute_and_store_context_kv`` forward.

Checkpoints are saved in the released ``mtp.*`` layout (via
``dspark_v4_state_dict_to_checkpoint``) so a trained drafter drops straight into
vLLM's ``DSparkDeepseekV4ForCausalLM`` (experts saved bf16 — the pre-quantization
form; fp4/fp8 packing is a downstream deploy step).
"""

import argparse
import functools
import json
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist

# Per-rank compile caches + generous PG timeouts (large MoE draft: kernel
# autotune stragglers otherwise trip the 10-min torch default).
_lr = os.environ.get("LOCAL_RANK", "0")
_cb = os.environ.get("SPECFORGE_RANK_CACHE_BASE", "/tmp/sf_caches")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{_cb}/inductor_rank{_lr}")
os.environ.setdefault("TRITON_CACHE_DIR", f"{_cb}/triton_rank{_lr}")
from datetime import timedelta as _td  # noqa: E402
from torch.distributed import distributed_c10d as _c10d  # noqa: E402

_ong, _oip = _c10d.new_group, _c10d.init_process_group


def _ng(*a, **k):
    k["timeout"] = _td(minutes=45)
    return _ong(*a, **k)


def _ip(*a, **k):
    k["timeout"] = _td(minutes=45)
    return _oip(*a, **k)


_c10d.new_group = _ng
_c10d.init_process_group = _ip
dist.new_group = _ng
dist.init_process_group = _ip

from accelerate.utils import set_seed  # noqa: E402
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: E402
from torch.distributed.fsdp import (  # noqa: E402
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from datasets import load_dataset  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from specforge.args import SGLangBackendArgs, TrackerArgs  # noqa: E402
from specforge.core.dspark_v4 import OnlineDSparkV4Model  # noqa: E402
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders  # noqa: E402
from specforge.distributed import (  # noqa: E402
    destroy_distributed,
    get_dp_group,
    init_distributed,
)
from specforge.modeling.draft.dspark_v4 import (  # noqa: E402
    DSparkV4DraftModel,
    build_dspark_v4_config,
    dspark_v4_state_dict_to_checkpoint,
)
from specforge.modeling.target.dflash_target_model import (  # noqa: E402
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead  # noqa: E402
from specforge.optimizer import BF16Optimizer  # noqa: E402
from specforge.tracker import create_tracker  # noqa: E402
from specforge.utils import (  # noqa: E402
    get_last_checkpoint,
    get_local_device,
    print_on_rank0,
    print_with_rank,
)

_DEFAULT_DRAFT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "deepseek-v4-flash-dspark.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train faithful DeepSeek-V4 DSpark draft")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend", type=str, default="sglang", choices=["sglang", "hf"]
    )
    model_group.add_argument(
        "--draft-config-path",
        type=str,
        default=_DEFAULT_DRAFT_CONFIG,
        help="Faithful DeepSeek-V4-Flash-DSpark draft config JSON.",
    )
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Noise/MASK token id. Defaults to config.dspark_noise_token_id (128799).",
    )
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--num-anchors", type=int, default=48)
    model_group.add_argument("--loss-decay-gamma", type=float, default=4.0)
    model_group.add_argument("--embedding-key", type=str, default=None)
    model_group.add_argument("--lm-head-key", type=str, default=None)

    dspark_group = parser.add_argument_group("dspark")
    dspark_group.add_argument("--ce-loss-alpha", type=float, default=0.1)
    dspark_group.add_argument("--l1-loss-alpha", type=float, default=0.9)
    dspark_group.add_argument("--confidence-head-alpha", type=float, default=1.0)

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="deepseek-v3")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=0)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=1)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=1024)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument("--max-steps", type=int, default=None)

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=8)
    output_group.add_argument("--save-interval", type=int, default=320)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    TrackerArgs.add_args(parser.add_argument_group("tracker"))
    parser.add_argument_group("distributed").add_argument(
        "--dist-timeout", type=int, default=30
    )
    SGLangBackendArgs.add_args(parser.add_argument_group("sglang backend"))
    return parser.parse_args()


def _load_draft_config_dict(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_models(args, device, config_dict) -> Tuple[DFlashTargetModel, DSparkV4DraftModel]:
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using "
        f"{args.target_model_backend} backend"
    )
    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    # For the sglang backend the target is a quantized (fp8 / fp4+fp8) DeepSeek-V4
    # checkpoint: forcing dtype=bfloat16 makes the loader mis-handle the native
    # quantization (bf16<->fp8 downcast errors, and mis-resolving the MoE method
    # for fp4-packed experts). Use "auto" so sglang respects the checkpoint's
    # quantization_config. The HF backend still needs a concrete compute dtype.
    target_torch_dtype = (
        torch.bfloat16 if args.target_model_backend == "hf" else "auto"
    )
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=target_torch_dtype,
        device=device.type if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    draft_config = build_dspark_v4_config(config_dict)
    draft_model = DSparkV4DraftModel(draft_config).to(device=device, dtype=torch.bfloat16)
    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"DSpark-V4 draft: layers={draft_config.num_hidden_layers} "
        f"experts={draft_config.n_routed_experts} block_size={draft_model.block_size} "
        f"target_layer_ids={draft_model.target_layer_ids} "
        f"markov_rank={draft_model.markov_rank} sliding_window={draft_model.sliding_window}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )
    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    import hashlib

    cache_key = hashlib.md5(
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-"
        f"{args.target_model_path}".encode()
    ).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_ds = build_eagle3_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_ds)
    train_ds = train_ds.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens,
        num_proc=args.build_dataset_num_proc,
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_ds)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_ds,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_ds = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_ds,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )
    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dspark_model, draft_model, optimizer, config_dict):
    """Save a checkpoint in the released ``mtp.*`` layout (vLLM-loadable)."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dspark_model, StateDictType.FULL_STATE_DICT):
        state_dict = dspark_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }
        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )
            # Native (HF-name) weights for exact resume of this trainer.
            save_file(
                {k: v.contiguous() for k, v in draft_state_dict.items()},
                os.path.join(save_dir, "model.safetensors"),
            )
            # Released mtp.* layout for vLLM deployment.
            ckpt = dspark_v4_state_dict_to_checkpoint(
                draft_state_dict, draft_model.config
            )
            save_file(ckpt, os.path.join(save_dir, "model.dspark_mtp.safetensors"))
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=2)
            modeling_dir = os.path.join(
                os.path.dirname(__file__), "..", "specforge", "modeling", "draft"
            )
            for fname in ("dspark_v4.py", "dspark.py"):
                src = os.path.join(modeling_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(save_dir, fname))
            print_on_rank0(f"Saved checkpoint to {save_dir}")
    dist.barrier()


def record_metrics(
    args, loss, accuracy, components, global_step, tracker, optimizer,
    train_dataloader=None, mode="train",
):
    logdict = {}
    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    for key, value in components.items():
        logdict[f"{mode}/{key}"] = value
    comp_str = " ".join(f"{k}={v:.4f}" for k, v in components.items())
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}, Loss: {loss:.4f}, "
        f"Acc: {accuracy:.4f}, {comp_str}"
    )
    tracker.log(logdict, step=global_step)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    device = get_local_device()
    device_type = device.type

    needs_target_hidden = (args.l1_loss_alpha > 0) or (args.confidence_head_alpha > 0)

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")
        cfg_path = os.path.join(draft_model_last_checkpoint or "", "config.json")
        if draft_model_last_checkpoint and os.path.exists(cfg_path):
            args.draft_config_path = cfg_path

    config_dict = _load_draft_config_dict(args.draft_config_path)
    # block_size is config-owned; expose it on args for dataloader filtering.
    args.block_size = int(config_dict.get("dspark_block_size", 5))

    target_model, draft_model = build_models(args, device, config_dict)

    resume_state = None
    if draft_model_last_checkpoint:
        sd = load_file(os.path.join(draft_model_last_checkpoint, "model.safetensors"))
        draft_model.load_state_dict(sd, strict=False)
        del sd
        print(f"Loaded draft weights from {draft_model_last_checkpoint}")
        ts_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")
        if os.path.exists(ts_path):
            resume_state = torch.load(ts_path, map_location="cpu", weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    mask_token_id = (
        args.mask_token_id
        if args.mask_token_id is not None
        else int(config_dict["dspark_noise_token_id"])
    )
    draft_model.mask_token_id = mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device_type,
        trust_remote_code=args.trust_remote_code,
    )

    dspark_model = OnlineDSparkV4Model(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend="eager",
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        ce_loss_alpha=args.ce_loss_alpha,
        l1_loss_alpha=args.l1_loss_alpha,
        confidence_head_alpha=args.confidence_head_alpha,
    )

    # FULL_SHARD is required here: BF16Optimizer builds its fp32 master + AdamW moments
    # from model.parameters(), which are only SHARDED under FULL_SHARD (~60GB/rank).
    # ZeRO-2/SHARD_GRAD_OP keeps params unsharded, so the optimizer would clone the full
    # 19.85B params -> ~237GB fp32 master+moments -> OOM alongside the resident target.
    # Instead recover FULL_SHARD's backward-overlap gap with prefetch: backward_prefetch
    # =BACKWARD_PRE + forward_prefetch=True overlap the all-gathers with compute (they
    # were fully serialized before, backward_prefetch=None — a big part of the slow bwd).
    # Sharding strategy (env SPECFORGE_FSDP_STRATEGY):
    #   full_shard (default): shards params+grads+opt. model.parameters() are sharded so
    #     BF16Optimizer's fp32 master+moments are ~60GB/rank. But every block's 40GB expert
    #     params are all-gathered in fwd AND re-all-gathered in bwd -> ~2s/step overhead
    #     (single-GPU draft is only ~0.65s; FSDP adds the rest).
    #   no_shard (DDP, DeepSpec's choice): params RESIDENT (no all-gather), grads all-reduced.
    #     Removes the all-gather overhead but model.parameters() are the FULL 19.85B, so
    #     BF16Optimizer clones ~237GB fp32 master+moments -> REQUIRES SPECFORGE_OFFLOAD_MASTER=1
    #     (keeps them on CPU; ~237GB host RAM, fine).
    _strat = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[os.environ.get("SPECFORGE_FSDP_STRATEGY", "full_shard")]
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
        ),
        sharding_strategy=_strat,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dspark_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=block_classes
        )
    dspark_model = FSDP(dspark_model, **fsdp_kwargs)
    print_with_rank("Initialized FSDP")

    start_epoch, global_step = ckpt_info

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
        offload_master=os.environ.get("SPECFORGE_OFFLOAD_MASTER", "0") == "1",
    )

    if resume_state is not None:
        if os.environ.get("SPECFORGE_RESUME_FULL_OPTIM") == "1":
            # Full resume incl. AdamW moments. Only correct if the checkpoint's flat-param
            # sharding matches this run's exactly; under FSDP FULL_SHARD the raw per-shard
            # AdamW state saved by BF16Optimizer does NOT reshard on a fresh wrap -> exp_avg
            # vs grad size mismatch at optimizer.step(). Leave unset unless save/load is
            # made FSDP-aware (FSDP.optim_state_dict).
            optimizer.load_state_dict(resume_state)
            print_on_rank0("Restored FULL optimizer + scheduler state")
        else:
            # Reshard-safe resume (default): restore the LR scheduler + step exactly, but
            # reset the Adam moments (they re-warm in ~tens of steps, negligible mid-run).
            # Avoids the FSDP flat-param reshard mismatch that crashes optimizer.step().
            optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            print_on_rank0("Restored LR scheduler + step; Adam moments reset (reshard-safe)")
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Resumed training state: epoch={start_epoch}, step={global_step}"
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)
    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")
    stop = False

    for epoch in range(start_epoch, args.num_epochs):
        if stop:
            break
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()
        progress_bar = (
            tqdm(train_dataloader, desc=f"Training Epoch {epoch}", leave=True)
            if dist.get_rank() == 0
            else train_dataloader
        )
        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            _prof = os.environ.get("SPECFORGE_PROFILE_STEP") == "1" and dist.get_rank() == 0

            def _psync(tag, t0):
                if _prof:
                    torch.cuda.synchronize()
                    print(f"[PROFILE step {global_step}] {tag}: {time.time()-t0:.2f}s", flush=True)
                    return time.time()
                return t0

            input_ids = data["input_ids"].to(device, non_blocking=True)
            attention_mask = data["attention_mask"].to(device, non_blocking=True)
            loss_mask = data["loss_mask"].to(device, non_blocking=True)
            _t = time.time()
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            _t = _psync("target_fwd", _t)
            hidden_states = target_output.hidden_states.to(device, non_blocking=True)
            last_hidden_states = target_output.last_hidden_states
            if last_hidden_states is not None:
                last_hidden_states = last_hidden_states.to(device, non_blocking=True)
            elif needs_target_hidden:
                raise RuntimeError(
                    "DSpark L1/confidence losses are enabled but the target backend "
                    f"({args.target_model_backend}) did not surface last_hidden_states. "
                    "Use --target-model-backend hf, or set --l1-loss-alpha 0 "
                    "--confidence-head-alpha 0."
                )

            (
                loss,
                accuracy,
                loss_per_position,
                acc_per_position,
                count_per_position,
                loss_components,
            ) = dspark_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                last_hidden_states=last_hidden_states,
            )
            _t = _psync("draft_fwd", _t)

            (loss / args.accumulation_steps).backward()
            _t = _psync("draft_bwd", _t)
            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log = loss_log / dist.get_world_size()
                acc_log = acc_log / dist.get_world_size()
                comp_log = {}
                for key, value in loss_components.items():
                    v = value.clone().float()
                    dist.all_reduce(v)
                    comp_log[key] = (v / dist.get_world_size()).item()
                record_metrics(
                    args, loss_log.item(), acc_log.item(), comp_log, global_step,
                    tracker, optimizer, train_dataloader, mode="train",
                )

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if global_step % args.save_interval == 0:
                save_checkpoint(
                    args, epoch, global_step, dspark_model, draft_model, optimizer,
                    config_dict,
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                print_on_rank0(f"Reached max_steps={args.max_steps}; stopping.")
                stop = True
                break

    save_checkpoint(
        args, args.num_epochs, global_step, dspark_model, draft_model, optimizer,
        config_dict,
    )
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
