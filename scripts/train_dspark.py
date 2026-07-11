#!/usr/bin/env python3
# coding=utf-8
"""DSpark Training Script.

DSpark = DFlash block-diffusion drafter + EAGLE-style Markov & confidence heads,
trained with cross-entropy + L1 distribution distillation + confidence BCE. The
L1 / confidence terms need the target model's FINAL hidden state, so the target
backend must surface it (HF always does; sglang does when it returns both the
captured aux stream and the final hidden state). Set ``--l1-loss-alpha 0`` and
``--no-confidence-head`` to train CE-only without the target final hidden state.

Cloned from ``scripts/train_dflash.py`` and adapted: builds a DSparkDraftModel +
OnlineDSparkModel, plumbs ``last_hidden_states`` into the forward, and logs the
per-component (ce / l1 / confidence) losses.
"""

import argparse
import functools
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist

# Per-rank compile caches + generous PG timeouts: a tp=16 sglang target load (~753B
# GLM-5.2-FP8) and the first flex/torch.compile autotune can exceed torch's 10-min
# default collective timeout on multi-node. Ported from train_dspark_v4.py.
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
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dspark import OnlineDSparkModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
)
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import (
    get_last_checkpoint,
    get_local_device,
    print_on_rank0,
    print_with_rank,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSpark Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local). "
        "DSpark's L1/confidence losses need the target's final hidden state; "
        "the 'hf' backend always surfaces it.",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=4.0,
        help="Gamma for exponential within-block loss decay (exp(-k/gamma), "
        "k = within-block slot index). None disables.",
    )
    model_group.add_argument(
        "--embedding-key",
        type=str,
        default=None,
        help="Embedding weight key in the target model. "
        "Default: 'model.embed_tokens.weight' for standard models.",
    )
    model_group.add_argument(
        "--lm-head-key",
        type=str,
        default=None,
        help="LM head weight key in the target model. Default: 'lm_head.weight'.",
    )

    # DSpark-specific knobs
    dspark_group = parser.add_argument_group("dspark")
    dspark_group.add_argument(
        "--markov-rank",
        type=int,
        default=256,
        help="Rank of the low-rank Markov (bigram) bias head. 0 disables it.",
    )
    dspark_group.add_argument(
        "--markov-head-type", type=str, default="vanilla", choices=["vanilla"]
    )
    dspark_group.add_argument(
        "--enable-confidence-head",
        action="store_true",
        default=True,
        help="Enable the per-position accept-rate (confidence) head.",
    )
    dspark_group.add_argument(
        "--no-confidence-head",
        dest="enable_confidence_head",
        action="store_false",
        help="Disable the confidence head.",
    )
    dspark_group.add_argument(
        "--confidence-head-with-markov",
        action="store_true",
        default=True,
        help="Fuse the Markov prev-token embedding into the confidence features.",
    )
    dspark_group.add_argument(
        "--ce-loss-alpha", type=float, default=0.1, help="Weight on cross-entropy."
    )
    dspark_group.add_argument(
        "--l1-loss-alpha",
        type=float,
        default=0.9,
        help="Weight on L1 distribution distillation (needs target last hidden).",
    )
    dspark_group.add_argument(
        "--confidence-head-alpha",
        type=float,
        default=1.0,
        help="Weight on the confidence-head BCE (needs target last hidden).",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument(
        "--eval-datasets-dir",
        type=str,
        default=None,
        help="Directory of DeepSpec accept-length benchmark jsonl files. If set, "
        "runs the DeepSpec-style mean-accepted-length eval (greedy) every "
        "--eval-interval steps, reusing the loaded sglang target. Off if unset.",
    )
    dataset_group.add_argument(
        "--eval-limit-per-task",
        type=int,
        default=32,
        help="Max prompts per benchmark for the in-loop accept-length eval.",
    )
    dataset_group.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens per prompt for the in-loop accept-length eval. "
        "1024 (not 256) so a thinking-ON generation's reasoning chain is not "
        "fully truncated; the KV-reuse verify keeps it affordable. tau_"
        "probabilistic remains the cheap per-step proxy between decoded evals.",
    )
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="If set, stop after this many optimizer steps (smoke testing).",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument(
        "--evals-per-epoch",
        type=int,
        default=None,
        help="If set, overrides --eval-interval so the accept-length eval runs "
        "this many times per epoch (eval_interval = micro_steps_per_epoch // N).",
    )
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def _apply_dspark_config(draft_config, args) -> None:
    """Set DSpark head fields on the draft config, preferring values already in
    the config JSON and falling back to CLI args."""
    defaults = {
        "markov_rank": args.markov_rank,
        "markov_head_type": args.markov_head_type,
        "enable_confidence_head": args.enable_confidence_head,
        "confidence_head_with_markov": args.confidence_head_with_markov,
    }
    for key, value in defaults.items():
        if not hasattr(draft_config, key) or getattr(draft_config, key) is None:
            setattr(draft_config, key, value)


def build_models(args, device) -> Tuple[DFlashTargetModel, DSparkDraftModel]:
    """Build target model (backend wrapper) and DSpark draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using "
        f"{args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    device_type = device.type

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device=device_type if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: config block_size ({draft_config.block_size}) differs from "
                f"command-line arg ({args.block_size}). Using config value."
            )
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    _apply_dspark_config(draft_config, args)
    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DSparkDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}, "
        f"markov_rank={getattr(draft_config, 'markov_rank', 0)}, "
        f"enable_confidence_head={getattr(draft_config, 'enable_confidence_head', False)}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    # Bump when the chat template / loss-mask logic changes so the processed-
    # dataset cache invalidates. The base key uses only the template NAME, so a
    # template *content* change (e.g. the GLM thinking-ON mask fix that recovered
    # ~50% of samples) would otherwise silently reuse the stale tokenized/masked
    # cache. v2 = GLM thinking-ON hybrid loss mask.
    mask_logic_version = "maskv2-glm-thinkhybrid"
    cache_params_string = (
        f"{args.train_data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}-"
        f"{mask_logic_version}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
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
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens,
        num_proc=args.build_dataset_num_proc,
    )
    retained = len(train_eagle3_dataset)
    frac = retained / max(original_size, 1)
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {retained} samples "
        f"({100*frac:.1f}% retained)"
    )
    # Guard against a silent chat-template/loss-mask mismatch. A large drop here
    # usually means the assistant_pattern did not match the rendered turns (e.g.
    # the GLM thinking-ON header regression that zero-masked ~50% of the corpus),
    # not genuinely short samples. Fail loud rather than train on half the data.
    _min_frac = float(os.environ.get("SPECFORGE_MIN_RETENTION", "0.9"))
    if frac < _min_frac:
        raise RuntimeError(
            f"Only {100*frac:.1f}% of samples survived the loss-mask filter "
            f"(< {100*_min_frac:.0f}%). This almost always means the chat "
            f"template's assistant_pattern does not match the rendered assistant "
            f"turns (zero loss mask -> filtered), NOT that samples are too short. "
            f"Inspect a rendered sample vs parser.assistant_pattern. Override with "
            f"SPECFORGE_MIN_RETENTION=0 only if the drop is genuinely expected."
        )

    # Under sglang DP-attention the target runs data-parallel across ALL ranks (each
    # rank forwards a distinct shard), so the draft must see a distinct shard per rank
    # too -> shard over the whole world (process_group=None -> world-wide sampler).
    # Without DP-attention the target is TP-replicated and ranks in a TP group must
    # consume identical data, so we shard over the DP group only. The DSpark
    # pooled-global-mean objective (core/dspark.py) is correct for both. Mirrors
    # train_dspark_v4.py.
    use_dp_attention = (
        args.target_model_backend == "sglang" and args.sglang_enable_dp_attention
    )
    data_parallel_group = None if use_dp_attention else get_dp_group()

    # Load data IN-PROCESS (num_workers=0). A fork()ed DataLoader worker inherits
    # wandb's public-API service object from the wandb-initialised main process;
    # its weakref finalizer fires on the first in-worker GC (mid-batch, while
    # holding the import lock) and blocks forever on the dead service socket. That
    # worker never yields its batch -> the rank stalls at the dataloader -> the
    # target's tp collective on the peer ranks deadlocks. (Exactly why the
    # REPORT_TO=none smoke passed but the REPORT_TO=wandb run hung at step ~47.)
    # `spawn` workers avoid the inherit but re-exec this heavy module per worker
    # and are fragile here; the workload is target-prefill-bound and the data is
    # pre-tokenised (Arrow mmap), so in-process loading costs ~1-2%/step.
    num_workers = 0
    if args.dataloader_num_workers > 0:
        print_on_rank0(
            f"DSpark: forcing dataloader num_workers=0 (requested "
            f"{args.dataloader_num_workers}) to avoid the wandb + forked-worker "
            f"deadlock; data is pre-tokenised so the cost is negligible."
        )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=num_workers,
        shuffle=True,
        process_group=data_parallel_group,
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=num_workers,
            shuffle=False,
            process_group=data_parallel_group,
        )

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dspark_model, draft_model, optimizer):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dspark_model, StateDictType.FULL_STATE_DICT):
        state_dict = dspark_model.state_dict()
        # Strip both the torch.compile wrapper prefix (_orig_mod.) and the
        # OnlineDSparkModel wrapper prefix (draft_model.) so the saved keys match
        # a bare DSparkDraftModel on reload. Missing the _orig_mod. strip silently
        # produces a checkpoint whose keys no reload can match -> random resume.
        draft_state_dict = {
            k.replace("_orig_mod.", "").replace("draft_model.", ""): v
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

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            # Copy the modeling files next to the checkpoint so auto_map can
            # resolve DSparkDraftModel (which subclasses DFlashDraftModel) on
            # reload with trust_remote_code.
            modeling_dir = os.path.join(
                os.path.dirname(__file__), "..", "specforge", "modeling", "draft"
            )
            for fname in ("dspark.py", "dflash.py"):
                src = os.path.join(modeling_dir, fname)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(save_dir, fname))

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    components: dict,
    global_step: int,
    tracker,
    optimizer,
    train_dataloader=None,
    mode: str = "train",
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    for key, value in components.items():
        logdict[f"{mode}/{key}"] = value

    comp_str = " ".join(f"{k}={v:.4f}" for k, v in components.items())
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}"
        f"[{global_step}/{args.num_epochs * len(train_dataloader) // args.accumulation_steps}?],"
        f" Loss: {loss:.4f}, Acc: {accuracy:.4f}, {comp_str}"
    )

    tracker.log(logdict, step=global_step)


def _maybe_run_accept_length_eval(
    args, dspark_model, draft_model, target_model, target_components, tokenizer,
    tracker, global_step,
):
    """Best-effort in-loop DeepSpec accept-length eval (greedy), reusing the loaded
    sglang target. Guarded: only runs when --eval-datasets-dir is set, and any
    failure is swallowed so it can never kill a long training run. The standalone
    scripts/eval_dspark_deepspec.py is the primary/validated eval path."""
    if not args.eval_datasets_dir:
        return
    import importlib.util

    try:
        spec_path = os.path.join(os.path.dirname(__file__), "eval_dspark_deepspec.py")
        spec = importlib.util.spec_from_file_location("eval_dspark_deepspec", spec_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        device = next(draft_model.parameters()).device
        with FSDP.summon_full_params(dspark_model, recurse=True, writeback=False):
            draft_model.eval()
            result = mod.run_deepspec_eval(
                target_model=target_model,
                draft_model=draft_model,
                target_lm_head=target_components.lm_head,
                target_embed_tokens=target_components.embed_tokens,
                tokenizer=tokenizer,
                tasks=None,
                eval_datasets_dir=args.eval_datasets_dir,
                limit_per_task=args.eval_limit_per_task,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=0.0,
                device=device,
                verbose=(dist.get_rank() == 0),
            )
        draft_model.train()
        if dist.get_rank() == 0 and result:
            overall = result.get("overall", {})
            logd = {}
            mal = overall.get("mean_accepted_length")
            if mal is not None:
                logd["eval/mean_accepted_length"] = mal
            for task, m in result.get("per_dataset", {}).items():
                v = m.get("mean_accepted_length")
                if v is not None:
                    logd[f"eval/{task}/accept_len"] = v
            if logd:
                tracker.log(logd, step=global_step)
            print_on_rank0(f"[accept-length eval @ step {global_step}] {overall}")
    except Exception as e:  # noqa: BLE001
        # Do NOT silently swallow: print the full traceback so eval failures are
        # visible in the logs. Training still continues (a periodic-eval failure
        # must not kill a multi-day run), but the problem is not hidden.
        import traceback

        print_on_rank0(
            f"[accept-length eval] FAILED at step {global_step} (training "
            f"continues) — {type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
        try:
            draft_model.train()
        except Exception:
            pass


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

    needs_target_hidden = (args.l1_loss_alpha > 0) or (
        args.enable_confidence_head and args.confidence_head_alpha > 0
    )

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)
        print(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(
            draft_model_last_checkpoint, "config.json"
        )
        if os.path.exists(checkpoint_config_path):
            print(f"Loading draft config from checkpoint: {checkpoint_config_path}")
            args.draft_config_path = checkpoint_config_path

    target_model, draft_model = build_models(args, device)

    resume_state = None
    if draft_model_last_checkpoint:
        # Load weights straight from the checkpoint's safetensors, stripping any
        # _orig_mod. (torch.compile wrapper) / draft_model. prefixes first, then
        # validate. Going through from_pretrained silently drops keys carrying a
        # _orig_mod. prefix (compile-era checkpoints) -> a fully random "resume".
        import glob as _glob

        from safetensors.torch import load_file as _load_sft

        _sft = sorted(
            _glob.glob(os.path.join(draft_model_last_checkpoint, "*.safetensors"))
        )
        if not _sft:
            raise FileNotFoundError(
                f"No .safetensors found in checkpoint {draft_model_last_checkpoint}"
            )
        _sd = {}
        for _f in _sft:
            for _k, _v in _load_sft(_f).items():
                _ck = _k.replace("_orig_mod.", "").replace("draft_model.", "")
                _sd[_ck] = _v.to(torch.bfloat16)
        _missing, _unexpected = draft_model.load_state_dict(_sd, strict=False)
        _n_expected = len(draft_model.state_dict())
        if len(_missing) >= _n_expected:
            raise RuntimeError(
                f"Resume matched 0 params from {draft_model_last_checkpoint} "
                f"({len(_missing)} missing / {len(_unexpected)} unexpected) — "
                f"checkpoint key-prefix mismatch. Refusing to train a random draft."
            )
        if _missing or _unexpected:
            print(
                f"Resume: {len(_missing)} missing / {len(_unexpected)} unexpected keys "
                f"(loaded {_n_expected - len(_missing)}/{_n_expected})"
            )
        print("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch
    print_on_rank0(f"Total training steps: {total_steps}")

    # eval cadence: --evals-per-epoch overrides --eval-interval (in micro-steps, the
    # unit global_step counts). e.g. 10 evals/epoch => every len(dataloader)//10 steps.
    if args.evals_per_epoch:
        args.eval_interval = max(1, len(train_dataloader) // args.evals_per_epoch)
        print_on_rank0(
            f"eval every {args.eval_interval} micro-steps "
            f"(~{args.evals_per_epoch} evals/epoch)"
        )

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device_type,
        trust_remote_code=args.trust_remote_code,
    )

    dspark_model = OnlineDSparkModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        ce_loss_alpha=args.ce_loss_alpha,
        l1_loss_alpha=args.l1_loss_alpha,
        confidence_head_alpha=args.confidence_head_alpha,
    )

    # Wrap each transformer block as its own FSDP unit (compute/comm overlap).
    # Sharding strategy (env SPECFORGE_FSDP_STRATEGY): the ~3.8B dense draft is small,
    # so shard_grad_op (default) keeps params resident (no fwd/bwd all-gather) while
    # sharding grads+optimizer -> fast and fits easily beside the 47GB/rank FP8 target.
    # full_shard / no_shard available for larger drafts or debugging.
    _strat = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[os.environ.get("SPECFORGE_FSDP_STRATEGY", "shard_grad_op")]
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=_strat,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {
        type(m) for m in dspark_model.modules() if type(m).__name__ in block_names
    }
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=block_classes,
        )
    else:
        print_with_rank(
            "No _no_split_modules on draft model; falling back to single-unit "
            "FSDP wrap (no compute-comm overlap)."
        )
    dspark_model = FSDP(dspark_model, **fsdp_kwargs)
    print_with_rank("Initialized FSDP")

    # torch.compile the (FSDP-wrapped) model. Compiled AFTER FSDP is the supported
    # ordering (FSDP comm hooks stay outermost; the draft decoder blocks + flex
    # attention compile, while the objective's data-dependent ops — all_reduce,
    # .item(), the vocab-softmax — graph-break cleanly). dynamic=True: draft
    # Q/context lengths vary per batch (data-dependent num_anchors). Enabled by
    # SPECFORGE_COMPILE_DRAFT (default OFF, incl. the GLM run script); set =1 to
    # enable only after the FSDP-recompile crash is fixed.
    if os.environ.get("SPECFORGE_COMPILE_DRAFT", "0") == "1":
        dspark_model = torch.compile(dspark_model, dynamic=True)
        print_with_rank("Applied torch.compile to dspark_model (SPECFORGE_COMPILE_DRAFT=1)")

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
        offload_master=os.environ.get("SPECFORGE_OFFLOAD_MASTER", "0") == "1",
    )

    # Persistent LR scale (SPECFORGE_LR_SCALE, default 1.0 = the DeepSpec-parity
    # schedule). The converged drafter went edge-of-stability at the schedule's
    # peak LR twice (collapse onset within ~3 optimizer steps of running at
    # 5.76e-4, at two different data positions; stable through the entire damped
    # re-warm both times) -> run the same cosine shape scaled down. Applied via
    # optimizer.step(lr_scale=...), which restores the group lr before the
    # recurrent scheduler advances, so the schedule itself stays exact.
    lr_scale_global = float(os.environ.get("SPECFORGE_LR_SCALE", "1.0"))
    if lr_scale_global != 1.0:
        print_on_rank0(f"SPECFORGE_LR_SCALE={lr_scale_global}: effective LR = "
                       f"{lr_scale_global} x schedule")
    # Every rank MUST use the same scale or the sharded updates diverge — verify.
    _scale_t = torch.tensor([lr_scale_global], device=device, dtype=torch.float64)
    _scale_min, _scale_max = _scale_t.clone(), _scale_t.clone()
    dist.all_reduce(_scale_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(_scale_max, op=dist.ReduceOp.MAX)
    if not torch.equal(_scale_min, _scale_max):
        raise RuntimeError(
            f"SPECFORGE_LR_SCALE differs across ranks "
            f"(min={_scale_min.item()}, max={_scale_max.item()}); set the same "
            f"value on every node."
        )

    rewarm_total = rewarm_left = 0
    if resume_state is not None:
        if os.environ.get("SPECFORGE_RESUME_FULL_OPTIM") == "1":
            # Full resume incl. AdamW moments. Only correct if the checkpoint's
            # flat-param sharding matches this run's exactly; under FSDP the raw
            # per-shard AdamW state saved by BF16Optimizer does NOT reshard onto a
            # fresh wrap -> exp_avg vs grad size mismatch at optimizer.step().
            # (Also: the checkpoint only holds RANK 0's optimizer shard, so this
            # is only usable single-rank.)
            optimizer.load_state_dict(resume_state)
            print_on_rank0("Restored FULL optimizer + scheduler state")
        else:
            # Reshard-safe resume (default): restore the LR scheduler + step exactly,
            # but reset the Adam moments. Avoids the FSDP flat-param reshard
            # mismatch that crashes optimizer.step().
            optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            # With freshly reset moments the first optimizer steps are sign-like
            # kicks of ~lr per coordinate (m_hat/sqrt(v_hat) = +-1 at t=1): fatal
            # for a converged model at full LR (observed: tau 4.06 -> 1.1 within
            # ~5 opt steps of a mid-run resume). Linearly re-warm the LR over the
            # first N optimizer steps so v_hat re-estimates on real gradients
            # while the weights barely move. All ranks compute the identical
            # factor, so sharded updates stay consistent.
            rewarm_total = rewarm_left = int(
                os.environ.get("SPECFORGE_RESUME_LR_REWARM_OPT_STEPS", "64")
            )
            print_on_rank0(
                "Restored LR scheduler + step; Adam moments reset (reshard-safe); "
                f"LR re-warm over next {rewarm_total} optimizer steps"
            )
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state
        print_on_rank0(
            f"Resumed training state: epoch={start_epoch}, step={global_step}, "
            f"lr={optimizer.get_learning_rate():.6f}"
        )

    skip_steps = global_step - start_epoch * len(train_dataloader)

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    # ---- TP-batch scatter (tp-replicated-target topology, e.g. Approach 2) ----
    # Without DP-attention the target runs TP over the node's ranks, so every
    # rank holds the SAME node-batch and identical target hiddens after the
    # cooperative prefill. Training the draft on the full batch on every rank
    # just computes tp_size identical-gradient copies (FSDP averages them back
    # to the same update). Scatter instead: each rank keeps a distinct
    # 1/tp_size slice (trimmed to its own max true length) -> tp_size x less
    # draft compute per sample and per-rank-unique data, at bit-identical
    # optimization semantics (the pooled-global-mean loss all-reduces its
    # denominator over the world either way).
    _use_dp_attention = (
        args.target_model_backend == "sglang" and args.sglang_enable_dp_attention
    )
    _tp_group = get_tp_group()
    _tp_size = dist.get_world_size(_tp_group) if _tp_group is not None else 1
    tp_scatter = (
        os.environ.get("SPECFORGE_TP_BATCH_SCATTER", "1") == "1"
        and not _use_dp_attention
        and _tp_size > 1
        and args.batch_size % _tp_size == 0
    )
    tp_scatter_rank = dist.get_rank(_tp_group) if tp_scatter else 0
    if _tp_size > 1 and not _use_dp_attention:
        if tp_scatter:
            print_on_rank0(
                f"TP-batch scatter ON: node batch {args.batch_size} -> "
                f"{args.batch_size // _tp_size} sample(s)/rank across tp={_tp_size}"
            )
        else:
            print_on_rank0(
                f"TP-batch scatter OFF (env or batch_size {args.batch_size} "
                f"not divisible by tp={_tp_size}); draft compute is replicated "
                f"{_tp_size}x per node"
            )

    last_time = time.time()
    last_global_grad_norm = None
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")
    stop = False

    for epoch in range(start_epoch, args.num_epochs):
        if stop:
            break
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].to(device, non_blocking=True)
            attention_mask = data["attention_mask"].to(device, non_blocking=True)
            loss_mask = data["loss_mask"].to(device, non_blocking=True)
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = target_output.hidden_states.to(device, non_blocking=True)

            last_hidden_states = target_output.last_hidden_states
            if last_hidden_states is not None:
                last_hidden_states = last_hidden_states.to(device, non_blocking=True)
            elif needs_target_hidden:
                raise RuntimeError(
                    "DSpark L1/confidence losses are enabled but the target backend "
                    f"({args.target_model_backend}) did not surface last_hidden_states. "
                    "Use --target-model-backend hf, or run CE-only with "
                    "--l1-loss-alpha 0 --no-confidence-head."
                )

            if tp_scatter:
                # Keep this rank's slice of the node batch and trim its right
                # padding (collator pads right; positions >= true length carry
                # no loss tokens, so anchors never reference them). contiguous()
                # matters: sliced views send cuBLAS down a ~35x slower batched-
                # GEMM path in the fc/lm_head linears.
                _per = input_ids.size(0) // _tp_size
                _sl = slice(tp_scatter_rank * _per, (tp_scatter_rank + 1) * _per)
                _keep = max(int(attention_mask[_sl].sum(dim=1).max().item()), 1)
                input_ids = input_ids[_sl, :_keep].contiguous()
                loss_mask = loss_mask[_sl, :_keep].contiguous()
                hidden_states = hidden_states[_sl, :_keep].contiguous()
                if last_hidden_states is not None:
                    last_hidden_states = last_hidden_states[
                        _sl, :_keep
                    ].contiguous()

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

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                # DeepSpec parity (base_trainer.py): clip by the GLOBAL grad norm
                # before the optimizer step. FSDP.clip_grad_norm_ all-reduces the
                # norm across shards -> one uniform scale on every rank. The
                # BF16Optimizer's internal local-shard clip then no-ops (each
                # post-clip local norm <= global norm <= max_norm), whereas on
                # its own it clips each shard by its LOCAL norm — a ~sqrt(world)x
                # looser, non-uniform threshold under sharding.
                if hasattr(dspark_model, "clip_grad_norm_"):
                    # Returned value = the PRE-clip global grad norm across all
                    # shards — the collapse-forensics signal we were missing:
                    # gradient explosion shows here as >> max_norm; curvature /
                    # step-size instability (Adam's update magnitude ~lr,
                    # independent of grad scale, which clipping cannot bound)
                    # shows a collapse WITHOUT this ever spiking.
                    last_global_grad_norm = float(
                        dspark_model.clip_grad_norm_(args.max_grad_norm)
                    )
                _scale = lr_scale_global
                if rewarm_left > 0:
                    # Post-resume LR re-warm (see resume block), composed with
                    # the persistent scale. Both are applied transiently inside
                    # optimizer.step() and restored before the (recurrent)
                    # scheduler advances.
                    _scale *= (rewarm_total - rewarm_left + 1) / rewarm_total
                    rewarm_left -= 1
                    if rewarm_left == 0:
                        print_on_rank0(
                            f"LR re-warm complete; effective LR = "
                            f"{lr_scale_global} x schedule"
                        )
                optimizer.step(lr_scale=_scale)

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
                if last_global_grad_norm is not None:
                    # Already global (FSDP all-reduces inside clip_grad_norm_);
                    # value from the most recent optimizer step (<= ACC micro
                    # steps stale).
                    comp_log["grad_norm"] = last_global_grad_norm

                record_metrics(
                    args,
                    loss_log.item(),
                    acc_log.item(),
                    comp_log,
                    global_step,
                    tracker,
                    optimizer,
                    train_dataloader,
                    mode="train",
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
                    args, epoch, global_step, dspark_model, draft_model, optimizer
                )

            if args.eval_datasets_dir and global_step % args.eval_interval == 0:
                _maybe_run_accept_length_eval(
                    args, dspark_model, draft_model, target_model,
                    target_components, tokenizer, tracker, global_step,
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                print_on_rank0(f"Reached max_steps={args.max_steps}; stopping.")
                stop = True
                break

    save_checkpoint(
        args, args.num_epochs, global_step, dspark_model, draft_model, optimizer
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
