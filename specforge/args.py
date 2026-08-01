import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES


@dataclass
class TrackerArgs:
    report_to: str = "none"
    wandb_project: str = None
    wandb_name: str = None
    wandb_key: str = None
    wandb_offline: bool = False
    wandb_dir: str = None
    swanlab_project: str = None
    swanlab_name: str = None
    swanlab_key: str = None
    mlflow_experiment_id: str = None
    mlflow_run_name: str = None
    mlflow_run_id: str = None
    mlflow_tracking_uri: str = None
    mlflow_registry_uri: str = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--report-to",
            type=str,
            default="none",
            choices=["wandb", "tensorboard", "swanlab", "mlflow", "none"],
            help="The integration to report results and logs to.",
        )
        # wandb-specific args
        parser.add_argument("--wandb-project", type=str, default=None)
        parser.add_argument("--wandb-name", type=str, default=None)
        parser.add_argument("--wandb-key", type=str, default=None, help="W&B API key.")
        parser.add_argument(
            "--wandb-offline",
            action="store_true",
            help="Enable W&B offline mode and store logs locally.",
        )
        parser.add_argument(
            "--wandb-dir",
            type=str,
            default=None,
            help="Directory to store W&B files. Defaults to './wandb' under the project root when using W&B.",
        )
        # swanlab-specific args
        parser.add_argument(
            "--swanlab-project",
            type=str,
            default=None,
            help="The project name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-name",
            type=str,
            default=None,
            help="The experiment name for swanlab.",
        )
        parser.add_argument(
            "--swanlab-key",
            type=str,
            default=None,
            help="The API key for swanlab non-interactive login.",
        )
        # mlflow-specific args
        parser.add_argument(
            "--mlflow-tracking-uri",
            type=str,
            default=None,
            help="The MLflow tracking URI. If not set, uses MLFLOW_TRACKING_URI environment variable or defaults to local './mlruns'.",
        )
        parser.add_argument(
            "--mlflow-experiment-name",
            type=str,
            default=None,
            help="The MLflow experiment name. If not set, uses MLFLOW_EXPERIMENT_NAME environment variable.",
        )
        parser.add_argument(
            "--mlflow-run-name",
            type=str,
            default=None,
            help="The MLflow run name. If not set, MLflow will auto-generate one.",
        )


@dataclass
class SGLangBackendArgs:
    sglang_attention_backend: str = "fa3"
    sglang_mem_fraction_static: float = 0.4
    sglang_context_length: int = None
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = True
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_enable_piecewise_cuda_graph: bool = False
    sglang_piecewise_cuda_graph_max_tokens: int = 4096
    sglang_piecewise_cuda_graph_tokens: List[int] = None
    sglang_ep_size: int = 1
    sglang_dp_size: int = 1
    sglang_moe_a2a_backend: str = "none"
    sglang_moe_runner_backend: str = "auto"
    sglang_max_running_requests: int = None  # assign based on batch size
    sglang_max_total_tokens: int = None  # assign based on batch size and seq length
    # Optional engine knobs (passed through only when set — needed by targets
    # with hybrid/mamba state and quantized checkpoints, e.g. NDA/Inkling).
    sglang_page_size: int = None
    sglang_quantization: str = None
    sglang_fp4_gemm_runner_backend: str = None
    sglang_mamba_radix_cache_strategy: str = None
    sglang_max_mamba_cache_size: int = None
    sglang_swa_full_tokens_ratio: float = None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # sglang arguments
        parser.add_argument(
            "--sglang-attention-backend",
            type=str,
            default="flashinfer",
            choices=ATTENTION_BACKEND_CHOICES,
            help="The attention backend of SGLang backend",
        )
        parser.add_argument(
            "--sglang-mem-fraction-static",
            type=float,
            default=0.4,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--sglang-context-length",
            type=int,
            default=None,
            help="The context length of the SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-symm-mem",
            action="store_true",
            help="Enable NCCL symmetric memory for fast collectives for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-dp-attention",
            action="store_true",
            help="Enable DP attention for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-dp-lm-head",
            action="store_true",
            help="Enable piecewise CUDA graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-piecewise-cuda-graph",
            action="store_true",
            help="Enable piecewise CUDA graph for SGLang backend's prefill",
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-max-tokens",
            type=int,
            default=4096,
            help="Set the max tokens for piecewise CUDA graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            default=None,
            help="Set the list of tokens when using piecewise cuda graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-ep-size",
            type=int,
            default=1,
            help="The ep size of the SGLang backend",
        )
        parser.add_argument(
            "--sglang-dp-size",
            type=int,
            default=1,
            help="The data-parallel size of the SGLang backend. With "
            "--sglang-enable-dp-attention set this to the world/tp size to make "
            "attention data-parallel (attn_tp = tp_size // dp_size).",
        )
        parser.add_argument(
            "--sglang-moe-a2a-backend",
            type=str,
            default="none",
            help="SGLang MoE all-to-all backend (e.g. 'deepep' for DP-attention + "
            "expert-parallel MoE, or 'none'). Validated by sglang ServerArgs.",
        )
        parser.add_argument(
            "--sglang-moe-runner-backend",
            type=str,
            default="auto",
            help="SGLang MoE runner backend (default 'auto' lets sglang resolve it, "
            "e.g. flashinfer_trtllm for fp4/fp8 DeepSeek-V4). Validated by ServerArgs.",
        )
        parser.add_argument(
            "--sglang-page-size",
            type=int,
            default=None,
            help="KV page size for the SGLang backend (e.g. 128 for NDA/Inkling "
            "serve parity). Default: sglang's own default.",
        )
        parser.add_argument(
            "--sglang-quantization",
            type=str,
            default=None,
            help="Checkpoint quantization method (e.g. modelopt_fp4). Usually "
            "auto-detected from hf_quant_config.json; pass to pin explicitly.",
        )
        parser.add_argument(
            "--sglang-fp4-gemm-runner-backend",
            type=str,
            default=None,
            help="FP4 GEMM runner backend (ServerArgs fp4_gemm_runner_backend; "
            "serve CLI alias --fp4-gemm-backend), e.g. flashinfer_trtllm.",
        )
        parser.add_argument(
            "--sglang-mamba-radix-cache-strategy",
            type=str,
            default=None,
            help="Mamba/SConv radix-cache strategy (e.g. extra_buffer — required "
            "by NDA/Inkling's model-constructor asserts).",
        )
        parser.add_argument(
            "--sglang-max-mamba-cache-size",
            type=int,
            default=None,
            help="Max mamba/SConv state slots. Training prefill uses <= batch "
            "requests at a time, so a small value (e.g. 64) saves pool memory.",
        )
        parser.add_argument(
            "--sglang-swa-full-tokens-ratio",
            type=float,
            default=None,
            help="Hybrid SWA memory split (full-attention share), e.g. 0.2 for "
            "NDA/Inkling serve parity.",
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "SGLangBackendArgs":
        return SGLangBackendArgs(
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_context_length=args.sglang_context_length,
            sglang_enable_nccl_nvls=args.sglang_enable_nccl_nvls,
            sglang_enable_symm_mem=args.sglang_enable_symm_mem,
            sglang_enable_torch_compile=args.sglang_enable_torch_compile,
            sglang_enable_dp_attention=args.sglang_enable_dp_attention,
            sglang_enable_dp_lm_head=args.sglang_enable_dp_lm_head,
            sglang_enable_piecewise_cuda_graph=args.sglang_enable_piecewise_cuda_graph,
            sglang_piecewise_cuda_graph_max_tokens=args.sglang_piecewise_cuda_graph_max_tokens,
            sglang_piecewise_cuda_graph_tokens=args.sglang_piecewise_cuda_graph_tokens,
            sglang_ep_size=args.sglang_ep_size,
            sglang_dp_size=args.sglang_dp_size,
            sglang_moe_a2a_backend=args.sglang_moe_a2a_backend,
            sglang_moe_runner_backend=args.sglang_moe_runner_backend,
            sglang_page_size=args.sglang_page_size,
            sglang_quantization=args.sglang_quantization,
            sglang_fp4_gemm_runner_backend=args.sglang_fp4_gemm_runner_backend,
            sglang_mamba_radix_cache_strategy=args.sglang_mamba_radix_cache_strategy,
            sglang_max_mamba_cache_size=args.sglang_max_mamba_cache_size,
            sglang_swa_full_tokens_ratio=args.sglang_swa_full_tokens_ratio,
            sglang_max_running_requests=(
                args.target_batch_size if hasattr(args, "target_batch_size") else None
            ),
            sglang_max_total_tokens=(
                args.target_batch_size * args.max_length
                if hasattr(args, "target_batch_size") and hasattr(args, "max_length")
                else None
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(
            attention_backend=self.sglang_attention_backend,
            mem_fraction_static=self.sglang_mem_fraction_static,
            context_length=self.sglang_context_length,
            enable_nccl_nvls=self.sglang_enable_nccl_nvls,
            enable_symm_mem=self.sglang_enable_symm_mem,
            enable_torch_compile=self.sglang_enable_torch_compile,
            enable_dp_attention=self.sglang_enable_dp_attention,
            enable_dp_lm_head=self.sglang_enable_dp_lm_head,
            ep_size=self.sglang_ep_size,
            dp_size=self.sglang_dp_size,
            moe_a2a_backend=self.sglang_moe_a2a_backend,
            moe_runner_backend=self.sglang_moe_runner_backend,
            max_running_requests=self.sglang_max_running_requests,
            max_total_tokens=self.sglang_max_total_tokens,
        )
        # Optional knobs ride along only when explicitly set, so sglang's own
        # defaults stay in force otherwise (None would override e.g. page_size).
        optional = dict(
            page_size=self.sglang_page_size,
            quantization=self.sglang_quantization,
            fp4_gemm_runner_backend=self.sglang_fp4_gemm_runner_backend,
            mamba_radix_cache_strategy=self.sglang_mamba_radix_cache_strategy,
            max_mamba_cache_size=self.sglang_max_mamba_cache_size,
            swa_full_tokens_ratio=self.sglang_swa_full_tokens_ratio,
        )
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        # Piecewise-CUDA-graph fields were renamed/removed in newer sglang; send
        # them only when the feature is actually requested, so a fork without
        # them keeps working (and an explicit request fails loud downstream
        # instead of silently dropping).
        if self.sglang_enable_piecewise_cuda_graph:
            kwargs.update(
                enable_piecewise_cuda_graph=True,
                piecewise_cuda_graph_max_tokens=self.sglang_piecewise_cuda_graph_max_tokens,
                piecewise_cuda_graph_tokens=self.sglang_piecewise_cuda_graph_tokens,
            )
        return kwargs
