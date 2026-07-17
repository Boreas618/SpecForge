"""DEPRECATED one-release compatibility wrapper over ``specforge data regen``.

This script keeps the historical flag surface and three-file output contract
(``<output>.jsonl`` + ``<output>_error.jsonl`` + ``<output>_skipped.jsonl``)
while executing through the supported regeneration pipeline
(``specforge.data.regen``). It partitions valid rows by operation — rows whose
conversations contain assistant turns use ``replay_assistants``; prompt-only
rows use ``complete_prompt`` — runs one artifact per operation, and derives
the legacy files from the finalized ``DatasetArtifact``s. New workflows should
author a recipe and call ``specforge data regen run`` directly; this wrapper
will be removed after one release.

Contract notes versus the legacy implementation:

- Output rows are ordered deterministically by input position instead of by
  request completion time, and every run rewrites the three files from the
  artifacts instead of appending. ``--resume`` resumes the pipeline's exact
  per-attempt journal rather than counting previously written lines.
- Rows the model cannot produce validly (leaked think markers, truncated
  completions, missing required reasoning, tool trajectories) are terminal
  rejects and land in the skipped file with a category and diagnostic.
  Unresolved transport errors land in the error file. The legacy script
  classified some of these differently and silently kept truncated rows.
- Rows that mix assistant history with a trailing user turn are skipped:
  they are neither a replay nor a prompt completion. The legacy script
  answered the trailing turn; author a recipe for that workload.
- ``--is-gpt-oss`` maps to a fixed ``reasoning_effort=medium`` request field.
  The legacy per-request random effort is not reproducible under the
  pipeline's deterministic request-seed contract.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
    --model Qwen/Qwen3.5-35B-A3B \
    --tp 1 \
    --host 0.0.0.0 \
    --port 30000 \
    --reasoning-parser qwen3

2. Regenerate the dataset.

python scripts/regenerate_train_data.py \
    --model Qwen/Qwen3.5-35B-A3B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/opc_train_first_turn.jsonl \
    --output-file-path ./cache/dataset/opc_train_regen_first_turn.jsonl \
    --resume \
    --reasoning save
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from scripts.conversation_validation import validate_conversation
except ModuleNotFoundError:
    from conversation_validation import validate_conversation

try:
    import specforge  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEPRECATION_NOTICE = (
    "scripts/regenerate_train_data.py is a deprecated compatibility wrapper "
    "and will be removed after one release. Author a regeneration recipe and "
    "run `specforge data regen run --config recipe.yaml` instead; see "
    "docs/concepts/data-regeneration.md."
)

MIXED_ROW_REASON = (
    "conversation mixes assistant history with a trailing user turn; the "
    "compatibility wrapper supports replay and prompt completion only"
)

# The canonical markers emitted by reasoning-parser-enabled servers. The
# structured_chat codec rejects any assistant content containing them.
THINK_MARKER_CONTROL_TOKENS = ["<think>", "</think>"]

OPERATIONS = {
    "replay": "replay_assistants",
    "complete": "complete_prompt",
}


def validate_regen_input(data: Any) -> Optional[str]:
    """Return why a row cannot be regenerated, or ``None``.

    A row must satisfy both the historical conversation shape check and the
    pipeline's record normalization, so that planning can never fail on a row
    this precheck admitted.
    """

    if not isinstance(data, dict):
        return "Expected a JSON object"
    legacy_reason = validate_conversation(
        data.get("conversations"),
        error_style="regeneration",
    )
    if legacy_reason is not None:
        return legacy_reason
    from specforge.data.regen.errors import ContractError
    from specforge.data.regen.records.messages import normalize_sharegpt_record

    try:
        normalize_sharegpt_record(data, source_name="legacy", position=0)
    except ContractError as exc:
        return str(exc)
    return None


def classify_operation(data: Dict[str, Any]) -> Optional[str]:
    """Return the operation bucket for a valid row, or ``None`` if neither."""

    conversations = data.get("conversations", [])
    has_assistant = any(
        message.get("role") == "assistant" for message in conversations
    )
    if not has_assistant:
        return "complete"
    if conversations and conversations[-1].get("role") == "assistant":
        return "replay"
    return None


def set_skipped(data: Any, error: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"status": "skipped", "error": error, "data": data}
    data["status"] = "skipped"
    data["error"] = error
    return data


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse the historical command line surface."""

    parser = argparse.ArgumentParser(
        description=(
            "DEPRECATED wrapper: re-generate training data through the "
            "specforge.data.regen pipeline"
        )
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--model-revision",
        type=str,
        default="unpinned",
        help=(
            "Immutable model revision recorded in the artifact recipe. "
            "Pin this for reproducible artifacts."
        ),
    )
    model_group.add_argument(
        "--reasoning",
        choices=["none", "save", "disable"],
        default="none",
        help=(
            "Reasoning mode: 'none' for standard models, 'save' to store "
            "reasoning_content, or 'disable' to disable thinking via extra_body"
        ),
    )
    model_group.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )

    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
    )
    sampling_params_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    sampling_params_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value",
    )
    sampling_params_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    sampling_params_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help=(
            "The number of concurrent requests per server; the total number "
            "of concurrent shard workers is concurrency * number of servers"
        ),
    )
    optimization_group.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-file-path", type=str, required=True, help="Path to the input file"
    )
    data_group.add_argument(
        "--output-file-path", type=str, required=True, help="Path to the output file"
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Regenerate only the first N valid samples",
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume the existing artifacts' exact per-attempt journals",
    )
    data_group.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help=(
            "Artifact workspace directory "
            "(default: <output-file-path minus .jsonl>.regen-artifact)"
        ),
    )

    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        required=True,
        help="Server address and port for sglang model server",
    )
    return parser.parse_args(argv)


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    """Rough context length estimate in whitespace-separated tokens."""

    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def _legacy_paths(output_file_path: str) -> Tuple[str, str]:
    error_path = output_file_path.replace(".jsonl", "_error.jsonl")
    skipped_path = output_file_path.replace(".jsonl", "_skipped.jsonl")
    return error_path, skipped_path


def _endpoint_url(server_address: str) -> str:
    if server_address.startswith(("http://", "https://")):
        return server_address.rstrip("/")
    return f"http://{server_address}"


def _probe_endpoint(endpoint: str, model: str, timeout: float) -> bool:
    from specforge.data.regen.backends import openai_chat

    url = endpoint + (
        "/chat/completions"
        if endpoint.endswith("/v1")
        else "/v1/chat/completions"
    )
    try:
        body = openai_chat.post_json(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 1,
                "stream": False,
            },
            api_key=None,
            timeout=timeout,
        )
    except Exception:
        return False
    choices = body.get("choices")
    return isinstance(choices, list) and bool(choices)


def _probe_endpoints(addresses: List[str], model: str, timeout: float) -> List[str]:
    valid = []
    for address in addresses:
        endpoint = _endpoint_url(address)
        if _probe_endpoint(endpoint, model, timeout):
            valid.append(endpoint)
        else:
            print(f"Server {address} is not available")
    if not valid:
        raise ValueError("No server address is available")
    return valid


@dataclass
class OperationBucket:
    """One operation's filtered pipeline input and its artifact workspace."""

    name: str
    filtered_path: Path
    artifact_dir: Path
    mapping: List[int] = field(default_factory=list)
    layout: Any = None


def _prepare_filtered_inputs(
    input_file_path: str,
    buckets: Dict[str, OperationBucket],
    skipped_path: str,
    num_samples: Optional[int],
) -> int:
    """Partition valid rows into operation buckets; skip the rest.

    Each bucket's ``mapping`` records the original input position of every
    filtered row. Scanning stops after ``num_samples`` valid rows, matching
    the legacy script, so later rows are neither regenerated nor accounted.
    """

    from specforge.data.regen.contracts import canonical_json

    seen_ids: set = set()
    handles = {}
    selected = 0
    try:
        for bucket in buckets.values():
            handles[bucket.name] = bucket.filtered_path.open(
                "w", encoding="utf-8"
            )
        with (
            open(input_file_path, encoding="utf-8") as input_file,
            open(skipped_path, "w", encoding="utf-8") as skipped_handle,
        ):
            for position, line in enumerate(input_file):
                if num_samples is not None and selected >= num_samples:
                    break
                data = json.loads(line.strip())
                invalid_reason = validate_regen_input(data)
                if invalid_reason is None and classify_operation(data) is None:
                    invalid_reason = MIXED_ROW_REASON
                if invalid_reason is not None:
                    skipped_handle.write(
                        json.dumps(
                            set_skipped(data, invalid_reason), ensure_ascii=False
                        )
                        + "\n"
                    )
                    continue
                bucket = buckets[classify_operation(data)]
                filtered_row = dict(data)
                identity = canonical_json(filtered_row.get("id", position))
                if identity in seen_ids:
                    # The pipeline requires unique type-preserving ids. The
                    # output derivation restores the original row, so this
                    # synthetic id never reaches the legacy output files.
                    filtered_row["id"] = f"{identity}#regen-dup-{position}"
                else:
                    seen_ids.add(identity)
                handles[bucket.name].write(
                    json.dumps(filtered_row, ensure_ascii=False) + "\n"
                )
                bucket.mapping.append(position)
                selected += 1
    finally:
        for handle in handles.values():
            handle.close()
    return selected


def _build_recipe(
    args: argparse.Namespace, bucket: OperationBucket, shards: int
):
    from specforge.data.regen.recipe import RegenerationRecipe

    reasoning_mode = {"none": "preserve", "save": "required", "disable": "disabled"}
    sampling: Dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning": reasoning_mode[args.reasoning],
        "extra": {},
    }
    if args.top_p is not None:
        sampling["top_p"] = args.top_p
    if args.top_k is not None:
        sampling["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        sampling["extra"]["presence_penalty"] = args.repetition_penalty
    if args.reasoning == "save":
        sampling["extra"]["chat_template_kwargs"] = {"enable_thinking": True}
    elif args.reasoning == "disable":
        sampling["extra"]["chat_template_kwargs"] = {"enable_thinking": False}

    generator_config: Dict[str, Any] = {}
    if args.reasoning in {"save", "disable"}:
        generator_config["control_tokens"] = list(THINK_MARKER_CONTROL_TOKENS)
    if args.reasoning == "save":
        # Serving templates for reasoning models drop earlier-turn thinking;
        # requests must match that, while the artifact keeps full reasoning.
        generator_config["history_reasoning"] = "strip"
    if args.is_gpt_oss:
        generator_config["request_extra"] = {"reasoning_effort": "medium"}

    return RegenerationRecipe.model_validate(
        {
            "version": 1,
            "seed": 0,
            "sources": {
                "legacy": {
                    "adapter": "jsonl",
                    "record_adapter": "sharegpt",
                    "config": {"path": str(bucket.filtered_path)},
                    "selection": {"mode": "all"},
                }
            },
            "generators": {
                "teacher": {
                    "backend": "openai_chat",
                    "model": args.model,
                    "revision": args.model_revision,
                    "codec": "structured_chat",
                    "sampling": sampling,
                    "config": generator_config,
                }
            },
            "workflow": [
                {
                    "id": "regenerate",
                    "operation": OPERATIONS[bucket.name],
                    "generator": "teacher",
                    "tool_policy": "reject",
                }
            ],
            "validation": {
                "profiles": ["baseline"],
                "max_unresolved_error_rate": 1.0,
                "max_policy_reject_rate": 1.0,
            },
            "output": {
                "uri": str(bucket.artifact_dir),
                "format": "jsonl",
                "shards": shards,
            },
        }
    )


def _prepare_artifact_dir(artifact_dir: Path, resume: bool) -> None:
    if not artifact_dir.exists():
        return
    if resume:
        return
    known_content = {"replay", "complete"}
    unexpected = [
        entry.name
        for entry in artifact_dir.iterdir()
        if entry.name not in known_content
    ]
    if unexpected:
        raise ValueError(
            f"refusing to overwrite {artifact_dir}: it does not look like a "
            "wrapper artifact workspace; pass --artifact-dir or remove it "
            "explicitly"
        )
    shutil.rmtree(artifact_dir)


def _check_resume_recipe(layout, recipe) -> None:
    """Fail fast when --resume is combined with changed generation flags."""

    from specforge.data.regen.artifact import read_json

    existing = read_json(layout.recipe)
    current = recipe.canonical_payload(for_identity=True)
    for payload in (existing, current):
        for source in payload.get("sources", {}).values():
            source["config"] = None
    if existing != current:
        raise ValueError(
            "--resume flags differ from the artifact's pinned recipe; rerun "
            "with the original flags or start a fresh run without --resume"
        )


def _run_pipeline(
    args: argparse.Namespace,
    bucket: OperationBucket,
    endpoints: List[str],
) -> None:
    from specforge.data.regen.artifact import ArtifactLayout, read_json
    from specforge.data.regen.executor import run_worker
    from specforge.data.regen.finalize import (
        publish_artifact,
        resolve_attempts,
        validate_artifact,
    )
    from specforge.data.regen.planner import plan_recipe

    from specforge.data.regen.artifact import read_state

    shards = max(
        1, min(len(bucket.mapping), len(endpoints) * args.concurrency)
    )
    recipe = _build_recipe(args, bucket, shards)
    layout = ArtifactLayout.from_uri(str(bucket.artifact_dir))
    if layout.plan.exists():
        _check_resume_recipe(layout, recipe)
        if read_state(layout) == "FINALIZED":
            manifest = publish_artifact(layout)
            print(
                f"Artifact already finalized ({bucket.name}): "
                f"digest {manifest.get('artifact_digest', '')}"
            )
            bucket.layout = layout
            return
        print(f"Resuming existing artifact plan at {bucket.artifact_dir}")
    else:
        layout = plan_recipe(recipe)

    runtime = {
        "teacher": {"endpoints": endpoints, "timeout": args.request_timeout}
    }
    # Shard ownership is pinned by the immutable plan; the flags only choose
    # how many workers run at once.
    shard_indexes = [
        int(shard["index"]) for shard in read_json(layout.plan)["shards"]
    ]
    workers = max(
        1, min(len(shard_indexes), len(endpoints) * args.concurrency)
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_worker, layout, shard_index, runtime=runtime)
            for shard_index in shard_indexes
        ]
        for future in futures:
            future.result()

    resolve_attempts(layout)
    validation = validate_artifact(layout)
    if not validation.get("passed", False):
        raise ValueError(
            "artifact validation failed; inspect "
            f"{layout.reports / 'validation.json'}"
        )
    manifest = publish_artifact(layout)
    print(
        f"Artifact finalized ({bucket.name}): "
        f"digest {manifest.get('artifact_digest', '')}"
    )
    bucket.layout = layout


def _iter_bucket_results(
    bucket: OperationBucket,
) -> Iterator[Tuple[int, str, Any]]:
    """Yield ``(original_position, kind, value)`` in input order."""

    from specforge.data.regen.contracts import RecordEnvelope, iter_jsonl

    def successes():
        for path in sorted(bucket.layout.data.glob("part-*.jsonl")):
            for _, value in iter_jsonl(path):
                envelope = RecordEnvelope.from_dict(value)
                yield envelope.input_position, list(
                    envelope.payload["conversations"]
                )

    def rejects():
        for path in sorted(bucket.layout.rejects.glob("part-*.jsonl")):
            for _, value in iter_jsonl(path):
                yield value

    success_iter = successes()
    reject_iter = rejects()
    next_success = next(success_iter, None)
    next_reject = next(reject_iter, None)
    for filtered_position, original_position in enumerate(bucket.mapping):
        if next_success is not None and next_success[0] == filtered_position:
            yield original_position, "success", next_success[1]
            next_success = next(success_iter, None)
        elif next_reject is not None and (
            next_reject["task_ordinal"] == filtered_position
        ):
            yield original_position, "reject", next_reject
            next_reject = next(reject_iter, None)
        else:
            # Finalization guarantees coverage, so this is defensive.
            yield original_position, "missing", None


def _derive_legacy_outputs(
    buckets: List[OperationBucket],
    input_file_path: str,
    output_file_path: str,
    error_path: str,
    skipped_path: str,
) -> Dict[str, Any]:
    """Merge artifact results back into the legacy three-file layout.

    The skipped file already holds precheck skips; terminal and policy
    rejects are appended to it, unresolved errors go to the error file, and
    successes rewrite the original row with the regenerated conversation.
    """

    import heapq

    stats = {
        "success": 0,
        "error": 0,
        "skipped": 0,
        "context_sum": 0,
        "context_min": None,
        "context_max": 0,
    }
    merged = heapq.merge(
        *(_iter_bucket_results(bucket) for bucket in buckets if bucket.layout),
        key=lambda item: item[0],
    )
    next_result = next(merged, None)
    with (
        open(input_file_path, encoding="utf-8") as input_file,
        open(output_file_path, "w", encoding="utf-8") as output_handle,
        open(error_path, "w", encoding="utf-8") as error_handle,
        open(skipped_path, "a", encoding="utf-8") as skipped_handle,
    ):
        for original_position, line in enumerate(input_file):
            if next_result is None:
                break
            if next_result[0] != original_position:
                continue
            _, kind, value = next_result
            next_result = next(merged, None)
            row = json.loads(line.strip())
            if kind == "success":
                row["conversations"] = value
                row["status"] = "success"
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["success"] += 1
                context = compute_context_length(row["conversations"])
                stats["context_sum"] += context
                stats["context_min"] = (
                    context
                    if stats["context_min"] is None
                    else min(stats["context_min"], context)
                )
                stats["context_max"] = max(stats["context_max"], context)
            elif kind == "reject":
                diagnostic = f"{value.get('category')}: {value.get('diagnostic')}"
                if value.get("status") == "unresolved_error":
                    row["status"] = "error"
                    row["error"] = diagnostic
                    error_handle.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    stats["error"] += 1
                else:
                    skipped_handle.write(
                        json.dumps(
                            set_skipped(row, diagnostic), ensure_ascii=False
                        )
                        + "\n"
                    )
                    stats["skipped"] += 1
            else:
                row["status"] = "error"
                row["error"] = (
                    "internal_error: row missing from artifact accounting"
                )
                error_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["error"] += 1
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_arguments(argv)
    print(DEPRECATION_NOTICE, file=sys.stderr)

    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")
    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")
    if args.concurrency <= 0:
        raise ValueError("Concurrency must be greater than 0")
    if not args.output_file_path.endswith(".jsonl"):
        raise ValueError("--output-file-path must end in .jsonl")
    if args.model_revision == "unpinned":
        print(
            "warning: --model-revision is unpinned; pin it for reproducible "
            "artifact provenance",
            file=sys.stderr,
        )

    print("Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {args.output_file_path}")
    print(f"  Resume mode: {args.resume}")
    print("-" * 50)

    base = args.output_file_path[: -len(".jsonl")]
    artifact_dir = Path(args.artifact_dir or f"{base}.regen-artifact")
    error_path, skipped_path = _legacy_paths(args.output_file_path)
    buckets = {
        name: OperationBucket(
            name=name,
            filtered_path=Path(f"{base}.regen-input-{name}.jsonl"),
            artifact_dir=artifact_dir / name,
        )
        for name in OPERATIONS
    }

    _prepare_artifact_dir(artifact_dir, args.resume)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    endpoints = _probe_endpoints(
        args.server_address, args.model, args.request_timeout
    )
    print(f"Using {len(endpoints)} server addresses: {endpoints}")
    print("-" * 50)

    selected = _prepare_filtered_inputs(
        args.input_file_path, buckets, skipped_path, args.num_samples
    )
    if selected == 0:
        open(args.output_file_path, "w", encoding="utf-8").close()
        open(error_path, "w", encoding="utf-8").close()
        precheck_skips = count_lines(skipped_path)
        print("No valid samples to regenerate.")
        print(
            f"\nProcessing completed! 0 samples regenerated, 0 samples "
            f"failed, {precheck_skips} samples skipped."
        )
        return

    active = [bucket for bucket in buckets.values() if bucket.mapping]
    for bucket in active:
        print(
            f"Regenerating {len(bucket.mapping)} samples via "
            f"{OPERATIONS[bucket.name]}; artifact: {bucket.artifact_dir}"
        )
        _run_pipeline(args, bucket, endpoints)
    print("-" * 50)

    stats = _derive_legacy_outputs(
        active,
        args.input_file_path,
        args.output_file_path,
        error_path,
        skipped_path,
    )
    precheck_skips = count_lines(skipped_path) - stats["skipped"]

    print("\nProcessing completed!")
    if stats["success"] > 0:
        average = stats["context_sum"] / stats["success"]
        print("Context length statistics (token count over conversations):")
        print(f"Number of successful examples: {stats['success']}")
        print(f"Shortest context length: {stats['context_min']}")
        print(f"Longest context length: {stats['context_max']}")
        print(f"Average context length: {average:.2f}")
    else:
        print("No successful examples to compute context length statistics.")

    total_skipped = stats["skipped"] + precheck_skips
    print(
        f"\nProcessing completed! {stats['success']} samples regenerated, "
        f"{stats['error']} samples failed, {total_skipped} samples skipped."
    )


if __name__ == "__main__":
    main()
