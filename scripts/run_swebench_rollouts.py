#!/usr/bin/env python3
# coding=utf-8
"""Generate agentic SWE-bench trajectories for DFlash training.

Phase A of the staged agentic-DFlash pipeline. This driver:

  1. Discovers SWE-bench Harbor task directories (produced by the harbor
     ``swebench`` adapter under ``harbor/datasets/swebench-verified``).
  2. Runs a Harbor ``Job`` with the chosen agent (``--agent``) pointed at an
     OpenAI-compatible SGLang server that serves the *target* model
     (Kimi-K2.7-Code), and captures per-turn supervision in one of two ways:

       * ``terminus-2`` (token-exact): makes its LLM calls through Harbor's own
         LiteLLM layer with ``collect_rollout_details=True``, so the server's
         exact ``prompt_token_ids`` / ``completion_token_ids`` are recorded.
       * ``codex`` (re-tokenized): the Codex CLI talks to the target over
         ``OPENAI_BASE_URL`` and only emits an ATIF ``trajectory.json`` of text
         messages + tool calls (no token ids). We therefore RE-TOKENIZE that
         trajectory through the target tokenizer's chat template, masking the
         assistant-generated spans. This approximates the true inference token
         stream but yields a self-consistent supervised signal.
  3. Reconstructs, per trajectory, contiguous ``(input_ids, loss_mask)`` pairs
     where ``loss_mask == 1`` exactly on the target model's generated tokens
     (the tokens DFlash must learn to draft) and ``0`` on system/user/tool
     context.
  4. Windows long trajectories to ``--max-length`` and writes pretokenized
     train / eval JSONL consumable by ``train_dflash.py --data-format
     pretokenized``.

The pure-Python reconstruction/windowing helpers at the top of this module do
NOT import harbor or torch, so they can be unit-tested anywhere (see
``--self-test``). Harbor is imported lazily inside :func:`run_rollouts`; the
target tokenizer (transformers) is imported lazily only for the codex path.

Run this in harbor's environment (e.g. ``uv run``), since it imports harbor.

Codex wire-API caveat
---------------------
The Codex CLI defaults to OpenAI's *Responses* API. If your SGLang server only
implements ``/v1/chat/completions``, configure Codex to use the chat wire API
(a custom ``model_provider`` with ``wire_api = "chat"`` in Codex's
``config.toml``); otherwise the rollouts will fail to reach the model.

Example
-------
    # Codex agent driving the local target, re-tokenized for DFlash:
    python scripts/run_swebench_rollouts.py \
        --agent codex \
        --api-base http://127.0.0.1:30000/v1 \
        --model-name openai/kimi-k2.7-code \
        --target-model moonshotai/Kimi-K2.7-Code \
        --tasks-dir ../harbor/datasets/swebench-verified \
        --limit 50 --n-concurrent 8 \
        --max-length 16384 \
        --output-dir cache/dataset/swebench-agentic
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

# --------------------------------------------------------------------------- #
# Pure-Python trajectory reconstruction (no harbor / torch imports)
# --------------------------------------------------------------------------- #

# When a re-rendered prompt diverges from the running token stream by at most
# this many tokens, treat it as a benign tokenizer seam (a BPE merge straddling
# the assistant/end-of-turn boundary) and resync in place. Larger divergences
# (e.g. agent-side context summarization/reset) finalize the current segment
# and start a fresh one.
DEFAULT_SEAM_TOLERANCE = 8


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two integer sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def reconstruct_trajectory(
    prompt_token_ids: Sequence[Sequence[int]],
    completion_token_ids: Sequence[Sequence[int]],
    seam_tolerance: int = DEFAULT_SEAM_TOLERANCE,
) -> List[Tuple[List[int], List[int]]]:
    """Reconstruct contiguous ``(input_ids, loss_mask)`` segments.

    Harbor's ``Chat`` records, per LLM turn ``i``:
      * ``prompt_token_ids[i]``  -- the FULL prompt (system+user+prior turns+
        tool results, with chat-template tokens) the server tokenized for turn
        ``i``; and
      * ``completion_token_ids[i]`` -- the tokens the model generated at turn
        ``i`` (stop tokens excluded by the server).

    For a linear chat history ``prompt_token_ids[i+1]`` extends
    ``prompt_token_ids[i] + completion_token_ids[i]`` by the inserted
    end-of-turn / tool / next-assistant-header tokens. We rebuild the running
    token stream ``full`` and mark generated tokens with ``loss_mask = 1``:

      full   = P0 | C0 | (P1 - prefix) | C1 | (P2 - prefix) | C2 | ...
      mask   =  0 |  1 |       0       |  1 |       0       |  1 | ...

    Returns a list of segments. Normally one segment; a new segment is started
    whenever the next prompt diverges from ``full`` by more than
    ``seam_tolerance`` tokens (context summarization / reset), which keeps every
    emitted segment internally consistent (its hidden states can be recomputed
    by a single prefill).
    """
    segments: List[Tuple[List[int], List[int]]] = []
    full: List[int] = []
    mask: List[int] = []

    n_turns = min(len(prompt_token_ids), len(completion_token_ids))
    for i in range(n_turns):
        prompt = list(prompt_token_ids[i])
        completion = list(completion_token_ids[i])

        prefix = common_prefix_len(full, prompt)
        discarded = len(full) - prefix
        if discarded > seam_tolerance and full:
            # Hard divergence (likely summarization). Finalize the current
            # segment and restart from this turn's fresh prompt.
            segments.append((full, mask))
            full, mask = [], []
            prefix = 0
        elif discarded > 0:
            # Benign seam: trust the freshly rendered prompt going forward.
            full = full[:prefix]
            mask = mask[:prefix]

        delta = prompt[prefix:]
        full.extend(delta)
        mask.extend([0] * len(delta))

        full.extend(completion)
        mask.extend([1] * len(completion))

    if full:
        segments.append((full, mask))
    return segments


def window_sequence(
    input_ids: Sequence[int],
    loss_mask: Sequence[int],
    max_length: Optional[int],
    stride: Optional[int],
    min_loss_tokens: int,
) -> List[Tuple[List[int], List[int]]]:
    """Slice a (possibly long) sequence into <= ``max_length`` windows.

    Windows with fewer than ``min_loss_tokens`` supervised tokens are dropped
    (they would be filtered by the trainer anyway). With ``stride <
    max_length`` windows overlap so supervised tokens near a window start still
    appear with longer preceding context in an adjacent window.
    """
    input_ids = list(input_ids)
    loss_mask = list(loss_mask)
    n = len(input_ids)

    if max_length is None or n <= max_length:
        if sum(loss_mask) >= min_loss_tokens:
            return [(input_ids, loss_mask)]
        return []

    if stride is None or stride <= 0:
        stride = max(1, max_length // 2)

    windows: List[Tuple[List[int], List[int]]] = []
    start = 0
    while start < n:
        end = min(start + max_length, n)
        w_ids = input_ids[start:end]
        w_mask = loss_mask[start:end]
        if sum(w_mask) >= min_loss_tokens:
            windows.append((w_ids, w_mask))
        if end == n:
            break
        start += stride
    return windows


# --------------------------------------------------------------------------- #
# Codex (ATIF trajectory) -> pretokenized rows via target-tokenizer re-render
# --------------------------------------------------------------------------- #
#
# Unlike Terminus-2 (which streams exact prompt/completion token ids from the
# SGLang server via collect_rollout_details), the Codex CLI agent only emits an
# ATIF ``trajectory.json`` of text messages + tool calls. To produce DFlash
# (input_ids, loss_mask) pairs we therefore RE-TOKENIZE the trajectory through
# the *target* model's chat template, masking exactly the assistant-generated
# spans. This is an approximation of the true inference token stream (the chat
# template / tool-schema framing may differ from what Codex actually sent over
# the wire), but it yields a self-consistent supervised signal on the target's
# own tokenization.


def extract_text_content(message: object) -> str:
    """Flatten an ATIF ``message``/``content`` field to plain text.

    ATIF allows these fields to be either a string or a list of ContentPart
    dicts (multimodal, ATIF-v1.6+). We keep only the text parts.
    """
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: List[str] = []
        for part in message:
            if isinstance(part, dict):
                text = part.get("text")
                if part.get("type") in (None, "text") and isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(message)


def reconstruct_codex_messages(
    trajectory: Dict,
) -> Tuple[List[Dict], List[int]]:
    """Convert an ATIF trajectory dict into OpenAI-style chat messages.

    Returns ``(messages, assistant_indices)`` where ``assistant_indices`` lists
    the positions in ``messages`` generated by the model (whose tokens must
    carry ``loss_mask = 1``). Tool results are emitted as separate
    ``role="tool"`` context messages right after their assistant turn.

    Steps flagged ``is_copied_context`` (ATIF-v1.5+, re-injected after a context
    reset/summarization) are skipped so we never double-train on copied turns.
    Reasoning summaries are intentionally NOT folded into content: Codex stores
    only lossy summaries, not the raw CoT the target actually emitted.
    """
    messages: List[Dict] = []
    assistant_indices: List[int] = []

    for step in trajectory.get("steps", []):
        if step.get("is_copied_context"):
            continue
        source = step.get("source")
        text = extract_text_content(step.get("message", ""))

        if source == "system":
            messages.append({"role": "system", "content": text})
        elif source == "user":
            messages.append({"role": "user", "content": text})
        elif source == "agent":
            assistant_msg: Dict = {"role": "assistant", "content": text}
            tool_calls = step.get("tool_calls") or []
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.get("tool_call_id") or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.get("function_name", ""),
                            "arguments": json.dumps(
                                tc.get("arguments", {}), ensure_ascii=False
                            ),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ]
            assistant_indices.append(len(messages))
            messages.append(assistant_msg)

            observation = step.get("observation") or {}
            for result in observation.get("results", []) or []:
                tool_msg: Dict = {
                    "role": "tool",
                    "content": extract_text_content(result.get("content", "")),
                }
                call_id = result.get("source_call_id")
                if call_id:
                    tool_msg["tool_call_id"] = call_id
                messages.append(tool_msg)
        # Unknown sources are ignored.

    return messages, assistant_indices


def derive_tools_from_trajectory(trajectory: Dict) -> Optional[List[Dict]]:
    """Synthesize a minimal ``tools`` schema from tool names in the trajectory.

    Many chat templates require a ``tools`` argument to render assistant
    ``tool_calls`` (and to emit the tool-definition preamble). We don't have
    Codex's exact JSON schemas, so we build permissive stubs keyed by the
    function names actually invoked. Returns ``None`` when no tools appear.
    """
    names: List[str] = []
    seen: set[str] = set()
    for step in trajectory.get("steps", []):
        for tc in step.get("tool_calls") or []:
            name = tc.get("function_name")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    if not names:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def mask_assistant_tokens(
    messages: List[Dict],
    assistant_indices: Sequence[int],
    tokenizer: object,
    tools: Optional[List[Dict]] = None,
) -> Tuple[List[int], List[int]]:
    """Render ``messages`` with the target chat template and mask agent tokens.

    For every assistant turn ``k`` we diff the rendered token stream of
    ``messages[:k]`` (with a generation prompt) against ``messages[:k+1]``; the
    suffix present only in the latter is exactly that turn's model-generated
    span, which receives ``loss_mask = 1``. ``input_ids`` is the render of all
    messages.

    Assumes a prefix-stable template (rendering more messages only appends
    tokens), which holds for Llama/Qwen/Kimi-style chat templates.
    """

    def render(msgs: List[Dict], add_generation_prompt: bool) -> List[int]:
        return list(
            tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                msgs,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
            )
        )

    spans: List[Tuple[int, int]] = []
    for k in sorted(set(assistant_indices)):
        prefix_ids = render(messages[:k], True) if k > 0 else []
        through_ids = render(messages[: k + 1], False)
        start = common_prefix_len(prefix_ids, through_ids)
        if len(through_ids) > start:
            spans.append((start, len(through_ids)))

    input_ids = render(messages, False)
    loss_mask = [0] * len(input_ids)
    n = len(input_ids)
    for start, end in spans:
        for i in range(start, min(end, n)):
            loss_mask[i] = 1
    return input_ids, loss_mask


# --------------------------------------------------------------------------- #
# Harbor job orchestration (lazy harbor import)
# --------------------------------------------------------------------------- #


def discover_tasks(
    tasks_dir: Path,
    task_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Path]:
    """Find Harbor task directories (those containing a ``task.toml``)."""
    root = Path(tasks_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"tasks-dir does not exist: {root}\n"
            "Generate SWE-bench tasks first, e.g.:\n"
            "  cd harbor/adapters/swebench && uv run swebench --limit 50"
        )
    dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "task.toml").exists()
    )
    if task_ids:
        wanted = set(task_ids)
        dirs = [p for p in dirs if p.name in wanted]
    if limit is not None:
        dirs = dirs[:limit]
    return dirs


async def run_rollouts(task_paths: List[Path], args: argparse.Namespace):
    """Build and run a Harbor Job; return its JobResult."""
    # Imported lazily so the reconstruction helpers (and --self-test) work in
    # environments without harbor's heavy optional deps (e.g. litellm).
    from harbor.job import Job
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.job.config import JobConfig
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
    )

    if args.agent == "codex":
        # Codex is a CLI agent: it talks to an OpenAI-compatible endpoint via
        # OPENAI_BASE_URL / OPENAI_API_KEY (here, the local SGLang target) and
        # writes an ATIF trajectory we re-tokenize afterwards. model_name is
        # passed straight to `codex exec --model` after stripping any provider
        # prefix, so 'openai/kimi-k2.7-code' -> '--model kimi-k2.7-code'.
        agent_kwargs: Dict[str, object] = {
            "reasoning_effort": args.reasoning_effort,
        }
        if args.web_search:
            agent_kwargs["web_search"] = args.web_search
        agent_config = AgentConfig(
            name="codex",
            model_name=args.model_name,
            kwargs=agent_kwargs,
            env={
                "OPENAI_BASE_URL": args.api_base,
                "OPENAI_API_KEY": args.openai_api_key or "sk-local-sglang",
            },
        )
    else:
        # Terminus-2 streams exact prompt/completion token ids from the server.
        # NOTE: AgentConfig.kwargs is splatted directly into Terminus2.__init__,
        # so use the constructor's real parameter names (parser_name, not the
        # YAML's 'parser'). collect_rollout_details=True makes LiteLLM request
        # token ids.
        agent_kwargs = {
            "parser_name": args.parser,
            "api_base": args.api_base,
            "collect_rollout_details": True,
        }
        if args.temperature is not None:
            agent_kwargs["temperature"] = args.temperature
        if args.max_turns is not None:
            agent_kwargs["max_turns"] = args.max_turns
        agent_config = AgentConfig(
            name="terminus-2",
            model_name=args.model_name,
            kwargs=agent_kwargs,
        )

    config = JobConfig(
        job_name=args.job_name,
        jobs_dir=Path(args.jobs_dir),
        n_concurrent_trials=args.n_concurrent,
        n_attempts=args.n_attempts,
        environment=EnvironmentConfig(
            type=EnvironmentType.DOCKER,
            force_build=args.force_build,
            delete=not args.keep_containers,
        ),
        agents=[agent_config],
        tasks=[TaskConfig(path=p) for p in task_paths],
    )

    job = Job(config=config)
    return await job.run()


def extract_rows_from_result(
    job_result, args: argparse.Namespace
) -> Tuple[List[Dict], Dict[str, int]]:
    """Convert a JobResult into pretokenized rows + summary stats."""
    rows: List[Dict] = []
    stats = {
        "trials": 0,
        "with_tokens": 0,
        "missing_tokens": 0,
        "resolved": 0,
        "segments": 0,
        "windows": 0,
    }

    for tr in job_result.trial_results:
        stats["trials"] += 1
        instance_id = getattr(tr, "task_name", None) or getattr(
            tr, "trial_name", "unknown"
        )

        reward = 0.0
        vr = getattr(tr, "verifier_result", None)
        if vr is not None and vr.rewards:
            reward = float(vr.rewards.get("reward", 0) or 0)
        if reward >= 1.0:
            stats["resolved"] += 1

        ctx = getattr(tr, "agent_result", None)
        rollout_details = getattr(ctx, "rollout_details", None) if ctx else None
        if not rollout_details:
            stats["missing_tokens"] += 1
            print(
                f"  [warn] {instance_id}: no rollout_details "
                "(server did not return token ids; ensure the SGLang OpenAI "
                "endpoint returns token ids and collect_rollout_details=True)"
            )
            continue

        rd = rollout_details[0]
        prompt_ids = rd.get("prompt_token_ids")
        completion_ids = rd.get("completion_token_ids")
        if not prompt_ids or not completion_ids:
            stats["missing_tokens"] += 1
            print(
                f"  [warn] {instance_id}: rollout_details present but missing "
                "prompt/completion token ids; skipping."
            )
            continue

        stats["with_tokens"] += 1
        segments = reconstruct_trajectory(
            prompt_ids, completion_ids, seam_tolerance=args.seam_tolerance
        )
        for seg_input_ids, seg_loss_mask in segments:
            stats["segments"] += 1
            windows = window_sequence(
                seg_input_ids,
                seg_loss_mask,
                max_length=args.max_length,
                stride=args.stride,
                min_loss_tokens=args.min_loss_tokens,
            )
            for w_ids, w_mask in windows:
                stats["windows"] += 1
                rows.append(
                    {
                        "instance_id": instance_id,
                        "reward": reward,
                        "input_ids": w_ids,
                        "loss_mask": w_mask,
                    }
                )

    return rows, stats


def _trial_dir_from_uri(trial_uri: Optional[str]) -> Optional[Path]:
    """Resolve a trial's on-disk directory from its ``file://`` ``trial_uri``."""
    if not trial_uri:
        return None
    parsed = urlparse(trial_uri)
    if parsed.scheme and parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    return Path(path) if path else None


def _codex_trajectory_path(trial_dir: Optional[Path]) -> Optional[Path]:
    """Locate the Codex ATIF ``trajectory.json`` written under ``agent/``.

    The Codex agent writes it to its ``logs_dir`` (the trial's ``agent/`` dir);
    multi-step trials relocate logs under ``steps/<name>/agent/``, so fall back
    to a recursive search.
    """
    if trial_dir is None:
        return None
    candidate = trial_dir / "agent" / "trajectory.json"
    if candidate.exists():
        return candidate
    for base in (trial_dir / "agent", trial_dir):
        if base.exists():
            matches = sorted(base.rglob("trajectory.json"))
            if matches:
                return matches[0]
    return None


def extract_rows_from_codex(
    job_result, args: argparse.Namespace, tokenizer: object
) -> Tuple[List[Dict], Dict[str, int]]:
    """Convert Codex ATIF trajectories into pretokenized rows + summary stats.

    Mirrors :func:`extract_rows_from_result` but sources tokens by re-rendering
    each trial's ``trajectory.json`` through the target tokenizer instead of
    reading server-streamed ``rollout_details``.
    """
    rows: List[Dict] = []
    stats = {
        "trials": 0,
        "with_tokens": 0,
        "missing_tokens": 0,
        "resolved": 0,
        "segments": 0,
        "windows": 0,
    }

    for tr in job_result.trial_results:
        stats["trials"] += 1
        instance_id = getattr(tr, "task_name", None) or getattr(
            tr, "trial_name", "unknown"
        )

        reward = 0.0
        vr = getattr(tr, "verifier_result", None)
        if vr is not None and vr.rewards:
            reward = float(vr.rewards.get("reward", 0) or 0)
        if reward >= 1.0:
            stats["resolved"] += 1

        traj_path = _codex_trajectory_path(
            _trial_dir_from_uri(getattr(tr, "trial_uri", None))
        )
        if traj_path is None:
            stats["missing_tokens"] += 1
            print(
                f"  [warn] {instance_id}: no Codex trajectory.json found "
                "(agent may have failed before producing a session)."
            )
            continue

        try:
            trajectory = json.loads(traj_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            stats["missing_tokens"] += 1
            print(f"  [warn] {instance_id}: failed to read {traj_path}: {exc}")
            continue

        messages, assistant_indices = reconstruct_codex_messages(trajectory)
        if not assistant_indices:
            stats["missing_tokens"] += 1
            print(f"  [warn] {instance_id}: trajectory has no assistant turns.")
            continue

        tools = derive_tools_from_trajectory(trajectory)
        try:
            input_ids, loss_mask = mask_assistant_tokens(
                messages, assistant_indices, tokenizer, tools=tools
            )
        except Exception as exc:  # chat-template rendering is best-effort
            stats["missing_tokens"] += 1
            print(
                f"  [warn] {instance_id}: chat-template re-tokenization failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if sum(loss_mask) == 0:
            stats["missing_tokens"] += 1
            print(f"  [warn] {instance_id}: no supervised tokens after masking.")
            continue

        stats["with_tokens"] += 1
        stats["segments"] += 1
        windows = window_sequence(
            input_ids,
            loss_mask,
            max_length=args.max_length,
            stride=args.stride,
            min_loss_tokens=args.min_loss_tokens,
        )
        for w_ids, w_mask in windows:
            stats["windows"] += 1
            rows.append(
                {
                    "instance_id": instance_id,
                    "reward": reward,
                    "input_ids": w_ids,
                    "loss_mask": w_mask,
                }
            )

    return rows, stats


def _load_target_tokenizer(args: argparse.Namespace):
    """Load the target model's tokenizer for Codex trajectory re-tokenization."""
    model = args.target_model or args.model_name
    if model and model.startswith("openai/"):
        # 'openai/<served-name>' is a LiteLLM routing alias, not an HF id.
        model = model.split("/", 1)[1]
    if not model:
        raise SystemExit(
            "codex agent requires --target-model (or $TARGET_MODEL) so the "
            "trajectory can be re-tokenized with the target chat template."
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "codex agent requires the 'transformers' package to re-tokenize "
            f"trajectories ({exc})."
        )
    print(f"Loading target tokenizer for re-tokenization: {model}")
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def write_split(
    rows: List[Dict],
    output_dir: Path,
    train_name: str,
    eval_name: str,
    eval_ratio: float,
    seed: int,
) -> Tuple[Path, Optional[Path], int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(rows)

    n_eval = 0
    if eval_ratio > 0 and len(rows) > 1:
        n_eval = max(1, int(round(len(rows) * eval_ratio)))
        n_eval = min(n_eval, len(rows) - 1)  # keep at least 1 train row

    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    train_path = output_dir / train_name
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    eval_path: Optional[Path] = None
    if eval_rows:
        eval_path = output_dir / eval_name
        with eval_path.open("w", encoding="utf-8") as f:
            for r in eval_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return train_path, eval_path, len(train_rows), len(eval_rows)


# --------------------------------------------------------------------------- #
# Self-test (no harbor / torch required)
# --------------------------------------------------------------------------- #


def _self_test() -> int:
    failures = 0

    def check(name, cond):
        nonlocal failures
        status = "ok" if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{status}] {name}")

    # Clean linear chat: P1 extends P0+C0 by end-of-turn+tool+header tokens.
    P0 = [1, 2, 3, 90]  # system+user+assistant header
    C0 = [10, 11, 12]  # assistant generated
    # next prompt = P0 + C0 + [eot, tool tokens..., assistant header]
    P1 = P0 + C0 + [99, 4, 5, 90]
    C1 = [20, 21]
    segs = reconstruct_trajectory([P0, P1], [C0, C1])
    check("linear: single segment", len(segs) == 1)
    ids, mask = segs[0]
    expected_ids = P0 + C0 + [99, 4, 5, 90] + C1
    expected_mask = (
        [0] * len(P0) + [1] * len(C0) + [0] * 4 + [1] * len(C1)
    )
    check("linear: input_ids", ids == expected_ids)
    check("linear: loss_mask", mask == expected_mask)
    check("linear: loss on generated only",
          [ids[i] for i, m in enumerate(mask) if m] == C0 + C1)

    # Benign seam: last token of C0 merged with eot at the boundary (<= tol).
    P0b = [1, 2, 90]
    C0b = [10, 11, 12]
    # re-render keeps 10,11 but merges 12+eot -> token 777 (1-token divergence)
    P1b = [1, 2, 90, 10, 11, 777, 4, 90]
    C1b = [22]
    segs_b = reconstruct_trajectory([P0b, P1b], [C0b, C1b], seam_tolerance=8)
    check("seam: single segment", len(segs_b) == 1)

    # Hard divergence (summarization) -> two segments.
    P0c = list(range(100, 140))
    C0c = [200, 201, 202]
    P1c = [1, 2, 3, 4]  # totally different (context reset)
    C1c = [300, 301]
    segs_c = reconstruct_trajectory([P0c, P1c], [C0c, C1c], seam_tolerance=8)
    check("summarization: two segments", len(segs_c) == 2)

    # Windowing: long sequence splits; short keeps; low-loss dropped.
    long_ids = list(range(1000))
    long_mask = [0] * 900 + [1] * 100
    w = window_sequence(long_ids, long_mask, max_length=400, stride=200,
                        min_loss_tokens=16)
    check("window: produced windows", len(w) >= 1)
    check("window: all <= max_length", all(len(x[0]) <= 400 for x in w))
    check("window: all meet min loss", all(sum(x[1]) >= 16 for x in w))

    short = window_sequence([1, 2, 3, 4, 5], [0, 1, 1, 1, 1], max_length=64,
                            stride=32, min_loss_tokens=2)
    check("window: short kept", len(short) == 1)
    dropped = window_sequence([1, 2, 3], [0, 0, 0], max_length=64, stride=32,
                              min_loss_tokens=2)
    check("window: zero-loss dropped", dropped == [])

    # --- Codex ATIF reconstruction + assistant masking -------------------- #
    class _FakeTok:
        """Deterministic role/word tokenizer for masking tests (no deps)."""

        def __init__(self) -> None:
            self._vocab: dict = {}

        def _id(self, tok: str) -> int:
            return self._vocab.setdefault(tok, len(self._vocab) + 1)

        def apply_chat_template(self, messages, tokenize=True,
                                add_generation_prompt=False, tools=None):
            toks: List[str] = []
            for m in messages:
                toks.append(f"<{m['role']}>")
                toks.extend((m.get("content") or "").split())
                for tc in m.get("tool_calls") or []:
                    toks.append(f"<call:{tc['function']['name']}>")
                toks.append(f"</{m['role']}>")
            if add_generation_prompt:
                toks.append("<assistant>")
            return [self._id(t) for t in toks] if tokenize else " ".join(toks)

    traj = {
        "steps": [
            {"step_id": 1, "source": "system", "message": "be good"},
            {"step_id": 2, "source": "user", "message": "fix bug"},
            {"step_id": 3, "source": "agent", "message": "looking now",
             "tool_calls": [{"tool_call_id": "c1", "function_name": "shell",
                             "arguments": {"cmd": "ls"}}],
             "observation": {"results": [{"source_call_id": "c1",
                                          "content": "file.py"}]}},
            {"step_id": 4, "source": "agent", "message": "all done"},
            # copied-context steps must be skipped entirely:
            {"step_id": 5, "source": "agent", "message": "ghost",
             "is_copied_context": True},
        ]
    }
    msgs, a_idx = reconstruct_codex_messages(traj)
    check("codex: message count", len(msgs) == 5)  # sys,user,asst,tool,asst
    check("codex: assistant indices", a_idx == [2, 4])
    check("codex: tool message role", msgs[3]["role"] == "tool")
    check("codex: tool result linked", msgs[3].get("tool_call_id") == "c1")
    check("codex: tool call mapped",
          msgs[2]["tool_calls"][0]["function"]["name"] == "shell")
    check("codex: copied-context skipped",
          all("ghost" != m.get("content") for m in msgs))

    check("codex: derive tools", derive_tools_from_trajectory(traj) ==
          [{"type": "function", "function": {"name": "shell", "description": "",
            "parameters": {"type": "object", "properties": {}}}}])

    tok = _FakeTok()
    ids, mask = mask_assistant_tokens(msgs, a_idx, tok, tools=None)
    check("codex: ids/mask aligned", len(ids) == len(mask) and len(ids) > 0)
    inv = {v: k for k, v in tok._vocab.items()}
    masked = [inv[i] for i, m in zip(ids, mask) if m]
    check("codex: masked agent content",
          "looking" in masked and "done" in masked)
    check("codex: masked tool-call token", "<call:shell>" in masked)
    check("codex: did not mask user text", "fix" not in masked)
    check("codex: did not mask tool output", "file.py" not in masked)
    check("codex: did not mask system text", "good" not in masked)

    # file:// trial_uri -> Path round-trip used to locate trajectory.json.
    check("codex: trial uri parse",
          _trial_dir_from_uri("file:///tmp/jobs/t__abc1234")
          == Path("/tmp/jobs/t__abc1234"))

    print(f"\nself-test: {'PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    specforge_root = here.parents[1]              # SpecForge/
    workspace_root = here.parents[2]              # Spec/ (holds SpecForge/, harbor/, sglang/)
    default_tasks = workspace_root / "harbor" / "datasets" / "swebench-verified"
    default_out = specforge_root / "cache" / "dataset" / "swebench-agentic"
    default_jobs = specforge_root / "cache" / "harbor_jobs"

    p = argparse.ArgumentParser(
        description="Generate agentic SWE-bench trajectories for DFlash.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # task selection
    p.add_argument("--tasks-dir", type=Path, default=default_tasks)
    p.add_argument("--task-ids", nargs="+", default=None,
                   help="Only run these task (instance) ids.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of tasks to run.")

    # serving / agent
    p.add_argument("--agent", type=str, default="codex",
                   choices=["codex", "terminus-2"],
                   help="Harbor agent driving the rollouts. 'codex' runs the "
                        "Codex CLI against the target and re-tokenizes its ATIF "
                        "trajectory (requires --target-model); 'terminus-2' "
                        "streams exact token ids from the server.")
    p.add_argument("--api-base", type=str, default="http://127.0.0.1:30000/v1",
                   help="OpenAI-compatible base URL of the SGLang target server. "
                        "For codex this is exported as OPENAI_BASE_URL.")
    p.add_argument("--model-name", type=str, default="openai/kimi-k2.7-code",
                   help="LiteLLM model name (provider-prefixed) for the target. "
                        "For codex the provider prefix is stripped for "
                        "`codex exec --model`.")
    p.add_argument("--parser", type=str, default="xml", choices=["xml", "json"],
                   help="Terminus-2 action parser (maps to parser_name).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (terminus-2 only).")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Max agent turns (terminus-2 only).")

    # codex agent
    p.add_argument("--target-model", type=str,
                   default=os.environ.get("TARGET_MODEL"),
                   help="HF model id/path whose tokenizer + chat template "
                        "re-tokenizes Codex trajectories (codex only). "
                        "Defaults to $TARGET_MODEL.")
    p.add_argument("--reasoning-effort", type=str, default="high",
                   choices=["minimal", "low", "medium", "high"],
                   help="Codex model_reasoning_effort (codex only).")
    p.add_argument("--web-search", type=str, default=None,
                   choices=["disabled", "cached", "live"],
                   help="Codex web_search mode (codex only; default off).")
    p.add_argument("--openai-api-key", type=str,
                   default=os.environ.get("OPENAI_API_KEY"),
                   help="API key Codex sends to the endpoint. Any non-empty "
                        "value works for a local SGLang server.")

    # orchestration
    p.add_argument("--n-concurrent", type=int, default=8)
    p.add_argument("--n-attempts", type=int, default=1)
    p.add_argument("--jobs-dir", type=Path, default=default_jobs)
    p.add_argument("--job-name", type=str,
                   default=datetime.now(timezone.utc).strftime("swebench-%Y%m%d-%H%M%S"))
    p.add_argument("--force-build", action="store_true",
                   help="Force rebuild of task Docker images.")
    p.add_argument("--keep-containers", action="store_true",
                   help="Do not delete sandboxes after each trial (debug).")

    # reconstruction / windowing
    p.add_argument("--max-length", type=int, default=16384,
                   help="Window length for long trajectories.")
    p.add_argument("--stride", type=int, default=None,
                   help="Window stride (default max_length // 2).")
    p.add_argument("--min-loss-tokens", type=int, default=16,
                   help="Drop windows with fewer supervised tokens "
                        "(should be >= 2 * draft block_size).")
    p.add_argument("--seam-tolerance", type=int, default=DEFAULT_SEAM_TOLERANCE,
                   help="Max token divergence treated as a benign tokenizer "
                        "seam before splitting into a new segment.")

    # output
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--train-name", type=str, default="swebench_agentic_train.jsonl")
    p.add_argument("--eval-name", type=str, default="swebench_agentic_eval.jsonl")
    p.add_argument("--eval-ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)

    # misc
    p.add_argument("--dry-run", action="store_true",
                   help="List discovered tasks and exit (no rollouts).")
    p.add_argument("--self-test", action="store_true",
                   help="Run reconstruction/windowing unit tests and exit.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.self_test:
        return _self_test()

    import asyncio

    task_paths = discover_tasks(args.tasks_dir, args.task_ids, args.limit)
    print(f"Discovered {len(task_paths)} SWE-bench task(s) under {args.tasks_dir}")
    if not task_paths:
        print("No tasks to run. Did you run the harbor swebench adapter?")
        return 1

    if args.dry_run:
        for p in task_paths:
            print(f"  - {p.name}")
        return 0

    print(
        f"Running rollouts: agent={args.agent} model={args.model_name} "
        f"api_base={args.api_base} n_concurrent={args.n_concurrent}"
    )

    # For codex we re-tokenize ATIF trajectories afterwards, so fail fast on a
    # missing/unloadable target tokenizer *before* spending GPU time rolling out.
    tokenizer = _load_target_tokenizer(args) if args.agent == "codex" else None

    job_result = asyncio.run(run_rollouts(task_paths, args))

    if args.agent == "codex":
        rows, stats = extract_rows_from_codex(job_result, args, tokenizer)
    else:
        rows, stats = extract_rows_from_result(job_result, args)
    print(
        "\nRollout summary: "
        f"trials={stats['trials']} with_tokens={stats['with_tokens']} "
        f"missing_tokens={stats['missing_tokens']} resolved={stats['resolved']} "
        f"segments={stats['segments']} windows={stats['windows']}"
    )

    if not rows:
        if args.agent == "codex":
            print(
                "No training rows produced. Check that Codex actually reached "
                "the target (the Codex CLI defaults to the OpenAI Responses "
                "API; if your SGLang server only serves /v1/chat/completions, "
                "configure a Codex model_provider with wire_api=\"chat\"), and "
                "that trajectory.json files exist under each trial's agent/ dir."
            )
        else:
            print(
                "No training rows produced. Most likely the server did not "
                "return token ids. Verify the SGLang OpenAI endpoint returns "
                "token ids (prompt/completion) for chat completions."
            )
        return 1

    train_path, eval_path, n_train, n_eval = write_split(
        rows,
        args.output_dir,
        args.train_name,
        args.eval_name,
        args.eval_ratio,
        args.seed,
    )
    print(f"Wrote {n_train} train rows -> {train_path}")
    if eval_path is not None:
        print(f"Wrote {n_eval} eval rows -> {eval_path}")
    print(
        "\nNext: train with\n"
        f"  --train-data-path {train_path} "
        f"--eval-data-path {eval_path or '<none>'} --data-format pretokenized"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
