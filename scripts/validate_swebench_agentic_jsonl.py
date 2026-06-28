#!/usr/bin/env python3
# coding=utf-8
"""Validate Codex SWE-bench rollout JSONL against saved trajectories.

This verifier is intentionally independent of Harbor runtime objects. It reads
the on-disk ``agent/trajectory.json`` files from a completed Harbor jobs dir,
re-renders them with the same target tokenizer/chat template used by
``run_swebench_rollouts.py``, windows the resulting ``input_ids``/``loss_mask``
pairs, and compares those windows with the emitted train/eval JSONL rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_rollout_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("run_swebench_rollouts", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _row_digest(row: Dict) -> str:
    payload = json.dumps(
        {
            "instance_id": row.get("instance_id"),
            "reward": row.get("reward"),
            "input_ids": row.get("input_ids"),
            "loss_mask": row.get("loss_mask"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_rows(rows: Iterable[Dict], max_length: int, min_loss_tokens: int) -> Dict[str, int]:
    stats = {"rows": 0, "tokens": 0, "loss_tokens": 0}
    for idx, row in enumerate(rows):
        input_ids = row.get("input_ids")
        loss_mask = row.get("loss_mask")
        if not isinstance(input_ids, list) or not isinstance(loss_mask, list):
            raise AssertionError(f"row {idx}: input_ids/loss_mask must be lists")
        if len(input_ids) != len(loss_mask):
            raise AssertionError(
                f"row {idx}: length mismatch input_ids={len(input_ids)} loss_mask={len(loss_mask)}"
            )
        if not input_ids:
            raise AssertionError(f"row {idx}: empty input_ids")
        if len(input_ids) > max_length:
            raise AssertionError(f"row {idx}: length {len(input_ids)} exceeds max_length={max_length}")
        if any(m not in (0, 1) for m in loss_mask):
            raise AssertionError(f"row {idx}: loss_mask contains values outside 0/1")
        loss_sum = sum(loss_mask)
        if loss_sum < min_loss_tokens:
            raise AssertionError(
                f"row {idx}: loss tokens {loss_sum} below min_loss_tokens={min_loss_tokens}"
            )
        if not all(isinstance(tok, int) and tok >= 0 for tok in input_ids):
            raise AssertionError(f"row {idx}: input_ids must be non-negative ints")

        stats["rows"] += 1
        stats["tokens"] += len(input_ids)
        stats["loss_tokens"] += loss_sum
    return stats


def _trajectory_paths(jobs_dir: Path) -> List[Path]:
    return sorted(jobs_dir.glob("*/agent/trajectory.json"))


def _rows_from_trajectories(
    rollout,
    tokenizer,
    jobs_dir: Path,
    max_length: int,
    stride: int | None,
    min_loss_tokens: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    rows: List[Dict] = []
    stats = {
        "trajectories": 0,
        "with_assistant": 0,
        "with_loss": 0,
        "windows": 0,
        "missing_loss": 0,
    }

    for traj_path in _trajectory_paths(jobs_dir):
        stats["trajectories"] += 1
        trial_dir = traj_path.parents[1]
        # Trial dir names are <instance_id>__<shortuuid>; instance ids also use
        # double underscores, so remove only the final suffix.
        instance_id = trial_dir.name.rsplit("__", 1)[0]
        trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
        messages, assistant_indices = rollout.reconstruct_codex_messages(trajectory)
        if not assistant_indices:
            continue
        stats["with_assistant"] += 1
        tools = rollout.derive_tools_from_trajectory(trajectory)
        input_ids, loss_mask = rollout.mask_assistant_tokens(
            messages, assistant_indices, tokenizer, tools=tools
        )
        if sum(loss_mask) == 0:
            stats["missing_loss"] += 1
            continue
        stats["with_loss"] += 1
        for w_ids, w_mask in rollout.window_sequence(
            input_ids,
            loss_mask,
            max_length=max_length,
            stride=stride,
            min_loss_tokens=min_loss_tokens,
        ):
            stats["windows"] += 1
            rows.append(
                {
                    "instance_id": instance_id,
                    # The rollout rows use verifier reward. Tokenization and
                    # masks do not depend on reward, so this is overwritten
                    # before multiset comparison.
                    "reward": 0.0,
                    "input_ids": w_ids,
                    "loss_mask": w_mask,
                }
            )

    return rows, stats


def main() -> int:
    p = argparse.ArgumentParser(
        description="Validate pretokenized Codex SWE-bench rollout JSONL."
    )
    p.add_argument("--jobs-dir", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, default=None)
    p.add_argument("--target-model", required=True)
    p.add_argument("--rollout-script", type=Path, default=Path(__file__).with_name("run_swebench_rollouts.py"))
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--min-loss-tokens", type=int, default=32)
    args = p.parse_args()

    rollout = _load_rollout_module(args.rollout_script)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"transformers is required: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    train_rows = _load_jsonl(args.train)
    eval_rows = _load_jsonl(args.eval) if args.eval else []
    emitted_rows = train_rows + eval_rows
    emitted_stats = _validate_rows(emitted_rows, args.max_length, args.min_loss_tokens)

    recomputed_rows, recomputed_stats = _rows_from_trajectories(
        rollout,
        tokenizer,
        args.jobs_dir,
        max_length=args.max_length,
        stride=args.stride,
        min_loss_tokens=args.min_loss_tokens,
    )
    _validate_rows(recomputed_rows, args.max_length, args.min_loss_tokens)

    # Rewards are verifier metadata, not part of tokenization. Preserve emitted
    # reward values by comparing token/mask/window identity separately.
    emitted_token_counter = Counter(
        hashlib.sha256(
            json.dumps(
                {
                    "input_ids": r.get("input_ids"),
                    "loss_mask": r.get("loss_mask"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for r in emitted_rows
    )
    recomputed_token_counter = Counter(
        hashlib.sha256(
            json.dumps(
                {
                    "input_ids": r.get("input_ids"),
                    "loss_mask": r.get("loss_mask"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for r in recomputed_rows
    )
    if emitted_token_counter != recomputed_token_counter:
        missing = recomputed_token_counter - emitted_token_counter
        extra = emitted_token_counter - recomputed_token_counter
        raise AssertionError(
            "emitted JSONL rows do not match re-rendered trajectory windows: "
            f"missing={sum(missing.values())} extra={sum(extra.values())}"
        )

    print("validation: PASS")
    print(
        "emitted rows: "
        f"train={len(train_rows)} eval={len(eval_rows)} total={emitted_stats['rows']} "
        f"tokens={emitted_stats['tokens']} loss_tokens={emitted_stats['loss_tokens']}"
    )
    print(
        "trajectories: "
        f"total={recomputed_stats['trajectories']} "
        f"with_assistant={recomputed_stats['with_assistant']} "
        f"with_loss={recomputed_stats['with_loss']} "
        f"windows={recomputed_stats['windows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
