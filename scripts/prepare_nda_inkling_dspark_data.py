#!/usr/bin/env python3
# coding=utf-8
"""Build the NDA/Inkling DSpark training corpus from regen output shards.

Unlike the GLM prepare script (which downloads two finished HF datasets), the
Inkling corpus is assembled from however many Step-2 regeneration outputs exist
at kickoff — complete shards, partial shards, or files gathered from several
worker boxes (owner: "we won't get full shards; maybe only 400k samples").
So this script deliberately does NOT run the 5-shard `regen.validate` contract;
it ingests any set of regen SUCCESS files and applies per-row gates only:

  * `status == "ok"` when the field is present (error rows live in separate
    `*_error.jsonl` files anyway — do not pass those);
  * every assistant turn has non-empty `content` (reasoning_content may be
    empty only if content is not — matches the regen acceptance rule);
  * no literal Inkling control token inside any text field (defense in depth;
    regen's ingress gate already rejects these);
  * >= 2 messages and >= 1 assistant turn.

Rows are deduplicated by the regen sidecar identity — `regeneration.shard
.input_index` (the global input position, unique across shards) with
`(source, id)` as fallback — so overlapping partial copies of the same shard
merge cleanly. The assistant messages KEEP the split reasoning_content/content
form: SpecForge's packaged `nda-inkling-thinking` Jinja renders that split
byte-identically to what SGLang served during regeneration.

Output: shuffled (seed) train jsonl + a small held-out eval jsonl, each row
`{"conversations": [...]}` only (regen metadata stripped).

Example:
    python scripts/prepare_nda_inkling_dspark_data.py \
        --inputs '/data/regen/regen.shard*of5.jsonl' \
        --output-dir /scratch/nda_data --eval-size 2000
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys

# Literal control tokens that must never appear inside stored text fields.
_CONTROL_TOKENS = (
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
)

_KEEP_KEYS = ("role", "content", "reasoning_content", "name")


def _clean_message(message: dict):
    """Strip a message to the render-relevant keys; None -> absent."""
    out = {}
    for key in _KEEP_KEYS:
        value = message.get(key)
        if value is not None and value != "":
            out[key] = value
    if "role" not in out:
        return None
    return out


def _row_identity(row: dict, fallback: str):
    # regen_contract.source_index is the row position in the ORIGINAL source
    # sample and is globally unique across ALL prepared inputs (the 400k input
    # and the extra-400k-600k input carry disjoint source-index sets).
    # shard.input_index is only unique WITHIN one prepared input: the delivered
    # S2/S3 shards (extra input, local positions [0,200000)) collide with
    # S0/S1 under it, so keying on input_index alone silently drops ~175k
    # legitimate rows when the two corpora are merged.
    contract = row.get("regen_contract") or {}
    source_index = contract.get("source_index")
    if source_index is not None:
        return ("source_index", int(source_index))
    regen = row.get("regeneration") or {}
    shard = regen.get("shard") or {}
    input_index = shard.get("input_index")
    if input_index is not None:
        return ("input_index", int(input_index))
    source, row_id = row.get("source"), row.get("id")
    if source is not None and row_id is not None:
        return ("source_id", str(source), str(row_id))
    return ("line", fallback)


_OK_STATUSES = ("success", "ok")


def _gate_row(row: dict):
    """Return (conversations, None) when the row passes, (None, reason) else."""
    status = row.get("status")
    if status is not None and status not in _OK_STATUSES:
        return None, f"status={status}"
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None, "too_few_messages"

    cleaned, num_assistant = [], 0
    for message in conversations:
        if not isinstance(message, dict):
            return None, "non_dict_message"
        msg = _clean_message(message)
        if msg is None:
            return None, "missing_role"
        for field in ("content", "reasoning_content"):
            text = msg.get(field)
            if isinstance(text, str) and any(t in text for t in _CONTROL_TOKENS):
                return None, "control_token_leak"
        if msg["role"] == "assistant":
            num_assistant += 1
            if not msg.get("content"):
                return None, "empty_assistant_content"
            finish_reason = message.get("finish_reason")
            if finish_reason is not None and finish_reason != "stop":
                return None, f"finish_reason={finish_reason}"
        cleaned.append(msg)
    if num_assistant == 0:
        return None, "no_assistant_turn"
    return cleaned, None


def main():
    ap = argparse.ArgumentParser(
        description="Prepare NDA/Inkling DSpark corpus from regen success files"
    )
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Regen SUCCESS jsonl paths or globs (partial shards fine; never "
        "pass *_error.jsonl files).",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-name", default="nda_inkling_dspark_train.jsonl")
    ap.add_argument("--eval-name", default="nda_inkling_dspark_eval.jsonl")
    ap.add_argument("--eval-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-rows", type=int, default=None, help="Optional cap after dedup+shuffle."
    )
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        matched = sorted(glob.glob(pattern))
        if not matched and os.path.exists(pattern):
            matched = [pattern]
        if not matched:
            print(f"WARNING: no files match {pattern!r}", file=sys.stderr)
        paths.extend(matched)
    if not paths:
        raise SystemExit("no input files found")
    for path in paths:
        if "_error" in os.path.basename(path):
            raise SystemExit(
                f"{path} looks like an ERROR file; pass success files only"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    kept, seen, seen_content = [], set(), set()
    stats = {"rows": 0, "dup": 0, "dup_content": 0, "bad_json": 0}
    rejects = {}
    for path in paths:
        before = len(kept)
        with open(path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                stats["rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                identity = _row_identity(row, f"{path}:{line_no}")
                if identity in seen:
                    stats["dup"] += 1
                    continue
                conversations, reason = _gate_row(row)
                if conversations is None:
                    rejects[reason] = rejects.get(reason, 0) + 1
                    continue
                serialized = json.dumps(
                    {"conversations": conversations}, ensure_ascii=False
                )
                # Content-level dedup: identity dedup cannot see the same
                # conversation arriving under different identity schemes
                # (e.g. regen shards keyed by input_index vs node files
                # keyed by (source, id)). First occurrence wins, so input
                # order sets precedence.
                content_key = hashlib.md5(serialized.encode("utf-8")).hexdigest()
                if content_key in seen_content:
                    stats["dup_content"] += 1
                    continue
                seen.add(identity)
                seen_content.add(content_key)
                kept.append(serialized)
        print(f"{path}: +{len(kept) - before} rows (total {len(kept)})", flush=True)

    if not kept:
        raise SystemExit("no rows survived the gates")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    if args.max_rows is not None:
        kept = kept[: args.max_rows]

    eval_size = min(args.eval_size, max(0, len(kept) - 1))
    eval_rows, train_rows = kept[:eval_size], kept[eval_size:]

    train_path = os.path.join(args.output_dir, args.train_name)
    eval_path = os.path.join(args.output_dir, args.eval_name)
    for out_path, rows in ((train_path, train_rows), (eval_path, eval_rows)):
        with open(out_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row + "\n")

    print(
        json.dumps(
            {
                "inputs": len(paths),
                "rows_read": stats["rows"],
                "bad_json": stats["bad_json"],
                "duplicates": stats["dup"],
                "duplicates_content": stats["dup_content"],
                "rejected": rejects,
                "train": {"path": train_path, "rows": len(train_rows)},
                "eval": {"path": eval_path, "rows": len(eval_rows)},
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
