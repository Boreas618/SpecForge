#!/usr/bin/env python3
# coding=utf-8
"""Seeded uniform sample of a regen success jsonl, for corpus assembly.

Written for regen_node2.jsonl (owner: "sample 100k from node2"): its rows
carry only {id, conversations, status} — no regeneration sidecar and no
source — so the identity dedup in prepare_nda_inkling_dspark_data.py would
fall back to per-line identity. This script therefore:

  1. drops later re-occurrences of an id (re-rolls: the same input row
     regenerated again and appended; keep-first matches the pipeline's
     existing dedup semantics),
  2. draws a seeded uniform sample of --num rows from the distinct-id pool,
  3. stamps --source into each row so prepare's (source, id) fallback
     identity applies downstream.

File order is preserved (prepare shuffles the final corpus anyway), so the
output is byte-reproducible for a given (input, num, seed, source) — safe to
re-run independently on every node.
"""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--source",
        default=None,
        help="Stamped into rows that LACK a `source` (dedup identity for "
        "sidecar-less node files); rows with an existing source keep it.",
    )
    args = ap.parse_args()

    rows, seen = [], set()
    dup_id = bad_json = 0
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            row_id = row.get("id")
            if row_id is not None:
                if row_id in seen:
                    dup_id += 1
                    continue
                seen.add(row_id)
            rows.append(row)

    take = min(args.num, len(rows))
    picked = sorted(random.Random(args.seed).sample(range(len(rows)), take))
    with open(args.output, "w", encoding="utf-8") as handle:
        for i in picked:
            row = rows[i]
            if args.source and not row.get("source"):
                row["source"] = args.source
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "rows_read": len(rows) + dup_id + bad_json,
                "dup_id_dropped": dup_id,
                "bad_json": bad_json,
                "distinct_pool": len(rows),
                "sampled": take,
                "seed": args.seed,
                "output": args.output,
            }
        )
    )


if __name__ == "__main__":
    main()
