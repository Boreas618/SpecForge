#!/usr/bin/env python3
"""Build the stage-2 (YaRN adapt) corpus: all regen rows unseen by stage 1,
back-filled with seen rows to the stage-1 per-epoch budget, plus the
normalized agentic 16K-64K trajectories upsampled x4 (holdout excluded).

Deterministic (seed 42). Chat rows are gated/deduped by the standard prepare
script; agentic rows bypass it (its message cleaner strips tool_calls) — they
were already normalized + render/mask-validated (V8, 2026-07-28).
"""
import hashlib, json, os, random, subprocess, sys

RO = "/scratch/yi/regen_out"
STAGE1_INPUTS = (
    sorted(__import__("glob").glob(f"{RO}/base/data/train-*.jsonl"))
    + sorted(__import__("glob").glob(f"{RO}/takeover/data/regen.shard*.partial-*.jsonl"))
)
NEW_INPUTS = (
    sorted(__import__("glob").glob(f"{RO}/takeover_full/data/*.full-*.jsonl"))
    + sorted(__import__("glob").glob(f"{RO}/gen800k/data/*.partial-*.jsonl"))
)
AG_DIR = "/scratch/yi/nda_data/agentic-normalized-20260728"
OUT_DIR = "/scratch/yi/nda_data_stage2c"
CHAT_TOTAL = 250_000  # owner 2026-07-30: unseen + seen fill together total 250k chat rows
UPSAMPLE = 8  # owner 2026-07-30: x8 (was x4)

def src_index(line):
    return int(json.loads(line)["regen_contract"]["source_index"])

seen, prev_lines = set(), []
for p in STAGE1_INPUTS:
    for line in open(p, encoding="utf-8"):
        if line.strip():
            seen.add(src_index(line))
            prev_lines.append(line)
assert len(prev_lines) == 446_352, len(prev_lines)

new_lines, new_idx = [], set()
for p in NEW_INPUTS:
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        i = src_index(line)
        if i in seen:
            continue
        assert i not in new_idx, i
        new_idx.add(i)
        new_lines.append(line)

fill_n = CHAT_TOTAL - len(new_lines)
fill = random.Random(42).sample(prev_lines, fill_n)
os.makedirs(OUT_DIR, exist_ok=True)
raw = os.path.join(OUT_DIR, "stage2_chat_raw.jsonl")
with open(raw, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
    f.writelines(fill)
print(f"unseen={len(new_lines)} fill={fill_n} raw={len(new_lines)+fill_n}")

subprocess.run(
    [sys.executable, "scripts/prepare_nda_inkling_dspark_data.py",
     "--inputs", raw, "--output-dir", OUT_DIR, "--eval-size", "2000"],
    check=True)

hold = {hashlib.md5(json.dumps(json.loads(l)["conversations"]).encode()).hexdigest()
        for l in open(f"{AG_DIR}/agentic_holdout.jsonl", encoding="utf-8")}
agentic = []
for l in open(f"{AG_DIR}/agentic_normalized.jsonl", encoding="utf-8"):
    row = json.loads(l)
    if hashlib.md5(json.dumps(row["conversations"]).encode()).hexdigest() in hold:
        continue
    agentic.append(json.dumps(
        {"conversations": row["conversations"], "tools": row["tools"]},
        ensure_ascii=False))
print(f"agentic train rows={len(agentic)} x{UPSAMPLE}")

def with_tools(path):
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        r["tools"] = []
        yield json.dumps(r, ensure_ascii=False)

train = list(with_tools(f"{OUT_DIR}/nda_inkling_dspark_train.jsonl")) + agentic * UPSAMPLE
random.Random(42).shuffle(train)
with open(f"{OUT_DIR}/nda_inkling_dspark_train.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(train) + "\n")
evr = list(with_tools(f"{OUT_DIR}/nda_inkling_dspark_eval.jsonl"))
with open(f"{OUT_DIR}/nda_inkling_dspark_eval.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(evr) + "\n")
os.remove(raw)
for n in ("nda_inkling_dspark_train.jsonl", "nda_inkling_dspark_eval.jsonl"):
    p = os.path.join(OUT_DIR, n)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"{n}: rows={sum(1 for _ in open(p))} sha256={h}")
