"""Stage-2 v2 corpus: the full stage-1 v2 corpus plus agentic x8 (holdout
excluded), tools column on every row, seed-42 shuffle."""
import hashlib, json, os, random

SRC = "/scratch/yi/nda_data_v2"
OUT = "/scratch/yi/nda_data_v2_stage2"
AG = "/scratch/yi/nda_data/agentic-normalized-20260728"
UPSAMPLE = 8

os.makedirs(OUT, exist_ok=True)
hold = {hashlib.md5(json.dumps(json.loads(l)["conversations"]).encode()).hexdigest()
        for l in open(AG + "/agentic_holdout.jsonl", encoding="utf-8")}
agentic = []
for l in open(AG + "/agentic_normalized.jsonl", encoding="utf-8"):
    row = json.loads(l)
    if hashlib.md5(json.dumps(row["conversations"]).encode()).hexdigest() in hold:
        continue
    agentic.append(json.dumps({"conversations": row["conversations"], "tools": row["tools"]},
                              ensure_ascii=False))

def with_tools(path):
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        r["tools"] = []
        yield json.dumps(r, ensure_ascii=False)

train = list(with_tools(SRC + "/nda_inkling_dspark_train.jsonl")) + agentic * UPSAMPLE
random.Random(42).shuffle(train)
with open(OUT + "/nda_inkling_dspark_train.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(train) + "\n")
with open(OUT + "/nda_inkling_dspark_eval.jsonl", "w", encoding="utf-8") as f:
    f.write("\n".join(with_tools(SRC + "/nda_inkling_dspark_eval.jsonl")) + "\n")
print("agentic", len(agentic), "x", UPSAMPLE)
for n in ("nda_inkling_dspark_train.jsonl", "nda_inkling_dspark_eval.jsonl"):
    p = os.path.join(OUT, n)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(n, "rows", sum(1 for _ in open(p)), "sha", h[:16])
