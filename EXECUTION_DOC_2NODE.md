# DeepSeek-V4-Flash DSpark — Two-Node Training: Execution Doc (PLAN — pending review)

**Goal:** Scale the faithful DSpark drafter training from one GB300 node (branch
`dspark-trainer`, see `EXECUTION_DOC.md`) to **two GB300 nodes**, following the
two-node DFlash practice on `origin/reproduction`
(`examples/run_kimi_k2.7_code_dflash.sh`). Same recipe, same architecture, same
precision — only the topology (and therefore throughput) changes.

Owner: yi.sun@radixark.ai. Planned 2026-07-07. **STATUS: EXECUTING — rank-0
cold-start running (teacher download + dataset build in background); rank-1 is
owner-driven (see §11). WORLD=8, Approach 1, warmup 0.008.**

The two nodes (cluster `gb300-coreweave`, instance `dspark-yisun`):

| rank | hostname (rendezvous IP) | role |
|---|---|---|
| 0 | `10.41.203.7` | master (rdzv endpoint, checkpoint writer) |
| 1 | `10.41.203.9` | worker |

---

## 0. What changes vs the single-node run (`EXECUTION_DOC.md`)

The single-node run is **tp=4 / dp=1**: one sglang target engine spans the 4
GPUs, the draft is FSDP-FULL_SHARD over the 4 ranks, and **data is replicated**
across ranks (effective global batch = `BATCH_SIZE × ACC`, no dp multiplier).
16 GPUs of pure-TP would give *no* throughput gain (all ranks recompute the same
sample) and add cross-node TP all-reduce — pointless.

The DFlash two-node practice instead uses **genuine data-parallelism via sglang
DP-attention**: every rank consumes a *unique* data shard, so effective global
batch = `WORLD × BATCH_SIZE` and throughput scales ~with the GPU count. This doc
ports that mechanism onto the DSpark trainer.

**Everything faithful stays fixed** (recipe from `EXECUTION_DOC.md §2`): bf16
training → fp4+fp8 deploy pack, 10 epochs, lr 6e-4, warmup 0.04, **global batch
512**, loss `0.1·CE + 0.9·L1 + 1.0·conf`, `loss_decay_gamma 4.0`, block_size 5,
markov_rank 256, num_anchors 512, 3 mtp layers, chat-template deepseek-v3,
teacher = `sgl-project/DeepSeek-V4-Flash-FP8`. Global batch 512 is preserved by
lowering `ACC` (see §7); the *recipe* is identical, the run is just faster and —
because each step now sees 512 **unique** samples instead of `BATCH_SIZE` unique
samples replicated `WORLD/…`× — marginally *more* faithful, not less.

---

## 1. Environment — verified on rank 0 (`10.41.203.7`), + open item

| Item | Value (rank 0, measured this session) |
|---|---|
| GPUs visible in container | **4× GB300, 284 GB each** (`nvidia-smi -L`; MIG disabled; `CUDA_VISIBLE_DEVICES` empty) |
| Cluster listing claims | **8× gb300 per node** — **DISCREPANCY, see Decision D1** |
| rank-1 reachability | `10.41.203.9:22` **open** (TCP) |
| `/scratch` | **node-local** `/dev/md127` xfs RAID, 27 TB free — **NOT shared** between nodes (`shared_s3=false`) |
| Interconnect | 8× `mlx5_*` IB HCAs + `ibs*p*` netdevs → GPUDirect RDMA available |
| Ethernet (rdzv) | `10.41.203.7` on an `enP…`/`enP5p9s0` iface; `10.88.8.91` = k8s pod net |
| `deep_ep` | **installed** ✅ (`/usr/local/lib/python3.12/.../deep_ep`) |
| sglang 0.5.14 | supports `enable_dp_attention` / `dp_size` / `ep_size` / `moe_a2a_backend`; **NEW** dp_attn API (`scheduler_components.dp_attn.prepare_mlp_sync_batch_raw`, `get_attention_tp_size`) — the OLD `Scheduler.prepare_mlp_sync_batch_raw` is **gone** |
| `yunchang` | **missing** ❌ (trainer import) — reinstall on both nodes |
| repo / model / data | **env freshly reset** — `/scratch` holds only the 21 MB git repo. The 274 GB FP8 teacher, dataset, tokenized cache, and all single-node checkpoints are **GONE**; `/scratch/hf_cache` no longer exists. No single-node run is live (only earlyoom). ⇒ prepare = full cold start on **both** nodes |
| `/scratch` persistence | backed by a persistent `/data-volume/<uuid>` subvolume (the git repo survived this reset) → provisioning to 8/node is **data-safe** (nothing valuable to lose); confirm with the cluster regardless |

Two hard consequences drive the whole plan:
1. **`/scratch` is node-local** → the repo, the ~274 GB FP8 teacher, the dataset
   jsonl, and the tokenized cache must exist **independently on both nodes**; and
   checkpoints (written only by global rank 0) must be **synced to rank 1** for a
   clean resume. The kimi script assumed a shared FS — that assumption is false here.
2. **GPU count is unconfirmed** (4 visible vs 8 claimed) → see Decision **D1**. The
   plan is written parametric in `NUM_GPUS`; `WORLD = NNODES × NUM_GPUS`.

---

## 2. Topology decision — Approach 1 (genuine DP), with Approach 2 as fallback

### Approach 1 — DP-attention genuine data-parallel (**recommended; mirrors the practice**)
One logical sglang engine spanning **all** ranks, but with **DP-attention** so
attention is per-rank (`attn_tp=1`) and the MoE is expert-parallel over `deepep`:

```
--tp-size WORLD  --sglang-dp-size WORLD  --sglang-ep-size WORLD \
--sglang-enable-dp-attention  --sglang-moe-a2a-backend deepep
```

- **Data:** sharded over the **whole world** (each rank a unique `1/WORLD` shard).
- **Target:** each rank forwards only its own shard, returns its own aux + final
  hidden states; attention has **no cross-node all-reduce** (attn_tp=1) — only the
  MoE `deepep` all-to-all crosses nodes (over IB). This is exactly why the practice
  uses DP-attention (it minimizes cross-node target traffic).
- **Draft:** FSDP **FULL_SHARD** over the default 16-rank world; gradients
  mean-reduced by FSDP's reduce-scatter.
- **Effective global batch = `WORLD × BATCH_SIZE`** → ~`WORLD/4`× the single-node
  throughput. **Primary risk:** does DP-attention + `deepep` compose with the
  DSV4 FP8 target + the auto-forced `dsv4` attention backend in sglang 0.5.14?
  → validated by the §6 smoke test before any long run.

### Approach 2 — per-node TP engines, cross-node DP (**fallback, low target risk**)
`--tp-size = NUM_GPUS` (per node), `dp = NNODES`, **no** DP-attention, **no**
`deepep`. The device mesh `(dp=NNODES, tp=NUM_GPUS)` gives one **intra-node**
tp engine per node (identical to the *proven* single-node target config, just
tp=NUM_GPUS), and data is sharded across the `dp` group (`get_dp_group()`, the
existing default path — no dataloader change).
- **Pro:** reuses the byte-for-byte proven single-node target serving path; only
  cross-node traffic is the draft FSDP collectives + a scalar loss all-reduce.
- **Con:** only **`NNODES`-way** unique-data parallelism (each tp engine's ranks
  see identical data) → effective global batch = `NNODES × BATCH_SIZE` → ~2×
  single-node (for 2 nodes), vs ~`WORLD/4`× for Approach 1.

**Recommendation:** implement **Approach 1** (it is the practice the task points
to and yields ~4× more throughput than Approach 2), and keep Approach 2 as a
one-flag fallback if the smoke test shows DP-attention+deepep don't compose with
the FP8 DSV4 target. Both are loss-correct (see §3); both need the same infra (§5).

---

## 3. Why the existing DSpark objective is already DP-correct (no loss change)

`OnlineDSparkModel._dspark_objective` (`specforge/core/dspark.py:340-356`) uses
DeepSpec's **pooled global-mean** reduction:

```python
world_size = dist.get_world_size()                      # default group = all ranks
global_den = local_den.detach().clone()
dist.all_reduce(global_den, op=SUM)                     # Σ over ALL ranks
loss = (α_ce·ce_num + α_l1·l1_num + α_cf·conf_num) / (global_den + eps) * world_size
```

Each rank contributes its **local numerators** over a **cross-rank-summed
denominator**, times `world_size`. FSDP's reduce-scatter then mean-divides the
gradient by `world_size`, so the `× world_size` cancels it and the effective
gradient is the **true token-pooled global mean** `Σ_r num_r / Σ_r den_r`. This
is correct **iff** (a) each rank holds unique data, (b) FSDP shards over the whole
world, (c) `dist.get_world_size()` == the FSDP shard group. Under **Approach 1**
all three hold (data sharded over world; single default-world FULL_SHARD group).
Under **Approach 2** the `T=NUM_GPUS`-fold replication inside each tp engine
cancels in numerator and denominator, so it reduces to the same global mean over
the `NNODES` unique streams. **⇒ the objective is unchanged for both approaches.**

(This differs from DFlash's loss, which is a per-rank *local* mean relying on FSDP
to average — also correct, just equal-weighted per rank instead of token-weighted.
DSpark keeps DeepSpec's token-weighted form, which is the faithful DSpark objective.)

Caveat carried over from single-node: `BF16Optimizer` grad-norm clipping is
per-shard-local, not a true global norm (`specforge/optimizer.py`). Pre-existing,
identical in single-node, not a blocker.

---

## 4. Code port list (small, surgical — all onto `dspark-trainer`)

The `_WORLD`-rebind sglang patch is **already present** on this branch
(`specforge/modeling/target/sglang_backend/patch.py:53-78`), so the target already
knows how to reuse the torchrun world. Three focused changes remain:

**C1 — `specforge/args.py`: add the two missing sglang knobs.**
`dspark-trainer` already has `sglang_enable_dp_attention` and `sglang_ep_size`
but **not** `sglang_dp_size` or `sglang_moe_a2a_backend`. Add both (dataclass
field + `add_args` + `from_args` + `to_kwargs`), mirroring `origin/reproduction`'s
`args.py` (`--sglang-dp-size` int default 1; `--sglang-moe-a2a-backend` str
default `"none"`; optionally `--sglang-moe-runner-backend`).

**C2 — `specforge/modeling/target/dflash_target_model.py`: dp_attn API compat.**
The dp-attention sync path currently calls the **removed** old API
(`Scheduler.prepare_mlp_sync_batch_raw(..., attn_tp_size=1, spec_algorithm=…)`),
dormant at dp=1 but **crashes the instant DP-attention is on**. Port the exact
fix from `origin/reproduction`:
```python
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.managers.scheduler_components.dp_attn import prepare_mlp_sync_batch_raw
...
prepare_mlp_sync_batch_raw(
    batch, dp_size=self.model_runner.server_args.dp_size,
    attn_tp_size=get_attention_tp_size(),
    attn_cp_size=getattr(self.model_runner, "attn_cp_size", 1),
    tp_group=self.model_runner.tp_group, get_idle_batch=None,
    disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
    require_mlp_tp_gather=require_mlp_tp_gather(self.model_runner.server_args),
    ...)
```
**Preserve every DSpark-specific target feature** already in this file that
`reproduction` lacks: `last_hidden_states` surfacing (L1/confidence), serial
weight-load host-RAM bounding (`SPECFORGE_SGLANG_SERIAL_LOAD`), ServerArgs kwarg
filtering, `initialize_moe_config`, and the newer-ModelRunner
`alloc_memory_pool/init_attention_backends/init_cuda_graphs` handling. This is a
**surgical edit of the sync call**, *not* a file replacement.

**C3 — `scripts/train_dspark_v4.py` `build_dataloader`: shard over world under DP-attn.**
Currently hardcodes `process_group=get_dp_group()`. Mirror reproduction:
```python
use_dp_attention = (args.target_model_backend == "sglang"
                    and args.sglang_enable_dp_attention)
data_parallel_group = None if use_dp_attention else get_dp_group()   # None => whole world
```
and pass it to both `prepare_dp_dataloaders(...)` calls. Nothing else in the
trainer changes: FSDP already wraps with the default world PG (correct), the
objective is already DP-correct (§3), `set_epoch` + `skip_steps` already replay a
deterministic per-rank shuffle for resume.

**No change needed:** `distributed.py` (`init_distributed(tp_size=WORLD)` gives the
right mesh), the objective, FSDP wrap, checkpoint *save*.

---

## 5. Infra / setup (node-local FS is the theme)

**S1 — deps on BOTH nodes** (env resets wipe pip; see `EXECUTION_DOC §7b`):
`pip install -e . && pip install accelerate yunchang wandb` (deep_ep already
present). Preflight import check: `torch, sglang, transformers, datasets,
accelerate, yunchang, deep_ep, flash_attn, specforge`.

**S2 — cold-start the data plane on BOTH nodes** (the reset wiped everything but
the git repo; nothing to replicate from — this is the largest upfront cost):
- **repo** → already on rank 0 (git, survives resets); `git`/`rsync` the working
  tree incl. the C1–C3 edits to `10.41.203.9:/scratch/SpecForge`.
- **teacher** `sgl-project/DeepSeek-V4-Flash-FP8` (~274 GB) → `snapshot_download`
  on rank 0 (HF_HOME=`/scratch/hf_cache`), then **rsync rank0→rank1 over the
  10.41.203.x net** (one download, far faster than two); ~hours either way. HF_TOKEN
  needed. (Both `embed.weight`/`head.weight` are bf16.)
- **dataset** → `prepare_data.py --dataset perfectblend` (rank 0) → 1.42 M →
  train split + 3 k eval (`EXECUTION_DOC §4f`); rsync the jsonl to rank 1. Then
  **warm the tokenized `cache/processed_dataset/<md5>`** on rank 0 and rsync it so
  rank 1 doesn't re-tokenize 1.35 M samples (md5 key is identical for identical
  inputs). This is the reset's biggest tax; budget it explicitly (D4).

**S3 — networking** (pin before launch): `MASTER_ADDR=10.41.203.7`,
`MASTER_PORT=29500`; `NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` = **`enP22p3s0np0`**
(resolved on rank 0 = the iface holding 10.41.203.7, via psutil since `ip` hangs
in-container). The run script auto-detects this per-node (matches each node's own
10.41.203.x iface), default `enP22p3s0np0`. IB HCAs auto-detected for the
NCCL/deepep data plane. **Do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** on
the deepep path (the practice warns it breaks sglang pynccl cuMem/NVLS →
`ncclCommInitRank invalid usage`); this *reverses* the single-node setting, so
watch the objective's transient-alloc behaviour in the smoke test.

**S4 — checkpoint sync for resume** (node-local FS + rank-0-only writer):
`save_checkpoint` gathers FULL_STATE_DICT and only global rank 0 writes → the
checkpoint lands on **rank 0's** `/scratch` only, but **resume loads
`model.safetensors` on every rank** (`train_dspark_v4.py:392`). So before any
resume, rank 1 needs the checkpoint dir. Plan: a detached **rank0→rank1 rsync
loop** mirroring `OUTPUT_DIR/epoch_*_step_*` (skipping dirs <3 min old to avoid
mid-write), reusing the single-node `ckpt_rotate.sh` idea. (Alternative, cleaner
but more code: change resume to load on rank 0 and broadcast — deferred; rsync is
lower-risk for the first run.)

---

## 6. Smoke test (mandatory gate before the long run)

Run on both nodes with `--max-steps 4` on a tiny data slice, Approach-1 flags,
`NUM_GPUS` per Decision D1. Validates, in order:
1. **Rendezvous** across nodes over Ethernet + NCCL init over IB (16-rank world up).
2. **Target load** under DP-attention + deepep + `dsv4` backend + FP8 teacher —
   *the primary Approach-1 risk*. If it faults in the fused MoE / dp_attn sync,
   fall back to **Approach 2** (per-node tp engines) and re-smoke.
3. **Per-rank forward** returns aux hidden states **and** `last_hidden_states`
   (else L1/confidence can't run) with per-rank-distinct data.
4. **Draft FSDP FULL_SHARD over 16 ranks** cross-node: no OOM; **measure
   micro-step time** (the cross-node all-gather is the perf unknown — see §7).
5. **Loss decreases**, `teacher_top1_prob ≈ 0.98` ceiling sane.
6. **Checkpoint save → rsync to rank 1 → resume** round-trips (LR+step restored,
   Adam reset per `train_dspark_v4.py:500-505`; expect the known one-time loss
   bump on resume, `EXECUTION_DOC §7g`).

Also keep the host-RAM guards from single-node (`SPECFORGE_SGLANG_SERIAL_LOAD=1`;
operator-pause earlyoom through the DeepGEMM-JIT startup spike, resume at steady
state) — now on **both** nodes.

---

## 7. Batch / throughput / wall-clock

- **Hold global batch 512:** `ACC = 512 / (WORLD × BATCH_SIZE)`.
  - `WORLD=16, BS=4 → ACC=8`; `WORLD=16, BS=1 → ACC=32`.
  - `WORLD=8,  BS=4 → ACC=16`; `WORLD=8,  BS=1 → ACC=64`.
- **`MEM_FRAC`:** with `tp=WORLD` the FP8 teacher shrinks to ~`274/WORLD` GB/rank
  (~17 GB at 16, ~34 GB at 8) → start `~0.5` (the practice's value) and tune; far
  more headroom than single-node's 0.4.
- **FSDP strategy: keep FULL_SHARD** (the 19.85 B MoE draft OOMs under
  SHARD_GRAD_OP — `EXECUTION_DOC §4i`; reproduction's DFlash uses SHARD_GRAD_OP
  only because its draft is tiny). Cross-node FULL_SHARD all-gather of the ~40 GB
  expert params over IB is the main perf question → **measured in the smoke test**;
  mitigated by `backward_prefetch=BACKWARD_PRE` + `forward_prefetch=True` (already
  wired) and amortized by `BATCH_SIZE`.
- **Estimate** (single-node baseline 3.45 samples/s, ~4.5 d/epoch): Approach 1 at
  `WORLD=8` (8 unique-data streams vs the single node's 4-way tp-replicated) ≈
  **~2× throughput → ~2 d/epoch → ~20 d for 10 epochs**; checkpoint every 500
  micro-steps and stop at convergence (goal probe vs the released drafter) rather
  than always running all 10. All estimates pending smoke-test calibration.

---

## 8. Run script + launch/supervision (mirror the kimi practice)

New `examples/run_dsv4_flash_dspark_2node.sh`, structured exactly like
`run_kimi_k2.7_code_dflash.sh` — subcommands `setup` / `prepare` / `train` /
`watchdog`, `NNODES`/`NUM_GPUS`/`WORLD`, `torchrun --nnodes --nproc-per-node
--node-rank --rdzv-backend c10d --rdzv-endpoint $MASTER:$PORT --rdzv-id <fixed>
--max-restarts 0`, fixed `RDZV_ID` — but carrying the **DSpark** args/env:
teacher = FP8 repack, `configs/deepseek-v4-flash-dspark.json`, block/anchors/loss
alphas/gamma/chat-template from §0, `--sglang-dp-size WORLD --sglang-ep-size WORLD
--sglang-enable-dp-attention --sglang-moe-a2a-backend deepep`,
`SGLANG_OPT_FP8_WO_A_GEMM=0`, `SPECFORGE_SGLANG_SERIAL_LOAD=1`,
`SPECFORGE_OFFLOAD_MASTER=1`, FULL_SHARD + prefetch, grouped_mm experts, flex
draft-attn, `gradient_checkpointing=false`, MoE auto→FlashInfer TRTLLM Fp8, wandb.

- **`MAX_RESTARTS=0`** (in-place restart on one 2-node worker desyncs the other →
  gloo storms; let torchrun exit cleanly, watchdog does a coordinated relaunch).
- **Launch per node in tmux**, detached (`setsid`) so no harness/session event can
  SIGTERM it (`EXECUTION_DOC §7f`); `RESUME=1` → auto-resume from latest checkpoint.
- **Per-node `watchdog`** (relaunch-on-death, crash-loop guard) + **rank0→rank1
  checkpoint rsync loop** (§S4) + **checkpoint rotation** (keep last 5, ~112 GB
  each). All detached.

---

## 9. Decisions

- **D1 — GPUs per node → ✅ RESOLVED: a GB300 node is physically 4 GPUs → `WORLD=8`.**
  `NUM_GPUS=4`, `NNODES=2`, `WORLD=8`. (The cluster "8x gb300" listing was
  misleading; 4/node is the hardware. No provisioning needed.)
- **D2 — Topology → ✅ DECIDED: Approach 1 (genuine DP-attention).** Approach 2
  kept as the automatic fallback if the §6 smoke-test gate (DP-attn+deepep on the
  FP8 DSV4 target) fails.
- **D3 — Faithfulness.** Genuine DP keeps the recipe + global batch 512 identical
  and makes each step's 512 samples *unique* → **≥ as faithful**. Assumed OK unless
  owner objects.
- **D4 — Budget/target.** 10 epochs ≈ **~6 days** at WORLD=16, **plus a larger
  upfront tax than first estimated** because the reset wiped everything: ~hours to
  re-download the 274 GB teacher + rebuild/tokenize 1.4 M samples + install deps,
  ×(rank 0 then rsync to rank 1). Assume run-to-10-or-convergence unless owner says
  otherwise.

### Concrete settings locked by D1/D2 (WORLD=8, Approach 1)
`NNODES=2 NUM_GPUS=4 WORLD=8`, `--tp-size 8 --sglang-dp-size 8 --sglang-ep-size 8
--sglang-enable-dp-attention --sglang-moe-a2a-backend deepep`. Global batch 512 ⇒
`BATCH_SIZE=1 → ACC=64` (start here; the DP-attn per-rank objective tensors scale
with BATCH_SIZE, and single-node showed BS drives the vocab-softmax memory wall —
raise BS only if the smoke test shows headroom, `ACC=512/(8·BS)`). `MEM_FRAC≈0.5`
(teacher ~34 GB/rank at tp8). FULL_SHARD + prefetch, grouped_mm, flex draft-attn,
`gradient_checkpointing=false`, `SGLANG_OPT_FP8_WO_A_GEMM=0`,
`SPECFORGE_SGLANG_SERIAL_LOAD=1`, `SPECFORGE_OFFLOAD_MASTER=1`, **no**
`expandable_segments` (deepep). Re-tune after the smoke test.

## 10. Risk register
- **R1 (high):** DP-attention + `deepep` may not compose with the FP8 DSV4 target
  (auto `dsv4` backend). → §6 gate; Approach-2 fallback.
- **R2 (med):** cross-node FSDP FULL_SHARD all-gather over IB may dominate the
  micro-step. → measure in smoke test; amortize with BATCH_SIZE / consider
  Approach 2 (draft-only cross-node traffic) if severe.
- **R3 (med):** node-local FS resume — checkpoint must reach rank 1. → §S4 rsync loop.
- **R4 (low):** `expandable_segments` off (deepep) may resurface the objective's
  transient-alloc stalls. → watch in smoke test; chunk objective only if it bites.
- **R5 (low):** env resets wipe pip on both nodes. → `setup` subcommand + preflight.

## 11. Status
- [x] Recon: infra probed, reproduction two-node mechanism mapped end-to-end,
      objective proven DP-correct, port list scoped, sglang API gap confirmed.
- [x] **D1 (WORLD=16) + D2 (Approach 1) decided by owner.** D3/D4 assumed OK.
- [x] **C1–C3 code edits landed + verified** (compile + import + flags parse; on
      `dspark-trainer`, uncommitted). C1 `specforge/args.py` (`--sglang-dp-size`,
      `--sglang-moe-a2a-backend`, `--sglang-moe-runner-backend`). C2
      `specforge/modeling/target/dflash_target_model.py` (dp_attn sync ported to
      sglang 0.5.14's `prepare_mlp_sync_batch_raw` free-fn + `attn_cp_size` +
      `get_attention_tp_size()`; verified the installed signature matches; all
      DSpark-only target features preserved). C3 `scripts/train_dspark_v4.py`
      (`build_dataloader` shards over world when DP-attention is on).
- [x] **Run script** `examples/run_dsv4_flash_dspark_2node.sh` (setup/prepare/train/
      watchdog; WORLD=8 / NUM_GPUS=4 defaults; Approach-1 flags; SMOKE=1 gate; syntax OK).
- [x] S1: deps installed on **rank 0** (accelerate/yunchang/wandb); import chain green.
- [x] Warmup ramp 5× faster (`--warmup-ratio 0.008`, env `WARMUP_RATIO`); iface
      resolved (`enP22p3s0np0`) + auto-detect wired; `prepare` now carves the
      train/eval split (deterministic seed-42 shuffle, 3,000 eval).
- [x] **S2 rank-0 cold start DONE + preflighted:** teacher 276 GB ✓; perfectblend
      built + carved (trainsplit 1,417,909 / eval 3,000) ✓; tokenized cache 64/64
      shards (key `9c3c57e…`, matches trainer) ✓; wandb `~/.netrc` ✓; full WORLD=8
      train arg-set parses clean ✓. **rank-0 is ready to launch.**
- **Transport = owner-driven, NO git push / NO node-to-node ssh.** rank-0↔rank-1
      direct ssh isn't available (the `10.41.203.9:22` sshd ≠ the `rx` devbox), and
      owner declined git-pull. ⇒ **owner gets the code to rank-1 and launches the
      rank-1 side their way** (`rx`/tmux); each node runs setup+prepare independently.
- [ ] **OWNER ACTION (raised):** on rank 1 — get this repo tree there, then
      `setup` + `prepare` (independent 274 GB download + data build). Details in msg.
- [ ] **TODO — tune `BATCH_SIZE`** (currently 1 → ACC 64). After the smoke test
      confirms per-rank memory headroom under DP-attention, try BS=2 (ACC 32) / BS=4
      (ACC 16) for fewer, fatter optimizer steps; keep global batch 512. Non-blocking.
- [ ] C4 (optional): resume loads on rank-0 + broadcasts, so checkpoints never need
      to cross nodes (else owner copies the checkpoint on a restart). Non-blocking for launch.
- [ ] §6 smoke test (`SMOKE=1`, both nodes; auto-fallback to Approach 2 if the gate fails).
- [ ] Launch + supervise; goal-probe vs the released drafter.
