# Hand-off: DFlash draft for `moonshotai/Kimi-K2.7-Code` (2 nodes, 16× H200)

Train the DFlash drafter for **Kimi-K2.7-Code** online, with the **SGLang** target
backend, across two nodes — launched over **ssh or slurm**. Everything is driven
by one script: `**examples/run_kimi_k2.7_code_dflash.sh`**.

Recipe mirrors `z-lab/Kimi-K2.5-DFlash` (Kimi-K2.7-Code shares K2.5's text dims, so
the draft config transfers verbatim); data is Nemotron-Post-Training v2, same as
`run_qwen3.6_27b_dflash_nemotron_v2.sh`.

## TL;DR

You ssh **from your local machine into each node separately** (there is no
node→node ssh). The repo lives on shared storage (`/personal/SpecForge`, visible to
both nodes). Pick rank 0 = the rendezvous master. Run inside `tmux` so the job
survives an ssh disconnect.

```bash
RANK0=gpu1-10-220-51-47     # rendezvous master
RANK1=gpu1-10-220-51-42

# --- ssh into RANK0, in its session: ------------------------------------
cd /personal/SpecForge
bash examples/run_kimi_k2.7_code_dflash.sh setup                       # deps (node-local)
HF_TOKEN=hf_xxx bash examples/run_kimi_k2.7_code_dflash.sh prepare      # model+data, ONCE (shared FS)
tmux new -s kimi "NODE_RANK=0 MASTER_ADDR=$RANK0 bash examples/run_kimi_k2.7_code_dflash.sh watchdog"

# --- ssh into RANK1, in a separate session: -----------------------------
cd /personal/SpecForge
bash examples/run_kimi_k2.7_code_dflash.sh setup                       # deps (node-local)
tmux new -s kimi "NODE_RANK=1 MASTER_ADDR=$RANK0 bash examples/run_kimi_k2.7_code_dflash.sh watchdog"
```

(slurm alternative: `PARTITION=… TIME=… bash examples/run_kimi_k2.7_code_dflash.sh slurm` from a login node — see Step 2.)

## What's in the bundle


| File                                           | Role                                                                              |
| ---------------------------------------------- | --------------------------------------------------------------------------------- |
| `examples/run_kimi_k2.7_code_dflash.sh`        | **The** entrypoint: `setup` / `prepare` / `train` / `watchdog` / `ssh` / `slurm`. |
| `configs/kimi-k2.7-code-dflash.json`           | Draft model config (copied from `z-lab/Kimi-K2.5-DFlash`).                        |
| `scripts/train_dflash.py`                      | Training script (edited: DP-attention-aware dataloader sharding).                 |
| `specforge/args.py`                            | Edited: adds `--sglang-dp-size`.                                                  |
| `scripts/prepare_nemotron_post_training_v2.py` | Nemotron→JSONL converter (called by `prepare`).                                   |


## Topology & why

Kimi-K2.7-Code is **595 GB** (INT4-quantized MoE; attention/embeddings/lm_head
stay bf16). We use the DeepSeek/Kimi large-MoE serving layout:

- `**tp = ep = dp = 16`** (the whole world) → ONE target instance across both nodes.
- **DP-attention**: each rank runs its own sequences' attention (own KV, no
attention all-reduce). `--sglang-dp-size 16` makes this real (`attn_tp = tp/dp = 1`).
- **EP-16 via `deepep`**: experts are expert-parallel across all 16 GPUs with an
all-to-all dispatch over InfiniBand. Per-GPU expert weights drop to ~37 GB.
- **16-way data sharding**: `train_dflash.py` detects `--sglang-enable-dp-attention`
and shards the dataloader over the full world, so the 16 ranks consume DISTINCT
samples and FSDP grad-averaging gives a true 16-way data-parallel gradient.

The 16-rank torch.distributed world (from torchrun's rendezvous) is reused by
SGLang's parallel groups (see `specforge/.../sglang_backend/patch.py`), so this
needs **no SGLang launch-server / no extra rendezvous** — just torchrun across the
two nodes.

> **Effective global batch = WORLD × BATCH_SIZE = 16 × 4 = 64 sequences/step**,
> ~32× the old tp=8/dp=2/batch=1 setup. **Scale the learning rate** (`LEARNING_RATE`
> env; default `6e-4` is the small-batch starting point — expect to raise it).

## Prerequisites

- **Two nodes** you can ssh into **from your local machine** (one session each;
  node→node ssh is *not* required) **or** a **slurm** allocation. Decide which node
  is rank 0 — it is the rendezvous master, and the other node connects to it at
  `MASTER_ADDR:MASTER_PORT` (29500), so that port must be reachable node→node.
- **HF token** with access to `moonshotai/Kimi-K2.7-Code`, exported as `HF_TOKEN`
for the `prepare` step. Never commit it.
- **Shared filesystem** for the repo, `cache/`, `outputs/`, and the HF model cache
(`HF_HOME`, default `/cluster-storage/models`), visible to both nodes — so
`prepare` runs once and checkpoints written by rank 0 are read on resume by either
node.
- Image with torch/sglang/flashinfer/flash-attn/deep_ep (e.g. `lmsysorg/sglang`).

## Step 0 — Environment setup (run on EACH node, once)

ssh into each node and run, in its session:

```bash
cd /personal/SpecForge
bash examples/run_kimi_k2.7_code_dflash.sh setup
```

The image pins are *older* than what ships, so `setup` installs SpecForge with
`pip install --no-deps -e .` plus the missing light deps (`accelerate`,
`tensorboard`, `yunchang`, `qwen-vl-utils`) — it will **not** downgrade
torch/sglang. It ends with `PREFLIGHT OK`. The launcher also refuses to start if
deps are missing. site-packages is node-local, hence "on both nodes".

## Step 1 — Prepare model + data (run ONCE, shared FS)

```bash
export HF_TOKEN=hf_xxx
export HF_HOME=/cluster-storage/models
bash examples/run_kimi_k2.7_code_dflash.sh prepare
```

Idempotent. Downloads the 595 GB model (skips complete files), builds
`nemotron_v2_{train,eval}.jsonl` + a deterministic `nemotron_v2_eval_2k.jsonl`, and
warms the tokenized train cache once (so the 16 ranks don't race tokenizing on the
shared FS). Skip the giant download with `SKIP_MODEL_DOWNLOAD=1`.

## Step 2 — Launch

**Per-node (recommended — matches "ssh from my machine into each node").** In each
node's ssh session, inside `tmux` so it survives disconnect:

```bash
# on rank 0 (gpu1-10-220-51-47):
cd /personal/SpecForge
tmux new -s kimi "NODE_RANK=0 MASTER_ADDR=gpu1-10-220-51-47 bash examples/run_kimi_k2.7_code_dflash.sh watchdog"

# on rank 1 (gpu1-10-220-51-42):
cd /personal/SpecForge
tmux new -s kimi "NODE_RANK=1 MASTER_ADDR=gpu1-10-220-51-47 bash examples/run_kimi_k2.7_code_dflash.sh watchdog"
```

`watchdog` runs `train` and auto-resumes it on crash; use `train` instead of
`watchdog` for a one-shot foreground run (e.g. a smoke test). Order doesn't matter —
c10d rendezvous waits for both. Reattach later with `tmux attach -t kimi`.

**slurm** (from a login node — submits both nodes for you):

```bash
PARTITION=<your-partition> TIME=24:00:00 \
  bash examples/run_kimi_k2.7_code_dflash.sh slurm
```

Submits `--nodes=2 --ntasks-per-node=1 --gpus-per-node=8`; each node task re-enters
the script as `train`, deriving `NODE_RANK`/`MASTER_ADDR` from `SLURM_*`. Add
`#SBATCH --requeue` semantics via your cluster for node-fault resubmission.

**ssh fan-out** (optional — only if you launch from a host with passwordless ssh to
*both* nodes): `bash examples/run_kimi_k2.7_code_dflash.sh ssh gpu1-10-220-51-47 gpu1-10-220-51-42`.
It ssh-launches the `watchdog` on each node (detached). Set `REMOTE_ROOT` if the repo
path differs from where you run the script. Not needed for the per-node flow above.

## Monitoring

```bash
ssh $RANK0_HOST tail -f outputs/kimi-k2.7-code-dflash-nemotron/train.rank0.log
```

TensorBoard (default `REPORT_TO=tensorboard`): point it at
`outputs/kimi-k2.7-code-dflash-nemotron/`. For W&B: `REPORT_TO=wandb` + export
`WANDB_API_KEY` on both nodes.

## Checkpointing & resume

- Rank 0 writes `outputs/kimi-k2.7-code-dflash-nemotron/epoch_<E>_step_<S>/` every
`--save-interval` (10000) steps, plus a final `epoch_6_step_*`.
- The launcher always passes `--resume`: on (re)start it loads the latest
checkpoint (model + optimizer + scheduler + step). The `watchdog` relies on this
to auto-resume after a crash; checkpoints live on shared storage so either node
can read them.

## Tuning knobs (env vars on the launch / both nodes)


| Goal / symptom              | Change                                                           |
| --------------------------- | ---------------------------------------------------------------- |
| **LR for the 64-seq batch** | `LEARNING_RATE=...` (start by scaling up from `6e-4`).           |
| CUDA OOM                    | Lower `MEM_FRACTION` (default `0.8`) → `0.7`; or `BATCH_SIZE=2`. |
| More throughput             | Raise `BATCH_SIZE` (EP-16 frees memory) while watching KV pool.  |
| NCCL/EP debugging           | Uncomment `NCCL_DEBUG=INFO` / `NCCL_IB_HCA=...` in `cmd_train`.  |
| Thinking-style data         | `CHAT_TEMPLATE=kimi-k2.5-thinking` (same loss-mask headers).     |


## First-run smoke test (do this before the full job)

1. On both nodes with `NCCL_DEBUG=INFO`, launch and confirm you reach the first
  `Training - Step 50` log line.
2. Check the logs show: SGLang loads the target with `tp_size=16` and
  `enable_dp_attention`; `deepep` initializes for EP; the line
   `Data-parallel sharding over WORLD (dp-attention)`; the draft FSDP initializes;
   step loss/acc are logged and decreasing.

## Known risks to validate on the first run

- **SGLang `kimi_k25` aux-hidden-state capture** of layers `[1,12,24,35,47,58]`
(`set_eagle3_layers_to_capture`). The `z-lab/Kimi-K2.5-DFlash` card pins a
specific SGLang PR — if capture fails, align the SGLang version with that card.
- **Cross-node `deepep`** uses RDMA/NVSHMEM; if EP init hangs/fails, your fabric may
need NVSHMEM/IBGDA env (e.g. `NVSHMEM_IB_ENABLE_IBGDA=1`) or `NCCL_IB_HCA` set.
- **Rendezvous port** `MASTER_PORT` (29500) must be reachable node→node; change it
if filtered.
- **LR**: the effective batch is 32× larger than the original recipe — retune.

