# PR report: world-consistent checkpoint selection before distributed resume

**Suggested PR title:** `fix(checkpoint): negotiate one resume target across all ranks`

**Source material:** `f9f0b3a`, corrected by `4675b15`

**Migration status:** move the invariant into the current
`specforge/training/checkpoint.py`; do not transplant the legacy trainer guard.

## PR description

### Background

The source incident occurred on node-local storage. Global rank zero found a new
checkpoint and resumed at a later step, while ranks on other nodes found no copy
and started earlier. The ranks then entered different collective schedules: one
group reached gradient-norm reduction while another was still in FSDP forward
collectives, producing a deterministic deadlock.

The first source fix compared local steps and restarted every rank from scratch
on mismatch. That was unsafe: it could discard a valid common initialization and
resume a converged continual run from random weights. The follow-up selected the
largest checkpoint step available everywhere and started fresh only when no
common step existed.

Current `CheckpointManager` is much stronger than the source trainer: saves are
atomic, every rank writes its optimizer/RNG shard, save failures are shared, and
checkpoint metadata records world size. Its documented layout still assumes a
shared filesystem, and `Trainer` reads `resume_from` locally before any
world-consistency negotiation.

### Motivation

A distributed training job must never enter the first model or optimizer
collective with different resume state across ranks. Missing, stale, partially
synced, or mismatched checkpoints should produce one world-wide decision and one
actionable error—not a hang minutes later.

Step equality alone is insufficient. Two directories can share a step number but
belong to different runs or contain different model/configuration state. Resume
selection should compare a stable checkpoint identity and verify that every rank
has the per-rank state it needs.

### Design and implementation

Add a distributed resume-resolution phase before `CheckpointManager.read_resume_state`.
Each rank reports locally usable candidates as small metadata records; all ranks
gather those records and deterministically choose the same target.

Support two explicit policies:

- `exact` (default): the supplied checkpoint must be present and have the same
  identity on every rank, otherwise every rank raises;
- `latest-common`: when the supplied path is a run root, intersect the locally
  usable checkpoint identities and select the highest common step.

There should be no implicit “start fresh” fallback. A caller that wants a new run
can omit `resume_from`; a caller that requested resume should not silently lose
training state.

A locally usable checkpoint contains:

- complete shared state (`training_state.pt`);
- this global rank's `training_state_rank{rank}.pt` for non-zero training steps;
- matching saved and current world size;
- run ID, strategy, step, and a persisted checkpoint identity;
- configuration metadata already checked by `Trainer` (dataset size,
  accumulation steps) after selection.

Add a generated `checkpoint_id` to new shared checkpoint payloads. Synced copies
retain that ID, allowing cheap identity comparison without hashing a multi-GB
state dict. For legacy checkpoints, derive a conservative metadata fingerprint
from run ID, strategy, step, world size, and stable state metadata; if identity
cannot be established, `exact` should fail with migration guidance rather than
guess.

This PR prevents divergent resume. It does not magically make node-local
checkpoints available: operators must still replicate the full shared payload and
the required rank-local shards to the nodes that consume them.

## Implementation plan (code walkthrough)

### 1. Introduce resume candidate records

In `specforge/training/checkpoint.py`, add an internal frozen record such as
`ResumeCandidate` containing:

- normalized local checkpoint path;
- `checkpoint_id`;
- `run_id`, `strategy`, `global_step`, and `world_size`;
- booleans for shared state and current-rank state availability;
- a redacted validation error when unusable.

Add helpers that inspect metadata with `map_location="cpu"` and never load
optimizer tensors until selection succeeds. For a run root, scan only complete
`{run_id}-stepN` directories; do not trust a stale `latest` symlink over the
directory contents.

### 2. Persist checkpoint identity

In `TrainerController.save_checkpoint`, add a stable `checkpoint_id` to the
shared payload. Generate it once on rank zero, broadcast it before save, and
store the same value in every rank-local payload or a small sidecar manifest.
The ID must remain unchanged when the checkpoint directory is copied.

Include `run_id`, strategy, world size, global step, and checkpoint ID in a small
JSON manifest written atomically. Candidate discovery can read this manifest
without deserializing model weights. Existing `training_state.pt` remains the
source of truth during the compatibility window.

### 3. Negotiate one target

Add `CheckpointManager.resolve_distributed_resume(path, *, policy, run_id)`:

1. discover local candidates and local validation failures;
2. use one fixed `all_gather_object` schedule so every rank participates even
   when its local path is missing;
3. for `exact`, require one identical checkpoint ID on every rank;
4. for `latest-common`, intersect IDs and select by `(global_step, checkpoint_id)`;
5. broadcast or independently derive the chosen metadata;
6. return each rank's local path for that common identity.

If resolution fails, every rank raises the same aggregated exception listing
which ranks/nodes lack which step or identity. Do not include checkpoint tensor
contents or sensitive paths beyond what operators need to diagnose the copy.

### 4. Wire resolution before state load

In `specforge/training/trainer.py::Trainer.__init__`, call the resolver before
`read_resume_state`. Only after all ranks agree should the code load draft weights,
wrap with FSDP, and restore optimizer/RNG state.

Extend `specforge/config/schema.py::TrainingConfig` with:

```python
resume_policy: Literal["exact", "latest-common"] = "exact"
```

The CLI must reject `latest-common` when `resume_from` names a single checkpoint
file rather than a run root. Preserve the existing strategy, dataset-size,
accumulation, and world-size validation after target resolution.

### 5. Clarify storage contracts

Update the `CheckpointManager` module documentation:

- shared filesystem: all ranks can use one physical checkpoint directory;
- pre-synced node-local filesystem: every node must hold the shared payload and
  each rank must see its own rank-state file at the same logical step;
- unsynced node-local filesystem: resume fails before training and reports the
  missing common checkpoint.

Do not have `CheckpointManager` launch `rsync`, SSH, or cloud-copy operations.
Artifact replication belongs to orchestration and requires separate authority.

### 6. Add distributed CPU tests

Extend `tests/test_runtime/test_checkpoint_manager.py` with multi-process Gloo
tests using a distinct temporary root per rank:

- exact same ID/step resolves on every rank;
- rank zero has step 10 while rank one has only step 5: `exact` fails;
- the same setup with `latest-common` selects step 5;
- equal steps with different checkpoint IDs fail;
- a missing rank-local optimizer/RNG file makes that candidate unusable;
- a step-zero weights-only checkpoint is accepted according to the existing
  contract;
- no common candidate raises identically on all ranks and never returns “fresh”;
- world-size mismatch fails before model construction;
- shared-filesystem behavior remains unchanged.

Add a timeout to every distributed test so a collective-schedule regression
fails quickly rather than hanging CI.

### 7. Add an integration continuity gate

Extend `tests/test_runtime/test_checkpoint_resume.py` to train a tiny model,
save, present different local `latest` views, resolve the common checkpoint, and
continue. Assert identical selected step, restored optimizer/RNG state, and
loss-curve continuity on all ranks.

## Scope and non-goals

- No automatic checkpoint copying or remote object-store client is added.
- No resume across a changed world size is enabled.
- No silent fresh start is permitted after an explicit resume request.
- The PR does not revive the source trainer's `get_last_checkpoint` logic.

## Acceptance criteria

- Every rank agrees on checkpoint identity before loading model or optimizer
  state.
- Missing or inconsistent local copies fail synchronously with an actionable
  error instead of deadlocking later.
- `latest-common` selects the highest fully usable common checkpoint and never a
  merely equal step with different contents.
- Existing shared-filesystem checkpoint/resume tests remain green.
