# PR report: portable distributed freeze-forensics bundle

**Suggested PR title:** `feat(scripts): capture distributed GPU and NCCL freeze evidence`

**Source material:** `5e19d77`, with the final diagnostic additions from
`72614f6`

**Migration status:** `5e19d77` cherry-picks cleanly, but its paths and process
matcher target the removed standalone trainer. Port the final behavior as a
general operational tool.

## PR description

### Background

Distributed GPU stalls often leave the training log unchanged while GPUs remain
busy. Killing and restarting the job destroys the most useful evidence: Python
stacks, the active CUDA kernel, NCCL flight-recorder entries, GPU state, and
kernel/Xid messages at the time of the freeze.

The source script captured Python stacks with `py-spy`, GPU and host memory,
NCCL traces, and later added an optional `cuda-gdb` attachment, trace parsing,
and Xid/NVLink checks. It assumed one output directory and one
`scripts/train_dspark` process name, and its later commit also modified a private
watchdog launcher.

### Motivation

A standalone, best-effort evidence collector is useful for every SpecForge
strategy and topology. It shortens attribution from “the world hung” to one of:

- a rank never entered a collective;
- a collective was launched but did not complete;
- a target-engine CUDA kernel was still running;
- a process was blocked in Python or the kernel;
- the host reported an Xid, NVLink, or memory failure.

The tool must be safe to run while a job is wedged, must not require the private
launcher, and must avoid collecting secrets from the environment or command line.

### Design and implementation

Add a parameterized `scripts/freeze_forensics.sh` that creates one timestamped
bundle per host. Required/optional inputs are supplied through flags or
environment variables:

- output root;
- process pattern or explicit PID list;
- training-log glob;
- maximum debugger attachment time;
- whether intrusive CUDA debugger attachment is enabled;
- additional trusted NCCL trace locations.

The default path is non-intrusive: enumerate matching PIDs, capture command and
thread/stack summaries, query GPU/host state, copy flight-recorder files, and
inspect recent kernel messages. `cuda-gdb` is opt-in because attaching can pause
or perturb a process. Every probe is best-effort and records its own failure
without aborting the rest of the bundle.

Write a small manifest containing UTC time, hostname, SpecForge commit, selected
PIDs, tool versions, and probe results. Only whitelist relevant NCCL/Torch/CUDA
environment variable names; never dump the full environment. The script captures
evidence but never kills, restarts, signals, or mutates the training job.

## Implementation plan (code walkthrough)

### 1. Generalize process and path discovery

Add `scripts/freeze_forensics.sh` with `set -u` and explicit best-effort handling
rather than `set -e`. Parse flags such as:

```text
--output-dir PATH
--process-pattern REGEX
--pid PID                 # repeatable
--log-glob GLOB
--cuda-gdb
--attach-timeout SECONDS
--trace-dir PATH          # repeatable
```

Default the process pattern to current SpecForge training/example entry points,
including the disaggregated DSpark consumer, but print the selected commands and
require at least one match unless `--allow-empty` is set. Exclude the collector's
own PID and validate every PID is numeric before reading `/proc`.

Use a restrictive directory mode and names of the form
`forensics_<UTC>_<hostname>`. Write to a temporary directory and rename it when
collection completes so operators can distinguish complete and interrupted runs.

### 2. Capture process evidence

For each PID, record a redacted command summary, process status, open thread
count, and one of:

- `py-spy dump --pid` when installed and permitted;
- `/proc/<pid>/stack` plus `/proc/<pid>/task/*/stack` as a fallback;
- a clear error file when ptrace restrictions block both.

Do not copy `/proc/<pid>/environ`. If command arguments may contain access tokens,
redact common token/authorization patterns before writing the manifest.

### 3. Capture GPU and host state

Collect bounded outputs from:

- `nvidia-smi` utilization, memory, temperature, power, ECC, and process tables;
- `nvidia-smi -q` with a timeout;
- `free`, `uptime`, load average, and disk availability for the output path;
- recent `dmesg` lines matching Xid, NVLink, GPU reset, OOM, and remapping errors,
  when permissions allow.

Every external command must have a timeout. Missing tools should produce a
one-line status in the manifest rather than terminate the script.

### 4. Add optional CUDA-kernel inspection

When `--cuda-gdb` is explicitly set and `cuda-gdb` exists, attach to at most one
selected worker for a bounded interval and run `info cuda kernels`. Record that
the probe is intrusive in stdout and the manifest. A timeout or attach failure
must leave the remaining collection intact.

Never enable this probe automatically from a generic watchdog.

### 5. Collect and summarize NCCL flight-recorder traces

Search both current Torch trace locations and user-supplied directories. Copy
files rather than moving them. Add `scripts/parse_nccl_trace.py` if parsing needs
more than shell:

- accept only explicitly named local files;
- catch format/version errors per file;
- report rank, process-group/collective sequence, profiling name, and state for a
  bounded tail;
- never execute or import data from the trace.

If the installed Torch trace format requires pickle, document that parsing is
only safe for locally generated trusted files and provide `--no-parse` to copy
without deserialization.

### 6. Produce operator-friendly output

Print a short stdout summary suitable for pasting into an incident:

- bundle path and host;
- selected PIDs and top Python/kernel frame;
- GPU utilization/memory summary;
- whether a CUDA kernel was active;
- whether NCCL traces were found and their last collective states;
- any Xid/NVLink findings.

Keep full outputs in separate files. Do not archive automatically; bundles can be
large and may need privacy review before upload.

### 7. Add tests and lint gates

Create `tests/test_scripts/test_freeze_forensics.py` with a temporary fake `PATH`
containing deterministic `nvidia-smi`, `py-spy`, and `cuda-gdb` executables. Add
test overrides for `/proc` and trace roots where needed. Cover:

- no matching process and `--allow-empty` behavior;
- PID selection and self-exclusion;
- missing optional tools;
- timeout handling;
- CUDA debugger opt-in;
- trace copy and parser failure isolation;
- environment/command redaction;
- complete-directory rename and manifest contents;
- no signal/kill commands are invoked.

Run `bash -n` and, when available in CI, `shellcheck`. Keep the script compatible
with the Bash version declared by SpecForge's supported images.

### 8. Document integration without coupling

Add a short operations section explaining how to invoke the collector manually on
every node while the job is frozen. A future watchdog may call it before recovery,
but that integration must pass explicit paths/PIDs and wait for bounded completion.
The collector itself remains independent of checkpointing and restart policy.

## Scope and non-goals

- The tool does not detect stalls; it captures evidence after an operator or
  watchdog detects one.
- It does not kill processes, free GPUs, or restart training.
- It does not upload bundles or require network credentials.
- It does not promise stable parsing for every historical Torch trace format;
  raw traces are preserved when parsing is unavailable.

## Acceptance criteria

- One command produces a complete, timestamped per-host diagnostic bundle for
  current SpecForge launchers.
- Missing permissions or optional tools degrade individual probes without losing
  the rest of the evidence.
- No credential-bearing environment dump or automatic process mutation occurs.
- Unit tests cover selection, redaction, timeouts, and non-intrusive defaults.
