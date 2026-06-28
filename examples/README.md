# Run SpecForge Examples

This folder contains the examples of running SpecForge on different models. The scripts can be invoked by the following command:

```bash
bash examples/<script-name>.sh [NUM_GPUS] [TP_SIZE]
```

We use the ShareGPT dataset for all the examples for now, but you can replace it with more robust datasets such as perfectblend, magpie-qwen2.5-pro-1m-v0.1, etc.

## DFlash SWE-bench Examples

- `run_qwen3_4b_dflash_swebench_rollout.sh`: Phase A helper for serving Qwen3-4B and generating agentic SWE-bench rollout JSONL.
- `run_qwen3_4b_dflash_swebench.sh`: Phase B helper for training/evaluating DFlash from the pretokenized rollout JSONL.

See `docs/examples/qwen3-4b-dflash-swebench-codex-rollout.md` for the full Codex + Daytona runbook.
