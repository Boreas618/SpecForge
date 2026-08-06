# Inkling-Small DSpark v2 — reference configs and results

Program record for the Inkling-Small drafter trained with the configs in
`configs/inkling-small-dspark-v2-{stage1,yarn}.json`. Target:
`thinkingmachines/Inkling-Small-NVFP4` (42-layer hybrid-SWA MoE, NVFP4,
1M addressable context), frozen and served by a colocated per-node TP4
SGLang engine during training.

## Draft

1.39B parameters: 6 layers, all full attention (the DSPARK serving kernel
has no SWA path), hidden 4096, 32 heads / 8 KV heads, head_dim 128
(target-matched GQA geometry), FFN 12288, 8 uniform target taps
`[1, 6, 12, 17, 23, 28, 34, 39]`, Markov rank 256 + confidence head,
`block_size=7` — **train = serve**.

## Recipe

| | Stage 1 | Stage 2 |
|:--|:--|:--|
| RoPE | native (`rope_scaling: null`), 8192 | YaRN factor 128 → 1,048,576, no mscale keys |
| Sequences | 8,192 | 65,536 |
| Corpus | 699,637 deduplicated self-regenerations | stage-1 corpus + 1,092 agentic trajectories ×8 (708,373 rows) |
| Epochs | 3 | 1 (weights-only warm start from stage 1) |
| LR | constant 5e-4, 4% warmup then hold | constant 5e-4, 64-step re-warm |
| Batch | global 512 (b16/ACC16 over 2 nodes) | global 512 (b8/ACC32 over 2 nodes) |
| Loss | 0.1 CE + 0.9 L1 + 1.0 confidence BCE, 512 anchors, within-block decay γ=4 | same |

Regeneration sampling: temperature 1.0, top-p 0.95, reasoning effort 0.99.

## Accept length, temperature 0

DSPARK block size 7, `top_k=-1`, thinking effort 0.99, 128 prompts/task
(`scripts/bench_dspark_accept.py` protocol). v1 is the previous
5-layer/0.9B block-15-trained drafter evaluated identically.

| Dataset | v1 | v2 | Δ |
|:--|--:|--:|--:|
| GSM8K | 4.787 | **5.321** | +11% |
| MATH500 | 4.143 | **4.731** | +14% |
| MBPP | 3.439 | **4.149** | +21% |
| HumanEval | 3.349 | **3.986** | +19% |
| MT-Bench | 3.114 | **3.721** | +19% |
| LiveCodeBench | 2.929 | **3.438** | +17% |
| AIME25 | 2.894 | **3.428** | +18% |
| Alpaca | 2.782 | **3.384** | +22% |
| Arena-Hard-v2 | 2.698 | **3.262** | +21% |
| **Mean** | 3.348 | **3.936** | **+17.6%** |

## Long-context holdout

67 held-out agentic trajectories (never trained on), each rendered as its
full conversation prefix up to the final assistant turn, temperature 0:

| Prompt tokens | n | accept length |
|:--|--:|--:|
| < 8K | 3 | 4.482 |
| 8K–16K | 16 | 3.592 |
| 16K–32K | 16 | 3.560 |
| ≥ 32K | 32 | 3.534 |
| **overall** | 67 | **3.597** |

Accept length is flat across context length — the two-stage YaRN recipe
shows no long-context degradation cliff.

## Operational notes

- 64K training memory is a cliff-edge around the engine's
  `mem_fraction_static`: 0.35 passes a worst-case all-long-batch smoke at
  283 GB steady (GB300, expandable_segments on); 0.25 fails prefill
  admission (SWA pool below the 8×64K token need); 0.30 admits but
  driver-OOMs — a smaller pool forces multi-wave prefill whose overlapping
  capture transients exceed the single-wave peak. Do not lower it blindly.
- Serving the 1.39B unquant draft alongside the NVFP4 target on B300
  (267 GB) needs `--mem-fraction-static 0.60` (0.68 OOMs at load).
