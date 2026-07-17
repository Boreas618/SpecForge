# Migration candidate reports

This folder turns the reusable work found in `_temp/nda_specforge` into PR-ready
migration proposals for the current SpecForge architecture. The source branch
and `main` share commit `fd6b16f`, but both evolved independently after that
point. An exact audit found 61 non-merge source commits: only three apply cleanly
to `main`, and none is both safe and immediately usable without adaptation.

These documents therefore describe **behavioral ports**, not literal commit
replays. File paths and symbols in each implementation plan refer to the current
tree unless explicitly labeled as source material.

## Candidate index

| # | Candidate | Source commits | Size | Dependencies | Recommended disposition |
|---|---|---|---|---|---|
| 1 | [Checkpoint vs generatable vocabulary](01-padded-checkpoint-vocabulary.md) | `f43c983` | M | none | Port the geometry contract; not standalone Inkling enablement |
| 2 | [GQA validation and GQA16 config](02-gqa-validation-and-gqa16.md) | `b7f5c8f`, `40be4f8` | S | #1 for the private target | Port the guard; adapt the final config |
| 3 | [Content-addressed dataset cache keys](03-content-addressed-dataset-cache.md) | `1f236bc` | M | none | Port across all active cache callers |
| 4 | [Deterministic regenerated-data sampling](04-deterministic-regen-sampling.md) | `0155ee1`, `71b1b41` | S | none | Extract the final standalone utility |
| 5 | [GLM-5.2 DSpark training and DeepSpec evaluation](05-glm52-dspark-and-deepspec-eval.md) | `e5755a7` through `114b1f2` | XL | #3 recommended | Implement as a stacked feature series |
| 6 | [Private typed-token target integration](06-private-typed-token-target.md) | `7490759`, `0bcd37f` through `40be4f8` | XL | #1–#5 in part | Restricted, selective port only |
| 7 | [World-consistent checkpoint selection](07-world-consistent-resume.md) | `f9f0b3a`, `4675b15` | M | none | Move the invariant into `CheckpointManager` |
| 8 | [Distributed freeze forensics](08-distributed-freeze-forensics.md) | `5e19d77`, `72614f6` | S | none | Generalize the final script for current launchers |

## Recommended delivery order

1. Land candidates 1–4 independently. They are small, broadly useful, and
   establish correctness primitives needed by the model-specific work.
2. Land candidate 7 and candidate 8 independently. Together they make failures
   fail closed and leave useful evidence, without changing model math.
3. Deliver candidate 5 as stacked PRs: data contract, DSpark capture/training,
   evaluator, then recipes and GPU gates.
4. Start candidate 6 only after an NDA/publication review. Reuse the generic
   pieces already landed rather than copying its legacy trainer or launcher.

## Migration rules

- Preserve the current separation of concerns: model definitions under
  `specforge/modeling`, algorithm behavior in `specforge/training/strategies`,
  target extraction in `specforge/inference/target_engine`, and topology
  assembly in `specforge/launch.py` or `examples/disagg`.
- Prefer the current `DSparkDraftModel`, `DSparkTrainStrategy`, server-capture
  schema, `Evaluator`, and `CheckpointManager`. Do not restore the removed
  standalone `scripts/train_dspark.py` architecture.
- Add CPU tests for contracts and transforms before GPU validation. Every
  model-specific PR must name its minimal GPU gate and its expected metric.
- Configuration fields that describe DSpark belong in `dflash_config`; older
  source configs placed several of them at the top level.
- Never copy credentials, private clone URLs, fixed cluster IPs, or personal
  repository names from the source branch. Examples must consume credentials
  only from the environment and must fail with an actionable message when a
  required credential is absent.
- A source commit that contains both generic and cluster-specific changes must
  be split. Preserve the behavior and attribution in the new PR description,
  but keep unrelated launch policy out of the code diff.

## Explicitly excluded source changes

- The early DSpark implementation (`1c2c647`, `3a4f432`) is superseded by the
  current implementation (`832263f`, `8b2ce67`) and would reintroduce duplicate
  model and trainer classes.
- The detached preview uploader (`fb58d8b`) is not a migration candidate. Its
  source contains a committed credential and assumes obsolete log and checkpoint
  layouts.
- Intermediate MoE backend, learning-rate, GQA8, and resume-policy toggles are
  not replayed. Each report targets the final validated behavior only.
- `967a2c1` is supporting documentation for the GLM work, not an independent
  feature. It can be rewritten after candidate 5 lands so its commands and file
  links match the final implementation.
