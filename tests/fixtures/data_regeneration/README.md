# Synthetic data-regeneration fixtures

These fixtures exercise semantic text/reasoning/tool behavior only. They are
synthetic, public-safe, and deliberately contain no server locations, model
paths, credentials, private dataset text, tensor fields, or raw token streams.

- `source_rows.jsonl`: multi-turn reasoning and recorded-tool trajectories.
- `failure_cases.json`: transport, truncation, control-token, tool-shape,
  non-canonical rebuild, and crash/resume stimuli.
- `legacy_snapshot.json`: redacted contract shape retained from the prototype.

The checksum test makes changes deliberate and reviewable.
