"""DEPRECATED wrapper validating regenerated JSONL via the baseline profile.

Finalized regeneration artifacts are validated with
``specforge data regen validate --artifact <dir>``; that path enforces the
full publication gates (coverage, thresholds, tampering). This wrapper keeps
the historical loose-JSONL checks for one release by routing each row through
the pipeline's record normalization and the registered ``baseline`` validator,
plus the legacy expectations (single success status, terminal assistant turn,
reasoning-mode contract, think-marker defense).
"""

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from scripts.conversation_validation import has_think_marker, validate_conversation
except ModuleNotFoundError:
    from conversation_validation import has_think_marker, validate_conversation

try:
    import specforge  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEPRECATION_NOTICE = (
    "scripts/validate_regenerated_data.py is a deprecated compatibility "
    "wrapper and will be removed after one release. Finalized artifacts are "
    "validated with `specforge data regen validate --artifact <dir>`."
)


@dataclass(frozen=True)
class ValidationSummary:
    rows: int
    assistant_messages: int
    duplicate_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DEPRECATED wrapper: validate regenerated ShareGPT JSONL through "
            "the baseline regeneration profile."
        )
    )
    parser.add_argument("--data-path", required=True, type=Path)
    reasoning_group = parser.add_mutually_exclusive_group()
    reasoning_group.add_argument(
        "--expect-non-reasoning",
        action="store_true",
        help="Reject non-empty assistant reasoning_content.",
    )
    reasoning_group.add_argument(
        "--expect-reasoning",
        action="store_true",
        help="Require non-empty assistant reasoning_content.",
    )
    parser.add_argument(
        "--strict-think-markers",
        action="store_true",
        help="Reject <think> and </think> in assistant content or reasoning_content.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            yield line_number, row


def _baseline_findings(row: Dict[str, Any]) -> list:
    """Run the row through the pipeline contracts it must satisfy to train."""

    from specforge.data.regen.contracts import (
        RecordEnvelope,
        RecordKey,
        canonical_digest,
    )
    from specforge.data.regen.errors import ContractError
    from specforge.data.regen.records.messages import (
        normalize_sharegpt_record,
        preserved_digest,
    )
    from specforge.data.regen.validators.baseline import create_baseline_validator

    try:
        normalized = normalize_sharegpt_record(row, source_name="legacy", position=0)
    except ContractError as exc:
        raise ValueError(str(exc)) from None
    envelope = RecordEnvelope(
        key=RecordKey(source="legacy", source_id=normalized.source_id),
        input_position=0,
        payload=normalized.payload,
        source_fingerprint="sha256:unverified-legacy-jsonl",
        source_payload_digest=canonical_digest(normalized.payload),
        source_preserved_digest=preserved_digest(normalized.payload),
    )
    return create_baseline_validator().validate(envelope)


def validate_row(
    row: Dict[str, Any],
    *,
    expect_non_reasoning: bool,
    expect_reasoning: bool = False,
    strict_think_markers: bool,
) -> int:
    """Validate one generated training row and return its assistant count."""

    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError("id must be a non-empty string")
    if row.get("status") != "success":
        raise ValueError("status must be 'success'")

    messages = row.get("conversations")
    conversation_error = validate_conversation(messages)
    if conversation_error is not None:
        raise ValueError(conversation_error)

    findings = _baseline_findings(row)
    if findings:
        first = findings[0]
        raise ValueError(f"{first.code}: {first.message}")

    assistant_count = 0
    for index, message in enumerate(messages):
        role = message.get("role")
        if role != "assistant":
            continue

        content = message["content"]
        assistant_count += 1
        reasoning = message.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(
                f"assistant message {index} reasoning_content must be a string or null"
            )
        if expect_non_reasoning and reasoning and reasoning.strip():
            raise ValueError(
                f"assistant message {index} has non-empty reasoning_content"
            )
        if expect_reasoning and (
            not isinstance(reasoning, str) or not reasoning.strip()
        ):
            raise ValueError(f"assistant message {index} has empty reasoning_content")
        if strict_think_markers and has_think_marker(content):
            raise ValueError(f"assistant message {index} contains a thinking marker")
        if (
            strict_think_markers
            and isinstance(reasoning, str)
            and has_think_marker(reasoning)
        ):
            raise ValueError(
                f"assistant message {index} reasoning_content contains a thinking marker"
            )

    if assistant_count == 0:
        raise ValueError("conversation has no assistant message")
    if messages[-1].get("role") != "assistant":
        raise ValueError("conversation must end with an assistant message")
    return assistant_count


def validate_dataset(
    path: Path,
    *,
    expect_non_reasoning: bool = False,
    expect_reasoning: bool = False,
    strict_think_markers: bool = False,
) -> ValidationSummary:
    if expect_non_reasoning and expect_reasoning:
        raise ValueError(
            "expect_non_reasoning and expect_reasoning are mutually exclusive"
        )
    seen_ids = set()
    duplicate_rows = 0
    rows = 0
    assistant_messages = 0

    for line_number, row in iter_jsonl(path):
        try:
            assistant_messages += validate_row(
                row,
                expect_non_reasoning=expect_non_reasoning,
                expect_reasoning=expect_reasoning,
                strict_think_markers=strict_think_markers,
            )
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc

        row_id = row["id"]
        if row_id in seen_ids:
            duplicate_rows += 1
        else:
            seen_ids.add(row_id)
        rows += 1

    if rows == 0:
        raise ValueError(f"{path} contains no data rows")
    if duplicate_rows:
        warnings.warn(
            f"found {duplicate_rows} rows with duplicate ids; duplicates are allowed",
            stacklevel=2,
        )
    return ValidationSummary(
        rows=rows,
        assistant_messages=assistant_messages,
        duplicate_rows=duplicate_rows,
    )


def main() -> None:
    args = parse_args()
    print(DEPRECATION_NOTICE, file=sys.stderr)
    try:
        summary = validate_dataset(
            args.data_path,
            expect_non_reasoning=args.expect_non_reasoning,
            expect_reasoning=args.expect_reasoning,
            strict_think_markers=args.strict_think_markers,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"validation failed: {exc}") from exc

    print(f"rows: {summary.rows}")
    print(f"assistant messages: {summary.assistant_messages}")
    print(f"duplicate rows (allowed): {summary.duplicate_rows}")
    print("validation passed")


if __name__ == "__main__":
    main()
