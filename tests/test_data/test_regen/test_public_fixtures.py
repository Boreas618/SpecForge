from __future__ import annotations

import hashlib
import json
from pathlib import Path

from specforge.data.regen.records.messages import normalize_openai_record

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "data_regeneration"


def test_public_fixture_contracts_and_checksums():
    expected = {
        # Updated only when a fixture change is intentionally reviewed.
        "failure_cases.json": "85ea59027400f676d420b3fbfe07aaaad527f7b72000d08e8efb9dcfa6d84b9a",
        "legacy_snapshot.json": "446d3e22eb3f438e78f51aa9adfbe0d046e9c2b05b462544b92ab6b98238433e",
        "source_rows.jsonl": "299ac0dee04f561fa97487d4c291fbe0711fdab9c9177988b17b16705fcb34bb",
    }
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in FIXTURES.iterdir()
        if path.suffix in {".json", ".jsonl"}
    }
    assert actual == expected


def test_public_source_rows_normalize_without_private_adapters():
    rows = [
        json.loads(line)
        for line in (FIXTURES / "source_rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    normalized = [
        normalize_openai_record(row, source_name="fixture", position=index)
        for index, row in enumerate(rows)
    ]
    assert normalized[0].payload["conversations"][2]["reasoning_content"]
    assert normalized[1].payload["conversations"][1]["tool_calls"]
