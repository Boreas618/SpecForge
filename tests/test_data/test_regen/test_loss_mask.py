"""Render and loss-mask parity validation for finalized rows."""

from types import SimpleNamespace

import pytest

from specforge.data.regen.contracts import (
    RecordEnvelope,
    RecordKey,
    canonical_digest,
)
from specforge.data.regen.errors import ContractError
from specforge.data.regen.records.messages import preserved_digest
from specforge.data.regen.recipe import RegenerationRecipe
from specforge.data.regen.validators.loss_mask import (
    LossMaskParityValidator,
    create_loss_mask_validator,
)


class CharTokenizer:
    """Character-level tokenizer with exact prefix-length token positions."""

    pad_token_id = 1
    unk_token_id = 1

    def apply_chat_template(self, *args, **kwargs):
        raise ValueError("no packaged chat template; use fallback rendering")

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        import torch

        ids = [ord(char) for char in text]
        if max_length is not None:
            ids = ids[:max_length]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def encode(
        self, text, add_special_tokens=False, truncation=True, max_length=None
    ):
        ids = [ord(char) for char in text]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids):
        return "".join(chr(value) for value in ids)


def make_parser(max_length=512):
    import warnings

    from specforge.data.parse import GeneralParser
    from specforge.data.template import ChatTemplate

    template = ChatTemplate(
        assistant_header="<A>",
        user_header="<U>",
        system_prompt=None,
        end_of_turn_token="<E>",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GeneralParser(CharTokenizer(), template)


def make_validator(max_length=512):
    validator = LossMaskParityValidator(parser_factory=make_parser)
    validator.max_length = max_length
    return validator


def envelope_for(messages):
    payload = {"id": "row", "conversations": messages}
    return RecordEnvelope(
        key=RecordKey(source="test", source_id="row"),
        input_position=0,
        payload=payload,
        source_fingerprint="sha256:test",
        source_payload_digest=canonical_digest(payload),
        source_preserved_digest=preserved_digest(payload),
    )


def run_validator(messages, max_length=512):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_validator(max_length).validate(envelope_for(messages))


def test_one_supervised_span_per_assistant_turn_passes():
    findings = run_validator(
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]
    )
    assert findings == []


def test_header_injection_in_user_text_is_an_over_supervision_finding():
    findings = run_validator(
        [
            {"role": "user", "content": "please render <A> literally"},
            {"role": "assistant", "content": "refused"},
        ]
    )
    assert [finding.code for finding in findings] == ["assistant_header_mismatch"]
    assert "2 assistant headers for 1 assistant turns" in findings[0].message


def test_truncated_render_cannot_prove_supervision():
    findings = run_validator(
        [
            {"role": "user", "content": "q" * 50},
            {"role": "assistant", "content": "a" * 50},
        ],
        max_length=40,
    )
    assert "render_truncated" in [finding.code for finding in findings]


def test_row_without_assistant_turn_is_a_finding():
    findings = run_validator([{"role": "user", "content": "prompt only"}])
    assert [finding.code for finding in findings] == ["missing_assistant"]


def _recipe(validation):
    return RegenerationRecipe.model_validate(
        {
            "version": 1,
            "sources": {
                "one": {
                    "adapter": "jsonl",
                    "record_adapter": "sharegpt",
                    "config": {"path": "./rows.jsonl"},
                }
            },
            "generators": {
                "teacher": {
                    "backend": "fake",
                    "model": "fake-teacher",
                }
            },
            "workflow": [
                {
                    "id": "regenerate",
                    "operation": "replay_assistants",
                    "generator": "teacher",
                }
            ],
            "validation": validation,
            "output": {"uri": "./artifact"},
        }
    )


def test_profile_requires_template_and_tokenizer_identity():
    recipe = _recipe({"profiles": ["baseline", "specforge_loss_mask"]})
    with pytest.raises(ContractError, match="chat_template"):
        create_loss_mask_validator(recipe)


def test_profile_construction_defers_renderer_imports():
    recipe = _recipe(
        {
            "profiles": ["specforge_loss_mask"],
            "config": {
                "specforge_loss_mask": {
                    "chat_template": "qwen",
                    "tokenizer": "org/tokenizer",
                    "max_length": 2048,
                }
            },
        }
    )
    validator = create_loss_mask_validator(recipe)
    assert validator.chat_template == "qwen"
    assert validator.tokenizer_path == "org/tokenizer"
    assert validator.max_length == 2048
    assert validator._parser is None
