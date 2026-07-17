"""SpecForge render and loss-mask parity validator.

Finalized rows must render through the packaged serving-compatible chat
template and the SpecForge parser with exactly one supervised span per
assistant turn. Torch, transformers, and the parser are imported lazily so
selecting other profiles keeps planning and inspection dependency-light.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..contracts import Finding, RecordEnvelope
from ..errors import ContractError

PROFILE_NAME = "specforge_loss_mask"


class LossMaskParityValidator:
    name = PROFILE_NAME

    def __init__(
        self,
        recipe: Any = None,
        parser_factory: Callable[[], Any] | None = None,
    ) -> None:
        config: Mapping[str, Any] = {}
        if recipe is not None:
            config = recipe.validation.config.get(PROFILE_NAME, {})
            if not isinstance(config, Mapping):
                raise ContractError(
                    f"validation.config.{PROFILE_NAME} must be an object"
                )
        self.chat_template = str(config.get("chat_template", ""))
        self.tokenizer_path = str(config.get("tokenizer", ""))
        self.max_length = int(config.get("max_length", 8192))
        if self.max_length <= 0:
            raise ContractError(f"{PROFILE_NAME}.max_length must be positive")
        #: deterministic sample: validate rows whose key-hash lands in bucket
        #: zero of ``sample_modulus`` buckets (1 = full scan). The baseline
        #: profile always full-scans regardless of this setting.
        self.sample_modulus = int(config.get("sample_modulus", 1))
        if self.sample_modulus <= 0:
            raise ContractError(f"{PROFILE_NAME}.sample_modulus must be positive")
        self._parser_factory = parser_factory
        self._parser: Any = None
        if parser_factory is None and (
            not self.chat_template or not self.tokenizer_path
        ):
            raise ContractError(
                f"the {PROFILE_NAME} profile requires validation.config."
                f"{PROFILE_NAME}.chat_template and .tokenizer so rows render "
                "through the packaged serving template"
            )

    def _resolve_parser(self) -> Any:
        if self._parser is None:
            if self._parser_factory is not None:
                self._parser = self._parser_factory()
            else:
                from transformers import AutoTokenizer

                from specforge.data.parse import GeneralParser
                from specforge.data.template import TEMPLATE_REGISTRY

                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                template = TEMPLATE_REGISTRY.get(self.chat_template)
                self._parser = GeneralParser(tokenizer, template)
        return self._parser

    def validate(self, envelope: RecordEnvelope) -> list[Finding]:
        findings: list[Finding] = []
        messages = [
            dict(message)
            for message in envelope.payload.get("conversations", [])
        ]
        assistant_turns = sum(
            1 for message in messages if message.get("role") == "assistant"
        )
        if assistant_turns == 0:
            return [
                Finding(
                    self.name,
                    "missing_assistant",
                    "row has no assistant turn to supervise",
                    record_key=envelope.key,
                )
            ]

        parser = self._resolve_parser()
        input_ids, loss_mask = parser.parse(messages, max_length=self.max_length)
        mask = [int(value) for value in loss_mask.tolist()]

        if len(input_ids) >= self.max_length:
            findings.append(
                Finding(
                    self.name,
                    "render_truncated",
                    f"rendered row fills max_length={self.max_length}; "
                    "supervised spans cannot be verified complete",
                    record_key=envelope.key,
                )
            )
        if not any(mask):
            findings.append(
                Finding(
                    self.name,
                    "no_supervised_tokens",
                    "rendered row has an all-zero loss mask",
                    record_key=envelope.key,
                )
            )
            return findings

        segments = sum(
            1
            for index, value in enumerate(mask)
            if value and (index == 0 or not mask[index - 1])
        )
        if segments != assistant_turns:
            findings.append(
                Finding(
                    self.name,
                    "supervised_span_mismatch",
                    f"expected {assistant_turns} supervised assistant spans "
                    f"but the parser produced {segments}",
                    record_key=envelope.key,
                )
            )

        # The training masker locates spans by the assistant header, so any
        # extra occurrence (for example header text inside a source-authored
        # message) silently supervises non-assistant tokens even when the
        # merged span count still matches.
        decode = getattr(parser.tokenizer, "decode", None)
        header = getattr(parser, "assistant_message_separator", "")
        if decode is not None and header:
            rendered = decode(input_ids.tolist())
            occurrences = rendered.count(header)
            if occurrences != assistant_turns:
                findings.append(
                    Finding(
                        self.name,
                        "assistant_header_mismatch",
                        f"rendered row contains {occurrences} assistant "
                        f"headers for {assistant_turns} assistant turns; "
                        "supervision boundaries are ambiguous",
                        record_key=envelope.key,
                    )
                )
        return findings


def create_loss_mask_validator(recipe=None) -> LossMaskParityValidator:
    return LossMaskParityValidator(recipe)


__all__ = ["LossMaskParityValidator", "create_loss_mask_validator"]
