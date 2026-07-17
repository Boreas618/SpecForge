from __future__ import annotations

from dataclasses import replace

import pytest

from specforge.data.regen.backends.base import BackendResponse
from specforge.data.regen.backends.openai_chat import OpenAIChatBackend
from specforge.data.regen.codecs.exact_raw import ExactRawTokenCodec
from specforge.data.regen.codecs.kimi_k3 import (
    K3_CONTROL_TOKENS,
    KimiK3Codec,
    parse_k3_completion,
)
from specforge.data.regen.contracts import GenerationRequest, RecordKey
from specforge.data.regen.errors import FailureCategory, GenerationError
from specforge.data.regen.recipe import GeneratorSpec


def _generator(**changes) -> GeneratorSpec:
    value = {
        "backend": "fake",
        "model": "model",
        "codec": "structured_chat",
        "sampling": {"reasoning": "required"},
    }
    value.update(changes)
    return GeneratorSpec.model_validate(value)


def _request() -> GenerationRequest:
    return GenerationRequest(
        record_key=RecordKey("source", 1),
        stage_id="stage",
        variant="base",
        generation_ordinal=1,
        messages=({"role": "user", "content": "hello"},),
        sampling={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": None,
            "max_tokens": 32,
            "stop": [],
            "reasoning": "required",
            "extra": {},
        },
        seed=7,
    )


def test_openai_backend_preserves_split_reasoning_and_tool_semantics(monkeypatch):
    captured = {}

    def fake_post(url, payload, **kwargs):
        captured.update({"url": url, "payload": payload, **kwargs})
        return {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "I should call the tool.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"q": "x"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "ignored": "x"},
        }

    monkeypatch.setattr(
        "specforge.data.regen.backends.openai_chat.post_json", fake_post
    )
    backend = OpenAIChatBackend(
        _generator(backend="openai_chat", revision="commit"),
        {"endpoint": "https://inference.invalid", "api_key": "do-not-record"},
    )
    result = backend.generate(_request())
    assert result.message["reasoning_content"] == "I should call the tool."
    assert result.message["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert result.usage == {"prompt_tokens": 4, "completion_tokens": 5}
    assert captured["payload"]["seed"] == 7
    assert "do-not-record" not in repr(result)


def test_exact_raw_codec_keeps_only_digest_and_proves_rebuild():
    spec = _generator(
        backend="sglang_raw",
        revision="commit",
        codec="exact_raw_tokens",
        config={"format_identity": "synthetic-v1"},
    )

    def render(messages, tools, sampling):
        return [10, len(messages), len(tools)]

    def parse(ids, request):
        return {
            "role": "assistant",
            "reasoning_content": "reason",
            "content": "answer",
        }

    def rebuild(message, request):
        return [20, 21, 22]

    codec = ExactRawTokenCodec(
        spec,
        {
            "format_identity": "synthetic-v1",
            "render_request": render,
            "parse_response": parse,
            "rebuild_response": rebuild,
        },
    )
    request = codec.prepare_request(_request())
    assert request.input_token_ids == (10, 1, 0)
    result = codec.decode(
        BackendResponse(finish_reason="stop", raw_token_ids=(20, 21, 22)),
        request,
        backend="sglang_raw",
        model="model",
    )
    serialized = result.to_dict()
    assert result.exactness == "token_identical"
    assert serialized["raw_token_count"] == 3
    assert "raw_token_ids" not in serialized
    assert "input_token_ids" not in request.to_dict()

    bad = ExactRawTokenCodec(
        spec,
        {
            "format_identity": "synthetic-v1",
            "render_request": render,
            "parse_response": parse,
            "rebuild_response": lambda message, request: [20, 99, 22],
        },
    )
    with pytest.raises(GenerationError) as error:
        bad.decode(
            BackendResponse(finish_reason="stop", raw_token_ids=(20, 21, 22)),
            bad.prepare_request(_request()),
            backend="sglang_raw",
            model="model",
        )
    assert error.value.category == FailureCategory.EXACT_REBUILD
    assert "99" not in str(error.value)


def test_kimi_k3_strict_grammar_and_control_token_defense():
    transition = "<|close|>think<|sep|><|open|>response<|sep|>"
    tail = "<|close|>response<|sep|><|close|>message<|sep|>"
    assert parse_k3_completion(f"reason{transition}answer{tail}") == (
        "reason",
        "answer",
    )
    with pytest.raises(GenerationError, match="unique"):
        parse_k3_completion("plain text")
    with pytest.raises(GenerationError, match="control token"):
        parse_k3_completion(f"reason{transition}bad {K3_CONTROL_TOKENS[0]}{tail}")

    codec = KimiK3Codec(_generator(codec="kimi_k3"))
    result = codec.decode(
        BackendResponse(
            message={
                "role": "assistant",
                "content": f"reason{transition}answer{tail}",
            }
        ),
        _request(),
        backend="openai_chat",
        model="model",
    )
    assert result.message == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "reason",
    }
    assert result.raw_evidence_digest


def test_structured_chat_enforces_reasoning_policy_and_control_tokens():
    from specforge.data.regen.codecs.structured_chat import StructuredChatCodec

    spec = _generator(
        sampling={"reasoning": "disabled"},
        config={"control_tokens": ["<think>", "</think>"]},
    )
    codec = StructuredChatCodec(spec)
    request = _request()

    with pytest.raises(GenerationError, match="control token"):
        codec.decode(
            BackendResponse(
                message={"role": "assistant", "content": "<think>x</think>y"},
                finish_reason="stop",
            ),
            request,
            backend="fake",
            model="model",
        )
    with pytest.raises(GenerationError, match="thinking was disabled"):
        codec.decode(
            BackendResponse(
                message={
                    "role": "assistant",
                    "content": "y",
                    "reasoning_content": "sneaky",
                },
                finish_reason="stop",
            ),
            request,
            backend="fake",
            model="model",
        )

    required = StructuredChatCodec(_generator(sampling={"reasoning": "required"}))
    with pytest.raises(GenerationError, match="required and empty"):
        required.decode(
            BackendResponse(
                message={"role": "assistant", "content": "answer"},
                finish_reason="stop",
            ),
            request,
            backend="fake",
            model="model",
        )


def test_structured_chat_history_reasoning_strip_matches_serving_context():
    from specforge.data.regen.codecs.structured_chat import StructuredChatCodec

    spec = _generator(config={"history_reasoning": "strip"})
    codec = StructuredChatCodec(spec)
    request = replace(
        _request(),
        messages=(
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "earlier answer",
                "reasoning_content": "earlier thinking",
            },
            {"role": "user", "content": "second"},
        ),
    )
    prepared = codec.prepare_request(request)
    assistant = prepared.messages[1]
    assert assistant["content"] == "earlier answer"
    assert "reasoning_content" not in assistant
    # Non-assistant turns and the default preserve mode are untouched.
    assert prepared.messages[0] == request.messages[0]
    preserve = StructuredChatCodec(_generator())
    assert preserve.prepare_request(request).messages == request.messages
