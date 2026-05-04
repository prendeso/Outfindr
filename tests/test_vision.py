from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from outfindr.core import config, vision
from outfindr.core.vision import VisionParseError, analyze_outfit


def _fake_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def _make_client(text: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _fake_response(text)
    return client


def test_parses_valid_response(sample_analysis_dict, sample_image_bytes):
    client = _make_client(json.dumps(sample_analysis_dict))
    result = analyze_outfit(
        sample_image_bytes,
        content_type="image/jpeg",
        client=client,
    )
    assert len(result.items) == 3
    assert result.overall_style == "streetwear"


def test_sends_cache_control_on_system(sample_analysis_dict, sample_image_bytes):
    client = _make_client(json.dumps(sample_analysis_dict))
    analyze_outfit(sample_image_bytes, content_type="image/jpeg", client=client)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == config.VISION_MODEL_ID
    system = kwargs["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "JSON" in system[0]["text"]


def test_sends_image_in_user_message(sample_analysis_dict, sample_image_bytes):
    client = _make_client(json.dumps(sample_analysis_dict))
    analyze_outfit(sample_image_bytes, content_type="image/png", client=client)
    kwargs = client.messages.create.call_args.kwargs
    user_msg = kwargs["messages"][0]
    assert user_msg["role"] == "user"
    image_block = user_msg["content"][0]
    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["type"] == "base64"


def test_raises_on_unsupported_content_type(sample_image_bytes):
    with pytest.raises(VisionParseError, match="unsupported content type"):
        analyze_outfit(sample_image_bytes, content_type="image/bmp", client=MagicMock())


def test_raises_on_invalid_json(sample_image_bytes):
    client = _make_client("definitely not json")
    with pytest.raises(VisionParseError, match="not valid JSON"):
        analyze_outfit(sample_image_bytes, content_type="image/jpeg", client=client)


def test_raises_on_schema_mismatch(sample_image_bytes):
    client = _make_client(json.dumps({"items": "nope", "overall_confidence": 0.5}))
    with pytest.raises(VisionParseError, match="schema"):
        analyze_outfit(sample_image_bytes, content_type="image/jpeg", client=client)


def test_strips_markdown_fences(sample_analysis_dict, sample_image_bytes):
    fenced = "```json\n" + json.dumps(sample_analysis_dict) + "\n```"
    client = _make_client(fenced)
    result = analyze_outfit(sample_image_bytes, content_type="image/jpeg", client=client)
    assert len(result.items) == 3


def test_fills_in_model_id_and_prompt_version_if_missing(
    sample_analysis_dict, sample_image_bytes
):
    sample_analysis_dict.pop("model_id")
    sample_analysis_dict.pop("prompt_version")
    client = _make_client(json.dumps(sample_analysis_dict))
    result = analyze_outfit(sample_image_bytes, content_type="image/jpeg", client=client)
    assert result.model_id == config.VISION_MODEL_ID
    assert result.prompt_version == config.PROMPT_VERSION


def test_raises_on_empty_text(sample_image_bytes):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=[])
    with pytest.raises(VisionParseError, match="no text content"):
        analyze_outfit(sample_image_bytes, content_type="image/jpeg", client=client)
