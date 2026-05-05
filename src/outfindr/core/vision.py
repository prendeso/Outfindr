"""Call Claude vision to identify clothing in an image.

The system prompt is large and static, so we send it with
`cache_control: ephemeral` for prompt caching. The user message is the image
only; the schema lives entirely in the cached system prompt.
"""
from __future__ import annotations

import base64
import json
from importlib import resources
from typing import Any

import anthropic

from . import config
from .models import OutfitAnalysis

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    ["image/jpeg", "image/png", "image/webp", "image/gif"]
)


class VisionParseError(Exception):
    """Raised when the model's response is not valid JSON or doesn't match the schema."""


_SYSTEM_PROMPT_CACHE: dict[str, str] = {}


def _system_prompt() -> str:
    version = config.PROMPT_VERSION
    cached = _SYSTEM_PROMPT_CACHE.get(version)
    if cached is None:
        cached = (
            resources.files("outfindr.prompts")
            .joinpath(f"{version}.txt")
            .read_text(encoding="utf-8")
        )
        _SYSTEM_PROMPT_CACHE[version] = cached
    return cached


def analyze_outfit(
    image_bytes: bytes,
    *,
    content_type: str,
    user_query: str | None = None,
    client: anthropic.Anthropic | None = None,
    api_key: str | None = None,
    max_tokens: int = 1500,
) -> OutfitAnalysis:
    """Analyze a single image and return an OutfitAnalysis.

    `user_query`, when given, is appended to the user message after the image
    so the model can disambiguate ("the yellow jacket", "second from left",
    etc.). It does NOT go in the system prompt — the system prompt is the
    cache target, the user query is dynamic.
    """
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise VisionParseError(f"unsupported content type: {content_type}")

    if client is None:
        client = anthropic.Anthropic(api_key=api_key)

    image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    user_content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": content_type,
                "data": image_b64,
            },
        }
    ]
    cleaned_query = (user_query or "").strip()
    if cleaned_query:
        user_content.append({"type": "text", "text": f"User query: {cleaned_query}"})

    response = client.messages.create(
        model=config.VISION_MODEL_ID,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": _system_prompt(),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )

    text = _extract_text(response)
    return _parse_analysis(text)


def _extract_text(response: Any) -> str:
    blocks = getattr(response, "content", None) or []
    chunks: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if text:
                chunks.append(text)
    if not chunks:
        raise VisionParseError("model returned no text content")
    return "".join(chunks).strip()


def _parse_analysis(text: str) -> OutfitAnalysis:
    raw = _strip_fences(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise VisionParseError(f"model output is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise VisionParseError("model output is not a JSON object")

    data.setdefault("model_id", config.VISION_MODEL_ID)
    data.setdefault("prompt_version", config.PROMPT_VERSION)

    try:
        return OutfitAnalysis.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise VisionParseError(f"model output failed schema validation: {exc}") from exc


def _strip_fences(text: str) -> str:
    """Defensive: strip ```json fences if the model added them anyway."""
    s = text.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()
