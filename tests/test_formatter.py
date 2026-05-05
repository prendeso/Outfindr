from __future__ import annotations

from outfindr.core.formatter import (
    DISCLOSURE,
    LOW_CONFIDENCE_REPLY,
    format_markdown,
)
from outfindr.core.models import OutfitAnalysis


def test_includes_disclosure(sample_analysis):
    out = format_markdown(sample_analysis)
    assert DISCLOSURE in out


def test_includes_each_item(sample_analysis):
    out = format_markdown(sample_analysis)
    for item in sample_analysis.items:
        assert item.description in out


def test_includes_search_links(sample_analysis):
    out = format_markdown(sample_analysis)
    assert "google.com" in out
    assert "amazon.com" in out
    assert "tbm=shop" in out


def test_low_confidence_reply_when_no_items():
    empty = OutfitAnalysis(
        items=[],
        overall_style=None,
        overall_confidence=0.0,
        notes="no clothing visible",
        model_id="m",
        prompt_version="p",
    )
    out = format_markdown(empty)
    assert out == LOW_CONFIDENCE_REPLY


def test_includes_overall_style_when_present(sample_analysis):
    out = format_markdown(sample_analysis)
    assert "streetwear" in out


def test_renders_confidence_as_percent(sample_analysis):
    out = format_markdown(sample_analysis)
    # 0.92 -> 92%
    assert "92%" in out


def test_reddit_markdown_uses_link_syntax(sample_analysis):
    out = format_markdown(sample_analysis, platform="reddit")
    # markdown link: [text](url)
    assert "[Google](" in out
    assert "[Amazon](" in out
