"""Render an OutfitAnalysis as a per-platform reply string."""
from __future__ import annotations

from typing import Literal

from .models import OutfitAnalysis, OutfitItem
from .search_links import links_for

Platform = Literal["reddit", "bluesky", "web"]

DISCLOSURE = (
    "^(I'm a bot identifying outfits from images. Reply 'opt out' to be ignored. "
    "Suggestions are guesses, not endorsements.)"
)

LOW_CONFIDENCE_REPLY = (
    "I couldn't identify the outfit with enough confidence to be useful. "
    "A clearer, full-body image with good lighting usually helps.\n\n" + DISCLOSURE
)


def format_markdown(analysis: OutfitAnalysis, *, platform: Platform = "reddit") -> str:
    if not analysis.items:
        return LOW_CONFIDENCE_REPLY

    lines: list[str] = []
    if analysis.overall_style:
        lines.append(f"**Overall style:** {analysis.overall_style}")
        lines.append("")

    for idx, item in enumerate(analysis.items, start=1):
        lines.append(_format_item(idx, item, platform=platform))
        lines.append("")

    if analysis.notes:
        lines.append(f"_Notes: {analysis.notes}_")
        lines.append("")

    lines.append(DISCLOSURE)
    return "\n".join(lines).rstrip() + "\n"


def _format_item(idx: int, item: OutfitItem, *, platform: Platform) -> str:
    header = f"**{idx}. {item.category.title()}** — {item.description}"
    bits: list[str] = []
    if item.color:
        bits.append(f"color: {item.color}")
    if item.pattern:
        bits.append(f"pattern: {item.pattern}")
    if item.material:
        bits.append(f"material: {item.material}")
    if item.brand_guess:
        bits.append(f"brand guess: {item.brand_guess}")
    bits.append(f"confidence: {item.confidence:.0%}")

    links = links_for(item)
    link_md = (
        f"[Google]({links['google']}) · "
        f"[Shopping]({links['google_shopping']}) · "
        f"[Amazon]({links['amazon']})"
    )

    return f"{header}\n- " + "\n- ".join(bits) + f"\n- Search: {link_md}"
