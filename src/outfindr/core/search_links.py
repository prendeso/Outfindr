"""Build shopping search links for an OutfitItem. No network calls."""
from __future__ import annotations

from urllib.parse import quote_plus

from .models import OutfitItem


def _query_for(item: OutfitItem) -> str:
    if item.search_terms:
        return item.search_terms[0]
    parts = [item.color, item.material, item.description, item.category]
    return " ".join(p for p in parts if p)


def links_for(
    item: OutfitItem,
    *,
    amazon_affiliate_tag: str | None = None,
) -> dict[str, str]:
    """Return shopping search URLs for one item.

    `amazon_affiliate_tag`, when provided, is appended to the Amazon URL as
    the `tag` query parameter for affiliate-revenue tracking.
    """
    q = quote_plus(_query_for(item))
    amazon_url = f"https://www.amazon.com/s?k={q}"
    if amazon_affiliate_tag:
        amazon_url += f"&tag={quote_plus(amazon_affiliate_tag)}"
    return {
        "google": f"https://www.google.com/search?q={q}",
        "google_shopping": f"https://www.google.com/search?tbm=shop&q={q}",
        "amazon": amazon_url,
    }
