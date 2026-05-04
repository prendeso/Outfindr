"""Build shopping search links for an OutfitItem. No network calls."""
from __future__ import annotations

from urllib.parse import quote_plus

from .models import OutfitItem


def _query_for(item: OutfitItem) -> str:
    if item.search_terms:
        return item.search_terms[0]
    parts = [item.color, item.material, item.description, item.category]
    return " ".join(p for p in parts if p)


def links_for(item: OutfitItem) -> dict[str, str]:
    """Return shopping search URLs for one item."""
    q = quote_plus(_query_for(item))
    return {
        "google": f"https://www.google.com/search?q={q}",
        "google_shopping": f"https://www.google.com/search?tbm=shop&q={q}",
        "amazon": f"https://www.amazon.com/s?k={q}",
    }
