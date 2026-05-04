"""Platform-agnostic data model for outfit analysis."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Category = Literal[
    "top",
    "bottom",
    "outerwear",
    "footwear",
    "accessory",
    "bag",
    "headwear",
    "jewelry",
]

VALID_CATEGORIES: frozenset[str] = frozenset(
    ["top", "bottom", "outerwear", "footwear", "accessory", "bag", "headwear", "jewelry"]
)


@dataclass
class OutfitItem:
    category: Category
    description: str
    color: str
    pattern: str | None
    material: str | None
    brand_guess: str | None
    confidence: float
    search_terms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"invalid category: {self.category!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        if not isinstance(self.search_terms, list):
            raise TypeError("search_terms must be a list")


@dataclass
class OutfitAnalysis:
    items: list[OutfitItem]
    overall_style: str | None
    overall_confidence: float
    notes: str | None
    model_id: str
    prompt_version: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError(f"overall_confidence out of range: {self.overall_confidence}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "OutfitAnalysis":
        data = json.loads(s)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutfitAnalysis":
        items_raw = data.get("items")
        if not isinstance(items_raw, list):
            raise TypeError("items must be a list")
        items = [OutfitItem(**item) for item in items_raw]
        return cls(
            items=items,
            overall_style=data.get("overall_style"),
            overall_confidence=float(data["overall_confidence"]),
            notes=data.get("notes"),
            model_id=data["model_id"],
            prompt_version=data["prompt_version"],
        )
