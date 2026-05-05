from __future__ import annotations

import json

import pytest

from outfindr.core.models import OutfitAnalysis, OutfitItem


def test_round_trip_json(sample_analysis):
    s = sample_analysis.to_json()
    parsed = OutfitAnalysis.from_json(s)
    assert parsed.items[0].description == sample_analysis.items[0].description
    assert parsed.overall_confidence == sample_analysis.overall_confidence
    assert parsed.model_id == sample_analysis.model_id


def test_invalid_category_rejected():
    with pytest.raises(ValueError, match="invalid category"):
        OutfitItem(
            category="hat",  # not in VALID_CATEGORIES
            description="x",
            color="black",
            pattern=None,
            material=None,
            brand_guess=None,
            confidence=0.9,
            search_terms=[],
        )


def test_confidence_out_of_range():
    with pytest.raises(ValueError):
        OutfitItem(
            category="top",
            description="x",
            color="black",
            pattern=None,
            material=None,
            brand_guess=None,
            confidence=1.5,
            search_terms=[],
        )


def test_overall_confidence_out_of_range(sample_analysis_dict):
    sample_analysis_dict["overall_confidence"] = 2.0
    with pytest.raises(ValueError):
        OutfitAnalysis.from_dict(sample_analysis_dict)


def test_missing_items_field_rejected():
    with pytest.raises(TypeError):
        OutfitAnalysis.from_dict({
            "items": "not a list",
            "overall_style": None,
            "overall_confidence": 0.5,
            "notes": None,
            "model_id": "x",
            "prompt_version": "y",
        })


def test_from_json_handles_unicode():
    a = OutfitAnalysis(
        items=[],
        overall_style="résumé chic",
        overall_confidence=0.0,
        notes=None,
        model_id="m",
        prompt_version="p",
    )
    parsed = OutfitAnalysis.from_json(a.to_json())
    assert parsed.overall_style == "résumé chic"
