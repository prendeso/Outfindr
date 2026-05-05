from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from outfindr.core.models import OutfitItem
from outfindr.core.search_links import links_for


def _item(**overrides):
    base = dict(
        category="outerwear",
        description="cropped black leather moto jacket",
        color="black",
        pattern="solid",
        material="leather",
        brand_guess=None,
        confidence=0.9,
        search_terms=["black leather moto jacket cropped"],
    )
    base.update(overrides)
    return OutfitItem(**base)


def test_uses_first_search_term():
    item = _item()
    links = links_for(item)
    google_q = parse_qs(urlparse(links["google"]).query)["q"][0]
    assert google_q == "black leather moto jacket cropped"


def test_falls_back_to_color_material_description_when_no_search_terms():
    item = _item(search_terms=[])
    links = links_for(item)
    q = parse_qs(urlparse(links["google_shopping"]).query)["q"][0]
    assert "black" in q
    assert "leather" in q
    assert "cropped" in q


def test_amazon_link_uses_k_param():
    item = _item()
    links = links_for(item)
    q = parse_qs(urlparse(links["amazon"]).query)["k"][0]
    assert "moto jacket" in q


def test_shopping_link_uses_tbm_shop():
    item = _item()
    links = links_for(item)
    parsed = urlparse(links["google_shopping"])
    qs = parse_qs(parsed.query)
    assert qs["tbm"] == ["shop"]


def test_url_encoded():
    item = _item(search_terms=["pink & blue tie-dye t-shirt"])
    links = links_for(item)
    # Spaces become + (quote_plus). Ampersand and other reserved chars must be escaped.
    assert " " not in links["google"]
    # The literal '&' inside the query value should be %26, so the parsed value
    # should include the ampersand back when decoded.
    q = parse_qs(urlparse(links["google"]).query)["q"][0]
    assert q == "pink & blue tie-dye t-shirt"


def test_amazon_affiliate_tag_appended_when_provided():
    item = _item()
    links = links_for(item, amazon_affiliate_tag="myaffil-20")
    qs = parse_qs(urlparse(links["amazon"]).query)
    assert qs["tag"] == ["myaffil-20"]
    # Other links unchanged
    assert "tag" not in parse_qs(urlparse(links["google"]).query)


def test_amazon_no_tag_by_default():
    item = _item()
    links = links_for(item)
    assert "tag" not in parse_qs(urlparse(links["amazon"]).query)


def test_amazon_empty_tag_string_is_ignored():
    item = _item()
    links = links_for(item, amazon_affiliate_tag="")
    assert "tag" not in parse_qs(urlparse(links["amazon"]).query)
