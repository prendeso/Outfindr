from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from outfindr.core import cache, config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "web.db"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Re-import to pick up env. The app instance is created at import time but
    # the DB path is read lazily inside handlers, so a single import is fine.
    from outfindr.web.app import create_app

    app = create_app()
    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.text == "ok"


def test_index_shows_form(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Identify an outfit" in r.text
    assert 'name="file"' in r.text
    assert 'name="url"' in r.text


def test_post_with_file_runs_vision_and_renders(
    client, monkeypatch, sample_image_bytes, sample_analysis
):
    analyze = MagicMock(return_value=sample_analysis)
    monkeypatch.setattr("outfindr.web.app.vision.analyze_outfit", analyze)

    r = client.post(
        "/identify",
        files={"file": ("photo.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    analyze.assert_called_once()
    # First item from sample analysis appears in rendered HTML.
    assert "moto jacket" in r.text
    assert "fresh analysis" in r.text


def test_post_with_url_runs_vision_and_renders(
    client, monkeypatch, sample_image_bytes, sample_analysis
):
    analyze = MagicMock(return_value=sample_analysis)
    monkeypatch.setattr("outfindr.web.app.vision.analyze_outfit", analyze)

    fake_resp = SimpleNamespace(
        content=sample_image_bytes,
        headers={"content-type": "image/jpeg"},
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr("outfindr.web.app.requests.get", lambda *a, **k: fake_resp)

    r = client.post(
        "/identify",
        data={"url": "https://i.redd.it/test.jpg"},
    )
    assert r.status_code == 200, r.text
    analyze.assert_called_once()
    assert "moto jacket" in r.text


def test_cache_hit_skips_vision(
    client, monkeypatch, tmp_path, sample_image_bytes, sample_analysis
):
    # Pre-populate the cache so vision should never be called.
    from outfindr.core import db as db_mod

    conn = db_mod.connect(str(tmp_path / "web.db"))
    db_mod.init_db(conn)
    sha = cache.sha256_bytes(sample_image_bytes)
    cache.put(conn, sha, config.VISION_MODEL_ID, config.PROMPT_VERSION, sample_analysis)
    conn.close()

    analyze = MagicMock()
    monkeypatch.setattr("outfindr.web.app.vision.analyze_outfit", analyze)

    r = client.post(
        "/identify",
        files={"file": ("photo.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    analyze.assert_not_called()
    assert "cache hit" in r.text


def test_post_with_neither_file_nor_url_returns_400(client):
    r = client.post("/identify", data={"url": ""})
    assert r.status_code == 400
    assert "Provide a file or a URL" in r.text


def test_post_with_unsupported_content_type_returns_400(client, sample_image_bytes):
    r = client.post(
        "/identify",
        files={"file": ("photo.bmp", sample_image_bytes, "image/bmp")},
    )
    assert r.status_code == 400
    assert "Unsupported content type" in r.text


def test_history_lists_cached_runs(
    client, tmp_path, sample_image_bytes, sample_analysis
):
    from outfindr.core import db as db_mod

    conn = db_mod.connect(str(tmp_path / "web.db"))
    db_mod.init_db(conn)
    sha_a = cache.sha256_bytes(sample_image_bytes)
    sha_b = cache.sha256_bytes(sample_image_bytes + b"diff")
    cache.put(conn, sha_a, config.VISION_MODEL_ID, config.PROMPT_VERSION, sample_analysis)
    cache.put(conn, sha_b, config.VISION_MODEL_ID, config.PROMPT_VERSION, sample_analysis)
    conn.close()

    r = client.get("/history")
    assert r.status_code == 200
    assert sha_a[:12] in r.text
    assert sha_b[:12] in r.text


def test_query_field_passed_to_vision(
    client, monkeypatch, sample_image_bytes, sample_analysis
):
    analyze = MagicMock(return_value=sample_analysis)
    monkeypatch.setattr("outfindr.web.app.vision.analyze_outfit", analyze)

    r = client.post(
        "/identify",
        files={"file": ("photo.jpg", sample_image_bytes, "image/jpeg")},
        data={"query": "the yellow jacket"},
    )
    assert r.status_code == 200, r.text
    assert analyze.call_args.kwargs["user_query"] == "the yellow jacket"
    assert "You asked" in r.text
    assert "yellow jacket" in r.text


def test_query_separates_cache_buckets(
    client, monkeypatch, sample_image_bytes, sample_analysis
):
    analyze = MagicMock(return_value=sample_analysis)
    monkeypatch.setattr("outfindr.web.app.vision.analyze_outfit", analyze)

    client.post(
        "/identify",
        files={"file": ("p.jpg", sample_image_bytes, "image/jpeg")},
        data={"query": "the yellow jacket"},
    )
    client.post(
        "/identify",
        files={"file": ("p.jpg", sample_image_bytes, "image/jpeg")},
        data={"query": "the shoes"},
    )
    # Different queries on the same image must invoke vision twice.
    assert analyze.call_count == 2

    # Repeating the first query is a cache hit, no third call.
    client.post(
        "/identify",
        files={"file": ("p.jpg", sample_image_bytes, "image/jpeg")},
        data={"query": "the yellow jacket"},
    )
    assert analyze.call_count == 2


def test_amazon_affiliate_tag_appears_in_rendered_links(
    client, monkeypatch, sample_image_bytes, sample_analysis
):
    monkeypatch.setenv("AMAZON_AFFILIATE_TAG", "webtag-20")
    analyze = MagicMock(return_value=sample_analysis)
    monkeypatch.setattr("outfindr.web.app.vision.analyze_outfit", analyze)

    r = client.post(
        "/identify",
        files={"file": ("photo.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    assert "tag=webtag-20" in r.text
