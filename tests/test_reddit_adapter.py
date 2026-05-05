from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from outfindr.adapters import reddit_bot
from outfindr.adapters.reddit_bot import (
    DownloadedImage,
    RedditAdapter,
    resolve_image_url,
)
from outfindr.core import cache, config


def _settings() -> config.Settings:
    return config.Settings(
        anthropic_api_key="sk-test",
        reddit_client_id="cid",
        reddit_client_secret="csec",
        reddit_username="outfindr",
        reddit_password="pw",
        reddit_user_agent="ua",
        bot_username="outfindr",
        database_path=":memory:",
        confidence_floor=0.55,
        daily_reply_budget=200,
        log_level="INFO",
    )


def _mention(
    *,
    item_id="m1",
    submission=None,
    author_name="alice",
    was_comment=True,
    subject="username mention",
):
    submission = submission or _submission()
    item = MagicMock()
    item.id = item_id
    item.was_comment = was_comment
    item.subject = subject
    item.author = SimpleNamespace(name=author_name) if author_name else None
    item.submission = submission
    item.reply.return_value = SimpleNamespace(id="reply-id")
    return item


def _submission(
    url="https://i.redd.it/foo.jpg",
    over_18=False,
    sub_id="sub1",
    preview=None,
    media_metadata=None,
):
    s = MagicMock()
    s.id = sub_id
    s.url = url
    s.over_18 = over_18
    s.preview = preview
    s.media_metadata = media_metadata
    return s


def _adapter(
    db_conn,
    *,
    analyze_fn=None,
    download_fn=None,
):
    return RedditAdapter(
        settings=_settings(),
        conn=db_conn,
        reddit=MagicMock(),
        analyze_fn=analyze_fn or MagicMock(),
        download_fn=download_fn or (lambda _url: DownloadedImage(b"img", "image/jpeg")),
    )


# ---------- resolve_image_url ----------

def test_resolve_direct_jpg_url():
    s = _submission(url="https://example.com/foo.JPG")
    assert resolve_image_url(s) == "https://example.com/foo.JPG"


def test_resolve_i_redd_it():
    s = _submission(url="https://i.redd.it/abcd1234")
    assert resolve_image_url(s) == "https://i.redd.it/abcd1234"


def test_resolve_uses_preview_when_no_direct_url():
    preview = {
        "images": [
            {"source": {"url": "https://preview.redd.it/x.jpg?width=1080&amp;crop=smart"}}
        ]
    }
    s = _submission(url="https://reddit.com/r/foo/comments/xyz/", preview=preview)
    out = resolve_image_url(s)
    assert out == "https://preview.redd.it/x.jpg?width=1080&crop=smart"


def test_resolve_uses_gallery_metadata():
    media_metadata = {
        "abc": {"s": {"u": "https://i.redd.it/g1.jpg?foo=bar&amp;baz=qux"}},
        "def": {"s": {"u": "https://i.redd.it/g2.jpg"}},
    }
    s = _submission(url="https://reddit.com/gallery/abc", media_metadata=media_metadata)
    out = resolve_image_url(s)
    assert out and out.endswith(".jpg") or "redd.it/g" in out


def test_resolve_returns_none_for_unknown():
    s = _submission(url="https://example.com/something.html")
    assert resolve_image_url(s) is None


# ---------- handle() flow ----------

def test_non_comment_routes_to_dm_handler(db_conn):
    """A non-comment inbox item is a DM. Vision is never called for DMs."""
    item = _pm(subject="hello", body="what is this?")
    analyze = MagicMock()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)
    analyze.assert_not_called()
    # DMs do get a reply (help text), unlike outfit-ID mentions.
    item.reply.assert_called_once()


def test_skips_non_username_mention(db_conn):
    item = _mention(subject="comment reply")
    analyze = MagicMock()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)
    analyze.assert_not_called()


def test_skips_nsfw(db_conn):
    item = _mention(submission=_submission(over_18=True))
    analyze = MagicMock()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)
    analyze.assert_not_called()
    item.reply.assert_not_called()


def test_skips_when_no_image(db_conn):
    item = _mention(submission=_submission(url="https://example.com/no-image.html"))
    analyze = MagicMock()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)
    analyze.assert_not_called()
    # skip is recorded so we don't reprocess on next stream pass
    assert cache.reply_exists(db_conn, "reddit", item.id)


def test_skips_opted_out_user(db_conn):
    db_conn.execute("INSERT INTO opt_outs (platform, user_id) VALUES (?, ?)", ("reddit", "alice"))
    item = _mention(author_name="alice")
    analyze = MagicMock()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)
    analyze.assert_not_called()
    item.reply.assert_not_called()


def test_happy_path_replies_and_records(db_conn, sample_analysis):
    analyze = MagicMock(return_value=sample_analysis)
    item = _mention()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)

    analyze.assert_called_once()
    item.reply.assert_called_once()
    body = item.reply.call_args.args[0]
    assert "google" in body.lower()
    assert cache.reply_exists(db_conn, "reddit", item.id)


def test_idempotency_second_call_does_not_invoke_vision(db_conn, sample_analysis):
    analyze = MagicMock(return_value=sample_analysis)
    item = _mention()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)
    a.handle(item)  # same id
    assert analyze.call_count == 1
    assert item.reply.call_count == 1


def test_cache_hit_skips_vision(db_conn, sample_analysis):
    # Pre-populate cache with the sha that download_fn's bytes will produce.
    img_bytes = b"img"
    sha = cache.sha256_bytes(img_bytes)
    cache.put(db_conn, sha, config.VISION_MODEL_ID, config.PROMPT_VERSION, sample_analysis)

    analyze = MagicMock()
    item = _mention()
    a = _adapter(
        db_conn,
        analyze_fn=analyze,
        download_fn=lambda _u: DownloadedImage(img_bytes, "image/jpeg"),
    )
    a.handle(item)
    analyze.assert_not_called()
    item.reply.assert_called_once()


def test_low_confidence_short_circuits_to_low_conf_message(db_conn, sample_analysis_dict):
    sample_analysis_dict["overall_confidence"] = 0.30
    from outfindr.core.models import OutfitAnalysis

    low = OutfitAnalysis.from_dict(sample_analysis_dict)
    analyze = MagicMock(return_value=low)
    item = _mention()
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)

    body = item.reply.call_args.args[0]
    assert "couldn't identify" in body.lower()


def test_daily_budget_blocks_reply(db_conn, sample_analysis):
    # Fill the budget.
    settings = _settings()
    for i in range(settings.daily_reply_budget):
        cache.insert_reply(
            db_conn,
            platform="reddit",
            source_id=f"prev-{i}",
            post_id=None,
            requesting_user=None,
            image_sha256=None,
            reply_id=None,
            response_text="x",
            confidence=None,
        )

    analyze = MagicMock(return_value=sample_analysis)
    item = _mention(item_id="new")
    a = _adapter(db_conn, analyze_fn=analyze)
    a.handle(item)

    analyze.assert_not_called()
    item.reply.assert_not_called()


def test_vision_parse_error_skips_reply_and_records(db_conn, monkeypatch):
    from outfindr.core.vision import VisionParseError

    def boom(*a, **k):
        raise VisionParseError("bad json")

    item = _mention()
    a = _adapter(db_conn, analyze_fn=boom)
    a.handle(item)
    item.reply.assert_not_called()
    assert cache.reply_exists(db_conn, "reddit", item.id)


def test_failed_download_skips(db_conn):
    analyze = MagicMock()
    item = _mention()
    a = _adapter(db_conn, analyze_fn=analyze, download_fn=lambda _u: None)
    a.handle(item)
    analyze.assert_not_called()
    item.reply.assert_not_called()
    assert cache.reply_exists(db_conn, "reddit", item.id)


# ---------- DM / private message flow ----------

def _pm(*, item_id="pm1", subject="opt out", body="", author_name="alice"):
    item = MagicMock()
    item.id = item_id
    item.was_comment = False
    item.subject = subject
    item.body = body
    item.author = SimpleNamespace(name=author_name) if author_name else None
    item.reply.return_value = SimpleNamespace(id="reply-id")
    return item


def test_opt_out_pm_records_user_and_replies(db_conn):
    item = _pm(subject="opt out", author_name="alice")
    a = _adapter(db_conn)
    a.handle(item)

    assert cache.is_opted_out(db_conn, "reddit", "alice")
    item.reply.assert_called_once()
    body = item.reply.call_args.args[0]
    assert "opted out" in body.lower()


def test_opt_out_in_body_works(db_conn):
    item = _pm(subject="hello", body="please opt out, thanks", author_name="bob")
    a = _adapter(db_conn)
    a.handle(item)
    assert cache.is_opted_out(db_conn, "reddit", "bob")


def test_unsubscribe_keyword_opts_out(db_conn):
    item = _pm(subject="unsubscribe", body="", author_name="carol")
    a = _adapter(db_conn)
    a.handle(item)
    assert cache.is_opted_out(db_conn, "reddit", "carol")


def test_opt_in_pm_removes_opt_out(db_conn):
    cache.record_opt_out(db_conn, "reddit", "alice")
    item = _pm(subject="opt in", author_name="alice")
    a = _adapter(db_conn)
    a.handle(item)

    assert not cache.is_opted_out(db_conn, "reddit", "alice")
    body = item.reply.call_args.args[0]
    assert "opted back in" in body.lower()


def test_unrecognized_pm_replies_with_help(db_conn):
    item = _pm(subject="hi", body="how does this thing work?", author_name="dan")
    a = _adapter(db_conn)
    a.handle(item)

    body = item.reply.call_args.args[0]
    assert "outfindr" in body.lower()
    assert "u/outfindr" in body
    # Did not opt them out.
    assert not cache.is_opted_out(db_conn, "reddit", "dan")


def test_pm_idempotent(db_conn):
    item = _pm(subject="opt out", author_name="alice")
    a = _adapter(db_conn)
    a.handle(item)
    a.handle(item)  # same id
    assert item.reply.call_count == 1


def test_pm_without_author_still_handled(db_conn):
    """A deleted-account DM has no author; we should still reply gracefully."""
    item = _pm(author_name=None, subject="hello")
    a = _adapter(db_conn)
    a.handle(item)
    item.reply.assert_called_once()
