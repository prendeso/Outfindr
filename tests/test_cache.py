from __future__ import annotations

from outfindr.core import cache


def test_put_and_get_round_trip(db_conn, sample_analysis):
    cache.put(db_conn, "sha-a", "model-x", "v1", sample_analysis)
    got = cache.get(db_conn, "sha-a", "model-x", "v1")
    assert got is not None
    assert got.overall_style == sample_analysis.overall_style
    assert len(got.items) == len(sample_analysis.items)


def test_cache_keyed_by_prompt_version(db_conn, sample_analysis):
    cache.put(db_conn, "sha-a", "model-x", "v1", sample_analysis)
    assert cache.get(db_conn, "sha-a", "model-x", "v2") is None


def test_cache_keyed_by_model_id(db_conn, sample_analysis):
    cache.put(db_conn, "sha-a", "model-x", "v1", sample_analysis)
    assert cache.get(db_conn, "sha-a", "model-y", "v1") is None


def test_cache_keyed_by_query_hash(db_conn, sample_analysis):
    qa = cache.query_hash("the yellow jacket")
    qb = cache.query_hash("the second person from the left")
    cache.put(db_conn, "sha-a", "model-x", "v1", sample_analysis, query_hash=qa)

    # same image + different query => miss
    assert cache.get(db_conn, "sha-a", "model-x", "v1", query_hash=qb) is None
    # same image + same query => hit
    assert cache.get(db_conn, "sha-a", "model-x", "v1", query_hash=qa) is not None
    # same image + no query => miss (empty query is its own bucket)
    assert cache.get(db_conn, "sha-a", "model-x", "v1") is None


def test_query_hash_normalizes_whitespace_and_case():
    assert cache.query_hash("  Hello  ") == cache.query_hash("hello")
    assert cache.query_hash(None) == ""
    assert cache.query_hash("") == ""
    assert cache.query_hash("   ") == ""
    assert cache.query_hash("nonempty") != ""


def test_miss_returns_none(db_conn):
    assert cache.get(db_conn, "nope", "m", "p") is None


def test_sha256_bytes_stable():
    a = cache.sha256_bytes(b"hello")
    b = cache.sha256_bytes(b"hello")
    assert a == b
    assert a != cache.sha256_bytes(b"hello!")


def test_reply_log_idempotency(db_conn):
    assert not cache.reply_exists(db_conn, "reddit", "abc")
    cache.insert_reply(
        db_conn,
        platform="reddit",
        source_id="abc",
        post_id="p1",
        requesting_user="alice",
        image_sha256="sha",
        reply_id="r1",
        response_text="hi",
        confidence=0.8,
    )
    assert cache.reply_exists(db_conn, "reddit", "abc")
    # second insert is a no-op due to UNIQUE constraint + INSERT OR IGNORE
    cache.insert_reply(
        db_conn,
        platform="reddit",
        source_id="abc",
        post_id="p1",
        requesting_user="alice",
        image_sha256="sha",
        reply_id="r2",
        response_text="hi again",
        confidence=0.9,
    )
    rows = db_conn.execute(
        "SELECT reply_id FROM bot_replies WHERE platform = 'reddit' AND source_id = 'abc'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["reply_id"] == "r1"


def test_replies_today_counts_only_today(db_conn):
    assert cache.replies_today(db_conn, "reddit") == 0
    cache.insert_reply(
        db_conn,
        platform="reddit",
        source_id="x",
        post_id=None,
        requesting_user=None,
        image_sha256=None,
        reply_id=None,
        response_text="hi",
        confidence=None,
    )
    assert cache.replies_today(db_conn, "reddit") == 1
    assert cache.replies_today(db_conn, "bluesky") == 0


def test_opt_outs(db_conn):
    assert not cache.is_opted_out(db_conn, "reddit", "alice")
    db_conn.execute(
        "INSERT INTO opt_outs (platform, user_id) VALUES (?, ?)",
        ("reddit", "alice"),
    )
    assert cache.is_opted_out(db_conn, "reddit", "alice")
    assert not cache.is_opted_out(db_conn, "reddit", "bob")


def test_record_opt_out_and_in(db_conn):
    cache.record_opt_out(db_conn, "reddit", "alice")
    assert cache.is_opted_out(db_conn, "reddit", "alice")
    cache.record_opt_in(db_conn, "reddit", "alice")
    assert not cache.is_opted_out(db_conn, "reddit", "alice")


def test_record_opt_out_idempotent(db_conn):
    cache.record_opt_out(db_conn, "reddit", "alice")
    cache.record_opt_out(db_conn, "reddit", "alice")  # second call is a no-op
    rows = db_conn.execute(
        "SELECT * FROM opt_outs WHERE platform = 'reddit' AND user_id = 'alice'"
    ).fetchall()
    assert len(rows) == 1


def test_record_opt_in_when_not_opted_out_is_noop(db_conn):
    cache.record_opt_in(db_conn, "reddit", "ghost")
    assert not cache.is_opted_out(db_conn, "reddit", "ghost")
