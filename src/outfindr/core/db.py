"""SQLite connection helpers and idempotent schema init."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS vision_cache (
    image_sha256    TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    analysis_json   TEXT NOT NULL,
    created         TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (image_sha256, model_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS bot_replies (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    platform          TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    post_id           TEXT,
    requesting_user   TEXT,
    image_sha256      TEXT,
    reply_id          TEXT,
    response_text     TEXT NOT NULL,
    confidence        REAL,
    created           TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bot_replies_unique
    ON bot_replies(platform, source_id);
CREATE INDEX IF NOT EXISTS idx_bot_replies_post ON bot_replies(post_id);

CREATE TABLE IF NOT EXISTS opt_outs (
    platform   TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    created    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (platform, user_id)
);
"""


def connect(database_path: str) -> sqlite3.Connection:
    parent = os.path.dirname(database_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(database_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
