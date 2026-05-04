"""Vision-result cache and bot-reply log keyed by (sha, model, prompt version)."""
from __future__ import annotations

import hashlib
import sqlite3

from .models import OutfitAnalysis


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(
    conn: sqlite3.Connection,
    image_sha256: str,
    model_id: str,
    prompt_version: str,
) -> OutfitAnalysis | None:
    row = conn.execute(
        "SELECT analysis_json FROM vision_cache "
        "WHERE image_sha256 = ? AND model_id = ? AND prompt_version = ?",
        (image_sha256, model_id, prompt_version),
    ).fetchone()
    if row is None:
        return None
    return OutfitAnalysis.from_json(row["analysis_json"])


def put(
    conn: sqlite3.Connection,
    image_sha256: str,
    model_id: str,
    prompt_version: str,
    analysis: OutfitAnalysis,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO vision_cache "
        "(image_sha256, model_id, prompt_version, analysis_json) "
        "VALUES (?, ?, ?, ?)",
        (image_sha256, model_id, prompt_version, analysis.to_json()),
    )


def reply_exists(conn: sqlite3.Connection, platform: str, source_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM bot_replies WHERE platform = ? AND source_id = ?",
        (platform, source_id),
    ).fetchone()
    return row is not None


def insert_reply(
    conn: sqlite3.Connection,
    *,
    platform: str,
    source_id: str,
    post_id: str | None,
    requesting_user: str | None,
    image_sha256: str | None,
    reply_id: str | None,
    response_text: str,
    confidence: float | None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO bot_replies "
        "(platform, source_id, post_id, requesting_user, image_sha256, "
        "reply_id, response_text, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            platform,
            source_id,
            post_id,
            requesting_user,
            image_sha256,
            reply_id,
            response_text,
            confidence,
        ),
    )


def replies_today(conn: sqlite3.Connection, platform: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM bot_replies "
        "WHERE platform = ? AND date(created) = date('now')",
        (platform,),
    ).fetchone()
    return int(row["c"])


def is_opted_out(conn: sqlite3.Connection, platform: str, user_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM opt_outs WHERE platform = ? AND user_id = ?",
        (platform, user_id),
    ).fetchone()
    return row is not None
