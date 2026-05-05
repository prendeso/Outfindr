"""Reddit adapter: stream username mentions, identify outfits, reply.

Imports `praw` only here. `core/` stays platform-agnostic.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import praw
import praw.exceptions
import praw.models
import requests

from ..core import cache, config, formatter, vision
from ..core.models import OutfitAnalysis

PLATFORM = "reddit"
DOWNLOAD_TIMEOUT_SEC = 10
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DIRECT_IMAGE_HOSTS = ("i.redd.it", "i.imgur.com")
DIRECT_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

log = logging.getLogger(__name__)


@dataclass
class DownloadedImage:
    data: bytes
    content_type: str


class RedditAdapter:
    def __init__(
        self,
        *,
        settings: config.Settings,
        conn: sqlite3.Connection,
        reddit: praw.Reddit | None = None,
        analyze_fn: Callable[..., OutfitAnalysis] = vision.analyze_outfit,
        download_fn: Callable[[str], DownloadedImage | None] | None = None,
    ) -> None:
        self.settings = settings
        self.conn = conn
        self.reddit = reddit or praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            username=settings.reddit_username,
            password=settings.reddit_password,
            user_agent=settings.reddit_user_agent,
        )
        self._analyze_fn = analyze_fn
        self._download_fn = download_fn or download_image

    def run_forever(self) -> None:
        log.info("starting reddit inbox stream as u/%s", self.settings.reddit_username)
        while True:
            try:
                stream = self.reddit.inbox.stream(skip_existing=True)
                for item in stream:
                    self._handle_safely(item)
            except praw.exceptions.RedditAPIException as exc:
                log.warning("reddit api exception, backing off: %s", exc)
                time.sleep(30)
            except Exception:
                log.exception("inbox stream crashed, restarting in 30s")
                time.sleep(30)

    def _handle_safely(self, item: Any) -> None:
        try:
            self.handle(item)
        except Exception:
            log.exception("error handling inbox item id=%s", getattr(item, "id", "?"))

    def handle(self, item: Any) -> None:
        if getattr(item, "was_comment", False):
            if getattr(item, "subject", None) == "username mention":
                self._handle_mention(item)
            return
        self._handle_message(item)

    def _handle_mention(self, item: Any) -> None:
        item_id = item.id
        if cache.reply_exists(self.conn, PLATFORM, item_id):
            log.info("already replied to %s, skipping", item_id)
            _safe_mark_read(item)
            return

        if cache.replies_today(self.conn, PLATFORM) >= self.settings.daily_reply_budget:
            log.warning("daily reply budget reached; skipping %s", item_id)
            return

        author_name = _author_name(item)
        if author_name and cache.is_opted_out(self.conn, PLATFORM, author_name):
            log.info("user %s is opted out; skipping", author_name)
            _safe_mark_read(item)
            return

        submission = item.submission
        post_id = getattr(submission, "id", None)
        if getattr(submission, "over_18", False):
            log.info("skipping NSFW submission %s", post_id)
            _safe_mark_read(item)
            return

        url = resolve_image_url(submission)
        if not url:
            log.info("no image found on submission %s; skipping", post_id)
            _record_skip(self.conn, item, post_id, "no image found")
            _safe_mark_read(item)
            return

        downloaded = self._download_fn(url)
        if downloaded is None:
            log.info("download failed for %s; skipping", url)
            _record_skip(self.conn, item, post_id, "image download failed")
            _safe_mark_read(item)
            return

        sha = cache.sha256_bytes(downloaded.data)
        cached = cache.get(self.conn, sha, config.VISION_MODEL_ID, config.PROMPT_VERSION)
        if cached is not None:
            analysis = cached
            log.info("cache hit for sha=%s", sha[:12])
        else:
            try:
                analysis = self._analyze_fn(
                    downloaded.data,
                    content_type=downloaded.content_type,
                    api_key=self.settings.anthropic_api_key,
                )
            except vision.VisionParseError:
                log.exception("vision parse error on %s; skipping reply", item_id)
                _record_skip(self.conn, item, post_id, "vision parse error")
                _safe_mark_read(item)
                return
            cache.put(self.conn, sha, config.VISION_MODEL_ID, config.PROMPT_VERSION, analysis)

        if analysis.overall_confidence < self.settings.confidence_floor:
            reply_md = formatter.LOW_CONFIDENCE_REPLY
        else:
            reply_md = formatter.format_markdown(analysis, platform="reddit")

        reply = item.reply(reply_md)
        reply_id = getattr(reply, "id", None) if reply is not None else None

        cache.insert_reply(
            self.conn,
            platform=PLATFORM,
            source_id=item_id,
            post_id=post_id,
            requesting_user=author_name,
            image_sha256=sha,
            reply_id=reply_id,
            response_text=reply_md,
            confidence=analysis.overall_confidence,
        )
        _safe_mark_read(item)
        log.info("replied to %s with %d items", item_id, len(analysis.items))

    def _handle_message(self, item: Any) -> None:
        item_id = item.id
        if cache.reply_exists(self.conn, PLATFORM, item_id):
            _safe_mark_read(item)
            return

        author_name = _author_name(item)
        text = _message_text(item)
        intent = _classify_message(text)

        if intent == "opt_out":
            if author_name:
                cache.record_opt_out(self.conn, PLATFORM, author_name)
            reply_text = OPT_OUT_REPLY
        elif intent == "opt_in":
            if author_name:
                cache.record_opt_in(self.conn, PLATFORM, author_name)
            reply_text = OPT_IN_REPLY
        else:
            reply_text = HELP_REPLY

        reply = item.reply(reply_text) if hasattr(item, "reply") else None
        reply_id = getattr(reply, "id", None) if reply is not None else None

        cache.insert_reply(
            self.conn,
            platform=PLATFORM,
            source_id=item_id,
            post_id=None,
            requesting_user=author_name,
            image_sha256=None,
            reply_id=reply_id,
            response_text=reply_text,
            confidence=None,
        )
        _safe_mark_read(item)
        log.info("handled DM from %s as %s", author_name, intent)


OPT_OUT_REPLY = (
    "You're opted out. I won't reply to mentions involving you on Reddit. "
    "Reply 'opt in' anytime to undo."
)

OPT_IN_REPLY = "You're opted back in. I'll respond to mentions normally."

HELP_REPLY = (
    "I'm Outfindr, a bot that identifies clothing in images. "
    "Mention me with `u/outfindr` in a comment on a post that contains an image "
    "and I'll reply with a breakdown of the outfit and shopping search links. "
    "Reply 'opt out' to this DM to be ignored, 'opt in' to undo."
)

_OPT_OUT_PHRASES = ("opt out", "optout", "unsubscribe", "stop")
_OPT_IN_PHRASES = ("opt in", "optin", "subscribe", "resubscribe")


def _classify_message(text: str) -> str:
    lowered = text.lower()
    if any(p in lowered for p in _OPT_OUT_PHRASES):
        return "opt_out"
    if any(p in lowered for p in _OPT_IN_PHRASES):
        return "opt_in"
    return "help"


def _message_text(item: Any) -> str:
    parts = [
        getattr(item, "subject", "") or "",
        getattr(item, "body", "") or "",
    ]
    return " ".join(parts).strip()


def resolve_image_url(submission: Any) -> str | None:
    """Return a direct image URL for the submission, or None if we can't get one."""
    url = getattr(submission, "url", None) or ""
    lower = url.lower()

    if any(lower.endswith(s) for s in DIRECT_IMAGE_SUFFIXES):
        return url

    if any(host in lower for host in DIRECT_IMAGE_HOSTS):
        return url

    preview = getattr(submission, "preview", None)
    if isinstance(preview, dict):
        images = preview.get("images") or []
        if images:
            source = images[0].get("source") or {}
            preview_url = source.get("url")
            if preview_url:
                return _decode_html_amp(preview_url)

    media_metadata = getattr(submission, "media_metadata", None)
    if isinstance(media_metadata, dict):
        for entry in media_metadata.values():
            if not isinstance(entry, dict):
                continue
            source = entry.get("s") or {}
            for key in ("u", "gif", "mp4"):
                candidate = source.get(key)
                if candidate and candidate.lower().endswith(DIRECT_IMAGE_SUFFIXES):
                    return _decode_html_amp(candidate)

    return None


def download_image(url: str) -> DownloadedImage | None:
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").split(";")[0].strip()
            if content_type not in vision.ALLOWED_CONTENT_TYPES:
                log.info("unsupported content type %r at %s", content_type, url)
                return None
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    log.info("image too large at %s (>%d bytes)", url, MAX_IMAGE_BYTES)
                    return None
                chunks.append(chunk)
            return DownloadedImage(data=b"".join(chunks), content_type=content_type)
    except requests.RequestException as exc:
        log.info("download error for %s: %s", url, exc)
        return None


def _author_name(item: Any) -> str | None:
    author = getattr(item, "author", None)
    return getattr(author, "name", None) if author else None


def _decode_html_amp(url: str) -> str:
    return url.replace("&amp;", "&")


def _safe_mark_read(item: Any) -> None:
    mark = getattr(item, "mark_read", None)
    if callable(mark):
        try:
            mark()
        except Exception:
            log.exception("failed to mark item read")


def _record_skip(conn: sqlite3.Connection, item: Any, post_id: str | None, reason: str) -> None:
    cache.insert_reply(
        conn,
        platform=PLATFORM,
        source_id=item.id,
        post_id=post_id,
        requesting_user=_author_name(item),
        image_sha256=None,
        reply_id=None,
        response_text=f"[skipped: {reason}]",
        confidence=None,
    )
