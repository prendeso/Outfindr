"""Environment-driven config. Read once at startup; never read os.environ elsewhere."""
from __future__ import annotations

import os
from dataclasses import dataclass

VISION_MODEL_ID = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "vision_system_v2"


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_username: str
    reddit_password: str
    reddit_user_agent: str
    bot_username: str
    database_path: str
    confidence_floor: float
    daily_reply_budget: int
    log_level: str
    amazon_affiliate_tag: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=_required("ANTHROPIC_API_KEY"),
            reddit_client_id=_required("REDDIT_CLIENT_ID"),
            reddit_client_secret=_required("REDDIT_CLIENT_SECRET"),
            reddit_username=_required("REDDIT_USERNAME"),
            reddit_password=_required("REDDIT_PASSWORD"),
            reddit_user_agent=_required("REDDIT_USER_AGENT"),
            bot_username=os.environ.get("BOT_USERNAME", "outfindr"),
            database_path=os.environ.get("DATABASE_PATH", "/data/outfindr.db"),
            confidence_floor=float(os.environ.get("CONFIDENCE_FLOOR", "0.55")),
            daily_reply_budget=int(os.environ.get("DAILY_REPLY_BUDGET", "200")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            amazon_affiliate_tag=(os.environ.get("AMAZON_AFFILIATE_TAG") or None),
        )


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value
