"""Worker entrypoint. Wires settings + DB + Reddit adapter and runs forever."""
from __future__ import annotations

import logging

from .adapters.reddit_bot import RedditAdapter
from .core import db
from .core.config import Settings


def main() -> None:
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    conn = db.connect(settings.database_path)
    db.init_db(conn)
    adapter = RedditAdapter(settings=settings, conn=conn)
    adapter.run_forever()


if __name__ == "__main__":
    main()
