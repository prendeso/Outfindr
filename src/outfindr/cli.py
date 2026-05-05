"""Local CLI: identify outfit from a local file or URL. Prints JSON to stdout."""
from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path

import requests

from .core import vision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Identify the outfit in an image.")
    parser.add_argument("source", help="path to a local image file, or an image URL")
    parser.add_argument(
        "--query",
        default=None,
        help="optional free-text question (e.g. 'the yellow jacket', 'second from left')",
    )
    args = parser.parse_args(argv)

    image_bytes, content_type = _load(args.source)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    analysis = vision.analyze_outfit(
        image_bytes,
        content_type=content_type,
        user_query=args.query,
        api_key=api_key,
    )
    print(analysis.to_json())
    return 0


def _load(source: str) -> tuple[bytes, str]:
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        ct = (resp.headers.get("content-type") or "").split(";")[0].strip()
        if ct not in vision.ALLOWED_CONTENT_TYPES:
            raise SystemExit(f"unsupported content type from URL: {ct!r}")
        return resp.content, ct

    path = Path(source)
    if not path.is_file():
        raise SystemExit(f"file not found: {source}")
    ct, _ = mimetypes.guess_type(str(path))
    if ct not in vision.ALLOWED_CONTENT_TYPES:
        raise SystemExit(f"unsupported file type: {ct!r}")
    return path.read_bytes(), ct


if __name__ == "__main__":
    raise SystemExit(main())
