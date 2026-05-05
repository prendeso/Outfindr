"""FastAPI eval / preview UI for Outfindr.

Single-page upload form. Accepts a multipart file or a pasted URL, runs the
core vision pipeline, and renders a result page. Uses the shared SQLite
vision cache, so re-submitting the same image is free.

Imports `core` only — same invariant as the Reddit adapter.
"""
from __future__ import annotations

import base64
import logging
import os
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core import cache, config, db, search_links, vision

DOWNLOAD_TIMEOUT_SEC = 15
MAX_IMAGE_BYTES = 10 * 1024 * 1024

log = logging.getLogger(__name__)


def _resource_path(*parts: str) -> Path:
    """Locate a path inside the installed package (works for editable installs)."""
    return Path(str(resources.files("outfindr.web").joinpath(*parts)))


_TEMPLATES_DIR = _resource_path("templates")
_STATIC_DIR = _resource_path("static")

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _connect() -> sqlite3.Connection:
    path = os.environ.get("DATABASE_PATH", "./outfindr-dev.db")
    conn = db.connect(path)
    db.init_db(conn)
    return conn


def create_app() -> FastAPI:
    app = FastAPI(title="Outfindr")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        return templates.TemplateResponse(request, "index.html", {})

    @app.post("/identify", response_class=HTMLResponse)
    async def identify(
        request: Request,
        file: UploadFile | None = None,
        url: str = Form(default=""),
    ) -> Any:
        image_bytes, content_type, image_src = await _load_input(file, url)

        conn = _connect()
        try:
            sha = cache.sha256_bytes(image_bytes)
            analysis = cache.get(conn, sha, config.VISION_MODEL_ID, config.PROMPT_VERSION)
            cache_hit = analysis is not None
            if analysis is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise HTTPException(
                        status_code=500,
                        detail="ANTHROPIC_API_KEY is not set on the server.",
                    )
                try:
                    analysis = vision.analyze_outfit(
                        image_bytes,
                        content_type=content_type,
                        api_key=api_key,
                    )
                except vision.VisionParseError as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                cache.put(conn, sha, config.VISION_MODEL_ID, config.PROMPT_VERSION, analysis)
        finally:
            conn.close()

        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "analysis": analysis,
                "image_src": image_src,
                "sha": sha,
                "cache_hit": cache_hit,
                "links_for": search_links.links_for,
            },
        )

    @app.get("/history", response_class=HTMLResponse)
    def history(request: Request) -> Any:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT image_sha256, model_id, prompt_version, analysis_json, created "
                "FROM vision_cache ORDER BY created DESC LIMIT 50"
            ).fetchall()
        finally:
            conn.close()

        from ..core.models import OutfitAnalysis

        entries = []
        for row in rows:
            try:
                analysis = OutfitAnalysis.from_json(row["analysis_json"])
            except Exception:
                continue
            entries.append(
                {
                    "sha": row["image_sha256"],
                    "sha_short": row["image_sha256"][:12],
                    "created": row["created"],
                    "item_count": len(analysis.items),
                    "overall_style": analysis.overall_style,
                    "overall_confidence": analysis.overall_confidence,
                }
            )
        return templates.TemplateResponse(request, "history.html", {"entries": entries})

    return app


async def _load_input(
    file: UploadFile | None,
    url: str,
) -> tuple[bytes, str, str]:
    """Return (image_bytes, content_type, image_src_for_template)."""
    has_file = file is not None and (file.filename or "")
    has_url = bool(url and url.strip())

    if has_file == has_url:
        # both empty, or both provided
        if not has_file and not has_url:
            raise HTTPException(status_code=400, detail="Provide a file or a URL.")
        raise HTTPException(
            status_code=400,
            detail="Provide a file OR a URL, not both.",
        )

    if has_file:
        content_type = (file.content_type or "").split(";")[0].strip()
        if content_type not in vision.ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {content_type or 'unknown'}",
            )
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="Image larger than 10MB.")
        b64 = base64.standard_b64encode(data).decode("ascii")
        return data, content_type, f"data:{content_type};base64,{b64}"

    clean_url = url.strip()
    try:
        resp = requests.get(clean_url, timeout=DOWNLOAD_TIMEOUT_SEC)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {exc}") from exc

    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip()
    if content_type not in vision.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type from URL: {content_type or 'unknown'}",
        )
    if len(resp.content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image at URL is larger than 10MB.")
    return resp.content, content_type, clean_url


app = create_app()
