from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from outfindr.core import db
from outfindr.core.models import OutfitAnalysis

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_analysis_dict() -> dict:
    return json.loads((FIXTURES / "sample_outfit_analysis.json").read_text())


@pytest.fixture
def sample_analysis(sample_analysis_dict) -> OutfitAnalysis:
    return OutfitAnalysis.from_dict(sample_analysis_dict)


@pytest.fixture
def sample_image_bytes() -> bytes:
    return (FIXTURES / "sample_image.jpg").read_bytes()


@pytest.fixture
def db_conn(tmp_path) -> sqlite3.Connection:
    conn = db.connect(str(tmp_path / "test.db"))
    db.init_db(conn)
    yield conn
    conn.close()
