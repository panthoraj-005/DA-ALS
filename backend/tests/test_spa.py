"""
SPA serving rules.

The fallback has to be precise: a browser navigating to a client route gets
index.html, but a missing fingerprinted asset must 404. Serving HTML in place of
a missing bundle turns "the deploy is stale" into an unreadable syntax error in
the console, which is a genuinely hard bug to trace back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("DEMO_MODE", "1")
os.environ.setdefault("ENABLE_FLORENCE", "0")
os.environ.setdefault("JSON_LOGS", "0")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spa import SpaStaticFiles, mount_frontend


def _scope(accept: str = "text/html"):
    return {"headers": [(b"accept", accept.encode())]}


@pytest.mark.parametrize(
    ("path", "accept", "expected"),
    [
        ("some/deep/route", "text/html,application/xhtml+xml", True),
        ("", "text/html", True),
        ("patients/42", "text/html", True),
        ("index.html", "text/html", True),
        ("assets/index-abc123.js", "*/*", False),
        ("assets/index-abc123.css", "text/css", False),
        ("logo.png", "image/png", False),
        ("data.json", "application/json", False),
        ("some/route", "application/json", False),
    ],
)
def test_only_navigations_fall_back(path, accept, expected):
    assert SpaStaticFiles._wants_document(path, _scope(accept)) is expected


@pytest.fixture
def built_app(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")

    app = FastAPI()
    assert mount_frontend(app, dist) is True
    return app


def test_client_route_serves_the_app(built_app):
    with TestClient(built_app) as client:
        response = client.get("/reports/1", headers={"Accept": "text/html"})
        assert response.status_code == 200
        assert 'id="root"' in response.text
        assert "no-cache" in response.headers["Cache-Control"]


def test_missing_asset_is_a_real_404(built_app):
    with TestClient(built_app) as client:
        assert client.get("/assets/deleted-bundle.js").status_code == 404


def test_hashed_asset_is_immutable(built_app):
    with TestClient(built_app) as client:
        response = client.get("/assets/index-abc123.js")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"


def test_no_build_means_no_mount(tmp_path):
    assert mount_frontend(FastAPI(), tmp_path / "nothing") is False
