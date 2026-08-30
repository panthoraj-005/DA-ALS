"""
Serving the built React app from FastAPI, with SPA fallback and cache headers.

Vite fingerprints everything under /assets (index-BPKsUMs9.js), so those files
are immutable and can be cached for a year. index.html is the one file that must
never be cached — it is what points at the current fingerprints, and a stale copy
pins a browser to a deleted bundle.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, Response

# Starlette's HTTPException, not FastAPI's: StaticFiles raises the parent class,
# and `except fastapi.HTTPException` would not catch it.
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

log = logging.getLogger("als.spa")

IMMUTABLE = "public, max-age=31536000, immutable"
NO_CACHE = "no-cache, must-revalidate"


class SpaStaticFiles(StaticFiles):
    """
    StaticFiles that returns index.html for unknown paths so client-side routes
    survive a refresh — but only for document requests. A missing asset or a
    stray /api call should 404 honestly rather than being handed a page of HTML
    that the caller will fail to parse.
    """

    @staticmethod
    def _wants_document(path: str, scope) -> bool:
        """
        True only for what looks like a browser navigation. A fingerprinted
        asset that is missing means a stale index.html is pinned to a deleted
        bundle — answering 200 with HTML hides that as a syntax error in the
        console instead of a clean 404.
        """
        normalised = path.lstrip("/")
        if normalised.startswith("assets/") or normalised.startswith("api/") or normalised == "api":
            return False

        suffix = Path(normalised).suffix.lower()
        if suffix and suffix != ".html":
            return False

        headers = {k.lower(): v for k, v in (scope.get("headers") or [])}
        accept = headers.get(b"accept", b"").decode("latin-1")
        return "text/html" in accept or accept.strip() in {"", "*/*"}

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or not self._wants_document(path, scope):
                raise
            return await super().get_response("index.html", scope)

        if response.status_code == 404 and self._wants_document(path, scope):
            return await super().get_response("index.html", scope)
        return response

    def file_response(self, full_path, stat_result, scope, status_code=200) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        name = Path(full_path).name

        if name == "index.html":
            response.headers["Cache-Control"] = NO_CACHE
        elif "/assets/" in str(full_path).replace("\\", "/"):
            response.headers["Cache-Control"] = IMMUTABLE
        else:
            response.headers["Cache-Control"] = "public, max-age=3600"

        return response


def mount_frontend(app, dist_dir: Path) -> bool:
    """Mount the built SPA at /. Returns False (and logs) if there is no build."""
    index = dist_dir / "index.html"
    if not index.is_file():
        log.info(
            "no frontend build at %s - serving the API only "
            "(run `npm run build` in frontend/, or use the Docker image)",
            dist_dir,
        )
        return False

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon(request: Request):  # noqa: ARG001
        icon = dist_dir / "favicon.ico"
        if icon.is_file():
            return FileResponse(icon)
        return Response(status_code=204)

    app.mount("/", SpaStaticFiles(directory=str(dist_dir), html=True), name="spa")
    log.info("serving frontend from %s", dist_dir)
    return True
