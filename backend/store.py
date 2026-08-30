"""
Prediction store, shared across Gunicorn workers.

An in-process dict breaks the moment there is more than one worker: worker A
serves /predict, worker B gets the /report/{id} and 404s. So a result is written
to OUTPUT_DIR as `{id}.json` next to the images it references, and any worker can
rebuild the PDF from it.

This is deliberately a flat directory of small JSON files, not a database. Results
are transient — they expire with their images — and adding Postgres to hold two
minutes of state would be the wrong trade. If you need predictions to persist,
swap this module; nothing else touches the filesystem for results.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import config

log = logging.getLogger("als.store")

SUFFIX = ".result.json"


def _path(record_id: str) -> Path:
    # ids are hex from uuid4; refuse anything else rather than trust the caller
    if not record_id.isalnum() or len(record_id) > 64:
        raise ValueError("invalid record id")
    return config.OUTPUT_DIR / f"{record_id}{SUFFIX}"


def save(result: dict) -> dict:
    """Persist a result and return the public (underscore-free) view of it."""
    public = {k: v for k, v in result.items() if not k.startswith("_")}
    record = dict(public)
    record["_created"] = time.time()

    try:
        path = _path(result["id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(path)  # atomic, so a concurrent reader never sees half a file
    except (OSError, ValueError) as exc:
        log.warning("could not persist result %s: %s", result.get("id"), exc)

    return public


def load(record_id: str) -> dict | None:
    try:
        path = _path(record_id)
    except ValueError:
        return None

    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read result %s: %s", record_id, exc)
        return None


def sweep(ttl_minutes: int | None = None) -> int:
    """
    Delete generated images, PDFs and result records past their TTL.

    Returns the number of files removed. Safe to call from several workers at
    once — a file another worker already unlinked is not an error.
    """
    ttl = config.ARTIFACT_TTL_MINUTES if ttl_minutes is None else ttl_minutes
    if ttl <= 0:
        return 0

    cutoff = time.time() - ttl * 60
    removed = 0

    try:
        entries = list(config.OUTPUT_DIR.iterdir())
    except OSError:
        return 0

    for path in entries:
        if path.suffix not in {".png", ".pdf", ".json", ".tmp"} and not path.name.endswith(SUFFIX):
            continue
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            pass  # another worker got there first
        except OSError as exc:
            log.debug("sweep skipped %s: %s", path.name, exc)

    if removed:
        log.info("swept expired artifacts", extra={"removed": removed, "ttl_minutes": ttl})

    return removed


def purge_all() -> int:
    """Remove every generated artifact. Used on shutdown when configured."""
    removed = 0
    try:
        entries = list(config.OUTPUT_DIR.iterdir())
    except OSError:
        return 0

    for path in entries:
        if path.suffix not in {".png", ".pdf", ".json", ".tmp"} and not path.name.endswith(SUFFIX):
            continue
        try:
            if path.is_file():
                path.unlink()
                removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return removed
