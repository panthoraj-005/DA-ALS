"""
Structured JSON logging and per-request context.

Every log line is one JSON object so a collector (CloudWatch, Loki, Datadog)
can index it without a grok pattern. Request-scoped fields — request_id,
client_ip, path — ride along on a ContextVar, so a log call deep inside the
pipeline carries them without every function taking a request argument.

Nothing here logs signal contents or file bytes. Uploaded EMG data is patient
data; only its filename, size and shape are ever recorded.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

# default=None, not {}: a mutable ContextVar default is shared across contexts.
REQUEST_CTX: ContextVar[dict | None] = ContextVar("request_ctx", default=None)

# Fields LogRecord always carries; anything else an adapter attached is "extra".
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the request context merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": record.process,
        }

        ctx = REQUEST_CTX.get()
        if ctx:
            payload.update(ctx)

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Readable output for local development."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = REQUEST_CTX.get() or {}
        rid = f" [{ctx['request_id'][:8]}]" if ctx.get("request_id") else ""
        base = (
            f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
            f"{record.levelname:<7}{rid} {record.name} | {record.getMessage()}"
        )
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else PlainFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn duplicates access logs in its own format; we emit our own with
    # latency and request_id, so silence its access logger.
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = False
    for noisy in ("uvicorn.error", "matplotlib", "PIL", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex


def client_ip(headers, fallback: str | None) -> str:
    """
    Behind a proxy the socket address is the proxy. Trust X-Forwarded-For only
    when TRUST_PROXY_HEADERS is on, because a client can forge it otherwise.
    """
    if os.getenv("TRUST_PROXY_HEADERS", "0").strip().lower() in {"1", "true", "yes", "on"}:
        forwarded = headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real = headers.get("x-real-ip")
        if real:
            return real.strip()
    return fallback or "unknown"
