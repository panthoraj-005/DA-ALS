"""
Gunicorn configuration for production.

    gunicorn -c gunicorn_conf.py main:app

Worker count is the one setting that needs thought here. Each worker loads its
own copy of every model — Florence-2 alone is roughly 1 GB in fp32 — so workers
multiply memory, they do not share it. Two workers is a sane default for the
CNN-only fast path on a 4 GB box. If you enable Florence-2, run ONE worker unless
you have measured the memory headroom.
"""

from __future__ import annotations

import multiprocessing
import os


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


bind = os.getenv("BIND", "0.0.0.0:8000")

# Not (2 * cores + 1): these workers hold models, not sockets.
_default_workers = 1 if os.getenv("FLORENCE_DEFAULT", "0") in {"1", "true"} else 2
workers = _int("WEB_CONCURRENCY", min(_default_workers, max(1, multiprocessing.cpu_count())))
worker_class = "uvicorn.workers.UvicornWorker"

# One inference can hold a worker for tens of seconds on CPU. The timeout has to
# clear the slowest legitimate request or Gunicorn will kill mid-prediction.
timeout = _int("GUNICORN_TIMEOUT", 120)
graceful_timeout = _int("GUNICORN_GRACEFUL_TIMEOUT", 60)
keepalive = _int("GUNICORN_KEEPALIVE", 5)

# Models take seconds to load; give each worker room to boot before the arbiter
# decides it has hung.
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None

max_requests = _int("GUNICORN_MAX_REQUESTS", 0)  # 0 = never recycle
max_requests_jitter = _int("GUNICORN_MAX_REQUESTS_JITTER", 0)

# The app configures its own JSON logging; let it own stdout.
accesslog = None
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()

preload_app = False  # each worker must load models AFTER fork, not share a handle


def on_starting(server):
    server.log.info("ALS screening API starting with %s worker(s)", workers)


def worker_exit(server, worker):
    server.log.info("worker %s exited", worker.pid)
