"""
Request middleware: context and access logging, security headers, rate limiting.

The rate limiter is a per-process in-memory token bucket. That is deliberate —
it needs no Redis and cannot fail open on a network blip — but it means the
effective limit is `RATE_LIMIT x number of workers`. With the default 2 workers
a limit of 10/min admits up to 20/min. DEPLOYMENT.md says so, and if you need an
exact global limit, enforce it at the reverse proxy instead.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import config
from observability import REQUEST_CTX, client_ip, new_request_id

log = logging.getLogger("als.http")


# ---------------------------------------------------------------------------
# Request context + access log
# ---------------------------------------------------------------------------
class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or new_request_id()
        ip = client_ip(request.headers, request.client.host if request.client else None)

        token = REQUEST_CTX.set(
            {
                "request_id": request_id,
                "client_ip": ip,
                "method": request.method,
                "path": request.url.path,
            }
        )
        request.state.request_id = request_id
        request.state.client_ip = ip

        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            # Liveness probes fire constantly; logging them buries everything else.
            if request.url.path not in {"/health/live", "/favicon.ico"}:
                log.info(
                    "request",
                    extra={"status": status, "latency_ms": latency_ms},
                )
            REQUEST_CTX.reset(token)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    The CSP is written for this app specifically: the SPA's own bundle, inline
    styles React sets on elements, images from our own origin (the generated
    PNGs are same-origin), and Google Fonts. No third-party scripts, no frames,
    no form posts anywhere but here.
    """

    def __init__(self, app, csp: str, hsts: bool):
        super().__init__(app)
        self.csp = csp
        self.hsts = hsts

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        headers = response.headers

        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), interest-cohort=()",
        )
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        headers.setdefault("X-Permitted-Cross-Domain-Policies", "none")

        if self.csp:
            headers.setdefault("Content-Security-Policy", self.csp)

        # Only meaningful over TLS, and harmful to send on plain HTTP in dev.
        if self.hsts:
            headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )

        # A screening result is patient-adjacent. Never let a proxy cache it.
        if request.url.path.startswith(("/predict", "/report", "/health")):
            headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            headers["Pragma"] = "no-cache"

        return response


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
@dataclass
class Bucket:
    tokens: float
    updated: float


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket keyed on client IP. Inference is the expensive thing to
    protect: one /predict with the VLM on can hold a CPU for 30 seconds, so a
    handful of concurrent callers is a denial of service without this.
    """

    def __init__(self, app, limit: int, window_seconds: int, paths: tuple[str, ...]):
        super().__init__(app)
        self.limit = float(limit)
        self.window = float(window_seconds)
        self.rate = self.limit / self.window if self.window > 0 else 0.0
        self.paths = paths
        self.buckets: OrderedDict[str, Bucket] = OrderedDict()
        self.max_keys = 4096

    def _consume(self, key: str) -> tuple[bool, float]:
        now = time.monotonic()
        bucket = self.buckets.get(key)

        if bucket is None:
            bucket = Bucket(tokens=self.limit, updated=now)
            self.buckets[key] = bucket
            if len(self.buckets) > self.max_keys:
                self.buckets.popitem(last=False)  # evict the oldest key
        else:
            self.buckets.move_to_end(key)
            bucket.tokens = min(self.limit, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, bucket.tokens

        retry_after = (1.0 - bucket.tokens) / self.rate if self.rate > 0 else self.window
        return False, retry_after

    async def dispatch(self, request: Request, call_next):
        if self.limit <= 0 or not request.url.path.startswith(self.paths):
            return await call_next(request)

        key = getattr(request.state, "client_ip", None) or (
            request.client.host if request.client else "unknown"
        )
        allowed, value = self._consume(key)

        if not allowed:
            retry_after = max(1, int(value) + 1)
            log.warning("rate limit exceeded", extra={"retry_after_s": retry_after})
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": (
                        f"Too many screening requests. This server accepts "
                        f"{int(self.limit)} every {int(self.window)}s per client. "
                        f"Try again in {retry_after}s."
                    ),
                },
                headers={"Retry-After": str(retry_after)},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(int(self.limit))
        response.headers["X-RateLimit-Remaining"] = str(int(value))
        return response


# ---------------------------------------------------------------------------
# CORS origin sanitising
# ---------------------------------------------------------------------------
def resolve_cors_origins() -> list[str]:
    """
    Parse CORS_ORIGINS strictly. A wildcard is refused outright in production:
    this API returns screening results, and `*` would let any page on the
    internet read them from a logged-in browser.
    """
    origins = [o.strip() for o in config.CORS_ORIGINS if o.strip()]

    if "*" in origins:
        if config.IS_PRODUCTION:
            raise RuntimeError(
                "CORS_ORIGINS contains '*', which is refused when ENV=production. "
                "List the exact frontend origins, e.g. "
                "CORS_ORIGINS=https://als.example.org"
            )
        log.warning("CORS is set to '*' - acceptable in development only")
        return ["*"]

    bad = [o for o in origins if not o.startswith(("http://", "https://"))]
    if bad:
        raise RuntimeError(
            f"CORS_ORIGINS entries must include a scheme. Invalid: {', '.join(bad)}"
        )

    if config.IS_PRODUCTION:
        insecure = [o for o in origins if o.startswith("http://") and "localhost" not in o]
        if insecure:
            log.warning(
                "CORS origins are served over plain HTTP in production: %s",
                ", ".join(insecure),
            )

    return origins
