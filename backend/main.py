"""
FastAPI application.

Models load once in the lifespan handler. Request handlers never load a model,
never train, and never touch Google Drive.

Development:  uvicorn main:app --reload --port 8000
Production:   gunicorn -c gunicorn_conf.py main:app
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pydantic import BaseModel

import config
import models as model_module
import signal_io
import store
import chat
from middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    resolve_cors_origins,
)
from observability import configure_logging
from pipeline import run_pipeline
from report import build_report
from signal_io import SignalFormatError
from spa import mount_frontend

configure_logging(level=config.LOG_LEVEL, json_logs=config.JSON_LOGS)
log = logging.getLogger("als.api")

# Florence-2 and Qwen are not thread-safe and are memory-hungry: one inference at
# a time per worker process. Concurrency comes from running more workers, which
# is a memory trade — see DEPLOYMENT.md.
INFERENCE_LOCK = asyncio.Lock()

api_router = APIRouter()


async def _sweeper() -> None:
    """Background TTL cleanup. Cancelled on shutdown."""
    interval = max(30, config.SWEEP_INTERVAL_SECONDS)
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await asyncio.to_thread(store.sweep)
            except Exception:
                log.exception("artifact sweep failed")
    except asyncio.CancelledError:
        raise


def _install_signal_logging() -> None:
    """
    Note the signal, then let uvicorn/gunicorn run their own handler. The actual
    cleanup happens in the lifespan shutdown, which the server calls after it
    has drained in-flight requests — killing files here would yank a PDF out
    from under a request that is still streaming it.
    """
    loop = asyncio.get_running_loop()

    def note(sig: signal.Signals) -> None:
        log.info("received %s, draining before shutdown", sig.name)

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            previous = signal.getsignal(sig)

            def handler(s=sig, prev=previous):
                note(s)
                if callable(prev):
                    prev(s, None)

            loop.add_signal_handler(sig, handler)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "starting ALS screening API",
        extra={"env": config.ENV, "version": app.version},
    )

    app.state.registry = await asyncio.to_thread(model_module.load_all)
    app.state.ready = app.state.registry.cnn_ready

    store.sweep()
    _install_signal_logging()
    app.state.sweeper = asyncio.create_task(_sweeper())

    try:
        yield
    finally:
        log.info("shutting down")
        app.state.ready = False

        sweeper = getattr(app.state, "sweeper", None)
        if sweeper is not None:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper

        if config.PURGE_ON_SHUTDOWN:
            removed = await asyncio.to_thread(store.purge_all)
            log.info("purged generated artifacts", extra={"removed": removed})
        else:
            await asyncio.to_thread(store.sweep)

        # Release model memory deterministically rather than at interpreter exit.
        registry = getattr(app.state, "registry", None)
        if registry is not None:
            registry.cnn = None
            registry.florence_model = None
            registry.florence_head = None
            registry.florence_processor = None
            registry.meta_learner = None
            registry.shap_explainer = None
            registry.llm_model = None
            registry.llm_tokenizer = None

        log.info("shutdown complete")


app = FastAPI(
    title="ALS screening API",
    version="1.1.0",
    description=(
        "EMG-based ALS screening pipeline: 1-D CNN, Florence-2 vision model, and an "
        "XGBoost meta-learner, with Grad-CAM and SHAP explainability. "
        "Screening aid for research use. Not a diagnostic device."
    ),
    lifespan=lifespan,
    docs_url=None if config.IS_PRODUCTION else "/docs",
    redoc_url=None,
    openapi_url=None if config.IS_PRODUCTION else "/openapi.json",
)

# Order matters: context first so the rate limiter and the access log both see
# the resolved client IP, headers last so they wrap every response including 429s.
app.add_middleware(
    SecurityHeadersMiddleware,
    csp=config.CONTENT_SECURITY_POLICY,
    hsts=config.ENABLE_HSTS,
)
app.add_middleware(
    RateLimitMiddleware,
    limit=config.RATE_LIMIT,
    window_seconds=config.RATE_LIMIT_WINDOW,
    paths=("/predict",),
)
app.add_middleware(RequestContextMiddleware)

if config.TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config.TRUSTED_HOSTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=resolve_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
    max_age=600,
)


# ---------------------------------------------------------------------------
# Errors: structured JSON, never a stack trace
# ---------------------------------------------------------------------------
@app.exception_handler(SignalFormatError)
async def signal_format_handler(request: Request, exc: SignalFormatError):  # noqa: ARG001
    return JSONResponse(status_code=400, content={"error": "invalid_signal", "detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):  # noqa: ARG001
    log.exception("unhandled error")
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "The server hit an unexpected error. The server log has the details.",
            "request_id": request_id,
        },
    )


def _registry():
    reg = getattr(app.state, "registry", None)
    if reg is None:
        raise HTTPException(status_code=503, detail="Models are still loading. Try again.")
    return reg


def _require_cnn(reg):
    if reg.cnn_ready:
        return
    raise HTTPException(
        status_code=503,
        detail=(
            "The CNN checkpoint is not loaded, so no prediction can be made. Put "
            "cnn_best.pt in the models/ folder (see README), or start the server with "
            "DEMO_MODE=1 to explore the interface with placeholder weights. "
            "GET /health lists every artifact."
        ),
    )


async def _read_upload(file: UploadFile) -> bytes:
    """
    Size is checked before the body reaches memory: the declared size first when
    the client sent one, then a hard read cap so a lying Content-Length cannot
    make us buffer a gigabyte.
    """
    limit = int(config.MAX_UPLOAD_MB * 1024 * 1024)

    declared = getattr(file, "size", None)
    if declared is not None and declared > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File is {declared / 1048576:.1f} MB, over the "
            f"{config.MAX_UPLOAD_MB:.0f} MB upload limit.",
        )

    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {config.MAX_UPLOAD_MB:.0f} MB upload limit.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return data


# ---------------------------------------------------------------------------
# Health and readiness
# ---------------------------------------------------------------------------
@api_router.get("/health/live", tags=["health"])
async def live():
    """Liveness: the process is up and the event loop is turning. Never touches models."""
    return {"status": "alive"}


@api_router.get("/health/ready", tags=["health"])
async def ready():
    """
    Readiness: can this worker actually score a signal? Load balancers use this
    to hold traffic back while Florence-2 is still loading.
    """
    reg = getattr(app.state, "registry", None)
    is_ready = bool(reg and reg.cnn_ready and getattr(app.state, "ready", False))

    body = {
        "status": "ready" if is_ready else "not_ready",
        "cnn_loaded": bool(reg and reg.cnn_ready),
        "florence_loaded": bool(reg and reg.florence_ready),
        "fusion_available": bool(reg and reg.fusion_ready),
        "demo_mode": bool(reg and reg.demo_mode),
    }
    return JSONResponse(status_code=200 if is_ready else 503, content=body)


@api_router.get("/health", tags=["health"])
async def health():
    reg = _registry()

    if reg.demo_mode:
        state = "demo"
    elif reg.cnn_ready and reg.fusion_ready:
        state = "ready"
    elif reg.cnn_ready:
        state = "degraded"
    else:
        state = "unavailable"

    return {
        "status": state,
        "device": reg.device,
        "demo_mode": reg.demo_mode,
        "env": config.ENV,
        "version": app.version,
        "capabilities": {
            "cnn": reg.cnn_ready,
            "florence": reg.florence_ready,
            "meta_learner": reg.meta_ready,
            "full_fusion": reg.fusion_ready,
            "gradcam": reg.cnn_ready,
            "shap": reg.shap_explainer is not None,
            "llm_explanation": reg.llm_model is not None or (config.LLM_PROVIDER == "gemini" and bool(config.GEMINI_API_KEY)),
        },
        "artifacts": reg.status_list(),
        "config": {
            "signal_length": config.SIGNAL_LENGTH,
            "fs": config.FS,
            "bandpass": [config.BANDPASS_LOW, config.BANDPASS_HIGH],
            "n_segments": config.N_SEGMENTS,
            "meta_feature_dim": config.META_FEATURE_DIM,
            "florence_default": config.FLORENCE_DEFAULT,
            "max_upload_mb": config.MAX_UPLOAD_MB,
            "max_batch_size": config.MAX_BATCH_SIZE,
        },
        "disclaimer": config.DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@api_router.get("/samples", tags=["predict"])
async def samples():
    return {"samples": signal_io.list_samples(), "signal_length": config.SIGNAL_LENGTH}


@api_router.post("/predict", tags=["predict"])
async def predict(
    file: UploadFile | None = File(default=None),
    sample: str | None = Form(default=None),
    explain: bool = Form(default=True),
    use_vlm: bool | None = Form(default=None),
    use_llm: bool | None = Form(default=None),
):
    reg = _registry()
    _require_cnn(reg)

    if file is None and not sample:
        raise HTTPException(
            status_code=400,
            detail="Send either a signal file (field 'file') or a bundled sample name "
            "(field 'sample').",
        )

    if file is not None:
        data = await _read_upload(file)
        signal_array = signal_io.read_signal_bytes(data, file.filename or "upload")
        source = file.filename or "upload"
    else:
        path = signal_io.resolve_sample(sample)
        signal_array = signal_io.read_signal_path(path)
        source = path.name

    signal_array = signal_io.validate_length(signal_array)

    log.info(
        "scoring signal",
        extra={"source": source, "explain": explain, "use_vlm": use_vlm, "use_llm": use_llm},
    )

    async with INFERENCE_LOCK:
        result = await asyncio.to_thread(
            run_pipeline,
            reg,
            signal_array,
            explain=explain,
            use_vlm=use_vlm,
            use_llm=use_llm,
            source_name=source,
        )

    log.info(
        "scored signal",
        extra={
            "record_id": result["id"],
            "prediction": result["final_prediction"],
            "confidence": result["final_confidence"],
            "fusion": result["fusion"],
            "inference_ms": result["timings"].get("total_ms"),
        },
    )

    return store.save(result)


@api_router.post("/predict/batch", tags=["predict"])
async def predict_batch(
    files: list[UploadFile] = File(...),
    explain: bool = Form(default=False),
    use_vlm: bool | None = Form(default=None),
):
    reg = _registry()
    _require_cnn(reg)

    if len(files) > config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch of {len(files)} exceeds the limit of {config.MAX_BATCH_SIZE}.",
        )

    results, failures = [], []

    for upload in files:
        name = upload.filename or "upload"
        try:
            data = await _read_upload(upload)
            signal_array = signal_io.validate_length(signal_io.read_signal_bytes(data, name))
            async with INFERENCE_LOCK:
                result = await asyncio.to_thread(
                    run_pipeline,
                    reg,
                    signal_array,
                    explain=explain,
                    use_vlm=use_vlm,
                    source_name=name,
                )
            results.append(store.save(result))
        except (SignalFormatError, HTTPException) as exc:
            failures.append({"source": name, "detail": getattr(exc, "detail", str(exc))})
        except Exception as exc:
            log.exception("batch item failed", extra={"source": name})
            failures.append({"source": name, "detail": f"Inference failed: {exc}"})

    als = sum(1 for r in results if r["final_prediction"] == "ALS")
    agreed = [r["models_agree"] for r in results if r["models_agree"] is not None]

    return {
        "results": results,
        "failures": failures,
        "summary": {
            "scored": len(results),
            "failed": len(failures),
            "als": als,
            "normal": len(results) - als,
            "mean_confidence": (
                round(sum(r["final_confidence"] for r in results) / len(results), 4)
                if results
                else None
            ),
            "agreement_rate": (
                round(100.0 * sum(1 for a in agreed if a) / len(agreed), 2) if agreed else None
            ),
        },
        "disclaimer": config.DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
@api_router.get("/report/{record_id}", tags=["artifacts"])
async def report(record_id: str, download: bool = Query(default=True)):
    result = store.load(record_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No prediction with that id. Results expire with their images "
            f"({config.ARTIFACT_TTL_MINUTES} minutes) — run the signal again.",
        )

    pdf_path = config.OUTPUT_DIR / f"{record_id}_report.pdf"
    if not pdf_path.exists():
        await asyncio.to_thread(build_report, result, pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"als_report_{record_id}.pdf" if download else None,
        headers={"Cache-Control": "no-store, private"},
    )


@api_router.get("/images/{name}", tags=["artifacts"])
async def image(name: str):
    safe = (config.OUTPUT_DIR / Path(name).name).resolve()
    if config.OUTPUT_DIR.resolve() not in safe.parents or not safe.is_file():
        raise HTTPException(status_code=404, detail="Image not found or already expired.")
    if safe.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Image not found.")

    return FileResponse(
        safe,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=600"},
    )


class ChatMessageItem(BaseModel):
    role: str
    content: str


class ChatRequestBody(BaseModel):
    message: str
    history: list[ChatMessageItem] = []
    context: dict | None = None


@api_router.post("/chat", tags=["assistant"])
async def chat_endpoint(body: ChatRequestBody):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history_dicts = [{"role": h.role, "content": h.content} for h in body.history]
    reply, suggestions = await asyncio.to_thread(
        chat.generate_chat_response,
        message=body.message.strip(),
        history=history_dicts,
        context=body.context,
    )
    return {"reply": reply, "suggestions": suggestions}


@api_router.get("/api", include_in_schema=False)
async def api_root():
    return {
        "name": "ALS screening API",
        "version": app.version,
        "health": "/health",
        "docs": "/docs" if not config.IS_PRODUCTION else None,
        "disclaimer": config.DISCLAIMER,
    }


# Mount API routes at both root and /api prefix so all client configurations work
app.include_router(api_router)
app.include_router(api_router, prefix="/api")

# Mounted last: the SPA claims "/", so every API route above must already exist.
FRONTEND_MOUNTED = False
if config.SERVE_FRONTEND:
    FRONTEND_MOUNTED = mount_frontend(app, config.FRONTEND_DIR)

if not FRONTEND_MOUNTED:

    @app.get("/", include_in_schema=False)
    async def root():
        return await api_root()
