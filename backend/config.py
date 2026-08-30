"""
Configuration for the ALS screening backend.

Everything that used to be a hard-coded Colab path in the notebook lives here and
is overridable through environment variables (see .env.example at the repo root).
Nothing in this file touches Google Drive.
"""

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --- load .env if python-dotenv is available (optional dependency) ------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


def _resolve_path(env_val: str | None, default: Path) -> Path:
    if not env_val:
        return default
    p = Path(env_val).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# --- model artifacts ---------------------------------------------------------
MODEL_DIR = _resolve_path(os.getenv("MODEL_DIR"), PROJECT_ROOT / "models")

# The notebook saves BOTH `cnn_best.pt` (best val_loss checkpoint) and
# `cnn_model.pt` / `emg_cnn.pt` (last epoch). We prefer the BEST checkpoint.
CNN_CANDIDATES = [
    Path(p).expanduser() if os.path.isabs(p) else MODEL_DIR / p
    for p in os.getenv("CNN_PATH", "cnn_best.pt,cnn_model.pt,emg_cnn.pt").split(",")
    if p.strip()
]

FLORENCE_DIR = _resolve_path(os.getenv("FLORENCE_DIR"), MODEL_DIR / "florence_finetuned")
FLORENCE_HEAD_PATH = _resolve_path(os.getenv("FLORENCE_HEAD_PATH"), MODEL_DIR / "florence_als_head.pt")
META_PATH = _resolve_path(os.getenv("META_PATH"), MODEL_DIR / "meta_learner.pkl")

SAMPLE_DIR = _resolve_path(os.getenv("SAMPLE_DIR"), PROJECT_ROOT / "sample_signals")
OUTPUT_DIR = _resolve_path(os.getenv("OUTPUT_DIR"), PROJECT_ROOT / "outputs")


# --- signal / preprocessing (must match the notebook exactly) ----------------
# FS is the sampling rate assumed by the bandpass filter. The trained artifacts
# were produced with FS=1000; changing it changes preprocessing and therefore
# invalidates the weights. Only change it if you retrain.
FS = _env_float("FS", 1000.0)
BANDPASS_LOW = _env_float("BANDPASS_LOW", 5.0)
BANDPASS_HIGH = _env_float("BANDPASS_HIGH", 450.0)
BANDPASS_ORDER = _env_int("BANDPASS_ORDER", 4)

SIGNAL_LENGTH = _env_int("SIGNAL_LENGTH", 23437)  # X_train.shape[1] in the notebook
LENGTH_TOLERANCE = _env_int("LENGTH_TOLERANCE", 0)  # 0 = exact length required
N_SEGMENTS = _env_int("N_SEGMENTS", 10)
FLORENCE_EMBED_DIM = _env_int("FLORENCE_EMBED_DIM", 768)
CNN_FEATURE_DIM = _env_int("CNN_FEATURE_DIM", 64)
META_FEATURE_DIM = 3 + CNN_FEATURE_DIM + 2 + FLORENCE_EMBED_DIM + 1  # 838


# --- runtime -----------------------------------------------------------------
DEVICE = os.getenv("DEVICE", "auto")  # auto | cpu | cuda
SEED = _env_int("SEED", 42)

# Florence-2 is heavy. On CPU a single signal takes ~10-30s, so the fast path
# (CNN only) is the default and the VLM is opt-in per request.
ENABLE_FLORENCE = _env_bool("ENABLE_FLORENCE", True)  # may the VLM be used at all
FLORENCE_DEFAULT = _env_bool("FLORENCE_DEFAULT", False)  # used when not requested
LOAD_FLORENCE_AT_STARTUP = _env_bool("LOAD_FLORENCE_AT_STARTUP", True)

# LLM explanation smoothing. Supports Gemini API (fast, 0 extra RAM) or local HuggingFace.
ENABLE_LLM = _env_bool("ENABLE_LLM", True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "local").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.7-flash" if LLM_PROVIDER == "gemini" else "Qwen/Qwen2.5-1.5B-Instruct")

# Demo mode lets the whole app run without trained artifacts (random weights).
# Every response is flagged `demo_mode: true` and the UI shows a loud banner.
DEMO_MODE = _env_bool("DEMO_MODE", False)


# --- server ------------------------------------------------------------------
ENV = os.getenv("ENV", "development").strip().lower()
IS_PRODUCTION = ENV in {"production", "prod"}

MAX_UPLOAD_MB = _env_float("MAX_UPLOAD_MB", 25.0)
MAX_BATCH_SIZE = _env_int("MAX_BATCH_SIZE", 16)
ARTIFACT_TTL_MINUTES = _env_int("ARTIFACT_TTL_MINUTES", 120)
# How often the background worker sweeps expired images, PDFs and results.
SWEEP_INTERVAL_SECONDS = _env_int("SWEEP_INTERVAL_SECONDS", 300)
# Delete every generated artifact on shutdown (containers restart clean).
PURGE_ON_SHUTDOWN = _env_bool("PURGE_ON_SHUTDOWN", False)

CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

# --- logging -----------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# JSON in production for log collectors; human-readable locally.
JSON_LOGS = _env_bool("JSON_LOGS", IS_PRODUCTION)

# --- rate limiting -----------------------------------------------------------
# Per client IP, per worker process. See middleware.py for why that caveat matters.
RATE_LIMIT = _env_int("RATE_LIMIT", 20)
RATE_LIMIT_WINDOW = _env_int("RATE_LIMIT_WINDOW", 60)

# --- security ----------------------------------------------------------------
ENABLE_HSTS = _env_bool("ENABLE_HSTS", IS_PRODUCTION)
TRUSTED_HOSTS = [h.strip() for h in os.getenv("TRUSTED_HOSTS", "").split(",") if h.strip()]

# Self-hosted bundle, same-origin images, Google Fonts. No third-party scripts.
CONTENT_SECURITY_POLICY = os.getenv(
    "CONTENT_SECURITY_POLICY",
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com data:; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'",
)

# --- frontend ----------------------------------------------------------------
SERVE_FRONTEND = _env_bool("SERVE_FRONTEND", True)
FRONTEND_DIR = _resolve_path(os.getenv("FRONTEND_DIR"), PROJECT_ROOT / "frontend" / "dist")

DISCLAIMER = (
    "Research and screening aid only. This result is produced by an experimental "
    "machine-learning pipeline, is not a medical diagnosis, and must be confirmed "
    "by a qualified clinician before any clinical decision is made."
)

# Feature names for SHAP, in the exact order the meta-learner was trained on.
FEATURE_NAMES = (
    ["CNN_Prob", "CNN_Confidence", "CNN_Agreement"]
    + [f"CNN_feat_{i}" for i in range(CNN_FEATURE_DIM)]
    + ["Florence_Prob", "Florence_Confidence"]
    + [f"Florence_emb_{i}" for i in range(FLORENCE_EMBED_DIM)]
    + ["CNN_Florence_Agreement"]
)

# Human-readable labels for the feature families, used in the explanation text.
FEATURE_FAMILY_LABEL = {
    "CNN_Prob": "the 1-D CNN's own ALS probability",
    "CNN_Confidence": "how far the CNN's probability sits from the 0.5 decision boundary",
    "CNN_Agreement": "whether the CNN was confidently on one side of the boundary",
    "Florence_Prob": "the Florence-2 vision model's ALS probability",
    "Florence_Confidence": "how decisive the Florence-2 vision model was",
    "CNN_Florence_Agreement": "whether the CNN and the vision model agreed",
}


def resolve_device() -> str:
    if DEVICE in {"cpu", "cuda"}:
        return DEVICE
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def feature_family_label(name: str) -> str:
    if name in FEATURE_FAMILY_LABEL:
        return FEATURE_FAMILY_LABEL[name]
    if name.startswith("CNN_feat_"):
        return f"a learned waveform feature from the CNN ({name})"
    if name.startswith("Florence_emb_"):
        return f"a learned image feature from the vision model ({name})"
    return name


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
