"""
API tests. Run with `pytest` from the backend directory.

Everything runs in DEMO_MODE with placeholder weights: these assert that the
service behaves, not that the model is accurate. Accuracy is a property of the
trained artifacts and is not testable here.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("DEMO_MODE", "1")
os.environ.setdefault("ENABLE_FLORENCE", "0")
os.environ.setdefault("ENABLE_LLM", "0")
os.environ.setdefault("SERVE_FRONTEND", "0")
os.environ.setdefault("JSON_LOGS", "0")
os.environ.setdefault("RATE_LIMIT", "0")  # off by default; one test turns it on

from fastapi.testclient import TestClient

import config
import main


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def signal_bytes() -> bytes:
    buf = io.BytesIO()
    rng = np.random.default_rng(3)
    np.save(buf, rng.normal(0, 1, config.SIGNAL_LENGTH).astype(np.float32))
    return buf.getvalue()


# --- health ---------------------------------------------------------------
def test_liveness_does_not_touch_models(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_reports_cnn(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["cnn_loaded"] is True
    assert body["demo_mode"] is True


def test_health_lists_every_artifact(client):
    body = client.get("/health").json()
    assert body["status"] == "demo"
    names = {a["name"] for a in body["artifacts"]}
    assert {"cnn", "florence", "meta_learner"} <= names
    assert body["config"]["meta_feature_dim"] == 838
    assert "not a medical diagnosis" in body["disclaimer"]


# --- security -------------------------------------------------------------
def test_security_headers_present(client):
    headers = client.get("/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_results_are_never_cached(client):
    assert "no-store" in client.get("/health").headers["Cache-Control"]


def test_every_response_carries_a_request_id(client):
    assert client.get("/health/live").headers["X-Request-ID"]


def test_wildcard_cors_refused_in_production(monkeypatch):
    import middleware

    monkeypatch.setattr(config, "CORS_ORIGINS", ["*"])
    monkeypatch.setattr(config, "IS_PRODUCTION", True)
    with pytest.raises(RuntimeError, match="refused when ENV=production"):
        middleware.resolve_cors_origins()


def test_cors_origins_require_a_scheme(monkeypatch):
    import middleware

    monkeypatch.setattr(config, "CORS_ORIGINS", ["als.example.org"])
    monkeypatch.setattr(config, "IS_PRODUCTION", False)
    with pytest.raises(RuntimeError, match="must include a scheme"):
        middleware.resolve_cors_origins()


# --- prediction -----------------------------------------------------------
def test_predict_from_upload(client, signal_bytes):
    response = client.post(
        "/predict",
        files={"file": ("unit.npy", signal_bytes, "application/octet-stream")},
        data={"explain": "true", "use_vlm": "false"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["final_prediction"] in {"ALS", "Normal"}
    assert 0.0 <= body["final_confidence"] <= 1.0
    assert body["fusion"] == "cnn_only"
    assert body["florence_probability"] is None
    assert body["demo_mode"] is True
    assert body["gradcam_image_url"]
    assert "not a medical diagnosis" in body["explanation"]


def test_predict_is_deterministic(client, signal_bytes):
    def score():
        return client.post(
            "/predict",
            files={"file": ("unit.npy", signal_bytes, "application/octet-stream")},
            data={"explain": "false", "use_vlm": "false"},
        ).json()["cnn_probability"]

    assert score() == score()


def test_predict_from_bundled_sample(client):
    names = [s["name"] for s in client.get("/samples").json()["samples"]]
    if not names:
        pytest.skip("no bundled samples on this checkout")

    response = client.post("/predict", data={"sample": names[0], "explain": "false"})
    assert response.status_code == 200


def test_wrong_length_is_a_clear_400(client):
    buf = io.BytesIO()
    np.save(buf, np.zeros(1234, dtype=np.float32))

    response = client.post(
        "/predict", files={"file": ("short.npy", buf.getvalue(), "application/octet-stream")}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "1,234" in detail and str(config.SIGNAL_LENGTH) in detail.replace(",", "")


def test_unsupported_extension_rejected(client):
    response = client.post(
        "/predict", files={"file": ("scan.dcm", b"not a signal", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_signal"


def test_empty_upload_rejected(client):
    response = client.post(
        "/predict", files={"file": ("empty.npy", b"", "application/octet-stream")}
    )
    assert response.status_code == 400


def test_missing_input_rejected(client):
    assert client.post("/predict", data={"explain": "false"}).status_code == 400


def test_sample_path_traversal_blocked(client):
    response = client.post("/predict", data={"sample": "../backend/main.py"})
    assert response.status_code == 400


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)
    response = client.post(
        "/predict", files={"file": ("big.npy", b"x" * 20_000, "application/octet-stream")}
    )
    assert response.status_code == 413


# --- artifacts ------------------------------------------------------------
def test_report_and_image_round_trip(client, signal_bytes):
    body = client.post(
        "/predict",
        files={"file": ("unit.npy", signal_bytes, "application/octet-stream")},
        data={"explain": "true"},
    ).json()

    pdf = client.get(f"/report/{body['id']}")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")

    image = client.get(body["signal_image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"


def test_report_survives_a_different_worker(client, signal_bytes):
    """The store is on disk, so a worker that never saw the prediction can serve it."""
    import store

    body = client.post(
        "/predict",
        files={"file": ("unit.npy", signal_bytes, "application/octet-stream")},
        data={"explain": "false"},
    ).json()

    reloaded = store.load(body["id"])
    assert reloaded is not None
    assert reloaded["final_prediction"] == body["final_prediction"]


def test_unknown_report_is_404_not_a_crash(client):
    response = client.get("/report/deadbeefdeadbeef")
    assert response.status_code == 404
    assert "expire" in response.json()["detail"]


def test_image_traversal_blocked(client):
    assert client.get("/images/..%2f..%2fmain.py").status_code == 404


# --- rate limiting --------------------------------------------------------
def test_rate_limit_returns_429_with_retry_after():
    from middleware import RateLimitMiddleware

    app = main.app
    limiter = RateLimitMiddleware(app, limit=2, window_seconds=60, paths=("/predict",))

    assert limiter._consume("10.0.0.1")[0] is True
    assert limiter._consume("10.0.0.1")[0] is True
    allowed, retry_after = limiter._consume("10.0.0.1")
    assert allowed is False
    assert retry_after > 0

    # a different client is unaffected
    assert limiter._consume("10.0.0.2")[0] is True


# --- store ----------------------------------------------------------------
def test_sweep_removes_expired_artifacts(tmp_path, monkeypatch):
    import store

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    stale = tmp_path / "old_signal.png"
    stale.write_bytes(b"x")
    os.utime(stale, (0, 0))
    fresh = tmp_path / "new_signal.png"
    fresh.write_bytes(b"x")

    removed = store.sweep(ttl_minutes=1)
    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()


def test_store_rejects_a_hostile_id():
    import store

    assert store.load("../../etc/passwd") is None


# --- multi-provider settings and assistant ---------------------------------
def test_get_settings_returns_providers_and_catalog(client):
    res = client.get("/settings")
    assert res.status_code == 200
    data = res.json()
    assert "providers" in data
    provider_ids = {p["id"] for p in data["providers"]}
    assert {"gemini", "openai", "anthropic", "groq", "openrouter", "ollama", "local"}.issubset(provider_ids)
    assert "available_models" in data
    assert "llm_provider" in data


def test_update_settings_multi_provider(client, monkeypatch):
    res = client.post(
        "/settings",
        json={
            "llm_provider": "groq",
            "llm_model": "llama-3.3-70b-versatile",
            "groq_api_key": "gsk_mock_test_key_12345",
            "ollama_base_url": "http://127.0.0.1:11434",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["llm_provider"] == "groq"
    assert data["llm_model"] == "llama-3.3-70b-versatile"
    assert config.GROQ_API_KEY == "gsk_mock_test_key_12345"
    assert config.OLLAMA_BASE_URL == "http://127.0.0.1:11434"


def test_test_key_endpoint_validation(client):
    # Missing key for provider requiring key
    res = client.post("/settings/test-key", json={"provider": "anthropic", "api_key": ""})
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is False
    assert "No API key" in data["message"] or "valid API key" in data["message"]


def test_chat_endpoint_missing_key_guidance(client, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    res = client.post("/chat", json={"message": "What does this EMG trace indicate?"})
    assert res.status_code == 200
    data = res.json()
    assert "OpenAI API key" in data["reply"] or "Settings" in data["reply"]


def test_chat_endpoint_mock_provider(client, monkeypatch):
    import llm_client

    def _mock_gen(messages, **kwargs):
        return (
            "The EMG signal displays normal motor unit potentials with no denervation signs.\n\n"
            "---\n"
            "### Related inquiries:\n"
            "- What is the normal recruitment ratio?\n"
            "- How does Grad-CAM highlight waveform peaks?"
        )

    monkeypatch.setattr(llm_client, "generate_llm_completion", _mock_gen)
    monkeypatch.setattr(llm_client, "is_provider_configured", lambda p: True)

    res = client.post(
        "/chat",
        json={
            "message": "Analyze this waveform",
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "denervation" in data["reply"]
    assert len(data["suggestions"]) >= 1

