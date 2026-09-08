"""
Unified Multi-Provider LLM Client for ALS Screening Platform.

Supports:
- Google Gemini API
- OpenAI (GPT-4o, GPT-4o-mini, o3-mini)
- Anthropic Claude (Claude 3.5 Haiku, Claude 3.5 Sonnet, Claude 3.7 Sonnet)
- Groq Cloud (Llama 3.3 70B, Llama 3.1 8B, DeepSeek R1 Distill)
- OpenRouter (Unified multi-model aggregator)
- Ollama (Local/Self-hosted e.g. MedGemma, Llama 3.2, Qwen 2.5)
- Local HuggingFace Qwen2.5 (when registry model loaded)
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request
from typing import Any

import config

log = logging.getLogger("als.llm_client")

# ---------------------------------------------------------------------------
# Provider Definitions & Metadata
# ---------------------------------------------------------------------------
PROVIDER_CATALOG: dict[str, dict[str, Any]] = {
    "gemini": {
        "id": "gemini",
        "name": "Google Gemini",
        "description": "Google AI Studio API (Fast, generous free tier, clinical-grade reasoning)",
        "env_key": "GEMINI_API_KEY",
        "requires_key": True,
        "key_url": "https://aistudio.google.com/app/apikey",
        "default_model": "gemini-3.1-flash-lite",
        "models": [
            {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite (Fastest, Recommended)"},
            {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash (Balanced Clinical Reasoning)"},
            {"id": "gemini-3.7-flash", "name": "Gemini 3.7 Flash (Deep Reasoning)"},
            {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Low-Latency)"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (Complex Multimodal)"},
        ],
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI",
        "description": "OpenAI official API (GPT-4o, GPT-4o-mini)",
        "env_key": "OPENAI_API_KEY",
        "requires_key": True,
        "key_url": "https://platform.openai.com/api-keys",
        "default_model": "gpt-4o-mini",
        "models": [
            {"id": "gpt-4o-mini", "name": "GPT-4o Mini (Fast & Cost-Effective, Recommended)"},
            {"id": "gpt-4o", "name": "GPT-4o (Omni Clinical Intelligence)"},
            {"id": "o3-mini", "name": "o3-mini (Advanced Reasoning)"},
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
        ],
    },
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic Claude",
        "description": "Anthropic Claude API (Constitutional AI, safe clinical summarization)",
        "env_key": "ANTHROPIC_API_KEY",
        "requires_key": True,
        "key_url": "https://console.anthropic.com/settings/keys",
        "default_model": "claude-3-5-haiku-20241022",
        "models": [
            {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku (High Speed, Concise)"},
            {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet (State-of-the-Art Analysis)"},
            {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet (Hybrid Reasoning)"},
        ],
    },
    "groq": {
        "id": "groq",
        "name": "Groq Cloud (LPU Inference)",
        "description": "Ultra high-speed open-source model inference on LPUs",
        "env_key": "GROQ_API_KEY",
        "requires_key": True,
        "key_url": "https://console.groq.com/keys",
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile (Recommended)"},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant (Sub-second response)"},
            {"id": "deepseek-r1-distill-llama-70b", "name": "DeepSeek R1 Distill Llama 70B (Reasoning)"},
            {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B (Long Context)"},
        ],
    },
    "openrouter": {
        "id": "openrouter",
        "name": "OpenRouter",
        "description": "Unified API gateway supporting hundreds of leading open and proprietary models",
        "env_key": "OPENROUTER_API_KEY",
        "requires_key": True,
        "key_url": "https://openrouter.ai/keys",
        "default_model": "google/gemini-2.5-flash",
        "models": [
            {"id": "google/gemini-2.5-flash", "name": "Google Gemini 2.5 Flash via OpenRouter"},
            {"id": "anthropic/claude-3.5-haiku", "name": "Claude 3.5 Haiku via OpenRouter"},
            {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Meta Llama 3.3 70B Instruct"},
            {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3 Chat"},
            {"id": "mistralai/mistral-large-2411", "name": "Mistral Large 2411"},
        ],
    },
    "ollama": {
        "id": "ollama",
        "name": "Ollama (Local / Self-Hosted)",
        "description": "Run open models completely private & offline on your own machine/server",
        "env_key": "OLLAMA_BASE_URL",
        "requires_key": False,
        "key_url": "https://ollama.com",
        "default_model": "medgemma",
        "models": [
            {"id": "medgemma", "name": "MedGemma (Medical Fine-tune)"},
            {"id": "llama3.2", "name": "Llama 3.2 (Lightweight Local)"},
            {"id": "llama3.3", "name": "Llama 3.3 (High Capability)"},
            {"id": "mistral", "name": "Mistral 7B"},
            {"id": "qwen2.5", "name": "Qwen 2.5"},
        ],
    },
    "local": {
        "id": "local",
        "name": "Local HuggingFace Model",
        "description": "Built-in Qwen2.5-1.5B loaded directly into Python runtime",
        "env_key": "",
        "requires_key": False,
        "key_url": "",
        "default_model": "Qwen/Qwen2.5-1.5B",
        "models": [
            {"id": "Qwen/Qwen2.5-1.5B", "name": "Qwen2.5-1.5B (Local Weights)"},
        ],
    },
}


def get_provider_key(provider: str) -> str:
    """Retrieve the configured API key or base URL for a given provider."""
    provider = provider.lower()
    if provider == "gemini":
        return config.GEMINI_API_KEY
    if provider == "openai":
        return config.OPENAI_API_KEY
    if provider == "anthropic":
        return config.ANTHROPIC_API_KEY
    if provider == "groq":
        return config.GROQ_API_KEY
    if provider == "openrouter":
        return config.OPENROUTER_API_KEY
    if provider == "ollama":
        return config.OLLAMA_BASE_URL
    return ""


def is_provider_configured(provider: str) -> bool:
    """Check if the provider is ready for calls."""
    provider = provider.lower()
    if provider == "ollama":
        return bool(config.OLLAMA_BASE_URL)
    if provider == "local":
        return True
    return bool(get_provider_key(provider))


# ---------------------------------------------------------------------------
# HTTP Helpers
# ---------------------------------------------------------------------------
def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int = 30) -> dict:
    """Execute a POST request with JSON payload using standard library urllib."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as err:
        try:
            err_body = json.loads(err.read().decode("utf-8"))
        except Exception:
            err_body = str(err)
        log.warning("HTTP error calling %s: %s (response: %s)", url, err, err_body)
        raise RuntimeError(f"HTTP {err.code}: {err_body}") from err
    except urllib.error.URLError as err:
        log.warning("Network connection error calling %s: %s", url, err)
        raise RuntimeError(f"Connection failed: {err.reason}") from err


# ---------------------------------------------------------------------------
# Provider Call Implementations
# ---------------------------------------------------------------------------
def _call_gemini(
    messages: list[dict[str, str]],
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """Execute Gemini generateContent API."""
    contents = []
    first_user_injected = False

    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        content = msg["content"]
        if role == "user" and not first_user_injected and system_prompt:
            content = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER:\n{content}"
            first_user_injected = True
        contents.append({"role": role, "parts": [{"text": content}]})

    if not first_user_injected and system_prompt:
        contents.insert(0, {"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTIONS:\n{system_prompt}"}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    resp = _post_json(url, payload, headers, timeout=25)
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Invalid Gemini response structure: {resp}") from exc


def _call_openai_compatible(
    url: str,
    headers: dict[str, str],
    messages: list[dict[str, str]],
    system_prompt: str,
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """Execute standard OpenAI-compatible /chat/completions endpoint."""
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = "assistant" if msg["role"] in {"assistant", "model"} else msg["role"]
        formatted_messages.append({"role": role, "content": msg["content"]})

    payload = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = _post_json(url, payload, headers, timeout=35)
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Invalid OpenAI-compatible response: {resp}") from exc


def _call_anthropic(
    messages: list[dict[str, str]],
    system_prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """Execute Anthropic /v1/messages API."""
    formatted_messages = []
    for msg in messages:
        role = "assistant" if msg["role"] in {"assistant", "model"} else "user"
        formatted_messages.append({"role": role, "content": msg["content"]})

    # Anthropic requires the first message to be from the 'user'
    if not formatted_messages or formatted_messages[0]["role"] != "user":
        formatted_messages.insert(0, {"role": "user", "content": "Hello"})

    payload: dict[str, Any] = {
        "model": model,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        payload["system"] = system_prompt

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    resp = _post_json(url, payload, headers, timeout=35)
    try:
        return resp["content"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Invalid Anthropic response: {resp}") from exc


def _call_ollama(
    messages: list[dict[str, str]],
    system_prompt: str,
    model: str,
    base_url: str,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """Execute Ollama local /api/chat endpoint."""
    formatted_messages = []
    if system_prompt:
        formatted_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = "assistant" if msg["role"] in {"assistant", "model"} else "user"
        formatted_messages.append({"role": role, "content": msg["content"]})

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": formatted_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    headers = {"Content-Type": "application/json"}

    resp = _post_json(endpoint, payload, headers, timeout=45)
    try:
        return resp["message"]["content"].strip()
    except KeyError as exc:
        raise RuntimeError(f"Invalid Ollama response: {resp}") from exc


# ---------------------------------------------------------------------------
# Public Unified Interface
# ---------------------------------------------------------------------------
def generate_llm_completion(
    messages: list[dict[str, str]],
    *,
    system_prompt: str = "",
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> str:
    """
    Route conversation or prompt to active or specified LLM provider.
    """
    active_provider = (provider or config.LLM_PROVIDER).lower().strip()
    catalog_entry = PROVIDER_CATALOG.get(active_provider, PROVIDER_CATALOG["gemini"])
    active_model = (model or config.LLM_MODEL or catalog_entry["default_model"]).strip()
    key = api_key if api_key is not None else get_provider_key(active_provider)

    if catalog_entry.get("requires_key", False) and not key:
        env_var = catalog_entry.get("env_key", "API_KEY")
        raise ValueError(
            f"No API key configured for provider '{catalog_entry['name']}'. "
            f"Please configure {env_var} in platform settings or .env."
        )

    if active_provider == "gemini":
        return _call_gemini(messages, system_prompt, active_model, key, temperature, max_tokens)

    if active_provider == "openai":
        base_url = getattr(config, "OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        return _call_openai_compatible(url, headers, messages, system_prompt, active_model, temperature, max_tokens)

    if active_provider == "anthropic":
        return _call_anthropic(messages, system_prompt, active_model, key, temperature, max_tokens)

    if active_provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        return _call_openai_compatible(url, headers, messages, system_prompt, active_model, temperature, max_tokens)

    if active_provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/als-screening",
            "X-Title": "EMG ALS Screening Platform",
        }
        return _call_openai_compatible(url, headers, messages, system_prompt, active_model, temperature, max_tokens)

    if active_provider == "ollama":
        base_url = (api_key or config.OLLAMA_BASE_URL or "http://localhost:11434").strip()
        return _call_ollama(messages, system_prompt, active_model, base_url, temperature, max_tokens)

    raise ValueError(f"Unsupported AI provider: {active_provider}")


def test_provider_connection(
    provider: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """
    Test connectivity and credentials for a given provider with a minimal probe query.
    Returns: {"valid": bool, "message": str}
    """
    prov = provider.lower().strip()
    cat = PROVIDER_CATALOG.get(prov)
    if not cat:
        return {"valid": False, "message": f"Unknown provider: {prov}"}

    key = api_key.strip() if api_key else get_provider_key(prov)
    test_model = (model or cat["default_model"]).strip()

    if cat.get("requires_key", False) and not key:
        return {
            "valid": False,
            "message": f"No API key provided. Please enter a valid API key for {cat['name']}.",
        }

    try:
        messages = [{"role": "user", "content": "Ping"}]
        test_tokens = 5

        if prov == "gemini":
            _call_gemini(messages, "", test_model, key, temperature=0.1, max_tokens=test_tokens)
        elif prov == "openai":
            b_url = getattr(config, "OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
            url = f"{b_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            _call_openai_compatible(url, headers, messages, "", test_model, temperature=0.1, max_tokens=test_tokens)
        elif prov == "anthropic":
            _call_anthropic(messages, "", test_model, key, temperature=0.1, max_tokens=test_tokens)
        elif prov == "groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            _call_openai_compatible(url, headers, messages, "", test_model, temperature=0.1, max_tokens=test_tokens)
        elif prov == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://github.com/als-screening",
                "X-Title": "EMG ALS Screening Platform",
            }
            _call_openai_compatible(url, headers, messages, "", test_model, temperature=0.1, max_tokens=test_tokens)
        elif prov == "ollama":
            b_url = (base_url or key or config.OLLAMA_BASE_URL or "http://localhost:11434").strip()
            _call_ollama(messages, "", test_model, b_url, temperature=0.1, max_tokens=test_tokens)
        elif prov == "local":
            return {"valid": True, "message": "Local HuggingFace model registry ready."}
        else:
            return {"valid": False, "message": f"Provider {prov} testing is not supported."}

        return {"valid": True, "message": f"Successfully connected to {cat['name']} ({test_model})!"}
    except Exception as exc:
        msg = str(exc)
        if "HTTP 401" in msg or "401" in msg:
            msg = "Authentication failed (401). Please verify your API key is correct and active."
        elif "HTTP 404" in msg or "404" in msg:
            msg = f"Model '{test_model}' or endpoint not found (404). Please verify model name."
        elif "HTTP 429" in msg or "429" in msg:
            msg = "Rate limit or quota reached (429). Check your billing/credits."
        elif "Connection failed" in msg:
            msg = f"Cannot reach server: {msg}"
        return {"valid": False, "message": msg}
