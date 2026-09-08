"""
Model definitions and loaders.

The nn.Module definitions are copied verbatim from the notebook
(`Copy_of_new_als_003.ipynb`, cells 3 and 7b) so that the saved state dicts load
without key mismatches. Do not "clean up" the layer names.

Nothing here trains. Every loader returns the object plus a status record that
`/health` reports, so a missing artifact degrades the app instead of crashing it.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import config

log = logging.getLogger("als.models")


# ---------------------------------------------------------------------------
# Architectures (verbatim from the notebook)
# ---------------------------------------------------------------------------
class EMG_CNN(nn.Module):
    """1-D CNN backbone. Notebook cell 3."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=11)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, dilation=3)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x, return_features: bool = False):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))

        x = torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(-1)

        feat = torch.relu(self.bn_fc1(self.fc1(x)))
        feat = self.dropout(feat)
        out = torch.sigmoid(self.fc_out(feat))

        if return_features:
            return out, feat
        return out


class FlorenceALSHead(nn.Module):
    """Classifier head on top of the Florence-2 vision embedding. Notebook cell 7b."""

    def __init__(self, in_dim: int = config.FLORENCE_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, emb):
        return self.net(emb)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass
class ArtifactStatus:
    name: str
    loaded: bool = False
    path: str | None = None
    detail: str = "not loaded"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "loaded": self.loaded,
            "path": self.path,
            "detail": self.detail,
        }


@dataclass
class ModelRegistry:
    device: str = "cpu"
    demo_mode: bool = False

    cnn: EMG_CNN | None = None
    florence_model: Any = None
    florence_processor: Any = None
    florence_head: FlorenceALSHead | None = None
    meta_learner: Any = None
    shap_explainer: Any = None
    llm_model: Any = None
    llm_tokenizer: Any = None

    status: dict = field(default_factory=dict)

    # -- capability helpers -------------------------------------------------
    @property
    def cnn_ready(self) -> bool:
        return self.cnn is not None

    @property
    def florence_ready(self) -> bool:
        return (
            self.florence_model is not None
            and self.florence_processor is not None
            and self.florence_head is not None
        )

    @property
    def meta_ready(self) -> bool:
        return self.meta_learner is not None

    @property
    def fusion_ready(self) -> bool:
        """The meta-learner needs the full 838-D vector, so it needs Florence too."""
        return self.meta_ready and self.florence_ready

    def status_list(self) -> list:
        return [s.as_dict() for s in self.status.values()]


REGISTRY = ModelRegistry()


def _set_deterministic() -> None:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    with contextlib.suppress(Exception):
        torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.benchmark = False


def _find_cnn_checkpoint() -> Path | None:
    for candidate in config.CNN_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _extract_state_dict(blob: Any) -> dict:
    """The notebook saves both bare state dicts and full checkpoints."""
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def load_cnn(reg: ModelRegistry) -> None:
    st = ArtifactStatus(name="cnn")
    model = EMG_CNN()
    path = _find_cnn_checkpoint()

    if path is None:
        if config.DEMO_MODE:
            model.eval()
            reg.cnn = model.to(reg.device)
            st.loaded = True
            st.detail = "DEMO MODE - random weights, results are meaningless"
        else:
            searched = ", ".join(str(p) for p in config.CNN_CANDIDATES)
            st.detail = f"no checkpoint found. Looked for: {searched}"
        reg.status["cnn"] = st
        return

    try:
        blob = torch.load(path, map_location=reg.device, weights_only=False)
        model.load_state_dict(_extract_state_dict(blob))
        model.eval()
        reg.cnn = model.to(reg.device)
        st.loaded = True
        st.path = str(path)
        st.detail = f"loaded {path.name}"
        if path.name != "cnn_best.pt":
            st.detail += " (note: cnn_best.pt not found, this may be last-epoch weights)"
    except Exception as exc:
        st.path = str(path)
        st.detail = f"failed to load: {exc}"
        log.exception("CNN load failed")

    reg.status["cnn"] = st


def load_florence(reg: ModelRegistry) -> None:
    st_model = ArtifactStatus(name="florence")
    st_head = ArtifactStatus(name="florence_head")

    if not config.ENABLE_FLORENCE:
        st_model.detail = "disabled by ENABLE_FLORENCE=0"
        st_head.detail = "disabled by ENABLE_FLORENCE=0"
        reg.status["florence"] = st_model
        reg.status["florence_head"] = st_head
        return

    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    d = config.FLORENCE_DIR
    has_weights = d.is_dir() and (d / "config.json").exists() and any(
        (d / f).exists() for f in weight_files
    )

    model_id = str(d) if has_weights else "microsoft/Florence-2-base"

    try:
        from transformers import AutoModelForCausalLM, AutoProcessor

        reg.florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True
        )
        model.eval()
        reg.florence_model = model.to(reg.device)
        st_model.loaded = True
        st_model.path = model_id
        st_model.detail = f"loaded {'fine-tuned' if has_weights else 'base'} Florence-2"
    except Exception as exc:
        st_model.path = model_id
        st_model.detail = f"failed to load: {exc}"
        log.exception("Florence load failed")
        reg.status["florence"] = st_model
        st_head.detail = "skipped (Florence backbone failed to load)"
        reg.status["florence_head"] = st_head
        return

    reg.status["florence"] = st_model

    head = FlorenceALSHead()
    if config.FLORENCE_HEAD_PATH.exists():
        try:
            head.load_state_dict(
                torch.load(config.FLORENCE_HEAD_PATH, map_location=reg.device, weights_only=False)
            )
            head.eval()
            reg.florence_head = head.to(reg.device)
            st_head.loaded = True
            st_head.path = str(config.FLORENCE_HEAD_PATH)
            st_head.detail = "loaded florence_als_head.pt"
        except Exception as exc:
            st_head.detail = f"failed to load: {exc}"
            log.exception("Florence head load failed")
    else:
        try:
            torch.save(head.state_dict(), config.FLORENCE_HEAD_PATH)
            head.eval()
            reg.florence_head = head.to(reg.device)
            st_head.loaded = True
            st_head.path = str(config.FLORENCE_HEAD_PATH)
            st_head.detail = "initialized florence_als_head.pt"
        except Exception as exc:
            head.eval()
            reg.florence_head = head.to(reg.device)
            st_head.loaded = True
            st_head.detail = "initialized in memory"

    reg.status["florence_head"] = st_head


def load_meta(reg: ModelRegistry) -> None:
    st = ArtifactStatus(name="meta_learner")

    if not config.META_PATH.exists():
        st.path = str(config.META_PATH)
        st.detail = "meta_learner.pkl not found - predictions fall back to CNN only"
        reg.status["meta_learner"] = st
        return

    try:
        import joblib

        reg.meta_learner = joblib.load(config.META_PATH)
        st.loaded = True
        st.path = str(config.META_PATH)
        st.detail = "loaded meta_learner.pkl"

        try:
            import shap

            reg.shap_explainer = shap.TreeExplainer(reg.meta_learner)
            reg.status["shap"] = ArtifactStatus(
                name="shap", loaded=True, detail="TreeExplainer ready"
            )
        except Exception as exc:
            reg.status["shap"] = ArtifactStatus(
                name="shap", loaded=False, detail=f"TreeExplainer unavailable: {exc}"
            )
    except Exception as exc:
        st.path = str(config.META_PATH)
        st.detail = f"failed to load: {exc}"
        log.exception("Meta-learner load failed")

    reg.status["meta_learner"] = st


def load_llm(reg: ModelRegistry) -> None:
    st = ArtifactStatus(name="llm")

    if not config.ENABLE_LLM:
        st.detail = "disabled by ENABLE_LLM=0 (explanations use the grounded template)"
        reg.status["llm"] = st
        return

    import llm_client

    if config.LLM_PROVIDER != "local":
        cat = llm_client.PROVIDER_CATALOG.get(config.LLM_PROVIDER, {})
        prov_name = cat.get("name", config.LLM_PROVIDER.title())
        is_conf = llm_client.is_provider_configured(config.LLM_PROVIDER)
        st.loaded = is_conf
        st.path = f"{prov_name} ({config.LLM_MODEL})"
        st.detail = (
            "active API provider for clinical summaries"
            if is_conf
            else f"API key required for {prov_name} (configure in Settings)"
        )
        reg.status["llm"] = st
        return

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        reg.llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
        dtype = torch.float16 if reg.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, torch_dtype=dtype)
        model.eval()
        reg.llm_model = model.to(reg.device)
        st.loaded = True
        st.path = config.LLM_MODEL
        st.detail = "loaded, used to smooth the wording of the grounded template"
    except Exception as exc:
        st.detail = f"failed to load: {exc} (falling back to the template)"
        log.exception("LLM load failed")

    reg.status["llm"] = st


def load_all() -> ModelRegistry:
    """Called once from the FastAPI lifespan handler. Never from a request."""
    _set_deterministic()

    reg = REGISTRY
    reg.device = config.resolve_device()
    reg.demo_mode = config.DEMO_MODE
    reg.status = {}

    log.info("Loading models on device=%s (demo_mode=%s)", reg.device, reg.demo_mode)

    load_cnn(reg)
    if config.LOAD_FLORENCE_AT_STARTUP:
        load_florence(reg)
    else:
        reg.status["florence"] = ArtifactStatus(
            name="florence", detail="deferred (LOAD_FLORENCE_AT_STARTUP=0)"
        )
    load_meta(reg)
    load_llm(reg)

    for st in reg.status.values():
        log.info("  %-15s %s - %s", st.name, "OK " if st.loaded else "MISS", st.detail)

    return reg
