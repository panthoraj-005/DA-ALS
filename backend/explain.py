"""
Explainability: Grad-CAM (which part of the signal mattered), SHAP (which feature
mattered), and a plain-language explanation.

The explanation is a TEMPLATE built from the numbers the pipeline actually
produced. When ENABLE_LLM=1, Qwen2.5 is asked to smooth the wording of that
template and is explicitly forbidden from adding clinical detail — it never sees
the image and never invents what the waveform "shows".

Grad-CAM is a port of notebook cell 9 (hooks on conv3, ReLU of the weighted
activation sum, interpolated back to signal length). The annotated render is
cell 13.
"""

from __future__ import annotations

import json
import logging
import ssl
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from models import ModelRegistry

log = logging.getLogger("als.explain")


# ---------------------------------------------------------------------------
# Grad-CAM - notebook cell 9
# ---------------------------------------------------------------------------
def compute_gradcam(reg: ModelRegistry, raw: np.ndarray) -> np.ndarray | None:
    activations: dict = {}
    gradients: dict = {}

    def fwd_hook(module, inp, out):  # noqa: ARG001 - torch fixes this signature
        activations["conv3"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):  # noqa: ARG001 - torch fixes this signature
        gradients["conv3"] = grad_out[0].detach()

    h1 = reg.cnn.conv3.register_forward_hook(fwd_hook)
    h2 = reg.cnn.conv3.register_full_backward_hook(bwd_hook)

    try:
        x = torch.tensor(np.asarray(raw, dtype=np.float32), dtype=torch.float32)
        x = x.view(1, 1, -1).to(reg.device)
        x.requires_grad_(True)

        reg.cnn.zero_grad(set_to_none=True)
        out = reg.cnn(x)
        out.backward()

        act = activations["conv3"][0]
        grad = gradients["conv3"][0]

        weights = grad.mean(dim=1)
        cam = torch.relu((weights[:, None] * act).sum(dim=0)).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_upsampled = np.interp(
            np.linspace(0, len(cam) - 1, len(raw)),
            np.arange(len(cam)),
            cam,
        )
        return cam_upsampled
    except Exception as exc:
        log.exception("Grad-CAM failed: %s", exc)
        return None
    finally:
        h1.remove()
        h2.remove()
        reg.cnn.zero_grad(set_to_none=True)


def render_annotated_image(
    processed: np.ndarray,
    cam: np.ndarray | None,
    abnormal_regions,
    predicted_text: str,
    confidence: float,
    out_path: Path,
) -> Path:
    """Notebook cell 13: Grad-CAM heat strip under the trace, abnormal spans in red."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_facecolor("white")

    if cam is not None:
        cam_norm = cam / (cam.max() + 1e-8)
        ax.imshow(
            cam_norm[np.newaxis, :],
            aspect="auto",
            cmap="Oranges",
            alpha=0.35,
            extent=[0, len(processed), processed.min(), processed.max()],
            origin="lower",
            interpolation="bilinear",
        )

    ax.plot(processed, color="blue", linewidth=1.1, label="EMG signal")

    for (s, e) in abnormal_regions:
        ax.axvspan(s, e, color="red", alpha=0.25)
        mid = (s + e) // 2
        ax.annotate(
            f"{predicted_text}\n{confidence * 100:.1f}%",
            xy=(mid, processed[mid]),
            xytext=(mid, processed.max() * 1.05),
            fontsize=8,
            color="darkred",
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "darkred", "lw": 1},
        )

    ax.set_title(
        f"{predicted_text} - confidence {confidence * 100:.1f}%"
        + ("" if cam is not None else "  (Grad-CAM unavailable)"),
        fontsize=11,
    )
    ax.set_xlabel("Time Sample")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# SHAP - notebook cell 10
# ---------------------------------------------------------------------------
def compute_shap(reg: ModelRegistry, meta_features: np.ndarray, top_k: int = 5):
    if reg.shap_explainer is None or meta_features is None:
        return None, None

    try:
        values = reg.shap_explainer.shap_values(meta_features)
        values = np.array(values)

        if values.ndim == 3:  # (classes, samples, features) on some xgboost versions
            values = values[-1]
        row = values.reshape(-1)[: config.META_FEATURE_DIM]

        order = np.argsort(np.abs(row))[::-1][:top_k]
        top = [
            {
                "feature": config.FEATURE_NAMES[int(i)],
                "label": config.feature_family_label(config.FEATURE_NAMES[int(i)]),
                "value": round(float(row[int(i)]), 6),
                "direction": "toward ALS" if row[int(i)] > 0 else "toward Normal",
            }
            for i in order
        ]
        return config.FEATURE_NAMES[int(order[0])], top
    except Exception as exc:
        log.exception("SHAP failed: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
def build_template_explanation(result: dict) -> str:
    """Grounded in the numbers only. No claim the models did not produce."""
    verdict = result["final_prediction"]
    conf = result["final_confidence"] * 100
    parts: list[str] = []

    if result["fusion"] == "cnn+florence+meta":
        origin = "The fused meta-learner (CNN + Florence-2 vision model, combined by XGBoost)"
    elif result["fusion"] == "cnn+florence":
        origin = "The 1-D CNN (the meta-learner was unavailable, so it was not fused)"
    else:
        origin = "The 1-D CNN, on the fast path with no vision model,"

    parts.append(
        f"{origin} scored this recording as {verdict} with {conf:.1f}% confidence"
        + (f", severity {result['severity']}." if result["severity"] != "N/A" else ".")
    )

    cnn_pct = result["cnn_probability"] * 100
    if result["florence_probability"] is not None:
        vlm_pct = result["florence_probability"] * 100
        agreement = "agree" if result["models_agree"] else "disagree"
        parts.append(
            f"The CNN put the ALS probability at {cnn_pct:.1f}% and the Florence-2 vision "
            f"model at {vlm_pct:.1f}%, so the two {agreement}."
        )
    else:
        parts.append(
            f"The CNN put the ALS probability at {cnn_pct:.1f}%. The vision model did not "
            f"run for this signal, so this is a single-model result."
        )

    n_abn = result["abnormal_segment_count"]
    parts.append(
        f"Segment energy flagged {n_abn} of {result['n_segments']} windows as unusually "
        f"high-energy relative to this recording's own mean"
        + (
            "; those windows are shaded red on the trace."
            if n_abn
            else ", so no window stood out."
        )
    )

    if result.get("top_shap_feature"):
        parts.append(
            f"The input that moved the fused decision most was "
            f"{config.feature_family_label(result['top_shap_feature'])}."
        )

    if result["models_agree"] is False:
        parts.append(
            "Because the two models disagree, treat this result as low-certainty and "
            "review the trace directly."
        )

    parts.append(config.DISCLAIMER)
    return " ".join(parts)


LLM_SYSTEM = (
    "You rewrite a machine-generated screening summary so it reads smoothly for a "
    "clinician. Rules, strictly: keep every number exactly as given; do not add any "
    "clinical detail, diagnosis, symptom, anatomy, or claim about what the waveform "
    "looks like; do not remove the disclaimer sentence; do not add new sentences of "
    "your own. Return only the rewritten summary, 3-4 sentences."
)


import llm_client


def smooth_with_llm(reg: ModelRegistry, template: str) -> tuple[str, str]:
    """
    Smooth grounded template using configured AI provider or local HuggingFace model.
    Returns (explanation_text, explanation_source).
    """
    prov = config.LLM_PROVIDER.lower()

    # 1. Cloud / Remote Provider path
    if prov != "local" and llm_client.is_provider_configured(prov):
        cat = llm_client.PROVIDER_CATALOG.get(prov, {})
        prov_name = cat.get("name", prov.title())
        model_name = config.LLM_MODEL or cat.get("default_model", "")
        try:
            smoothed = llm_client.generate_llm_completion(
                [{"role": "user", "content": f"Template to rewrite:\n{template}"}],
                system_prompt=LLM_SYSTEM,
                provider=prov,
                model=model_name,
                temperature=0.1,
                max_tokens=280,
            )
            if len(smoothed) >= 50:
                if "not a medical diagnosis" not in smoothed.lower():
                    smoothed = f"{smoothed} {config.DISCLAIMER}"
                return smoothed, f"template + {model_name} ({prov_name})"
        except Exception as exc:
            log.warning("Multi-provider smoothing with %s (%s) failed: %s", prov, model_name, exc)

    # 2. Local HuggingFace model fallback
    if reg.llm_model is not None and reg.llm_tokenizer is not None:
        try:
            messages = [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": template},
            ]
            text = reg.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = reg.llm_tokenizer(text, return_tensors="pt").to(reg.llm_model.device)

            with torch.no_grad():
                out = reg.llm_model.generate(
                    **inputs, max_new_tokens=220, do_sample=False, temperature=None, top_p=None
                )

            response = reg.llm_tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            if len(response) >= 60:
                if "not a medical diagnosis" not in response.lower():
                    response = f"{response} {config.DISCLAIMER}"
                return response, "template + Qwen2.5 wording pass"
        except Exception as exc:
            log.exception("LLM smoothing failed: %s", exc)

    return template, "grounded template"


# ---------------------------------------------------------------------------
# Entry point used by pipeline.run_pipeline
# ---------------------------------------------------------------------------
def attach_explanations(
    reg: ModelRegistry,
    result: dict,
    *,
    raw: np.ndarray,
    processed: np.ndarray,
    abnormal_regions,
    meta_features,
    record_id: str,
    timings: dict,
    use_llm: bool | None = None,
) -> None:
    t0 = time.perf_counter()
    cam = compute_gradcam(reg, raw)
    timings["gradcam_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    annotated = config.OUTPUT_DIR / f"{record_id}_annotated.png"
    render_annotated_image(
        processed,
        cam,
        abnormal_regions,
        result["final_prediction"],
        result["final_confidence"],
        annotated,
    )
    result["gradcam_image_url"] = f"/images/{annotated.name}"
    result["gradcam_available"] = cam is not None

    if cam is not None:
        # A coarse profile for the frontend to draw without shipping 23k floats.
        buckets = np.array_split(cam, 60)
        result["gradcam_profile"] = [round(float(b.mean()), 4) for b in buckets]
        peak = int(np.argmax(cam))
        result["gradcam_peak_index"] = peak
        result["gradcam_peak_percent"] = round(100.0 * peak / max(len(cam) - 1, 1), 1)

    t0 = time.perf_counter()
    top_feature, top_features = compute_shap(reg, meta_features)
    timings["shap_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["top_shap_feature"] = top_feature
    result["top_shap_features"] = top_features

    template = build_template_explanation(result)
    want_llm = config.ENABLE_LLM if use_llm is None else bool(use_llm)
    provider_available = (
        (config.LLM_PROVIDER != "local" and llm_client.is_provider_configured(config.LLM_PROVIDER))
        or reg.llm_model is not None
    )

    if want_llm and provider_available:
        t0 = time.perf_counter()
        smoothed, source = smooth_with_llm(reg, template)
        result["explanation"] = smoothed
        result["explanation_source"] = source
        timings["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    else:
        result["explanation"] = template
        result["explanation_source"] = "grounded template"

