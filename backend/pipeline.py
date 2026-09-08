"""
Inference pipeline: preprocess -> CNN -> signal image -> Florence-2 -> XGBoost.

Every numeric step is a direct port of `Copy_of_new_als_003.ipynb` so that a
signal scored here gets the same answer it got in the notebook. Inference only:
there is no optimizer, no backward pass except the one Grad-CAM needs, and no
path that writes model weights.

Notebook provenance
  preprocess_signal / segment_signal   cell 3
  EMG_CNN forward + features           cells 3, 5
  signal image rendering               cell 6 (test-set render, dpi=150)
  Florence embedding + head            cell 7b
  838-D meta feature vector            cell 8
  severity thresholds                  cell 14

IMPORTANT, and easy to get wrong: the notebook trains and scores the CNN on the
RAW signal (`X_train.transpose(0, 2, 1)` straight off the pickle, cells 4 and 5).
`preprocess_signal` is applied only when rendering the image the VLM reads
(cells 6, 13). We keep that split — bandpass-filtering the CNN's input here would
feed it a distribution it never saw in training.

Known notebook quirk, preserved deliberately: when the meta-learner was TRAINED,
the Florence probability column held the head's raw logit, while at TEST time it
held sigmoid(logit). We reproduce the TEST-time behaviour, because that is the
path the reported 96.5% accuracy was measured on. See README > Known quirks.
"""

from __future__ import annotations

import contextlib
import logging
import time
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt

import config
from models import ModelRegistry

log = logging.getLogger("als.pipeline")


# ---------------------------------------------------------------------------
# Preprocessing - notebook cell 3, unchanged
# ---------------------------------------------------------------------------
def bandpass_filter(signal, low=None, high=None, fs=None, order=None):
    low = config.BANDPASS_LOW if low is None else low
    high = config.BANDPASS_HIGH if high is None else high
    fs = config.FS if fs is None else fs
    order = config.BANDPASS_ORDER if order is None else order

    nyq = 0.5 * fs
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.99)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal, axis=0)


def zscore_normalize(signal):
    mu = np.mean(signal)
    sd = np.std(signal) + 1e-8
    return (signal - mu) / sd


def preprocess_signal(raw_signal):
    sig = np.asarray(raw_signal).astype(np.float32).flatten()
    # The notebook falls back to the raw signal if the filter fails; keep that.
    with contextlib.suppress(Exception):
        sig = bandpass_filter(sig)
    return zscore_normalize(sig)


def segment_signal(signal, n_segments: int | None = None):
    n_segments = config.N_SEGMENTS if n_segments is None else n_segments
    length = len(signal)
    seg_len = length // n_segments
    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else length
        segments.append((start, end))
    return segments


def find_abnormal_regions(signal, n_segments: int | None = None):
    """Segment energy above mean+std, exactly as the notebook flags regions."""
    n_segments = config.N_SEGMENTS if n_segments is None else n_segments
    segments = segment_signal(signal, n_segments)
    seg_scores = np.array([np.mean(signal[s:e] ** 2) for (s, e) in segments])
    threshold = seg_scores.mean() + seg_scores.std()
    regions = [(s, e) for i, (s, e) in enumerate(segments) if seg_scores[i] > threshold]
    return regions, seg_scores


# ---------------------------------------------------------------------------
# Signal image - notebook cell 6
# ---------------------------------------------------------------------------
def render_signal_image(signal, abnormal_regions, out_path: Path, title: str = "EMG signal"):
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.set_facecolor("white")
    ax.plot(signal, color="blue", linewidth=1.2)

    for (s, e) in abnormal_regions:
        ax.axvspan(s, e, color="red", alpha=0.25)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Sample")
    ax.set_ylabel("Amplitude")

    info_text = (
        f"Mean: {np.mean(signal):.2f}\n"
        f"STD: {np.std(signal):.2f}\n"
        f"Max: {np.max(signal):.2f}\n"
        f"Min: {np.min(signal):.2f}\n"
        f"Abnormal: {len(abnormal_regions)}/{config.N_SEGMENTS}"
    )
    ax.text(
        0.01,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.7},
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Stage 1: CNN
# ---------------------------------------------------------------------------
def run_cnn(reg: ModelRegistry, raw: np.ndarray):
    """The CNN sees the RAW signal - that is what it was trained on."""
    x = torch.tensor(np.asarray(raw, dtype=np.float32), dtype=torch.float32)
    x = x.view(1, 1, -1).to(reg.device)

    reg.cnn.eval()
    with torch.no_grad():
        prob, feat = reg.cnn(x, return_features=True)

    return float(prob.item()), feat.detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# Stage 2: Florence-2
# ---------------------------------------------------------------------------
CAPTION_PROMPT = "<MORE_DETAILED_CAPTION>"


def _florence_embedding(reg: ModelRegistry, pixel_values):
    with torch.no_grad():
        vision_out = reg.florence_model._encode_image(pixel_values)
        if isinstance(vision_out, (tuple, list)):
            vision_out = vision_out[0]
        emb = vision_out.mean(dim=1)
    return emb.float()


def run_florence(reg: ModelRegistry, image_path: Path, want_caption: bool = True):
    from PIL import Image as PILImage

    image = PILImage.open(image_path).convert("RGB")
    inputs = reg.florence_processor(
        text=[CAPTION_PROMPT], images=[image], return_tensors="pt"
    ).to(reg.device)
    pixel_values = inputs["pixel_values"].to(reg.device, torch.float32)

    caption = ""
    if want_caption:
        try:
            with torch.no_grad():
                gen_ids = reg.florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=pixel_values,
                    max_new_tokens=64,
                    num_beams=1,
                    do_sample=False,
                )
            raw = reg.florence_processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            parsed = reg.florence_processor.post_process_generation(
                raw, task=CAPTION_PROMPT, image_size=(image.width, image.height)
            )
            caption = (parsed.get(CAPTION_PROMPT, "") or "").strip()
            if not caption:
                caption = raw.replace("</s>", "").replace("<s>", "").strip()
        except Exception as exc:
            log.warning("Florence caption failed: %s", exc)
            caption = ""

    emb = _florence_embedding(reg, pixel_values)
    with torch.no_grad():
        logit = reg.florence_head(emb)
        prob = float(torch.sigmoid(logit).item())

    image.close()
    return prob, emb.detach().cpu().numpy().reshape(-1), caption


# ---------------------------------------------------------------------------
# Stage 3: meta feature vector - notebook cell 8
# ---------------------------------------------------------------------------
def build_meta_features(cnn_prob, cnn_feat, florence_prob, florence_emb):
    cnn_confidence = abs(cnn_prob - 0.5)
    cnn_agreement = float(cnn_confidence > 0.3)
    florence_confidence = abs(florence_prob - 0.5) * 2
    agree = float((cnn_prob > 0.5) == (florence_prob > 0.5))

    vec = np.concatenate(
        [
            np.array([cnn_prob, cnn_confidence, cnn_agreement], dtype=np.float32),
            np.asarray(cnn_feat, dtype=np.float32),
            np.array([florence_prob, florence_confidence], dtype=np.float32),
            np.asarray(florence_emb, dtype=np.float32),
            np.array([agree], dtype=np.float32),
        ]
    ).reshape(1, -1)

    if vec.shape[1] != config.META_FEATURE_DIM:
        raise RuntimeError(
            f"Meta feature vector is {vec.shape[1]}-D, expected {config.META_FEATURE_DIM}-D. "
            "CNN_FEATURE_DIM / FLORENCE_EMBED_DIM in config do not match the artifacts."
        )

    return vec


def severity_for(final_label: int, confidence: float) -> str:
    """Notebook cell 14 thresholds."""
    if confidence >= 0.85:
        return "Severe"
    if confidence >= 0.65:
        return "Moderate"
    if final_label == 1:
        return "Mild"
    return "N/A"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(
    reg: ModelRegistry,
    raw_signal: np.ndarray,
    *,
    explain: bool = True,
    use_vlm: bool | None = None,
    use_llm: bool | None = None,
    source_name: str = "signal",
    record_id: str | None = None,
) -> dict:
    """Score one signal. Returns the dict the API serialises."""
    if not reg.cnn_ready:
        raise RuntimeError(
            "The CNN is not loaded, so no prediction can be made. Check /health for "
            "which artifacts are missing."
        )

    started = time.perf_counter()
    record_id = record_id or uuid.uuid4().hex[:12]
    timings: dict[str, float] = {}

    # --- preprocess (for the rendered image only; the CNN gets the raw signal)
    t0 = time.perf_counter()
    raw = np.asarray(raw_signal, dtype=np.float32).flatten()
    processed = preprocess_signal(raw)
    abnormal_regions, seg_scores = find_abnormal_regions(processed)
    timings["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # --- signal image ------------------------------------------------------
    t0 = time.perf_counter()
    signal_image = config.OUTPUT_DIR / f"{record_id}_signal.png"
    render_signal_image(processed, abnormal_regions, signal_image, title=source_name)
    timings["render_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # --- audio samples -----------------------------------------------------
    # The same preprocessed trace the image shows, dumped as raw little-endian
    # float32 so the browser can play it through the Web Audio API. Purely an
    # output artifact: nothing downstream reads it back.
    audio_path = config.OUTPUT_DIR / f"{record_id}_signal.f32"
    audio_url = None
    try:
        audio_path.write_bytes(np.ascontiguousarray(processed, dtype="<f4").tobytes())
        audio_url = f"/audio/{record_id}"
    except OSError as exc:
        log.warning("could not write audio samples for %s: %s", record_id, exc)

    # --- CNN ---------------------------------------------------------------
    t0 = time.perf_counter()
    cnn_prob, cnn_feat = run_cnn(reg, processed)
    timings["cnn_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # --- Florence-2 --------------------------------------------------------
    want_vlm = config.FLORENCE_DEFAULT if use_vlm is None else bool(use_vlm)
    florence_prob = None
    florence_emb = None
    caption = None
    vlm_note = None

    if want_vlm and config.ENABLE_FLORENCE:
        if not reg.florence_ready:
            vlm_note = "Florence-2 was requested but is not loaded; see /health."
        else:
            t0 = time.perf_counter()
            try:
                florence_prob, florence_emb, caption = run_florence(reg, signal_image)
                timings["florence_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            except Exception as exc:
                log.exception("Florence inference failed")
                vlm_note = f"Florence-2 inference failed: {exc}"
                florence_prob = None
                florence_emb = None
    elif want_vlm and not config.ENABLE_FLORENCE:
        vlm_note = "Florence-2 is disabled on this server (ENABLE_FLORENCE=0)."

    # --- fusion ------------------------------------------------------------
    meta_features = None
    meta_prob = None
    fusion = "cnn_only"
    fusion_note = None

    if florence_prob is not None and reg.meta_ready:
        try:
            meta_features = build_meta_features(cnn_prob, cnn_feat, florence_prob, florence_emb)
            t0 = time.perf_counter()
            meta_prob = float(reg.meta_learner.predict_proba(meta_features)[:, 1][0])
            timings["meta_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            fusion = "cnn+florence+meta"
        except Exception as exc:
            log.exception("Meta-learner inference failed")
            fusion_note = f"Meta-learner failed, falling back to the CNN: {exc}"
    elif florence_prob is not None and not reg.meta_ready:
        fusion = "cnn+florence"
        fusion_note = "meta_learner.pkl is not loaded, so the CNN carries the decision."
    elif not want_vlm:
        fusion_note = "Fast path: CNN only. Turn on the vision model for the fused result."
    else:
        fusion_note = vlm_note

    final_prob = meta_prob if meta_prob is not None else cnn_prob
    final_label = int(final_prob > 0.5)
    final_confidence = final_prob if final_label == 1 else 1.0 - final_prob

    models_agree = None
    if florence_prob is not None:
        models_agree = bool((cnn_prob > 0.5) == (florence_prob > 0.5))

    result = {
        "id": record_id,
        "source": source_name,
        "final_prediction": "ALS" if final_label == 1 else "Normal",
        "final_probability": round(float(final_prob), 6),
        "final_confidence": round(float(final_confidence), 6),
        "severity": severity_for(final_label, float(final_prob)),
        "cnn_probability": round(float(cnn_prob), 6),
        "cnn_prediction": "ALS" if cnn_prob > 0.5 else "Normal",
        "florence_probability": None if florence_prob is None else round(float(florence_prob), 6),
        "florence_prediction": (
            None if florence_prob is None else ("ALS" if florence_prob > 0.5 else "Normal")
        ),
        "florence_confidence": (
            None if florence_prob is None else round(abs(float(florence_prob) - 0.5) * 2, 6)
        ),
        "florence_caption": caption or None,
        "meta_probability": None if meta_prob is None else round(float(meta_prob), 6),
        "models_agree": models_agree,
        "fusion": fusion,
        "fusion_note": fusion_note,
        "abnormal_segments": f"{len(abnormal_regions)}/{config.N_SEGMENTS}",
        "abnormal_segment_count": len(abnormal_regions),
        "n_segments": config.N_SEGMENTS,
        "segment_energies": [round(float(s), 6) for s in seg_scores],
        "signal_length": len(processed),
        "signal_image_url": f"/images/{signal_image.name}",
        "audio_url": audio_url,
        "audio_sample_rate": float(config.FS),
        "gradcam_image_url": None,
        "top_shap_feature": None,
        "top_shap_features": None,
        "explanation": None,
        "explain": bool(explain),
        "demo_mode": reg.demo_mode,
        "device": reg.device,
        "disclaimer": config.DISCLAIMER,
        "timings": timings,
    }

    # --- explainability ----------------------------------------------------
    if explain:
        from explain import attach_explanations

        attach_explanations(
            reg,
            result,
            raw=raw,
            processed=processed,
            abnormal_regions=abnormal_regions,
            meta_features=meta_features,
            record_id=record_id,
            timings=timings,
            use_llm=use_llm,
        )

    if result["explanation"] is None:
        from explain import build_template_explanation

        result["explanation"] = build_template_explanation(result)

    result["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result
