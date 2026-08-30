"""
Wiring check for the inference pipeline. No trained artifacts required.

It runs the real code paths - preprocessing, CNN forward, the 838-D feature
assembly, XGBoost fusion, Grad-CAM, SHAP, the explanation and the PDF - using
placeholder weights and a throwaway meta-learner fitted on random noise. It
proves the plumbing holds together and the shapes line up. It says nothing about
model accuracy.

    python smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("DEMO_MODE", "1")
os.environ.setdefault("ENABLE_FLORENCE", "0")

import numpy as np

import config
import models as model_module
import pipeline
from report import build_report

PASS = "  ok   "
FAIL = "  FAIL "
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL} {name}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(name)


def main() -> int:
    print("ALS pipeline smoke test (placeholder weights - accuracy is meaningless)\n")

    reg = model_module.load_all()
    check("CNN instantiated", reg.cnn_ready)

    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, config.SIGNAL_LENGTH).astype(np.float32)

    # --- preprocessing -----------------------------------------------------
    processed = pipeline.preprocess_signal(signal)
    check("preprocess keeps length", processed.shape[0] == config.SIGNAL_LENGTH)
    check("preprocess z-scores", abs(float(processed.mean())) < 1e-3, f"mean={processed.mean():.2e}")

    regions, scores = pipeline.find_abnormal_regions(processed)
    check("segment scores", len(scores) == config.N_SEGMENTS)
    check("abnormal regions within bounds", all(0 <= s < e <= len(processed) for s, e in regions))

    # --- CNN ---------------------------------------------------------------
    prob, feat = pipeline.run_cnn(reg, signal)
    check("CNN probability in [0,1]", 0.0 <= prob <= 1.0, f"p={prob:.4f}")
    check("CNN feature vector is 64-D", feat.shape == (config.CNN_FEATURE_DIM,), str(feat.shape))

    prob2, _ = pipeline.run_cnn(reg, signal)
    check("CNN is deterministic", prob == prob2, f"{prob} vs {prob2}")

    # --- fusion ------------------------------------------------------------
    fake_emb = rng.normal(0, 1, config.FLORENCE_EMBED_DIM).astype(np.float32)
    vec = pipeline.build_meta_features(prob, feat, 0.61, fake_emb)
    check(
        f"meta vector is {config.META_FEATURE_DIM}-D",
        vec.shape == (1, config.META_FEATURE_DIM),
        str(vec.shape),
    )
    check("meta vector column order", vec[0, 0] == np.float32(prob), "col 0 is CNN_Prob")
    check(
        "meta vector Florence slot",
        abs(float(vec[0, 3 + config.CNN_FEATURE_DIM]) - 0.61) < 1e-6,
        "col 67 is Florence_Prob",
    )
    check("feature names match width", len(config.FEATURE_NAMES) == config.META_FEATURE_DIM)

    try:
        import xgboost as xgb

        throwaway = xgb.XGBClassifier(n_estimators=8, max_depth=2, eval_metric="logloss")
        X = rng.normal(0, 1, (40, config.META_FEATURE_DIM))
        throwaway.fit(X, (rng.random(40) > 0.5).astype(int))
        reg.meta_learner = throwaway

        import shap

        reg.shap_explainer = shap.TreeExplainer(throwaway)
        meta_p = float(throwaway.predict_proba(vec)[:, 1][0])
        check("XGBoost accepts the vector", 0.0 <= meta_p <= 1.0, f"p={meta_p:.4f}")
    except Exception as exc:
        check("XGBoost / SHAP wiring", False, str(exc))

    # --- severity ----------------------------------------------------------
    check("severity Severe", pipeline.severity_for(1, 0.91) == "Severe")
    check("severity Moderate", pipeline.severity_for(1, 0.70) == "Moderate")
    check("severity Mild", pipeline.severity_for(1, 0.55) == "Mild")
    check("severity N/A for Normal", pipeline.severity_for(0, 0.40) == "N/A")

    # --- full run + explainability ----------------------------------------
    result = pipeline.run_pipeline(reg, signal, explain=True, use_vlm=False, source_name="smoke")
    check("pipeline returns a verdict", result["final_prediction"] in {"ALS", "Normal"})
    check("Grad-CAM produced", result.get("gradcam_available") is True)
    check("Grad-CAM profile has 60 buckets", len(result.get("gradcam_profile", [])) == 60)
    check("signal image written", (config.OUTPUT_DIR / f"{result['id']}_signal.png").exists())
    check("annotated image written", (config.OUTPUT_DIR / f"{result['id']}_annotated.png").exists())
    check("explanation carries the disclaimer", "not a medical diagnosis" in result["explanation"])
    check("demo mode is flagged", result["demo_mode"] is True)

    # --- PDF ---------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path

        pdf = build_report(result, Path(tmp) / "r.pdf")
        check("PDF generated", pdf.exists() and pdf.stat().st_size > 5000, f"{pdf.stat().st_size} B")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed. (Wiring only - load the trained artifacts for real results.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
