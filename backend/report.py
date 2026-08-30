"""
PDF report generation (ReportLab). Ported from notebook cell 16 and extended to
carry the evidence table and the disclaimer, which the notebook version omitted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus import (
    Image as RLImage,
)

import config

INK = colors.HexColor("#12161C")
MUTED = colors.HexColor("#5A6472")
RULE = colors.HexColor("#D5DAE1")
ALS_COLOR = colors.HexColor("#B4342A")
NORMAL_COLOR = colors.HexColor("#1E6F5C")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontSize=20, textColor=INK, spaceAfter=2
        ),
        "sub": ParagraphStyle(
            "ReportSub", parent=base["Normal"], fontSize=9, textColor=MUTED, spaceAfter=14
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontSize=11,
            textColor=INK,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ReportBody", parent=base["BodyText"], fontSize=9.5, leading=14, textColor=INK
        ),
        "verdict": ParagraphStyle(
            "Verdict", parent=base["Heading1"], fontSize=24, spaceAfter=2, spaceBefore=4
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer",
            parent=base["BodyText"],
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#7A2F27"),
            backColor=colors.HexColor("#FBF0EE"),
            borderPadding=8,
            spaceBefore=10,
        ),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "not run"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(result: dict, out_path: Path) -> Path:
    s = _styles()
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        title=f"ALS screening report {result['id']}",
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    story = []
    story.append(Paragraph("EMG ALS screening report", s["title"]))
    story.append(
        Paragraph(
            f"Record {result['id']} &nbsp;·&nbsp; {result.get('source', 'signal')} "
            f"&nbsp;·&nbsp; generated "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            s["sub"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=RULE))

    if result.get("demo_mode"):
        story.append(
            Paragraph(
                "<b>DEMO MODE.</b> This server is running with untrained placeholder "
                "weights. The numbers below are not a real screening result.",
                s["disclaimer"],
            )
        )

    verdict_style = ParagraphStyle(
        "V",
        parent=s["verdict"],
        textColor=ALS_COLOR if result["final_prediction"] == "ALS" else NORMAL_COLOR,
    )
    story.append(Paragraph(result["final_prediction"], verdict_style))
    story.append(
        Paragraph(
            f"Confidence {result['final_confidence'] * 100:.1f}% &nbsp;·&nbsp; "
            f"severity {result['severity']} &nbsp;·&nbsp; decided by {result['fusion']}",
            s["sub"],
        )
    )

    # --- evidence table ----------------------------------------------------
    story.append(Paragraph("Model evidence", s["h2"]))
    rows = [
        ["Signal", "ALS probability", "Reading"],
        [
            "1-D CNN (waveform)",
            _fmt(result["cnn_probability"]),
            result["cnn_prediction"],
        ],
        [
            "Florence-2 (signal image)",
            _fmt(result["florence_probability"]),
            result["florence_prediction"] or "not run",
        ],
        [
            "XGBoost meta-learner (fused)",
            _fmt(result["meta_probability"]),
            result["final_prediction"] if result["meta_probability"] is not None else "not run",
        ],
        ["Models agree", _fmt(result["models_agree"]), ""],
        ["Abnormal segments", result["abnormal_segments"], ""],
        ["Top SHAP feature", result.get("top_shap_feature") or "not computed", ""],
    ]
    table = Table(rows, colWidths=[2.6 * inch, 1.7 * inch, 2.7 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, RULE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#ECEFF3")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(table)

    # --- images ------------------------------------------------------------
    for label, url in (
        ("Signal with Grad-CAM overlay", result.get("gradcam_image_url")),
        ("Preprocessed signal", result.get("signal_image_url")),
    ):
        if not url:
            continue
        path = config.OUTPUT_DIR / Path(url).name
        if not path.exists():
            continue
        story.append(Paragraph(label, s["h2"]))
        img = RLImage(str(path))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = 7.0 * inch
        img.drawHeight = min(7.0 * inch * ratio, 3.0 * inch)
        story.append(img)

    # --- explanation -------------------------------------------------------
    story.append(Paragraph("Explanation", s["h2"]))
    story.append(Paragraph(result.get("explanation", ""), s["body"]))

    if result.get("florence_caption"):
        story.append(Paragraph("Vision model caption (unverified)", s["h2"]))
        story.append(Paragraph(result["florence_caption"], s["body"]))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))
    story.append(Paragraph(config.DISCLAIMER, s["disclaimer"]))

    doc.build(story)
    return out_path
