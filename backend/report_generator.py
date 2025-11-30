# backend/report_generator.py

import os
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# For simple embedded charts
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart


# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------


def _get_styles():
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        fontName="Helvetica-Bold",
        fontSize=24,
        alignment=1,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=12,
    )

    section_header = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=colors.HexColor("#1565C0"),
        spaceAfter=8,
    )

    sub_header = ParagraphStyle(
        "SubHeader",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=4,
    )

    normal = ParagraphStyle(
        "Normal",
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#212121"),
    )

    small_grey = ParagraphStyle(
        "SmallGrey",
        fontName="Helvetica",
        fontSize=8.5,
        alignment=1,
        textColor=colors.grey,
        leading=10,
    )

    return {
        "title": title_style,
        "section": section_header,
        "sub": sub_header,
        "normal": normal,
        "small_grey": small_grey,
    }


def _baseline_risk(disease: str) -> float:
    """Rough reference risk to show in bar chart."""
    d = disease.lower()
    if d == "heart":
        return 12.0  # population CAD prevalence-ish
    if d == "diabetes":
        return 10.0
    if d == "breast":
        return 5.0
    if d == "brain":
        return 1.0
    return 5.0


def _make_risk_chart(prob_pct: float, disease: str) -> Drawing:
    """
    Creates a simple vertical bar chart comparing:
    - Your estimated risk vs
    - A rough reference baseline
    """
    baseline = _baseline_risk(disease)

    drawing = Drawing(300, 180)

    chart = VerticalBarChart()
    chart.x = 40
    chart.y = 40
    chart.width = 220
    chart.height = 100

    chart.data = [[prob_pct, baseline]]
    chart.categoryAxis.categoryNames = ["Your risk", "Reference"]
    chart.categoryAxis.labels.boxAnchor = "n"
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(100, max(prob_pct, baseline) + 10)
    chart.valueAxis.valueStep = 20

    chart.bars[0].fillColor = colors.HexColor("#2196F3")
    if len(chart.data[0]) > 1:
        chart.bars[1].fillColor = colors.HexColor("#90CAF9")

    # Title above chart
    title = String(
        150,
        150,
        "Estimated risk vs typical population",
        fontName="Helvetica",
        fontSize=10,
        textAnchor="middle",
        fillColor=colors.HexColor("#0D47A1"),
    )

    drawing.add(chart)
    drawing.add(title)
    return drawing


def _disease_context_paragraph(disease: str) -> str:
    """Short explanation block for each disease."""
    d = disease.lower()
    if d == "heart":
        return (
            "This section summarises the model’s estimate of coronary artery disease (CAD) "
            "risk based on your entered heart-related parameters (blood pressure, cholesterol, "
            "exercise tolerance and related features). The score is intended to support, not "
            "replace, a full clinical assessment."
        )
    if d == "diabetes":
        return (
            "This section summarises the estimated likelihood of type 2 diabetes or prediabetes "
            "using lifestyle, weight, blood pressure and self-reported health indicators. "
            "The result helps identify individuals who may benefit from earlier screening or intervention."
        )
    if d == "breast":
        return (
            "This section reflects the model’s interpretation of breast mass features derived from imaging "
            "or pathology (size, shape and border irregularities). The prediction differentiates between "
            "patterns more typical of benign versus malignant lesions."
        )
    if d == "brain":
        return (
            "This section summarises the model’s interpretation of your brain MRI slice, including "
            "whether the pattern resembles common tumor types such as glioma, meningioma or pituitary "
            "lesions. Final decisions must always be made by a neurologist or neurosurgeon."
        )
    return (
        "This section summarises the model’s risk estimation for the evaluated condition. "
        "The output is designed as a decision-support aid and must be reviewed alongside "
        "clinical history, examination and additional testing."
    )


# ---------------------------------------------------------
# MAIN PDF GENERATOR
# ---------------------------------------------------------


def generate_pdf(report_id: str, data: Dict[str, Any]) -> str:
    """
    Creates a hospital-style multi-page AI diagnostic report.

    data should contain:
    - disease: str
    - label: str
    - probability: float (0–1)
    - advisory: str
    - gradcam_path: str (optional)
    - comparisons: List[dict] (optional, for tabular models)
    """

    os.makedirs("backend/reports", exist_ok=True)
    pdf_path = f"backend/reports/{report_id}.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=40,
        bottomMargin=45,
        title="AI Diagnostic Report",
    )

    styles = _get_styles()
    story: List[Any] = []

    disease = str(data.get("disease", "Unknown")).capitalize()
    label = str(data.get("label", "N/A"))
    prob_pct = round(float(data.get("probability", 0.0)) * 100, 2)

    # =========================================================
    # PAGE 1 – SUMMARY & RISK VIEW
    # =========================================================

    # Header bar
    story.append(
        Paragraph(
            '<para alignment="center"><font color="#0D47A1"><b>'
            '═══════════════════════════════════════════════════'
            "</b></font></para>",
            styles["normal"],
        )
    )
    story.append(Spacer(1, 6))

    story.append(Paragraph("AI Diagnostic Report", styles["title"]))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"<b>Disease Module:</b> {disease}  &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Prediction:</b> {label}",
            styles["normal"],
        )
    )
    story.append(Spacer(1, 12))

    # Summary table
    story.append(Paragraph("Summary Findings", styles["section"]))
    story.append(Spacer(1, 4))

    summary_data = [
        ["Condition evaluated", disease],
        ["Prediction label", label],
        ["Estimated probability", f"{prob_pct} %"],
    ]

    summary_table = Table(summary_data, colWidths=[170, 270])
    summary_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eceff1")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # Disease-specific context
    story.append(Paragraph("Clinical Context", styles["section"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph(_disease_context_paragraph(disease), styles["normal"]))
    story.append(Spacer(1, 16))

    # Risk chart for tabular modules (heart / diabetes / breast)
    if disease.lower() in {"heart", "diabetes", "breast"}:
        story.append(Paragraph("Risk Visualisation", styles["section"]))
        story.append(Spacer(1, 4))
        drawing = _make_risk_chart(prob_pct, disease)
        story.append(KeepTogether([drawing, Spacer(1, 6)]))

        story.append(
            Paragraph(
                "The bar chart compares your estimated risk with a rough reference risk in a "
                "general adult population. It is not a formal epidemiological estimate but "
                "helps visualise relative risk.",
                styles["small_grey"],
            )
        )
        story.append(Spacer(1, 10))

    # End of page 1
    story.append(PageBreak())

    # =========================================================
    # PAGE 2 – IMAGING & FEATURE ANALYSIS
    # =========================================================
    grad_present = bool(data.get("gradcam_path"))
    comparisons: List[Dict[str, Any]] = list(data.get("comparisons", []))

    if grad_present or comparisons:
        story.append(Paragraph("Model Evidence & Feature Analysis", styles["section"]))
        story.append(Spacer(1, 8))

    # Grad-CAM (for imaging models)
    if grad_present:
        story.append(Paragraph("Imaging Focus Map (Grad-CAM)", styles["sub"]))
        story.append(Spacer(1, 4))

        grad_path = "backend" + data["gradcam_path"]
        if os.path.exists(grad_path):
            img = Image(grad_path, width=360, height=360)
            img.hAlign = "CENTER"
            story.append(KeepTogether([img, Spacer(1, 10)]))
        else:
            story.append(Paragraph("<i>Grad-CAM image unavailable.</i>", styles["normal"]))
            story.append(Spacer(1, 10))

        story.append(
            Paragraph(
                "The Grad-CAM map highlights regions of the image that contributed most strongly "
                "to the model’s decision. Bright or warm colours often indicate regions with higher influence.",
                styles["small_grey"],
            )
        )
        story.append(Spacer(1, 16))

    # Feature comparison table (tabular models)
    if comparisons:
        story.append(Paragraph("Key Feature Comparisons", styles["sub"]))
        story.append(Spacer(1, 4))

        table_data = [["Feature", "Your value", "Normal min", "Normal max", "Status"]]
        for c in comparisons:
            table_data.append(
                [
                    c.get("feature", ""),
                    str(c.get("user_value", "")),
                    str(c.get("normal_min", "")),
                    str(c.get("normal_max", "")),
                    str(c.get("status", "")),
                ]
            )

        comp_table = Table(
            table_data,
            colWidths=[110, 70, 70, 70, 90],
            repeatRows=1,
        )
        comp_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E3F2FD")),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(comp_table)
        story.append(Spacer(1, 8))

        story.append(
            Paragraph(
                "Values outside the typical reference range are flagged in the status column. "
                "These ranges are approximate and may differ from laboratory-specific reference values.",
                styles["small_grey"],
            )
        )

    if grad_present or comparisons:
        story.append(PageBreak())

    # =========================================================
    # PAGE 3 – ADVISORY & DISCLAIMER
    # =========================================================
    story.append(Paragraph("AI-Generated Advisory", styles["section"]))
    story.append(Spacer(1, 4))

    advisory_text = str(data.get("advisory", "No advisory text available.")).strip()

    if not advisory_text:
        advisory_text = "No advisory text available."

    # Advisory may contain headings and bullet-like sections from GPT-2
    for para in advisory_text.split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), styles["normal"]))
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 16))

    # Disclaimer & usage note
    story.append(Paragraph("Important Notes", styles["section"]))
    story.append(Spacer(1, 4))

    disclaimer = (
        "This report was generated by an AI-based clinical decision-support system. "
        "The outputs are probabilistic estimates and pattern interpretations based on the data "
        "you provided and model training datasets. They do NOT constitute a diagnosis.\n\n"
        "All decisions about investigations, treatment and follow-up must be made by a licensed "
        "healthcare professional who can review your full medical history, examination findings "
        "and additional test results. If you have concerning or rapidly worsening symptoms, "
        "seek urgent medical care regardless of the values shown in this report."
    )

    story.append(Paragraph(disclaimer, styles["normal"]))
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "© Hybrid AI Diagnostics System – generated automatically. ",
            styles["small_grey"],
        )
    )

    # Build PDF
    doc.build(story)
    return pdf_path
