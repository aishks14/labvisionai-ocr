"""
LabVisionAI — Exporters
========================
Turns one extraction record into the deliverables customers actually
want: Excel (styled), CSV, JSON, PDF (ReportLab), and FHIR R4
DiagnosticReport+Observation JSON for hospital interoperability.
"""

import io
import json
import re


def _rows_frame(record: dict):
    import pandas as pd
    rows = record.get("rows", [])
    cols = ["test_name", "value", "unit", "reference_range", "flag"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].rename(columns={
        "test_name": "Test Name", "value": "Value", "unit": "Unit",
        "reference_range": "Reference Range", "flag": "Flag"})


def to_csv(record: dict) -> bytes:
    return _rows_frame(record).to_csv(index=False).encode()


def to_json(record: dict) -> bytes:
    payload = {"patient": record.get("header", {}), "results": record.get("rows", [])}
    return json.dumps(payload, indent=2).encode()


def to_excel(record: dict) -> bytes:
    """Two-sheet styled workbook: Patient Info + Test Results."""
    import pandas as pd
    from openpyxl.styles import Font, PatternFill

    buf = io.BytesIO()
    header = record.get("header", {})
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([{"Field": k.replace("_", " ").title(), "Value": v}
                      for k, v in header.items()]
                     ).to_excel(writer, sheet_name="Patient Info", index=False)
        _rows_frame(record).to_excel(writer, sheet_name="Test Results", index=False)

        for sheet in writer.sheets.values():
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E79")
            for col in sheet.columns:
                width = max((len(str(c.value or "")) for c in col), default=8)
                sheet.column_dimensions[col[0].column_letter].width = min(width + 3, 45)
    return buf.getvalue()


def to_pdf(record: dict) -> bytes:
    """Clean single-page(ish) PDF report via ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("LabVisionAI — Extracted Lab Report", styles["Title"]),
             Spacer(1, 6 * mm)]

    for key, value in record.get("header", {}).items():
        story.append(Paragraph(
            f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles["Normal"]))
    story.append(Spacer(1, 6 * mm))

    data = [["Test Name", "Value", "Unit", "Reference Range", "Flag"]]
    for r in record.get("rows", []):
        data.append([r.get("test_name", ""), r.get("value", ""), r.get("unit", ""),
                     r.get("reference_range", ""), r.get("flag", "")])
    table = Table(data, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("FONTSIZE", (0, 0), (-1, -1), 8.5),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1),
              [colors.white, colors.HexColor("#EEF3F8")])]
    for i, r in enumerate(record.get("rows", []), start=1):
        if r.get("flag") in ("HIGH", "LOW"):
            style.append(("TEXTCOLOR", (4, i), (4, i), colors.red))
    table.setStyle(TableStyle(style))
    story.append(table)
    doc.build(story)
    return buf.getvalue()


def to_fhir(record: dict) -> bytes:
    """FHIR R4 Bundle: DiagnosticReport + one Observation per test row."""
    header = record.get("header", {})
    observations, refs = [], []
    for i, r in enumerate(record.get("rows", []), start=1):
        m = re.search(r"[\d.]+", r.get("value") or "")
        obs = {
            "resourceType": "Observation", "id": f"obs-{i}", "status": "final",
            "code": {"text": r.get("test_name", "")},
            "referenceRange": [{"text": r.get("reference_range", "")}],
        }
        if m:
            obs["valueQuantity"] = {"value": float(m.group()),
                                    "unit": r.get("unit", "")}
        else:
            obs["valueString"] = r.get("value", "")
        observations.append({"resource": obs})
        refs.append({"reference": f"Observation/obs-{i}"})

    bundle = {
        "resourceType": "Bundle", "type": "collection",
        "entry": [{"resource": {
            "resourceType": "DiagnosticReport", "status": "final",
            "code": {"text": "Laboratory Report (LabVisionAI extraction)"},
            "subject": {"display": header.get("patient_name", "Unknown")},
            "result": refs}}] + observations,
    }
    return json.dumps(bundle, indent=2).encode()


EXPORTERS = {"csv": (to_csv, "text/csv"),
             "json": (to_json, "application/json"),
             "xlsx": (to_excel,
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
             "pdf": (to_pdf, "application/pdf"),
             "fhir": (to_fhir, "application/fhir+json")}


def export(record: dict, fmt: str) -> tuple[bytes, str, str]:
    """Return (bytes, mime_type, extension) for the requested format."""
    if fmt not in EXPORTERS:
        raise ValueError(f"Unknown format: {fmt}")
    fn, mime = EXPORTERS[fmt]
    ext = "json" if fmt == "fhir" else fmt
    return fn(record), mime, ext
