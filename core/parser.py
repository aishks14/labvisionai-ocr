"""
LabVisionAI — Post-processor / Parser
======================================
Turns raw (class, text, box) detections into the final structured
record: a header dict (patient info) plus clean test-result rows.
Includes noise filtering, value normalization, row assembly by
vertical alignment, and abnormal-flag computation vs reference range.
"""

import re

HEADER_FIELDS = {"patient_name", "age", "gender", "doctor_name", "report_date"}
ROW_FIELDS = {"test_name", "value", "unit", "reference_range"}

NOISE_PATTERNS = re.compile(
    r"(page \d+|thank you|end of report|www\.|http|@|barcode|"
    r"registered|lab no|sample collected)", re.I)

_VALUE_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                              "S": "5", "B": "8", ",": "."})


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" :;|-_")
    return "" if NOISE_PATTERNS.search(text) else text


def normalize_value(text: str) -> str:
    """Fix common OCR digit confusions inside numeric values."""
    candidate = text.translate(_VALUE_FIXES)
    return candidate if re.fullmatch(r"[\d.]+", candidate.replace(" ", "")) else text


def parse_range(range_text: str) -> tuple[float, float] | None:
    m = re.search(r"([\d.]+)\s*[-–to]+\s*([\d.]+)", range_text)
    if not m:
        return None
    try:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo, hi) if lo <= hi else None
    except ValueError:
        return None


def compute_flag(value_text: str, range_text: str) -> str:
    """Return LOW / HIGH / NORMAL / '' by comparing value to range."""
    bounds = parse_range(range_text)
    m = re.search(r"[\d.]+", value_text or "")
    if not bounds or not m:
        return ""
    try:
        v = float(m.group())
    except ValueError:
        return ""
    lo, hi = bounds
    return "LOW" if v < lo else "HIGH" if v > hi else "NORMAL"


def _center_y(det):
    return (det["box"][1] + det["box"][3]) / 2


def assemble_rows(detections: list[dict]) -> list[dict]:
    """
    Group row-field detections into table rows by vertical alignment:
    each test_name anchors a row; value/unit/range join whichever
    anchor's center they sit closest to (within a max band), not just
    any anchor within range — this prevents a single detection from
    being claimed by two adjacent rows when boxes sit close together.
    """
    anchors = sorted([d for d in detections if d["class"] == "test_name"],
                     key=_center_y)
    others = [d for d in detections if d["class"] in ROW_FIELDS - {"test_name"}]

    rows = [{"test_name": a["text"], "value": "", "unit": "",
            "reference_range": ""} for a in anchors]

    for det in others:
        det_y = _center_y(det)
        best_i, best_dist = None, None
        for i, anchor in enumerate(anchors):
            y1, y2 = anchor["box"][1], anchor["box"][3]
            band = max(14, (y2 - y1) * 0.8)
            dist = abs(det_y - (y1 + y2) / 2)
            if dist <= band and (best_dist is None or dist < best_dist):
                best_i, best_dist = i, dist
        if best_i is not None and not rows[best_i][det["class"]]:
            rows[best_i][det["class"]] = det["text"]

    final = []
    for row in rows:
        row["value"] = normalize_value(row["value"])
        row["flag"] = compute_flag(row["value"], row["reference_range"])
        if row["test_name"]:
            final.append(row)
    return final


def build_record(detections: list[dict]) -> dict:
    """Full post-processing: noise-clean, split header vs rows, assemble."""
    cleaned = []
    for det in detections:
        text = clean_text(det.get("text", ""))
        if text:
            cleaned.append({**det, "text": text})

    header = {}
    for det in cleaned:
        if det["class"] in HEADER_FIELDS and det["class"] not in header:
            header[det["class"]] = det["text"]

    rows = assemble_rows([d for d in cleaned if d["class"] in ROW_FIELDS])
    return {"header": header, "rows": rows}