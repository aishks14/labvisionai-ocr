"""
LabVisionAI — Deterministic table & header parser
===================================================
Given a cropped region image (a header block or a results table, found
by the coarse 2-class YOLO detector), extracts structured data via OCR
word positions and layout rules — no per-field ML classification.

Why: classifying every single cell (test_name/value/unit/range) as its
own YOLO class needs a large, diverse training set to get reliable
per-field recall and tight box regression. Detecting just two coarse
regions is a much easier task for a small model, and everything inside
each region is then parsed deterministically by reading actual pixel
positions — the same technique already proven in auto_annotate.py.

Shared by the customer inference pipeline (core/pipeline.py) and the
admin auto-annotation assistant (core/auto_annotate.py).
"""

from __future__ import annotations

import re

import numpy as np
import pytesseract
from pytesseract import Output

from config.settings import OCR_LANG, TESSERACT_CMD
from core.parser import clean_text, compute_flag, normalize_value, split_unit_range

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

GAP_THRESHOLD = 35  # px gap that separates two columns/fields on the same line

_NAME_AGE_GENDER = re.compile(
    r"^[Ss]?\s*(.*?)\s*\(?\s*([\dOoSsIlB]{1,3})\s*Y\s*/\s*([A-Za-z])\)?\s*$", re.I)
_AGE_FIXES = str.maketrans({"O": "0", "o": "0", "S": "5", "s": "5",
                            "I": "1", "l": "1", "B": "8"})
_LABEL_NOISE = re.compile(r"^[:+;|=\s]+")

_RANGE_RE = re.compile(r"^[<>]?\s*[\d.]+\s*([-–to]+\s*[\d.]+)?$", re.I)
_VALUE_RE = re.compile(r"^[\d.]+$")

# Known lab methodology names — dropped from test_name if found sitting
# between the test name and the value column. Finite, well-documented
# vocabulary, so a lookup here is a legitimate rule, not a guess.
_TECHNOLOGY_WORDS = {
    "photometry", "calculated", "elisa", "clia", "c.l.i.a", "cliaa",
    "immunoassay", "chemiluminescence", "turbidimetry", "nephelometry",
    "electrophoresis", "flowcytometry", "flow cytometry", "ise",
}


def get_lines(image: np.ndarray) -> list[dict]:
    """OCR a region and group words into lines with word-level boxes
    and Tesseract's own per-word confidence scores."""
    data = pytesseract.image_to_data(image, lang=OCR_LANG, config="--psm 6",
                                     output_type=Output.DICT)
    lines: dict[tuple, dict] = {}
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        lines.setdefault(key, {"words": []})["words"].append(
            {"text": text, "x1": x, "y1": y, "x2": x + w, "y2": y + h, "conf": conf})

    result = []
    for L in lines.values():
        L["words"].sort(key=lambda w: w["x1"])
        L["y1"] = min(w["y1"] for w in L["words"])
        L["y2"] = max(w["y2"] for w in L["words"])
        result.append(L)
    result.sort(key=lambda L: L["y1"])
    return result


def cluster_columns(words: list[dict]) -> list[dict]:
    """Split a line's words into column clusters based on x-gaps."""
    clusters, current = [], [words[0]]
    for prev, cur in zip(words, words[1:]):
        if cur["x1"] - prev["x2"] > GAP_THRESHOLD:
            clusters.append(current)
            current = [cur]
        else:
            current.append(cur)
    clusters.append(current)

    return [{
        "text": " ".join(w["text"] for w in c),
        "x1": min(w["x1"] for w in c), "y1": min(w["y1"] for w in c),
        "x2": max(w["x2"] for w in c), "y2": max(w["y2"] for w in c),
    } for c in clusters]


def _mean_confidence(lines: list[dict]) -> float:
    """Average Tesseract's own per-word confidence across a set of lines."""
    confs = [w["conf"] for L in lines for w in L["words"] if w["conf"] >= 0]
    return round(sum(confs) / len(confs), 1) if confs else 0.0


def parse_header_region(image: np.ndarray) -> tuple[dict, float]:
    """
    Deterministically extract patient_name/age/gender/doctor_name/
    report_date from a cropped header-block region. Returns
    (header_dict, mean_ocr_confidence).
    """
    lines = get_lines(image)
    header: dict = {}
    for L in lines:
        clusters = cluster_columns(L["words"])
        for i, c in enumerate(clusters):
            t = clean_text(c["text"])
            if not t:
                continue
            up = t.upper()

            if up.startswith("NAME") and "TEST" not in up and i + 1 < len(clusters):
                value = _LABEL_NOISE.sub("", clean_text(clusters[i + 1]["text"]))
                m = _NAME_AGE_GENDER.match(value)
                if m:
                    name, age, gender = m.groups()
                    name = re.sub(r"^\d{1,2}\s+", "", name.strip())
                    age = age.translate(_AGE_FIXES)
                    if len(name.strip()) >= 2 and age.isdigit():
                        header.setdefault("patient_name", name.strip())
                        header.setdefault("age", age)
                        header.setdefault("gender", gender.upper())
                elif value:
                    header.setdefault("patient_name", value)

            elif up.startswith("REF") and i + 1 < len(clusters):
                value = _LABEL_NOISE.sub("", clean_text(clusters[i + 1]["text"]))
                value = re.sub(r"^\d{1,2}\s+", "", value)
                if value:
                    header.setdefault("doctor_name", value)

            elif up.startswith("DATE") and i + 1 < len(clusters):
                value = _LABEL_NOISE.sub("", clean_text(clusters[i + 1]["text"]))
                value = re.sub(r"^\d{1,2}\s+", "", value)
                if value:
                    header.setdefault("report_date", value)

    return header, _mean_confidence(lines)


_NOISE_STRIP = re.compile(r"[^0-9A-Za-z.<>\-–/ ]+")


def _clean_field(text: str) -> str:
    """Strip stray OCR punctuation noise (°, !, :, ;, |, etc.) while
    keeping the characters that actually carry meaning in a value,
    unit, or range field. Converts comma-for-period OCR misreads
    (e.g. '0,3-1,2') instead of silently dropping the comma, which
    would otherwise corrupt the number (0.3 -> 03)."""
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return _NOISE_STRIP.sub("", text).strip(" .-")


def _classify_row_clusters(clusters: list[dict]) -> dict | None:
    """
    Map a table row's column clusters to fields using position PLUS a
    minimal content check (does this cluster contain a digit) — pure
    cluster-count heuristics break because the same count can mean
    different things: e.g. 4 clusters might be [test_name, value,
    unit, range] on one report layout, or [test_name, technology,
    value, unit+range-merged] on another. A technology name never
    contains a digit; a value always does — that one signal is enough
    to disambiguate reliably without going back to fragile full
    pattern-matching on noisy OCR text.
    """
    texts = [clean_text(c["text"]) for c in clusters]
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return None

    test_name = texts[0]
    idx = 1

    # Optional technology column: present if the next cluster has no
    # digit in it at all (a real value always does).
    if idx < len(texts) and not any(ch.isdigit() for ch in texts[idx]):
        idx += 1  # drop it — not one of our output fields

    if idx >= len(texts):
        return None
    value = texts[idx]
    idx += 1

    remaining = texts[idx:]
    if len(remaining) >= 2:
        unit, rng = remaining[0], remaining[1]
    elif len(remaining) == 1:
        split = split_unit_range(remaining[0])
        if split:
            unit, rng = split
        else:
            unit, rng = remaining[0], ""
    else:
        unit, rng = "", ""

    value = normalize_value(_clean_field(value))
    unit = _clean_field(unit)
    rng = _clean_field(rng)

    if not test_name or not any(ch.isdigit() for ch in value):
        return None

    return {"test_name": test_name, "value": value, "unit": unit,
           "reference_range": rng}


def parse_table_region(image: np.ndarray) -> tuple[list[dict], float]:
    """
    Deterministically extract test rows from a cropped results-table
    region — every OCR'd line is a candidate row, so nothing is
    skipped the way a per-row YOLO detection could be missed. Returns
    (rows, mean_ocr_confidence).
    """
    lines = get_lines(image)

    header_idx = None
    for i, L in enumerate(lines):
        line_text = " ".join(w["text"] for w in L["words"]).upper()
        if ("TEST" in line_text and ("VALUE" in line_text or "TECHNOLOGY" in line_text)
                or ("VALUE" in line_text and "UNIT" in line_text)):
            header_idx = i
            break

    data_lines = lines[header_idx + 1:] if header_idx is not None else lines

    rows = []
    for L in data_lines:
        clusters = cluster_columns(L["words"])
        row = _classify_row_clusters(clusters)
        if row is None:
            continue
        row["flag"] = compute_flag(row["value"], row["reference_range"])
        rows.append(row)

    return rows, _mean_confidence(lines)