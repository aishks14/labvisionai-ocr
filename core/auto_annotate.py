"""
LabVisionAI — Auto-annotation assistant
=========================================
Runs Tesseract on a full report page, groups words into lines/columns
by position, and applies keyword + layout rules to pre-fill candidate
bounding boxes for the known FIELD_CLASSES.

This is a bootstrapping aid, not a replacement for the trained YOLO
detector — it exists to turn "draw every box by hand" into "review
and fix the boxes that are wrong." Typical accuracy on a clean scan
is high for test_name/value/unit; header fields (name/age/gender)
and reference_range are the most likely to need a manual nudge.
"""

from __future__ import annotations

import re

import numpy as np
import pytesseract
from pytesseract import Output

from config.settings import FIELD_CLASSES, OCR_LANG, TESSERACT_CMD

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

GAP_THRESHOLD = 35  # px gap that separates two columns/fields on the same line
HEADER_ZONE_RATIO = 0.45  # top portion of the page treated as patient/header info


def _get_lines(gray_or_bgr: np.ndarray) -> list[dict]:
    """OCR the page and group words into lines with word-level boxes."""
    data = pytesseract.image_to_data(gray_or_bgr, lang=OCR_LANG,
                                     config="--psm 6", output_type=Output.DICT)

    lines: dict[tuple, dict] = {}
    n = len(data["text"])

    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        lines.setdefault(key, {"words": []})["words"].append(
            {"text": text, "x1": x, "y1": y, "x2": x + w, "y2": y + h})

    result = []
    for L in lines.values():
        L["words"].sort(key=lambda w: w["x1"])
        L["y1"] = min(w["y1"] for w in L["words"])
        L["y2"] = max(w["y2"] for w in L["words"])
        result.append(L)

    result.sort(key=lambda L: L["y1"])
    return result


def _cluster_columns(words: list[dict]) -> list[dict]:
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


def _parse_patient_block(clusters: list[dict], boxes: list[dict]) -> None:
    for i, c in enumerate(clusters):
        t = c["text"].upper()

        if t.startswith("NAME") and "TEST" not in t and i + 1 < len(clusters):
            value = clusters[i + 1]
            m = re.match(r"^(.*?)\s*\(?(\d{1,3})\s*Y\s*/\s*(\w)\)?", value["text"], re.I)
            if m:
                name_part = m.group(1).strip(" :")
                if name_part:
                    boxes.append({"class": "patient_name", "x1": value["x1"], "y1": value["y1"],
                                 "x2": value["x1"] + int((value["x2"] - value["x1"]) * 0.6),
                                 "y2": value["y2"]})
                boxes.append({"class": "age", **{k: value[k] for k in ("x1", "y1", "x2", "y2")}})
                boxes.append({"class": "gender", **{k: value[k] for k in ("x1", "y1", "x2", "y2")}})
            elif value["text"].strip(" :"):
                boxes.append({"class": "patient_name", **{k: value[k] for k in ("x1", "y1", "x2", "y2")}})

        elif t.startswith("REF") and i + 1 < len(clusters):
            value = clusters[i + 1]
            if value["text"].strip(" :"):
                boxes.append({"class": "doctor_name", **{k: value[k] for k in ("x1", "y1", "x2", "y2")}})

        elif t.startswith("DATE") and i + 1 < len(clusters):
            value = clusters[i + 1]
            if value["text"].strip(" :"):
                boxes.append({"class": "report_date", **{k: value[k] for k in ("x1", "y1", "x2", "y2")}})


def _parse_table(lines: list[dict], boxes: list[dict]) -> None:
    header_idx = None
    for i, L in enumerate(lines):
        line_text = " ".join(w["text"] for w in L["words"]).upper()
        if "TEST" in line_text and ("VALUE" in line_text or "TECHNOLOGY" in line_text):
            header_idx = i
            break
    if header_idx is None:
        return

    # column order left-to-right; only classes present in FIELD_CLASSES are kept
    column_labels = ["test_name", "technology", "value", "unit", "reference_range"]

    for L in lines[header_idx + 1:]:
        clusters = _cluster_columns(L["words"])
        if len(clusters) < 3:
            continue
        for label, c in zip(column_labels, clusters):
            if label in FIELD_CLASSES:
                boxes.append({"class": label, "x1": c["x1"], "y1": c["y1"],
                             "x2": c["x2"], "y2": c["y2"]})


def auto_annotate(image_bgr: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Run OCR + layout rules on a full report page image (BGR, as decoded
    by cv2) and return candidate boxes in the same format used by the
    Annotation page's session state: (class_id, x1, y1, x2, y2).

    Unrecognised regions are simply omitted — this is meant to reduce
    manual box-drawing, not to be a complete or final label set.
    """
    h = image_bgr.shape[0]
    lines = _get_lines(image_bgr)

    boxes: list[dict] = []

    header_lines = [L for L in lines if L["y1"] < h * HEADER_ZONE_RATIO]
    for L in header_lines:
        _parse_patient_block(_cluster_columns(L["words"]), boxes)

    _parse_table(lines, boxes)

    return [
        (FIELD_CLASSES.index(b["class"]), b["x1"], b["y1"], b["x2"], b["y2"])
        for b in boxes if b["class"] in FIELD_CLASSES
    ]