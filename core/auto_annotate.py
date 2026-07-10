"""
LabVisionAI — Auto-annotation assistant
=========================================
Runs Tesseract on a full report page and finds the bounding envelope
of the header info block and the results table, so the Annotation
page can pre-fill 2 candidate region boxes instead of one per field.

v2 note: this used to emit ~50 tiny per-field boxes for a 9-class
scheme. Detecting individual field boxes reliably needs a lot more
training data than a small dataset provides — see core/table_parser.py
for the full reasoning. Annotating 2 coarse regions per image is both
far faster to review by hand AND a much easier detection task for a
small model to actually learn well.
"""

from __future__ import annotations

import numpy as np

from config.settings import FIELD_CLASSES
from core.table_parser import cluster_columns, get_lines

END_OF_TABLE_MARKERS = ("comment", "please correlate", "method", "end of report",
                        "interpretation", "note:")


def _line_text(L: dict) -> str:
    return " ".join(w["text"] for w in L["words"])


def _bbox(lines: list[dict]) -> tuple[int, int, int, int] | None:
    if not lines:
        return None
    x1 = min(w["x1"] for L in lines for w in L["words"])
    y1 = min(L["y1"] for L in lines)
    x2 = max(w["x2"] for L in lines for w in L["words"])
    y2 = max(L["y2"] for L in lines)
    return x1, y1, x2, y2


def auto_annotate(image_bgr: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """
    Run OCR + layout rules on a full report page image (BGR, as decoded
    by cv2) and return candidate region boxes in the same format used
    by the Annotation page's session state: (class_id, x1, y1, x2, y2).

    Finds two regions: everything from the patient "NAME" line down to
    (not including) the results-table header row = header_block; the
    table header row through the last recognizable data row =
    results_table. Falls back gracefully if either marker isn't found.
    """
    lines = get_lines(image_bgr)
    if not lines:
        return []

    name_idx = next((i for i, L in enumerate(lines)
                     if "NAME" in _line_text(L).upper()
                     and "TEST NAME" not in _line_text(L).upper()), None)

    table_start_idx = next((i for i, L in enumerate(lines)
                            if "TEST" in _line_text(L).upper()
                            and ("VALUE" in _line_text(L).upper()
                                 or "TECHNOLOGY" in _line_text(L).upper())), None)

    table_end_idx = None
    if table_start_idx is not None:
        for i in range(table_start_idx + 1, len(lines)):
            text_lower = _line_text(lines[i]).lower()
            if any(marker in text_lower for marker in END_OF_TABLE_MARKERS):
                table_end_idx = i
                break

    boxes: list[dict] = []

    header_start = name_idx if name_idx is not None else 0
    header_end = table_start_idx if table_start_idx is not None else min(len(lines), header_start + 5)
    header_bbox = _bbox(lines[header_start:header_end])
    if header_bbox:
        boxes.append({"class": "header_block", "box": header_bbox})

    if table_start_idx is not None:
        table_lines = lines[table_start_idx:table_end_idx]
        table_bbox = _bbox(table_lines)
        if table_bbox:
            boxes.append({"class": "results_table", "box": table_bbox})

    return [
        (FIELD_CLASSES.index(b["class"]), *b["box"])
        for b in boxes if b["class"] in FIELD_CLASSES
    ]