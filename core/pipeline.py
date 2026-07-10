"""
LabVisionAI — Inference Pipeline (the product's engine room)
=============================================================
The single entry point the Customer Portal and API call:

    upload -> load pages -> enhance -> YOLO detect 2 coarse regions
    (deployed model) -> crop header_block + results_table -> parse
    each region deterministically (core/table_parser.py) -> persist

No training happens here. Ever. Inference only.

v2 note: this used to crop and OCR one small region per individual
field (9 classes) and hand every detection to core/parser.py's
build_record for ML-driven field assembly. It now detects only 2
coarse regions and parses their contents deterministically — see
core/table_parser.py for why. core/parser.py's utility functions
(clean_text, normalize_value, compute_flag) are still used internally
by table_parser.py.
"""

import time
from pathlib import Path

import numpy as np

from core.detector import detect_fields
from core.preprocessing import enhance_page, load_pages
from core.table_parser import parse_header_region, parse_table_region
from database.db import log_event, session_scope
from database.models import Document, Extraction


def _crop(image: np.ndarray, box: list[int], pad: int = 6) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    return image[max(0, y1 - pad):min(h, y2 + pad),
                 max(0, x1 - pad):min(w, x2 + pad)]


def _best_box(detections: list[dict], field_class: str) -> list[int] | None:
    """Highest-confidence detection for a given coarse class, if any."""
    candidates = [d for d in detections if d["class"] == field_class]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d["confidence"])["box"]


def process_document(document_id: int) -> dict:
    """
    Run the full inference pipeline for a stored Document row.
    Updates status through processing -> done/failed and returns the record.
    """
    started = time.time()

    with session_scope() as s:
        doc = s.get(Document, document_id)
        if doc is None:
            raise ValueError(f"Document {document_id} not found")
        doc.status = "processing"
        path, owner = doc.stored_path, doc.owner.email if doc.owner else "?"

    try:
        pages = load_pages(path)
        header: dict = {}
        rows: list[dict] = []
        ocr_confidences: list[float] = []
        model_version = ""

        for page in pages:
            page = enhance_page(page)
            detections, model_version = detect_fields(page)
            h, w = page.shape[:2]

            header_box = _best_box(detections, "header_block")
            if header_box and not header:
                header, conf = parse_header_region(_crop(page, header_box))
                if header:
                    ocr_confidences.append(conf)

            table_box = _best_box(detections, "results_table")
            if table_box:
                table_rows, conf = parse_table_region(_crop(page, table_box))

                if not table_rows:
                    # The detected box may be too tight or slightly
                    # mispositioned — retry against a generously
                    # widened crop before giving up on this page.
                    x1, y1, x2, y2 = table_box
                    pad_x, pad_y = int(0.05 * w), int(0.08 * h)
                    wide_box = [max(0, x1 - pad_x), max(0, y1 - pad_y),
                               min(w, x2 + pad_x), min(h, y2 + pad_y)]
                    table_rows, conf = parse_table_region(_crop(page, wide_box))

                if table_rows:
                    ocr_confidences.append(conf)
                rows.extend(table_rows)

        mean_conf = round(float(np.mean(ocr_confidences)), 1) if ocr_confidences else 0.0
        elapsed_ms = int((time.time() - started) * 1000)
        record = {"header": header, "rows": rows}

        with session_scope() as s:
            doc = s.get(Document, document_id)
            doc.status, doc.pages, doc.model_version = "done", len(pages), model_version
            s.add(Extraction(document_id=document_id, header=record["header"],
                             rows=record["rows"], raw_detections=[],
                             mean_ocr_confidence=mean_conf, processing_ms=elapsed_ms))

        log_event(owner, "document_processed",
                  f"doc={document_id} rows={len(record['rows'])} "
                  f"conf={mean_conf} model={model_version}")
        return {**record, "mean_ocr_confidence": mean_conf,
                "processing_ms": elapsed_ms, "model_version": model_version}

    except Exception as exc:
        with session_scope() as s:
            doc = s.get(Document, document_id)
            doc.status, doc.error = "failed", str(exc)[:1000]
        log_event(owner, "document_failed", f"doc={document_id}: {exc}")
        raise


def save_upload(owner_id: int, filename: str, data: bytes) -> int:
    """Persist an uploaded file to storage and create its Document row."""
    from config.settings import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, UPLOAD_DIR

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File exceeds {MAX_UPLOAD_MB} MB limit")

    with session_scope() as s:
        doc = Document(owner_id=owner_id, filename=filename,
                       file_type=suffix.lstrip("."), stored_path="")
        s.add(doc)
        s.flush()
        stored = UPLOAD_DIR / f"{doc.id}{suffix}"
        stored.write_bytes(data)
        doc.stored_path = str(stored)
        return doc.id