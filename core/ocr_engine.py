"""
LabVisionAI — OCR Engine
=========================
Tesseract wrapper that reads each cropped, preprocessed field region
with a per-field PSM and returns (text, confidence). Designed so the
backend can later swap in PaddleOCR behind the same read_region() API.
"""

import numpy as np
import pytesseract

from config.settings import (FIELD_PSM, OCR_LANG, OCR_MIN_CONFIDENCE,
                             TESSERACT_CMD)
from core.preprocessing import prepare_crop_for_ocr

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def read_region(crop: np.ndarray, field_class: str) -> tuple[str, float]:
    """OCR one detected field crop. Returns (clean_text, mean_confidence)."""
    if crop is None or crop.size == 0:
        return "", 0.0

    processed = prepare_crop_for_ocr(crop)
    psm = FIELD_PSM.get(field_class, 6)
    config = f"--oem 3 --psm {psm}"

    data = pytesseract.image_to_data(processed, lang=OCR_LANG, config=config,
                                     output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = -1.0
        if text and conf >= OCR_MIN_CONFIDENCE:
            words.append(text)
            confs.append(conf)

    joined = " ".join(words)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return joined, round(mean_conf, 1)
