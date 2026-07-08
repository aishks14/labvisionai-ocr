"""
LabVisionAI — Image preprocessing
==================================
Everything that happens to pixels before OCR: PDF rasterization,
deskew, CLAHE contrast, and the crop-level pipeline (upscale ->
grayscale -> blur -> Otsu threshold -> invert) that Tesseract prefers.
"""

from pathlib import Path

import cv2
import numpy as np

from config.settings import OCR_UPSCALE, PDF_DPI


def load_pages(file_path: str | Path) -> list[np.ndarray]:
    """Load an image or PDF into a list of BGR page arrays."""
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        pil_pages = convert_from_path(str(file_path), dpi=PDF_DPI)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pil_pages]
    img = cv2.imread(str(file_path))
    if img is None:
        raise ValueError(f"Unreadable image: {file_path}")
    return [img]


def deskew(image: np.ndarray) -> np.ndarray:
    """Estimate global text skew via minAreaRect and rotate to correct it."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:  # ignore noise / wild estimates
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_page(image: np.ndarray) -> np.ndarray:
    """Page-level cleanup before detection: deskew + CLAHE contrast."""
    image = deskew(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def prepare_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Crop-level OCR pipeline (from the reference workflow):
    upscale 3x -> grayscale -> Gaussian blur -> Otsu threshold
    (white text / black bg) -> bitwise_not (black text / white bg).
    """
    if crop.size == 0:
        return crop
    crop = cv2.resize(crop, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE,
                      interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # despeckle
    return cv2.bitwise_not(thresh)
