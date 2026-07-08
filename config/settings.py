"""
LabVisionAI — Central Configuration
====================================
Single source of truth for every path, threshold, and secret in the system.
All values can be overridden via environment variables (.env file).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- Paths ----
BASE_DIR = Path(__file__).resolve().parent.parent

STORAGE_DIR = Path(os.getenv("LVA_STORAGE_DIR", BASE_DIR / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
RESULT_DIR = STORAGE_DIR / "results"
EXPORT_DIR = STORAGE_DIR / "exports"
DATASET_DIR = STORAGE_DIR / "datasets"
LOG_DIR = STORAGE_DIR / "logs"

MODEL_DIR = Path(os.getenv("LVA_MODEL_DIR", BASE_DIR / "models"))
REGISTRY_DIR = MODEL_DIR / "registry"
REGISTRY_INDEX = REGISTRY_DIR / "registry.json"

for _d in (UPLOAD_DIR, RESULT_DIR, EXPORT_DIR, DATASET_DIR, LOG_DIR, REGISTRY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- Database ----
DATABASE_URL = os.getenv("LVA_DATABASE_URL", f"sqlite:///{BASE_DIR / 'labvisionai.db'}")

# ----------------------------------------------------------------- Auth ----
SECRET_KEY = os.getenv("LVA_SECRET_KEY", "change-me-in-production-9f8a7b6c")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("LVA_TOKEN_EXPIRE_MIN", "480"))

# ------------------------------------------------------------ Detection ----
YOLO_CONFIDENCE = float(os.getenv("LVA_YOLO_CONF", "0.35"))
YOLO_IOU = float(os.getenv("LVA_YOLO_IOU", "0.45"))
YOLO_IMG_SIZE = int(os.getenv("LVA_YOLO_IMGSZ", "1024"))

# Field classes the detector is trained on (index = YOLO class id).
FIELD_CLASSES = [
    "patient_name",
    "age",
    "gender",
    "doctor_name",
    "report_date",
    "test_name",
    "value",
    "unit",
    "reference_range",
]

# --------------------------------------------------------------- OCR -------
TESSERACT_CMD = os.getenv("LVA_TESSERACT_CMD", "")  # blank = system default
OCR_LANG = os.getenv("LVA_OCR_LANG", "eng")
OCR_UPSCALE = 3          # crop upscale factor before OCR
OCR_MIN_CONFIDENCE = 30  # discard OCR words below this confidence

# Per-field Tesseract page-segmentation modes (single line vs block).
FIELD_PSM = {
    "patient_name": 7, "age": 7, "gender": 7, "doctor_name": 7,
    "report_date": 7, "test_name": 6, "value": 6, "unit": 6,
    "reference_range": 6,
}

# ------------------------------------------------------------- Product -----
APP_NAME = "LabVisionAI"
APP_VERSION = "3.0.0"
MAX_UPLOAD_MB = int(os.getenv("LVA_MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
PDF_DPI = 300
