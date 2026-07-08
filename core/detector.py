"""
LabVisionAI — Field Detector (YOLO inference)
==============================================
Loads the ACTIVE model from the registry exactly once (singleton) and
returns labelled bounding boxes for the 9 lab-report field classes.
Customers never touch this — the pipeline calls it internally.
"""

import numpy as np

from config.settings import (FIELD_CLASSES, YOLO_CONFIDENCE, YOLO_IMG_SIZE,
                             YOLO_IOU)
from core.registry import get_active_model

_model_cache: dict = {}


class NoDeployedModelError(RuntimeError):
    """Raised when no model has been promoted to 'active' in the registry."""


def _load_model():
    active = get_active_model()
    if active is None:
        raise NoDeployedModelError(
            "No deployed model. An admin must register and promote a model "
            "in the Admin Portal -> Model Registry before customers can process reports."
        )
    version, weights = active
    if _model_cache.get("version") != version:
        from ultralytics import YOLO
        _model_cache["model"] = YOLO(str(weights))
        _model_cache["version"] = version
    return _model_cache["model"], version


def detect_fields(image: np.ndarray) -> tuple[list[dict], str]:
    """
    Run the deployed YOLO model on one page.

    Returns ([{class, confidence, box:[x1,y1,x2,y2]}], model_version).
    """
    model, version = _load_model()
    results = model.predict(image, conf=YOLO_CONFIDENCE, iou=YOLO_IOU,
                            imgsz=YOLO_IMG_SIZE, verbose=False)[0]
    detections = []
    names = results.names or {i: c for i, c in enumerate(FIELD_CLASSES)}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class": names.get(cls_id, f"class_{cls_id}"),
            "confidence": float(box.conf[0]),
            "box": [int(v) for v in box.xyxy[0].tolist()],
        })
    return detections, version
