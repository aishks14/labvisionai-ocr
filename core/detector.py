"""
LabVisionAI — Field Detector (YOLO inference)
==============================================
Loads the ACTIVE model from the registry exactly once (singleton) and
returns labelled bounding boxes for the current FIELD_CLASSES scheme.
Customers never touch this — the pipeline calls it internally.
"""

import numpy as np

from config.settings import (FIELD_CLASSES, YOLO_CONFIDENCE, YOLO_IMG_SIZE,
                             YOLO_IOU)
from core.registry import get_active_model

_model_cache: dict = {}


class NoDeployedModelError(RuntimeError):
    """Raised when no model has been promoted to 'active' in the registry."""


class IncompatibleModelError(RuntimeError):
    """
    Raised when the deployed model's own class names don't overlap
    with the current FIELD_CLASSES scheme — e.g. a model trained
    before FIELD_CLASSES changed (a 9-per-field scheme vs the current
    coarse header_block/results_table scheme). This fails loudly and
    immediately rather than silently returning detections that never
    match anything downstream, which otherwise looks like "0 results,
    no error" and is very hard to diagnose from the UI alone.
    """


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
        model = YOLO(str(weights))

        model_classes = set((model.names or {}).values())
        expected = set(FIELD_CLASSES)
        if model_classes and not (model_classes & expected):
            raise IncompatibleModelError(
                f"Deployed model '{version}' was trained on class scheme "
                f"{sorted(model_classes)}, but this app now expects "
                f"{sorted(expected)}. This happens after FIELD_CLASSES "
                f"changes in config/settings.py — retrain and promote a "
                f"new model using the current Annotation classes before "
                f"customers can process reports."
            )

        _model_cache["model"] = model
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