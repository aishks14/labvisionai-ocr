"""
LabVisionAI — YOLO training (Admin/ML team only)
=================================================
Fine-tunes YOLOv8 on a prepared dataset and registers the resulting
best.pt as a *candidate* in the model registry. It is NOT deployed
automatically — promotion is a deliberate human decision.

Usage:
    python -m training.train_yolo --dataset labreports_v3 --version v3.1.0 --epochs 100
"""

import argparse
from pathlib import Path

from config.settings import DATASET_DIR, MODEL_DIR, YOLO_IMG_SIZE
from core.registry import register_model


def train(dataset_name: str, version: str, epochs: int = 100,
          base_model: str = "yolov8s.pt", batch: int = 8):
    from ultralytics import YOLO

    data_yaml = DATASET_DIR / dataset_name / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Dataset not prepared: {data_yaml}. "
                         "Run training.prepare_dataset first.")

    model = YOLO(base_model)
    results = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=YOLO_IMG_SIZE, batch=batch,
        project=str(MODEL_DIR / "runs"), name=version,
        patience=25, workers=0,  # workers=0 -> Windows-safe
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    metrics = {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
    }
    register_model(version, str(best), metrics=metrics,
                   dataset_name=dataset_name, notes=f"epochs={epochs} base={base_model}",
                   actor="ml-team")
    print(f"Registered candidate {version} with metrics {metrics}")
    print("Promote it via Admin Portal -> Model Registry (or core.registry.promote_model).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--base-model", default="yolov8s.pt")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    train(args.dataset, args.version, args.epochs, args.base_model, args.batch)
