"""
LabVisionAI — Model evaluation (Admin/ML team only)
====================================================
Runs val-set evaluation on any registered version and prints a
side-by-side comparison table so admins can decide which candidate
deserves promotion.

Usage:
    python -m training.evaluate --dataset labreports_v3            # compare all
    python -m training.evaluate --dataset labreports_v3 --version v3.1.0
"""

import argparse

from config.settings import DATASET_DIR
from core.registry import list_models


def evaluate(dataset_name: str, version: str | None = None):
    from ultralytics import YOLO

    data_yaml = DATASET_DIR / dataset_name / "data.yaml"
    index = list_models()
    targets = {version: index["versions"][version]} if version else index["versions"]

    print(f"{'Version':<12}{'mAP50':>8}{'mAP50-95':>10}{'Prec':>8}{'Rec':>8}  Status")
    for ver, info in targets.items():
        model = YOLO(info["weights"])
        m = model.val(data=str(data_yaml), verbose=False)
        active = " <- ACTIVE" if index.get("active") == ver else ""
        print(f"{ver:<12}{m.box.map50:>8.3f}{m.box.map:>10.3f}"
              f"{m.box.mp:>8.3f}{m.box.mr:>8.3f}{active}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--version", default=None)
    args = ap.parse_args()
    evaluate(args.dataset, args.version)
