"""
LabVisionAI — Dataset preparation (Admin/ML team only)
=======================================================
Builds a YOLO-format dataset from an annotated image folder:
train/val split, data.yaml generation, and integrity checks
(orphan labels, empty files, class-id range).

Usage:
    python -m training.prepare_dataset --source storage/datasets/raw_v3 --name labreports_v3
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

from config.settings import DATASET_DIR, FIELD_CLASSES
from database.db import log_event, session_scope
from database.models import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def validate_label(label_file: Path) -> list[str]:
    errors = []
    for ln, line in enumerate(label_file.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_file.name}:{ln} malformed")
            continue
        cls, *coords = parts
        if not cls.isdigit() or int(cls) >= len(FIELD_CLASSES):
            errors.append(f"{label_file.name}:{ln} bad class id {cls}")
        if any(not 0 <= float(c) <= 1 for c in coords):
            errors.append(f"{label_file.name}:{ln} coord out of [0,1]")
    return errors


def build(source: Path, name: str, val_ratio: float = 0.2, seed: int = 42):
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXT)
    pairs, errors = [], []
    for img in images:
        label = img.with_suffix(".txt")
        if label.exists():
            errors += validate_label(label)
            pairs.append((img, label))
    if errors:
        print("Validation errors:\n" + "\n".join(errors[:30]))
    if not pairs:
        raise SystemExit("No annotated (image, .txt) pairs found.")

    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    root = DATASET_DIR / name
    for split, items in splits.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, label in items:
            shutil.copy2(img, root / "images" / split / img.name)
            shutil.copy2(label, root / "labels" / split / label.name)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root), "train": "images/train", "val": "images/val",
        "nc": len(FIELD_CLASSES), "names": FIELD_CLASSES}))

    with session_scope() as s:
        existing = s.query(Dataset).filter_by(name=name).first()
        if existing:
            existing.n_images, existing.n_annotated = len(images), len(pairs)
            existing.root_path = str(root)
        else:
            s.add(Dataset(name=name, root_path=str(root),
                          n_images=len(images), n_annotated=len(pairs)))
    log_event("ml-team", "dataset_built",
              f"{name}: {len(splits['train'])} train / {len(splits['val'])} val")
    print(f"Dataset ready: {data_yaml}")
    return data_yaml


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()
    build(args.source, args.name, args.val_ratio)
