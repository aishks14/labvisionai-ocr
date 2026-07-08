"""
LabVisionAI — Model Registry
=============================
Versioned store of trained YOLO weights with a single ACTIVE pointer.
The customer pipeline only ever loads the active (deployed, frozen)
model. Admins register candidates, promote one to active, and retire
old versions. Backed by models/registry/registry.json + the DB.
"""

import json
import shutil
from pathlib import Path

from config.settings import REGISTRY_DIR, REGISTRY_INDEX
from database.db import log_event, session_scope
from database.models import ModelVersion


def _load_index() -> dict:
    if REGISTRY_INDEX.exists():
        return json.loads(REGISTRY_INDEX.read_text())
    return {"active": None, "versions": {}}


def _save_index(index: dict):
    REGISTRY_INDEX.write_text(json.dumps(index, indent=2))


def register_model(version: str, weights_file: str, metrics: dict | None = None,
                   dataset_name: str = "", notes: str = "", actor: str = "system") -> Path:
    """Copy a trained best.pt into the registry as a new candidate version."""
    src = Path(weights_file)
    if not src.exists():
        raise FileNotFoundError(f"Weights not found: {src}")

    dest_dir = REGISTRY_DIR / version
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "best.pt"
    shutil.copy2(src, dest)

    index = _load_index()
    index["versions"][version] = {"weights": str(dest), "metrics": metrics or {}}
    _save_index(index)

    with session_scope() as s:
        s.add(ModelVersion(version=version, weights_path=str(dest),
                           metrics=metrics or {}, dataset_name=dataset_name,
                           notes=notes, status="candidate", created_by=actor))
    log_event(actor, "model_registered", f"{version} <- {src.name}")
    return dest


def promote_model(version: str, actor: str = "system"):
    """Make `version` the deployed model; freeze the previous active one."""
    index = _load_index()
    if version not in index["versions"]:
        raise KeyError(f"Unknown version: {version}")

    previous = index.get("active")
    index["active"] = version
    _save_index(index)

    with session_scope() as s:
        if previous:
            prev_row = s.query(ModelVersion).filter_by(version=previous).first()
            if prev_row:
                prev_row.status = "frozen"
        row = s.query(ModelVersion).filter_by(version=version).first()
        if row:
            row.status = "active"
    log_event(actor, "model_promoted", f"{version} (previous: {previous})")


def retire_model(version: str, actor: str = "system"):
    index = _load_index()
    if index.get("active") == version:
        raise ValueError("Cannot retire the active model. Promote another first.")
    with session_scope() as s:
        row = s.query(ModelVersion).filter_by(version=version).first()
        if row:
            row.status = "retired"
    log_event(actor, "model_retired", version)


def get_active_model() -> tuple[str, Path] | None:
    """Return (version, weights_path) of the deployed model, or None."""
    index = _load_index()
    active = index.get("active")
    if not active:
        return None
    weights = Path(index["versions"][active]["weights"])
    return (active, weights) if weights.exists() else None


def list_models() -> dict:
    return _load_index()
