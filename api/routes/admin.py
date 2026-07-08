"""Admin-only routes: model registry, users, audit log."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import admin_only, get_db
from core.registry import list_models, promote_model, retire_model
from database.models import AuditLog, ModelVersion, User

router = APIRouter(dependencies=[Depends(admin_only)])


@router.get("/models")
def models(db=Depends(get_db)):
    index = list_models()
    rows = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return {"active": index.get("active"),
            "versions": [{"version": m.version, "status": m.status,
                          "metrics": m.metrics, "dataset": m.dataset_name,
                          "created_at": m.created_at.isoformat()} for m in rows]}


class PromoteIn(BaseModel):
    version: str


@router.post("/models/promote")
def promote(body: PromoteIn, user: User = Depends(admin_only)):
    try:
        promote_model(body.version, actor=user.email)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"active": body.version}


@router.post("/models/retire")
def retire(body: PromoteIn, user: User = Depends(admin_only)):
    try:
        retire_model(body.version, actor=user.email)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"retired": body.version}


@router.get("/logs")
def logs(limit: int = 100, db=Depends(get_db)):
    rows = (db.query(AuditLog).order_by(AuditLog.created_at.desc())
            .limit(min(limit, 500)).all())
    return [{"when": r.created_at.isoformat(), "actor": r.actor,
             "action": r.action, "detail": r.detail} for r in rows]


@router.get("/users")
def users(db=Depends(get_db)):
    return [{"id": u.id, "email": u.email, "role": u.role,
             "organization": u.organization, "active": u.is_active}
            for u in db.query(User).all()]
