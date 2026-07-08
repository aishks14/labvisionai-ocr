"""
LabVisionAI — REST API (FastAPI)
=================================
Programmatic access for hospital LIS/HIS integrations. Mirrors the
Customer Portal: auth, upload, process, results, export. Admin-only
routes cover the model registry. Run:  uvicorn api.main:app --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import admin, auth, documents
from config.settings import APP_NAME, APP_VERSION
from database.db import init_db

app = FastAPI(title=f"{APP_NAME} API", version=APP_VERSION,
              description="Lab report OCR extraction — inference API for "
                          "customers, registry API for admins.")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health", tags=["system"])
def health():
    from core.registry import get_active_model
    active = get_active_model()
    return {"status": "ok", "version": APP_VERSION,
            "deployed_model": active[0] if active else None}
