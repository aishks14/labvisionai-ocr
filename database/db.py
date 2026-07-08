"""
LabVisionAI — Database session management
==========================================
Engine + session factory + init helper. SQLite by default; swap
LVA_DATABASE_URL to postgres:// for production without code changes.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import DATABASE_URL
from database.models import AuditLog, Base

_engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db():
    """Create all tables (idempotent)."""
    Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    """Transactional scope: commit on success, rollback on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def log_event(actor: str, action: str, detail: str = ""):
    """Append one row to the audit trail."""
    with session_scope() as s:
        s.add(AuditLog(actor=actor, action=action, detail=detail))
