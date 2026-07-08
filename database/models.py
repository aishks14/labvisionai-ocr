"""
LabVisionAI — SQLAlchemy ORM models
====================================
Users (role-based), Documents, Extractions, ModelVersions, Datasets,
AuditLogs. One schema shared by the API and both portals.
"""

import datetime as dt

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow():
    return dt.datetime.utcnow()


class User(Base):
    """Portal/API account. role: 'admin' (AI team) or 'customer'."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), default="")
    organization = Column(String(255), default="")
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="customer")  # admin | customer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)

    documents = relationship("Document", back_populates="owner")


class Document(Base):
    """One uploaded lab report (PDF or image) belonging to a customer."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    owner_id = Column(Integer, ForeignKey("users.id"), index=True)
    filename = Column(String(512))
    stored_path = Column(String(1024))
    file_type = Column(String(10))
    pages = Column(Integer, default=1)
    status = Column(String(20), default="uploaded")  # uploaded|processing|done|failed
    error = Column(Text, default="")
    model_version = Column(String(50), default="")
    created_at = Column(DateTime, default=utcnow)

    owner = relationship("User", back_populates="documents")
    extractions = relationship("Extraction", back_populates="document",
                               cascade="all, delete-orphan")


class Extraction(Base):
    """Structured OCR result for one document (header + test rows as JSON)."""
    __tablename__ = "extractions"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True)
    header = Column(JSON, default=dict)      # patient_name, age, gender, ...
    rows = Column(JSON, default=list)        # [{test_name, value, unit, reference_range, flag}]
    raw_detections = Column(JSON, default=list)
    mean_ocr_confidence = Column(Float, default=0.0)
    processing_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)

    document = relationship("Document", back_populates="extractions")


class ModelVersion(Base):
    """Registry entry for one trained YOLO weight file."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True)
    version = Column(String(50), unique=True)      # e.g. v3.1.0
    weights_path = Column(String(1024))
    metrics = Column(JSON, default=dict)           # mAP50, precision, recall
    dataset_name = Column(String(255), default="")
    notes = Column(Text, default="")
    status = Column(String(20), default="candidate")  # candidate|active|frozen|retired
    created_by = Column(String(255), default="")
    created_at = Column(DateTime, default=utcnow)


class Dataset(Base):
    """Annotation dataset tracked by the admin portal."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True)
    root_path = Column(String(1024))
    n_images = Column(Integer, default=0)
    n_annotated = Column(Integer, default=0)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)


class AuditLog(Base):
    """System event trail shown in the admin portal (Logs page)."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    actor = Column(String(255), default="system")
    action = Column(String(100))
    detail = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow, index=True)
