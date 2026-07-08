"""Customer routes: upload, process, results, export, history."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response

from api.deps import current_user, get_db
from core.detector import NoDeployedModelError
from core.exporters import export
from core.pipeline import process_document, save_upload
from database.models import Document, Extraction

router = APIRouter()


def _own_doc(doc_id: int, user, db) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None or (doc.owner_id != user.id and user.role != "admin"):
        raise HTTPException(404, "Document not found")
    return doc


@router.post("/upload")
async def upload(file: UploadFile, user=Depends(current_user)):
    try:
        doc_id = save_upload(user.id, file.filename, await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"document_id": doc_id, "status": "uploaded"}


@router.post("/{doc_id}/process")
def process(doc_id: int, user=Depends(current_user), db=Depends(get_db)):
    _own_doc(doc_id, user, db)
    try:
        record = process_document(doc_id)
    except NoDeployedModelError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")
    return record


@router.get("")
def history(user=Depends(current_user), db=Depends(get_db)):
    docs = (db.query(Document).filter_by(owner_id=user.id)
            .order_by(Document.created_at.desc()).limit(200).all())
    return [{"id": d.id, "filename": d.filename, "status": d.status,
             "pages": d.pages, "model_version": d.model_version,
             "created_at": d.created_at.isoformat()} for d in docs]


@router.get("/{doc_id}/result")
def result(doc_id: int, user=Depends(current_user), db=Depends(get_db)):
    _own_doc(doc_id, user, db)
    ext = (db.query(Extraction).filter_by(document_id=doc_id)
           .order_by(Extraction.id.desc()).first())
    if ext is None:
        raise HTTPException(404, "No extraction yet — call /process first")
    return {"header": ext.header, "rows": ext.rows,
            "mean_ocr_confidence": ext.mean_ocr_confidence,
            "processing_ms": ext.processing_ms}


@router.get("/{doc_id}/export/{fmt}")
def download(doc_id: int, fmt: str, user=Depends(current_user), db=Depends(get_db)):
    doc = _own_doc(doc_id, user, db)
    ext = (db.query(Extraction).filter_by(document_id=doc_id)
           .order_by(Extraction.id.desc()).first())
    if ext is None:
        raise HTTPException(404, "No extraction yet")
    try:
        data, mime, suffix = export({"header": ext.header, "rows": ext.rows}, fmt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    name = f"{doc.filename.rsplit('.', 1)[0]}_extracted.{suffix}"
    return Response(data, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
