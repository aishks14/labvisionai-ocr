"""
LabVisionAI — CUSTOMER PORTAL
==============================
What hospitals and diagnostic labs see. Five pages, nothing else:
Dashboard · Upload Report · History · Export · Settings.

There is no dataset manager, no annotation tool, no training, no
deployment screen here. The deployed model is invisible plumbing.

Run:  streamlit run portals/customer/app.py --server.port 8501
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from config.settings import ALLOWED_EXTENSIONS, APP_NAME, MAX_UPLOAD_MB
from core.detector import NoDeployedModelError
from core.exporters import EXPORTERS, export
from core.pipeline import process_document, save_upload
from database.db import SessionLocal, init_db
from database.models import Document, Extraction
from portals.common import logout_button, require_login

st.set_page_config(page_title=f"{APP_NAME} — Customer Portal",
                   page_icon="🔬", layout="wide")
init_db()

user = require_login(required_role="customer", allow_signup=True)
logout_button()

st.sidebar.title(f"🔬 {APP_NAME}")
page = st.sidebar.radio("Navigation",
                        ["Dashboard", "Upload Report", "History", "Export",
                         "Settings"])


def _db():
    return SessionLocal()


def _latest_extraction(db, doc_id: int) -> Extraction | None:
    return (db.query(Extraction).filter_by(document_id=doc_id)
            .order_by(Extraction.id.desc()).first())


def _render_result(ext: Extraction):
    """Header card + editable results table shared by several pages."""
    if ext.header:
        cols = st.columns(min(len(ext.header), 5))
        for col, (key, value) in zip(cols, ext.header.items()):
            col.metric(key.replace("_", " ").title(), value or "—")
    st.caption(f"Mean OCR confidence: {ext.mean_ocr_confidence}% · "
               f"Processed in {ext.processing_ms} ms")

    df = pd.DataFrame(ext.rows) if ext.rows else pd.DataFrame(
        columns=["test_name", "value", "unit", "reference_range", "flag"])
    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic",
                            key=f"editor_{ext.id}")
    if st.button("Save corrections", key=f"save_{ext.id}"):
        db = _db()
        try:
            row = db.get(Extraction, ext.id)
            row.rows = edited.fillna("").to_dict("records")
            db.commit()
            st.success("Corrections saved.")
        finally:
            db.close()


# ------------------------------------------------------------ Dashboard ----
if page == "Dashboard":
    st.title("Dashboard")
    db = _db()
    try:
        docs = db.query(Document).filter_by(owner_id=user["id"]).all()
        done = [d for d in docs if d.status == "done"]
        failed = [d for d in docs if d.status == "failed"]
        confs = [_latest_extraction(db, d.id).mean_ocr_confidence
                 for d in done if _latest_extraction(db, d.id)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reports uploaded", len(docs))
        c2.metric("Processed", len(done))
        c3.metric("Failed", len(failed))
        c4.metric("Avg OCR confidence",
                  f"{sum(confs)/len(confs):.1f}%" if confs else "—")

        st.subheader("Recent activity")
        recent = sorted(docs, key=lambda d: d.created_at, reverse=True)[:10]
        if recent:
            st.dataframe(pd.DataFrame(
                [{"File": d.filename, "Status": d.status,
                  "Uploaded": d.created_at.strftime("%d %b %Y %H:%M")}
                 for d in recent]), use_container_width=True, hide_index=True)
        else:
            st.info("No reports yet. Head to **Upload Report** to get started.")
    finally:
        db.close()

# -------------------------------------------------------- Upload Report ----
elif page == "Upload Report":
    st.title("Upload Report")
    st.caption(f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))} · "
               f"Max {MAX_UPLOAD_MB} MB per file")

    files = st.file_uploader("Drop lab reports here", accept_multiple_files=True,
                             type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS])
    if files and st.button(f"Process {len(files)} report(s)", type="primary"):
        progress = st.progress(0.0)
        for i, f in enumerate(files, start=1):
            with st.status(f"Processing {f.name}…", expanded=False) as status:
                try:
                    doc_id = save_upload(user["id"], f.name, f.getvalue())
                    process_document(doc_id)
                    status.update(label=f"✅ {f.name}", state="complete")
                except NoDeployedModelError as e:
                    status.update(label=f"⛔ {f.name}: service unavailable",
                                  state="error")
                    st.error(str(e))
                    break
                except Exception as e:
                    status.update(label=f"❌ {f.name}: {e}", state="error")
            progress.progress(i / len(files))
        st.success("Done. Review results below or in **History**.")

    st.divider()
    st.subheader("Latest result")
    db = _db()
    try:
        latest = (db.query(Document)
                  .filter_by(owner_id=user["id"], status="done")
                  .order_by(Document.id.desc()).first())
        if latest:
            st.markdown(f"**{latest.filename}**")
            ext = _latest_extraction(db, latest.id)
            if ext:
                _render_result(ext)
        else:
            st.info("Processed results will appear here.")
    finally:
        db.close()

# --------------------------------------------------------------- History ---
elif page == "History":
    st.title("History")
    db = _db()
    try:
        docs = (db.query(Document).filter_by(owner_id=user["id"])
                .order_by(Document.created_at.desc()).all())
        if not docs:
            st.info("No reports uploaded yet.")
        for d in docs:
            icon = {"done": "✅", "failed": "❌",
                    "processing": "⏳"}.get(d.status, "📄")
            with st.expander(f"{icon} {d.filename} — "
                             f"{d.created_at.strftime('%d %b %Y %H:%M')}"):
                if d.status == "failed":
                    st.error(d.error or "Processing failed.")
                    if st.button("Retry", key=f"retry_{d.id}"):
                        try:
                            process_document(d.id)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                elif d.status == "done":
                    ext = _latest_extraction(db, d.id)
                    if ext:
                        _render_result(ext)
                else:
                    st.info(f"Status: {d.status}")
    finally:
        db.close()

# ---------------------------------------------------------------- Export ---
elif page == "Export":
    st.title("Export")
    db = _db()
    try:
        docs = (db.query(Document)
                .filter_by(owner_id=user["id"], status="done")
                .order_by(Document.created_at.desc()).all())
        if not docs:
            st.info("Process at least one report to enable exports.")
        else:
            choice = st.selectbox(
                "Report", docs,
                format_func=lambda d: f"{d.filename} "
                                      f"({d.created_at.strftime('%d %b %Y')})")
            ext = _latest_extraction(db, choice.id)
            if ext:
                record = {"header": ext.header, "rows": ext.rows}
                base = choice.filename.rsplit(".", 1)[0]
                labels = {"xlsx": "Excel (.xlsx)", "csv": "CSV", "json": "JSON",
                          "pdf": "PDF report", "fhir": "FHIR R4 (JSON)"}
                cols = st.columns(len(EXPORTERS))
                for col, fmt in zip(cols, EXPORTERS):
                    data, mime, suffix = export(record, fmt)
                    col.download_button(labels[fmt], data,
                                        file_name=f"{base}_extracted.{suffix}",
                                        mime=mime, use_container_width=True)
                st.divider()
                _render_result(ext)
    finally:
        db.close()

# -------------------------------------------------------------- Settings ---
elif page == "Settings":
    st.title("Settings")
    st.text_input("Email", value=user["email"], disabled=True)
    st.text_input("Name", value=user.get("name", ""), disabled=True)
    st.divider()
    st.subheader("Change password")
    with st.form("pw"):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password (min 8 chars)", type="password")
        if st.form_submit_button("Update password"):
            from core.security import hash_password, verify_password
            from database.models import User
            db = _db()
            try:
                row = db.get(User, user["id"])
                if not verify_password(current, row.password_hash):
                    st.error("Current password is incorrect.")
                elif len(new) < 8:
                    st.error("New password too short.")
                else:
                    row.password_hash = hash_password(new)
                    db.commit()
                    st.success("Password updated.")
            finally:
                db.close()
