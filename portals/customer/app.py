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

import cv2
import pandas as pd
import streamlit as st

from config.settings import ALLOWED_EXTENSIONS, APP_NAME, MAX_UPLOAD_MB
from core.detector import NoDeployedModelError
from core.exporters import EXPORTERS, export
from core.pipeline import process_document, save_upload
from core.preprocessing import load_pages
from database.db import SessionLocal, init_db
from database.models import Document, Extraction
from portals.common import render_sidebar, require_login, styled_df

st.set_page_config(page_title=f"{APP_NAME} — Customer Portal",
                   page_icon="🔬", layout="wide")
init_db()

user = require_login(required_role="customer", allow_signup=True)

NAV_ITEMS = [("dashboard", "Dashboard"), ("upload_file", "Upload Report"),
            ("history", "History"), ("download", "Export"),
            ("settings", "Settings")]
page = render_sidebar(APP_NAME, "Pathology Report Intelligence", NAV_ITEMS, user)

st.markdown(
    """<style>
    .lv-trust-strip {
        display: flex; gap: 18px; flex-wrap: wrap;
        font-family: 'JetBrains Mono', monospace; font-size: 11px;
        color: #6B6472; letter-spacing: 0.02em; margin: -6px 0 18px 0;
    }
    .lv-thumb { border: 1px solid #E4DEE6; overflow: hidden; background: #FAF7FB; }
    .lv-thumb-caption {
        font-family: 'JetBrains Mono', monospace; font-size: 11px;
        color: #6B6472; padding: 4px 2px 10px 2px; word-break: break-all;
    }
    </style>""",
    unsafe_allow_html=True,
)


def _db():
    return SessionLocal()


def _latest_extraction(db, doc_id: int) -> Extraction | None:
    return (db.query(Extraction).filter_by(document_id=doc_id)
            .order_by(Extraction.id.desc()).first())


@st.cache_data(show_spinner=False)
def _preview_array(path: str, max_dim: int = 700):
    """
    Load the first page of a stored document as an RGB array for
    st.image, downscaled to max_dim on its longest side and cached
    by path — avoids re-decoding full-resolution scans on every rerun.
    """
    try:
        pages = load_pages(path)
        arr = cv2.cvtColor(pages[0], cv2.COLOR_BGR2RGB)
        h, w = arr.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        return arr
    except Exception:
        return None


def _delete_document(doc_id: int) -> None:
    db = _db()
    try:
        doc = db.get(Document, doc_id)
        if doc:
            try:
                Path(doc.stored_path).unlink(missing_ok=True)
            except Exception:
                pass
            db.delete(doc)
            db.commit()
    finally:
        db.close()


def _delete_all_documents(owner_id: int) -> None:
    db = _db()
    try:
        docs = db.query(Document).filter_by(owner_id=owner_id).all()
        for doc in docs:
            try:
                Path(doc.stored_path).unlink(missing_ok=True)
            except Exception:
                pass
            db.delete(doc)
        db.commit()
    finally:
        db.close()


@st.dialog("Delete this report?")
def _confirm_delete_one(doc_id: int, filename: str):
    st.write(f"This permanently deletes **{filename}**, its extracted "
             "data, and the stored file. This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width='stretch', key=f"cancel_{doc_id}"):
        st.rerun()
    if c2.button("Delete", type="primary", width='stretch',
                key=f"confirm_del_{doc_id}"):
        _delete_document(doc_id)
        st.session_state.pop("_history_page", None)
        st.rerun()


@st.dialog("Delete all reports?")
def _confirm_delete_all(owner_id: int):
    st.write("This permanently deletes **every** uploaded report, its "
             "extracted data, and the stored files. This can't be undone.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width='stretch', key="cancel_all"):
        st.rerun()
    if c2.button("Delete all", type="primary", width='stretch',
                key="confirm_del_all"):
        _delete_all_documents(owner_id)
        st.session_state.pop("_history_page", None)
        st.rerun()


def _render_result(ext: Extraction, source_path: str | None = None):
    """Small patient/ref-by line + metrics + source image + editable table."""
    header = ext.header or {}
    patient_name = header.get("patient_name", "")
    ref_by = header.get("doctor_name", "")
    other_fields = {k: v for k, v in header.items()
                    if k not in ("patient_name", "doctor_name") and v}

    if patient_name or ref_by:
        left, right = st.columns([2, 1])
        with right:
            parts = []
            if patient_name:
                parts.append(f"🧑 {patient_name}")
            if ref_by:
                parts.append(f"Ref by: {ref_by}")
            st.markdown(
                f"<div style='text-align:right; font-size:11px; "
                f"color:#6B6472; line-height:1.6; margin-top:4px;'>"
                f"{'<br>'.join(parts)}</div>",
                unsafe_allow_html=True,
            )

    if other_fields:
        cols = st.columns(min(len(other_fields), 5))
        for col, (key, value) in zip(cols, other_fields.items()):
            col.metric(key.replace("_", " ").title(), value or "—")
    st.caption(f"Mean OCR confidence: {ext.mean_ocr_confidence}% · "
               f"Processed in {ext.processing_ms} ms")

    df = pd.DataFrame(ext.rows) if ext.rows else pd.DataFrame(
        columns=["test_name", "value", "unit", "reference_range", "flag"])

    if source_path:
        img_col, table_col = st.columns([1, 2])
        with img_col:
            arr = _preview_array(source_path)
            if arr is not None:
                st.image(arr, width='stretch', caption="Source document")
    else:
        table_col = st.container()

    with table_col:
        edited = st.data_editor(
            df, width='stretch', num_rows="dynamic",
            key=f"editor_{ext.id}",
            column_config={
                "test_name": st.column_config.TextColumn("🧪 Test Name", width="large"),
                "value": st.column_config.TextColumn("📊 Value"),
                "unit": st.column_config.TextColumn("📐 Unit"),
                "reference_range": st.column_config.TextColumn("📏 Reference Range"),
                "flag": st.column_config.SelectboxColumn(
                    "🚩 Flag", options=["", "LOW", "NORMAL", "HIGH"]),
            },
        )
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
    st.markdown(
        """<div class="lv-trust-strip">
        <span>🔬 YOLOv8 field detection</span>
        <span>📄 Multi-format export</span>
        <span>🧬 Built for pathology &amp; diagnostic labs</span>
        </div>""", unsafe_allow_html=True)
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
            recent_df = pd.DataFrame(
                [{"File": d.filename, "Status": d.status,
                  "Uploaded": d.created_at.strftime("%d %b %Y %H:%M")}
                 for d in recent])
            st.dataframe(styled_df(recent_df), width='stretch', hide_index=True)
        else:
            st.info("No reports yet. Head to **Upload Report** to get started.")
    finally:
        db.close()

# -------------------------------------------------------- Upload Report ----
elif page == "Upload Report":
    st.title("Upload Report")
    st.caption(f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))} · "
               f"Max {MAX_UPLOAD_MB} MB per file")

    st.markdown('<div class="lv-card">', unsafe_allow_html=True)
    files = st.file_uploader("Drop lab reports here", accept_multiple_files=True,
                             type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS])

    if files:
        st.markdown("**Preview**")
        cols = st.columns(min(len(files), 4))
        for i, f in enumerate(files):
            with cols[i % len(cols)]:
                st.markdown('<div class="lv-thumb">', unsafe_allow_html=True)
                try:
                    if f.name.lower().endswith(".pdf"):
                        from pdf2image import convert_from_bytes
                        page_img = convert_from_bytes(f.getvalue(), dpi=100)[0]
                        st.image(page_img, width='stretch')
                    else:
                        st.image(f.getvalue(), width='stretch')
                except Exception:
                    st.info("Preview unavailable")
                st.markdown(f'<div class="lv-thumb-caption">{f.name} · '
                           f'{f.size / 1024:.1f} KB</div></div>',
                           unsafe_allow_html=True)

    process_clicked = st.button(
        f"Process {len(files)} report(s)" if files else "Process report(s)",
        type="primary", disabled=not files)
    st.markdown('</div>', unsafe_allow_html=True)

    if files and process_clicked:
        progress = st.progress(0.0)
        for i, f in enumerate(files, start=1):
            with st.status(f"Processing {f.name}…", expanded=False) as status:
                try:
                    doc_id = save_upload(user["id"], f.name, f.getvalue())
                    process_document(doc_id)
                    status.update(label=f.name, state="complete")
                except NoDeployedModelError as e:
                    status.update(label=f"⛔ {f.name}: service unavailable",
                                  state="error")
                    st.error(str(e))
                    break
                except Exception as e:
                    status.update(label=f"{f.name}: {e}", state="error")
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
                _render_result(ext, source_path=latest.stored_path)
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
        else:
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.caption(f"{len(docs)} report(s)")
            with top_r:
                if st.button("Delete all", icon=":material/delete_sweep:",
                            width='stretch'):
                    _confirm_delete_all(user["id"])

            PAGE_SIZE = 8
            total_pages = max(1, -(-len(docs) // PAGE_SIZE))
            current_page = min(st.session_state.get("_history_page", 1), total_pages)
            st.session_state["_history_page"] = current_page

            start = (current_page - 1) * PAGE_SIZE
            page_docs = docs[start:start + PAGE_SIZE]

            for d in page_docs:
                status_icon = {"done": "check_circle", "failed": "cancel",
                              "processing": "hourglass_top"}.get(d.status, "description")
                status_color = {"done": "#1F7A46", "failed": "#A32D2D",
                               "processing": "#8A5A00"}.get(d.status, "#6B6472")

                with st.container(border=True):
                    exp_l, exp_r = st.columns([6, 1], vertical_alignment="center")
                    with exp_l:
                        expanded = st.expander(
                            f"{d.filename} — "
                            f"{d.created_at.strftime('%d %b %Y %H:%M')}",
                            icon=f":material/{status_icon}:")
                    with exp_r:
                        if st.button("", icon=":material/delete:", key=f"del_{d.id}",
                                    help="Delete this report", width='stretch'):
                            _confirm_delete_one(d.id, d.filename)

                    with expanded:
                        if d.status == "failed":
                            st.error(d.error or "Processing failed.")
                            if st.button("Retry", icon=":material/refresh:", key=f"retry_{d.id}"):
                                try:
                                    process_document(d.id)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                        elif d.status == "done":
                            ext = _latest_extraction(db, d.id)
                            if ext:
                                _render_result(ext, source_path=d.stored_path)
                        else:
                            st.info(f"Status: {d.status}")

            if total_pages > 1:
                st.divider()
                p1, p2, p3 = st.columns([1, 2, 1])
                with p1:
                    if st.button("← Previous", disabled=current_page <= 1,
                                width='stretch'):
                        st.session_state["_history_page"] -= 1
                        st.rerun()
                with p2:
                    st.markdown(
                        f"<div style='text-align:center; font-size:13px; "
                        f"color:#6B6472; padding-top:8px;'>"
                        f"Page {current_page} of {total_pages}</div>",
                        unsafe_allow_html=True,
                    )
                with p3:
                    if st.button("Next →", disabled=current_page >= total_pages,
                                width='stretch'):
                        st.session_state["_history_page"] += 1
                        st.rerun()
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
                                        mime=mime, width='stretch')
                st.divider()
                _render_result(ext, source_path=choice.stored_path)
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