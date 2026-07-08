"""
LabVisionAI — ADMIN PORTAL (Internal AI Platform)
==================================================
The company's AI lab. Only role='admin' accounts can sign in.
Pages: Dashboard · Datasets · Annotation · Training · Model Registry
       · Deployments · Users · System Logs

Customers never see this app — it runs on a different port (and in
production, behind a VPN / internal network).

Run:  streamlit run portals/admin/app.py --server.port 8502
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from config.settings import (APP_NAME, APP_VERSION, DATASET_DIR,
                             FIELD_CLASSES)
from core.auto_annotate import auto_annotate
from core.registry import (get_active_model, list_models, promote_model,
                           retire_model)
from database.db import SessionLocal, init_db
from database.models import (AuditLog, Dataset, Document, ModelVersion, User)
from portals.common import logout_button, require_login

st.set_page_config(page_title=f"{APP_NAME} — Admin", page_icon="🛠️",
                   layout="wide")
init_db()

user = require_login(required_role="admin")
logout_button()

st.sidebar.title(f"🛠️ {APP_NAME} Admin")
page = st.sidebar.radio("Navigation",
                        ["Dashboard", "Datasets", "Annotation", "Training",
                         "Model Registry", "Deployments", "Users",
                         "System Logs"])


def _db():
    return SessionLocal()


# ------------------------------------------------------------ Dashboard ----
if page == "Dashboard":
    st.title("Platform Dashboard")
    db = _db()
    try:
        active = get_active_model()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Deployed model", active[0] if active else "NONE ⚠️")
        c2.metric("Registered models", db.query(ModelVersion).count())
        c3.metric("Customers",
                  db.query(User).filter_by(role="customer").count())
        c4.metric("Documents processed",
                  db.query(Document).filter_by(status="done").count())

        if active is None:
            st.error("No model is deployed. Customers cannot process reports "
                     "until you promote a version in **Model Registry**.")

        st.subheader("Recent processing activity")
        docs = (db.query(Document).order_by(Document.created_at.desc())
                .limit(15).all())
        if docs:
            st.dataframe(pd.DataFrame(
                [{"ID": d.id, "File": d.filename, "Owner": d.owner.email
                  if d.owner else "?", "Status": d.status,
                  "Model": d.model_version,
                  "When": d.created_at.strftime("%d %b %H:%M")}
                 for d in docs]), use_container_width=True, hide_index=True)
    finally:
        db.close()

# -------------------------------------------------------------- Datasets ---
elif page == "Datasets":
    st.title("Datasets")
    st.caption("Prepared YOLO datasets. Build new ones with "
               "`python -m training.prepare_dataset --source <dir> --name <name>` "
               "or the form below.")
    db = _db()
    try:
        rows = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
        if rows:
            st.dataframe(pd.DataFrame(
                [{"Name": d.name, "Images": d.n_images,
                  "Annotated": d.n_annotated, "Path": d.root_path,
                  "Created": d.created_at.strftime("%d %b %Y")}
                 for d in rows]), use_container_width=True, hide_index=True)
        else:
            st.info("No datasets registered yet.")
    finally:
        db.close()

    st.subheader("Build dataset from annotated folder")
    with st.form("build_ds"):
        source = st.text_input("Source folder (images + YOLO .txt labels)")
        name = st.text_input("Dataset name", value="labreports_v1")
        ratio = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
        if st.form_submit_button("Build"):
            try:
                from training.prepare_dataset import build
                yaml_path = build(Path(source), name, ratio)
                st.success(f"Dataset ready: {yaml_path}")
            except Exception as e:
                st.error(str(e))

# ------------------------------------------------------------ Annotation ---
elif page == "Annotation":
    st.title("Annotation Tool")
    st.caption("Draw one box per field, pick its class, and export YOLO "
               "labels. For large batches use LabelImg/CVAT and import the "
               "folder via **Datasets**.")

    img_file = st.file_uploader("Report image", type=["png", "jpg", "jpeg"])
    if img_file:
        import cv2
        import numpy as np

        arr = cv2.imdecode(np.frombuffer(img_file.getvalue(), np.uint8),
                           cv2.IMREAD_COLOR)
        h, w = arr.shape[:2]
        st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB),
                 caption=f"{img_file.name} — {w}×{h}px",
                 use_container_width=True)

        if "boxes" not in st.session_state:
            st.session_state.boxes = []

        st.subheader("Auto-annotate")
        st.caption("Runs OCR + layout rules to pre-fill candidate boxes. "
                   "Review and correct them below — this is a starting "
                   "point, not a final label set.")
        ac1, ac2 = st.columns([1, 3])
        with ac1:
            if st.button("🪄 Auto-annotate", use_container_width=True):
                with st.spinner("Reading page layout…"):
                    predicted = auto_annotate(arr)
                st.session_state.boxes = predicted
                st.session_state["_auto_count"] = len(predicted)
                st.rerun()
        if st.session_state.get("_auto_count"):
            with ac2:
                st.success(f"{st.session_state['_auto_count']} candidate "
                          "boxes added — check the table below, fix any "
                          "that are wrong, then add missed fields manually.")

        st.divider()

        st.subheader("Add box (pixel coordinates)")
        c1, c2, c3, c4, c5 = st.columns(5)
        x1 = c1.number_input("x1", 0, w, 0)
        y1 = c2.number_input("y1", 0, h, 0)
        x2 = c3.number_input("x2", 0, w, min(200, w))
        y2 = c4.number_input("y2", 0, h, min(50, h))
        cls = c5.selectbox("Class", FIELD_CLASSES)
        if st.button("Add box"):
            if x2 > x1 and y2 > y1:
                st.session_state.boxes.append((FIELD_CLASSES.index(cls),
                                               x1, y1, x2, y2))
            else:
                st.error("x2/y2 must be greater than x1/y1.")

        if st.session_state.boxes:
            st.dataframe(pd.DataFrame(
                [{"class": FIELD_CLASSES[b[0]], "x1": b[1], "y1": b[2],
                  "x2": b[3], "y2": b[4]} for b in st.session_state.boxes]),
                use_container_width=True, hide_index=True)

            lines = []
            for cid, bx1, by1, bx2, by2 in st.session_state.boxes:
                cx, cy = (bx1 + bx2) / 2 / w, (by1 + by2) / 2 / h
                bw, bh = (bx2 - bx1) / w, (by2 - by1) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label_name = Path(img_file.name).with_suffix(".txt").name

            st.divider()
            st.subheader("Save to dataset folder")
            st.caption("Writes this image + its YOLO label straight to disk, "
                      "paired by filename — ready for **Datasets → Build** "
                      "once you've saved a batch this way.")

            default_folder = str(DATASET_DIR / "raw_annotated")
            save_folder = st.text_input("Destination folder", value=default_folder)

            sc1, sc2 = st.columns([1, 3])
            with sc1:
                save_clicked = st.button("💾 Save to dataset folder",
                                         use_container_width=True, type="primary")
            with sc2:
                st.download_button("Download YOLO label (.txt) only",
                                   "\n".join(lines), file_name=label_name)

            if save_clicked:
                import cv2

                dest = Path(save_folder)
                dest.mkdir(parents=True, exist_ok=True)

                img_path = dest / img_file.name
                label_path = dest / label_name

                if img_path.exists() or label_path.exists():
                    st.warning(f"{img_file.name} already exists in this "
                              "folder — saving again will overwrite it.")

                cv2.imwrite(str(img_path), arr)
                label_path.write_text("\n".join(lines))

                pairs = [p for p in dest.glob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        and p.with_suffix(".txt").exists()]
                st.success(f"Saved **{img_file.name}** + **{label_name}** to "
                          f"`{dest}`. This folder now has "
                          f"**{len(pairs)}** annotated image(s).")

            if st.button("Clear boxes"):
                st.session_state.boxes = []
                st.rerun()

# -------------------------------------------------------------- Training ---
elif page == "Training":
    st.title("Training")
    st.caption("Trains YOLOv8 on a prepared dataset and registers the "
               "result as a *candidate*. Nothing is deployed automatically.")

    datasets = sorted(p.name for p in DATASET_DIR.iterdir()
                      if (p / "data.yaml").exists()) if DATASET_DIR.exists() else []
    if not datasets:
        st.warning("No prepared datasets found. Build one in **Datasets** first.")
    else:
        with st.form("train"):
            ds = st.selectbox("Dataset", datasets)
            version = st.text_input("New version tag", value="v1.0.0")
            c1, c2, c3 = st.columns(3)
            epochs = c1.number_input("Epochs", 10, 500, 100)
            batch = c2.number_input("Batch size", 2, 64, 8)
            base = c3.selectbox("Base model",
                                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
            if st.form_submit_button("Start training", type="primary"):
                cmd = [sys.executable, "-m", "training.train_yolo",
                       "--dataset", ds, "--version", version,
                       "--epochs", str(epochs), "--batch", str(batch),
                       "--base-model", base]
                st.code(" ".join(cmd), language="bash")
                with st.spinner("Training… live log below"):
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True,
                        encoding="utf-8", errors="replace",
                        cwd=str(Path(__file__).parents[2]))
                    log_box = st.empty()
                    tail: list[str] = []
                    for line in proc.stdout:
                        tail = (tail + [line.rstrip()])[-25:]
                        log_box.code("\n".join(tail))
                    proc.wait()
                if proc.returncode == 0:
                    st.success(f"Training complete — {version} registered "
                               "as candidate. Review it in **Model Registry**.")
                else:
                    st.error("Training failed. Check the log above.")

# -------------------------------------------------------- Model Registry ---
elif page == "Model Registry":
    st.title("Model Registry")
    db = _db()
    try:
        index = list_models()
        rows = (db.query(ModelVersion)
                .order_by(ModelVersion.created_at.desc()).all())
        if not rows:
            st.info("No models registered. Train one, or register external "
                    "weights below.")
        else:
            st.dataframe(pd.DataFrame(
                [{"Version": m.version, "Status": m.status,
                  "mAP50": (m.metrics or {}).get("mAP50", "—"),
                  "Precision": (m.metrics or {}).get("precision", "—"),
                  "Recall": (m.metrics or {}).get("recall", "—"),
                  "Dataset": m.dataset_name,
                  "By": m.created_by,
                  "Created": m.created_at.strftime("%d %b %Y")}
                 for m in rows]), use_container_width=True, hide_index=True)

            versions = [m.version for m in rows]
            c1, c2 = st.columns(2)
            with c1:
                pv = st.selectbox("Promote to ACTIVE (deploy + freeze previous)",
                                  versions)
                if st.button("Promote", type="primary"):
                    promote_model(pv, actor=user["email"])
                    st.success(f"{pv} is now the deployed model.")
                    st.rerun()
            with c2:
                rv = st.selectbox("Retire version", versions)
                if st.button("Retire"):
                    try:
                        retire_model(rv, actor=user["email"])
                        st.success(f"{rv} retired.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        st.divider()
        st.subheader("Register external weights (e.g. trained on Colab GPU)")
        with st.form("reg_ext"):
            wpath = st.text_input("Path to best.pt")
            ver = st.text_input("Version tag", value="v1.0.0")
            note = st.text_input("Notes", value="trained on Colab")
            if st.form_submit_button("Register"):
                try:
                    from core.registry import register_model
                    register_model(ver, wpath, notes=note, actor=user["email"])
                    st.success(f"{ver} registered as candidate.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    finally:
        db.close()

# ------------------------------------------------------------ Deployments --
elif page == "Deployments":
    st.title("Deployments")
    active = get_active_model()
    if active:
        version, weights = active
        st.success(f"**Deployed model:** `{version}`")
        st.code(str(weights))
        size_mb = weights.stat().st_size / 1e6
        st.caption(f"Weights size: {size_mb:.1f} MB · Served to every "
                   "customer upload until a new version is promoted.")
    else:
        st.error("No active deployment. Promote a model in **Model Registry**.")

    st.subheader("Deployment history")
    db = _db()
    try:
        events = (db.query(AuditLog)
                  .filter(AuditLog.action.in_(["model_promoted",
                                               "model_registered",
                                               "model_retired"]))
                  .order_by(AuditLog.created_at.desc()).limit(50).all())
        if events:
            st.dataframe(pd.DataFrame(
                [{"When": e.created_at.strftime("%d %b %Y %H:%M"),
                  "Actor": e.actor, "Action": e.action, "Detail": e.detail}
                 for e in events]), use_container_width=True, hide_index=True)
    finally:
        db.close()

# ----------------------------------------------------------------- Users ---
elif page == "Users":
    st.title("Users")
    db = _db()
    try:
        rows = db.query(User).all()
        st.dataframe(pd.DataFrame(
            [{"ID": u.id, "Email": u.email, "Role": u.role,
              "Organization": u.organization, "Active": u.is_active,
              "Joined": u.created_at.strftime("%d %b %Y")} for u in rows]),
            use_container_width=True, hide_index=True)

        st.subheader("Toggle account status")
        target = st.selectbox("User", [u.email for u in rows
                                       if u.email != user["email"]] or ["—"])
        if target != "—" and st.button("Enable / disable"):
            row = db.query(User).filter_by(email=target).first()
            row.is_active = not row.is_active
            db.commit()
            st.success(f"{target} is now "
                       f"{'active' if row.is_active else 'disabled'}.")
            st.rerun()
    finally:
        db.close()

# ----------------------------------------------------------- System Logs ---
elif page == "System Logs":
    st.title("System Logs")
    db = _db()
    try:
        action_filter = st.text_input("Filter by action (blank = all)")
        q = db.query(AuditLog).order_by(AuditLog.created_at.desc())
        if action_filter.strip():
            q = q.filter(AuditLog.action.contains(action_filter.strip()))
        rows = q.limit(300).all()
        st.dataframe(pd.DataFrame(
            [{"When": r.created_at.strftime("%d %b %Y %H:%M:%S"),
              "Actor": r.actor, "Action": r.action, "Detail": r.detail}
             for r in rows]), use_container_width=True, hide_index=True)
    finally:
        db.close()

st.sidebar.caption(f"{APP_NAME} v{APP_VERSION} — internal platform")