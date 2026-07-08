# 🔬 LabVisionAI

**AI-powered lab report digitization.** Upload a scanned blood report — get a clean, structured, editable table in Excel, CSV, JSON, PDF, or FHIR. Built the way commercial document-AI products (Textract, Document AI, Nanonets) are actually built: **the customer only ever touches inference; the ML lifecycle lives in a separate internal platform.**

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│  CUSTOMER PORTAL :8501  │        │  ADMIN PORTAL (internal) :8502│
│  Dashboard              │        │  Dashboard    Model Registry  │
│  Upload Report          │        │  Datasets     Deployments     │
│  History                │        │  Annotation   Users           │
│  Export                 │        │  Training     System Logs     │
│  Settings               │        └──────────────┬───────────────┘
└───────────┬─────────────┘                       │ promote / freeze
            │ inference only                      ▼
            ▼                          ┌────────────────────┐
   ┌────────────────────┐   loads      │   MODEL REGISTRY   │
   │  INFERENCE PIPELINE │◄────────────│  v1.0.0  frozen    │
   │  PDF→pages→YOLO→    │  ACTIVE     │  v1.1.0  ACTIVE ◄──│
   │  crop→OCR→parse→DB  │  model only │  v1.2.0  candidate │
   └────────────────────┘              └────────────────────┘
```

The customer never sees datasets, annotation, or training. The admin never has to touch customer uploads. One frozen `best.pt` — promoted deliberately by a human — serves every customer request.

## How a customer experiences it

1. Signs up / logs in on the **Customer Portal**
2. Drops `BloodReport.pdf` on **Upload Report**
3. The backend loads the **deployed** model → YOLO detects the 9 field classes → each crop is preprocessed (3× upscale, grayscale, blur, Otsu, invert) → Tesseract reads it → the post-processor assembles clean rows with LOW/HIGH/NORMAL flags
4. Customer reviews & corrects the table inline, then downloads **Excel / CSV / JSON / PDF / FHIR** from **Export**

**Zero training. Inference only.**

## How the AI team experiences it

1. Collect reports → annotate (Annotation page, or LabelImg/CVAT)
2. `python -m training.prepare_dataset --source <folder> --name labreports_v2`
3. Train from the **Training** page (or `python -m training.train_yolo ... `, or Colab GPU) → the new `best.pt` lands in the **Model Registry** as a *candidate*
4. Compare metrics (`python -m training.evaluate`), then **Promote** — the previous active version is automatically frozen
5. Customers instantly get the better model; nothing else changes

## Quickstart (local, no cloud, ₹0)

```bash
pip install -r requirements.txt
# system deps: tesseract-ocr + poppler (Windows: install both, set LVA_TESSERACT_CMD in .env)

python -m scripts.verify_system                      # check everything
python -m scripts.init_system --weights path/to/best.pt   # admin acct + deploy your existing model
python -m scripts.demo_seed                          # demo customer acct

streamlit run portals/customer/app.py --server.port 8501   # what hospitals see
streamlit run portals/admin/app.py    --server.port 8502   # what your team sees
uvicorn api.main:app --port 8000                            # REST API (docs at /docs)
```

Default admin: `admin@labvisionai.local / ChangeMe#2026` · Demo customer: `customer@demo.local / Demo#2026`

Or with Docker: `docker compose up --build` (admin portal is bound to localhost only — mirror this with a VPN in production).

## Repository map

| Layer | Path | What it does |
|---|---|---|
| Config | `config/settings.py` | Every path, threshold, class list, and secret |
| Data | `database/models.py`, `db.py` | Users, documents, extractions, model versions, audit log |
| Engine | `core/` | Registry, detector, preprocessing, OCR, parser, pipeline, exporters, security |
| API | `api/` | FastAPI: auth, customer document routes, admin registry routes |
| Portals | `portals/customer`, `portals/admin` | The two products |
| ML lifecycle | `training/` | Dataset prep, YOLO training, evaluation |
| Ops | `scripts/`, `tests/`, `Dockerfile`, `docker-compose.yml` | Bootstrap, verification, CI-ready tests, deployment |

## Tech stack

Python 3.11 · YOLOv8 (Ultralytics) · Tesseract/pytesseract · OpenCV · Streamlit · FastAPI · SQLAlchemy (SQLite→Postgres-ready) · ReportLab · openpyxl · JWT + bcrypt · Docker

## Roadmap

- [ ] PaddleOCR as a drop-in alternative behind `core/ocr_engine.read_region`
- [ ] Async processing queue (Celery/RQ) for batch uploads
- [ ] Per-customer usage metering & API keys
- [ ] Canvas-based annotation (streamlit-drawable-canvas) replacing coordinate entry
