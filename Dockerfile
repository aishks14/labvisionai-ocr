# LabVisionAI — single image serving API + both portals (compose picks the command)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m scripts.init_system || true

EXPOSE 8000 8501 8502
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
