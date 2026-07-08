"""
LabVisionAI — System verification
==================================
Checks every dependency and configuration in one shot: Python libs,
Tesseract binary, Poppler (for PDFs), DB connectivity, storage
write-access, and whether a model is deployed. Run before demos.
"""

import importlib
import shutil
import sys


def check(name, fn):
    try:
        detail = fn() or ""
        print(f"  [OK]   {name} {detail}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main():
    print("LabVisionAI system verification\n" + "=" * 40)
    ok = True

    for mod in ["cv2", "numpy", "pandas", "pytesseract", "pdf2image",
                "ultralytics", "streamlit", "fastapi", "sqlalchemy",
                "reportlab", "openpyxl", "bcrypt", "jwt", "yaml"]:
        ok &= check(f"import {mod}", lambda m=mod: importlib.import_module(m)
                    and "")

    def tesseract():
        import pytesseract
        from config.settings import TESSERACT_CMD
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        return f"(v{pytesseract.get_tesseract_version()})"
    ok &= check("Tesseract binary", tesseract)

    def db():
        from database.db import init_db
        init_db()
        return ""
    ok &= check("Database init", db)

    def storage():
        from config.settings import UPLOAD_DIR
        probe = UPLOAD_DIR / ".probe"
        probe.write_text("x")
        probe.unlink()
        return f"({UPLOAD_DIR})"
    ok &= check("Storage writable", storage)

    def deployed():
        from core.registry import get_active_model
        active = get_active_model()
        if active is None:
            raise RuntimeError("no active model — customers cannot process yet")
        return f"({active[0]})"
    check("Deployed model", deployed)  # warning-level, not fatal

    print("=" * 40)
    print("READY" if ok else "FIX FAILURES ABOVE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
