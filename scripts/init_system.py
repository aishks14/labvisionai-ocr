"""
LabVisionAI — First-run bootstrap
==================================
Creates DB tables, a default admin account, and (optionally) registers
existing weights as v1.0.0 and deploys them.

Usage:
    python -m scripts.init_system --admin-email you@company.com --admin-pass Secret123
    python -m scripts.init_system --weights path/to/best.pt   # also deploy a model
"""

import argparse

from core.security import hash_password
from database.db import init_db, session_scope
from database.models import User


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-email", default="admin@labvisionai.local")
    ap.add_argument("--admin-pass", default="ChangeMe#2026")
    ap.add_argument("--weights", default=None,
                    help="Optional path to an existing best.pt to register + deploy")
    args = ap.parse_args()

    init_db()
    with session_scope() as s:
        if not s.query(User).filter_by(email=args.admin_email).first():
            s.add(User(email=args.admin_email,
                       password_hash=hash_password(args.admin_pass),
                       full_name="Platform Admin", role="admin"))
            print(f"Admin created: {args.admin_email}")
        else:
            print("Admin already exists — skipped.")

    if args.weights:
        from core.registry import promote_model, register_model
        register_model("v1.0.0", args.weights, notes="bootstrap import",
                       actor="bootstrap")
        promote_model("v1.0.0", actor="bootstrap")
        print("v1.0.0 registered and deployed as the active model.")

    print("\nNext steps:")
    print("  Customer portal : streamlit run portals/customer/app.py --server.port 8501")
    print("  Admin portal    : streamlit run portals/admin/app.py --server.port 8502")
    print("  REST API        : uvicorn api.main:app --port 8000")


if __name__ == "__main__":
    main()
