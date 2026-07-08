"""
LabVisionAI — Demo seeding
===========================
Creates a demo customer account so you can showcase both portals
immediately (customer@demo.local / Demo#2026).
"""

from core.security import hash_password
from database.db import init_db, session_scope
from database.models import User

init_db()
with session_scope() as s:
    if not s.query(User).filter_by(email="customer@demo.local").first():
        s.add(User(email="customer@demo.local",
                   password_hash=hash_password("Demo#2026"),
                   full_name="Demo Hospital", organization="City Diagnostics",
                   role="customer"))
        print("Demo customer created: customer@demo.local / Demo#2026")
    else:
        print("Demo customer already exists.")
