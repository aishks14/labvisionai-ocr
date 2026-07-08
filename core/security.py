"""
LabVisionAI — Password hashing & JWT tokens
============================================
bcrypt for password storage, HS256 JWT for API auth. Shared by the
FastAPI backend and both Streamlit portals.
"""

import datetime as dt

import bcrypt
import jwt

from config.settings import (ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM,
                             SECRET_KEY)


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        return False


def create_access_token(email: str, role: str) -> str:
    payload = {
        "sub": email,
        "role": role,
        "exp": dt.datetime.utcnow() + dt.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
