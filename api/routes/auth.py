"""Auth routes: register (customer) and login -> JWT."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

from api.deps import get_db
from core.security import create_access_token, hash_password, verify_password
from database.models import User

router = APIRouter()


class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    full_name: str = ""
    organization: str = ""


class LoginIn(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
def register(body: RegisterIn, db=Depends(get_db)):
    if db.query(User).filter_by(email=body.email).first():
        raise HTTPException(409, "Email already registered")
    user = User(email=body.email, password_hash=hash_password(body.password),
                full_name=body.full_name, organization=body.organization,
                role="customer")
    db.add(user)
    db.commit()
    return {"id": user.id, "email": user.email, "role": user.role}


@router.post("/login")
def login(body: LoginIn, db=Depends(get_db)):
    user = db.query(User).filter_by(email=body.email, is_active=True).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_access_token(user.email, user.role),
            "token_type": "bearer", "role": user.role}
