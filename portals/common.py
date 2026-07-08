"""
LabVisionAI — Shared portal helpers
====================================
Login form, session handling, and role gate used by both Streamlit
portals. The Customer Portal gate blocks admins-only pages entirely.
"""

import base64
from pathlib import Path

import streamlit as st

from database.db import SessionLocal
from database.models import User
from core.security import hash_password, verify_password


# ======================================================
# Logo
# ======================================================

# Expects the logo at <project_root>/assets/labvisionai-logo.png
# Adjust this path if your assets folder lives elsewhere.
LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "labvisionai-logo.png"


@st.cache_data
def _load_logo_base64() -> str | None:
    if not LOGO_PATH.exists():
        return None
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


# ======================================================
# Sign-in page CSS
# ======================================================

_SIGN_IN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');

.block-container {
    max-width: 560px;
    padding-top: 3rem;
}

.lv-logo {
    display: flex;
    justify-content: center;
}

.lv-logo img {
    height: 125px;
    width: 125px;
    width: auto;
}

.lv-brand {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-bottom: 2px;
}

.lv-brand-mark {
    width: 22px;
    height: 22px;
    border-radius: 5px;
    background: #0F6E56;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
}

.lv-brand-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: #57697A;
}

.lv-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 26px;
    color: #10202C;
    text-align: center;
}

.lv-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #57697A;
    margin: 0 0 18px 0;
    text-align: center;
}

.lv-ticks {
    display: flex;
    justify-content: center;
    gap: 2px;
    margin-bottom: 22px;
}

.lv-ticks span {
    display: inline-block;
    width: 3px;
    height: 14px;
    background: #0F6E56;
}

.lv-card {
    background: #FFFFFF;
    border: 1px solid #E0E6E9;
    border-radius: 12px;
    padding: 28px 36px 26px 36px;
}

div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 500;
    color: #57697A;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0F6E56;
}

div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {
    background-color: #0F6E56;
}

div[data-testid="stTextInput"] label {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #57697A;
}

div[data-testid="stTextInput"] input {
    height: 42px;
    border: 1px solid #E0E6E9;
    border-radius: 8px;
    font-size: 14px;
    color: #10202C;
    background: #FBFCFC;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #0F6E56;
    box-shadow: 0 0 0 3px rgba(15, 110, 86, 0.15);
}

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    height: 42px;
    background: #0F6E56;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 500;
    margin-top: 6px;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: #0B5847;
    color: #FFFFFF;
}

div[data-testid="stFormSubmitButton"] button p {
    color: #FFFFFF;
}

div[data-testid="stAlert"] {
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
}
</style>
"""


def authenticate(email: str, password: str) -> User | None:
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email, is_active=True).first()
        if user and verify_password(password, user.password_hash):
            return user
        return None
    finally:
        db.close()


def register_customer(email: str, password: str, full_name: str, org: str) -> str:
    db = SessionLocal()
    try:
        if db.query(User).filter_by(email=email).first():
            return "Email already registered."
        db.add(User(email=email, password_hash=hash_password(password),
                    full_name=full_name, organization=org, role="customer"))
        db.commit()
        return ""
    finally:
        db.close()


def require_login(required_role: str | None = None, allow_signup: bool = False):
    """Render login (and optional signup) until a valid session exists."""
    if "user" in st.session_state:
        user = st.session_state["user"]
        if required_role and user["role"] != required_role:
            st.error("You do not have access to this portal.")
            st.stop()
        return user

    st.markdown(_SIGN_IN_CSS, unsafe_allow_html=True)

    logo_b64 = _load_logo_base64()
    logo_html = (
        f'<div class="lv-logo"><img src="data:image/png;base64,{logo_b64}" /></div>'
        if logo_b64
        else ""
    )

    st.markdown(
        f"""
        {logo_html}
        <div class="lv-brand">
            <span class="lv-brand-name">LABVISIONAI</span>
        </div>
        <div class="lv-title">Sign in</div>
        <div class="lv-subtitle">Access your lab report workspace.</div>
        <div class="lv-ticks">
            <span style="opacity:1;"></span>
            <span style="opacity:0.6;width:2px;"></span>
            <span style="opacity:0.35;width:4px;"></span>
            <span style="opacity:0.7;width:2px;"></span>
            <span style="opacity:0.25;"></span>
            <span style="opacity:0.5;width:2px;"></span>
            <span style="opacity:0.2;width:5px;"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lv-card">', unsafe_allow_html=True)
    tabs = st.tabs(["Login", "Create account"]) if allow_signup else [st.container()]

    with tabs[0]:
        with st.form("login"):
            email = st.text_input("Email", placeholder="name@company.com")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            if st.form_submit_button("Sign in", use_container_width=True):
                user = authenticate(email.strip().lower(), password)
                if user is None:
                    st.error("Invalid credentials.")
                elif required_role and user.role != required_role:
                    st.error("This account cannot access this portal.")
                else:
                    st.session_state["user"] = {"id": user.id, "email": user.email,
                                                "name": user.full_name,
                                                "role": user.role}
                    st.rerun()

    if allow_signup:
        with tabs[1]:
            with st.form("signup"):
                email = st.text_input("Work email", placeholder="name@company.com")
                name = st.text_input("Full name", placeholder="Jane Doe")
                org = st.text_input("Organization / Hospital", placeholder="Acme Diagnostics")
                pw = st.text_input("Password", type="password", placeholder="At least 8 characters")
                if st.form_submit_button("Create account", use_container_width=True):
                    if len(pw) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        err = register_customer(email.strip().lower(), pw, name, org)
                        st.error(err) if err else st.success("Account created — sign in.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def logout_button():
    with st.sidebar:
        st.caption(f"Signed in as {st.session_state['user']['email']}")
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
