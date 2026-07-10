"""
LabVisionAI — Shared portal helpers
====================================
Login form, session handling, role gate, shared visual theme, and the
icon sidebar (with bottom-docked profile/sign-out) used by both
Streamlit portals.
"""

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from database.db import SessionLocal
from database.models import User
from core.security import hash_password, verify_password


# ======================================================
# Brand
# ======================================================

PRIMARY = "#52185A"
PRIMARY_DARK = "#3B1041"
PRIMARY_TINT = "#F4ECF5"
INK = "#1A1620"
INK_SECONDARY = "#6B6472"
BORDER = "#E4DEE6"

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "labvisionai-logo.png"


@st.cache_data
def _load_logo_base64() -> str | None:
    if not LOGO_PATH.exists():
        return None
    return base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")


# ======================================================
# Global theme — applied on every authenticated page
# ======================================================

GLOBAL_THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; color: {INK}; font-weight: 600 !important; }}

/* ---- Hard reset: zero border-radius everywhere ---- */
*, *::before, *::after {{ border-radius: 0 !important; }}

/* ---- Sidebar shell ---- */
section[data-testid="stSidebar"] {{
    background: #17111A;
    border-right: 1px solid #2A2130;
}}
section[data-testid="stSidebar"] * {{ color: #E7E1EA !important; }}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {{
    height: 100%;
}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {{
    height: 100%;
}}

.lv-side-brand {{
    display: flex; align-items: center; gap: 10px;
    padding: 6px 0 14px 0;
    border-bottom: 1px solid #2A2130;
    margin-bottom: 6px;
}}
.lv-side-brand img {{ height: 32px; width: auto; }}
.lv-side-name {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    font-size: 16px; color: #FFFFFF !important; line-height: 1.15;
}}
.lv-side-tag {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.06em; color: #B98FC2 !important;
    text-transform: uppercase; margin-top: 2px;
}}

/* ---- Nav (real buttons — tertiary = inactive, primary = active) ---- */
.lv-nav {{ display: flex; flex-direction: column; gap: 2px; margin-bottom: 4px; }}
.lv-nav button {{
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 9px 12px !important;
    border-left: 3px solid transparent !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="tertiary"] {{
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="tertiary"]:hover {{
    background: #241A29 !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {{
    background: #2C1F32 !important;
    border-left: 3px solid {PRIMARY} !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="tertiary"] p,
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="tertiary"] span {{
    color: #D8D0DC !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] p,
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] span {{
    color: #FFFFFF !important;
}}

/* ---- Bottom-docked profile block ---- */
.lv-sidebar-spacer {{ flex: 1 1 auto; min-height: 12px; }}
.lv-profile {{
    display: flex; align-items: center; gap: 10px;
    padding: 12px 4px 6px 4px;
    border-top: 1px solid #2A2130;
    margin-top: 10px;
}}
.lv-avatar {{
    width: 32px; height: 32px; flex-shrink: 0;
    background: {PRIMARY};
    color: #FFFFFF !important;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 14px;
}}
.lv-profile-text {{ overflow: hidden; }}
.lv-profile-name {{
    font-size: 13px; font-weight: 500; color: #FFFFFF !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.lv-profile-email {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    color: #9A8FA0 !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
section[data-testid="stSidebar"] .lv-signout div[data-testid="stButton"] button {{
    background: transparent !important;
    border: 1px solid #4A3D50 !important;
    font-size: 12.5px !important;
    padding: 6px 0 !important;
}}
section[data-testid="stSidebar"] .lv-signout div[data-testid="stButton"] button p,
section[data-testid="stSidebar"] .lv-signout div[data-testid="stButton"] button span {{
    color: #F0EAF2 !important;
}}
section[data-testid="stSidebar"] .lv-signout div[data-testid="stButton"] button:hover {{
    background: #2C1F32 !important;
    border-color: {PRIMARY} !important;
}}

/* ---- Buttons ---- */
button[kind="primary"] {{
    background: {PRIMARY} !important; border: none !important;
}}
button[kind="primary"]:hover {{ background: {PRIMARY_DARK} !important; }}
div[data-testid="stFormSubmitButton"] button[kind="primary"] p,
button[kind="primary"] p {{ color: #FFFFFF !important; }}

/* ---- Metrics ---- */
div[data-testid="stMetric"] {{
    background: {PRIMARY_TINT}; border: 1px solid {BORDER};
    padding: 14px 16px;
}}
div[data-testid="stMetricValue"] {{ color: {PRIMARY} !important; font-family: 'Space Grotesk', sans-serif; }}
div[data-testid="stMetricLabel"] {{ color: {INK_SECONDARY} !important; }}

/* ---- Cards / containers ---- */
.lv-card {{
    background: #FFFFFF; border: 1px solid {BORDER};
    padding: 20px 22px; margin-bottom: 4px;
}}
.lv-trust-strip {{
    display: flex; gap: 18px; flex-wrap: wrap;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: {INK_SECONDARY}; letter-spacing: 0.02em; margin: -6px 0 18px 0;
}}

/* ---- Uploader ---- */
div[data-testid="stFileUploader"] section {{
    border: 1.5px dashed #C6AECB; background: {PRIMARY_TINT};
}}

/* ---- Tables ---- */
div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {{
    border: 1px solid {BORDER} !important;
    overflow: hidden;
}}

/* ---- Tabs ---- */
div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {PRIMARY} !important; }}
div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{ background-color: {PRIMARY} !important; }}

/* ---- Text inputs ---- */
div[data-testid="stTextInput"] input:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 3px rgba(82, 24, 90, 0.15) !important;
}}
</style>
"""


def apply_theme() -> None:
    st.markdown(GLOBAL_THEME_CSS, unsafe_allow_html=True)


# ======================================================
# Advanced table styling
# ======================================================

_FLAG_COLORS = {
    "HIGH": "background-color:#FBE4E6;color:#8A1F2B;font-weight:600",
    "LOW": "background-color:#FFF1DC;color:#8A5A00;font-weight:600",
    "NORMAL": "background-color:#E9F5EC;color:#1F7A46;font-weight:600",
    "active": "background-color:#E9F5EC;color:#1F7A46;font-weight:600",
    "candidate": "background-color:#FFF1DC;color:#8A5A00;font-weight:600",
    "frozen": "background-color:#F1EEF2;color:#6B6472;font-weight:600",
    "done": "background-color:#E9F5EC;color:#1F7A46;font-weight:600",
    "failed": "background-color:#FBE4E6;color:#8A1F2B;font-weight:600",
    "processing": "background-color:#FFF1DC;color:#8A5A00;font-weight:600",
    "saved": "background-color:#E9F5EC;color:#1F7A46;font-weight:600",
    "skipped": "background-color:#F1EEF2;color:#6B6472;font-weight:600",
    "disabled": "background-color:#FBE4E6;color:#8A1F2B;font-weight:600",
}


def styled_df(df: pd.DataFrame, flag_col: str | None = None, color_map: dict | None = None):
    """
    Wrap a DataFrame with header + zebra-row + conditional coloring on
    one column (flag values, status values, etc.) for use with
    st.dataframe (read-only tables only — Styler objects aren't
    accepted by st.data_editor).
    """
    colors = color_map or _FLAG_COLORS
    styler = df.style
    if flag_col and flag_col in df.columns:
        def _match(v):
            text = str(v).strip().lower()
            for key, css in colors.items():
                if key.lower() in text:
                    return css
            return ""
        styler = styler.map(_match, subset=[flag_col])
    styler = styler.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", PRIMARY), ("color", "white"),
            ("font-weight", "600"), ("text-transform", "uppercase"),
            ("font-size", "11px"), ("letter-spacing", "0.04em"),
            ("text-align", "left"), ("padding", "8px 10px"),
        ]},
        {"selector": "td", "props": [("padding", "7px 10px")]},
        {"selector": "tr:nth-child(even)", "props": [("background-color", "#FAF7FB")]},
    ])
    return styler


# ======================================================
# Icon sidebar
# ======================================================

def render_sidebar(app_name: str, tagline: str, nav_items: list[tuple[str, str]],
                   user: dict) -> str:
    """
    Renders the themed sidebar: logo/brand, icon nav (real buttons — not
    a restyled radio, which fights BaseWeb's internals unreliably), and
    a bottom-docked avatar + name/email + Sign out control.

    nav_items: list of (material_icon_name, label) tuples, e.g.
               [("dashboard", "Dashboard"), ("upload_file", "Upload Report")]
               Icon names come from Google's Material Symbols set.
    Returns the selected page's plain label.
    """
    apply_theme()

    logo_b64 = _load_logo_base64()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}"/>' if logo_b64 else "🔬"

    nav_key = "_lv_page"
    if nav_key not in st.session_state or st.session_state[nav_key] not in [l for _, l in nav_items]:
        st.session_state[nav_key] = nav_items[0][1]

    with st.sidebar:
        st.markdown(
            f"""<div class="lv-side-brand">{logo_html}
                <div>
                    <div class="lv-side-name">{app_name}</div>
                    <div class="lv-side-tag">{tagline}</div>
                </div></div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="lv-nav">', unsafe_allow_html=True)
        for icon, label in nav_items:
            active = st.session_state[nav_key] == label
            if st.button(label, icon=f":material/{icon}:", key=f"_navbtn_{label}",
                        width='stretch',
                        type="primary" if active else "tertiary"):
                st.session_state[nav_key] = label
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="lv-sidebar-spacer"></div>', unsafe_allow_html=True)

        display_name = user.get("name") or user["email"].split("@")[0]
        initial = display_name[0].upper() if display_name else "?"
        st.markdown(
            f"""<div class="lv-profile">
                <div class="lv-avatar">{initial}</div>
                <div class="lv-profile-text">
                    <div class="lv-profile-name">{display_name}</div>
                    <div class="lv-profile-email">{user['email']}</div>
                </div></div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="lv-signout">', unsafe_allow_html=True)
        if st.button("Sign out", icon=":material/logout:",
                    width='stretch', key="_lv_signout"):
            st.session_state.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state[nav_key]


# ======================================================
# Sign-in page CSS
# ======================================================

_SIGN_IN_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500&family=Inter:wght@400;500&family=JetBrains+Mono:wght@500&display=swap');

*, *::before, *::after {{ border-radius: 0 !important; }}

.block-container {{
    max-width: 560px;
    padding-top: 3rem;
}}

.lv-logo {{
    display: flex;
    justify-content: center;
}}

.lv-logo img {{
    height: 125px;
    width: auto;
}}

.lv-brand {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-bottom: 2px;
}}

.lv-brand-name {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: {INK_SECONDARY};
}}

.lv-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 26px;
    color: {INK};
    text-align: center;
}}

.lv-subtitle {{
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: {INK_SECONDARY};
    margin: 0 0 18px 0;
    text-align: center;
}}

.lv-ticks {{
    display: flex;
    justify-content: center;
    gap: 2px;
    margin-bottom: 22px;
}}

.lv-ticks span {{
    display: inline-block;
    width: 3px;
    height: 14px;
    background: {PRIMARY};
}}

.lv-card {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
    padding: 28px 36px 26px 36px;
}}

div[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 500;
    color: {INK_SECONDARY};
}}

div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {PRIMARY};
}}

div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY};
}}

div[data-testid="stTextInput"] label {{
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: {INK_SECONDARY};
}}

div[data-testid="stTextInput"] input {{
    height: 42px;
    border: 1px solid {BORDER};
    font-size: 14px;
    color: {INK};
    background: #FBFAFB;
}}

div[data-testid="stTextInput"] input:focus {{
    border-color: {PRIMARY};
    box-shadow: 0 0 0 3px rgba(82, 24, 90, 0.15);
}}

div[data-testid="stFormSubmitButton"] button {{
    width: 100%;
    height: 42px;
    background: {PRIMARY};
    color: #FFFFFF;
    border: none;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 500;
    margin-top: 6px;
}}

div[data-testid="stFormSubmitButton"] button:hover {{
    background: {PRIMARY_DARK};
    color: #FFFFFF;
}}

div[data-testid="stFormSubmitButton"] button p {{
    color: #FFFFFF;
}}

div[data-testid="stAlert"] {{
    font-family: 'Inter', sans-serif;
    font-size: 13px;
}}
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


def register_admin(email: str, password: str, full_name: str) -> str:
    """Same shape as register_customer, for admin self-registration if
    the admin portal ever calls require_login(..., allow_signup=True).
    Off by default — see the note on require_login below."""
    db = SessionLocal()
    try:
        if db.query(User).filter_by(email=email).first():
            return "Email already registered."
        db.add(User(email=email, password_hash=hash_password(password),
                    full_name=full_name, organization="", role="admin"))
        db.commit()
        return ""
    finally:
        db.close()


def require_login(required_role: str | None = None, allow_signup: bool = False):
    """
    Render login (and optional signup) until a valid session exists.

    NOTE on admin signup: the admin portal currently calls this with
    allow_signup left at its default (False) — admin accounts are not
    self-registrable, which is the safer default for an internal ops
    tool. The signup form below already asks for a full name and works
    identically for either role; if you do want open admin signup,
    pass allow_signup=True from portals/admin/app.py and swap the
    register_customer(...) call a few lines down for register_admin(...)
    when required_role == "admin".
    """
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
            if st.form_submit_button("Sign in", width='stretch'):
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
                if st.form_submit_button("Create account", width='stretch'):
                    if len(pw) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        err = register_customer(email.strip().lower(), pw, name, org)
                        st.error(err) if err else st.success("Account created — sign in.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def logout_button():
    """Kept for backward compatibility — render_sidebar() now includes
    a bottom-docked sign-out control, so new code shouldn't need this."""
    with st.sidebar:
        st.caption(f"Signed in as {st.session_state['user']['email']}")
        if st.button("Sign out", width='stretch'):
            st.session_state.clear()
            st.rerun()