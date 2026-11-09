"""
Authentication and Role-Based Access Control (RBAC) Module for LoanWatch.

Provides:
- PBKDF2-HMAC-SHA256 password hashing and constant-time verification.
- Session-based user authentication backed by .streamlit/secrets.toml.
- Lockout protection (blocks attempts for 30s after 5 failures).
- Centralized role-to-page permission mappings and write authorization.
- UI gatekeeper (require_login) enforcing authentication and role barriers across all pages.
- Safe handling of missing or malformed secrets (generic error, no stack traces).
"""

import hashlib
import hmac
from pathlib import Path
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

# Pre-computed dummy hash to ensure constant-time response when user ID does not exist
DUMMY_SALT = "00000000000000000000000000000000"
DUMMY_HASH = "pbkdf2_sha256$200000$00000000000000000000000000000000$0000000000000000000000000000000000000000000000000000000000000000"

# Centralized Role-to-Page Permissions Mapping
PAGE_PERMISSIONS: Dict[str, List[str]] = {
    "1_Manager_Dashboard": ["admin", "collection_manager", "risk_manager", "business_head"],
    "2_Upload_Data": ["admin", "collection_manager", "risk_manager"],
    "3_Risk_Prediction": ["admin", "collection_manager", "risk_manager", "collector"],
    "4_Collector_Worklist": ["admin", "collection_manager", "risk_manager", "collector"],
    "5_Loan_Details": ["admin", "collection_manager", "risk_manager", "collector"],
}

ROLE_PERMISSIONS: Dict[str, List[str]] = {
    "admin": ["1_Manager_Dashboard", "2_Upload_Data", "3_Risk_Prediction", "4_Collector_Worklist", "5_Loan_Details"],
    "collection_manager": ["1_Manager_Dashboard", "2_Upload_Data", "3_Risk_Prediction", "4_Collector_Worklist", "5_Loan_Details"],
    "risk_manager": ["1_Manager_Dashboard", "2_Upload_Data", "3_Risk_Prediction", "4_Collector_Worklist", "5_Loan_Details"],
    "business_head": ["1_Manager_Dashboard"],
    "collector": ["3_Risk_Prediction", "4_Collector_Worklist", "5_Loan_Details"],
}

# Follow-up write permission: only collection_manager, admin, and collector (own accounts)
CAN_WRITE_FOLLOWUPS_ROLES = {"admin", "collection_manager", "collector"}


def can_write_followups(role: str) -> bool:
    """Returns True if the role has permission to write/update collection follow-ups."""
    return role in CAN_WRITE_FOLLOWUPS_ROLES


def is_page_allowed(role: str, page_name: str) -> bool:
    """Checks if a given role is authorized to view a specific page."""
    allowed = PAGE_PERMISSIONS.get(page_name, [])
    return role in allowed


def hash_password(password: str, iterations: int = 200000) -> str:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with a random 16-byte salt.
    Format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    salt = secrets.token_bytes(16)
    hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${hash_bytes.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verifies a plaintext password against a stored PBKDF2-HMAC-SHA256 hash.
    Format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    Returns False (never raises) for malformed hashes, invalid iterations, or empty/non-string inputs.
    """
    if not isinstance(password, str) or not isinstance(stored_hash, str):
        return False

    parts = stored_hash.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False

    try:
        iterations = int(parts[1])
        if iterations <= 0:
            return False
        salt = bytes.fromhex(parts[2])
        expected_hash = bytes.fromhex(parts[3])
    except (ValueError, TypeError):
        return False

    try:
        computed_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(computed_hash, expected_hash)
    except Exception:
        return False


def get_configured_users() -> Optional[Dict[str, Any]]:
    """
    Safely retrieves the users dictionary from st.secrets.
    Returns None if secrets are missing, unconfigured, or malformed.
    """
    try:
        if "auth" not in st.secrets:
            return None
        auth_conf = st.secrets["auth"]
        if "users" not in auth_conf:
            return None
        users = auth_conf["users"]
        if not hasattr(users, "items") and not isinstance(users, dict):
            return None
        return users
    except Exception:
        return None


def is_locked_out() -> Tuple[bool, int]:
    """Checks if the current session is locked out due to excessive failed attempts."""
    lockout_until = st.session_state.get("auth_lockout_until", 0.0)
    now = time.time()
    if now < lockout_until:
        remaining = int(lockout_until - now) + 1
        return True, remaining
    return False, 0


def login(user_id: str, password: str) -> bool:
    """
    Authenticates user credentials against st.secrets configuration.
    Includes constant-time dummy hash check on unknown users.
    Implements 5-attempt lockout for 30 seconds.
    Stores authenticated session data on success.
    """
    locked, remaining = is_locked_out()
    if locked:
        return False

    users = get_configured_users()
    if users is None:
        return False

    user_conf = users.get(user_id)
    if user_conf is None:
        # Run dummy verification to simulate consistent execution time
        verify_password(password, DUMMY_HASH)
        _record_failed_attempt()
        return False

    stored_hash = user_conf.get("password_hash", "")
    if verify_password(password, stored_hash):
        # Successful authentication: reset lockout metrics
        st.session_state["auth_failed_attempts"] = 0
        st.session_state["auth_lockout_until"] = 0.0
        st.session_state["auth_user"] = {
            "user_id": user_id,
            "name": str(user_conf.get("name", user_id)),
            "role": str(user_conf.get("role", "collector")),
            "collector_id": str(user_conf.get("collector_id", "ALL")),
        }
        return True
    else:
        _record_failed_attempt()
        return False


def _record_failed_attempt() -> None:
    """Increments the failed login attempt counter and sets lockout timestamp if threshold reached."""
    current_fails = st.session_state.get("auth_failed_attempts", 0) + 1
    st.session_state["auth_failed_attempts"] = current_fails
    if current_fails >= 5:
        st.session_state["auth_lockout_until"] = time.time() + 30.0
        st.session_state["auth_failed_attempts"] = 0


def logout() -> None:
    """Clears authentication keys from st.session_state and triggers a rerun."""
    if "auth_user" in st.session_state:
        del st.session_state["auth_user"]
    st.rerun()


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns the authenticated user dictionary, or None if unauthenticated."""
    return st.session_state.get("auth_user")


def render_sidebar_auth(current_user: Dict[str, Any]) -> None:
    """
    Renders user profile and sign-out control in the Streamlit sidebar.
    Injects CSS rules hiding navigation links for unauthorized pages.
    """
    with st.sidebar:
        st.markdown(f"👤 **{current_user['name']}**")
        st.caption(f"Role: `{current_user['role']}`")
        if current_user.get("collector_id") and current_user["collector_id"] != "ALL":
            st.caption(f"Assigned ID: `{current_user['collector_id']}`")
        if st.button("Sign Out", key="auth_signout_btn", use_container_width=True):
            logout()
        st.divider()

    # CSS to hide navigation items the user role is not permitted to see
    user_role = current_user.get("role", "")
    hidden_pages = [page for page, allowed in PAGE_PERMISSIONS.items() if user_role not in allowed]

    css_rules = []
    for p in hidden_pages:
        css_rules.append(f'[data-testid="stSidebarNav"] a[href*="{p}"] {{ display: none !important; }}')
        css_rules.append(f'[data-testid="stSidebarNav"] li:has(a[href*="{p}"]) {{ display: none !important; }}')

    if css_rules:
        st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)


def render_login_form() -> None:
    """Renders a clean login form. Never shows credentials, hints, or sample logins."""
    st.markdown(
        "<div style='text-align: center; margin-top: 40px; margin-bottom: 24px;'>"
        "<h2 style='color: #1e3d59;'>🏦 LoanWatch – Sign In</h2>"
        "<p style='color: #7f8c8d; font-size: 14px;'>Enter your credentials to access the early warning system.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        locked, remaining = is_locked_out()
        if locked:
            st.error(f"Too many failed attempts. Access blocked for {remaining} seconds.")
            st.stop()

        with st.form("login_form", clear_on_submit=False):
            user_id = st.text_input("User ID", key="auth_input_user_id").strip()
            password = st.text_input("Password", type="password", key="auth_input_password")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submitted:
                if not user_id or not password:
                    st.error("Invalid user ID or password")
                else:
                    success = login(user_id, password)
                    if success:
                        st.rerun()
                    else:
                        st.error("Invalid user ID or password")


def require_login(allowed_roles: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Gatekeeper function called at the very top of each page.
    - If secrets are missing or malformed, displays generic 'Application is not configured' and stops.
    - If unauthenticated, displays the clean login form and stops.
    - If authenticated but unauthorized for the page, displays an access error and stops.
    - Returns the authenticated user dictionary.
    """
    # 1. Validate secrets configuration safely without stack traces
    users = get_configured_users()
    if users is None:
        st.error("Application is not configured")
        st.stop()

    # 2. Check current session
    user = get_current_user()
    if user is None:
        # Hide sidebar completely when not logged in
        st.markdown("<style>[data-testid='stSidebar'] { display: none !important; }</style>", unsafe_allow_html=True)
        render_login_form()
        st.stop()

    # 3. Render sidebar controls and role-based page hiding
    render_sidebar_auth(user)

    # 4. Check page authorization
    if allowed_roles is not None and user["role"] not in allowed_roles:
        st.warning("You do not have access to this page")
        st.stop()

    return user
