"""
Authentication Module for Artha Drishti
Handles user login, registration, and session management.
User data is persisted in a local JSON file.
Passwords are hashed with SHA-256 + salt.
"""

import json
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / ".auth_data"
USERS_FILE = DATA_DIR / "users.json"


def _ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({"users": {}}, indent=2))


def _load_users() -> dict:
    _ensure_data_dir()
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return {"users": {}}


def _save_users(data: dict):
    _ensure_data_dir()
    USERS_FILE.write_text(json.dumps(data, indent=2, default=str))


def _hash_password(password: str, salt: str = None) -> tuple:
    """Hash a password with SHA-256 + salt. Returns (hash, salt)."""
    if salt is None:
        salt = uuid.uuid4().hex
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return hashed, salt


# ---- Public API ----

def register_user(username: str, password: str, full_name: str = "", email: str = "") -> tuple:
    """
    Register a new user. Returns (success: bool, message: str).
    """
    username = username.strip().lower()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."

    data = _load_users()
    if username in data["users"]:
        return False, "Username already exists."

    hashed, salt = _hash_password(password)
    data["users"][username] = {
        "password_hash": hashed,
        "salt": salt,
        "full_name": full_name.strip(),
        "email": email.strip(),
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "settings": {
            "default_exchange": "NSE",
            "theme": "dark",
            "notifications": True,
        },
        "watchlist": [],
        "portfolio": [],
        "alerts": [],
        "trade_journal": [],
    }
    _save_users(data)
    return True, "Registration successful!"


def authenticate(username: str, password: str) -> tuple:
    """
    Authenticate a user. Returns (success: bool, user_data: dict | message: str).
    """
    username = username.strip().lower()
    data = _load_users()
    user = data["users"].get(username)

    if not user:
        return False, "Invalid username or password."

    hashed, _ = _hash_password(password, user["salt"])
    if hashed != user["password_hash"]:
        return False, "Invalid username or password."

    # Update last login
    user["last_login"] = datetime.now().isoformat()
    _save_users(data)

    return True, user


def update_user_data(username: str, key: str, value):
    """Update a specific field in the user's profile."""
    username = username.strip().lower()
    data = _load_users()
    if username in data["users"]:
        data["users"][username][key] = value
        _save_users(data)
        return True
    return False


def get_user_data(username: str) -> dict:
    """Get the full user record."""
    username = username.strip().lower()
    data = _load_users()
    return data["users"].get(username, {})


def save_user_watchlist(username: str, watchlist: list):
    return update_user_data(username, "watchlist", watchlist)


def save_user_portfolio(username: str, portfolio: list):
    return update_user_data(username, "portfolio", portfolio)


def save_user_alerts(username: str, alerts: list):
    return update_user_data(username, "alerts", alerts)


def save_user_trade_journal(username: str, trade_journal: list):
    return update_user_data(username, "trade_journal", trade_journal)


def save_user_settings(username: str, settings: dict):
    return update_user_data(username, "settings", settings)


def change_password(username: str, old_password: str, new_password: str) -> tuple:
    """Change a user's password. Returns (success, message)."""
    username = username.strip().lower()
    data = _load_users()
    user = data["users"].get(username)
    if not user:
        return False, "User not found."

    hashed, _ = _hash_password(old_password, user["salt"])
    if hashed != user["password_hash"]:
        return False, "Current password is incorrect."

    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."

    new_hash, new_salt = _hash_password(new_password)
    user["password_hash"] = new_hash
    user["salt"] = new_salt
    _save_users(data)
    return True, "Password changed successfully."


def get_all_usernames() -> list:
    data = _load_users()
    return list(data["users"].keys())
