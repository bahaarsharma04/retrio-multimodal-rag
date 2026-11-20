import json
import uuid
import time
from typing import Optional, Dict, Any, List

from core.config import DATA_DIR
from core.logger import app_logger
from core.security import hash_password, verify_password
from db.session import is_db_connected, get_db_session
from db.models import User

USERS_FILE = DATA_DIR / "users.json"


def _load_local_users() -> Dict[str, Dict[str, Any]]:
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_local_users(users: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        app_logger.warning(f"Failed to save local users file: {e}")


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by email from PostgreSQL or local JSON storage."""
    email_clean = email.strip().lower()
    if is_db_connected():
        try:
            with get_db_session() as session:
                user = session.query(User).filter(User.email == email_clean).first()
                if user:
                    user_dict = user.to_dict()
                    user_dict["hashed_password"] = user.hashed_password
                    return user_dict
        except Exception as e:
            app_logger.warning(f"Error querying user by email in PostgreSQL: {e}")

    # Fallback to local users.json
    local_users = _load_local_users()
    for uid, udata in local_users.items():
        if udata.get("email", "").lower() == email_clean:
            return udata
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user by ID from PostgreSQL or local JSON storage."""
    if is_db_connected():
        try:
            with get_db_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if user:
                    user_dict = user.to_dict()
                    user_dict["hashed_password"] = user.hashed_password
                    return user_dict
        except Exception as e:
            app_logger.warning(f"Error querying user by ID in PostgreSQL: {e}")

    # Fallback to local users.json
    local_users = _load_local_users()
    return local_users.get(user_id)


def create_user(email: str, password: str, full_name: str = "", role: str = "user") -> Dict[str, Any]:
    """Creates a new user record with a bcrypt hashed password."""
    email_clean = email.strip().lower()
    existing = get_user_by_email(email_clean)
    if existing:
        raise ValueError(f"User with email '{email_clean}' already exists.")

    user_id = str(uuid.uuid4())
    hashed_pwd = hash_password(password)

    user_data = {
        "id": user_id,
        "email": email_clean,
        "hashed_password": hashed_pwd,
        "full_name": full_name.strip(),
        "role": role,
        "is_active": True,
        "created_at": time.strftime("%d/%m/%Y %H:%M"),
    }

    # Save to PostgreSQL if connected
    if is_db_connected():
        try:
            with get_db_session() as session:
                new_user = User(
                    id=user_id,
                    email=email_clean,
                    hashed_password=hashed_pwd,
                    full_name=full_name.strip(),
                    role=role,
                    is_active=True,
                )
                session.add(new_user)
                session.commit()
                app_logger.info(f"User '{email_clean}' ({user_id}) registered in PostgreSQL.")
        except Exception as e:
            app_logger.error(f"Failed to persist user in PostgreSQL: {e}")

    # Always persist to local users.json as mirror/backup
    local_users = _load_local_users()
    local_users[user_id] = user_data
    _save_local_users(local_users)

    safe_user = {k: v for k, v in user_data.items() if k != "hashed_password"}
    return safe_user


def list_all_users() -> List[Dict[str, Any]]:
    """Returns a list of all registered users (passwords excluded)."""
    if is_db_connected():
        try:
            with get_db_session() as session:
                users = session.query(User).order_by(User.created_at.desc()).all()
                return [u.to_dict() for u in users]
        except Exception as e:
            app_logger.warning(f"Error querying all users from PostgreSQL: {e}")

    # Fallback to local users.json
    local_users = _load_local_users()
    safe_list = []
    for uid, udata in local_users.items():
        safe_list.append({k: v for k, v in udata.items() if k != "hashed_password"})
    return safe_list


def set_user_role(user_id_or_email: str, new_role: str) -> Optional[Dict[str, Any]]:
    """Promotes or changes a user's role ('user' or 'admin')."""
    if new_role not in ("user", "admin"):
        raise ValueError("Invalid role. Role must be 'user' or 'admin'.")

    clean_target = user_id_or_email.strip().lower()

    # 1. Update in PostgreSQL
    if is_db_connected():
        try:
            with get_db_session() as session:
                user = session.query(User).filter(
                    (User.id == user_id_or_email) | (User.email == clean_target)
                ).first()
                if user:
                    user.role = new_role
                    session.commit()
                    app_logger.info(f"User '{user.email}' role updated to '{new_role}' in PostgreSQL.")
        except Exception as e:
            app_logger.warning(f"Error updating user role in PostgreSQL: {e}")

    # 2. Update in local users.json
    local_users = _load_local_users()
    found_key = None
    for uid, udata in local_users.items():
        if uid == user_id_or_email or udata.get("email", "").lower() == clean_target:
            found_key = uid
            break

    if found_key:
        local_users[found_key]["role"] = new_role
        _save_local_users(local_users)
        app_logger.info(f"User '{found_key}' role updated to '{new_role}' in local storage.")
        return {k: v for k, v in local_users[found_key].items() if k != "hashed_password"}

    return None


def ensure_default_admin() -> Optional[Dict[str, Any]]:
    """Deprecated: Retrio operates on a standard user model without special admin privileges."""
    return None


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Validates user credentials. Returns safe user dictionary if valid, None otherwise."""
    user = get_user_by_email(email)
    if not user:
        return None
    if not user.get("is_active", True):
        return None
    if not verify_password(password, user.get("hashed_password", "")):
        return None

    return {k: v for k, v in user.items() if k != "hashed_password"}
