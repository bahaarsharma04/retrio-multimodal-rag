import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy.sql import func
from core.config import PROJECTS_DIR, DEFAULT_LLM_MODEL, OPENAI_API_KEY
from db.session import get_db_session, is_db_connected, init_db
from db.models import Conversation, Message


def get_project_chats_dir(project_id: str) -> Path:
    """Returns the persistent chats directory for a project."""
    chats_dir = PROJECTS_DIR / project_id / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)
    return chats_dir


# ====================================================================
# JSON STORAGE FALLBACK IMPLEMENTATIONS
# ====================================================================

def _list_chats_json(project_id: str) -> List[Dict[str, Any]]:
    chats_dir = get_project_chats_dir(project_id)
    chats = []
    for f in chats_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                chats.append(data)
        except Exception:
            pass
    chats.sort(key=lambda c: c.get("updated_timestamp", c.get("created_timestamp", 0)), reverse=True)
    return chats


def _get_chat_json(project_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
    chats_dir = get_project_chats_dir(project_id)
    chat_file = chats_dir / f"{chat_id}.json"
    if chat_file.exists():
        try:
            with open(chat_file, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None
    return None


def _save_chat_json(
    project_id: str,
    chat_id: str,
    title: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    chats_dir = get_project_chats_dir(project_id)
    chat_file = chats_dir / f"{chat_id}.json"

    now_ts = time.time()
    now_str = time.strftime("%d/%m/%Y %H:%M")

    created_ts = now_ts
    created_str = now_str
    if chat_file.exists():
        existing = _get_chat_json(project_id, chat_id)
        if existing:
            created_ts = existing.get("created_timestamp", now_ts)
            created_str = existing.get("created_at", now_str)

    chat_data = {
        "chat_id": chat_id,
        "project_id": project_id,
        "title": title.strip() or "Untitled Chat",
        "created_at": created_str,
        "created_timestamp": created_ts,
        "updated_at": now_str,
        "updated_timestamp": now_ts,
        "messages": messages,
    }

    with open(chat_file, "w", encoding="utf-8") as fp:
        json.dump(chat_data, fp, indent=2, ensure_ascii=False)

    return chat_data


def _rename_chat_json(project_id: str, chat_id: str, new_title: str) -> bool:
    chat = _get_chat_json(project_id, chat_id)
    if not chat:
        return False
    chat["title"] = new_title.strip() or "Untitled Chat"
    chat["updated_at"] = time.strftime("%d/%m/%Y %H:%M")
    chat["updated_timestamp"] = time.time()
    chat_file = get_project_chats_dir(project_id) / f"{chat_id}.json"
    with open(chat_file, "w", encoding="utf-8") as fp:
        json.dump(chat, fp, indent=2, ensure_ascii=False)
    return True


def _delete_chat_json(project_id: str, chat_id: str) -> bool:
    chat_file = get_project_chats_dir(project_id) / f"{chat_id}.json"
    if chat_file.exists():
        chat_file.unlink()
        return True
    return False


def _sync_json_to_db(session, project_id: str):
    """Automatically migrates any legacy JSON chats on disk into PostgreSQL."""
    json_chats = _list_chats_json(project_id)
    for jc in json_chats:
        cid = jc.get("chat_id")
        if not cid:
            continue
        existing = session.query(Conversation).filter(Conversation.id == cid).first()
        if not existing:
            conv = Conversation(
                id=cid,
                project_id=project_id,
                title=jc.get("title", "Untitled Chat"),
            )
            session.add(conv)
            session.flush()

            for m in jc.get("messages", []):
                msg = Message(
                    conversation_id=cid,
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    sources=m.get("sources"),
                )
                session.add(msg)
    try:
        session.commit()
    except Exception:
        session.rollback()


# ====================================================================
# PRIMARY CHAT MANAGER (POSTGRESQL + ORM WITH DUAL-MODE FALLBACK)
# ====================================================================

def list_chats(project_id: str) -> List[Dict[str, Any]]:
    """Lists all saved chats for a given project, sorted newest first."""
    if is_db_connected():
        try:
            init_db()
            with get_db_session() as session:
                _sync_json_to_db(session, project_id)
                convs = (
                    session.query(Conversation)
                    .filter(Conversation.project_id == project_id)
                    .order_by(Conversation.updated_at.desc())
                    .all()
                )
                return [c.to_dict() for c in convs]
        except Exception as e:
            print(f"Warning: Database error in list_chats, falling back to JSON: {e}")

    return _list_chats_json(project_id)


def get_chat(project_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
    """Loads a specific chat session."""
    if is_db_connected():
        try:
            with get_db_session() as session:
                conv = (
                    session.query(Conversation)
                    .filter(Conversation.id == chat_id, Conversation.project_id == project_id)
                    .first()
                )
                if conv:
                    return conv.to_dict()
        except Exception as e:
            print(f"Warning: Database error in get_chat, falling back to JSON: {e}")

    return _get_chat_json(project_id, chat_id)


def save_chat(
    project_id: str,
    chat_id: str,
    title: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Saves or updates a conversation session in PostgreSQL (and syncs to JSON)."""
    if is_db_connected():
        try:
            init_db()
            with get_db_session() as session:
                conv = session.query(Conversation).filter(Conversation.id == chat_id).first()
                clean_title = title.strip() or "Untitled Chat"

                if not conv:
                    conv = Conversation(
                        id=chat_id,
                        project_id=project_id,
                        title=clean_title,
                    )
                    session.add(conv)
                else:
                    conv.title = clean_title
                    conv.updated_at = func.now()

                session.flush()

                # Replace messages with current dialogue state
                session.query(Message).filter(Message.conversation_id == chat_id).delete()
                for m in messages:
                    msg = Message(
                        conversation_id=chat_id,
                        role=m.get("role", "user"),
                        content=m.get("content", ""),
                        sources=m.get("sources"),
                    )
                    session.add(msg)

                session.commit()
                session.refresh(conv)

                # Keep JSON in sync as secondary backup
                _save_chat_json(project_id, chat_id, clean_title, messages)
                return conv.to_dict()
        except Exception as e:
            print(f"Warning: Database error in save_chat, falling back to JSON: {e}")

    return _save_chat_json(project_id, chat_id, title, messages)


def rename_chat(project_id: str, chat_id: str, new_title: str) -> bool:
    """Renames an existing chat."""
    if is_db_connected():
        try:
            with get_db_session() as session:
                conv = session.query(Conversation).filter(Conversation.id == chat_id).first()
                if conv:
                    conv.title = new_title.strip() or "Untitled Chat"
                    conv.updated_at = func.now()
                    session.commit()
                    _rename_chat_json(project_id, chat_id, new_title)
                    return True
        except Exception as e:
            print(f"Warning: Database error in rename_chat, falling back to JSON: {e}")

    return _rename_chat_json(project_id, chat_id, new_title)


def delete_chat(project_id: str, chat_id: str) -> bool:
    """Deletes a chat from database and disk."""
    if is_db_connected():
        try:
            with get_db_session() as session:
                conv = session.query(Conversation).filter(Conversation.id == chat_id).first()
                if conv:
                    session.delete(conv)
                    session.commit()
                    _delete_chat_json(project_id, chat_id)
                    return True
        except Exception as e:
            print(f"Warning: Database error in delete_chat, falling back to JSON: {e}")

    return _delete_chat_json(project_id, chat_id)


def generate_chat_title(messages: List[Dict[str, Any]]) -> str:
    """
    Generates a concise 3-5 word title based on the first user query.
    Uses OpenAI gpt-4o-mini if API key is present, otherwise cleans the first question.
    """
    first_user_msg = ""
    for m in messages:
        if m.get("role") == "user":
            first_user_msg = m.get("content", "").strip()
            break

    if not first_user_msg:
        return "New Conversation"

    # Fast heuristic title fallback
    words = first_user_msg.split()
    fallback_title = " ".join(words[:5])
    if len(fallback_title) > 35:
        fallback_title = fallback_title[:32] + "..."

    # If OpenAI key is set, generate a clean 3-5 word summary title
    api_key = OPENAI_API_KEY
    if api_key and "your-openai" not in api_key:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.0, api_key=api_key)
            resp = llm.invoke([
                SystemMessage(content="Generate a concise 3 to 5 word title for this user query. Return ONLY the title, no punctuation or quotes."),
                HumanMessage(content=first_user_msg)
            ])
            generated = resp.content.strip().strip('"').strip("'")
            if generated and len(generated) < 45:
                return generated
        except Exception:
            pass

    return fallback_title
