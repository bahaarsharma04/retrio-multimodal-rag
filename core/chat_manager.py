import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.config import PROJECTS_DIR, DEFAULT_LLM_MODEL, OPENAI_API_KEY


def get_project_chats_dir(project_id: str) -> Path:
    """Returns the persistent chats directory for a project."""
    chats_dir = PROJECTS_DIR / project_id / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)
    return chats_dir


def list_chats(project_id: str) -> List[Dict[str, Any]]:
    """Lists all saved chats for a given project, sorted newest first."""
    chats_dir = get_project_chats_dir(project_id)
    chats = []
    for f in chats_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                chats.append(data)
        except Exception:
            pass

    # Sort by updated_timestamp descending
    chats.sort(key=lambda c: c.get("updated_timestamp", c.get("created_timestamp", 0)), reverse=True)
    return chats


def get_chat(project_id: str, chat_id: str) -> Optional[Dict[str, Any]]:
    """Loads a specific chat session."""
    chats_dir = get_project_chats_dir(project_id)
    chat_file = chats_dir / f"{chat_id}.json"
    if chat_file.exists():
        try:
            with open(chat_file, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return None
    return None


def save_chat(
    project_id: str,
    chat_id: str,
    title: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Saves or updates a conversation session on disk."""
    chats_dir = get_project_chats_dir(project_id)
    chat_file = chats_dir / f"{chat_id}.json"

    now_ts = time.time()
    now_str = time.strftime("%d/%m/%Y %H:%M")

    # If already exists, preserve creation time
    created_ts = now_ts
    created_str = now_str
    if chat_file.exists():
        existing = get_chat(project_id, chat_id)
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


def rename_chat(project_id: str, chat_id: str, new_title: str) -> bool:
    """Renames an existing chat."""
    chat = get_chat(project_id, chat_id)
    if not chat:
        return False
    chat["title"] = new_title.strip() or "Untitled Chat"
    chat["updated_at"] = time.strftime("%d/%m/%Y %H:%M")
    chat["updated_timestamp"] = time.time()
    chat_file = get_project_chats_dir(project_id) / f"{chat_id}.json"
    with open(chat_file, "w", encoding="utf-8") as fp:
        json.dump(chat, fp, indent=2, ensure_ascii=False)
    return True


def delete_chat(project_id: str, chat_id: str) -> bool:
    """Deletes a chat file from disk."""
    chat_file = get_project_chats_dir(project_id) / f"{chat_id}.json"
    if chat_file.exists():
        chat_file.unlink()
        return True
    return False


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
