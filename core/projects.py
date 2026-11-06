import json
import uuid
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.config import PROJECTS_DIR
from core.logger import app_logger


def get_project_file_path(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "project.json"


def list_projects() -> List[Dict[str, Any]]:
    """Returns a list of all existing projects, sorted by creation date descending."""
    projects = []
    if not PROJECTS_DIR.exists():
        return projects

    for pdir in PROJECTS_DIR.iterdir():
        if pdir.is_dir():
            pfile = pdir / "project.json"
            if pfile.exists():
                try:
                    with open(pfile, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        projects.append(data)
                except Exception:
                    pass

    # Sort newest first
    projects.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return projects


def create_project(name: str, description: str = "") -> Dict[str, Any]:
    """Creates a new project workspace directory and metadata file."""
    project_id = str(uuid.uuid4())
    pdir = PROJECTS_DIR / project_id
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "documents").mkdir(exist_ok=True)
    (pdir / "chroma").mkdir(exist_ok=True)

    project_data = {
        "id": project_id,
        "name": name.strip(),
        "description": description.strip(),
        "created_at": time.strftime("%d/%m/%Y"),
        "created_timestamp": time.time(),
    }

    with open(pdir / "project.json", "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2)

    app_logger.info(f"Project '{project_id}' (Name: '{name.strip()}') created successfully.")
    return project_data


def get_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Fetches metadata for a single project by ID."""
    pfile = get_project_file_path(project_id)
    if pfile.exists():
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def delete_project(project_id: str) -> bool:
    """
    Completely deletes a project workspace:
    1. Removes all ChromaDB vector collections for this project.
    2. Removes any PostgreSQL chat conversations for this project.
    3. Deletes project.json, documents, and chats on disk.
    4. Recursively removes the physical project directory.
    """
    pdir = PROJECTS_DIR / project_id
    project_meta = get_project(project_id)
    if not pdir.exists() and not project_meta:
        return False

    project_name = project_meta.get("name", "Unknown") if project_meta else "Unknown"

    # 1. Delete ChromaDB collection and vectors
    try:
        from core.vectorstore import get_project_vectorstore
        store = get_project_vectorstore(project_id)
        collection_name = f"col_{project_id}"
        try:
            store._client.delete_collection(name=collection_name)
        except Exception:
            pass
    except Exception as e:
        app_logger.warning(f"Failed to delete ChromaDB collection for project {project_id}: {e}")

    # 2. Delete PostgreSQL records if DB is connected
    try:
        from db.session import is_db_connected, get_db_session
        from db.models import Conversation
        if is_db_connected():
            with get_db_session() as session:
                session.query(Conversation).filter(Conversation.project_id == project_id).delete(synchronize_session=False)
                session.commit()
    except Exception as e:
        app_logger.warning(f"Failed to delete PostgreSQL records for project {project_id}: {e}")

    # 3. Clean up files on disk
    if pdir.exists():
        # A. Unlink project.json first to immediately revoke project existence
        pfile = pdir / "project.json"
        if pfile.exists():
            try:
                pfile.unlink()
            except Exception as e:
                app_logger.warning(f"Failed to remove project.json for {project_id}: {e}")

        # B. Remove documents directory (raw files, checkpoints, pipelines)
        docs_dir = pdir / "documents"
        if docs_dir.exists():
            shutil.rmtree(docs_dir, ignore_errors=True)

        # C. Remove chats directory
        chats_dir = pdir / "chats"
        if chats_dir.exists():
            shutil.rmtree(chats_dir, ignore_errors=True)

        # D. Remove Chroma storage directory and root project directory
        # On Windows, ignore_errors=True prevents WinError 32 if SQLite file is still locked by active process
        shutil.rmtree(pdir, ignore_errors=True)

    app_logger.info(f"Project '{project_id}' (Name: '{project_name}') and all associated vectors, documents, and chat records deleted.")
    return True
