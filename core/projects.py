import json
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.config import PROJECTS_DIR


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
