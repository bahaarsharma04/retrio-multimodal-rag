import json
import uuid
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.config import PROJECTS_DIR
from core.logger import app_logger
from db.session import is_db_connected, get_db_session
from db.models import Project


def get_project_file_path(project_id: str) -> Path:
    return PROJECTS_DIR / project_id / "project.json"


def _attach_owner_info(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enriches project dictionaries with owner_email for administrative transparency."""
    from core.users import get_user_by_id
    user_cache: Dict[str, Optional[str]] = {}
    for p in projects:
        oid = p.get("owner_id")
        if oid and "owner_email" not in p:
            if oid not in user_cache:
                u = get_user_by_id(oid)
                user_cache[oid] = u.get("email") if u else None
            p["owner_email"] = user_cache[oid]
        elif "owner_email" not in p:
            p["owner_email"] = None
    return projects


def list_projects(owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of existing projects.
    When owner_id is provided (e.g. from authenticated API or UI sessions),
    strictly filters to that user's projects.
    """
    if is_db_connected():
        try:
            with get_db_session() as session:
                query = session.query(Project)
                if owner_id:
                    query = query.filter(Project.owner_id == owner_id)
                db_projects = query.order_by(Project.created_at.desc()).all()
                result = [p.to_dict() for p in db_projects]
                return _attach_owner_info(result)
        except Exception as e:
            app_logger.warning(f"Error querying projects from PostgreSQL: {e}")

    # Fallback to local disk storage
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
                        if owner_id and data.get("owner_id") != owner_id:
                            continue
                        projects.append(data)
                except Exception:
                    pass

    projects.sort(key=lambda p: p.get("created_timestamp", 0.0), reverse=True)
    return _attach_owner_info(projects)


def create_project(
    name: str,
    description: str = "",
    owner_id: Optional[str] = None,
    template_type: str = "default",
    is_master_template: bool = False,
) -> Dict[str, Any]:
    """Creates a new project workspace directory and metadata record."""
    project_id = str(uuid.uuid4())
    pdir = PROJECTS_DIR / project_id
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "documents").mkdir(exist_ok=True)

    project_data = {
        "id": project_id,
        "owner_id": owner_id,
        "name": name.strip(),
        "description": description.strip(),
        "template_type": template_type,
        "is_master_template": is_master_template,
        "created_at": time.strftime("%d/%m/%Y"),
        "created_timestamp": time.time(),
    }

    # Save to local disk
    with open(pdir / "project.json", "w", encoding="utf-8") as f:
        json.dump(project_data, f, indent=2)

    # Save to PostgreSQL if connected
    if is_db_connected():
        try:
            with get_db_session() as session:
                new_project = Project(
                    id=project_id,
                    name=name.strip(),
                    description=description.strip(),
                    owner_id=owner_id,
                    template_type=template_type,
                    is_master_template=is_master_template,
                )
                session.add(new_project)
                session.commit()
        except Exception as e:
            app_logger.warning(f"Failed to save project to PostgreSQL: {e}")

    app_logger.info(
        f"Project '{project_id}' (Name: '{name.strip()}', Owner: {owner_id}, "
        f"Template: {template_type}, Master: {is_master_template}) created successfully."
    )
    return project_data


def get_project(project_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetches metadata for a single project by ID.
    Strictly verifies ownership against owner_id. Returns None if not matched or not found.
    """
    if is_db_connected():
        try:
            with get_db_session() as session:
                query = session.query(Project).filter(Project.id == project_id)
                if owner_id:
                    query = query.filter(Project.owner_id == owner_id)
                proj = query.first()
                if proj:
                    res = proj.to_dict()
                    return _attach_owner_info([res])[0]
                return None
        except Exception as e:
            app_logger.warning(f"Error querying project by ID from PostgreSQL: {e}")

    # Fallback to local disk
    pfile = get_project_file_path(project_id)
    if pfile.exists():
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                data = json.load(f)
                if owner_id and data.get("owner_id") != owner_id:
                    return None
                return _attach_owner_info([data])[0]
        except Exception:
            return None
    return None


def delete_project(project_id: str, owner_id: Optional[str] = None) -> bool:
    """
    Completely deletes a project workspace and all associated resources.
    Strictly verifies ownership before deletion.
    """
    pdir = PROJECTS_DIR / project_id
    project_meta = get_project(project_id, owner_id=owner_id)
    if not project_meta:
        return False

    project_name = project_meta.get("name", "Unknown")

    # 1. Delete PostgreSQL records if DB is connected (cascades to documents, chunks, conversations)
    if is_db_connected():
        try:
            with get_db_session() as session:
                session.query(Project).filter(Project.id == project_id).delete(synchronize_session=False)
                session.commit()
        except Exception as e:
            app_logger.warning(f"Failed to delete PostgreSQL records for project {project_id}: {e}")

    # 2. Clean up files on disk
    if pdir.exists():
        pfile = pdir / "project.json"
        if pfile.exists():
            try:
                pfile.unlink()
            except Exception as e:
                app_logger.warning(f"Failed to remove project.json for {project_id}: {e}")

        docs_dir = pdir / "documents"
        if docs_dir.exists():
            shutil.rmtree(docs_dir, ignore_errors=True)

        chats_dir = pdir / "chats"
        if chats_dir.exists():
            shutil.rmtree(chats_dir, ignore_errors=True)

        shutil.rmtree(pdir, ignore_errors=True)

    app_logger.info(f"Project '{project_id}' (Name: '{project_name}') and all associated vectors, documents, and chat records deleted.")
    return True


def get_master_compliance_project_ids() -> List[str]:
    """
    Finds all Master Compliance project IDs.
    Returns project IDs:
    1. Where is_master_template == True and template_type == 'auditzo_compliance'
    2. Or owned by adminzestminds@gmail.com with template_type == 'auditzo_compliance' or is_master_template == True
    3. Or any project with template_type == 'auditzo_compliance' and is_master_template == True
    """
    master_ids = set()
    if is_db_connected():
        try:
            with get_db_session() as session:
                from db.models import User
                admin = session.query(User).filter(User.email == "adminzestminds@gmail.com").first()
                admin_id = admin.id if admin else None

                # Projects explicitly marked as master template
                p_masters = session.query(Project).filter(
                    (Project.is_master_template == True) | (Project.owner_id == admin_id)
                ).filter(
                    (Project.template_type == "auditzo_compliance") | (Project.is_master_template == True)
                ).all()

                for p in p_masters:
                    master_ids.add(p.id)
        except Exception as e:
            app_logger.warning(f"Error resolving master compliance project IDs: {e}")

    # Fallback to local disk storage
    if PROJECTS_DIR.exists():
        for pdir in PROJECTS_DIR.iterdir():
            if pdir.is_dir():
                pfile = pdir / "project.json"
                if pfile.exists():
                    try:
                        with open(pfile, "r", encoding="utf-8") as f:
                            d = json.load(f)
                            if d.get("is_master_template") or d.get("template_type") == "auditzo_compliance":
                                master_ids.add(d.get("id"))
                    except Exception:
                        pass
    return list(master_ids)


def get_master_compliance_project_id() -> Optional[str]:
    """Returns the primary master compliance project ID (preferably one with chunks)."""
    pids = get_master_compliance_project_ids()
    if not pids:
        return None
    # Prioritize project that has actual chunks
    if is_db_connected():
        try:
            with get_db_session() as session:
                from db.models import DocumentChunk
                for pid in pids:
                    count = session.query(DocumentChunk).filter(DocumentChunk.project_id == pid).count()
                    if count > 0:
                        return pid
        except Exception:
            pass
    return pids[0]


def get_target_project_ids(active_project_id: str) -> List[str]:
    """
    Returns the list of project IDs to search for retrieval.
    If the active project was created with template_type == 'auditzo_compliance',
    it includes ALL master compliance project IDs alongside the active project ID.
    """
    target_ids = [active_project_id]
    proj = get_project(active_project_id)
    if proj and proj.get("template_type") == "auditzo_compliance":
        masters = get_master_compliance_project_ids()
        for mid in masters:
            if mid and mid not in target_ids:
                target_ids.append(mid)
    return target_ids
