from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Path, Request, Depends, status

from api.schemas import ProjectCreateRequest, ProjectResponse
from core.projects import list_projects, create_project, get_project, delete_project
from core.limiter import limiter
from api.deps import get_current_active_user

router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])


@router.get("", response_model=List[ProjectResponse], summary="List all project workspaces owned by current user")
@limiter.limit("60/minute")
def get_all_projects(request: Request, current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Returns only projects belonging to the authenticated user."""
    return list_projects(owner_id=current_user["id"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED, summary="Create a new project workspace")
@limiter.limit("15/minute")
def create_new_project(
    request: Request,
    payload: ProjectCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Creates a new project owned exclusively by the authenticated user."""
    new_proj = create_project(payload.name, payload.description or "", owner_id=current_user["id"])
    return new_proj


@router.get("/{project_id}", response_model=ProjectResponse, summary="Get project details (ownership verified)")
@limiter.limit("60/minute")
def get_project_by_id(
    request: Request,
    project_id: str = Path(..., description="Project workspace ID"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Fetches details for a specific project. Strictly isolated to the owning user."""
    proj = get_project(project_id, owner_id=current_user["id"])
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )
    return proj


@router.delete("/{project_id}", summary="Delete a project workspace and all its data (ownership verified)")
@limiter.limit("10/minute")
def delete_project_by_id(
    request: Request,
    project_id: str = Path(..., description="Project workspace ID"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    """Deletes a project workspace. Strictly rejected if the project is not owned by the user."""
    proj = get_project(project_id, owner_id=current_user["id"])
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )
    success = delete_project(project_id, owner_id=current_user["id"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project '{project_id}'.",
        )
    return {
        "message": f"Project '{project_id}' and all associated data deleted successfully.",
        "project_id": project_id,
    }
