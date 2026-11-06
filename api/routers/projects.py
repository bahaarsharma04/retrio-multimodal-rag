from typing import List
from fastapi import APIRouter, HTTPException, Path, status
from api.schemas import ProjectCreateRequest, ProjectResponse
from core.projects import list_projects, create_project, get_project, delete_project

router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])


@router.get("", response_model=List[ProjectResponse], summary="List all project workspaces")
def get_all_projects():
    return list_projects()


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED, summary="Create a new project")
def create_new_project(payload: ProjectCreateRequest):
    new_proj = create_project(payload.name, payload.description or "")
    return new_proj


@router.get("/{project_id}", response_model=ProjectResponse, summary="Get project details")
def get_project_by_id(project_id: str = Path(..., description="Project workspace ID")):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )
    return proj


@router.delete("/{project_id}", summary="Delete a project workspace and all its data")
def delete_project_by_id(project_id: str = Path(..., description="Project workspace ID")):
    proj = get_project(project_id)
    if not proj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )
    success = delete_project(project_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete project '{project_id}'.",
        )
    return {
        "message": f"Project '{project_id}' and all associated data deleted successfully.",
        "project_id": project_id,
    }
