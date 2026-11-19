import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Path as FPath, Request, Depends, status
from typing import Dict, Any

from core.projects import get_project
from core.pipeline import (
    get_project_docs_dir,
    run_ingestion_pipeline,
    load_document_pipeline_data,
    delete_source_document,
)
from core.limiter import limiter
from api.deps import get_current_active_user

router = APIRouter(prefix="/api/v1/projects/{project_id}/documents", tags=["Knowledge Base & Ingestion"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".md"}


@router.get("", summary="List all source documents in a project (ownership verified)")
@limiter.limit("60/minute")
def list_project_documents(
    request: Request,
    project_id: str = FPath(..., description="Project workspace ID"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    docs_dir = get_project_docs_dir(project_id)
    files = []
    if docs_dir.exists():
        for f in docs_dir.iterdir():
            if f.is_file() and not f.name.endswith(".pipeline.json") and not f.name.endswith(".checkpoint.json"):
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                files.append({
                    "filename": f.name,
                    "size_mb": size_mb,
                    "extension": f.suffix.lower(),
                    "has_pipeline_data": (docs_dir / f"{f.name}.pipeline.json").exists(),
                })
    return files


@router.post("", status_code=status.HTTP_201_CREATED, summary="Upload and ingest a document into the knowledge base (ownership verified)")
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    project_id: str = FPath(..., description="Project workspace ID"),
    file: UploadFile = File(..., description="Document file (.pdf, .docx, .csv, .txt, .md)"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    safe_filename = Path(file.filename).name
    ext = Path(safe_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {list(ALLOWED_EXTENSIONS)}",
        )

    docs_dir = get_project_docs_dir(project_id)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest_path = docs_dir / safe_filename

    # Save uploaded file safely to disk
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Execute 5-stage ingestion pipeline
    try:
        pipeline_result = run_ingestion_pipeline(project_id, dest_path)
        return {
            "message": f"Document '{safe_filename}' uploaded and ingested successfully.",
            "filename": safe_filename,
            "status": pipeline_result.get("status", "completed"),
            "elements_discovered": pipeline_result.get("stages", {}).get("partitioning", {}).get("stats", {}),
            "chunks_created": pipeline_result.get("stages", {}).get("chunking", {}).get("stats", {}),
            "vectors_stored": pipeline_result.get("stages", {}).get("vectorization", {}).get("stats", {}),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline failed: {str(e)}",
        )


@router.get("/{filename}/pipeline", summary="Retrieve 5-stage pipeline telemetry for a document (ownership verified)")
@limiter.limit("60/minute")
def get_pipeline_telemetry(
    request: Request,
    project_id: str = FPath(..., description="Project workspace ID"),
    filename: str = FPath(..., description="Filename including extension"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    safe_filename = Path(filename).name
    data = load_document_pipeline_data(project_id, safe_filename)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pipeline data found for '{safe_filename}'.",
        )
    return data


@router.delete("/{filename}", summary="Delete document and remove its embeddings from ChromaDB (ownership verified)")
@limiter.limit("15/minute")
def delete_document(
    request: Request,
    project_id: str = FPath(..., description="Project workspace ID"),
    filename: str = FPath(..., description="Filename including extension"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    safe_filename = Path(filename).name
    success = delete_source_document(project_id, safe_filename)
    return {"message": f"Document '{safe_filename}' and associated vectors deleted successfully."}
