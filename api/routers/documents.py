import shutil
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Path as FPath, Request, Depends, status, BackgroundTasks
from typing import Dict, Any

from core.projects import get_project
from core.pipeline import (
    get_project_docs_dir,
    run_ingestion_pipeline,
    load_document_pipeline_data,
    load_document_checkpoint,
    delete_source_document,
    update_document_status_in_db,
    sha256_file,
)
from api.schemas import DocumentUploadAsyncResponse, DocumentStatusResponse
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


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentUploadAsyncResponse,
    summary="Upload and ingest a document asynchronously in background (ownership verified)",
)
@limiter.limit("15/minute")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
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

    file_hash = sha256_file(dest_path)

    # Register initial "processing" record in PostgreSQL
    doc_id = update_document_status_in_db(
        project_id=project_id,
        filename=safe_filename,
        file_path=dest_path,
        document_hash=file_hash,
        status="processing",
        user_id=current_user["id"],
    )

    # Enqueue background ingestion worker
    background_tasks.add_task(
        run_ingestion_pipeline,
        project_id=project_id,
        file_path=dest_path,
        restart=True,
        user_id=current_user["id"],
    )

    return DocumentUploadAsyncResponse(
        message=f"Document '{safe_filename}' uploaded successfully. Ingestion running in background.",
        filename=safe_filename,
        project_id=project_id,
        status="processing",
        document_id=doc_id,
    )


@router.get(
    "/{filename}/status",
    response_model=DocumentStatusResponse,
    summary="Retrieve background ingestion status for a document (ownership verified)",
)
@limiter.limit("60/minute")
def get_document_status(
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

    # 1. Check PostgreSQL Document and PipelineRun
    try:
        from db.session import is_db_connected, get_db_session
        from db.models import Document, PipelineRun
        if is_db_connected():
            with get_db_session() as s:
                doc = s.query(Document).filter(
                    Document.project_id == project_id,
                    Document.filename == safe_filename
                ).first()
                if doc:
                    run = s.query(PipelineRun).filter(
                        PipelineRun.document_id == doc.id
                    ).order_by(PipelineRun.started_at.desc()).first()
                    return DocumentStatusResponse(
                        filename=safe_filename,
                        project_id=project_id,
                        status=doc.ingestion_status or (run.status if run else "unknown"),
                        total_chunks=run.total_chunks if run else (len(doc.chunks) if doc.chunks else 0),
                        duration_seconds=run.duration_seconds if run else 0.0,
                        error_message=run.error_message if run else None,
                        started_at=run.started_at.strftime("%Y-%m-%d %H:%M:%S") if (run and run.started_at) else None,
                        finished_at=run.finished_at.strftime("%Y-%m-%d %H:%M:%S") if (run and run.finished_at) else None,
                        stages=[stg.to_dict() for stg in run.stages] if (run and run.stages) else [],
                    )
    except Exception:
        pass

    # 2. Check disk checkpoint fallback
    chk = load_document_checkpoint(project_id, safe_filename)
    if chk:
        return DocumentStatusResponse(
            filename=safe_filename,
            project_id=project_id,
            status="processing",
            total_chunks=0,
            error_message=None,
        )

    # 3. Check disk pipeline data fallback
    pipeline_data = load_document_pipeline_data(project_id, safe_filename)
    if pipeline_data:
        return DocumentStatusResponse(
            filename=safe_filename,
            project_id=project_id,
            status="completed",
            total_chunks=pipeline_data.get("stages", {}).get("chunking", {}).get("stats", {}).get("total_chunks", 0),
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No document or ingestion record found for '{safe_filename}'.",
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


@router.delete("/{filename}", summary="Delete document and remove its embeddings from PostgreSQL (ownership verified)")
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
