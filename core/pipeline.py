import json
import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from core.config import PROJECTS_DIR
from core.parser import parse_document
from core.chunker import chunk_elements
from core.summarizer import summarize_chunks
from core.vectorstore import index_chunks, source_ids_present
from core.logger import ingest_logger, log_timed_block


def get_project_docs_dir(project_id: str) -> Path:
    """Directory where uploaded raw files and pipeline telemetry are saved."""
    docs_dir = PROJECTS_DIR / project_id / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    return docs_dir


def get_document_record_path(project_id: str, filename: str) -> Path:
    """Path to the JSON metadata recording the pipeline stages for a document."""
    return get_project_docs_dir(project_id) / f"{filename}.pipeline.json"


def get_document_checkpoint_path(project_id: str, filename: str) -> Path:
    """Path to the temporary checkpoint metadata for an in-flight ingestion."""
    return get_project_docs_dir(project_id) / f"{filename}.checkpoint.json"


def load_document_checkpoint(project_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Loads active checkpoint data if an ingestion was previously interrupted."""
    path = get_document_checkpoint_path(project_id, filename)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_document_checkpoint(
    project_id: str,
    filename: str,
    stage_name: str,
    stage_data: Dict[str, Any],
) -> None:
    """Persists intermediate progress to disk after a stage completes."""
    checkpoint = load_document_checkpoint(project_id, filename) or {
        "filename": filename,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
    }
    checkpoint["last_completed_stage"] = stage_name
    checkpoint["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    checkpoint["stages"][stage_name] = stage_data

    path = get_document_checkpoint_path(project_id, filename)
    try:
        _atomic_json(path, checkpoint)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint for {filename}: {e}")


def delete_document_checkpoint(project_id: str, filename: str) -> None:
    """Removes the checkpoint file upon final completion or manual discard."""
    path = get_document_checkpoint_path(project_id, filename)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass


def get_interrupted_documents(project_id: str) -> List[Dict[str, Any]]:
    """Finds all documents in a project that have an incomplete ingestion checkpoint."""
    docs_dir = get_project_docs_dir(project_id)
    interrupted = []
    for f in docs_dir.iterdir():
        if f.is_file() and f.name.endswith(".checkpoint.json"):
            orig_filename = f.name[:-len(".checkpoint.json")]
            # Only consider interrupted if final pipeline record is not yet completed
            pipeline_rec = get_document_record_path(project_id, orig_filename)
            if not pipeline_rec.exists():
                chk_data = load_document_checkpoint(project_id, orig_filename) or {}
                interrupted.append({
                    "filename": orig_filename,
                    "last_stage": chk_data.get("last_completed_stage", "unknown"),
                    "updated_at": chk_data.get("updated_at", ""),
                })
    return interrupted


def load_document_pipeline_data(project_id: str, filename: str) -> Optional[Dict[str, Any]]:
    """Loads previously computed pipeline data for a file if it exists."""
    path = get_document_record_path(project_id, filename)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def sha256_file(path: Path) -> str:
    """Computes a streaming SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    """Safely writes JSON to a temp file and replaces the target atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _execute_pipeline_stages(
    project_id: str,
    file_path: Path,
    filename: str,
    file_size_kb: float,
    document_hash: str,
    source_key: str,
    restart: bool,
    progress_callback: Optional[Callable[[str, int, int], None]],
    user_id: Optional[str],
) -> Dict[str, Any]:
    previous = load_document_pipeline_data(project_id, filename)
    if not restart and previous and previous.get("status") == "completed":
        old_ids = previous.get("stages", {}).get("vectorization", {}).get("stats", {}).get("chunk_ids", [])
        if (previous.get("document_hash") == document_hash
                and old_ids and source_ids_present(project_id, old_ids)):
            ingest_logger.info(f"Ingestion skipped for unchanged '{filename}' (SHA256: {document_hash[:8]}...) | Project: {project_id}")
            result = dict(previous)
            result["status"] = "skipped_unchanged"
            result["skipped"] = True
            result["stages"]["vectorization"]["stats"] = dict(previous["stages"]["vectorization"]["stats"])
            result["stages"]["vectorization"]["stats"]["new_vectors_added"] = 0
            update_document_status_in_db(
                project_id=project_id,
                filename=filename,
                file_path=file_path,
                document_hash=document_hash,
                status="completed",
                user_id=user_id,
            )
            return result

    if restart:
        delete_document_checkpoint(project_id, filename)
    checkpoint = None if restart else load_document_checkpoint(project_id, filename)
    # Never replay stages from a different document version.
    if checkpoint and checkpoint.get("document_hash") != document_hash:
        delete_document_checkpoint(project_id, filename)
        checkpoint = None
    if checkpoint is None:
        checkpoint = {
            "filename": filename,
            "document_hash": document_hash,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": {},
        }
        _atomic_json(get_document_checkpoint_path(project_id, filename), checkpoint)
    cached_stages = checkpoint.get("stages", {})

    # ----------------- Stage 1: Partitioning -----------------
    if "partitioning" in cached_stages:
        if progress_callback:
            progress_callback("Partitioning (restored from checkpoint)", 1, 4)
        raw_elements = cached_stages["partitioning"]["raw_elements"]
        partition_stats = cached_stages["partitioning"]["stats"]
        ingest_logger.info(f"Stage 1: Partitioning restored from checkpoint for '{filename}' ({len(raw_elements)} elements)")
    else:
        if progress_callback:
            progress_callback("Partitioning", 1, 4)
        with log_timed_block(ingest_logger, f"Stage 1 Partitioning: '{filename}'", {"size_kb": file_size_kb}):
            raw_elements, partition_stats = parse_document(file_path)
            save_document_checkpoint(project_id, filename, "partitioning", {
                "raw_elements": raw_elements,
                "stats": partition_stats,
            })
            ingest_logger.info(f"Stage 1: Discovered elements in '{filename}': {partition_stats}")

    # ----------------- Stage 2: Chunking -----------------
    if "chunking" in cached_stages:
        if progress_callback:
            progress_callback("Chunking (restored from checkpoint)", 2, 4)
        chunks = cached_stages["chunking"]["chunks"]
        chunk_stats = cached_stages["chunking"]["stats"]
        ingest_logger.info(f"Stage 2: Chunking restored from checkpoint for '{filename}' ({len(chunks)} chunks)")
    else:
        if progress_callback:
            progress_callback("Chunking", 2, 4)
        with log_timed_block(ingest_logger, f"Stage 2 Chunking: '{filename}'", {"elements": len(raw_elements)}):
            chunks, chunk_stats = chunk_elements(raw_elements, source_key=source_key, project_id=project_id)
            save_document_checkpoint(project_id, filename, "chunking", {
                "chunks": chunks,
                "stats": chunk_stats,
            })
            ingest_logger.info(f"Stage 2: Chunking created {len(chunks)} chunks | Breakdown: {chunk_stats.get('chunk_type_breakdown')}")

    if not chunks:
        raise ValueError(f"No valid chunks parsed from {filename}; existing vectors preserved")

    # ----------------- Stage 3: Summarisation -----------------
    if "summarisation" in cached_stages:
        if progress_callback:
            progress_callback("Summarisation (restored from checkpoint)", 3, 4)
        enriched_chunks = cached_stages["summarisation"]["enriched_chunks"]
        summarizer_stats = cached_stages["summarisation"]["stats"]
        ingest_logger.info(f"Stage 3: Summarisation restored from checkpoint for '{filename}'")
    else:
        if progress_callback:
            progress_callback("Summarisation", 3, 4)
        with log_timed_block(ingest_logger, f"Stage 3 Summarisation: '{filename}'", {"total_chunks": len(chunks)}):
            enriched_chunks, summarizer_stats = summarize_chunks(chunks)
            save_document_checkpoint(project_id, filename, "summarisation", {
                "enriched_chunks": enriched_chunks,
                "stats": summarizer_stats,
            })
            ingest_logger.info(f"Stage 3: Summarisation completed | {summarizer_stats}")

    # ----------------- Stage 4: Vectorization & Storage -----------------
    if progress_callback:
        progress_callback("Vectorization & Storage", 4, 4)
    if not enriched_chunks:
        raise ValueError(f"No enriched chunks for {filename}; existing vectors preserved")
    with log_timed_block(ingest_logger, f"Stage 4 Vectorization: '{filename}'", {"chunks": len(enriched_chunks)}):
        vector_stats = index_chunks(
            project_id, enriched_chunks,
            filename=filename,
            document_hash=document_hash,
        )
        ingest_logger.info(f"Stage 4: Vector indexing completed | Added: {vector_stats.get('new_vectors_added')}, Reused: {vector_stats.get('reused_vectors', 0)}, Deleted: {vector_stats.get('old_vectors_deleted', 0)}, Total: {vector_stats.get('total_vectors_stored')}")

    # ----------------- Stage 5: View Chunks & Finalization -----------------
    all_chunks_for_file = [c for c in enriched_chunks if c.get("filename") == filename]

    pipeline_result = {
        "filename": filename,
        "document_hash": document_hash,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "completed",
        "stages": {
            "partitioning": {
                "stats": partition_stats,
                "elements_sample": raw_elements[:10],
            },
            "chunking": {
                "stats": chunk_stats,
            },
            "summarisation": {
                "stats": summarizer_stats,
            },
            "vectorization": {
                "stats": vector_stats,
            },
            "view_chunks": {
                "chunks": all_chunks_for_file,
                "total_chunks": len(all_chunks_for_file),
            }
        }
    }

    # Persist the final pipeline run metadata
    record_path = get_document_record_path(project_id, filename)
    _atomic_json(record_path, pipeline_result)

    # Clean up temporary checkpoint since ingestion completed successfully
    delete_document_checkpoint(project_id, filename)
    ingest_logger.info(f"✔ Pipeline successfully completed for '{filename}' | Recorded in: {record_path.name}")

    # Synchronize with PostgreSQL if database is active
    sync_document_chunks_to_db(
        project_id=project_id,
        filename=filename,
        file_path=file_path,
        document_hash=document_hash,
        chunks=all_chunks_for_file,
        user_id=user_id,
    )

    update_document_status_in_db(
        project_id=project_id,
        filename=filename,
        file_path=file_path,
        document_hash=document_hash,
        status="completed",
        user_id=user_id,
        total_chunks=len(all_chunks_for_file),
    )

    return pipeline_result


def run_ingestion_pipeline(
    project_id: str,
    file_path: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    restart: bool = False,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the complete 5-stage ingestion pipeline with checkpoint recovery:
    1. Partitioning (Text, Tables, Images, Titles)
    2. Chunking (Semantic chunking by title)
    3. Summarisation (AI summaries for tables/images)
    4. Vectorization & Storage (OpenAI Embeddings + PostgreSQL pgvector)
    5. View Chunks (Formatted chunk list with metadata)

    If interrupted, loads from the last checkpoint to avoid re-running expensive steps.
    """
    file_path = Path(file_path)
    filename = file_path.name
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    file_size_kb = round(file_path.stat().st_size / 1024, 1) if file_path.exists() else 0
    document_hash = sha256_file(file_path)
    # A document filename is the stable identity within a project.
    source_key = hashlib.sha256(f"{project_id}:{filename}".encode()).hexdigest()

    ingest_logger.info(f"Initiating ingestion pipeline for '{filename}' ({file_size_kb} KB) | Project: {project_id}")

    # Mark document as processing in PostgreSQL
    update_document_status_in_db(
        project_id=project_id,
        filename=filename,
        file_path=file_path,
        document_hash=document_hash,
        status="processing",
        user_id=user_id,
    )

    try:
        return _execute_pipeline_stages(
            project_id=project_id,
            file_path=file_path,
            filename=filename,
            file_size_kb=file_size_kb,
            document_hash=document_hash,
            source_key=source_key,
            restart=restart,
            progress_callback=progress_callback,
            user_id=user_id,
        )
    except Exception as e:
        ingest_logger.error(f"Ingestion pipeline failed for '{filename}': {e}", exc_info=True)
        update_document_status_in_db(
            project_id=project_id,
            filename=filename,
            file_path=file_path,
            document_hash=document_hash,
            status="failed",
            error_message=str(e),
            user_id=user_id,
        )
        raise


def update_document_status_in_db(
    project_id: str,
    filename: str,
    file_path: Optional[Path] = None,
    document_hash: Optional[str] = None,
    status: str = "processing",
    error_message: Optional[str] = None,
    user_id: Optional[str] = None,
    total_chunks: int = 0,
) -> Optional[str]:
    """Updates or creates a Document record and PipelineRun in PostgreSQL."""
    try:
        from db.session import is_db_connected, get_db_session
        if not is_db_connected():
            return None
        from db.models import Document, Project, PipelineRun
        from sqlalchemy import func
        import uuid

        with get_db_session() as s:
            if not user_id:
                proj = s.query(Project).filter(Project.id == project_id).first()
                if proj and proj.owner_id:
                    user_id = proj.owner_id

            doc = s.query(Document).filter(
                Document.project_id == project_id,
                Document.filename == filename
            ).first()

            if not doc:
                doc = Document(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    user_id=user_id,
                    filename=filename,
                    file_path=str(file_path) if file_path else "",
                    file_hash=document_hash or "",
                    ingestion_status=status,
                )
                s.add(doc)
                s.flush()
            else:
                if document_hash:
                    doc.file_hash = document_hash
                if file_path:
                    doc.file_path = str(file_path)
                doc.ingestion_status = status
                if user_id:
                    doc.user_id = user_id

            # Update or create PipelineRun
            run = s.query(PipelineRun).filter(
                PipelineRun.document_id == doc.id
            ).order_by(PipelineRun.started_at.desc()).first()

            if status == "processing":
                if not run or run.status != "processing":
                    run = PipelineRun(
                        id=str(uuid.uuid4()),
                        document_id=doc.id,
                        status="processing",
                    )
                    s.add(run)
            elif status == "completed":
                if run and run.status == "processing":
                    run.status = "completed"
                    run.total_chunks = total_chunks
                    run.finished_at = func.now()
            elif status == "failed":
                if run and run.status == "processing":
                    run.status = "failed"
                    run.error_message = error_message
                    run.finished_at = func.now()
                else:
                    s.add(PipelineRun(
                        id=str(uuid.uuid4()),
                        document_id=doc.id,
                        status="failed",
                        error_message=error_message,
                        finished_at=func.now(),
                    ))

            s.commit()
            return doc.id
    except Exception as e:
        ingest_logger.warning(f"Failed to update document status in DB for '{filename}': {e}")
        return None


def sync_document_chunks_to_db(
    project_id: str,
    filename: str,
    file_path: Path,
    document_hash: str,
    chunks: List[Dict[str, Any]],
    user_id: Optional[str] = None,
) -> None:
    """Synchronizes completed document chunks to PostgreSQL database if connected."""
    try:
        from db.session import is_db_connected, get_db_session
        if not is_db_connected():
            return
        from db.models import Document, DocumentChunk, Project
        import uuid

        def clean_pg_text(val):
            if val is None:
                return None
            return str(val).replace("\x00", "")

        with get_db_session() as s:
            if not user_id:
                proj = s.query(Project).filter(Project.id == project_id).first()
                if proj and proj.owner_id:
                    user_id = proj.owner_id

            doc = s.query(Document).filter(Document.project_id == project_id, Document.filename == filename).first()
            if not doc:
                doc = Document(
                    id=str(uuid.uuid4()),
                    project_id=project_id,
                    user_id=user_id,
                    filename=filename,
                    file_path=str(file_path),
                    file_hash=document_hash,
                    ingestion_status="completed",
                )
                s.add(doc)
                s.flush()
            else:
                doc.file_hash = document_hash
                doc.ingestion_status = "completed"
                if user_id:
                    doc.user_id = user_id

            # Fetch existing chunks to preserve vector embeddings during metadata sync
            existing_chunks = s.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
            existing_map = {c_obj.id: c_obj for c_obj in existing_chunks}
            new_chunk_ids = {str(c.get("chunk_id")) for c in chunks}

            # Delete any obsolete chunks no longer present
            for old_id, old_chunk in existing_map.items():
                if old_id not in new_chunk_ids:
                    s.delete(old_chunk)

            for idx, c in enumerate(chunks, 1):
                cid = str(c.get("chunk_id"))
                p_raw = c.get("page_range") or c.get("page_number") or 1
                if isinstance(p_raw, list):
                    pages = [int(x) for x in p_raw if str(x).isdigit()] or [1]
                elif "-" in str(p_raw):
                    pages = [int(p) for p in str(p_raw).split("-") if p.isdigit()] or [1]
                else:
                    pages = [int(p_raw)] if str(p_raw).isdigit() else [1]

                db_c = existing_map.get(cid)
                if db_c:
                    db_c.chunk_index = idx
                    db_c.chunk_type = c.get("type", "text")
                    db_c.title = clean_pg_text(c.get("title") or c.get("section") or "General")
                    db_c.section = clean_pg_text(c.get("section") or "General")
                    db_c.part_index = int(c.get("part_index") or 1)
                    db_c.total_parts = int(c.get("total_parts") or 1)
                    db_c.page_numbers = pages
                    db_c.content = clean_pg_text(c.get("content", ""))
                    db_c.text_as_html = clean_pg_text(c.get("text_as_html"))
                    db_c.summary = clean_pg_text(c.get("summary"))
                    db_c.indexed_text = clean_pg_text(c.get("indexed_text"))
                    db_c.content_hash = c.get("content_hash", "")
                    db_c.char_count = c.get("char_count", len(c.get("content", "")))
                else:
                    db_c = DocumentChunk(
                        id=cid,
                        document_id=doc.id,
                        project_id=project_id,
                        user_id=user_id,
                        chunk_index=idx,
                        chunk_type=c.get("type", "text"),
                        title=clean_pg_text(c.get("title") or c.get("section") or "General"),
                        section=clean_pg_text(c.get("section") or "General"),
                        part_index=int(c.get("part_index") or 1),
                        total_parts=int(c.get("total_parts") or 1),
                        page_numbers=pages,
                        content=clean_pg_text(c.get("content", "")),
                        text_as_html=clean_pg_text(c.get("text_as_html")),
                        summary=clean_pg_text(c.get("summary")),
                        indexed_text=clean_pg_text(c.get("indexed_text")),
                        content_hash=c.get("content_hash", ""),
                        char_count=c.get("char_count", len(c.get("content", ""))),
                    )
                    s.add(db_c)
            s.commit()
    except Exception as e:
        ingest_logger.warning(f"Notice: Background sync to PostgreSQL skipped: {e}")


def delete_source_document(project_id: str, filename: str) -> bool:
    """
    Deletes vector data from PostgreSQL (pgvector) and removes all associated files from disk.
    Ensures complete, case-insensitive purging with zero orphan remnants.
    """
    from core.vectorstore import delete_file_from_vectorstore
    delete_file_from_vectorstore(project_id, filename)

    documents = get_project_docs_dir(project_id)
    fn_lower = filename.lower()
    if documents.exists():
        for f in list(documents.iterdir()):
            if f.is_file():
                name_lower = f.name.lower()
                if (
                    name_lower == fn_lower
                    or name_lower == f"{fn_lower}.pipeline.json"
                    or name_lower == f"{fn_lower}.checkpoint.json"
                    or name_lower.startswith(f"{fn_lower}.")
                ):
                    try:
                        f.unlink()
                    except Exception:
                        pass

    try:
        from db.session import is_db_connected, get_db_session
        from db.models import Document
        from sqlalchemy import func
        if is_db_connected():
            with get_db_session() as s:
                docs = s.query(Document).filter(
                    Document.project_id == project_id,
                    func.lower(Document.filename) == filename.lower()
                ).all()
                for d in docs:
                    s.delete(d)
                s.commit()
    except Exception as e:
        ingest_logger.warning(f"Notice: Document record deletion from PostgreSQL skipped: {e}")

    ingest_logger.info(f"Source document '{filename}' and associated vectors deleted from Project: {project_id}")
    return True


def reconcile_project_documents(project_id: str) -> Dict[str, Any]:
    """
    Ensures a single source of truth between disk and database:
    1. Removes any ghost document records from PostgreSQL if their physical file is not on disk.
    2. Returns the synchronized list of active documents.
    """
    documents_dir = get_project_docs_dir(project_id)
    disk_files = {}
    if documents_dir.exists():
        for f in documents_dir.iterdir():
            if (
                f.is_file()
                and not f.name.endswith(".pipeline.json")
                and not f.name.endswith(".checkpoint.json")
            ):
                disk_files[f.name.lower()] = f.name

    purged_ghosts = []
    try:
        from db.session import is_db_connected, get_db_session
        from db.models import Document as DBDocument
        from core.vectorstore import delete_file_from_vectorstore

        if is_db_connected():
            with get_db_session() as s:
                db_docs = s.query(DBDocument).filter(DBDocument.project_id == project_id).all()
                for doc in db_docs:
                    if doc.filename.lower() not in disk_files:
                        purged_ghosts.append(doc.filename)

            for ghost in purged_ghosts:
                delete_file_from_vectorstore(project_id, ghost)
                ingest_logger.info(f"Reconciliation: Purged ghost document '{ghost}' from database for Project {project_id}")
    except Exception as e:
        ingest_logger.warning(f"Notice: reconcile_project_documents encountered an issue: {e}")

    if purged_ghosts:
        ingest_logger.info(f"Reconciled Project {project_id}: purged {len(purged_ghosts)} ghost documents: {purged_ghosts}")

    return {
        "active_disk_files": list(disk_files.values()),
        "purged_ghosts": purged_ghosts,
    }
