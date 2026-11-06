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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_ingestion_pipeline(
    project_id: str,
    file_path: Path,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    restart: bool = False,
) -> Dict[str, Any]:
    """
    Executes the complete 5-stage ingestion pipeline with checkpoint recovery:
    1. Partitioning (Text, Tables, Images, Titles)
    2. Chunking (Semantic chunking by title)
    3. Summarisation (AI summaries for tables/images)
    4. Vectorization & Storage (OpenAI Embeddings + ChromaDB)
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

    return pipeline_result


def delete_source_document(project_id: str, filename: str) -> bool:
    """Delete vector data first; only remove files after Chroma confirms deletion."""
    from core.vectorstore import delete_file_from_vectorstore
    delete_file_from_vectorstore(project_id, filename)
    documents = get_project_docs_dir(project_id)
    for path in (
        documents / filename,
        get_document_record_path(project_id, filename),
        get_document_checkpoint_path(project_id, filename),
    ):
        if path.exists():
            path.unlink()
    ingest_logger.info(f"Source document '{filename}' and associated vectors deleted from Project: {project_id}")
    return True
