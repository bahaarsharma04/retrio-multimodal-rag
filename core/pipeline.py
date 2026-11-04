import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from core.config import PROJECTS_DIR
from core.parser import parse_document
from core.chunker import chunk_elements
from core.summarizer import summarize_chunks
from core.vectorstore import index_chunks, get_all_chunks


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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
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
    filename = file_path.name

    if restart:
        delete_document_checkpoint(project_id, filename)
        checkpoint = None
    else:
        checkpoint = load_document_checkpoint(project_id, filename)

    cached_stages = checkpoint.get("stages", {}) if checkpoint else {}

    # ----------------- Stage 1: Partitioning -----------------
    if "partitioning" in cached_stages:
        if progress_callback:
            progress_callback("Partitioning (restored from checkpoint)", 1, 4)
        raw_elements = cached_stages["partitioning"]["raw_elements"]
        partition_stats = cached_stages["partitioning"]["stats"]
    else:
        if progress_callback:
            progress_callback("Partitioning", 1, 4)
        raw_elements, partition_stats = parse_document(file_path)
        save_document_checkpoint(project_id, filename, "partitioning", {
            "raw_elements": raw_elements,
            "stats": partition_stats,
        })

    # ----------------- Stage 2: Chunking -----------------
    if "chunking" in cached_stages:
        if progress_callback:
            progress_callback("Chunking (restored from checkpoint)", 2, 4)
        chunks = cached_stages["chunking"]["chunks"]
        chunk_stats = cached_stages["chunking"]["stats"]
    else:
        if progress_callback:
            progress_callback("Chunking", 2, 4)
        chunks, chunk_stats = chunk_elements(raw_elements)
        save_document_checkpoint(project_id, filename, "chunking", {
            "chunks": chunks,
            "stats": chunk_stats,
        })

    # ----------------- Stage 3: Summarisation -----------------
    if "summarisation" in cached_stages:
        if progress_callback:
            progress_callback("Summarisation (restored from checkpoint)", 3, 4)
        enriched_chunks = cached_stages["summarisation"]["enriched_chunks"]
        summarizer_stats = cached_stages["summarisation"]["stats"]
    else:
        if progress_callback:
            progress_callback("Summarisation", 3, 4)
        enriched_chunks, summarizer_stats = summarize_chunks(chunks)
        save_document_checkpoint(project_id, filename, "summarisation", {
            "enriched_chunks": enriched_chunks,
            "stats": summarizer_stats,
        })

    # ----------------- Stage 4: Vectorization & Storage -----------------
    if progress_callback:
        progress_callback("Vectorization & Storage", 4, 4)
    vector_stats = index_chunks(project_id, enriched_chunks)

    # ----------------- Stage 5: View Chunks & Finalization -----------------
    all_chunks_for_file = [c for c in enriched_chunks if c.get("filename") == filename]

    pipeline_result = {
        "filename": filename,
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
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_result, f, indent=2, ensure_ascii=False)

    # Clean up temporary checkpoint since ingestion completed successfully
    delete_document_checkpoint(project_id, filename)

    return pipeline_result


def delete_source_document(project_id: str, filename: str) -> bool:
    """
    Deletes a source document completely from a project:
    1. Removes the raw file from disk
    2. Removes the cached pipeline JSON
    3. Removes any pending checkpoint JSON
    4. Deletes all associated vector embeddings from ChromaDB
    """
    from core.vectorstore import delete_file_from_vectorstore

    # 1. Delete physical file
    docs_dir = get_project_docs_dir(project_id)
    doc_path = docs_dir / filename
    if doc_path.exists():
        try:
            doc_path.unlink()
        except Exception:
            pass

    # 2. Delete pipeline json
    pipe_path = get_document_record_path(project_id, filename)
    if pipe_path.exists():
        try:
            pipe_path.unlink()
        except Exception:
            pass

    # 3. Delete any intermediate checkpoint
    delete_document_checkpoint(project_id, filename)

    # 4. Delete from ChromaDB
    delete_file_from_vectorstore(project_id, filename)
    return True
