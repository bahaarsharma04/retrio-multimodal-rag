import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

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
) -> Dict[str, Any]:
    """
    Executes the complete 5-stage ingestion pipeline for an uploaded file:
    1. Partitioning (Text, Tables, Images, Titles)
    2. Chunking (Semantic chunking by title)
    3. Summarisation (AI summaries for tables/images)
    4. Vectorization & Storage (OpenAI Embeddings + ChromaDB)
    5. View Chunks (Formatted chunk list with metadata)

    Returns:
        A unified telemetry dictionary containing the data for each tab.
    """
    filename = file_path.name

    # ----------------- Stage 1: Partitioning -----------------
    if progress_callback:
        progress_callback("Partitioning", 1, 4)
    raw_elements, partition_stats = parse_document(file_path)

    # ----------------- Stage 2: Chunking -----------------
    if progress_callback:
        progress_callback("Chunking", 2, 4)
    chunks, chunk_stats = chunk_elements(raw_elements)

    # ----------------- Stage 3: Summarisation -----------------
    if progress_callback:
        progress_callback("Summarisation", 3, 4)
    enriched_chunks, summarizer_stats = summarize_chunks(chunks)

    # ----------------- Stage 4: Vectorization & Storage -----------------
    if progress_callback:
        progress_callback("Vectorization & Storage", 4, 4)
    vector_stats = index_chunks(project_id, enriched_chunks)

    # ----------------- Stage 5: View Chunks -----------------
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

    # Persist the pipeline run metadata for quick inspection
    record_path = get_document_record_path(project_id, filename)
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_result, f, indent=2, ensure_ascii=False)

    return pipeline_result


def delete_source_document(project_id: str, filename: str) -> bool:
    """
    Deletes a source document completely from a project:
    1. Removes the raw file from disk
    2. Removes the cached pipeline JSON
    3. Deletes all associated vector embeddings from ChromaDB
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

    # 3. Delete from ChromaDB
    delete_file_from_vectorstore(project_id, filename)
    return True
