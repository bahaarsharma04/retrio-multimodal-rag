"""
Vector store management with native PostgreSQL (pgvector) as the single source of truth.
"""
import hashlib
import os
import random
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from core.config import PROJECTS_DIR, DEFAULT_EMBEDDING_MODEL, OPENAI_API_KEY
from core.logger import embedding_logger
from db.session import is_db_connected, get_db_session
from db.models import Project as DBProject, Document as DBDocument, DocumentChunk as DBDocumentChunk
from sqlalchemy import func


class DeterministicFallbackEmbeddings(Embeddings):
    """For OFFLINE UNIT TESTS only; not semantically meaningful, never for production retrieval."""
    def __init__(self, size: int = 1536):
        self.size = size

    def _embed(self, value: str) -> List[float]:
        seed = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        return [rng.uniform(-1, 1) for _ in range(self.size)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embedding_function() -> Tuple[Embeddings, str]:
    """Returns the configured embeddings provider and its model name."""
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if api_key and api_key.strip() and "your-openai" not in api_key:
        return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, api_key=api_key), DEFAULT_EMBEDDING_MODEL
    # Explicit opt-in so missing API keys cannot silently degrade RAG search.
    if os.getenv("RAG_ALLOW_FALLBACK_EMBEDDINGS") == "1":
        return DeterministicFallbackEmbeddings(), "offline-fallback-v1"
    raise RuntimeError("OpenAI API key missing; set RAG_ALLOW_FALLBACK_EMBEDDINGS=1 only for tests")


def _require_db_connection():
    """Fail-fast check ensuring PostgreSQL is connected."""
    if not is_db_connected():
        raise RuntimeError("PostgreSQL database is disconnected. Please ensure PostgreSQL is running.")


def source_vector_ids(project_id: str, filename: str) -> List[str]:
    """Returns chunk IDs belonging to a specific document in PostgreSQL."""
    _require_db_connection()
    with get_db_session() as s:
        doc = s.query(DBDocument).filter(
            DBDocument.project_id == project_id,
            func.lower(DBDocument.filename) == filename.lower()
        ).first()
        if not doc:
            return []
        chunks = s.query(DBDocumentChunk.id).filter(DBDocumentChunk.document_id == doc.id).all()
        return [c[0] for c in chunks]


def source_ids_present(project_id: str, chunk_ids: List[str]) -> bool:
    """Checks whether all specified chunk IDs exist in the PostgreSQL project database."""
    if not chunk_ids:
        return False
    _require_db_connection()
    with get_db_session() as s:
        found = s.query(DBDocumentChunk.id).filter(
            DBDocumentChunk.project_id == project_id,
            DBDocumentChunk.id.in_(chunk_ids)
        ).all()
        found_set = {f[0] for f in found}
        return set(chunk_ids).issubset(found_set)


def index_chunks(
    project_id: str,
    chunks: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    *,
    filename: Optional[str] = None,
    document_hash: str = "",
) -> Dict[str, Any]:
    """
    Computes vector embeddings and indexes chunks directly into PostgreSQL (pgvector).
    """
    if not chunks:
        raise ValueError("Refusing to replace document with zero chunks")
    filename = filename or chunks[0].get("filename", "unknown")
    if any(c.get("filename", filename) != filename for c in chunks):
        raise ValueError("index_chunks accepts chunks from one file at a time")

    _require_db_connection()

    t0 = time.perf_counter()
    embedding_fn, model_name = get_embedding_function()
    new_ids = [str(c["chunk_id"]) for c in chunks]
    if len(new_ids) != len(set(new_ids)):
        raise ValueError("Duplicate chunk ID within batch: fix chunker, do not suffix at storage time")

    embedding_logger.info(
        f"EMBED_START | Project: {project_id} | File: '{filename}' | Total Chunks: {len(chunks)} | "
        f"Model: {model_name} | Primary Engine: PostgreSQL (pgvector)"
    )

    # 1. Prepare text bodies to embed
    texts_to_embed: List[str] = []
    chunk_metadata_list: List[Dict[str, Any]] = []

    for chunk in chunks:
        cid = str(chunk["chunk_id"])
        body = chunk.get("indexed_text") or chunk.get("content", "")
        if not body.strip():
            raise ValueError(f"Empty indexed text for {cid}")

        metadata = {
            "chunk_id": cid,
            "type": chunk.get("type", "text"),
            "filename": filename,
            "title": str(chunk.get("title") or chunk.get("metadata", {}).get("title") or chunk.get("section") or "General"),
            "section": str(chunk.get("section") or chunk.get("metadata", {}).get("section") or "General"),
            "part_index": int(chunk.get("part_index") or chunk.get("metadata", {}).get("part_index") or 1),
            "total_parts": int(chunk.get("total_parts") or chunk.get("metadata", {}).get("total_parts") or 1),
            "page_number": int(chunk.get("page_number", 1)),
            "page_range": str(chunk.get("page_range", "1")),
            "char_count": int(chunk.get("char_count", len(body))),
            "raw_content": chunk.get("content", ""),
            "summary": chunk.get("summary") or "",
            "project_id": str(project_id),
            "document_hash": document_hash,
            "embedding_model": model_name,
            "content_hash": chunk.get("content_hash") or hashlib.sha256(chunk.get("content", "").encode()).hexdigest(),
        }
        texts_to_embed.append(body)
        chunk_metadata_list.append(metadata)

    # 2. Generate embeddings
    raw_embeddings = embedding_fn.embed_documents(texts_to_embed)

    # 3. Store into PostgreSQL (pgvector)
    with get_db_session() as s:
        # Ensure Project exists
        proj = s.query(DBProject).filter(DBProject.id == project_id).first()
        if not proj:
            proj = DBProject(id=project_id, name="Default Project", description="Auto-created")
            s.add(proj)
            s.flush()

        # Ensure Document exists
        doc = s.query(DBDocument).filter(
            DBDocument.project_id == project_id,
            DBDocument.filename == filename
        ).first()
        if not doc:
            doc = DBDocument(
                id=str(uuid.uuid4()),
                project_id=project_id,
                filename=filename,
                file_path=str(filename),
                file_hash=document_hash,
                ingestion_status="completed",
            )
            s.add(doc)
            s.flush()
        else:
            doc.file_hash = document_hash
            doc.ingestion_status = "completed"

        # Delete obsolete chunks for this document
        s.query(DBDocumentChunk).filter(
            DBDocumentChunk.document_id == doc.id,
            ~DBDocumentChunk.id.in_(new_ids)
        ).delete(synchronize_session=False)

        # Upsert all chunks with embeddings
        for idx, (chunk, meta, emb) in enumerate(zip(chunks, chunk_metadata_list, raw_embeddings)):
            cid = meta["chunk_id"]
            db_chunk = s.query(DBDocumentChunk).filter(DBDocumentChunk.id == cid).first()
            p_raw = chunk.get("page_range") or chunk.get("page_number") or 1
            if isinstance(p_raw, list):
                pages = [int(x) for x in p_raw if str(x).isdigit()] or [1]
            elif "-" in str(p_raw):
                pages = [int(p) for p in str(p_raw).split("-") if p.isdigit()] or [1]
            else:
                pages = [int(p_raw)] if str(p_raw).isdigit() else [1]

            if not db_chunk:
                db_chunk = DBDocumentChunk(
                    id=cid,
                    document_id=doc.id,
                    project_id=project_id,
                    chunk_index=idx + 1,
                    chunk_type=meta["type"],
                    title=meta["title"],
                    section=meta["section"],
                    part_index=meta["part_index"],
                    total_parts=meta["total_parts"],
                    page_numbers=pages,
                    content=chunk.get("content", ""),
                    text_as_html=chunk.get("text_as_html"),
                    summary=chunk.get("summary"),
                    indexed_text=texts_to_embed[idx],
                    content_hash=meta["content_hash"],
                    char_count=meta["char_count"],
                    embedding=emb,
                )
                s.add(db_chunk)
            else:
                db_chunk.title = meta["title"]
                db_chunk.section = meta["section"]
                db_chunk.part_index = meta["part_index"]
                db_chunk.total_parts = meta["total_parts"]
                db_chunk.page_numbers = pages
                db_chunk.content = chunk.get("content", "")
                db_chunk.text_as_html = chunk.get("text_as_html")
                db_chunk.summary = chunk.get("summary")
                db_chunk.indexed_text = texts_to_embed[idx]
                db_chunk.content_hash = meta["content_hash"]
                db_chunk.char_count = meta["char_count"]
                db_chunk.embedding = emb

            if progress_callback:
                progress_callback(idx + 1, len(chunks))

        s.commit()
        total_pg_vectors = s.query(DBDocumentChunk).filter(DBDocumentChunk.project_id == project_id).count()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    embedding_logger.info(
        f"EMBED_COMPLETE | Project: {project_id} | File: '{filename}' | Primary Engine: PostgreSQL (pgvector) | "
        f"Stored Vectors: {len(chunks)} | Total In Database: {total_pg_vectors} | Elapsed: {elapsed_ms:.1f}ms"
    )

    return {
        "total_vectors_stored": total_pg_vectors,
        "new_vectors_added": len(chunks),
        "reused_vectors": 0,
        "old_vectors_deleted": 0,
        "chunk_ids": new_ids,
        "collection_name": f"project_{project_id}",
        "database_engine": "PostgreSQL (pgvector)",
        "storage_path": "PostgreSQL database (retrio_db.document_chunks)",
        "embedding_model": model_name,
    }


def dense_similarity_search(
    project_id: str,
    query: str,
    k: int = 4,
    filename_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Performs semantic vector similarity search strictly using native PostgreSQL pgvector cosine distance.
    Fails fast if PostgreSQL is offline.
    """
    _require_db_connection()

    embedding_fn, _ = get_embedding_function()
    t0 = time.perf_counter()

    query_vector = embedding_fn.embed_query(query)
    with get_db_session() as s:
        distance_col = DBDocumentChunk.embedding.cosine_distance(query_vector).label("distance")
        q = (
            s.query(DBDocumentChunk, DBDocument.filename, distance_col)
            .join(DBDocument, DBDocumentChunk.document_id == DBDocument.id)
            .filter(
                DBDocumentChunk.project_id == project_id,
                DBDocumentChunk.embedding.isnot(None),
            )
        )
        if filename_filter:
            q = q.filter(func.lower(DBDocument.filename) == filename_filter.lower())

        results = q.order_by(distance_col.asc()).limit(k).all()

        hits = []
        for chunk, fname, dist in results:
            content = chunk.content or ""
            summary = chunk.summary or None
            if content.strip().startswith(("[Embedded Figure:", "[Embedded Visual")) and not summary:
                continue

            dist_val = float(dist) if dist is not None else 0.0
            similarity_score = max(0.0, 1.0 - dist_val)
            hits.append({
                "chunk_id": chunk.id,
                "content": content,
                "filename": fname or chunk.document_id,
                "page_number": chunk.page_numbers[0] if (chunk.page_numbers and isinstance(chunk.page_numbers, list)) else 1,
                "type": chunk.chunk_type or "text",
                "title": chunk.title or "General",
                "section": chunk.section or "General",
                "summary": summary,
                "score": dist_val,
                "similarity": similarity_score,
            })
        elapsed_ms = (time.perf_counter() - t0) * 1000
        top_score = f"{hits[0]['score']:.4f}" if hits else "None"
        embedding_logger.info(
            f"Dense search via PostgreSQL (pgvector): retrieved {len(hits)} chunks in {elapsed_ms:.1f}ms | "
            f"Filter: {filename_filter or 'none'} | Top distance: {top_score}"
        )
        return hits


def get_all_chunks(project_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves all indexed chunks and metadata for a project directly from PostgreSQL.
    """
    _require_db_connection()
    with get_db_session() as s:
        rows = (
            s.query(DBDocumentChunk, DBDocument.filename)
            .join(DBDocument, DBDocumentChunk.document_id == DBDocument.id)
            .filter(DBDocumentChunk.project_id == project_id)
            .order_by(DBDocumentChunk.chunk_index.asc())
            .all()
        )
        chunks = []
        for chunk, fname in rows:
            chunks.append({
                "chunk_id": chunk.id,
                "type": chunk.chunk_type or "text",
                "filename": fname or "document",
                "title": chunk.title or "General",
                "section": chunk.section or "General",
                "part_index": chunk.part_index or 1,
                "total_parts": chunk.total_parts or 1,
                "page_number": chunk.page_numbers[0] if (chunk.page_numbers and isinstance(chunk.page_numbers, list)) else 1,
                "char_count": chunk.char_count or len(chunk.content or ""),
                "content": chunk.content or "",
                "summary": chunk.summary or None,
                "indexed_text": chunk.indexed_text or chunk.content or "",
            })
        return chunks


def get_live_vector_count(project_id: str) -> int:
    """Returns the total number of indexed vectors for a project directly from PostgreSQL."""
    _require_db_connection()
    with get_db_session() as s:
        return s.query(DBDocumentChunk).filter(
            DBDocumentChunk.project_id == project_id,
            DBDocumentChunk.embedding.isnot(None),
        ).count()


def delete_file_from_vectorstore(project_id: str, filename: str) -> bool:
    """Removes all vector embeddings belonging to a specific file from PostgreSQL."""
    _require_db_connection()
    with get_db_session() as s:
        docs = s.query(DBDocument).filter(
            DBDocument.project_id == project_id,
            func.lower(DBDocument.filename) == filename.lower()
        ).all()
        for doc in docs:
            s.delete(doc)  # Cascades to document_chunks and pipeline_runs

        # Also purge any loose chunks in this project matching the document's filename
        doc_ids_subquery = s.query(DBDocument.id).filter(
            DBDocument.project_id == project_id,
            func.lower(DBDocument.filename) == filename.lower()
        ).subquery()
        s.query(DBDocumentChunk).filter(
            DBDocumentChunk.project_id == project_id,
            DBDocumentChunk.document_id.in_(doc_ids_subquery)
        ).delete(synchronize_session=False)

        s.commit()
        embedding_logger.info(
            f"STATUS: DELETED_ALL | Project: {project_id} | File: '{filename}' | Purged from PostgreSQL (pgvector)"
        )

    return True
