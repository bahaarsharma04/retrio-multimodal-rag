"""Chroma storage with source-scoped IDs, embedding reuse and verified replacement."""
import hashlib
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from core.config import PROJECTS_DIR, DEFAULT_EMBEDDING_MODEL, OPENAI_API_KEY
from core.logger import embedding_logger


class DeterministicFallbackEmbeddings(Embeddings):
    """For OFFLINE TESTS only; not semantically meaningful, never for production retrieval."""
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
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if api_key and api_key.strip() and "your-openai" not in api_key:
        return OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL, api_key=api_key), DEFAULT_EMBEDDING_MODEL
    # Explicit opt-in so missing API keys cannot silently degrade RAG search.
    if os.getenv("RAG_ALLOW_FALLBACK_EMBEDDINGS") == "1":
        return DeterministicFallbackEmbeddings(), "offline-fallback-v1"
    raise RuntimeError("OpenAI API key missing; set RAG_ALLOW_FALLBACK_EMBEDDINGS=1 only for tests")


def get_project_storage_dir(project_id: str) -> Path:
    folder = PROJECTS_DIR / project_id / "chroma"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_project_vectorstore(project_id: str) -> Chroma:
    embedding, _ = get_embedding_function()
    return Chroma(
        collection_name=f"col_{project_id}",
        embedding_function=embedding,
        persist_directory=str(get_project_storage_dir(project_id)),
    )


def source_vector_ids(project_id: str, filename: str) -> List[str]:
    """Includes legacy rows, identified by filename (within this project's collection)."""
    store = get_project_vectorstore(project_id)
    return list(store._collection.get(where={"filename": filename}, include=[]).get("ids", []))


def source_ids_present(project_id: str, chunk_ids: List[str]) -> bool:
    if not chunk_ids:
        return False
    store = get_project_vectorstore(project_id)
    found = store._collection.get(ids=chunk_ids, include=[]).get("ids", [])
    return set(found) == set(chunk_ids)


def index_chunks(
    project_id: str,
    chunks: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    *,
    filename: Optional[str] = None,
    document_hash: str = "",
) -> Dict[str, Any]:
    """Prepare, upsert and verify new IDs, THEN remove obsolete IDs for this file.

    Reuses unchanged existing vectors if indexed_text and embedding_model match.
    A failed insertion never initiates old-vector deletion. Concurrent writes for
    the same filename need an application-level lock (see README).
    """
    if not chunks:
        raise ValueError("Refusing to replace document with zero chunks")
    filename = filename or chunks[0].get("filename", "unknown")
    if any(c.get("filename", filename) != filename for c in chunks):
        raise ValueError("index_chunks accepts chunks from one file at a time")

    t0 = time.perf_counter()
    store = get_project_vectorstore(project_id)
    _, model_name = get_embedding_function()
    collection = store._collection
    new_ids = [str(c["chunk_id"]) for c in chunks]
    if len(new_ids) != len(set(new_ids)):
        raise ValueError("Duplicate chunk ID within batch: fix chunker, do not suffix at storage time")

    embedding_logger.info(
        f"EMBED_START | Project: {project_id} | File: '{filename}' | Total Chunks: {len(chunks)} | Embedding Model: {model_name}"
    )

    old_ids = set(collection.get(where={"filename": filename}, include=[]).get("ids", []))
    existing = collection.get(ids=new_ids, include=["documents", "metadatas"])
    existing_map = {
        cid: (doc, meta or {})
        for cid, doc, meta in zip(existing["ids"], existing.get("documents", []), existing.get("metadatas", []))
    }
    insert_docs, insert_ids = [], []
    meta_updates, meta_update_ids = [], []
    reused = 0

    for index, chunk in enumerate(chunks):
        cid = str(chunk["chunk_id"])
        body = chunk.get("indexed_text") or chunk.get("content", "")
        if not body.strip():
            raise ValueError(f"Empty indexed text for {cid}")
        metadata = {
            "chunk_id": cid,
            "type": chunk.get("type", "text"),
            "filename": filename,
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

        snippet = body.replace("\n", " ").strip()
        if len(snippet) > 100:
            snippet = snippet[:100] + "..."

        previous = existing_map.get(cid)
        if previous and previous[0] == body and previous[1].get("embedding_model") == model_name:
            # No embedding API call needed; keep metadata/citations current.
            reused += 1
            meta_update_ids.append(cid)
            meta_updates.append(metadata)
            embedding_logger.info(
                f"STATUS: REUSED | Chunk [{index + 1}/{len(chunks)}] | ID: {cid} | Project: {project_id} | "
                f"Model: {model_name} | Chars: {len(body)} | Metadata: {metadata} | "
                f"Reason: Identical text & model match in ChromaDB (0 API tokens consumed)"
            )
        else:
            insert_ids.append(cid)
            insert_docs.append(Document(page_content=body, metadata=metadata))
            embedding_logger.info(
                f"STATUS: INDEXED | Chunk [{index + 1}/{len(chunks)}] | ID: {cid} | Project: {project_id} | "
                f"Model: {model_name} | Chars: {len(body)} | Page: {metadata['page_number']} | "
                f"Metadata: {metadata} | Data: \"{snippet}\""
            )

        if progress_callback:
            progress_callback(index + 1, len(chunks))

    # Stage all new embeddings first. If the call fails, old IDs remain intact.
    if insert_docs:
        store.add_documents(documents=insert_docs, ids=insert_ids)

    found = set(collection.get(ids=new_ids, include=[]).get("ids", []))
    if found != set(new_ids):
        raise RuntimeError("New vector version incomplete; refusing to delete old vectors")

    if meta_update_ids:
        collection.update(ids=meta_update_ids, metadatas=meta_updates)

    # Only after successful verification do we remove old/legacy rows.
    obsolete = sorted(old_ids - set(new_ids))
    if obsolete:
        collection.delete(ids=obsolete)
        for del_id in obsolete:
            embedding_logger.info(
                f"STATUS: DELETED | Chunk [Old ID: {del_id}] | Project: {project_id} | File: '{filename}' | Reason: Obsolete version replaced"
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    embedding_logger.info(
        f"EMBED_COMPLETE | Project: {project_id} | File: '{filename}' | Collection: col_{project_id} | "
        f"New Added: {len(insert_ids)} | Reused: {reused} | Deleted: {len(obsolete)} | "
        f"Total In Store: {collection.count()} | Elapsed: {elapsed_ms:.1f}ms"
    )

    return {
        "total_vectors_stored": collection.count(),
        "new_vectors_added": len(insert_ids),
        "reused_vectors": reused,
        "old_vectors_deleted": len(obsolete),
        "chunk_ids": new_ids,
        "collection_name": f"col_{project_id}",
        "storage_path": str(get_project_storage_dir(project_id)),
        "embedding_model": model_name,
    }


def get_all_chunks(project_id: str) -> List[Dict[str, Any]]:
    store = get_project_vectorstore(project_id)
    data = store.get(include=["documents", "metadatas"])
    chunks = []
    for cid, text, meta in zip(data.get("ids", []), data.get("documents", []), data.get("metadatas", [])):
        meta = meta or {}
        chunks.append({
            "chunk_id": cid,
            "type": meta.get("type", "text"),
            "filename": meta.get("filename", "document"),
            "page_number": meta.get("page_number", 1),
            "char_count": meta.get("char_count", len(text)),
            "content": meta.get("raw_content") or text,
            "summary": meta.get("summary") or None,
            "indexed_text": text,
        })
    return chunks


def delete_file_from_vectorstore(project_id: str, filename: str) -> bool:
    store = get_project_vectorstore(project_id)
    store._collection.delete(where={"filename": filename})
    embedding_logger.info(
        f"STATUS: DELETED_ALL | Project: {project_id} | File: '{filename}' | Purged all vector embeddings from ChromaDB"
    )
    return True
