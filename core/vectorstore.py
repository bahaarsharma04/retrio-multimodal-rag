import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from core.config import (
    PROJECTS_DIR,
    DEFAULT_EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


class DeterministicFallbackEmbeddings(Embeddings):
    """
    Lightweight deterministic pseudo-embeddings for offline testing
    when no OpenAI API key is configured. Ensures the app never crashes.
    """
    def __init__(self, size: int = 1536):
        self.size = size

    def _embed(self, text: str) -> List[float]:
        # Hash text to a pseudo-random reproducible vector of floats
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        import random
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.size)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embedding_function() -> Tuple[Embeddings, str]:
    """
    Initializes OpenAI embeddings if key is present,
    otherwise returns the fallback embeddings.
    """
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if api_key and api_key.strip() and "your-openai" not in api_key:
        try:
            return OpenAIEmbeddings(
                model=DEFAULT_EMBEDDING_MODEL,
                api_key=api_key,
            ), DEFAULT_EMBEDDING_MODEL
        except Exception:
            pass
    
    return DeterministicFallbackEmbeddings(size=1536), "Fallback Offline Embeddings"


def get_project_storage_dir(project_id: str) -> Path:
    """Returns the persistent storage directory for a specific project."""
    project_dir = PROJECTS_DIR / project_id / "chroma"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def get_project_vectorstore(project_id: str) -> Chroma:
    """
    Retrieves or creates a persistent Chroma vector store for a project.
    Each project is strictly isolated in its own collection and directory.
    """
    persist_dir = get_project_storage_dir(project_id)
    embeddings, _ = get_embedding_function()
    collection_name = f"col_{project_id}"

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def index_chunks(
    project_id: str,
    chunks: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Embeds semantic chunks and persists them into ChromaDB.
    
    Args:
        project_id: The project ID namespace.
        chunks: List of enriched chunks (with indexed_text, summary, metadata).
        progress_callback: Optional UI callback(current, total).
        
    Returns:
        stats: Vectorization and storage telemetry for the UI tab.
    """
    if not chunks:
        return {
            "total_vectors_stored": 0,
            "new_vectors_added": 0,
            "collection_name": f"col_{project_id}",
            "storage_path": str(get_project_storage_dir(project_id)),
            "embedding_model": "None",
        }

    vectorstore = get_project_vectorstore(project_id)
    _, model_name = get_embedding_function()

    documents: List[Document] = []
    ids: List[str] = []

    for idx, c in enumerate(chunks):
        # We index the enriched text (summary + raw content) for optimal search
        content_to_index = c.get("indexed_text") or c.get("content", "")
        
        doc = Document(
            page_content=content_to_index,
            metadata={
                "chunk_id": c.get("chunk_id", f"c_{idx}"),
                "type": c.get("type", "text"),
                "filename": c.get("filename", "unknown"),
                "page_number": c.get("page_number", 1),
                "char_count": c.get("char_count", len(content_to_index)),
                "raw_content": c.get("content", ""),
                "summary": c.get("summary") or "",
                "project_id": project_id,
            }
        )
        documents.append(doc)
        ids.append(c.get("chunk_id", f"c_{idx}"))

        if progress_callback:
            progress_callback(idx + 1, len(chunks))

    # Add to Chroma vector database
    vectorstore.add_documents(documents=documents, ids=ids)

    # Fetch total count of vectors in the collection
    total_count = vectorstore._collection.count()

    stats = {
        "total_vectors_stored": total_count,
        "new_vectors_added": len(documents),
        "collection_name": f"col_{project_id}",
        "storage_path": str(get_project_storage_dir(project_id)),
        "embedding_model": model_name,
    }

    return stats


def get_all_chunks(project_id: str) -> List[Dict[str, Any]]:
    """
    Fetches all indexed chunks from ChromaDB for the 'View Chunks' screen.
    """
    vectorstore = get_project_vectorstore(project_id)
    try:
        data = vectorstore.get()
        if not data or not data.get("documents"):
            return []

        chunks = []
        for i, text in enumerate(data["documents"]):
            meta = data["metadatas"][i] if data.get("metadatas") else {}
            chunk_id = data["ids"][i] if data.get("ids") else f"c_{i}"

            chunks.append({
                "chunk_id": chunk_id,
                "type": meta.get("type", "text"),
                "filename": meta.get("filename", "document"),
                "page_number": meta.get("page_number", 1),
                "char_count": meta.get("char_count", len(text)),
                "content": meta.get("raw_content") or text,
                "summary": meta.get("summary") or None,
                "indexed_text": text,
            })
        return chunks
    except Exception:
        return []


def delete_file_from_vectorstore(project_id: str, filename: str) -> bool:
    """Removes all vector embeddings and chunks associated with a specific file."""
    vectorstore = get_project_vectorstore(project_id)
    try:
        vectorstore._collection.delete(where={"filename": filename})
        return True
    except Exception:
        return False
