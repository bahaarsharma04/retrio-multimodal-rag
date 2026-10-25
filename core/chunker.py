import uuid
from typing import List, Dict, Any, Tuple
from core.config import (
    MAX_CHUNK_CHARACTERS,
    COMBINE_UNDER_N_CHARS,
)


def create_chunk_object(
    content: str,
    chunk_type: str,
    filename: str,
    page_numbers: List[int],
    merged_elements_count: int,
    extra_metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Helper to construct a standardized chunk dictionary with complete metadata."""
    clean_text = content.strip()
    pages = sorted(list(set(page_numbers))) if page_numbers else [1]
    
    chunk = {
        "chunk_id": f"chunk_{uuid.uuid4().hex[:8]}",
        "type": chunk_type,  # 'text', 'table', or 'image'
        "content": clean_text,
        "char_count": len(clean_text),
        "page_number": pages[0] if pages else 1,
        "page_range": f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]),
        "elements_merged": merged_elements_count,
        "filename": filename,
        "summary": None,  # Will be populated in Phase 2 Part 3 (Summarisation)
        "metadata": extra_metadata or {},
    }
    return chunk


def chunk_elements(
    elements: List[Dict[str, Any]],
    max_chars: int = MAX_CHUNK_CHARACTERS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Performs layout-aware semantic chunking on parsed document elements.
    
    Rules:
    1. Titles act as semantic boundaries: starts a new chunk when a title is detected.
    2. Tables are preserved intact as dedicated 'table' chunks.
    3. Images/figures are preserved intact as dedicated 'image' chunks.
    4. Text sections under the same heading are combined up to max_chars.
    
    Returns:
        chunks: List of structured chunks with metadata.
        stats: Aggregated metrics for the 'Chunking' UI tab.
    """
    if not elements:
        return [], {
            "atomic_elements_count": 0,
            "chunks_created": 0,
            "avg_chunk_size": 0,
            "chunk_type_breakdown": {},
        }

    chunks: List[Dict[str, Any]] = []
    
    # State tracking for the currently accumulated text chunk
    current_text_parts: List[str] = []
    current_pages: List[int] = []
    current_merged_count = 0
    current_filename = elements[0]["metadata"].get("filename", "document")

    def flush_text_chunk():
        nonlocal current_text_parts, current_pages, current_merged_count
        if current_text_parts:
            combined_content = "\n\n".join(current_text_parts).strip()
            if combined_content:
                chunk = create_chunk_object(
                    content=combined_content,
                    chunk_type="text",
                    filename=current_filename,
                    page_numbers=current_pages,
                    merged_elements_count=current_merged_count,
                )
                chunks.append(chunk)
            current_text_parts = []
            current_pages = []
            current_merged_count = 0

    for el in elements:
        el_type = el.get("type", "text")
        el_text = el.get("text", "").strip()
        el_page = el.get("metadata", {}).get("page_number", 1) or 1
        el_filename = el.get("metadata", {}).get("filename", current_filename)
        current_filename = el_filename

        # Case 1: Table - keep intact as dedicated chunk
        if el_type == "table":
            flush_text_chunk()
            table_html = el.get("metadata", {}).get("text_as_html")
            table_chunk = create_chunk_object(
                content=el_text,
                chunk_type="table",
                filename=el_filename,
                page_numbers=[el_page],
                merged_elements_count=1,
                extra_metadata={"text_as_html": table_html},
            )
            chunks.append(table_chunk)
            continue

        # Case 2: Image - keep intact as dedicated chunk
        if el_type == "image":
            flush_text_chunk()
            image_chunk = create_chunk_object(
                content=el_text or "[Image element detected]",
                chunk_type="image",
                filename=el_filename,
                page_numbers=[el_page],
                merged_elements_count=1,
            )
            chunks.append(image_chunk)
            continue

        # Case 3: Title - natural section boundary
        if el_type == "title":
            # If current buffer already has meaningful content, flush it
            if current_text_parts:
                flush_text_chunk()
            current_text_parts.append(el_text)
            current_pages.append(el_page)
            current_merged_count += 1
            continue

        # Case 4: Text or Other elements
        if el_text:
            projected_len = sum(len(p) for p in current_text_parts) + len(el_text) + 2
            if projected_len > max_chars and current_text_parts:
                flush_text_chunk()

            current_text_parts.append(el_text)
            current_pages.append(el_page)
            current_merged_count += 1

    # Flush any remaining text elements
    flush_text_chunk()

    # Calculate statistics for the Chunking UI tab
    total_atomic = len(elements)
    total_chunks = len(chunks)
    avg_size = int(sum(c["char_count"] for c in chunks) / total_chunks) if total_chunks > 0 else 0

    stats = {
        "atomic_elements_count": total_atomic,
        "chunks_created": total_chunks,
        "avg_chunk_size": avg_size,
        "chunk_type_breakdown": {
            "text": sum(1 for c in chunks if c["type"] == "text"),
            "table": sum(1 for c in chunks if c["type"] == "table"),
            "image": sum(1 for c in chunks if c["type"] == "image"),
        },
    }

    return chunks, stats
