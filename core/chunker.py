import hashlib
import time
import re
from typing import List, Dict, Any, Tuple
from core.config import (
    MAX_CHUNK_CHARACTERS,
    COMBINE_UNDER_N_CHARS,
    CHUNK_OVERLAP_CHARACTERS,
)
from core.logger import chunking_logger


def extract_sentence_overlap(text_parts: List[str], max_overlap_chars: int = CHUNK_OVERLAP_CHARACTERS) -> List[str]:
    """Extracts trailing sentence(s) to seed the next chunk, preventing boundary severing."""
    if not text_parts:
        return []
    last_text = text_parts[-1].strip()
    if len(last_text) <= max_overlap_chars:
        return [last_text]
    sentences = re.split(r"(?<=[.!?])\s+", last_text)
    if len(sentences) >= 2:
        candidate = " ".join(sentences[-2:])
        if len(candidate) <= max_overlap_chars + 120:
            return [candidate]
        return [sentences[-1]]
    return [last_text[-max_overlap_chars:]]


def split_large_text_by_sentences(
    text: str,
    max_chars: int = MAX_CHUNK_CHARACTERS,
    overlap_chars: int = CHUNK_OVERLAP_CHARACTERS,
) -> List[str]:
    """
    Subdivides an oversized atomic text block (e.g. unstructured documents without headings
    or large multi-thousand-character paragraphs) into coherent, sentence-bounded chunks
    with trailing sentence overlap.
    """
    clean_text = text.strip()
    if len(clean_text) <= max_chars:
        return [clean_text]

    # Split into individual sentences preserving punctuation
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    # If no sentence boundaries were detected (e.g. continuous text without punctuation)
    if len(sentences) <= 1:
        words = clean_text.split()
        chunks = []
        cur_words = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > max_chars and cur_words:
                chunks.append(" ".join(cur_words))
                overlap_words = []
                overlap_len = 0
                for ow in reversed(cur_words):
                    if overlap_len + len(ow) + 1 <= overlap_chars:
                        overlap_words.insert(0, ow)
                        overlap_len += len(ow) + 1
                    else:
                        break
                cur_words = list(overlap_words)
                cur_len = sum(len(x) + 1 for x in cur_words)
            cur_words.append(w)
            cur_len += len(w) + 1
        if cur_words:
            chunks.append(" ".join(cur_words))
        return chunks

    chunks = []
    current_sentences = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len + 1 > max_chars and current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks.append(chunk_str)

            # Trailing sentence overlap for the next sub-chunk
            overlap_sentences = []
            overlap_acc = 0
            for s in reversed(current_sentences):
                if overlap_acc + len(s) + 1 <= overlap_chars + 100:
                    overlap_sentences.insert(0, s)
                    overlap_acc += len(s) + 1
                else:
                    break

            if not overlap_sentences and current_sentences:
                overlap_sentences = [current_sentences[-1]]
                overlap_acc = len(overlap_sentences[0])

            current_sentences = list(overlap_sentences)
            current_len = sum(len(s) + 1 for s in current_sentences)

        current_sentences.append(sent)
        current_len += sent_len + 1

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def create_chunk_object(
    content: str,
    chunk_type: str,
    filename: str,
    page_numbers: List[int],
    merged_elements_count: int,
    extra_metadata: Dict[str, Any] = None,
    seen_hashes: Dict[str, int] = None,
    source_key: str = "",
) -> Dict[str, Any]:
    """Helper to construct a standardized chunk dictionary with complete metadata."""
    clean_text = content.strip()
    pages = sorted(list(set(page_numbers))) if page_numbers else [1]
    
    # Full-length SHA-256 prevents accidental collisions between unrelated texts.
    # Occurrence counter preserves repeated paragraphs as separate source locations.
    content_digest = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    scope = source_key or filename  # Pipeline supplies project-scoped identity.
    occurrence_key = f"{chunk_type}:{content_digest}"
    occurrence = seen_hashes.get(occurrence_key, 0) if seen_hashes is not None else 0
    if seen_hashes is not None:
        seen_hashes[occurrence_key] = occurrence + 1
    identity = f"{scope}:{chunk_type}:{content_digest}:{occurrence}"
    chunk_id = "chunk_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()

    section_name = (extra_metadata or {}).get("section", "General")

    chunk = {
        "chunk_id": chunk_id,
        "content_hash": content_digest,
        "type": chunk_type,  # 'text', 'table', or 'image'
        "content": clean_text,
        "char_count": len(clean_text),
        "page_number": pages[0] if pages else 1,
        "page_range": f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0]),
        "elements_merged": merged_elements_count,
        "filename": filename,
        "section": section_name,
        "summary": None,  # Will be populated in Phase 2 Part 3 (Summarisation)
        "metadata": extra_metadata or {},
    }
    return chunk


def chunk_elements(
    elements: List[Dict[str, Any]],
    max_chars: int = MAX_CHUNK_CHARACTERS,
    source_key: str = "",
    project_id: str = "",
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

    t0 = time.perf_counter()
    current_filename = elements[0].get("metadata", {}).get("filename", "document")
    chunking_logger.info(
        f"CHUNK_START | Project: {project_id or 'unknown'} | File: '{current_filename}' | Input Elements: {len(elements)}"
    )

    chunks: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, int] = {}
    
    # State tracking for the currently accumulated text chunk
    current_text_parts: List[str] = []
    current_pages: List[int] = []
    current_merged_count = 0
    current_section = "General"

    def flush_text_chunk(keep_overlap: bool = False):
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
                    extra_metadata={"section": current_section} if current_section else {},
                    seen_hashes=seen_hashes,
                    source_key=source_key,
                )
                chunks.append(chunk)

            if keep_overlap and current_text_parts:
                overlap_seed = extract_sentence_overlap(current_text_parts)
                current_text_parts = list(overlap_seed)
                current_pages = [current_pages[-1]] if current_pages else [1]
                current_merged_count = len(overlap_seed)
            else:
                current_text_parts = []
                current_pages = []
                current_merged_count = 0

    for el in elements:
        el_type = el.get("type", "text")
        el_text = el.get("text", "").strip()
        el_page = el.get("metadata", {}).get("page_number", 1) or 1
        el_filename = el.get("metadata", {}).get("filename", current_filename)

        # Case 1: Table - keep intact as dedicated chunk
        if el_type == "table":
            flush_text_chunk(keep_overlap=False)
            current_filename = el_filename
            table_html = el.get("metadata", {}).get("text_as_html")
            table_chunk = create_chunk_object(
                content=el_text,
                chunk_type="table",
                filename=el_filename,
                page_numbers=[el_page],
                merged_elements_count=1,
                extra_metadata={"text_as_html": table_html, "section": current_section},
                seen_hashes=seen_hashes,
                source_key=source_key,
            )
            chunks.append(table_chunk)
            continue

        # Case 2: Image - keep intact as dedicated chunk
        if el_type == "image":
            flush_text_chunk(keep_overlap=False)
            current_filename = el_filename
            image_chunk = create_chunk_object(
                content=el_text or "[Image element detected]",
                chunk_type="image",
                filename=el_filename,
                page_numbers=[el_page],
                merged_elements_count=1,
                extra_metadata={"section": current_section},
                seen_hashes=seen_hashes,
                source_key=source_key,
            )
            chunks.append(image_chunk)
            continue

        # Case 3: Title - natural section boundary (titles must be reasonably concise)
        if el_type == "title" and len(el_text) <= 300:
            clean_title = " ".join(el_text.split())
            base_fname = re.sub(r"\.[^.]+$", "", el_filename).lower() if el_filename else ""
            is_running_header = (
                bool(re.search(r"\b(all rights reserved|copyright|page\s+\d+(\s+of\s+\d+)?)\b", clean_title, re.I))
                or (len(base_fname) >= 4 and base_fname in clean_title.lower())
            )
            is_section_header = bool(re.match(
                r"^(\d+(\.\d+)*\s+[A-Z]|Appendix|References|Table\s+[A-Z0-9]+|Figure\s+[A-Z0-9]+|[A-Z][a-zA-Z\s]{3,40}$)",
                clean_title
            ))
            if is_section_header and not is_running_header and len(clean_title) < 100:
                current_section = clean_title

            current_buffer_len = sum(len(p) for p in current_text_parts)
            # Only start a new chunk on a title if current buffer already has meaningful content
            if current_buffer_len >= COMBINE_UNDER_N_CHARS:
                flush_text_chunk(keep_overlap=False)
            current_filename = el_filename
            current_text_parts.append(el_text)
            current_pages.append(el_page)
            current_merged_count += 1
            continue

        # Case 4: Text or Other elements
        if el_text:
            if current_text_parts and el_filename != current_filename:
                flush_text_chunk(keep_overlap=False)
            current_filename = el_filename

            # Handle unstructured / oversized text where a single atomic element exceeds max_chars
            if len(el_text) > max_chars:
                current_buffer_len = sum(len(p) for p in current_text_parts)
                if current_buffer_len > 0:
                    if current_buffer_len < COMBINE_UNDER_N_CHARS:
                        # Prepend small preceding text (e.g. section title or stub) to this text block
                        el_text = "\n\n".join(current_text_parts) + "\n\n" + el_text
                        current_text_parts = []
                        current_pages = []
                        current_merged_count = 0
                    else:
                        flush_text_chunk(keep_overlap=False)

                sub_parts = split_large_text_by_sentences(
                    el_text, max_chars=max_chars, overlap_chars=CHUNK_OVERLAP_CHARACTERS
                )

                for sp in sub_parts[:-1]:
                    current_text_parts = [sp]
                    current_pages = [el_page]
                    current_merged_count = 1
                    flush_text_chunk(keep_overlap=False)

                if sub_parts:
                    current_text_parts = [sub_parts[-1]]
                    current_pages = [el_page]
                    current_merged_count = 1
                continue

            projected_len = sum(len(p) for p in current_text_parts) + len(el_text) + 2
            if projected_len > max_chars and current_text_parts:
                flush_text_chunk(keep_overlap=True)

            current_text_parts.append(el_text)
            current_pages.append(el_page)
            current_merged_count += 1

    # Flush any remaining text elements
    flush_text_chunk(keep_overlap=False)

    # Calculate statistics for the Chunking UI tab
    total_atomic = len(elements)
    total_chunks = len(chunks)
    avg_size = int(sum(c["char_count"] for c in chunks) / total_chunks) if total_chunks > 0 else 0
    elapsed_ms = (time.perf_counter() - t0) * 1000

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

    # Log each individual chunk in the exact requested schema
    for idx, c in enumerate(chunks, 1):
        snippet = (c.get("content") or "").replace("\n", " ").strip()
        if len(snippet) > 100:
            snippet = snippet[:100] + "..."
        meta = {
            "filename": c.get("filename"),
            "page_range": c.get("page_range"),
            "elements_merged": c.get("elements_merged"),
            "content_hash": c.get("content_hash"),
        }
        chunking_logger.info(
            f"STATUS: CREATED | Chunk [{idx}/{total_chunks}] | ID: {c['chunk_id']} | "
            f"Project: {project_id or 'unknown'} | Type: {c.get('type')} | Chars: {c.get('char_count')} | "
            f"Metadata: {meta} | Data: \"{snippet}\""
        )

    chunking_logger.info(
        f"CHUNK_COMPLETE | Project: {project_id or 'unknown'} | File: '{current_filename}' | "
        f"Total Chunks: {total_chunks} | Avg Size: {avg_size} chars | Breakdown: {stats['chunk_type_breakdown']} | "
        f"Elapsed: {elapsed_ms:.1f}ms"
    )

    return chunks, stats
