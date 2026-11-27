"""
Standalone CLI tool for generating embeddings and indexing chunks into PostgreSQL (pgvector).
Usage:
    python scripts/embed_document.py
    python scripts/embed_document.py --file path/to/document.pdf --project-id <project_id>
    python scripts/embed_document.py --chunks-json path/to/chunks.json --project-id <project_id>
"""
import sys
import argparse
import time
import json
import hashlib
from pathlib import Path
from typing import List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_document
from core.chunker import chunk_elements
from core.vectorstore import index_chunks, get_live_vector_count, dense_similarity_search
from core.retriever import count_tokens
from core.config import PROJECTS_DIR, DEFAULT_EMBEDDING_MODEL
from db.session import is_db_connected, get_db_session
from db.models import Project, DocumentChunk


def list_projects_for_embedding() -> List[Tuple[str, str, int]]:
    """Returns available projects from PostgreSQL with chunk counts."""
    try:
        with get_db_session() as s:
            projects = s.query(Project.id, Project.name).all()
            result = []
            for p in projects:
                count = s.query(DocumentChunk).filter(DocumentChunk.project_id == str(p.id)).count()
                result.append((str(p.id), p.name, count))
            return result
    except Exception:
        return []


def discover_available_documents() -> List[Path]:
    """Finds non-json document files in data/projects to allow auto-selection."""
    valid_extensions = {".pdf", ".docx", ".doc", ".csv", ".txt", ".md"}
    found = []
    if PROJECTS_DIR.exists():
        for doc_path in PROJECTS_DIR.glob("*/documents/*"):
            if doc_path.is_file() and doc_path.suffix.lower() in valid_extensions:
                found.append(doc_path)
    return sorted(found, key=lambda p: p.stat().st_size)


def main():
    parser = argparse.ArgumentParser(
        description="Retrio CLI: Compute vector embeddings and index directly into PostgreSQL (pgvector) with full telemetry."
    )
    parser.add_argument(
        "--project-id",
        "-p",
        type=str,
        default=None,
        help="Target PostgreSQL project ID (optional: auto-selects active project if omitted)",
    )
    parser.add_argument("--file", "-f", type=str, default=None, help="Document file to parse, chunk, and embed")
    parser.add_argument("--chunks-json", "-c", type=str, default=None, help="Pre-generated chunks JSON file")
    parser.add_argument("--preview", type=int, default=5, help="Number of chunks to preview before embedding (default: 5)")
    parser.add_argument("--skip-search-check", action="store_true", help="Skip the post-embedding vector search check")

    args = parser.parse_args()

    width = 85
    print("=" * width)
    print(" RETRIO VECTOR EMBEDDING & PGVECTOR INDEXING CLI ".center(width, "="))
    print("=" * width)

    # 1. Verify PostgreSQL Connection
    if not is_db_connected():
        print("Error: Could not connect to PostgreSQL. Please ensure the database is running.", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve Project ID
    project_id = args.project_id
    if not project_id:
        projects = list_projects_for_embedding()
        if not projects:
            print("Error: No projects found in PostgreSQL.", file=sys.stderr)
            sys.exit(1)
        # Choose the first project with chunks or first project
        project_id, proj_name, existing_chunks = projects[0]
        print(f"ℹ No --project-id provided. Auto-selected project:")
        print(f"  Name:   {proj_name}")
        print(f"  ID:     {project_id}")
        print(f"  Chunks in DB: {existing_chunks}\n")
        print("  Available projects in database:")
        for pid, pname, pchunks in projects[:5]:
            print(f"    • {pid} ({pname}) - {pchunks} chunks")
        print("-" * width)
    else:
        print(f" Target Project ID: {project_id}")

    # 3. Resolve Input Document / Chunks
    chunks = []
    filename = "unknown"
    doc_hash = ""

    if args.file:
        file_path = Path(args.file).resolve()
        if not file_path.is_file():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        filename = file_path.name
        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()

        print(f"\n[1/3] Parsing and chunking '{filename}' ({round(file_path.stat().st_size / 1024, 1)} KB)...")
        t0 = time.perf_counter()
        raw_elements, _ = parse_document(file_path)
        chunks, _ = chunk_elements(raw_elements, project_id=project_id)
        print(f"      ✔ Generated {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s")
    elif args.chunks_json:
        json_path = Path(args.chunks_json).resolve()
        if not json_path.is_file():
            print(f"Error: File not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", data if isinstance(data, list) else [])
        filename = data.get("filename", json_path.stem)
        doc_hash = hashlib.sha256(json_path.read_bytes()).hexdigest()
        print(f"\n[1/3] Loaded {len(chunks)} chunks from JSON file '{json_path.name}'")
    else:
        # Auto-discover a document
        docs = discover_available_documents()
        if not docs:
            print("Error: Must provide either --file or --chunks-json (no sample files discovered).", file=sys.stderr)
            sys.exit(1)
        target_doc = docs[0]
        filename = target_doc.name
        doc_hash = hashlib.sha256(target_doc.read_bytes()).hexdigest()
        print(f"\n[1/3] No input file specified. Auto-selected document from workspace:")
        print(f"      File: '{filename}' ({round(target_doc.stat().st_size / 1024, 1)} KB)")
        t0 = time.perf_counter()
        raw_elements, _ = parse_document(target_doc)
        chunks, _ = chunk_elements(raw_elements, project_id=project_id)
        print(f"      ✔ Parsed & generated {len(chunks)} chunks in {time.perf_counter() - t0:.2f}s")

    if not chunks:
        print("Error: No chunks available to embed.", file=sys.stderr)
        sys.exit(1)

    # Enrich and calculate tokens
    total_tokens = 0
    for idx, c in enumerate(chunks, 1):
        content = c.get("content") or ""
        tok = count_tokens(content)
        c["tokensize"] = tok
        c["first_100_chars"] = content.replace("\n", " ")[:100].strip()
        total_tokens += tok

    print("-" * width)
    print(f" Embedding Preparation Summary:")
    print(f"   • Document:          {filename}")
    print(f"   • Total Chunks:      {len(chunks)}")
    print(f"   • Total Tokens:      {total_tokens} tokens (cl100k_base)")
    print(f"   • Model:             {DEFAULT_EMBEDDING_MODEL} (1536 dimensions)")
    print(f"   • Storage Engine:    PostgreSQL (pgvector 'document_chunks' table)")
    print("-" * width)

    # Preview Chunks
    if args.preview > 0:
        preview_count = min(args.preview, len(chunks))
        print(f"\n[Preview] Inspecting first {preview_count} chunks to be embedded:")
        for i, c in enumerate(chunks[:preview_count], 1):
            cid = c.get("chunk_id", c.get("id", f"chunk_{i}"))[:18] + "..."
            print(f"  [{i}/{len(chunks)}] ID: {cid} | Type: {c.get('type', 'text')} | Tokens: {c['tokensize']}")
            print(f"       first_100_chars: \"{c['first_100_chars']}\"")
        print("-" * width)

    # Step 2: Index Chunks into PostgreSQL
    print("\n[2/3] Calling OpenAI Embeddings API & writing vectors to PostgreSQL...")
    t_start = time.perf_counter()

    result = index_chunks(
        project_id=project_id,
        chunks=chunks,
        filename=filename,
        document_hash=doc_hash,
    )

    t_embed = time.perf_counter() - t_start
    total_in_db = get_live_vector_count(project_id)

    print(f"✔ Embedding & PostgreSQL indexing completed in {t_embed:.2f}s:")
    print(f"   • Vectors Added/Updated:  {result.get('added', 0)}")
    print(f"   • Existing Reused:        {result.get('reused', 0)}")
    print(f"   • Total Chunks in DB:     {total_in_db}")

    # Step 3: Sanity Retrieval Test
    if not args.skip_search_check and chunks:
        sample_query = chunks[0]["first_100_chars"][:50].strip()
        print("\n[3/3] Sanity Vector Search Test (Verifying live pgvector query)...")
        t_search = time.perf_counter()
        search_results = dense_similarity_search(
            project_id=project_id,
            query=sample_query,
            top_k=3,
        )
        t_search_ms = (time.perf_counter() - t_search) * 1000.0
        print(f"✔ Executed test pgvector cosine search in {t_search_ms:.1f}ms:")
        print(f"   Query: \"{sample_query}\"")
        for rank, res in enumerate(search_results, 1):
            match_preview = (res.get("content") or "").replace("\n", " ")[:80]
            print(f"   [{rank}] Score: {res.get('score', 0):.4f} | \"{match_preview}...\"")

    print("\n" + "=" * width)
    print("✔ ALL EMBEDDING & PGVECTOR TESTS COMPLETED SUCCESSFULLY!")
    print("=" * width)


if __name__ == "__main__":
    main()
