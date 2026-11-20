"""
Standalone CLI tool for generating embeddings and indexing chunks into PostgreSQL (pgvector).
Usage:
    python scripts/embed_document.py --file path/to/document.pdf --project-id <project_id>
    python scripts/embed_document.py --chunks-json path/to/chunks.json --project-id <project_id>
"""
import sys
import argparse
import time
import json
import hashlib
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_document
from core.chunker import chunk_elements
from core.vectorstore import index_chunks, get_live_vector_count
from db.session import is_db_connected


def main():
    parser = argparse.ArgumentParser(
        description="Retrio CLI: Compute vector embeddings and index directly into PostgreSQL (pgvector)."
    )
    parser.add_argument("--project-id", type=str, required=True, help="Target PostgreSQL project ID")
    parser.add_argument("--file", type=str, default=None, help="Document file to parse, chunk, and embed")
    parser.add_argument("--chunks-json", type=str, default=None, help="Pre-generated chunks JSON file")

    args = parser.parse_args()

    if not args.file and not args.chunks_json:
        print("Error: Must provide either --file or --chunks-json", file=sys.stderr)
        sys.exit(1)

    width = 75
    print("=" * width)
    print(" RETRIO VECTOR EMBEDDING CLI (PostgreSQL pgvector) ".center(width, "="))
    print("=" * width)

    # Verify PostgreSQL Connection
    if not is_db_connected():
        print("Error: Could not connect to PostgreSQL. Please ensure the database is running.", file=sys.stderr)
        sys.exit(1)

    print(f" Target Project ID: {args.project_id}")
    print(f" Database Engine:   PostgreSQL (pgvector)")
    print("-" * width)

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

        print(f"▶ Step 1: Parsing and chunking '{filename}'...")
        t0 = time.perf_counter()
        raw_elements, _ = parse_document(file_path)
        chunks, stats = chunk_elements(raw_elements, project_id=args.project_id)
        print(f"✔ Ready to embed {len(chunks)} chunks (created in {time.perf_counter() - t0:.2f}s)")
    else:
        json_path = Path(args.chunks_json).resolve()
        if not json_path.is_file():
            print(f"Error: File not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = data.get("chunks", data if isinstance(data, list) else [])
        filename = data.get("filename", json_path.stem)
        doc_hash = hashlib.sha256(json_path.read_bytes()).hexdigest()
        print(f"✔ Loaded {len(chunks)} chunks from JSON file '{json_path.name}'")

    if not chunks:
        print("Error: No chunks available to embed.", file=sys.stderr)
        sys.exit(1)

    print("-" * width)
    print("▶ Step 2: Generating OpenAI embeddings and indexing into PostgreSQL...")
    t_start = time.perf_counter()

    result = index_chunks(
        project_id=args.project_id,
        chunks=chunks,
        filename=filename,
        document_hash=doc_hash,
    )

    t_embed = time.perf_counter() - t_start
    total_in_db = get_live_vector_count(args.project_id)

    print("-" * width)
    print("✔ EMBEDDING & INDEXING COMPLETED SUCCESSFULLY")
    print(f"  * Document:           {filename}")
    print(f"  * Vectors Added:      {result.get('new_vectors_added')}")
    print(f"  * Embedding Model:    {result.get('embedding_model')}")
    print(f"  * Total in Project:   {total_in_db} vectors in PostgreSQL (pgvector)")
    print(f"  * Elapsed Time:       {t_embed:.2f}s")
    print("=" * width)


if __name__ == "__main__":
    main()
