"""
Standalone CLI tool for Document Parsing and Semantic Title Chunking.
Usage:
    python scripts/chunk_document.py
    python scripts/chunk_document.py path/to/document.pdf
    python scripts/chunk_document.py path/to/document.pdf --max-chars 1200 --overlap 200 --output-json chunks.json
"""
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_document
from core.chunker import chunk_elements
from core.retriever import count_tokens
from core.config import PROJECTS_DIR


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
        description="Retrio CLI: Parse a document and generate semantic title-based chunks with full telemetry."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        type=str,
        default=None,
        help="Path to the document file (PDF, DOCX, CSV, TXT, MD). Optional: auto-selects if omitted.",
    )
    parser.add_argument("--project-id", "-p", type=str, default="cli_project", help="Scoping project ID")
    parser.add_argument("--max-chars", type=int, default=1500, help="Max chunk character size (default: 1500)")
    parser.add_argument("--overlap", type=int, default=250, help="Overlap character size (default: 250)")
    parser.add_argument("--output-json", "-o", type=str, default=None, help="Save generated chunks to a JSON file")
    parser.add_argument("--preview", type=int, default=10, help="Number of chunk previews to display (default: 10, -1 for all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full text of each chunk instead of first 100 characters")

    args = parser.parse_args()

    width = 85
    print("=" * width)
    print(" RETRIO DOCUMENT CHUNKING & TELEMETRY CLI ".center(width, "="))
    print("=" * width)

    # 1. Resolve Document File
    target_path = None
    if args.file_path:
        p = Path(args.file_path).resolve()
        if p.is_file():
            target_path = p
        else:
            print(f"Error: File not found: {p}", file=sys.stderr)
            sys.exit(1)
    else:
        docs = discover_available_documents()
        if not docs:
            print("Error: No document provided and no sample documents found in data/projects.", file=sys.stderr)
            print("Please provide a file: python scripts/chunk_document.py <path_to_file>", file=sys.stderr)
            sys.exit(1)
        target_path = docs[0]
        print(f"ℹ No file specified. Auto-selected available document:")
        print(f"  Path: {target_path}")
        print(f"  Size: {round(target_path.stat().st_size / 1024, 2)} KB\n")
        print("  Available documents discovered in workspace:")
        for d in docs[:6]:
            print(f"    • {d.name} ({round(d.stat().st_size / 1024, 1)} KB) in project {d.parent.parent.name[:8]}...")
        print("-" * width)

    print(f" Target File:    {target_path.name}")
    print(f" Absolute Path:  {target_path}")
    print(f" File Size:      {round(target_path.stat().st_size / 1024, 2)} KB ({target_path.stat().st_size} bytes)")
    print(f" Project ID:     {args.project_id}")
    print(f" Chunker Config: Max Chars: {args.max_chars} | Overlap: {args.overlap}")
    print("-" * width)

    # Stage 1: Partitioning
    t0 = time.perf_counter()
    print("\n[Stage 1/2] Parsing layout elements via Unstructured / PDF engine...")
    raw_elements, partition_stats = parse_document(target_path)
    t_parse = time.perf_counter() - t0
    print(f"✔ Parsed {len(raw_elements)} layout elements in {t_parse:.2f}s:")
    print(f"   • Text Sections:      {partition_stats.get('text_sections', 0)}")
    print(f"   • Tables:             {partition_stats.get('tables', 0)}")
    print(f"   • Images/Figures:     {partition_stats.get('images', 0)}")
    print(f"   • Titles/Headings:    {partition_stats.get('titles_headers', 0)}")
    print(f"   • Other Elements:     {partition_stats.get('other_elements', 0)}")
    print("-" * width)

    # Stage 2: Chunking
    t1 = time.perf_counter()
    print("\n[Stage 2/2] Running semantic title-aware chunking & metadata propagation...")
    chunks, chunk_stats = chunk_elements(
        raw_elements,
        max_chars=args.max_chars,
        project_id=args.project_id,
    )
    t_chunk = time.perf_counter() - t1

    # Enrich chunks with token sizes and first_100_chars
    total_tokens = 0
    total_chars = 0
    for c in chunks:
        content = c.get("content") or ""
        tok = count_tokens(content)
        c["tokensize"] = tok
        c["char_count"] = len(content)
        c["first_100_chars"] = content.replace("\n", " ")[:100].strip()
        total_tokens += tok
        total_chars += len(content)

    avg_tokens = round(total_tokens / len(chunks), 1) if chunks else 0
    avg_chars = round(total_chars / len(chunks), 1) if chunks else 0

    print(f"✔ Created {len(chunks)} chunks in {t_chunk:.3f}s:")
    print(f"   • Total Tokens:       {total_tokens} tokens (cl100k_base)")
    print(f"   • Average Tokensize:  {avg_tokens} tokens/chunk")
    print(f"   • Total Characters:   {total_chars} characters")
    print(f"   • Average Characters: {avg_chars} chars/chunk")
    print(f"   • Chunk Breakdown:    {chunk_stats.get('chunk_type_breakdown', {})}")
    print("=" * width)

    # Detailed Chunk Inspection
    if chunks:
        limit = len(chunks) if args.preview < 0 else min(args.preview, len(chunks))
        print(f"\n--- DETAILED CHUNK INSPECTION (Showing {limit} of {len(chunks)} chunks) ---\n")
        for i, c in enumerate(chunks[:limit], 1):
            cid = c.get("chunk_id", c.get("id", f"chunk_{i}"))
            ctype = c.get("type", "text").upper()
            title = c.get("title") or c.get("section") or "General"
            page = c.get("page_number") or c.get("page_range") or 1
            tok = c["tokensize"]
            chars = c["char_count"]
            first_100 = c["first_100_chars"]

            print(f"[{i:02d}/{len(chunks):02d}] [{ctype}] ID: {cid[:20]}... | Page: {page} | Section: \"{title}\"")
            print(f"     Metrics: tokensize={tok} tokens | length={chars} chars")
            print(f"     first_100_chars: \"{first_100}\"")
            if args.verbose:
                raw_full = (c.get("content") or "").strip()
                print(f"     FULL CONTENT:\n     " + "\n     ".join(raw_full.splitlines()))
            print("-" * width)

    # Save to JSON if requested
    if args.output_json:
        out_p = Path(args.output_json).resolve()
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "filename": target_path.name,
                "project_id": args.project_id,
                "total_chunks": len(chunks),
                "total_tokens": total_tokens,
                "avg_tokensize": avg_tokens,
                "partition_stats": partition_stats,
                "chunk_stats": chunk_stats,
                "chunks": chunks,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✔ Full chunks and telemetry saved to: {out_p}")

    print(f"\n✔ Chunking CLI execution finished successfully in {t_parse + t_chunk:.2f}s total.")


if __name__ == "__main__":
    main()
