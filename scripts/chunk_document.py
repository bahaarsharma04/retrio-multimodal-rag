"""
Standalone CLI tool for Document Parsing and Semantic Title Chunking.
Usage:
    python scripts/chunk_document.py path/to/document.pdf
    python scripts/chunk_document.py path/to/document.pdf --max-chars 1200 --overlap 200 --output-json chunks.json
"""
import sys
import argparse
import time
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_document
from core.chunker import chunk_elements


def main():
    parser = argparse.ArgumentParser(
        description="Retrio CLI: Parse a document and generate semantic title-based chunks."
    )
    parser.add_argument("file_path", type=str, help="Path to the document file (PDF, DOCX, TXT, MD)")
    parser.add_argument("--project-id", type=str, default="cli_project", help="Scoping project ID")
    parser.add_argument("--max-chars", type=int, default=1500, help="Max chunk character size (default: 1500)")
    parser.add_argument("--overlap", type=int, default=250, help="Overlap character size (default: 250)")
    parser.add_argument("--output-json", type=str, default=None, help="Save generated chunks to a JSON file")
    parser.add_argument("--preview", type=int, default=5, help="Number of chunk previews to display (default: 5)")

    args = parser.parse_args()

    file_path = Path(args.file_path).resolve()
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    width = 75
    print("=" * width)
    print(" RETRIO DOCUMENT CHUNKING CLI ".center(width, "="))
    print("=" * width)
    print(f" Target File:  {file_path.name} ({round(file_path.stat().st_size / 1024, 1)} KB)")
    print(f" Project ID:   {args.project_id}")
    print(f" Max Chars:    {args.max_chars} | Overlap: {args.overlap}")
    print("-" * width)

    # Stage 1: Partitioning
    t0 = time.perf_counter()
    print("▶ Running Stage 1: Parsing layout elements...")
    raw_elements, partition_stats = parse_document(file_path)
    t_parse = time.perf_counter() - t0
    print(f"✔ Parsed {len(raw_elements)} elements in {t_parse:.2f}s:")
    print(f"   - Text Sections:     {partition_stats.get('text_sections', 0)}")
    print(f"   - Tables:            {partition_stats.get('tables', 0)}")
    print(f"   - Images/Figures:    {partition_stats.get('images', 0)}")
    print(f"   - Titles/Headings:   {partition_stats.get('titles_headers', 0)}")
    print("-" * width)

    # Stage 2: Chunking
    t1 = time.perf_counter()
    print("▶ Running Stage 2: Semantic title chunking...")
    chunks, chunk_stats = chunk_elements(
        raw_elements,
        max_chars=args.max_chars,
        project_id=args.project_id,
    )
    t_chunk = time.perf_counter() - t1
    print(f"✔ Created {len(chunks)} chunks in {t_chunk:.3f}s:")
    print(f"   - Average Size:      {chunk_stats.get('avg_chunk_size', 0)} characters")
    print(f"   - Breakdown:         {chunk_stats.get('chunk_type_breakdown', {})}")
    print("=" * width)

    # Preview chunks
    if args.preview > 0 and chunks:
        shown = min(args.preview, len(chunks))
        print(f"\n--- CHUNK PREVIEWS (Showing {shown} of {len(chunks)}) ---\n")
        for i, c in enumerate(chunks[:shown], 1):
            cid = c.get("chunk_id", "")[:16] + "..."
            ctype = c.get("type", "text").upper()
            title = c.get("title", "General")
            page = c.get("page_number", 1)
            raw = (c.get("content") or "").replace("\n", " ").strip()
            preview_text = (raw[:140] + "...") if len(raw) > 140 else raw
            print(f"[{i}] [{ctype}] Title: \"{title}\" | Page {page} | ID: {cid}")
            print(f"    \"{preview_text}\"\n")

    # Output JSON if requested
    if args.output_json:
        out_p = Path(args.output_json).resolve()
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "filename": file_path.name,
                "project_id": args.project_id,
                "total_chunks": len(chunks),
                "stats": chunk_stats,
                "chunks": chunks,
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Chunks successfully saved to: {out_p}")


if __name__ == "__main__":
    main()
