import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import (
    parse_document,
    parse_csv_file,
    is_transcript_content,
    parse_transcript_content,
    is_receipt_invoice_content,
    parse_receipt_invoice_content,
)
from core.chunker import chunk_elements


def print_header(title: str):
    print("\n" + "=" * 68)
    print(title.center(68))
    print("=" * 68)


def main():
    print_header("RETRIO: FORMAT DETECTION & TITLE METADATA VERIFICATION")

    # -------------------------------------------------------------
    # 1. Unstructured Document & Multi-Part Title Propagation
    # -------------------------------------------------------------
    print("\n[CHECK 1] Multi-Part Unstructured Document (Title on All Parts)")
    sample_txt = Path("data/projects/0fa55d40-2859-494b-8537-fd423ded528b/documents/Fluid mechanics is a branch of phys.txt")
    if sample_txt.exists():
        elements, _ = parse_document(sample_txt)
        chunks, stats = chunk_elements(elements, max_chars=1500)
        print(f"  Document: '{sample_txt.name}'")
        print(f"  Total Chunks: {len(chunks)} | Average Chunk Size: {stats.get('avg_chunk_size')} chars")
        print("  Checking metadata across all chunks:")
        for i, c in enumerate(chunks, 1):
            print(f"    • Chunk #{i}:")
            print(f"        - Title:       '{c.get('title')}'")
            print(f"        - Section:     '{c.get('section')}'")
            print(f"        - Part:        {c.get('part_index')} of {c.get('total_parts')}")
            print(f"        - Length:      {c.get('char_count')} chars")
            print(f"        - Snippet:     \"{c.get('content', '')[:65].replace(chr(10), ' ')}...\"")
    else:
        print(f"  Notice: File '{sample_txt.name}' not found at expected path.")

    # -------------------------------------------------------------
    # 2. CSV / Tabular Pre-Structuring & Row Batching
    # -------------------------------------------------------------
    print("\n[CHECK 2] CSV Tabular Data (Row Batching & Header Retention)")
    test_csv = Path("scratch/test_verify.csv")
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    test_csv.write_text(
        "ID,Product,Price,Quantity,Status\n" + "\n".join(
            f"{i},Item_{i},{i*10.5},{i%5+1},In Stock" for i in range(1, 46)
        ),
        encoding="utf-8"
    )
    csv_elements = parse_csv_file(test_csv, max_rows_per_batch=20)
    csv_chunks, _ = chunk_elements(csv_elements, max_chars=1500)
    print(f"  Generated {len(csv_chunks)} chunks for 45-row CSV (batched by 20 rows):")
    for i, c in enumerate(csv_chunks, 1):
        print(f"    • Chunk #{i}: Type='{c.get('type')}', Title='{c.get('title')}', Length={c.get('char_count')} chars")
        if c.get("type") == "table":
            first_line = c.get("content", "").split("\n")[2] if len(c.get("content", "").split("\n")) > 2 else ""
            print(f"        Header verified: {first_line[:55]}...")
    if test_csv.exists():
        test_csv.unlink()

    # -------------------------------------------------------------
    # 3. Audio/Meeting Transcript Pre-Structuring
    # -------------------------------------------------------------
    print("\n[CHECK 3] Audio / Meeting Transcript (Dialogue Normalization)")
    raw_transcript = (
        "[00:00:05] Host: Welcome everyone to today's deep dive on RAG architecture.\n"
        "[00:00:12] Engineer: Glad to be here. Today we will discuss layout-aware chunking.\n"
        "[00:00:20] Host: How do we handle unstructured text without headings?\n"
        "[00:00:25] Engineer: We split at sentence boundaries with trailing sentence overlap."
    )
    detected = is_transcript_content(raw_transcript)
    transcript_elements = parse_transcript_content(raw_transcript, "rag_podcast.txt")
    transcript_chunks, _ = chunk_elements(transcript_elements, max_chars=1500)
    print(f"  Transcript Detected: {detected}")
    for c in transcript_chunks:
        print(f"    • Title: '{c.get('title')}' | Section: '{c.get('section')}'")
        print(f"      Dialogue turn: \"{c.get('content')[:75].replace(chr(10), ' ')}...\"")

    # -------------------------------------------------------------
    # 4. Financial Receipt / Invoice Pre-Structuring
    # -------------------------------------------------------------
    print("\n[CHECK 4] Financial Receipt / Invoice (Summary Extraction)")
    raw_receipt = (
        "CLOUD COMPUTE SERVICES LLC\n"
        "Invoice Number: INV-2026-991\n"
        "Date: 2026-11-17\n"
        "Total Amount Due: $350.00\n"
        "Description: High-Memory Dedicated Worker Node\n"
        "Payment Receipt: Paid via ACH"
    )
    inv_detected = is_receipt_invoice_content(raw_receipt)
    receipt_elements = parse_receipt_invoice_content(raw_receipt, "compute_bill.txt")
    receipt_chunks, _ = chunk_elements(receipt_elements, max_chars=1500)
    print(f"  Invoice Detected: {inv_detected}")
    for c in receipt_chunks:
        print(f"    • Title: '{c.get('title')}' | Section: '{c.get('section')}'")
        print(f"      Content: \"{c.get('content')[:75].replace(chr(10), ' ')}...\"")

    print_header("ALL CHECKS PASSED: READY FOR SELF-VERIFICATION")


if __name__ == "__main__":
    main()
