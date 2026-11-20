"""
Standalone CLI tool for testing Hybrid Retrieval (Dense pgvector + Sparse BM25 + RRF + Cohere Rerank).
Usage:
    python scripts/retrieve_context.py --project-id <project_id> --query "What is the Hubble constant?"
    python scripts/retrieve_context.py --project-id <project_id> --query "..." --top-n 8 --filter "p197.pdf"
"""
import sys
import argparse
import time
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retriever import HybridRetriever
from db.session import is_db_connected


def main():
    parser = argparse.ArgumentParser(
        description="Retrio CLI: Execute hybrid retrieval against PostgreSQL and inspect ranked snippets."
    )
    parser.add_argument("--project-id", type=str, required=True, help="Target PostgreSQL project ID")
    parser.add_argument("--query", type=str, required=True, help="Search query string")
    parser.add_argument("--top-k", type=int, default=35, help="Initial candidate depth for search (default: 35)")
    parser.add_argument("--top-n", type=int, default=12, help="Final reranked snippet count (default: 12)")
    parser.add_argument("--filter", type=str, default=None, help="Filter to a specific document filename")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hybrid", "dense-only", "bm25-only"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Save retrieved snippets to a JSON file")

    args = parser.parse_args()

    width = 80
    print("=" * width)
    print(" RETRIO HYBRID RETRIEVAL CLI (PostgreSQL pgvector) ".center(width, "="))
    print("=" * width)

    # Verify PostgreSQL Connection
    if not is_db_connected():
        print("Error: Could not connect to PostgreSQL. Please ensure the database is running.", file=sys.stderr)
        sys.exit(1)

    print(f" Target Project ID: {args.project_id}")
    print(f" Query:             \"{args.query}\"")
    print(f" Mode:              {args.mode.upper()} (Top-K: {args.top_k}, Top-N: {args.top_n})")
    if args.filter:
        print(f" Document Filter:   {args.filter}")
    print("-" * width)

    t0 = time.perf_counter()
    retriever = HybridRetriever(args.project_id)

    if args.mode == "dense-only":
        results = retriever._dense_search(args.query, k=args.top_n, filename_filter=args.filter)
    elif args.mode == "bm25-only":
        results = retriever._sparse_search(args.query, k=args.top_n, filename_filter=args.filter)
    else:
        results = retriever.retrieve(
            query=args.query,
            top_k=args.top_k,
            top_n=args.top_n,
            filename_filter=args.filter,
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"✔ Retrieved {len(results)} relevant snippets in {elapsed_ms:.1f}ms\n")
    print("-" * width)

    if not results:
        print("No matching snippets found.")
    else:
        for idx, s in enumerate(results, 1):
            fname = s.get("filename", "unknown")
            page = s.get("page_number", 1)
            stype = s.get("type", "text").upper()
            title = s.get("title", "General")
            rrf_score = s.get("rrf_score")
            relevance = s.get("rerank_score") or s.get("similarity")
            
            score_str = ""
            if relevance is not None:
                score_str += f" | Relevance: {relevance:.4f}"
            if rrf_score is not None:
                score_str += f" | RRF: {rrf_score:.5f}"

            content = (s.get("content") or "").replace("\n", " ").strip()
            preview = (content[:200] + "...") if len(content) > 200 else content

            print(f"[{idx:>2}] [{stype}] {fname} (Page {page}) - \"{title}\"{score_str}")
            print(f"     \"{preview}\"\n")

    print("=" * width)

    if args.output_json:
        out_p = Path(args.output_json).resolve()
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "project_id": args.project_id,
                "query": args.query,
                "mode": args.mode,
                "elapsed_ms": elapsed_ms,
                "total_retrieved": len(results),
                "snippets": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Retrieved results saved to: {out_p}")


if __name__ == "__main__":
    main()
