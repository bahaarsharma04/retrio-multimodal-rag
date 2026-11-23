"""
Standalone CLI tool for testing Hybrid Retrieval (Dense pgvector + Sparse BM25 + RRF + Cohere Rerank).
Uses the database chunks table directly via get_all_chunks, dense_similarity_search,
BM25Okapi, _reciprocal_rank_fusion, and _rerank_with_cohere.

Usage:
    python scripts/retrieve_context.py
    python scripts/retrieve_context.py --project-id <project_id> --query "What is the Hubble constant?"
    python scripts/retrieve_context.py --project-id <project_id> --query "..." --top-n 5 --filter "p197.pdf"
"""
import sys
import argparse
import time
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_bm25 import BM25Okapi
from core.vectorstore import get_all_chunks, dense_similarity_search
from core.retriever import HybridRetriever, tokenize_for_bm25, count_tokens
from db.session import is_db_connected, get_db_session
from db.models import Project, DocumentChunk


def list_projects_with_chunks():
    """Returns a list of projects that contain indexed chunks in PostgreSQL."""
    try:
        with get_db_session() as s:
            projects = s.query(Project.id, Project.name).all()
            valid = []
            for p in projects:
                count = s.query(DocumentChunk).filter(DocumentChunk.project_id == str(p.id)).count()
                if count > 0:
                    valid.append((str(p.id), p.name, count))
            return valid
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Retrio CLI: Step-by-step testing of retrieval using PostgreSQL chunks table."
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Target PostgreSQL project ID (optional: auto-selects active project if omitted)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query string (optional: prompted interactively if omitted)",
    )
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
    print(" RETRIO HYBRID RETRIEVAL TESTING CLI (PostgreSQL pgvector) ".center(width, "="))
    print("=" * width)

    # 1. Verify PostgreSQL Connection
    if not is_db_connected():
        print("Error: Could not connect to PostgreSQL. Please ensure the database is running.", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve Project ID if omitted
    project_id = args.project_id
    if not project_id:
        available = list_projects_with_chunks()
        if not available:
            print("Error: No projects with indexed chunks found in the database.", file=sys.stderr)
            sys.exit(1)
        # Auto-select the first available project with chunks
        project_id, proj_name, chunk_count = available[0]
        print(f"ℹ No --project-id provided. Auto-selected project:")
        print(f"  Name:   {proj_name}")
        print(f"  ID:     {project_id}")
        print(f"  Chunks: {chunk_count} in database\n")
        print("  Available projects with chunks in DB:")
        for pid, pname, pchunks in available:
            print(f"    • {pid} ({pname}) - {pchunks} chunks")
        print("-" * width)

    # 3. Resolve Query if omitted
    query = args.query
    if not query:
        if sys.stdin.isatty():
            try:
                query = input("\nEnter search query: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                sys.exit(0)
        if not query:
            query = "What are the key concepts and findings?"
            print(f"ℹ No query entered. Using default test query: \"{query}\"")

    print(f" Target Project ID: {project_id}")
    print(f" Query:             \"{query}\"")
    print(f" Mode:              {args.mode.upper()} (Top-K: {args.top_k}, Top-N: {args.top_n})")
    if args.filter:
        print(f" Document Filter:   {args.filter}")
    print("-" * width)

    t0 = time.perf_counter()

    # =========================================================================
    # Step 1: get_all_chunks from PostgreSQL chunks table
    # =========================================================================
    print(f"\n[1/5] Fetching chunks from database via get_all_chunks('{project_id}')...")
    all_chunks = get_all_chunks(project_id)
    print(f"      ✔ Loaded {len(all_chunks)} chunks from database table 'document_chunks'.")

    if not all_chunks:
        print("Error: No chunks found for this project in PostgreSQL.", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # Step 2: dense_similarity_search for dense semantic results
    # =========================================================================
    dense_results = []
    if args.mode in ("hybrid", "dense-only"):
        print(f"\n[2/5] Executing dense search via dense_similarity_search()...")
        t_dense = time.perf_counter()
        dense_results = dense_similarity_search(
            project_id=project_id,
            query=query,
            k=args.top_k if args.mode == "hybrid" else args.top_n,
            filename_filter=args.filter,
        )
        dense_ms = (time.perf_counter() - t_dense) * 1000
        print(f"      ✔ Retrieved {len(dense_results)} dense chunks in {dense_ms:.1f}ms")

    # =========================================================================
    # Step 3: BM25Okapi for sparse lexical results from chunks table
    # =========================================================================
    sparse_results = []
    if args.mode in ("hybrid", "bm25-only"):
        print(f"\n[3/5] Building BM25Okapi index on database chunks & searching...")
        t_bm25 = time.perf_counter()
        tokenized_corpus = [
            tokenize_for_bm25(c.get("indexed_text") or c.get("content", ""), filter_stopwords=False)
            for c in all_chunks
        ]
        bm25_index = BM25Okapi(tokenized_corpus)
        query_tokens = tokenize_for_bm25(query, filter_stopwords=True)
        
        if query_tokens:
            scores = bm25_index.get_scores(query_tokens)
            scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            k_sparse = args.top_k if args.mode == "hybrid" else args.top_n
            for idx in scored_indices:
                if scores[idx] <= 0:
                    break
                c = all_chunks[idx]
                if args.filter and (c.get("filename") or "").lower() != args.filter.lower():
                    continue
                content = c.get("content", "")
                summary = c.get("summary")
                if content.strip().startswith(("[Embedded Figure:", "[Embedded Visual")) and not summary:
                    continue
                sparse_results.append({
                    "chunk_id": c.get("chunk_id"),
                    "content": content,
                    "filename": c.get("filename", "unknown"),
                    "page_number": c.get("page_number", 1),
                    "type": c.get("type", "text"),
                    "section": c.get("section") or c.get("metadata", {}).get("section") or "General",
                    "summary": summary,
                    "score": float(scores[idx]),
                })
                if len(sparse_results) >= k_sparse:
                    break
        bm25_ms = (time.perf_counter() - t_bm25) * 1000
        print(f"      ✔ Retrieved {len(sparse_results)} BM25 chunks in {bm25_ms:.1f}ms (Tokens: {query_tokens[:5]})")

    # =========================================================================
    # Step 4: Reciprocal Rank Fusion (_reciprocal_rank_fusion)
    # =========================================================================
    retriever_helper = HybridRetriever(project_id)
    if args.mode == "dense-only":
        final_results = dense_results[:args.top_n]
    elif args.mode == "bm25-only":
        final_results = sparse_results[:args.top_n]
    else:
        print(f"\n[4/5] Merging results via _reciprocal_rank_fusion()...")
        fused_candidates = retriever_helper._reciprocal_rank_fusion(dense_results, sparse_results)
        print(f"      ✔ Fused into {len(fused_candidates)} unique ranked candidates")

        # =====================================================================
        # Step 5: Cohere neural reranking (_rerank_with_cohere)
        # =====================================================================
        print(f"\n[5/5] Reranking candidates via _rerank_with_cohere()...")
        t_rerank = time.perf_counter()
        final_results = retriever_helper._rerank_with_cohere(query, fused_candidates, top_n=args.top_n)
        rerank_ms = (time.perf_counter() - t_rerank) * 1000
        print(f"      ✔ Reranked into {len(final_results)} top snippets in {rerank_ms:.1f}ms")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    q_tokens = count_tokens(query)
    total_chunk_tokens = sum(count_tokens(s.get("content") or "") for s in final_results)

    print("\n" + "=" * width)
    print(" RETRIEVAL EXECUTION SUMMARY ".center(width, "="))
    print("=" * width)
    print(f"✔ Retrieved count: {len(final_results)} chunks in {elapsed_ms:.1f}ms")
    print(f"  Query tokensize: {q_tokens} tokens | Total chunk tokensize: {total_chunk_tokens} tokens\n")
    print("-" * width)

    if not final_results:
        print("No matching snippets found.")
    else:
        for idx, s in enumerate(final_results, 1):
            fname = s.get("filename", "unknown")
            page = s.get("page_number", 1)
            stype = s.get("type", "text").upper()
            title = s.get("title", "General")
            rrf_score = s.get("rrf_score")
            relevance = s.get("rerank_score") or s.get("similarity")
            chunk_tokens = count_tokens(s.get("content") or "")
            
            score_str = ""
            if relevance is not None:
                score_str += f"Relevance/Rerank: {relevance:.4f}"
            if rrf_score is not None:
                sep = " | " if score_str else ""
                score_str += f"{sep}RRF: {rrf_score:.5f}"
            if not score_str and s.get("score") is not None:
                score_str = f"Score: {float(s['score']):.4f}"

            content = (s.get("content") or "").replace("\n", " ").strip()
            first_100 = content[:100]

            print(f"[{idx:>2}] [{stype}] {fname} (Page {page}) - \"{title}\"")
            print(f"     Score: {score_str} | tokensize: {chunk_tokens} tokens")
            print(f"     First 100 chars: \"{first_100}\"\n")

    print("=" * width)

    if args.output_json:
        out_p = Path(args.output_json).resolve()
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump({
                "project_id": project_id,
                "query": query,
                "mode": args.mode,
                "elapsed_ms": elapsed_ms,
                "total_retrieved": len(final_results),
                "snippets": final_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Retrieved results saved to: {out_p}")


if __name__ == "__main__":
    main()
