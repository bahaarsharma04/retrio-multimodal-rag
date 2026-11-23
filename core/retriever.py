import os
import re
import time
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from core.config import (
    COHERE_API_KEY,
    HYBRID_SEARCH_TOP_K,
    RERANK_TOP_N,
)
from core.vectorstore import dense_similarity_search, get_all_chunks
from core.logger import retrieval_logger

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_enc.encode(text, disallowed_special=()))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len((text or "").split())) if text else 0


BM25_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


def tokenize_for_bm25(text: str, filter_stopwords: bool = True) -> List[str]:
    """Extracts lowercase alphanumeric tokens, stripping punctuation and optionally filtering stopwords."""
    if not text:
        return []
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    cleaned = []
    for t in raw_tokens:
        clean_tok = t.strip("-._")
        if len(clean_tok) < 2:
            continue
        if filter_stopwords and clean_tok in BM25_STOPWORDS:
            continue
        cleaned.append(clean_tok)
    if not cleaned and raw_tokens:
        return [t.strip("-._") for t in raw_tokens if len(t.strip("-._")) >= 2]
    return cleaned


class HybridRetriever:
    """
    Combines Dense Vector Retrieval (PostgreSQL pgvector) with
    Sparse Lexical Retrieval (BM25 Okapi) using Reciprocal Rank Fusion (RRF),
    followed by an optional Cohere neural reranking stage.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.chunks: List[Dict[str, Any]] = get_all_chunks(project_id)
        
        # Initialize BM25 corpus index with cleaned, punctuation-free tokens
        self.bm25: Optional[BM25Okapi] = None
        if self.chunks:
            tokenized_corpus = [
                tokenize_for_bm25(c.get("indexed_text") or c.get("content", ""), filter_stopwords=False)
                for c in self.chunks
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def _dense_search(self, query: str, k: int, filename_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs semantic similarity search via PostgreSQL (pgvector)."""
        if not self.chunks:
            retrieval_logger.warning(f"Dense search skipped for Project '{self.project_id}': no chunks indexed.")
            return []
        try:
            hits = dense_similarity_search(self.project_id, query, k=k, filename_filter=filename_filter)
            q_tokens = count_tokens(query)
            retrieval_logger.info(
                f"Dense search: count={len(hits)} chunks | query_tokensize={q_tokens} | filter={filename_filter or 'none'}"
            )
            for idx, h in enumerate(hits, 1):
                raw = (h.get("content") or "").replace("\n", " ").strip()
                preview_100 = raw[:100]
                chunk_tokens = count_tokens(h.get("content") or "")
                retrieval_logger.info(
                    f"  [Dense Chunk #{idx}] id={h.get('chunk_id')} | sim={h.get('similarity', 0.0):.4f} | "
                    f"tokensize={chunk_tokens} | first_100_chars: \"{preview_100}\""
                )
            return hits
        except Exception as e:
            retrieval_logger.error(f"Dense search failed: {e}", exc_info=True)
            return []

    def _sparse_search(self, query: str, k: int, filename_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Performs lexical BM25 keyword search across indexed chunks."""
        if not self.bm25 or not self.chunks:
            retrieval_logger.warning(f"BM25 search skipped for Project '{self.project_id}': index is empty.")
            return []
        
        t0 = time.perf_counter()
        query_tokens = tokenize_for_bm25(query, filter_stopwords=True)
        if not query_tokens:
            return []
            
        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        sparse_docs = []
        for idx in scored_indices:
            if scores[idx] <= 0:
                break
            c = self.chunks[idx]
            if filename_filter and (c.get("filename") or "").lower() != filename_filter.lower():
                continue
            content = c.get("content", "")
            summary = c.get("summary")
            # Skip bare placeholder chunks that contain no real visual summary or caption
            if content.strip().startswith(("[Embedded Figure:", "[Embedded Visual")) and not summary:
                continue
            sparse_docs.append({
                "chunk_id": c.get("chunk_id"),
                "content": content,
                "filename": c.get("filename", "unknown"),
                "page_number": c.get("page_number", 1),
                "type": c.get("type", "text"),
                "section": c.get("section") or c.get("metadata", {}).get("section") or "General",
                "summary": summary,
                "score": float(scores[idx]),
            })
            if len(sparse_docs) >= k:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        top_score = f"{sparse_docs[0]['score']:.4f}" if sparse_docs else "0.0"
        q_tokens = count_tokens(query)
        retrieval_logger.info(
            f"BM25 search: count={len(sparse_docs)} chunks in {elapsed_ms:.1f}ms | query_tokensize={q_tokens} | "
            f"tokens={query_tokens[:5]} | filter={filename_filter or 'none'} | top_score={top_score}"
        )
        for idx, doc in enumerate(sparse_docs, 1):
            raw = (doc.get("content") or "").replace("\n", " ").strip()
            preview_100 = raw[:100]
            chunk_tokens = count_tokens(doc.get("content") or "")
            retrieval_logger.info(
                f"  [BM25 Chunk #{idx}] id={doc.get('chunk_id')} | score={doc['score']:.4f} | "
                f"tokensize={chunk_tokens} | first_100_chars: \"{preview_100}\""
            )
        return sparse_docs

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k_constant: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Combines two ranked result lists using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score(d) = sum(1 / (k_constant + rank(d)))
        """
        doc_map: Dict[str, Dict[str, Any]] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, doc in enumerate(dense_results):
            cid = doc.get("chunk_id") or doc.get("content")[:50]
            doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_constant + rank + 1))

        dense_cids = set(doc_map.keys())
        overlap_count = 0

        for rank, doc in enumerate(sparse_results):
            cid = doc.get("chunk_id") or doc.get("content")[:50]
            if cid in dense_cids:
                overlap_count += 1
            if cid not in doc_map:
                doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_constant + rank + 1))

        # Sort by unified RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        
        fused = []
        for cid in sorted_cids:
            item = dict(doc_map[cid])
            item["rrf_score"] = rrf_scores[cid]
            item["retrieval_method"] = "hybrid_rrf"
            fused.append(item)

        top_rrf = f"{fused[0]['rrf_score']:.5f}" if fused else "0.0"
        retrieval_logger.info(
            f"RRF fusion: merged {len(dense_results)} dense + {len(sparse_results)} sparse into {len(fused)} candidates ({overlap_count} overlapping boosts) | top_score={top_rrf}"
        )
        for rank, doc in enumerate(fused[:10], 1):
            raw = (doc.get("content") or "").replace("\n", " ").strip()
            preview_100 = raw[:100]
            chunk_tokens = count_tokens(doc.get("content") or "")
            retrieval_logger.info(
                f"  [RRF Candidate #{rank}] rrf_score={doc.get('rrf_score', 0.0):.5f} | tokensize={chunk_tokens} | first_100_chars: \"{preview_100}\""
            )
        return fused

    def _rerank_with_cohere(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Neural cross-encoder reranking using Cohere.
        Falls back to RRF candidates if Cohere API key is absent or fails.
        """
        api_key = os.getenv("COHERE_API_KEY") or COHERE_API_KEY
        if not api_key or "your-cohere" in api_key or not candidates:
            retrieval_logger.info("Cohere rerank bypassed (no API key configured or empty candidates); using RRF order.")
            reranked_fallback = candidates[:top_n]
            for rank, doc in enumerate(reranked_fallback, 1):
                raw = (doc.get("content") or doc.get("summary") or "").replace("\n", " ").strip()
                preview_100 = raw[:100]
                chunk_tokens = count_tokens(doc.get("content") or "")
                score = doc.get("rrf_score", 0.0)
                retrieval_logger.info(
                    f"  [Rerank Score #{rank} (RRF)] score={score:.5f} | tokensize={chunk_tokens} | first_100_chars: \"{preview_100}\""
                )
            return reranked_fallback

        t0 = time.perf_counter()
        try:
            import cohere
            co = cohere.ClientV2(api_key=api_key)
            docs_text = []
            for c in candidates:
                txt = (c.get("content") or "").strip()
                if not txt:
                    txt = (c.get("summary") or "").strip() or "Document snippet"
                docs_text.append(txt[:4000])
            
            response = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs_text,
                top_n=min(top_n, len(candidates)),
            )

            reranked = []
            for item in response.results:
                original_doc = dict(candidates[item.index])
                original_doc["rerank_score"] = float(item.relevance_score)
                original_doc["retrieval_method"] = "cohere_reranked"
                reranked.append(original_doc)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            top_rel = f"{reranked[0]['rerank_score']:.4f}" if reranked else "0.0"
            retrieval_logger.info(
                f"Cohere rerank v3.5 completed in {elapsed_ms:.1f}ms: count={len(reranked)} (from {len(candidates)} candidates) | top_score={top_rel}"
            )
            for rank, doc in enumerate(reranked, 1):
                raw = (doc.get("content") or doc.get("summary") or "").replace("\n", " ").strip()
                preview_100 = raw[:100]
                chunk_tokens = count_tokens(doc.get("content") or "")
                score = doc.get("rerank_score", 0.0)
                retrieval_logger.info(
                    f"  [Rerank Score #{rank}] score={score:.4f} | tokensize={chunk_tokens} | first_100_chars: \"{preview_100}\""
                )
            return reranked
        except Exception as e:
            retrieval_logger.warning(f"Cohere rerank failed ({e}); falling back to top {top_n} RRF candidates.")
            fallback_reranked = candidates[:top_n]
            for rank, doc in enumerate(fallback_reranked, 1):
                raw = (doc.get("content") or doc.get("summary") or "").replace("\n", " ").strip()
                preview_100 = raw[:100]
                chunk_tokens = count_tokens(doc.get("content") or "")
                score = doc.get("rrf_score", 0.0)
                retrieval_logger.info(
                    f"  [Rerank Score #{rank} (Fallback)] score={score:.5f} | tokensize={chunk_tokens} | first_100_chars: \"{preview_100}\""
                )
            return fallback_reranked

    def retrieve(
        self,
        query: str,
        top_k: int = HYBRID_SEARCH_TOP_K,
        top_n: int = RERANK_TOP_N,
        filename_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        End-to-end Hybrid Retrieval:
        1. Dense Vector Search (PostgreSQL pgvector)
        2. Sparse Keyword Search (BM25)
        3. Reciprocal Rank Fusion (RRF)
        4. Neural Reranking (Cohere or RRF fallback)
        """
        t0 = time.perf_counter()
        q_tokens = count_tokens(query)
        filter_msg = f" | Filter: '{filename_filter}'" if filename_filter else ""
        retrieval_logger.info(
            f"Retrieval initiated for Project: {self.project_id}{filter_msg} | query_tokensize={q_tokens} | Query: '{query[:80]}...'"
        )

        dense_docs = self._dense_search(query, k=top_k, filename_filter=filename_filter)
        sparse_docs = self._sparse_search(query, k=top_k, filename_filter=filename_filter)

        # Merge with Reciprocal Rank Fusion
        fused_candidates = self._reciprocal_rank_fusion(dense_docs, sparse_docs)

        # Apply two-stage reranker
        final_results = self._rerank_with_cohere(query, fused_candidates, top_n=top_n)

        # Document metadata guardrail: Ensure page 1 / title block chunks are present for document metadata queries
        q_lower = query.lower()
        if any(w in q_lower for w in ["author", "who wrote", "written by", "title", "publisher", "isbn", "editor"]):
            page1_chunks = []
            for c in self.chunks:
                if filename_filter and (c.get("filename") or "").lower() != filename_filter.lower():
                    continue
                if c.get("page_number") == 1:
                    page1_chunks.append({
                        "chunk_id": c.get("chunk_id"),
                        "content": c.get("content", ""),
                        "filename": c.get("filename", "unknown"),
                        "page_number": 1,
                        "type": c.get("type", "text"),
                        "section": c.get("section") or "General",
                        "summary": c.get("summary"),
                        "score": 0.0,
                        "retrieval_method": "metadata_anchor",
                    })
            if page1_chunks:
                existing_cids = {r.get("chunk_id") for r in final_results}
                for p1 in reversed(page1_chunks):
                    if p1.get("chunk_id") not in existing_cids:
                        final_results.insert(0, p1)

        total_ms = (time.perf_counter() - t0) * 1000
        total_chunk_tokens = sum(count_tokens(c.get("content") or "") for c in final_results)
        retrieval_logger.info(
            f"✔ Retrieval pipeline completed in {total_ms:.1f}ms | query_tokensize={q_tokens} tokens | "
            f"retrieved_count={len(final_results)} chunks | total_chunk_tokensize={total_chunk_tokens} tokens"
        )
        for idx, c in enumerate(final_results, 1):
            raw = (c.get("content") or "").replace("\n", " ").strip()
            preview_100 = raw[:100]
            chunk_tokens = count_tokens(c.get("content") or "")
            score = c.get("rerank_score") if c.get("rerank_score") is not None else c.get("rrf_score", c.get("score", 0.0))
            score_type = "rerank" if c.get("rerank_score") is not None else ("rrf" if c.get("rrf_score") is not None else "score")
            retrieval_logger.info(
                f"  [Retrieved Chunk #{idx}] id={c.get('chunk_id')} | doc={c.get('filename')} (p.{c.get('page_number', 1)}) | "
                f"{score_type}_score={score:.4f} | tokensize={chunk_tokens} | first_100_chars: \"{preview_100}\""
            )
        return final_results

