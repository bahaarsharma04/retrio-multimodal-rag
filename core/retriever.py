import os
import time
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from core.config import (
    COHERE_API_KEY,
    HYBRID_SEARCH_TOP_K,
    RERANK_TOP_N,
)
from core.vectorstore import get_project_vectorstore, get_all_chunks
from core.logger import retrieval_logger


class HybridRetriever:
    """
    Combines Dense Vector Retrieval (ChromaDB + Embeddings) with
    Sparse Lexical Retrieval (BM25 Okapi) using Reciprocal Rank Fusion (RRF),
    followed by an optional Cohere neural reranking stage.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.vectorstore = get_project_vectorstore(project_id)
        self.chunks: List[Dict[str, Any]] = get_all_chunks(project_id)
        
        # Initialize BM25 corpus index
        self.bm25: Optional[BM25Okapi] = None
        if self.chunks:
            tokenized_corpus = [
                (c.get("indexed_text") or c.get("content", "")).lower().split()
                for c in self.chunks
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def _dense_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Performs semantic similarity search via ChromaDB."""
        if not self.chunks:
            retrieval_logger.warning(f"Dense search skipped for Project '{self.project_id}': no chunks indexed.")
            return []
        t0 = time.perf_counter()
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            dense_docs = []
            for doc, score in results:
                meta = doc.metadata or {}
                dense_docs.append({
                    "chunk_id": meta.get("chunk_id"),
                    "content": meta.get("raw_content") or doc.page_content,
                    "filename": meta.get("filename", "unknown"),
                    "page_number": meta.get("page_number", 1),
                    "type": meta.get("type", "text"),
                    "summary": meta.get("summary") or None,
                    "score": float(score) if score is not None else 0.0,
                })
            elapsed_ms = (time.perf_counter() - t0) * 1000
            top_score = f"{dense_docs[0]['score']:.4f}" if dense_docs else "None"
            retrieval_logger.info(f"Dense search: retrieved {len(dense_docs)} chunks in {elapsed_ms:.1f}ms | Top distance/score: {top_score}")
            return dense_docs
        except Exception as e:
            retrieval_logger.error(f"Dense search failed: {e}", exc_info=True)
            return []

    def _sparse_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Performs lexical BM25 keyword search across all indexed chunks."""
        if not self.bm25 or not self.chunks:
            retrieval_logger.warning(f"BM25 search skipped for Project '{self.project_id}': index is empty.")
            return []
        
        t0 = time.perf_counter()
        query_tokens = query.lower().split()
        if not query_tokens:
            return []
            
        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        sparse_docs = []
        for idx in scored_indices[:k]:
            if scores[idx] > 0:  # Only return documents that had at least one keyword match
                c = self.chunks[idx]
                sparse_docs.append({
                    "chunk_id": c.get("chunk_id"),
                    "content": c.get("content", ""),
                    "filename": c.get("filename", "unknown"),
                    "page_number": c.get("page_number", 1),
                    "type": c.get("type", "text"),
                    "summary": c.get("summary"),
                    "score": float(scores[idx]),
                })
        elapsed_ms = (time.perf_counter() - t0) * 1000
        top_score = f"{sparse_docs[0]['score']:.4f}" if sparse_docs else "0.0"
        retrieval_logger.info(f"BM25 search: retrieved {len(sparse_docs)} chunks in {elapsed_ms:.1f}ms | Tokens: {query_tokens[:5]} | Top score: {top_score}")
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
        retrieval_logger.info(f"RRF fusion: merged {len(dense_results)} dense + {len(sparse_results)} sparse into {len(fused)} candidates ({overlap_count} overlapping boosts) | Top RRF: {top_rrf}")
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
            return candidates[:top_n]

        t0 = time.perf_counter()
        try:
            import cohere
            co = cohere.ClientV2(api_key=api_key)
            docs_text = [c["content"] for c in candidates]
            
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
            retrieval_logger.info(f"Cohere rerank v3.5 completed in {elapsed_ms:.1f}ms: {len(candidates)} candidates -> {len(reranked)} top chunks | Top relevance: {top_rel}")
            return reranked
        except Exception as e:
            retrieval_logger.warning(f"Cohere rerank failed ({e}); falling back to top {top_n} RRF candidates.")
            return candidates[:top_n]

    def retrieve(
        self,
        query: str,
        top_k: int = HYBRID_SEARCH_TOP_K,
        top_n: int = RERANK_TOP_N,
    ) -> List[Dict[str, Any]]:
        """
        End-to-end Hybrid Retrieval:
        1. Dense Vector Search (Chroma)
        2. Sparse Keyword Search (BM25)
        3. Reciprocal Rank Fusion (RRF)
        4. Neural Reranking (Cohere or RRF fallback)
        """
        t0 = time.perf_counter()
        retrieval_logger.info(f"Retrieval initiated for Project: {self.project_id} | Query: '{query[:80]}...'")

        dense_docs = self._dense_search(query, k=top_k)
        sparse_docs = self._sparse_search(query, k=top_k)

        # Merge with Reciprocal Rank Fusion
        fused_candidates = self._reciprocal_rank_fusion(dense_docs, sparse_docs)

        # Apply two-stage reranker
        final_results = self._rerank_with_cohere(query, fused_candidates, top_n=top_n)

        total_ms = (time.perf_counter() - t0) * 1000
        retrieval_logger.info(f"✔ Retrieval pipeline completed in {total_ms:.1f}ms: returned {len(final_results)} chunks to LLM.")
        return final_results
