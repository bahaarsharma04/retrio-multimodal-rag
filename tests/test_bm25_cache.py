import unittest
from core.retriever import (
    get_cached_project_bm25,
    invalidate_bm25_cache,
    _PROJECT_BM25_CACHE,
)


class TestBM25Cache(unittest.TestCase):
    def setUp(self):
        invalidate_bm25_cache()

    def tearDown(self):
        invalidate_bm25_cache()

    def test_cache_stores_and_reuses_index(self):
        project_id = "test_project_cache_abc"
        
        # Pre-seed cache entry to test retrieval without database
        mock_chunks = [
            {"chunk_id": "c1", "content": "Machine learning and deep neural networks in RAG.", "filename": "doc1.pdf"},
            {"chunk_id": "c2", "content": "PostgreSQL vector database and hybrid search.", "filename": "doc2.pdf"},
        ]
        
        # First call builds/caches
        from rank_bm25 import BM25Okapi
        from core.retriever import tokenize_for_bm25
        tokenized = [tokenize_for_bm25(c["content"], filter_stopwords=False) for c in mock_chunks]
        bm25_obj = BM25Okapi(tokenized)
        
        _PROJECT_BM25_CACHE[project_id] = {
            "chunks": mock_chunks,
            "bm25": bm25_obj,
            "cached_at": 1000.0,
        }
        
        # Second call should fetch exact cached instance
        chunks, bm25 = get_cached_project_bm25(project_id)
        self.assertIs(chunks, mock_chunks)
        self.assertIs(bm25, bm25_obj)

    def test_cache_invalidation_purges_project(self):
        project_id = "test_project_purge_123"
        _PROJECT_BM25_CACHE[project_id] = {
            "chunks": [],
            "bm25": None,
            "cached_at": 1000.0,
        }
        self.assertIn(project_id, _PROJECT_BM25_CACHE)
        
        invalidate_bm25_cache(project_id)
        self.assertNotIn(project_id, _PROJECT_BM25_CACHE)

    def test_global_cache_invalidation_clears_all(self):
        _PROJECT_BM25_CACHE["proj_1"] = {"chunks": [], "bm25": None}
        _PROJECT_BM25_CACHE["proj_2"] = {"chunks": [], "bm25": None}
        self.assertEqual(len(_PROJECT_BM25_CACHE), 2)
        
        invalidate_bm25_cache(None)
        self.assertEqual(len(_PROJECT_BM25_CACHE), 0)


if __name__ == "__main__":
    unittest.main()
