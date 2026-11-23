# Standalone Hybrid Retrieval CLI (`scripts/retrieve_context.py`)

A standalone, production-grade CLI utility for testing, debugging, and inspecting **Retrio's 5-Stage Hybrid Retrieval Engine** directly against PostgreSQL (`pgvector`) and BM25 Okapi without running the FastAPI server or Streamlit UI.

---

## 1. Overview & Purpose

`scripts/retrieve_context.py` provides a transparent, step-by-step diagnostic view of how a user query traverses Retrio's hybrid retrieval stack:

```
[ PostgreSQL 'document_chunks' Table ]
                  │
                  ▼ (Step 1: get_all_chunks)
        [ In-Memory Corpus & BM25 Index ]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
(Step 2: Dense pgvector)  (Step 3: BM25Okapi)
        └─────────┬─────────┘
                  ▼ (Step 4: _reciprocal_rank_fusion)
      [ Fused Candidate Pool ]
                  │
                  ▼ (Step 5: _rerank_with_cohere)
      [ Top-N Neural Reranked Chunks ]
```

### Why Use This CLI Tool?
- **Zero Frontend/API Overhead:** Directly inspect retrieval quality, rank fusion, and cross-encoder scores in your terminal.
- **Granular Telemetry:** Every execution outputs exact **token sizes** (`tiktoken`), **candidate counts** at each stage, and **first-100-character previews** for every chunk and reranker score.
- **Fail-Fast Single Database:** Queries strictly PostgreSQL (`pgvector`); no reliance on local ChromaDB files.
- **Interactive Auto-Detection:** Automatically discovers projects with chunks in the database if no arguments are provided.

---

## 2. The 5-Stage Retrieval Pipeline

When you run `scripts/retrieve_context.py`, it executes the following explicit steps:

### Stage 1: Load Chunks from Database (`get_all_chunks`)
- Queries the PostgreSQL table `document_chunks` (joined with `documents`).
- Retrieves all indexed chunks, titles, section metadata, page numbers, and vector existence.

### Stage 2: Dense Semantic Search (`dense_similarity_search`)
- Embeds the query using OpenAI `text-embedding-3-small` (1536 dimensions).
- Performs native cosine distance search directly inside PostgreSQL using the `<=>` operator:
  $$\text{cosine\_distance} = 1 - \frac{u \cdot v}{\|u\| \|v\|}$$
- Returns the top $K$ nearest vectors ordered by distance ascending.

### Stage 3: Sparse Lexical Search (`BM25Okapi`)
- Builds a `BM25Okapi` index over the tokenized corpus extracted from the database.
- Cleans and tokenizes the search query, stripping English stopwords.
- Computes exact lexical match scores and retrieves top $K$ candidates.

### Stage 4: Reciprocal Rank Fusion (`_reciprocal_rank_fusion`)
- Merges the dense and sparse candidate lists into a unified, de-duplicated ranked list.
- Uses standard Reciprocal Rank Fusion (RRF) with constant $k = 60$:
  $$RRF\_Score(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- Candidates appearing in **both** dense and sparse lists receive a compound rank boost.

### Stage 5: Neural Cross-Encoder Reranking (`_rerank_with_cohere`)
- Feeds the top fused candidates and the user query to Cohere `rerank-v3.5`.
- Evaluates full query-document semantic relevance, outputting cross-encoder scores ($0.0 \dots 1.0$).
- *Fallback Resilience:* If no Cohere API key is configured or rate-limited, safely preserves top-N RRF candidate order.

---

## 3. CLI Command Reference & Arguments

```powershell
python scripts\retrieve_context.py [OPTIONS]
```

### Options:
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--project-id` | `str` | `None` | Target project UUID. If omitted, automatically lists and selects an active project from PostgreSQL. |
| `--query` | `str` | `None` | Search query string. If omitted, prompts interactively or defaults to a smoke test query. |
| `--top-k` | `int` | `35` | Initial candidate depth for dense and sparse search stages. |
| `--top-n` | `int` | `12` | Final number of reranked snippets returned. |
| `--mode` | `choice` | `hybrid` | Retrieval strategy: `hybrid` (dense + BM25 + RRF + Cohere), `dense-only`, or `bm25-only`. |
| `--filter` | `str` | `None` | Restrict search to a specific document filename (e.g. `report.pdf`). |
| `--output-json` | `str` | `None` | Save full structured retrieval output and timing telemetry to a JSON file. |

---

## 4. Usage Examples

### 1. Simple Run (Auto-Detection Mode)
If you don't know the project UUID or just want to quickly test retrieval:
```powershell
python scripts\retrieve_context.py
```
*Output will list all projects with chunks in the database and run on the first active project.*

---

### 2. Standard Hybrid Retrieval with Custom Top-N
```powershell
python scripts\retrieve_context.py --project-id "ade1ae6a-b265-4090-8d43-5875909edbf9" --query "What is the Friedmann equation?" --top-n 3
```

---

### 3. Filter to a Specific Document
Isolate retrieval to a single PDF within a multi-document project:
```powershell
python scripts\retrieve_context.py --project-id "ade1ae6a-b265-4090-8d43-5875909edbf9" --query "reheating temperature" --filter "p197.pdf" --top-n 5
```

---

### 4. Compare Dense-Only vs. BM25-Only Search
- **Dense semantic search only (pgvector):**
  ```powershell
  python scripts\retrieve_context.py --project-id "ade1ae6a-b265-4090-8d43-5875909edbf9" --query "inflationary expansion" --mode dense-only --top-n 5
  ```
- **BM25 lexical search only:**
  ```powershell
  python scripts\retrieve_context.py --project-id "ade1ae6a-b265-4090-8d43-5875909edbf9" --query "inflationary expansion" --mode bm25-only --top-n 5
  ```

---

### 5. Export Results to JSON for Evaluation
```powershell
python scripts\retrieve_context.py --project-id "ade1ae6a-b265-4090-8d43-5875909edbf9" --query "dark energy density parameter" --top-n 5 --output-json results.json
```

---

## 5. Sample Output & Telemetry Explanation

```text
================================================================================
========== RETRIO HYBRID RETRIEVAL TESTING CLI (PostgreSQL pgvector) ===========
================================================================================
 Target Project ID: ade1ae6a-b265-4090-8d43-5875909edbf9
 Query:             "Friedmann equation"
 Mode:              HYBRID (Top-K: 35, Top-N: 3)
--------------------------------------------------------------------------------

[1/5] Fetching chunks from database via get_all_chunks('ade1ae6a-b265-4090-8d43-5875909edbf9')...
      ✔ Loaded 184 chunks from database table 'document_chunks'.

[2/5] Executing dense search via dense_similarity_search()...
      ✔ Retrieved 35 dense chunks in 2255.3ms

[3/5] Building BM25Okapi index on database chunks & searching...
      ✔ Retrieved 19 BM25 chunks in 80.5ms (Tokens: ['friedmann', 'equation'])

[4/5] Merging results via _reciprocal_rank_fusion()...
      ✔ Fused into 40 unique ranked candidates

[5/5] Reranking candidates via _rerank_with_cohere()...
      ✔ Reranked into 3 top snippets in 2987.3ms

================================================================================
========================= RETRIEVAL EXECUTION SUMMARY ==========================
================================================================================
✔ Retrieved count: 3 chunks in 6239.7ms
  Query tokensize: 4 tokens | Total chunk tokensize: 795 tokens

--------------------------------------------------------------------------------
[ 1] [TEXT] p197.pdf (Page 7) - "V = a3"
     Score: Relevance/Rerank: 0.9049 | RRF: 0.03252 | tokensize: 384 tokens
     First 100 chars: "COSMOLOGY  2.4 Friedmann equation  The basic equation governing the expansion rate of the Universe i"

[ 2] [TEXT] p197.pdf (Page 43) - "V (φ)"
     Score: Relevance/Rerank: 0.8124 | RRF: 0.03252 | tokensize: 178 tokens
     First 100 chars: "φ V (φ)  ‘particle’ with coordinate moving with friction in the potential , the friction coefficient"

[ 3] [TEXT] p197.pdf (Page 24) - "COSMOLOGY"
     Score: Relevance/Rerank: 0.8026 | RRF: 0.03101 | tokensize: 233 tokens
     First 100 chars: "Owing to the dominance of dark energy, the Universe accelerates today. When matter was dominating,  "

================================================================================
```

### Metric Definitions:
- **`Query tokensize`**: Exact token count of the search query encoded via OpenAI `cl100k_base` (`tiktoken`).
- **`Total chunk tokensize`**: Sum of tokens across all final returned chunks, showing the exact prompt context load.
- **`Relevance/Rerank`**: Cross-encoder semantic score assigned by Cohere `rerank-v3.5` ($0.0 \dots 1.0$).
- **`RRF`**: Reciprocal Rank Fusion compound score from combining pgvector dense and BM25 sparse lists.
- **`First 100 chars`**: Unambiguous visual verification of the chunk's opening content.

---

## 6. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Error: Could not connect to PostgreSQL` | Docker container or PostgreSQL service is not running. | Run `docker compose up -d` and verify port 5432 is accessible. |
| `Error: No chunks found for this project in PostgreSQL` | Document has not yet been embedded or project ID is wrong. | Run `scripts\embed_document.py` to index document chunks into the project. |
| `Cohere rerank bypassed` | `COHERE_API_KEY` is missing in `.env`. | Add a free Cohere trial API key to `.env`, or let it fall back to RRF rankings. |
