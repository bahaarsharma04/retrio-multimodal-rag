# Operational Scripts & CLI Reference

This document provides a comprehensive operational guide for all command-line interface (CLI) scripts located in the `scripts/` directory.

These tools allow developers, test automation engineers, and system operators to test, inspect, embed, retrieve, benchmark, and verify data directly from the command line with full telemetry (tokensize, candidate counts, first 100 characters previews, and stage latencies).

---

## 📋 Script Inventory

| Script | Purpose | Auto-Selects Defaults | Primary Dependencies |
| :--- | :--- | :--- | :--- |
| **`scripts/chunk_document.py`** | Document partitioning & semantic title chunking CLI | ✔ Yes (workspace doc) | `core/parser.py`, `core/chunker.py` |
| **`scripts/embed_document.py`** | Vectorization & PostgreSQL (`pgvector`) indexing CLI | ✔ Yes (active DB project & doc) | `core/vectorstore.py`, OpenAI |
| **`scripts/retrieve_context.py`** | Hybrid (Dense + BM25) + Cohere Reranking search CLI | ✔ Yes (active DB project & prompt) | `core/retriever.py`, Cohere |
| **`scripts/test_db.py`** | PostgreSQL & `pgvector` connectivity and database audit | ✔ Yes (runs full suite) | `SQLAlchemy`, `pgvector` |
| **`scripts/run_rag_benchmark.py`** | Comprehensive 17-metric IR and Generation benchmark | ✔ Yes (active DB project) | `core/rag_chain.py` |
| **`scripts/evaluate_accuracy.py`** | Automated regression testing & accuracy evaluator | ✔ Yes (active workspace project) | `core/rag_chain.py` |
| **`scripts/verify_formats_and_metadata.py`** | Multi-format title metadata and chunking verification | ✔ Yes (runs all 4 format checks) | `core/parser.py`, `core/chunker.py` |
| **`scripts/test_background_ingest_and_greeting.py`** | Async background ingestion & greeting interception test | ✔ Yes (self-contained integration) | `FastAPI TestClient`, `core/pipeline.py` |
| **`scripts/test_security_rbac.py`** | Multi-tenant isolation & SlowAPI rate limiting test | ✔ Yes (self-contained integration) | `FastAPI TestClient`, SlowAPI |
| **`scripts/migrate_json_to_postgres.py`** | Legacy JSON files to PostgreSQL migration utility | ✔ Yes (scans `data/projects`) | `db/session.py` |

---

## 1. Standalone Document Chunker (`scripts/chunk_document.py`)

Parses a raw document (PDF, DOCX, CSV, TXT, MD) using Unstructured / PyPDF, applies title-aware semantic chunking, computes exact token counts (`tiktoken`), and displays full chunk metadata and first 100 characters previews.

### Syntax
```bash
# Run with auto-selected document from workspace
python scripts/chunk_document.py

# Run on a specific file
python scripts/chunk_document.py [FILE_PATH] [OPTIONS]
```

### Options & Flags
| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `file_path` | String | No | Auto-detected | Path to the document file. If omitted, automatically discovers available documents in `data/projects/`. |
| `--project-id`, `-p` | String | No | `cli_project` | Scoping project ID for metadata tagging. |
| `--max-chars` | Int | No | `1500` | Maximum character boundary per chunk. |
| `--overlap` | Int | No | `250` | Trailing character overlap across sequential chunks. |
| `--preview` | Int | No | `10` | Number of chunks to preview in detail (`-1` for all). |
| `--verbose`, `-v` | Flag | No | `False` | Display full chunk text in addition to first 100 characters. |
| `--output-json`, `-o` | String | No | None | Save full chunks and telemetry to a JSON file. |

### Telemetry Displayed
- Partitioning element breakdown (`text_sections`, `tables`, `images`, `titles_headers`)
- Chunk creation latency (ms)
- Total document tokens and average tokensize per chunk (`tiktoken` `cl100k_base`)
- Total characters and average length
- Per-chunk details: ID, Page, Section title, Tokensize, and **First 100 characters**

### Example Invocations
```bash
# 1. Quick test on auto-discovered document
python scripts/chunk_document.py

# 2. Parse specific PDF and save JSON telemetry
python scripts/chunk_document.py data/sample_docs/quarterly_report.pdf --max-chars 1200 --output-json chunks.json

# 3. Verbose mode showing full content of all chunks
python scripts/chunk_document.py data/sample_docs/research.pdf --preview -1 --verbose
```

---

## 2. Standalone Chunk Embedder (`scripts/embed_document.py`)

Parses a document (or reads pre-generated chunks), computes OpenAI `text-embedding-3-small` 1536-dimensional vectors, indexes them directly into PostgreSQL's `document_chunks` table, and runs an immediate sanity vector query to verify live retrieval.

### Syntax
```bash
# Run with auto-selected project and document
python scripts/embed_document.py

# Run on a specific file and project
python scripts/embed_document.py --file <PATH_TO_FILE> --project-id <PROJECT_ID> [OPTIONS]
```

### Options & Flags
| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--project-id`, `-p` | String | No | Auto-detected | Target PostgreSQL project ID. If omitted, auto-selects an active project from the database. |
| `--file`, `-f` | String | No | Auto-detected | Document file to parse, chunk, and embed. |
| `--chunks-json`, `-c`| String | No | None | Pre-generated chunks JSON file (from `chunk_document.py`). |
| `--preview` | Int | No | `5` | Number of chunks to preview before embedding. |
| `--skip-search-check`| Flag | No | `False` | Skip the post-embedding pgvector search verification. |

### Telemetry Displayed
- Total chunks and total tokens to be embedded
- Preview of chunk IDs, types, tokensizes, and **first 100 characters**
- API embedding & PostgreSQL write latency (s)
- Vectors added, reused, and new total chunk count in PostgreSQL
- **Live Vector Search Sanity Test:** Runs a sample query against the newly embedded vectors and prints top match cosine scores and previews.

### Example Invocations
```bash
# 1. Automatic test run
python scripts/embed_document.py

# 2. Embed a specific document into a project
python scripts/embed_document.py \
  --project-id "0fa55d40-2859-494b-8537-fd423ded528b" \
  --file "data/sample_docs/balance_sheet.pdf"
```

---

## 3. Standalone Context Retriever (`scripts/retrieve_context.py`)

Executes dense vector search, sparse BM25 search, Reciprocal Rank Fusion (RRF), and Cohere neural re-ranking directly from the CLI against the PostgreSQL `document_chunks` table.

### Syntax
```bash
# Interactive / Auto-selects project and prompt
python scripts/retrieve_context.py

# Targeted search
python scripts/retrieve_context.py --query "What is renewable energy?" [OPTIONS]
```

### Options & Flags
| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--project-id`, `-p` | String | No | Auto-detected | Target project UUID containing indexed documents. |
| `--query`, `-q` | String | No | Interactive | The search query string. Prompted if omitted. |
| `--top-k`, `-k` | Int | No | `35` | Number of dense and sparse candidates to retrieve before fusion. |
| `--top-n`, `-n` | Int | No | `12` | Number of final chunks to return after Cohere neural reranking. |
| `--mode`, `-m` | String | No | `hybrid` | Search mode: `hybrid` (Dense+BM25+Rerank), `dense-only`, or `bm25-only`. |
| `--filter` | String | No | None | Filter results to a specific document filename. |
| `--output-json`, `-o`| String | No | None | File path to write full JSON results including scores and tokens. |
| `--verbose`, `-v` | Flag | No | `False` | Display full chunk text in addition to first 100 characters. |

### Telemetry Displayed
- Total chunks queried via `get_all_chunks()`
- Dense retrieval candidate count and latency (ms)
- Sparse BM25 candidate count, latency (ms), and tokenized terms
- **RRF Candidate Rankings:** Each candidate logged with `rrf_score`, **`tokensize`**, and **`first_100_chars`**
- **Cohere Neural Reranker Scores:** Each top snippet logged with `score`, **`tokensize`**, and **`first_100_chars`**
- Final execution summary with query tokensize, total chunk tokensize, page numbers, and section titles

### Example Invocations
```bash
# 1. Full hybrid retrieval with Cohere reranking
python scripts/retrieve_context.py --query "What are the key financial highlights and revenue figures?"

# 2. Compare with dense-only vector search
python scripts/retrieve_context.py --query "operating expenses" --mode dense-only --top-n 5

# 3. Filter to a single document with verbose output
python scripts/retrieve_context.py --query "solar energy" --filter "Renewable Energy.docx" --verbose
```

---

## 4. PostgreSQL & pgvector Database Health Suite (`scripts/test_db.py`)

Verifies PostgreSQL server connectivity, extensions (`pgvector`, `pgcrypto`, `uuid-ossp`), complete table row counts, and runs a live native `pgvector` Cosine Distance operator (`<=>`) check.

### Syntax
```bash
python scripts/test_db.py
python scripts/test_db.py --verbose
```

### Telemetry Displayed
- Connection ping latency in milliseconds
- Installed PostgreSQL extensions and version numbers
- Row counts for all 8 tables (`users`, `projects`, `documents`, `document_chunks`, `conversations`, `messages`, `message_citations`)
- Project workspace breakdown (name, ID, chunks, and documents with ingestion statuses)
- **Live pgvector Cosine Search Test:** Computes nearest vector distance to a sample chunk, logs latency (ms), tokensize, and first 100 characters of top 3 matches
- Chat persistence test with preview of recent user/assistant turns

---

## 5. Comprehensive RAG Benchmark Suite (`scripts/run_rag_benchmark.py`)

Runs an exhaustive 17-metric Information Retrieval (IR) and Generation evaluation across single-document, multi-document, causal reasoning, negative constraints, and meta-conversational test cases.

### Syntax
```bash
python scripts/run_rag_benchmark.py
python scripts/run_rag_benchmark.py --project-id <PROJECT_ID> --output report.json --verbose
```

### Options & Flags
| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--project-id`, `-p` | String | No | Auto-detected | Target project ID containing benchmark documents. |
| `--output`, `-o` | String | No | `scratch/rag_benchmark_report.json` | Destination path for benchmark JSON report. |
| `--verbose`, `-v` | Flag | No | `False` | Print full text of answers and retrieved context. |

### Metrics Computed
- **Retrieval:** HitRate@k, Precision@k, Recall@k, MRR, nDCG@k, MAP@k, Context Relevance, Noise Rate, F1
- **Generation:** Faithfulness, Anti-Hallucination Assertion Accuracy, Citation Accuracy, Answer Correctness, Answer Relevance, Context Utilisation, Completeness, Multi-Turn History Recall

---

## 6. Automated Accuracy & Regression Evaluator (`scripts/evaluate_accuracy.py`)

Evaluates answer accuracy and checks for performance regressions against a saved baseline.

### Syntax
```bash
# Run evaluation on current project documents
python scripts/evaluate_accuracy.py

# Save current accuracy metrics as regression baseline
python scripts/evaluate_accuracy.py --set-baseline

# Compare current code against baseline (exits with code 1 if performance degrades > 5%)
python scripts/evaluate_accuracy.py --compare --threshold 0.05
```

---

## 7. Format Detection & Title Metadata Suite (`scripts/verify_formats_and_metadata.py`)

Verifies layout-aware partitioning and semantic title propagation across four diverse data formats:
1. **Multi-Part Text / Unstructured Documents:** Verifies section titles propagate across all chunk parts.
2. **CSV / Tabular Data:** Verifies row batching (e.g. 20 rows/batch) and column header retention.
3. **Audio / Meeting Transcripts:** Verifies speaker turns and dialogue timestamps.
4. **Financial Receipts / Invoices:** Verifies key-value financial summaries and vendor extraction.

Logs **`tokensize`**, **`char_count`**, and **`first_100_chars`** for all generated test chunks.

### Syntax
```bash
python scripts/verify_formats_and_metadata.py
```

---

## 8. Background Ingestion & Greeting Verification (`scripts/test_background_ingest_and_greeting.py`)

End-to-end integration test verifying:
1. **Greeting Interception:** Verifies greetings ("hi", "hello") bypass retrieval and return instantaneous friendly welcome messages.
2. **Asynchronous Background Ingestion:** Uploads a document via `POST /api/v1/projects/{id}/documents`, verifies HTTP `202 Accepted` response, polls `GET .../status` until `completed`, inspects `GET .../pipeline` execution audit, and cleans up.

### Syntax
```bash
python scripts/test_background_ingest_and_greeting.py
```

---

## 9. Security, JWT & RBAC Verification (`scripts/test_security_rbac.py`)

Verifies multi-tenant database isolation, JWT access control, and API rate-limiting:
- Confirms unauthenticated requests receive `401 Unauthorized`.
- Registers User A and User B.
- Confirms User B cannot access, read, chat with, or delete User A's projects (receives `404 Not Found`).
- Executes repeated login requests to verify SlowAPI rate limiting triggers `429 Too Many Requests`.

### Syntax
```bash
python scripts/test_security_rbac.py
```

---

## 10. JSON to PostgreSQL Migration Utility (`scripts/migrate_json_to_postgres.py`)

Migrates legacy file-based project directories and JSON records into PostgreSQL with null-byte sanitization.

### Syntax
```bash
python scripts/migrate_json_to_postgres.py
```
