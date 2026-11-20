# Operational Scripts & CLI Reference

This document provides a comprehensive operational guide for all command-line interface (CLI) scripts located in the `scripts/` directory.

These tools allow developers, test automation engineers, and system operators to test, inspect, embed, retrieve, benchmark, and migrate data without running the web UI.

---

## 📋 Script Inventory

| Script | Purpose | Primary Dependencies |
| :--- | :--- | :--- |
| **`scripts/chunk_document.py`** | Standalone document partitioning and title-aware chunking tool | `core/parser.py`, `core/chunker.py` |
| **`scripts/embed_document.py`** | Standalone vectorization and PostgreSQL (`pgvector`) ingestion tool | `core/vectorstore.py`, OpenAI |
| **`scripts/retrieve_context.py`** | Standalone Hybrid (Dense + BM25) + Cohere Reranking search tool | `core/retriever.py`, Cohere |
| **`scripts/test_background_ingest_and_greeting.py`** | Verification suite for async background ingestion and greeting interceptor | `FastAPI TestClient`, `core/pipeline.py` |
| **`scripts/test_security_rbac.py`** | Verification suite for multi-tenant user isolation and rate limiting | `FastAPI TestClient`, SlowAPI |
| **`scripts/test_db.py`** | PostgreSQL and `pgvector` connectivity and record audit suite | `SQLAlchemy`, `pgvector` |
| **`scripts/run_rag_benchmark.py`** | Comprehensive Information Retrieval (IR) and Generation metrics suite | `core/rag_chain.py` |
| **`scripts/evaluate_accuracy.py`** | Automated regression testing and precision/recall evaluation runner | `core/rag_chain.py` |
| **`scripts/migrate_json_to_postgres.py`** | Legacy JSON files to PostgreSQL database migration utility | `db/session.py` |

---

## 1. Standalone Document Chunker (`scripts/chunk_document.py`)

Parses a raw document (PDF, DOCX, CSV, TXT, MD) using Unstructured, applies title-aware semantic chunking, and writes the chunked JSON representation to disk or standard output.

### Syntax
```bash
python scripts/chunk_document.py --input-file <PATH_TO_FILE> [OPTIONS]
```

### Options & Flags
| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input-file`, `-i` | String | **Yes** | — | Absolute or relative path to the input document file. |
| `--output-file`, `-o`| String | No | None | Destination path to write the JSON array of chunks. If omitted, prints to stdout. |
| `--project-id` | String | No | `cli_test_project` | Associated project UUID for chunk namespace tagging. |
| `--strategy` | String | No | `fast` | Partitioning strategy: `fast`, `hi_res`, or `auto`. |
| `--extract-tables` | Flag | No | `True` | Whether to extract HTML table representations. |

### Example Invocations
```bash
# 1. Parse PDF and inspect chunk breakdown on console
python scripts/chunk_document.py --input-file data/sample_docs/quarterly_report.pdf

# 2. Output chunks to a JSON file for inspection
python scripts/chunk_document.py \
  --input-file data/sample_docs/balance_sheet.pdf \
  --output-file data/chunks_output.json \
  --strategy hi_res
```

---

## 2. Standalone Chunk Embedder (`scripts/embed_document.py`)

Takes a chunked JSON file (or reads chunks directly from an existing project document) and indexes them into PostgreSQL using `pgvector` and OpenAI's `text-embedding-3-small`.

### Syntax
```bash
python scripts/embed_document.py --project-id <PROJECT_ID> [OPTIONS]
```

### Options & Flags
| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--project-id`, `-p` | String | **Yes** | — | Target project UUID in PostgreSQL. |
| `--chunks-file`, `-c` | String | No | None | Path to a pre-generated chunks JSON file (from `chunk_document.py`). |
| `--filename`, `-f` | String | No | None | Document filename to associate with chunks in PostgreSQL. |
| `--batch-size` | Int | No | `64` | Embedding batch size sent to the OpenAI Embeddings API. |
| `--dry-run` | Flag | No | `False` | Computes embeddings without writing rows to PostgreSQL. |

### Example Invocations
```bash
# Embed pre-chunked JSON directly into the database
python scripts/embed_document.py \
  --project-id 0fa55d40-2859-494b-8537-fd423ded528b \
  --chunks-file data/chunks_output.json \
  --filename balance_sheet.pdf
```

---

## 3. Standalone Context Retriever (`scripts/retrieve_context.py`)

Executes dense vector search, sparse BM25 search, Reciprocal Rank Fusion (RRF), and Cohere neural re-ranking directly from the CLI.

Per system specifications, all candidate chunks are queried directly from the `document_chunks` table in PostgreSQL.

### Syntax
```bash
python scripts/retrieve_context.py --project-id <PROJECT_ID> --query <QUERY_STRING> [OPTIONS]
```

### Options & Flags
| Flag | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--project-id`, `-p` | String | **Yes** | — | Target project UUID containing indexed documents. |
| `--query`, `-q` | String | **Yes** | — | The search query string. |
| `--top-k`, `-k` | Int | No | `15` | Number of dense and sparse candidates to retrieve before fusion. |
| `--top-n`, `-n` | Int | No | `4` | Number of final chunks to return after Cohere neural reranking. |
| `--mode`, `-m` | String | No | `hybrid` | Search mode: `hybrid` (Dense+BM25+Rerank), `dense-only`, or `bm25-only`. |
| `--output-json`, `-o`| String | No | None | Optional file path to write full JSON results including scores and tokens. |

### Output Metrics Displayed
The CLI output reports:
- Total chunks queried from PostgreSQL
- Candidate counts for dense and sparse retrieval stages
- Token counts (calculated via `tiktoken` `cl100k_base`) for the query and every retrieved chunk
- First 100 characters of each chunk
- Cohere neural relevance score (0.0 to 1.0)

### Example Invocations
```bash
# 1. Run hybrid retrieval with Cohere reranking
python scripts/retrieve_context.py \
  --project-id 0fa55d40-2859-494b-8537-fd423ded528b \
  --query "What are the key financial highlights and revenue figures?"

# 2. Run dense-only search and save full JSON payload
python scripts/retrieve_context.py \
  --project-id 0fa55d40-2859-494b-8537-fd423ded528b \
  --query "operating expenses" \
  --mode dense-only \
  --top-k 10 \
  --output-json search_results.json
```

---

## 4. Background Ingestion & Greeting Verification (`scripts/test_background_ingest_and_greeting.py`)

Automated end-to-end integration test verifying two critical platform features:
1. **Greeting Interception:** Verifies that greetings ("hi", "hello") bypass retrieval and return instantaneous friendly welcome messages.
2. **Asynchronous Document Ingestion:** Uploads a document via `POST /api/v1/projects/{id}/documents`, confirms HTTP `202 Accepted` response, polls `GET .../status` until `completed`, and verifies clean database cleanup upon deletion.

### Syntax
```bash
python scripts/test_background_ingest_and_greeting.py
```

### Exit Codes & Assertions
- Exit `0`: All greeting patterns and background ingestion tests passed.
- Exit `1`: Assertion failure (e.g. upload did not return 202, polling timed out, or greeting failed).

---

## 5. Security, JWT & RBAC Verification (`scripts/test_security_rbac.py`)

Verifies multi-tenant database isolation, JWT access control, and API rate-limiting:
- Confirms unauthenticated requests receive `401 Unauthorized`.
- Registers User A and User B.
- Ensures User B cannot access, read, chat with, or delete User A's projects (receives `404 Not Found`).
- Executes repeated login requests to verify SlowAPI rate limiting triggers `429 Too Many Requests`.

### Syntax
```bash
python scripts/test_security_rbac.py
```

---

## 6. PostgreSQL & pgvector Status Audit (`scripts/test_db.py`)

Verifies PostgreSQL server connectivity, table row counts, and live `pgvector` cosine similarity calculations.

### Syntax
```bash
python scripts/test_db.py
```

### Output Sample
```
=================================================================
Retrio PostgreSQL & pgvector Verification Suite
=================================================================
1. Database Connection Status: CONNECTED

2. Database Record Overview:
   • Projects:          18
   • Documents:         34
   • Document Chunks:   4730
   • Stored Embeddings: 4730 in pgvector
   • Conversations:     39
   • Messages:          593

4. Testing Native pgvector Cosine Similarity Search:
   Target chunk: "Unveiling Our Times..."
   Top 3 nearest results retrieved by PostgreSQL:
      1. [Page [2] | Section: General] "Unveiling Our Times..." (Distance: 0.0000)
```

---

## 7. RAG Benchmark Suite (`scripts/run_rag_benchmark.py`)

Executes an exhaustive Information Retrieval (IR) and generation evaluation against gold-standard evaluation datasets.

### Metrics Computed
- **Hit Rate @ k:** Percentage of queries where at least one ground-truth document was retrieved.
- **Precision @ k & Recall @ k:** Exactness and completeness of retrieved context.
- **Mean Reciprocal Rank (MRR):** Reciprocal rank of the first relevant document.
- **Normalized Discounted Cumulative Gain (nDCG @ k):** Graded relevance ranking quality.
- **Mean Average Precision (MAP @ k):** Ranking precision across multiple relevant chunks.
- **Noise Rate:** Ratio of irrelevant chunks included in context.
- **Context Relevance Score:** Semantic alignment between query and retrieved chunks.

### Syntax
```bash
python scripts/run_rag_benchmark.py
```

---

## 8. Automated Accuracy & Regression Runner (`scripts/evaluate_accuracy.py`)

Evaluates answer accuracy and checks for performance regressions against a saved baseline.

### Syntax
```bash
# Evaluate accuracy on current project documents
python scripts/evaluate_accuracy.py

# Save current accuracy metrics as regression baseline
python scripts/evaluate_accuracy.py --set-baseline

# Compare current code against baseline (exits with code 1 if performance degrades > 5%)
python scripts/evaluate_accuracy.py --compare --threshold 0.05
```

---

## 9. JSON to PostgreSQL Migration Utility (`scripts/migrate_json_to_postgres.py`)

Migrates legacy file-based project directories and JSON records into PostgreSQL.

### Syntax
```bash
python scripts/migrate_json_to_postgres.py
```
- Reads existing projects from `data/projects/`.
- Inserts missing projects, documents, chunks, and chat history into PostgreSQL.
- Sanitizes null bytes (`\x00`) from raw text strings to prevent PostgreSQL driver errors.
