# Retrio: Multi-Modal Hybrid RAG System

A modular, production-grade Retrieval-Augmented Generation (RAG) platform featuring **Format-Aware Canonical Pre-Structuring**, **Layout Analysis**, **Sentence-Bounded Overlap Chunking**, **PostgreSQL + pgvector Single Database Storage**, **BM25 + Dense Hybrid Search**, and **Cohere Neural Reranking**.

---

## Architecture Overview

```
                                      [ Upload Documents ]
                        (PDF, DOCX, CSV, TXT, MD, Invoices, Transcripts)
                                               │
                                               ▼
                              [ 1. Format-Aware Pre-Structuring ]
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
              [ Tabular/CSV ]            [ Transcripts ]         [ Receipts/Invoices ]
           (pandas row-batching       (Dialogue & timestamp     (Vendor, Total, Date
            + repeated headers)         turn normalization)      + line-item tables)
                    └──────────────────────────┬──────────────────────────┘
                                               │
                                               ▼
                                 [ 2. Document Partitioning ]
                           (Unstructured layout analysis & OCR)
                                               │
                                               ▼
                                 [ 3. Layout & Title Chunking ]
                        (Sentence-bounded splitting, 250-char overlap,
                       metadata & title propagation across all N parts)
                                               │
                                               ▼
                                [ 4. Multi-Modal Summaries ]
                         (GPT-4o-mini table & figure summarization)
                                               │
                                               ▼
                             [ 5. Vectorization & Storage ]
                                               │
                                               ▼
                              [ Unified Database: PostgreSQL ]
                         (pgvector 1536d Cosine Embeddings +
                          Relational Documents, Chunks, Metadata)
                                               │
                                               ▼
                               [ 6. Hybrid Retrieval & RRF ]
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼                                                     ▼
         [ Dense Vector Search ]                                 [ Sparse Lexical ]
      (PostgreSQL pgvector <=> )                                    (BM25 Okapi)
                    └──────────────────────────┬──────────────────────────┘
                                               │
                                               ▼
                                  [ 7. Cohere Rerank v3.5 ]
                                               │
                                               ▼
                            [ 8. Grounded Generation & Citations ]
                                (LangChain LCEL + gpt-4o-mini)
```

---

## Key Features

### 1. Format-Aware Pre-Structuring
Before chunking, raw files are classified and normalized into canonical structured elements:
* **Tabular / CSV (`.csv`):** Automatically batched into 20–30 row segments using `pandas`. **Table column headers are repeated on every chunk** to prevent loss of tabular schema context.
* **Dialogue Transcripts (`.txt`, `.vtt`):** Detects timestamped lines and speaker turns (`[00:01:23] Speaker:`), grouping dialogues into structured conversational turns.
* **Receipts & Invoices:** Extracts financial entities (`Vendor`, `Invoice #`, `Date`, `Total Amount`) into a standardized summary header prepended to the line items table.
* **Unstructured Plain Text:** Synthesizes a clean document title element at the top if no headings exist, anchoring the entire document.

### 2. Title & Metadata Propagation Engine
* **No Orphaned Sub-Chunks:** When a document section or continuous paragraph is split into $N$ parts, **every single sub-chunk (from Part 1 to $N$) carries `title`, `section`, `part_index`, and `total_parts` in its metadata**.
* **Sentence-Bounded Splitting with Overlap:** Large text blocks exceeding 1,500 characters are subdivided along complete sentence boundaries (`[.!?]\s+`) with a **250-character trailing sentence overlap** to maintain conceptual continuity across chunk boundaries.

### 3. Unified Single Database: PostgreSQL (`pgvector`)
* **Single Database Architecture:** Unifies relational metadata (`projects`, `documents`, `conversations`, `messages`) and dense vectors (`document_chunks` with 1536-dimensional native `vector` column) inside PostgreSQL.
* **Native Cosine Similarity (`<=>`):** Dense vector distance is computed directly in PostgreSQL using `pgvector` HNSW/IVFFlat indexes, eliminating separate external vector services or local ChromaDB databases.
* **Strict Fail-Fast Single Source of Truth:** Direct connection to PostgreSQL is required. No fallbacks or duplicate vector silos.

### 4. Two-Stage Hybrid Retrieval & Neural Reranking
* **Dense Semantic Search:** OpenAI `text-embedding-3-small` vectors searched via native PostgreSQL `pgvector` cosine similarity (`<=>`).
* **Sparse Lexical Search:** BM25 Okapi for precise acronyms, numerical values, and keyword matching.
* **Reciprocal Rank Fusion (RRF):** Algorithmically merges dense semantic and sparse lexical score distributions.
* **Cohere Rerank (`rerank-v3.5`):** Cross-encoder reranks top retrieved candidates to maximize precision before LLM generation.

### 5. Multi-Turn Conversational Memory & Citations
* Full conversation management with automatic chat title generation.
* Grounded generation with inline, verified source citations and page numbers (`[document.pdf, Page 2]`).

---

## Project Structure

```
Rag application/
├── core/
│   ├── config.py           # Centralized environment & hyper-parameters
│   ├── projects.py         # Workspace project management
│   ├── chat_manager.py     # Conversation history & auto-titling
│   ├── parser.py           # Layout analysis & format pre-structuring (CSV, Transcripts, Invoices)
│   ├── chunker.py          # Sentence-bounded chunker with title/metadata propagation
│   ├── summarizer.py       # Multi-modal table & figure summarizer (GPT-4o-mini)
│   ├── vectorstore.py      # PostgreSQL pgvector & vector operations engine
│   ├── retriever.py        # BM25 + Dense Hybrid search + Cohere Rerank
│   ├── rag_chain.py        # LangChain LCEL RAG chain with citations
│   └── pipeline.py         # 5-stage ingestion coordinator with DB sync
├── db/
│   ├── models.py           # SQLAlchemy ORM models (Project, Document, DocumentChunk, Chat)
│   ├── session.py          # PostgreSQL session management & extension bootstrap
│   └── alembic/            # Database schema migration scripts
├── scripts/
│   ├── chunk_document.py               # Standalone layout & semantic chunking CLI
│   ├── embed_document.py               # Standalone embedding & pgvector indexing CLI
│   ├── retrieve_context.py             # Standalone hybrid/dense/BM25 retrieval & rerank CLI
│   ├── test_db.py                      # PostgreSQL & pgvector verification suite
│   ├── verify_formats_and_metadata.py  # End-to-end format & metadata test suite
│   └── migrate_json_to_postgres.py     # Data migration from local JSON to PostgreSQL
├── docker-compose.yml      # PostgreSQL 16 + pgvector container configuration
├── app.py                  # Streamlit 3-panel UI & pipeline inspector
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # Project documentation
```

---

## Quickstart

### 1. Set Up Environment
```powershell
# Clone the repository
git clone <your-repo-url>
cd "Rag application"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:
```env
OPENAI_API_KEY=your-openai-api-key-here
COHERE_API_KEY=your-cohere-api-key-here       # Optional: Free tier trial key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/retrio_db
```

### 3. Start Database (Docker)
Launch PostgreSQL with `pgvector`:
```powershell
docker compose up -d
```

### 4. Verification & Automated Testing Suite
Run security checks and model accuracy evaluations after every code change:
```powershell
# 1. Security, JWT & RBAC Verification (Multi-tenancy, Admin Super-Visibility, Rate Limiting)
.venv\Scripts\python.exe scripts\test_security_rbac.py

# 2. Automated Model Accuracy & Regression Benchmark:
# Quick smoke test (~25s):
.venv\Scripts\python.exe scripts\evaluate_accuracy.py --quick

# Compare against reference baseline (fails with exit code 1 if accuracy regressed):
.venv\Scripts\python.exe scripts\evaluate_accuracy.py --compare

# Set/update gold standard baseline after verified improvements:
.venv\Scripts\python.exe scripts\evaluate_accuracy.py --set-baseline

# Verify PostgreSQL & pgvector connection and vector search
.venv\Scripts\python.exe scripts\test_db.py

# Verify format pre-structuring (CSV, Transcripts, Invoices, Unstructured text)
.venv\Scripts\python.exe scripts\verify_formats_and_metadata.py
```

### 5. Standalone Executable CLI Utilities
You can independently run the ingestion stages (chunking, embedding, retrieval) from the command line:

```powershell
# 1. Chunking: Parse and chunk any document layout
.venv\Scripts\python.exe scripts\chunk_document.py "path/to/document.pdf" --preview
# Optional: save to JSON for decoupled pipelines
.venv\Scripts\python.exe scripts\chunk_document.py "path/to/document.pdf" --output-json chunks.json

# 2. Embedding: Generate embeddings & index into PostgreSQL pgvector
# Directly from file:
.venv\Scripts\python.exe scripts\embed_document.py --file "path/to/document.pdf" --project-id <project_id>
# Or from pre-generated chunks JSON:
.venv\Scripts\python.exe scripts\embed_document.py --chunks-json chunks.json --project-id <project_id>

# 3. Retrieval: Query indexed vectors with hybrid search (pgvector + BM25 + Cohere rerank)
.venv\Scripts\python.exe scripts\retrieve_context.py --project-id <project_id> --query "your question"
# Options: --top-k 10 --top-n 3 --mode hybrid|dense-only|bm25-only --output-json results.json
```

### 6. Launch Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
