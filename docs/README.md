# Retrio: Enterprise Multimodal Hybrid RAG Platform
## Engineering Handover & System Documentation

Welcome to the technical handover documentation for **Retrio**, an enterprise-grade Multimodal Retrieval-Augmented Generation (RAG) platform designed for complex technical documents, balance sheets, academic research papers, and corporate repositories.

This documentation suite provides a complete architectural blueprint, detailed inter-module data flow diagrams, CLI script manuals, and API references for incoming engineers and maintainers.

---

## 📚 Handover Documentation Map

The handover documentation is organized into modular specifications located within the `docs/` directory:

| Document | Purpose & Contents |
| :--- | :--- |
| **[Workflow & Architecture](file:///c:/Users/Dell/Desktop/Rag%20application/docs/WORKFLOW_AND_ARCHITECTURE.md)** | End-to-end architecture, 5-stage ingestion pipeline, background task execution, hybrid retrieval, Cohere neural reranking, and greeting interceptor. |
| **[File Structure & Data Flow](file:///c:/Users/Dell/Desktop/Rag%20application/docs/FILE_STRUCTURE_AND_DATA_FLOW.md)** | Full codebase directory tree, component responsibilities, and exact data schemas/contracts passed between every file. |
| **[CLI Scripts Reference](file:///c:/Users/Dell/Desktop/Rag%20application/docs/CLI_SCRIPTS_REFERENCE.md)** | Manual for all standalone CLI utilities (`chunk_document.py`, `embed_document.py`, `retrieve_context.py`), evaluation suites, and database tools with CLI flags and examples. |
| **[API Reference & cURL Suite](file:///c:/Users/Dell/Desktop/Rag%20application/docs/API_REFERENCE_AND_CURLS.md)** | Complete REST API endpoint reference covering Authentication, Projects, Documents (Async Ingestion & Polling), and Chat, complete with copy-paste `curl` commands and response payloads. |

---

## 🛠️ Technology Stack & Core Infrastructure

```
┌────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND                                  │
│         Streamlit Multi-Page UI (app.py) & REST Client                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST API (Port 8000)
┌───────────────────────────────────▼────────────────────────────────────┐
│                        API GATEWAY (FastAPI)                           │
│   • SlowAPI Rate Limiting (120/min authenticated, 10/min unauth)       │
│   • JWT Authentication (HS256) & Strict Multi-Tenant RBAC              │
│   • FastAPI BackgroundTasks for Non-Blocking Document Ingestion         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│     DOCUMENT INGESTION       │        │     RETRIEVAL & GENERATION   │
│ • Unstructured.io Partition  │        │ • Meta-Greeting Interceptor  │
│ • Title-Aware Chunking       │        │ • Contextual Query Rewriter  │
│ • GPT-4o-mini Vision Summary │        │ • Hybrid Search (Dense+BM25) │
│ • text-embedding-3-small     │        │ • Reciprocal Rank Fusion(RRF)│
│ • pgvector 1536-dim Vectors  │        │ • Cohere Rerank v3.5         │
└──────────────┬───────────────┘        │ • OpenAI GPT-4o-mini Gen     │
               │                        └──────────────┬───────────────┘
               ▼                                       ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   PERSISTENCE LAYER (PostgreSQL 16)                    │
│   • Extensions: pgvector, pgcrypto, uuid-ossp                          │
│   • SQLAlchemy 2.0 ORM + Alembic Migrations                            │
│   • Tables: users, projects, documents, document_chunks, conversations,│
│             messages, message_citations, pipeline_runs, stages         │
└────────────────────────────────────────────────────────────────────────┘
```

- **Backend Runtime:** Python 3.11+
- **API Framework:** FastAPI, Uvicorn, Pydantic v2
- **Vector Storage:** PostgreSQL 16 with `pgvector` extension (single source of truth; zero third-party vector store dependencies)
- **Relational Storage:** PostgreSQL (Users, Projects, Documents, Chunks, Chat Conversations, Messages, Citations)
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions, normalized cosine similarity)
- **Sparse Retrieval:** BM25 (`rank-bm25` with domain tokenization)
- **Re-ranking:** Cohere Rerank (`rerank-v3.5`)
- **Generation LLM:** OpenAI `gpt-4o-mini` (temperature: 0.1, strict grounding)
- **Document Processing:** Unstructured (`unstructured[all-docs]`), PyPDF, pdf2image, python-docx
- **Security:** Passwords hashed with `bcrypt` (12 rounds); Stateless JWT tokens (`HS256`); SlowAPI client rate limiting.

---

## ⚡ Quickstart Guide

### 1. Environment Configuration
Ensure your `.env` file in the project root defines the required database credentials and API keys:

```env
# Application Core
PROJECT_ROOT=.
DATA_DIR=data
PROJECTS_DIR=data/projects

# PostgreSQL Single Source of Truth
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/retrio_db

# External AI Services
OPENAI_API_KEY=sk-proj-...
COHERE_API_KEY=...

# Security & Authentication
JWT_SECRET_KEY=retrio-super-secret-jwt-key-production-change-me-32chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# RAG Hyperparameters
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
RAG_PRIMARY_MODEL=gpt-4o-mini
HYBRID_SEARCH_TOP_K=15
RERANK_TOP_N=4
```

### 2. Verify Database & Dependencies
Run the database verification suite to confirm PostgreSQL connectivity and table health:

```bash
python scripts/test_db.py
```

### 3. Launch the Backend Server
Start the FastAPI server on port 8000:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI Swagger docs will be accessible at: `http://localhost:8000/docs`

### 4. Launch the Frontend UI (Optional)
Start the Streamlit interface on port 8501:

```bash
streamlit run app.py
```

---

## 🔒 Key System Guarantees & Constraints

1. **Fail-Fast PostgreSQL (pgvector) Single Source of Truth:**
   All document chunks, vectors, metadata, chat history, and users are stored natively in PostgreSQL. There is zero reliance on ChromaDB or local fallback vector directories. If PostgreSQL is offline, requests fail fast with explicit diagnostic error responses.
2. **Asynchronous Background Document Ingestion:**
   Document uploads (`POST /api/v1/projects/{id}/documents`) return HTTP `202 Accepted` within milliseconds. Heavy document parsing, vision summarisation, and vectorization run in background tasks. Clients poll `GET /documents/{filename}/status` for atomic state transitions (`processing` $\rightarrow$ `completed` / `failed`).
3. **Conversational Greeting Interception:**
   Casual greetings ("hi", "hello", "good morning") bypass expensive RAG retrieval pipelines, returning an immediate welcoming response without hallucinating or falsely reporting "uploaded document does not contain sufficient info".
4. **Strict Multi-Tenant Isolation:**
   Every database query filters by `owner_id` (projects) and `project_id` (documents/chats). Cross-tenant queries are blocked at the ORM layer and return `404 Not Found` to prevent information enumeration.
