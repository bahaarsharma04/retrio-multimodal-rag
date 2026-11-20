# System Workflow & Architecture

This document describes the runtime execution flows and architectural design of the Retrio RAG platform.

---

## 1. High-Level Architecture Overview

Retrio is organized into a modular layered architecture separating presentation, API routing, RAG orchestration, core processing pipelines, and native PostgreSQL storage:

```mermaid
graph TD
    User([Client / Streamlit UI / REST API]) -->|HTTP REST + Bearer JWT| FastAPIGateway[FastAPI Application Gateway\napi/main.py]
    
    subgraph Security & Access
        FastAPIGateway --> Limiter[SlowAPI Rate Limiter\ncore/limiter.py]
        FastAPIGateway --> AuthGuard[JWT Auth & RBAC\ncore/security.py & api/deps.py]
    end

    subgraph API Routers
        AuthGuard --> AuthRouter[api/routers/auth.py]
        AuthGuard --> ProjRouter[api/routers/projects.py]
        AuthGuard --> DocRouter[api/routers/documents.py]
        AuthGuard --> ChatRouter[api/routers/chat.py]
    end

    subgraph Ingestion Subsystem
        DocRouter -->|HTTP 202 Accepted| BGWorker[FastAPI BackgroundTasks\ncore/pipeline.py]
        BGWorker --> Stage1[Stage 1: Partitioning\ncore/parser.py]
        Stage1 --> Stage2[Stage 2: Chunking\ncore/chunker.py]
        Stage2 --> Stage3[Stage 3: Summarization\ncore/summarizer.py]
        Stage3 --> Stage4[Stage 4: Vectorization\ncore/vectorstore.py]
        Stage4 --> Stage5[Stage 5: Verification & Commit\ncore/pipeline.py]
    end

    subgraph Retrieval & Generation Subsystem
        ChatRouter --> RAGChain[RAG Chain Controller\ncore/rag_chain.py]
        RAGChain --> GreetingCheck{Is Greeting?\n'hi', 'hello', etc.}
        GreetingCheck -->|Yes| GreetingResponse[Instant Conversational Reply\nZero Retrieval Overhead]
        GreetingCheck -->|No| Rewriter[Contextual Query Rewriter\nOpenAI gpt-4o-mini]
        Rewriter --> Retriever[HybridRetriever\ncore/retriever.py]
        
        subgraph Hybrid Search Engine
            Retriever --> DenseSearch[Dense Vector Search\npgvector Cosine Distance]
            Retriever --> SparseSearch[Sparse Lexical Search\nBM25Okapi]
            DenseSearch --> RRF[Reciprocal Rank Fusion\nk=60]
            SparseSearch --> RRF
            RRF --> CohereRerank[Cohere Neural Reranker\nrerank-v3.5]
        end

        CohereRerank --> LLMGen[Grounded Answer Generation\nOpenAI gpt-4o-mini]
        LLMGen --> Citations[Citation Linker & Deduplicator]
    end

    subgraph Database Layer
        Stage5 --> Postgres[(PostgreSQL 16 + pgvector\ndb/models.py)]
        DenseSearch --> Postgres
        ChatRouter --> Postgres
        Citations --> Postgres
    end
```

---

## 2. Document Ingestion Pipeline Workflow

Document ingestion transforms raw unstructured files (PDF, DOCX, CSV, TXT, MD) into searchable semantic vectors and database records. Ingestion is executed asynchronously via FastAPI's `BackgroundTasks`.

```mermaid
flowchart TD
    A[Client Uploads File\nmultipart/form-data] --> B[Save File to Disk\ndata/projects/{id}/documents/]
    B --> C[Register Document in PostgreSQL\nstatus='processing']
    C --> D[Return HTTP 202 Accepted\nPayload with document_id and filename]
    
    subgraph Background Pipeline Task
        D -.-> E[Stage 1: Partition Document\ncore/parser.py]
        E -->|Raw Unstructured Elements| F[Stage 2: Semantic Title-Aware Chunking\ncore/chunker.py]
        F -->|Composite Chunks| G[Stage 3: Multimodal Summarisation\ncore/summarizer.py]
        G -->|Enriched Chunks with Table/Image Summaries| H[Stage 4: Vectorization & Storage\ncore/vectorstore.py]
        H -->|OpenAI text-embedding-3-small| I[Store Chunks & 1536-dim Vectors in PostgreSQL\ndocument_chunks table]
        I --> J[Stage 5: Commit & Record Pipeline State\ncore/pipeline.py]
        J --> K[Update Document Status in PostgreSQL\nstatus='completed']
    end
```

### Ingestion Stages Detailed

1. **Stage 1 — Partitioning (`core/parser.py`):**
   - Automatically detects format:
     - **PDF:** Utilizes `unstructured.partition.pdf.partition_pdf` with `strategy="fast"` (or `"hi_res"` with PDF image extraction fallback) and PyPDF fallback.
     - **DOCX / DOC:** Extracts paragraphs, formatted runs, and native document tables.
     - **CSV:** Converts tabular data into clean Markdown format for text embedding.
     - **TXT / MD:** Preserves plain text headers and code blocks.
   - Outputs: A stream of normalized `UnstructuredElement` dicts tagged with element types (`Title`, `NarrativeText`, `Table`, `Image`, `ListItem`).

2. **Stage 2 — Semantic Title-Aware Chunking (`core/chunker.py`):**
   - Implements semantic section-boundary chunking rather than naive character slicing.
   - When a `Title` or header element is encountered, a new chunk boundary is established, and the heading text is propagated down to all subordinate narrative chunks as section metadata.
   - Tables and images are treated as discrete atomic chunks to preserve structural integrity.
   - Outputs: Clean chunk objects containing `id`, `section`, `page_numbers`, `content`, `text_as_html`, and `metadata`.

3. **Stage 3 — Multimodal LLM Summarisation (`core/summarizer.py`):**
   - Inspects chunk types:
     - For `table` chunks: If HTML representation exists, prompts `gpt-4o-mini` to generate a concise semantic summary describing rows, columns, metrics, and relationships.
     - For `image` chunks: Encodes image bytes to Base64 and invokes `gpt-4o-mini` vision capabilities to describe figures, charts, and diagrams.
   - The generated summary is prepended to the chunk text or indexed as search context, enabling text search to retrieve visual and tabular data.

4. **Stage 4 — Vectorization & Database Insertion (`core/vectorstore.py`):**
   - Chunks are vectorized using OpenAI's `text-embedding-3-small` (1536 dimensions).
   - Generates deterministic UUID5 hashes based on chunk content and section to prevent duplicate insertions upon re-ingestion.
   - Inserts records directly into PostgreSQL's `document_chunks` table utilizing the `pgvector` vector extension.

5. **Stage 5 — Verification & State Transition (`core/pipeline.py`):**
   - Verifies the chunk count and vector count in the database against the chunker output.
   - Atomically updates `documents.status` from `'processing'` to `'completed'`.
   - Writes an audit checkpoint to `data/projects/{id}/documents/{filename}.pipeline.json`.

---

## 3. Background Ingestion & Status Polling Lifecycle

Because multi-page PDF parsing and vision summarisation may take 5–30 seconds, file uploads must never block HTTP threads.

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant API as FastAPI Gateway (api/routers/documents.py)
    participant Worker as Background Task (core/pipeline.py)
    participant DB as PostgreSQL (db/models.py)

    Client->>API: POST /api/v1/projects/{id}/documents (file upload)
    API->>API: Save file to disk
    API->>DB: INSERT INTO documents (status='processing')
    API->>Worker: Enqueue run_ingestion_pipeline()
    API-->>Client: HTTP 202 Accepted {"status": "processing", "document_id": "..."}

    par Asynchronous Processing
        Worker->>Worker: Execute Stages 1-4 (Partition, Chunk, Summarize, Embed)
        Worker->>DB: INSERT INTO document_chunks (pgvector vectors)
        Worker->>DB: UPDATE documents SET status='completed', chunk_count=N
    and Client Polling
        loop Every 1-2 Seconds
            Client->>API: GET /api/v1/projects/{id}/documents/{filename}/status
            API->>DB: SELECT status, chunk_count FROM documents WHERE filename=...
            DB-->>API: status='processing' (or 'completed')
            API-->>Client: HTTP 200 {"status": "processing"}
        end
    end

    Client->>API: GET /api/v1/projects/{id}/documents/{filename}/status
    API-->>Client: HTTP 200 {"status": "completed", "total_chunks": 42}
```

---

## 4. Retrieval & Generation (RAG) Architecture

The query answering pipeline employs a multi-tiered architecture ensuring high recall, exact keyword precision, and grounded synthesis.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Router as Chat Router (api/routers/chat.py)
    participant Chain as RAG Chain (core/rag_chain.py)
    participant Rewriter as LLM Query Rewriter (gpt-4o-mini)
    participant Retriever as HybridRetriever (core/retriever.py)
    participant PG as PostgreSQL (pgvector)
    participant Cohere as Cohere Rerank API (rerank-v3.5)
    participant Generator as LLM Generator (gpt-4o-mini)

    User->>Router: POST /api/v1/projects/{id}/chat {"message": "what were the 2024 revenue figures?"}
    Router->>Chain: generate_rag_response(query, project_id, history)
    
    Chain->>Chain: is_greeting_query(query)?
    Note over Chain: Query is domain-specific, not a greeting

    alt Multi-turn Conversation
        Chain->>Rewriter: Rewrite query using conversation history
        Rewriter-->>Chain: Standalone query: "What were the Company revenue figures in fiscal year 2024?"
    end

    Chain->>Retriever: retrieve(standalone_query, top_k=15, top_n=4)
    
    par Parallel Retrieval
        Retriever->>PG: Dense vector search (text-embedding-3-small <=> embedding)
        PG-->>Retriever: Top 15 dense matches
    and
        Retriever->>Retriever: Sparse BM25 lexical search over project chunks
        Retriever-->>Retriever: Top 15 BM25 matches
    end

    Retriever->>Retriever: Reciprocal Rank Fusion (RRF, k=60)
    Note over Retriever: Combines dense & sparse candidates into unified ranking

    Retriever->>Cohere: Rerank fused candidates (query, docs, top_n=4)
    Cohere-->>Retriever: Top 4 reranked chunks with relevance scores

    Retriever-->>Chain: 4 high-relevance chunks with metadata & tokensize
    Chain->>Generator: Synthesize grounded answer with citations
    Generator-->>Chain: Grounded response text + bracketed references [1]
    Chain->>Router: Response payload (answer, sources, tokensize, search_metadata)
    Router->>PG: Persist User & Assistant messages to messages table
    Router-->>User: HTTP 200 JSON Response
```

### Retrieval Components Detailed

1. **Meta-Greeting Interception (`core/rag_chain.py`):**
   - Matches conversational pleasantries (`"hi"`, `"hello"`, `"hey"`, `"good morning"`, `"thanks"`) using regex guardrails.
   - Checks if project has documents:
     - **With documents:** Welcomes the user and states that documents are loaded and ready for questions.
     - **Without documents:** Prompts the user to upload documents first.
   - Bypasses DB retrieval entirely, saving API latency and token cost.

2. **Conversational Query Rewriting (`core/rag_chain.py`):**
   - Resolves ambiguous conversational pronouns (e.g. *"What about its fourth quarter?"* $\rightarrow$ *"What were Apple's financial results in fourth quarter 2024?"*).
   - Prevents multi-turn degradation where retrieval fails due to unreferenced antecedents.

3. **Hybrid Search & Reciprocal Rank Fusion (RRF) (`core/retriever.py`):**
   - **Dense Retrieval:** Calculates Cosine Distance using `pgvector` (`DocumentChunk.embedding.cosine_distance(query_vector)`).
   - **Sparse Retrieval:** Executes `BM25Okapi` over in-memory tokenized project chunks to capture exact product codes, acronyms, and numeric figures.
   - **RRF Equation:**
     $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$

4. **Neural Cross-Encoder Re-ranking (`Cohere rerank-v3.5`):**
   - Re-evaluates top RRF candidates using a cross-attention transformer model.
   - Returns top $N$ (default: 4) context passages with calibrated relevance scores (0.0 to 1.0).

5. **Grounded Generation & Citation Linking (`core/rag_chain.py`):**
   - Uses strict system prompts instructing `gpt-4o-mini` to answer **only** using provided context.
   - Injects structured source markers (filename, section, page numbers) to produce verifiable bracketed citations `[1]`, `[2]`.
