# Retrio: Multi-Modal Hybrid RAG System

A modular, production-grade Retrieval-Augmented Generation (RAG) platform featuring **Unstructured layout parsing**, **Multi-Vector Indexing**, **BM25 + Dense Hybrid Search**, and **Cohere Neural Reranking**.

---

## Architecture Overview

```
                                  [ Upload Documents ]
                               (PDF, DOCX, CSV, TXT, MD)
                                           │
                                           ▼
                            [ 1. Document Partitioning ]
                      (Unstructured layout analysis & OCR)
                                           │
                                           ▼
                             [ 2. Semantic Chunking ]
                        (By-Title boundary & table isolation)
                                           │
                                           ▼
                            [ 3. Multi-Modal Summaries ]
                      (GPT-4o-mini table & figure summarization)
                                           │
                                           ▼
                         [ 4. Vectorization & Hybrid Index ]
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          [ Dense Vectors ]                               [ Sparse Lexical ]
    (text-embedding-3-small + ChromaDB)                     (BM25 Okapi)
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                          [ 5. Reciprocal Rank Fusion (RRF) ]
                                           │
                                           ▼
                             [ 6. Cohere Rerank v3.5 ]
                                           │
                                           ▼
                       [ 7. Grounded Generation & Citations ]
                           (LangChain LCEL + gpt-4o-mini)
```

---

## Key Features

1. **Layout-Aware Document Parsing**:
   * Uses `unstructured` to parse complex document layouts across `.pdf`, `.docx`, `.csv`, and `.txt`.
   * Automatically isolates narrative text, tables, figures, headers, and metadata.

2. **Semantic / By-Title Chunking**:
   * Chunks documents by logical heading boundaries rather than naive character slicing.
   * Keeps tables and diagrams intact as dedicated multi-modal atomic units.

3. **Multi-Vector Representation**:
   * Generates natural language summaries for tables and figures using `gpt-4o-mini`.
   * Indexes the **summaries** for high vector similarity matching while retaining the **raw table/diagram structure** for response synthesis.

4. **Two-Stage Hybrid Retrieval & Reranking**:
   * **Dense Semantic Search**: OpenAI `text-embedding-3-small` in isolated ChromaDB collections.
   * **Sparse Lexical Search**: BM25 Okapi for precise acronyms, column headers, and keyword matching.
   * **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse rankings algorithmically.
   * **Neural Cross-Encoder**: Cohere Rerank (`rerank-v3.5`) scores top candidate chunks for maximum precision.

5. **Multi-Turn Conversational Memory**:
   * Retrio maintains dialogue history within each session to handle pronouns, context shifts, and follow-up inquiries.
   * Inline source citations with page numbers (`[attention.pdf, Page 2]`).

6. **Interactive 3-Panel Streamlit Interface**:
   * **Left Panel**: Workspace projects switch/create modals and conversation history with rename/delete actions.
   * **Center Panel**: Conversational interface with collapsible source telemetry.
   * **Right Panel**: Project knowledge base with 1-click 5-stage pipeline telemetry inspector.

---

## Project Structure

```
Rag application/
├── core/
│   ├── config.py           # Centralized configuration & hyper-parameters
│   ├── projects.py         # Workspace project management
│   ├── chat_manager.py     # Conversation persistence & auto-titling
│   ├── parser.py           # Unstructured document layout extraction
│   ├── chunker.py          # Semantic by-title chunking engine
│   ├── summarizer.py       # Multi-modal table & figure summarizer
│   ├── vectorstore.py      # ChromaDB client & vector operations
│   ├── retriever.py        # BM25 + ChromaDB Hybrid search + Cohere rerank
│   ├── rag_chain.py        # LangChain LCEL RAG chain with citations
│   └── pipeline.py         # End-to-end 5-stage ingestion coordinator
├── app.py                  # Streamlit 3-panel UI & pipeline inspector
├── requirements.txt        # Production dependencies
├── .env.example            # Environment variables template
└── .gitignore              # Git ignore rules (protects API keys & data)
```

---

## Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
git clone <your-repo-url>
cd "Rag application"

# Create virtual environment
python -m venv .venv
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy `.env.example` to `.env` and provide your keys:

```bash
OPENAI_API_KEY=your-openai-api-key-here
COHERE_API_KEY=your-cohere-api-key-here    # Optional: Free tier trial key
```

### 4. Run Application

```bash
streamlit run app.py
```

Access the UI at `http://localhost:8501`.
