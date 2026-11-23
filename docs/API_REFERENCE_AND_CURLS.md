# REST API Reference & cURL Suite

This document provides complete documentation for the Retrio REST API, including endpoint specifications, header requirements, request payloads, response schemas, and copy-paste `curl` commands.

Base URL: `http://localhost:8000`

---

## 🔐 1. Authentication Endpoints

### 1.1 Register User
- **Method:** `POST`
- **Path:** `/api/v1/auth/register`
- **Auth Required:** No
- **Rate Limit:** 10 requests / minute per IP

#### Request Payload
```json
{
  "email": "engineer@company.com",
  "password": "SecurePassword123!",
  "full_name": "Senior Software Engineer"
}
```

#### cURL Command
```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "engineer@company.com",
    "password": "SecurePassword123!",
    "full_name": "Senior Software Engineer"
  }'
```

#### Response (`HTTP 201 Created`)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "706c46e2-60ed-42b8-a2cf-60a2153c2b6e",
    "email": "engineer@company.com",
    "full_name": "Senior Software Engineer",
    "role": "user",
    "created_at": 1795177771.2
  }
}
```

---

### 1.2 User Login
- **Method:** `POST`
- **Path:** `/api/v1/auth/login`
- **Auth Required:** No
- **Rate Limit:** 10 requests / minute per IP

#### Request Payload
```json
{
  "email": "engineer@company.com",
  "password": "SecurePassword123!"
}
```

#### cURL Command
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "engineer@company.com",
    "password": "SecurePassword123!"
  }'
```

#### Response (`HTTP 200 OK`)
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "706c46e2-60ed-42b8-a2cf-60a2153c2b6e",
    "email": "engineer@company.com",
    "full_name": "Senior Software Engineer",
    "role": "user",
    "created_at": 1795177771.2
  }
}
```

---

### 1.3 Get Current User Profile
- **Method:** `GET`
- **Path:** `/api/v1/auth/me`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X GET "http://localhost:8000/api/v1/auth/me" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
{
  "id": "706c46e2-60ed-42b8-a2cf-60a2153c2b6e",
  "email": "engineer@company.com",
  "full_name": "Senior Software Engineer",
  "role": "user",
  "created_at": 1795177771.2
}
```

---

## 📁 2. Project Workspace Endpoints

### 2.1 List User Projects
- **Method:** `GET`
- **Path:** `/api/v1/projects`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X GET "http://localhost:8000/api/v1/projects" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
[
  {
    "id": "0fa55d40-2859-494b-8537-fd423ded528b",
    "name": "Financial Audit 2024",
    "description": "Quarterly balance sheets and income statements",
    "owner_id": "706c46e2-60ed-42b8-a2cf-60a2153c2b6e",
    "created_at": 1795170000.0,
    "document_count": 3
  }
]
```

---

### 2.2 Create New Project
- **Method:** `POST`
- **Path:** `/api/v1/projects`
- **Auth Required:** Yes (`Bearer <token>`)

#### Request Payload
```json
{
  "name": "Acquisition Due Diligence",
  "description": "Technical and legal audit documentation"
}
```

#### cURL Command
```bash
curl -X POST "http://localhost:8000/api/v1/projects" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Acquisition Due Diligence",
    "description": "Technical and legal audit documentation"
  }'
```

#### Response (`HTTP 201 Created`)
```json
{
  "id": "8a3d1f04-9a42-4e89-b13c-78d10b7e21a4",
  "name": "Acquisition Due Diligence",
  "description": "Technical and legal audit documentation",
  "owner_id": "706c46e2-60ed-42b8-a2cf-60a2153c2b6e",
  "created_at": 1795180120.5,
  "document_count": 0
}
```

---

### 2.3 Get Project Details
- **Method:** `GET`
- **Path:** `/api/v1/projects/{project_id}`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X GET "http://localhost:8000/api/v1/projects/8a3d1f04-9a42-4e89-b13c-78d10b7e21a4" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
{
  "id": "8a3d1f04-9a42-4e89-b13c-78d10b7e21a4",
  "name": "Acquisition Due Diligence",
  "description": "Technical and legal audit documentation",
  "owner_id": "706c46e2-60ed-42b8-a2cf-60a2153c2b6e",
  "created_at": 1795180120.5,
  "document_count": 0
}
```

---

### 2.4 Delete Project
- **Method:** `DELETE`
- **Path:** `/api/v1/projects/{project_id}`
- **Auth Required:** Yes (`Bearer <token>`)
- **Note:** Cascades and deletes all documents, vectors, conversations, and messages associated with the project.

#### cURL Command
```bash
curl -X DELETE "http://localhost:8000/api/v1/projects/8a3d1f04-9a42-4e89-b13c-78d10b7e21a4" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
{
  "message": "Project '8a3d1f04-9a42-4e89-b13c-78d10b7e21a4' and all associated data deleted successfully."
}
```

---

## 📄 3. Document Ingestion & Management Endpoints

### 3.1 Upload Document (Asynchronous Background Task)
- **Method:** `POST`
- **Path:** `/api/v1/projects/{project_id}/documents`
- **Auth Required:** Yes (`Bearer <token>`)
- **Content-Type:** `multipart/form-data`
- **Behavior:** Saves file to disk, registers record in PostgreSQL with `status="processing"`, enqueues background ingestion pipeline, and returns `HTTP 202 Accepted` immediately.

#### cURL Command
```bash
curl -X POST "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/documents" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -F "file=@/path/to/balance_sheet_2024.pdf"
```

#### Response (`HTTP 202 Accepted`)
```json
{
  "message": "Document 'balance_sheet_2024.pdf' uploaded successfully. Ingestion running in background.",
  "filename": "balance_sheet_2024.pdf",
  "project_id": "0fa55d40-2859-494b-8537-fd423ded528b",
  "status": "processing",
  "document_id": "338021f1-efab-4977-8c06-1689ce05e908"
}
```

---

### 3.2 Poll Document Ingestion Status
- **Method:** `GET`
- **Path:** `/api/v1/projects/{project_id}/documents/{filename}/status`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X GET "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/documents/balance_sheet_2024.pdf/status" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response when Processing (`HTTP 200 OK`)
```json
{
  "filename": "balance_sheet_2024.pdf",
  "status": "processing",
  "total_chunks": 0,
  "completed_stages": ["partitioning"]
}
```

#### Response when Completed (`HTTP 200 OK`)
```json
{
  "filename": "balance_sheet_2024.pdf",
  "status": "completed",
  "total_chunks": 42,
  "completed_stages": ["partitioning", "chunking", "summarisation", "vectorization", "complete"]
}
```

---

### 3.3 Inspect Ingestion Pipeline Audit
- **Method:** `GET`
- **Path:** `/api/v1/projects/{project_id}/documents/{filename}/pipeline`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X GET "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/documents/balance_sheet_2024.pdf/pipeline" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
{
  "filename": "balance_sheet_2024.pdf",
  "project_id": "0fa55d40-2859-494b-8537-fd423ded528b",
  "status": "completed",
  "stages": {
    "partitioning": {"status": "completed", "duration_ms": 2150.4},
    "chunking": {"status": "completed", "chunks_created": 42},
    "summarisation": {"status": "completed", "tables_summarized": 3},
    "vectorization": {"status": "completed", "vectors_stored": 42}
  }
}
```

---

### 3.4 Delete Document
- **Method:** `DELETE`
- **Path:** `/api/v1/projects/{project_id}/documents/{filename}`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X DELETE "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/documents/balance_sheet_2024.pdf" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
{
  "message": "Document 'balance_sheet_2024.pdf' and its associated vectors deleted successfully."
}
```

---

## 💬 4. Conversational RAG & Chat Endpoints

### 4.1 Send Chat Query (Hybrid RAG)
- **Method:** `POST`
- **Path:** `/api/v1/projects/{project_id}/chat`
- **Auth Required:** Yes (`Bearer <token>`)

#### Request Payload
```json
{
  "message": "What were the total assets and liabilities in 2024?",
  "conversation_id": "4e49298d-a41e-4508-b80c-7d9b9776f821"
}
```
*(Note: If `conversation_id` is omitted or null, a new conversation session is created automatically).*

#### cURL Command
```bash
curl -X POST "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/chat" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What were the total assets and liabilities in 2024?",
    "conversation_id": "4e49298d-a41e-4508-b80c-7d9b9776f821"
  }'
```

#### Response (`HTTP 200 OK`)
```json
{
  "conversation_id": "4e49298d-a41e-4508-b80c-7d9b9776f821",
  "answer": "According to the Consolidated Balance Sheets, total assets were $352.5 billion in fiscal year 2024, compared to total liabilities of $290.4 billion [1].",
  "sources": [
    {
      "citation_id": 1,
      "document_name": "balance_sheet_2024.pdf",
      "section": "Consolidated Financial Position",
      "page_numbers": [3],
      "page_range": "3",
      "text": "Total assets stood at $352,583 million as of September 28, 2024, with total liabilities at $290,437 million...",
      "relevance_score": 0.9821
    }
  ],
  "search_metadata": {
    "query": "What were the total assets and liabilities in 2024?",
    "rewritten_query": "What were the total assets and liabilities for the company in fiscal year 2024?",
    "dense_count": 15,
    "sparse_count": 15,
    "fused_count": 24,
    "reranked_count": 4,
    "query_tokensize": 13,
    "retrieved_chunks_tokensize": 512
  }
}
```

---

### 4.2 List Project Conversations
- **Method:** `GET`
- **Path:** `/api/v1/projects/{project_id}/conversations`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X GET "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/conversations" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
[
  {
    "id": "4e49298d-a41e-4508-b80c-7d9b9776f821",
    "title": "Balance Sheet Assets Query",
    "project_id": "0fa55d40-2859-494b-8537-fd423ded528b",
    "created_at": 1795171000.0,
    "updated_at": 1795171250.0,
    "message_count": 4
  }
]
```

---

### 4.3 Rename Conversation Title
- **Method:** `PATCH`
- **Path:** `/api/v1/projects/{project_id}/conversations/{conversation_id}/title`
- **Auth Required:** Yes (`Bearer <token>`)

#### Request Payload
```json
{
  "title": "2024 Asset Review"
}
```

#### cURL Command
```bash
curl -X PATCH "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/conversations/4e49298d-a41e-4508-b80c-7d9b9776f821/title" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "2024 Asset Review"
  }'
```

#### Response (`HTTP 200 OK`)
```json
{
  "id": "4e49298d-a41e-4508-b80c-7d9b9776f821",
  "title": "2024 Asset Review",
  "message": "Conversation title updated successfully."
}
```

---

### 4.4 Delete Conversation
- **Method:** `DELETE`
- **Path:** `/api/v1/projects/{project_id}/conversations/{conversation_id}`
- **Auth Required:** Yes (`Bearer <token>`)

#### cURL Command
```bash
curl -X DELETE "http://localhost:8000/api/v1/projects/0fa55d40-2859-494b-8537-fd423ded528b/conversations/4e49298d-a41e-4508-b80c-7d9b9776f821" \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>"
```

#### Response (`HTTP 200 OK`)
```json
{
  "message": "Conversation '4e49298d-a41e-4508-b80c-7d9b9776f821' deleted successfully."
}
```
