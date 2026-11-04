from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ----------------- Chat & Message Schemas -----------------

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user query or question for the RAG assistant")
    conversation_id: Optional[str] = Field(None, description="Optional conversation UUID to continue a dialogue")


class SourceCitation(BaseModel):
    filename: Optional[str] = None
    page_number: Optional[int] = 1
    type: Optional[str] = "text"
    content: Optional[str] = None
    summary: Optional[str] = None
    score: Optional[float] = 0.0
    retrieval_method: Optional[str] = "hybrid"


class ChatResponse(BaseModel):
    conversation_id: str
    project_id: str
    title: str
    answer: str
    sources: List[Dict[str, Any]] = []


class RenameChatRequest(BaseModel):
    title: str = Field(..., min_length=1, description="New title for the conversation")


class MessageItem(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None


class ConversationResponse(BaseModel):
    chat_id: str
    project_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[MessageItem] = []


class ConversationSummary(BaseModel):
    chat_id: str
    project_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


# ----------------- Project Schemas -----------------

class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Project workspace name")
    description: Optional[str] = Field("", description="Optional project description")


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str


# ----------------- Health & Telemetry Schemas -----------------

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    version: str = "1.0.0"
