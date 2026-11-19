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


# ----------------- User & Auth Schemas -----------------

class UserRegisterRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., min_length=6, max_length=128, description="User password (min 6 chars)")
    full_name: Optional[str] = Field("", max_length=100, description="User's display or full name")


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = ""
    role: str = "user"
    is_active: bool = True
    created_at: str = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ----------------- Project Schemas -----------------

class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Project workspace name")
    description: Optional[str] = Field("", description="Optional project description")


class ProjectResponse(BaseModel):
    id: str
    owner_id: Optional[str] = None
    owner_email: Optional[str] = None
    name: str
    description: str
    created_at: str


# ----------------- Health & Telemetry Schemas -----------------

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    version: str = "1.0.0"
