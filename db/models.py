import uuid
from typing import Dict, Any
from sqlalchemy import (
    Column, String, BigInteger, Integer, Float, Text,
    DateTime, ForeignKey, JSON, Boolean, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    from sqlalchemy.types import UserDefinedType
    class Vector(UserDefinedType):
        def __init__(self, dim):
            self.dim = dim
        def get_col_spec(self, **kw):
            return f"vector({self.dim})"

from db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True, default="")
    role = Column(String(32), default="user", nullable=False)  # "user", "admin"
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name or "",
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.strftime("%d/%m/%Y %H:%M") if self.created_at else "",
        }


class Project(Base):
    __tablename__ = "projects"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    owner = relationship("User", back_populates="projects")
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="project", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.strftime("%d/%m/%Y") if self.created_at else "",
            "created_timestamp": self.created_at.timestamp() if self.created_at else 0.0,
        }


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(
        String(64),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    filename = Column(String(512), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    mime_type = Column(String(128), default="application/pdf")
    ingestion_status = Column(String(50), default="pending", index=True)
    total_pages = Column(Integer, default=1)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    project = relationship("Project", back_populates="documents")
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    pipeline_runs = relationship("PipelineRun", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "mime_type": self.mime_type,
            "ingestion_status": self.ingestion_status,
            "total_pages": self.total_pages,
            "metadata": self.metadata_ or {},
            "created_at": self.created_at.strftime("%d/%m/%Y %H:%M") if self.created_at else "",
        }


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        String(64),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String(50), default="started")
    total_chunks = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    document = relationship("Document", back_populates="pipeline_runs")
    stages = relationship("PipelineStage", back_populates="run", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "status": self.status,
            "total_chunks": self.total_chunks,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "started_at": self.started_at.strftime("%Y-%m-%d %H:%M:%S") if self.started_at else None,
            "finished_at": self.finished_at.strftime("%Y-%m-%d %H:%M:%S") if self.finished_at else None,
            "stages": [s.to_dict() for s in self.stages],
        }


class PipelineStage(Base):
    __tablename__ = "pipeline_stages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pipeline_run_id = Column(
        String(64),
        ForeignKey("pipeline_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    duration_ms = Column(Float, default=0.0)
    stats = Column(JSON, default=dict)
    checkpoint_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    run = relationship("PipelineRun", back_populates="stages")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pipeline_run_id": self.pipeline_run_id,
            "stage_name": self.stage_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "stats": self.stats or {},
        }


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(128), primary_key=True)  # e.g. chunk_80c150b567b293...
    document_id = Column(
        String(64),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id = Column(
        String(64),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(50), default="text", index=True)
    title = Column(String(255), default="General", index=True)
    section = Column(String(255), default="General", index=True)
    part_index = Column(Integer, default=1)
    total_parts = Column(Integer, default=1)
    page_numbers = Column(JSON, nullable=False, default=list)
    content = Column(Text, nullable=False)
    text_as_html = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    indexed_text = Column(Text, nullable=True)
    content_hash = Column(String(64), nullable=False)
    char_count = Column(Integer, default=0)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    document = relationship("Document", back_populates="chunks")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "title": self.title or "General",
            "section": self.section or "General",
            "part_index": self.part_index or 1,
            "total_parts": self.total_parts or 1,
            "page_numbers": self.page_numbers or [1],
            "content": self.content,
            "text_as_html": self.text_as_html,
            "summary": self.summary,
            "char_count": self.char_count,
        }


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True)  # UUID string e.g. chat_id
    project_id = Column(
        String(64),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        String(64),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title = Column(String(255), nullable=False, default="Untitled Chat")
    active_filters = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    project = relationship("Project", back_populates="conversations")
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_id": self.id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "title": self.title,
            "active_filters": self.active_filters or [],
            "created_at": self.created_at.strftime("%d/%m/%Y %H:%M") if self.created_at else "",
            "created_timestamp": self.created_at.timestamp() if self.created_at else 0.0,
            "updated_at": self.updated_at.strftime("%d/%m/%Y %H:%M") if self.updated_at else "",
            "updated_timestamp": self.updated_at.timestamp() if self.updated_at else 0.0,
            "messages": [m.to_dict() for m in self.messages],
        }


class Message(Base):
    __tablename__ = "messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)  # List of source citations/chunks
    rewritten_query = Column(Text, nullable=True)
    latency_ms = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
    citations = relationship("MessageCitation", back_populates="message", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        msg_dict = {
            "role": self.role,
            "content": self.content,
        }
        if self.sources:
            msg_dict["sources"] = self.sources
        if self.rewritten_query:
            msg_dict["rewritten_query"] = self.rewritten_query
        if self.latency_ms:
            msg_dict["latency_ms"] = self.latency_ms
        return msg_dict


class MessageCitation(Base):
    __tablename__ = "message_citations"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    message_id = Column(
        BigInteger,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id = Column(
        String(128),
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_id = Column(
        String(64),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    citation_index = Column(Integer, nullable=False)
    page_number = Column(Integer, default=1)
    section = Column(String(255), nullable=True)
    relevance_score = Column(Float, nullable=True)
    retrieval_method = Column(String(50), nullable=True)
    snippet_preview = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    message = relationship("Message", back_populates="citations")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_index": self.citation_index,
            "page_number": self.page_number,
            "section": self.section,
            "relevance_score": self.relevance_score,
            "retrieval_method": self.retrieval_method,
            "snippet_preview": self.snippet_preview,
        }
