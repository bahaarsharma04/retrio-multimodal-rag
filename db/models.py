from sqlalchemy import Column, String, BigInteger, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.base import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True)  # UUID string e.g. chat_id
    project_id = Column(String(64), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="Untitled Chat")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def to_dict(self):
        return {
            "chat_id": self.id,
            "project_id": self.project_id,
            "title": self.title,
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
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    conversation = relationship("Conversation", back_populates="messages")

    def to_dict(self):
        msg_dict = {
            "role": self.role,
            "content": self.content,
        }
        if self.sources:
            msg_dict["sources"] = self.sources
        return msg_dict
