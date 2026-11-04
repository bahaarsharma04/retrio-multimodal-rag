import uuid
from typing import List
from fastapi import APIRouter, HTTPException, Path, status

from api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ConversationSummary,
    RenameChatRequest,
)
from core.projects import get_project
from core.rag_chain import generate_rag_response
from core.chat_manager import (
    list_chats,
    get_chat,
    save_chat,
    rename_chat,
    delete_chat,
    generate_chat_title,
)

router = APIRouter(prefix="/api/v1", tags=["Conversations & RAG Chat"])


@router.post(
    "/projects/{project_id}/chat",
    response_model=ChatResponse,
    summary="Ask a question grounded in project documents",
    description="Executes hybrid retrieval (BM25 + ChromaDB + Cohere) and multi-turn generation, persisting dialogue to PostgreSQL.",
)
def chat_with_project(
    project_id: str = Path(..., description="Project workspace ID"),
    payload: ChatRequest = ...,
):
    project = get_project(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    cid = payload.conversation_id or str(uuid.uuid4())
    existing_chat = get_chat(project_id, cid)

    if existing_chat:
        messages = existing_chat.get("messages", [])
        title = existing_chat.get("title", "Untitled Chat")
    else:
        messages = []
        title = "New Conversation"

    # 1. Append user message
    messages.append({
        "role": "user",
        "content": payload.message,
    })

    # Auto-generate title if this is the first turn
    if title in ["New Conversation", "Untitled Chat"] or len(messages) == 1:
        title = generate_chat_title(messages)

    # 2. Generate grounded response with multi-turn history
    rag_result = generate_rag_response(
        project_id=project_id,
        query=payload.message,
        chat_history=messages[:-1],
    )
    answer = rag_result.get("answer", "")
    sources = rag_result.get("sources", [])

    # 3. Append assistant response
    messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

    # 4. Save dialogue turn into PostgreSQL (with JSON backup)
    save_chat(
        project_id=project_id,
        chat_id=cid,
        title=title,
        messages=messages,
    )

    return ChatResponse(
        conversation_id=cid,
        project_id=project_id,
        title=title,
        answer=answer,
        sources=sources,
    )


@router.get(
    "/projects/{project_id}/conversations",
    response_model=List[ConversationSummary],
    summary="List all conversations in a project",
)
def list_project_conversations(
    project_id: str = Path(..., description="Project workspace ID"),
):
    project = get_project(project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    chats = list_chats(project_id)
    summaries = []
    for c in chats:
        summaries.append(
            ConversationSummary(
                chat_id=c.get("chat_id", ""),
                project_id=project_id,
                title=c.get("title", "Untitled Chat"),
                created_at=c.get("created_at", ""),
                updated_at=c.get("updated_at", ""),
                message_count=len(c.get("messages", [])),
            )
        )
    return summaries


@router.get(
    "/projects/{project_id}/conversations/{conversation_id}",
    response_model=ConversationResponse,
    summary="Retrieve full conversation history",
)
def get_conversation_details(
    project_id: str = Path(..., description="Project workspace ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
):
    chat = get_chat(project_id, conversation_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found in project '{project_id}'.",
        )
    return ConversationResponse(**chat)


@router.patch(
    "/projects/{project_id}/conversations/{conversation_id}",
    summary="Rename a conversation",
)
def rename_conversation(
    project_id: str = Path(..., description="Project workspace ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
    payload: RenameChatRequest = ...,
):
    success = rename_chat(project_id, conversation_id, payload.title)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return {"message": "Conversation renamed successfully.", "title": payload.title}


@router.delete(
    "/projects/{project_id}/conversations/{conversation_id}",
    summary="Delete a conversation",
)
def remove_conversation(
    project_id: str = Path(..., description="Project workspace ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
):
    success = delete_chat(project_id, conversation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return {"message": f"Conversation '{conversation_id}' deleted successfully."}
