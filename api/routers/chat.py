import uuid
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Path, Request, Depends, status

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
from core.limiter import limiter
from api.deps import get_current_active_user

router = APIRouter(prefix="/api/v1", tags=["Conversations & RAG Chat"])


@router.post(
    "/projects/{project_id}/chat",
    response_model=ChatResponse,
    summary="Ask a question grounded in project documents (ownership verified)",
    description="Executes hybrid retrieval (BM25 + ChromaDB + Cohere) and multi-turn generation. Rate limited to 15 queries per minute.",
)
@limiter.limit("15/minute")
def chat_with_project(
    request: Request,
    project_id: str = Path(..., description="Project workspace ID"),
    payload: ChatRequest = ...,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    cid = payload.conversation_id or str(uuid.uuid4())
    existing_chat = get_chat(project_id, cid, user_id=current_user["id"])

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

    # 4. Save dialogue turn into PostgreSQL (with JSON backup) scoped to current_user
    save_chat(
        project_id=project_id,
        chat_id=cid,
        title=title,
        messages=messages,
        user_id=current_user["id"],
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
    summary="List all conversations in a project (ownership verified)",
)
@limiter.limit("60/minute")
def list_project_conversations(
    request: Request,
    project_id: str = Path(..., description="Project workspace ID"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    chats = list_chats(project_id, user_id=current_user["id"])
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
    summary="Retrieve full conversation history (ownership verified)",
)
@limiter.limit("60/minute")
def get_conversation_details(
    request: Request,
    project_id: str = Path(..., description="Project workspace ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    chat = get_chat(project_id, conversation_id, user_id=current_user["id"])
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found in project '{project_id}'.",
        )
    return ConversationResponse(**chat)


@router.patch(
    "/projects/{project_id}/conversations/{conversation_id}",
    summary="Rename a conversation (ownership verified)",
)
@limiter.limit("30/minute")
def rename_conversation(
    request: Request,
    project_id: str = Path(..., description="Project workspace ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
    payload: RenameChatRequest = ...,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    success = rename_chat(project_id, conversation_id, payload.title, user_id=current_user["id"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return {"message": "Conversation renamed successfully.", "title": payload.title}


@router.delete(
    "/projects/{project_id}/conversations/{conversation_id}",
    summary="Delete a conversation (ownership verified)",
)
@limiter.limit("20/minute")
def remove_conversation(
    request: Request,
    project_id: str = Path(..., description="Project workspace ID"),
    conversation_id: str = Path(..., description="Conversation ID"),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
):
    project = get_project(project_id, owner_id=current_user["id"])
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with ID '{project_id}' not found.",
        )

    success = delete_chat(project_id, conversation_id, user_id=current_user["id"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return {"message": f"Conversation '{conversation_id}' deleted successfully."}
