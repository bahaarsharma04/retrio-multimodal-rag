import os
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from core.config import DEFAULT_LLM_MODEL, OPENAI_API_KEY
from core.retriever import HybridRetriever

RAG_SYSTEM_PROMPT = """You are Retrio, an advanced technical AI assistant answering questions using a multi-modal knowledge base.
You are provided with verified context snippets extracted from documents (including narrative text, tables, and figure descriptions) as well as the recent conversation history.

Follow these strict rules:
1. Ground your answers strictly in the provided context and conversation history. Do NOT make up facts or hallucinate.
2. When referencing specific figures, metrics, or table data, reference the exact table and page number.
3. Include inline source citations in brackets, for example: [attention.pdf, Page 2].
4. Maintain context across the conversation. If the user asks a follow-up question (e.g., "explain row 2" or "what about that model?"), use the conversation history to understand what they are referring to.
5. If the provided context does not contain enough information to answer the question, clearly state: "The uploaded documents do not contain sufficient information to answer this question."
6. Be concise, structured, and factual. Use bullet points or markdown tables when explaining complex technical concepts.
"""


def format_context_block(sources: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a clean, structured context string."""
    if not sources:
        return "No relevant context found in documents."

    formatted_parts = []
    for i, s in enumerate(sources):
        src_header = f"--- [Snippet {i+1} | Source: {s.get('filename', 'doc')} | Page: {s.get('page_number', 1)} | Type: {s.get('type', 'text').upper()}] ---"
        content = s.get("content", "").strip()
        summary = s.get("summary")
        if summary:
            snippet_body = f"AI Summary: {summary}\n\nContent:\n{content}"
        else:
            snippet_body = content

        formatted_parts.append(f"{src_header}\n{snippet_body}\n")

    return "\n".join(formatted_parts)


def format_chat_history_block(chat_history: Optional[List[Dict[str, Any]]]) -> str:
    """Formats recent conversation turns into a context block."""
    if not chat_history:
        return ""

    recent_turns = chat_history[-6:]  # Keep last 3 exchanges (6 messages)
    history_lines = []
    for msg in recent_turns:
        role = "User" if msg.get("role") == "user" else "Retrio (Assistant)"
        content = msg.get("content", "").strip()
        history_lines.append(f"{role}: {content}")

    return "\n".join(history_lines)


def generate_rag_response(
    project_id: str,
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 8,
    top_n: int = 4,
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline with multi-turn conversation memory:
    1. Hybrid Retrieval (BM25 + ChromaDB + RRF + Cohere)
    2. Context & Conversation History Formatting
    3. Grounded LLM Response Generation (gpt-4o-mini)
    """
    retriever = HybridRetriever(project_id)
    retrieved_sources = retriever.retrieve(query=query, top_k=top_k, top_n=top_n)

    formatted_context = format_context_block(retrieved_sources)
    formatted_history = format_chat_history_block(chat_history)

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key or "your-openai" in api_key:
        sources_summary = "\n".join([
            f"- [{s.get('type', 'text').upper()}] {s.get('filename')} (Page {s.get('page_number')}): {s.get('content', '')[:120]}..."
            for s in retrieved_sources
        ])
        mock_answer = (
            f"**[Offline Demo Mode - Please set OPENAI_API_KEY in .env for live generation]**\n\n"
            f"Successfully retrieved **{len(retrieved_sources)} relevant chunks** via Hybrid Search (BM25 + ChromaDB):\n\n"
            f"{sources_summary}\n\n"
            f"Add your OpenAI key in `.env` to enable live LLM generation!"
        )
        return {
            "query": query,
            "answer": mock_answer,
            "sources": retrieved_sources,
            "retrieval_method": "hybrid_offline_preview",
        }

    try:
        llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0.0,
            api_key=api_key,
        )

        user_prompt_parts = []
        if formatted_history:
            user_prompt_parts.append(f"Recent Conversation History:\n{formatted_history}\n")

        user_prompt_parts.append(f"Context from Knowledge Base:\n{formatted_context}\n")
        user_prompt_parts.append(f"Current User Question: {query}")

        full_user_prompt = "\n".join(user_prompt_parts)

        messages = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=full_user_prompt),
        ]

        response = llm.invoke(messages)
        answer_text = response.content.strip()

        return {
            "query": query,
            "answer": answer_text,
            "sources": retrieved_sources,
            "retrieval_method": retrieved_sources[0].get("retrieval_method", "hybrid_rrf") if retrieved_sources else "none",
        }
    except Exception as e:
        return {
            "query": query,
            "answer": f"Error generating answer: {str(e)}",
            "sources": retrieved_sources,
            "retrieval_method": "error",
        }
