import os
import re
import time
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from core.config import (
    DEFAULT_LLM_MODEL,
    OPENAI_API_KEY,
    HYBRID_SEARCH_TOP_K,
    RERANK_TOP_N,
)
from core.retriever import HybridRetriever
from core.logger import generation_logger

RAG_SYSTEM_PROMPT = """You are Retrio, an advanced technical AI assistant providing accurate, strictly document-grounded answers using a multi-modal knowledge base.
You are provided with verified context snippets extracted from documents (including narrative text, tables, and figure descriptions) as well as the recent conversation history.

Follow these universal rules for all uploaded documents:
1. Strict Grounding & No Unsupported Inferences: Base your answers ONLY on the retrieved document context. Do NOT invent missing details, durations, speed rates, or mathematical relationships beyond what is explicitly stated in the provided text.
   - If a specific attribute, timeframe, or measurement is NOT stated in the document, explicitly state: "The document does not specify [attribute]." Do not guess or infer it.
2. No Unverified External Knowledge: When asked what a document says, answer strictly from the source text and do NOT inject unmentioned external scientific, financial, or historical facts.
3. Distinguish Empirical Findings vs. Theoretical Models vs. Robustness Checks:
   - Check section headers and context labels (e.g., Empirical Results vs. Theoretical Model / Simulation vs. Robustness Checks vs. Appendix).
   - Explicitly identify the nature of reported data:
     - Differentiate observed empirical/econometric/experimental estimates from calibrated theoretical simulations or mathematical modeling results.
     - Never cite a theoretical simulation output or counterfactual scenario as an observed empirical real-world fact.
4. Exact Numerical Thresholds, Ranges & Classifications:
   - Carefully preserve inequality signs (<, >, ≤, ≥), units, and percentage ranges.
   - When a document defines category thresholds (e.g., lower tail vs. upper tail vs. intermediate range), preserve all boundaries and intermediate definitions. Never assign an intermediate value to an extreme category unless explicitly classified as such by the text.
5. Causal Mechanisms & Explanatory Sequence:
   - When explaining technical, economic, or scientific mechanisms, preserve the exact chronological sequence: Initial Cause -> Transmission Mechanism -> Direct Response -> Downstream Consequence.
   - Do NOT invert cause and consequence (e.g., do not explain an initial trigger by citing a later downstream side-effect).
6. Distinguish Proven Findings from Possibilities or Partial Offsets:
   - Clearly distinguish what the source explicitly demonstrates from what remains unproven, ambiguous, or only partially offsetting.
7. Compound Attribution:
   - When effects, risks, or properties are attributed to multiple phenomena together, attribute those consequences explicitly and equally to all mentioned causes.
8. Figures, Images, and Visual Explanations:
   - Rely strictly on figure captions, callouts, and diagrams provided in context. Accurately reflect the specific entities, labels, components, or relationships shown.
9. Table Integrity & Arithmetic Accuracy:
   - Carefully preserve row-and-column alignments and metric units.
   - Pay strict attention to table headers, footnotes, and conditional qualifiers like "Of these" (indicating percentages of a sub-group, not the total sample).
   - Verify all calculations, percentage-point differences, and comparative totals before answering.
10. Complete Enumeration:
   - When questions ask for lists, components, or specific counts, include every item explicitly stated in the context without truncating or substituting with vague generalizations.
11. Methodological Precision:
   - Clearly distinguish theoretical definitions, assumptions, or criteria from the statistical or empirical tests used to evaluate them.
12. Inline Citations:
   - Always cite the source document and page number in brackets, for example: [DocumentName.pdf, Page X].
13. Insufficient Information:
   - If the retrieved context does not contain enough evidence to answer a question or verify a claim, clearly state: "The uploaded documents do not contain sufficient information to answer this question." State what information is missing.
14. Multi-Document Separation:
   - When responding across multiple documents, keep the information clearly separated by document title and do not mix facts between them.
"""

REWRITE_SYSTEM_PROMPT = """You are an expert search query reformulation assistant for a technical RAG system.
Given a recent conversation history and a user's latest follow-up question, rewrite the follow-up question into an explicit, self-contained standalone search query.

Rules:
1. Resolve all pronouns and ambiguous references (e.g. replace "this region", "these particles", "it", "these percentages", "the table", "that document", "these skills") with the specific entity discussed in earlier turns.
2. If the user refers to a specific document (e.g. "Aurora", "AliyaMam"), include the document name in the rewritten query.
3. If the user asks a comparison or calculation based on previous context, mention the specific subjects and metrics.
4. If the user's question is already fully standalone, return it verbatim without changes.
5. Output ONLY the rewritten standalone query string. Do NOT add quotes, markdown formatting, explanations, or prefixes.
"""


def rewrite_conversational_query(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """
    Step 1: Conversational Query Rewriting.
    Transforms an ambiguous or contextual follow-up query into a standalone search query.
    """
    if not chat_history:
        return query

    # Check if query contains contextual clues that require history resolution
    q_lower = query.lower()
    needs_rewrite = any([
        len(query.split()) < 7,
        any(w in q_lower for w in ["it", "this", "that", "these", "those", "they", "them", "he", "she"]),
        any(w in q_lower for w in ["the same", "the region", "this region", "the table", "above", "mentioned"]),
        any(w in q_lower for w in ["now compare", "calculate the difference", "which of the", "these skills", "these percentages"]),
        any(w in q_lower for w in ["going back", "switching back", "as you said", "you mentioned"]),
    ])

    if not needs_rewrite:
        return query

    # Use the last 4 turns (2 user, 2 assistant) for rewriting context
    recent_turns = chat_history[-4:]
    history_lines = []
    for msg in recent_turns:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        # Truncate very long previous responses to keep latency low
        if len(content) > 400:
            content = content[:400] + "..."
        history_lines.append(f"{role}: {content}")

    history_str = "\n".join(history_lines)
    prompt = f"Recent Conversation:\n{history_str}\n\nLatest User Follow-Up:\n{query}\n\nStandalone Search Query:"

    try:
        if not llm:
            api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
            if not api_key or "your-openai" in api_key:
                return query
            llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.0, api_key=api_key)

        response = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        rewritten = response.content.strip().strip('"\'')
        if rewritten and len(rewritten) > 3 and not rewritten.startswith(("Error", "Sorry")):
            generation_logger.info(f"Query Rewriting (Step 1): '{query}' -> '{rewritten}'")
            return rewritten
        return query
    except Exception as e:
        generation_logger.warning(f"Query rewriting failed ({e}); falling back to original query.")
        return query


def detect_multi_document_query(query: str, available_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Step 4: Detects whether a query asks for information from multiple distinct documents.
    """
    if not available_chunks:
        return []

    available_files = sorted(list({c.get("filename") for c in available_chunks if c.get("filename")}))
    if len(available_files) < 2:
        return []

    q_lower = query.lower()
    explicitly_mentioned = [
        f for f in available_files
        if f.lower().replace(".pdf", "").replace(".docx", "") in q_lower or f.lower() in q_lower
    ]

    if len(explicitly_mentioned) >= 2:
        return explicitly_mentioned

    multi_doc_keywords = [
        "two documents", "both documents", "each document",
        "summarize the documents", "summarize both", "compare the documents",
        "summarize the two"
    ]
    if any(k in q_lower for k in multi_doc_keywords):
        return available_files[:2]

    return []


def detect_target_document(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]],
    available_chunks: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Step 5: Dynamically detects whether a query or conversation explicitly targets a specific document,
    using universal filename matching without hardcoded document dictionaries.
    """
    if not available_chunks:
        return None
    available_files = sorted(list({c.get("filename") for c in available_chunks if c.get("filename")}))
    if len(available_files) < 2:
        return None

    text_to_check = query.lower()
    if chat_history:
        for msg in chat_history[-2:]:
            text_to_check += " " + (msg.get("content") or "").lower()

    scores = {}
    for fname in available_files:
        fn_lower = fname.lower()
        base_name = re.sub(r"\.(pdf|docx|txt|csv|xlsx|pptx)$", "", fn_lower)
        score = 0
        if len(base_name) >= 3 and base_name in text_to_check:
            score += 10

        name_tokens = [tok for tok in re.split(r"[-_.\s]+", base_name) if len(tok) >= 3 and not tok.isdigit()]
        for tok in name_tokens:
            if re.search(r"\b" + re.escape(tok) + r"\b", text_to_check):
                score += 3

        scores[fname] = score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_doc, best_score = sorted_scores[0]
    second_doc, second_score = sorted_scores[1] if len(sorted_scores) > 1 else (None, 0)

    if best_score >= 3 and best_score > (second_score * 2):
        return best_doc
    return None


def format_context_block(sources: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a clean, structured context string."""
    if not sources:
        return "No relevant context found in documents."

    # Sort sources by document and page number to preserve natural reading / causal order
    sorted_sources = sorted(
        enumerate(sources),
        key=lambda item: (item[1].get("filename", ""), item[1].get("page_number", 1), item[0])
    )

    formatted_parts = []
    for orig_idx, s in sorted_sources:
        sec = s.get("section") or s.get("metadata", {}).get("section") or "General"
        src_header = f"--- [Snippet {orig_idx+1} | Source: {s.get('filename', 'doc')} | Page: {s.get('page_number', 1)} | Section: {sec} | Type: {s.get('type', 'text').upper()}] ---"
        content = s.get("content", "").strip()
        summary = s.get("summary")
        if summary:
            snippet_body = f"AI Summary: {summary}\n\nContent:\n{content}"
        else:
            snippet_body = content

        formatted_parts.append(f"{src_header}\n{snippet_body}\n")
        generation_logger.debug(f"Formatted snippet {orig_idx+1}: {len(snippet_body)} characters from {s.get('filename', 'doc')} (Page {s.get('page_number', 1)})")
        
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

    generation_logger.debug(f"Formatted chat history with {len(recent_turns)} turns.")
    return "\n".join(history_lines)


def generate_rag_response(
    project_id: str,
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    top_k: int = HYBRID_SEARCH_TOP_K,
    top_n: int = RERANK_TOP_N,
) -> Dict[str, Any]:
    """
    Executes the full RAG pipeline with multi-turn conversation memory:
    1. Conversational Query Rewriting (Context Resolution)
    2. Hybrid Retrieval (BM25 + ChromaDB + RRF + Cohere) with Multi-Doc Support
    3. Context & Conversation History Formatting
    4. Grounded LLM Response Generation (gpt-4o-mini)
    """
    t_start = time.perf_counter()
    generation_logger.info(f"RAG Chain query received for Project: {project_id} | Query: '{query[:80]}' | History turns: {len(chat_history or [])}")

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    llm = None
    if api_key and "your-openai" not in api_key:
        llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.0, api_key=api_key)

    # Step 1: Query Rewriting for follow-up questions
    search_query = rewrite_conversational_query(query, chat_history=chat_history, llm=llm)

    retriever = HybridRetriever(project_id)

    # Step 4 & 5: Multi-Document Query Handling & Target Document Routing
    multi_docs = detect_multi_document_query(search_query, retriever.chunks)
    target_doc = None
    if multi_docs:
        generation_logger.info(f"Multi-document query detected across: {multi_docs}. Executing per-document retrieval.")
        per_doc_n = max(3, top_n // len(multi_docs))
        combined_sources: List[Dict[str, Any]] = []
        for doc_name in multi_docs:
            doc_sources = retriever.retrieve(query=search_query, top_k=top_k, top_n=per_doc_n, filename_filter=doc_name)
            combined_sources.extend(doc_sources)
        retrieved_sources = combined_sources
    else:
        target_doc = detect_target_document(search_query, chat_history, retriever.chunks)
        if target_doc:
            generation_logger.info(f"Target document routing applied: '{target_doc}'")
        retrieved_sources = retriever.retrieve(query=search_query, top_k=top_k, top_n=top_n, filename_filter=target_doc)

    formatted_context = format_context_block(retrieved_sources)
    formatted_history = format_chat_history_block(chat_history)
    generation_logger.info(f"Context formatted: {len(formatted_context)} chars from {len(retrieved_sources)} snippets | History context: {len(formatted_history)} chars")

    if not api_key or "your-openai" in api_key:
        generation_logger.warning("OPENAI_API_KEY is not configured; using offline demo response preview.")
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

        t_llm = time.perf_counter()
        response = llm.invoke(messages)
        llm_ms = (time.perf_counter() - t_llm) * 1000
        answer_text = response.content.strip()

        total_ms = (time.perf_counter() - t_start) * 1000
        generation_logger.info(f"✔ LLM ({DEFAULT_LLM_MODEL}) generated {len(answer_text)} chars in {llm_ms:.1f}ms | Total RAG response time: {total_ms:.1f}ms")

        return {
            "query": query,
            "search_query": search_query,
            "target_document": target_doc or (", ".join(multi_docs) if multi_docs else "all"),
            "answer": answer_text,
            "sources": retrieved_sources,
            "retrieval_method": retrieved_sources[0].get("retrieval_method", "hybrid_rrf") if retrieved_sources else "none",
        }
    except Exception as e:
        generation_logger.error(f"❌ RAG generation failed: {e}", exc_info=True)
        return {
            "query": query,
            "answer": f"Error generating answer: {str(e)}",
            "sources": retrieved_sources,
            "retrieval_method": "error",
        }
