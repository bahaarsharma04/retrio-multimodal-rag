import os
from typing import List, Dict, Any, Callable, Optional, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from core.config import DEFAULT_LLM_MODEL, OPENAI_API_KEY

TABLE_SUMMARY_PROMPT = """You are an AI assistant specialized in analyzing tabular data for retrieval-augmented generation (RAG).
Analyze the following table and produce a concise, information-dense summary describing:
1. What the table represents (main topic, entities, key metrics).
2. Key trends, maximums, minimums, or notable comparative figures.
Keep the summary under 150 words. Do not repeat raw markdown syntax.

Table Content:
{content}
"""

IMAGE_SUMMARY_PROMPT = """You are an AI assistant specialized in analyzing document figures and diagrams for RAG.
Analyze the following figure description or caption and produce a clear, search-optimized summary explaining:
1. The type of visual (e.g., system architecture, flowchart, performance benchmark graph).
2. The core concepts, components, and relationships depicted.
Keep the summary under 150 words.

Figure / Diagram Description:
{content}
"""


def get_llm_client() -> Optional[ChatOpenAI]:
    """Initializes the OpenAI Chat model if API key is set."""
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key or api_key.strip() == "" or "your-openai" in api_key:
        return None
    try:
        return ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0.0,
            api_key=api_key,
        )
    except Exception:
        return None


def summarize_chunk_content(
    content: str,
    chunk_type: str,
    llm: Optional[ChatOpenAI] = None,
) -> str:
    """
    Generates a search-optimized summary for tables or figures.
    Falls back to a structured rule-based preview if no OpenAI key is configured.
    """
    if not llm:
        # Graceful fallback if user hasn't added OpenAI API key yet
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        first_few = " ".join(lines[:3]) if lines else content[:100]
        return f"[Generated preview of {chunk_type}]: {first_few[:150]}..."

    try:
        prompt_template = TABLE_SUMMARY_PROMPT if chunk_type == "table" else IMAGE_SUMMARY_PROMPT
        prompt = prompt_template.format(content=content)
        response = llm.invoke([
            SystemMessage(content="You are an expert technical data summarizer for vector search indexing."),
            HumanMessage(content=prompt)
        ])
        return response.content.strip()
    except Exception as e:
        # Fallback to truncated text on API error (e.g. rate limit / network issue)
        return f"Summary of {chunk_type}: {content[:150]}... (Error: {str(e)[:40]})"


def summarize_chunks(
    chunks: List[Dict[str, Any]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Processes all chunks and enriches tables and images with AI summaries.
    
    Why: Embedding natural language summaries of tables/images dramatically increases
    retrieval accuracy during dense vector search, while preserving the raw table/image
    for generation.
    
    Args:
        chunks: List of chunk dictionaries from the chunker.
        progress_callback: Optional callback(current, total) to report live UI progress.
        
    Returns:
        enriched_chunks: Chunks with 'summary' and 'indexed_text' populated.
        stats: Aggregated summarization stats matching the UI tab.
    """
    total = len(chunks)
    enriched_chunks = []
    llm = get_llm_client()
    
    tables_count = 0
    images_count = 0

    for idx, chunk in enumerate(chunks):
        c = dict(chunk)
        chunk_type = c.get("type", "text")
        raw_content = c.get("content", "")

        if chunk_type in ["table", "image"]:
            summary = summarize_chunk_content(raw_content, chunk_type, llm)
            c["summary"] = summary
            # For multi-vector indexing: index the summary alongside content for optimal retrieval
            c["indexed_text"] = f"{summary}\n\nRaw {chunk_type.title()} Content:\n{raw_content}"
            if chunk_type == "table":
                tables_count += 1
            else:
                images_count += 1
        else:
            # For narrative text, content is already optimal prose
            c["summary"] = None
            c["indexed_text"] = raw_content

        enriched_chunks.append(c)

        if progress_callback:
            progress_callback(idx + 1, total)

    stats = {
        "total_chunks": total,
        "processed_chunks": total,
        "tables_summarized": tables_count,
        "images_summarized": images_count,
        "llm_used": DEFAULT_LLM_MODEL if llm else "Rule-based fallback (No OpenAI Key)",
    }

    return enriched_chunks, stats
