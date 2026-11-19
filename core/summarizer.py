import os
import base64
from pathlib import Path
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
            temperature=0.2,
            api_key=api_key,
        )
    except Exception:
        return None


def encode_image_file(image_path: Optional[str]) -> Optional[str]:
    """Encodes an image file to a base64 string for OpenAI Vision API."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def summarize_chunk_content(
    content: str,
    chunk_type: str,
    llm: Optional[ChatOpenAI] = None,
    image_path: Optional[str] = None,
    caption: Optional[str] = None,
) -> str:
    """
    Generates a search-optimized summary for tables or figures.
    Uses multimodal Vision (GPT-4o-mini) when image bytes are available.
    """
    if not llm:
        # Graceful fallback if user hasn't added OpenAI API key yet
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        first_few = " ".join(lines[:3]) if lines else content[:100]
        return f"[Generated preview of {chunk_type}]: {first_few[:150]}..."

    try:
        if chunk_type == "table":
            prompt = TABLE_SUMMARY_PROMPT.format(content=content)
            response = llm.invoke([
                SystemMessage(content="You are an expert technical data summarizer for vector search indexing."),
                HumanMessage(content=prompt)
            ])
            return response.content.strip()

        # chunk_type == "image": Multimodal Vision Analysis
        b64_img = encode_image_file(image_path)
        if b64_img:
            ext = Path(image_path).suffix.lower() if image_path else ".png"
            mime = "image/png" if ext == ".png" else "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/webp" if ext == ".webp" else "image/png"
            caption_text = f"\nAccompanying Caption: {caption}" if caption else ""
            vision_prompt = (
                "You are an expert technical data summarizer for vector search indexing.\n"
                "Analyze this technical figure, diagram, or chart from a document.\n"
                f"{caption_text}\n"
                "Produce a clear, concise, search-optimized summary (under 150 words) describing:\n"
                "1. What kind of visual this is (e.g. flowchart, system architecture, performance plot, schematic).\n"
                "2. The key components, relationships, labels, data trends, or formulas depicted.\n"
                "Focus strictly on what is visually and textually present in the image. Do not ask for more information."
            )
            message = HumanMessage(
                content=[
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_img}"},
                    },
                ]
            )
            response = llm.invoke([message])
            return response.content.strip()

        # If image file is not on disk but a caption exists:
        if caption and caption.strip():
            caption_prompt = (
                f"You are an expert technical data summarizer for vector search indexing.\n"
                f"Analyze this document figure caption and produce a concise, search-optimized summary (under 120 words):\n\n"
                f"Caption: {caption.strip()}"
            )
            response = llm.invoke([
                SystemMessage(content="You are an expert technical data summarizer for vector search indexing."),
                HumanMessage(content=caption_prompt)
            ])
            return response.content.strip()

        # Fallback for images without pixels or captions: informative label rather than an apology
        return f"Document visual element: {content}."

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
            img_p = c.get("image_path") or c.get("metadata", {}).get("image_path")
            cap = c.get("caption") or c.get("metadata", {}).get("caption")
            summary = summarize_chunk_content(
                content=raw_content,
                chunk_type=chunk_type,
                llm=llm,
                image_path=img_p,
                caption=cap,
            )
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
