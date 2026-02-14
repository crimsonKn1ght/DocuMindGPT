"""
DocuMindGPT – Retrieval & Generation Agent
Embeds the user query, retrieves relevant chunks from Supabase,
and generates an answer with Gemini-2.5-Flash.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from supabase import create_client, Client

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION: str = (
    "You are a helpful assistant. Answer the user query ONLY using the provided context. "
    "If the answer is not in the context, state that you do not know."
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embed_query(query: str) -> List[float]:
    """Embed a single query string for retrieval."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        return result["embedding"]
    except Exception:
        logger.exception("Failed to embed query.")
        raise


def _retrieve_chunks(
    sb: Client,
    query_embedding: List[float],
    match_count: int = 5,
    match_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Call the Supabase RPC match_documents function."""
    try:
        response = sb.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "match_threshold": match_threshold,
            },
        ).execute()
        return response.data or []
    except Exception:
        logger.exception("Supabase RPC match_documents failed.")
        raise


def _build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Compose the generation prompt with retrieved context."""
    context_parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Chunk {i}]\n{chunk['content']}")
    context_block = "\n\n".join(context_parts)
    return (
        f"Context:\n{context_block}\n\n"
        f"User Query: {query}\n\n"
        "Answer:"
    )


def _generate_answer(prompt: str) -> str:
    """Call Gemini-2.5-Flash to generate an answer."""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_INSTRUCTION,
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        logger.exception("Gemini generation failed.")
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(
    user_query: str,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    top_k: int = 5,
    match_threshold: float = 0.5,
) -> Tuple[str, str]:
    """
    Run the full RAG pipeline: embed -> retrieve -> generate.

    Returns
    -------
    (answer, context)
        answer  – The generated response from Gemini.
        context – The concatenated context chunks used for generation.
    """
    # --- Configuration ---
    gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    supabase_url = supabase_url or os.getenv("SUPABASE_URL", "")
    supabase_key = supabase_key or os.getenv("SUPABASE_KEY", "")

    if not gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    if not supabase_url or not supabase_key:
        raise EnvironmentError("SUPABASE_URL / SUPABASE_KEY is not set.")

    genai.configure(api_key=gemini_api_key)
    sb: Client = create_client(supabase_url, supabase_key)

    # --- Embed query ---
    logger.info("Embedding user query …")
    query_embedding: List[float] = _embed_query(user_query)

    # --- Retrieve ---
    logger.info("Retrieving top-%d chunks …", top_k)
    chunks: List[Dict[str, Any]] = _retrieve_chunks(
        sb, query_embedding, match_count=top_k, match_threshold=match_threshold
    )
    if not chunks:
        logger.warning("No relevant chunks found.")
        return "I could not find any relevant information in the knowledge base.", ""

    logger.info("Retrieved %d chunks (best similarity: %.4f).", len(chunks), chunks[0].get("similarity", 0))

    # --- Generate ---
    prompt: str = _build_prompt(user_query, chunks)
    context_text: str = "\n\n".join(c["content"] for c in chunks)
    logger.info("Generating answer …")
    answer: str = _generate_answer(prompt)

    return answer, context_text
