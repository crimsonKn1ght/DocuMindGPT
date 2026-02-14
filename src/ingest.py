"""
DocuMindGPT – Ingestion Agent
Reads PDF or plain-text files, splits them into chunks,
generates embeddings via Gemini, and upserts into Supabase.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_text_from_file(file_path: str) -> str:
    """Return the raw text content of a PDF or plain-text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        if not pages:
            raise ValueError("PDF contains no extractable text.")
        return "\n".join(pages)

    # Fall back to plain text (.txt, .md, etc.)
    return path.read_text(encoding="utf-8")


def _chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using Gemini Embedding 004."""
    embeddings: List[List[float]] = []
    # The API supports batching, but to stay within rate limits we batch in
    # groups of 100 (the documented max per request).
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings.extend(result["embedding"])
        except Exception:
            logger.exception("Embedding API call failed for batch starting at index %d", i)
            raise
    return embeddings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    End-to-end ingestion pipeline.

    Returns the number of chunks successfully stored.
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

    # --- Load & chunk ---
    logger.info("Loading file: %s", file_path)
    raw_text: str = _load_text_from_file(file_path)
    chunks: List[str] = _chunk_text(raw_text, chunk_size, chunk_overlap)
    logger.info("Split into %d chunks.", len(chunks))

    if not chunks:
        logger.warning("No chunks produced – nothing to ingest.")
        return 0

    # --- Embed ---
    logger.info("Generating embeddings …")
    embeddings: List[List[float]] = _embed_texts(chunks)

    # --- Upsert into Supabase ---
    logger.info("Upserting %d chunks into Supabase …", len(chunks))
    file_name = Path(file_path).name
    rows: List[Dict[str, Any]] = [
        {
            "content": chunk,
            "metadata": {"source": file_name, "chunk_index": idx},
            "embedding": emb,
        }
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    try:
        sb.table("document_chunks").insert(rows).execute()
    except Exception:
        logger.exception("Supabase insert failed.")
        raise

    logger.info("Successfully stored %d chunks.", len(rows))
    return len(rows)
