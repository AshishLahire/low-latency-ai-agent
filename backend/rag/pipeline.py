"""
RAG Pipeline
============
Responsibilities:
  1. Embed KB articles and upsert into Supabase (ingestion)
  2. Embed a user query and retrieve top-K similar chunks (retrieval)

Design choices for low latency:
  - Embedding model (all-MiniLM-L6-v2) runs locally — no extra network hop
  - Model is loaded once at startup via module-level singleton
  - Retrieval uses Supabase's pgvector cosine similarity with IVFFlat index
  - top_k is kept small (default 3) — only high-signal context injected
  - Similarity threshold filters noise before it reaches the LLM
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.db.client import get_supabase
from backend.models.schemas import KBChunk

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Embedding model singleton ─────────────────────────────────────────────────

@lru_cache()
def _get_model() -> SentenceTransformer:
    """
    Loaded once at first call, cached for the process lifetime.
    all-MiniLM-L6-v2: 384 dims, ~80 MB RAM, ~5 ms per sentence on CPU.
    """
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


def embed(text: str) -> List[float]:
    """Return a normalised embedding vector for a single text."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Batch-embed for ingestion — much faster than one-by-one."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    return vecs.tolist()


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_knowledge_base() -> int:
    """
    Fetch all active KB rows without embeddings, compute and store them.
    Safe to run multiple times (idempotent).
    Returns the number of articles embedded.
    """
    db = get_supabase()

    # Fetch rows that still need embedding
    rows = (
        db.table("knowledge_base")
        .select("id, title, content")
        .eq("is_active", True)
        .is_("embedding", "null")
        .execute()
        .data
    )

    if not rows:
        logger.info("No KB rows need embedding — ingestion skipped.")
        return 0

    texts = [f"{r['title']}\n{r['content']}" for r in rows]
    embeddings = embed_batch(texts)

    for row, vec in zip(rows, embeddings):
        db.table("knowledge_base").update({"embedding": vec}).eq("id", row["id"]).execute()

    logger.info("Ingested %d KB articles.", len(rows))
    return len(rows)


# ── Retrieval ─────────────────────────────────────────────────────────────────

async def retrieve(query: str, top_k: int | None = None, category: str | None = None) -> List[KBChunk]:
    """
    Async RAG retrieval:
      1. Embed the query (sync, ~5 ms)
      2. Call Supabase RPC for vector similarity search
      3. Filter by threshold and return top-k chunks

    Running the sync embed() in a thread executor keeps the FastAPI
    event loop unblocked.
    """
    k = top_k or settings.rag_top_k

    # Offload CPU-bound embedding to thread pool
    loop = asyncio.get_event_loop()
    query_vec = await loop.run_in_executor(None, embed, query)

    db = get_supabase()

    # Call the Supabase RPC function (defined below)
    # Passing the vector as a plain list — supabase-py serialises it correctly
    params: dict = {
        "query_embedding": query_vec,
        "match_count": k,
        "similarity_threshold": settings.rag_similarity_threshold,
    }
    if category:
        params["filter_category"] = category

    result = db.rpc("match_knowledge_base", params).execute()

    chunks = []
    for row in (result.data or []):
        chunks.append(
            KBChunk(
                id=row["id"],
                title=row["title"],
                content=row["content"],
                category=row["category"],
                similarity=round(row["similarity"], 4),
            )
        )

    logger.debug("RAG retrieved %d chunks for query: %.60s", len(chunks), query)
    return chunks


def format_context(chunks: List[KBChunk]) -> str:
    """
    Render KB chunks into a compact context block for the LLM prompt.
    Only minimal, high-signal content — no padding.
    """
    if not chunks:
        return ""
    parts = []
    for c in chunks:
        parts.append(f"[{c.category.upper()}] {c.title}\n{c.content}")
    return "\n\n---\n\n".join(parts)
