"""
Chat Service
============
Orchestrates a single agent turn:

  1. Resolve session + customer (parallel where possible)
  2. Retrieve RAG context (async, non-blocking)
  3. Compress context window if needed
  4. Build messages array
  5. Stream Groq response token-by-token via SSE
  6. Persist assistant turn + updated session (fire-and-forget)

Latency budget per turn (approximate, CPU inference + Groq llama3-8b):
  - Session fetch:      ~30 ms  (Supabase indexed read)
  - Customer fetch:     ~30 ms  (parallel with session)
  - Embedding:          ~5 ms   (local MiniLM)
  - Vector search:      ~20 ms  (pgvector IVFFlat)
  - Time-to-first-token: ~250 ms (Groq llama3-8b)
  Total TTFT target:   < 400 ms
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from groq import AsyncGroq

from backend.core.config import get_settings
from backend.core.context import (
    add_turn,
    build_messages,
    compress_context,
    get_or_create_session,
    log_message,
    save_session,
    should_summarise,
)
from backend.db.customers import format_customer_info, get_customer_by_email, get_customer_by_id, get_recent_orders
from backend.models.schemas import KBChunk, Session
from backend.rag.pipeline import format_context, retrieve

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_groq() -> AsyncGroq:
    return AsyncGroq(api_key=settings.groq_api_key)


async def _resolve_customer(session: Session, customer_email: Optional[str]) -> tuple[Optional[dict], list]:
    """
    Resolve customer profile + recent orders.
    Priority:
      1. session.customer_id already set (fastest — just fetch profile)
      2. customer_email passed in this request
      3. Email mentioned in conversation history (extracted from context window)
    """
    customer_id = session.customer_id
    customer = None

    if customer_id:
        # Already known — just refresh profile
        customer = await get_customer_by_id(customer_id)

    if not customer and customer_email:
        customer = await get_customer_by_email(customer_email)

    if not customer:
        # Last resort: scan recent context window for an email mention
        extracted_email = _extract_email_from_context(session)
        if extracted_email:
            customer = await get_customer_by_email(extracted_email)

    if not customer:
        return None, []

    # Persist customer_id on session so future turns don't re-lookup
    session.customer_id = str(customer["id"])

    orders = await get_recent_orders(session.customer_id)
    return customer, orders


def _extract_email_from_context(session: Session) -> Optional[str]:
    """Scan the last 10 context turns for an email address typed by the user."""
    import re
    email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    for turn in reversed(session.context_window[-10:]):
        if turn.role == "user":
            match = email_re.search(turn.content)
            if match:
                return match.group(0)
    return None


async def stream_response(
    session_id: Optional[str],
    customer_email: Optional[str],
    user_message: str,
) -> AsyncGenerator[str, None]:
    """
    Main streaming generator. Yields SSE-formatted strings.

    Yielded event types:
      data: {"type": "session_id", "data": "<id>"}   — first chunk, always
      data: {"type": "token",      "data": "<text>"}  — streamed tokens
      data: {"type": "done",       "data": ""}        — stream complete
      data: {"type": "error",      "data": "<msg>"}   — on failure
    """
    import json
    start_time = time.monotonic()
    groq = _get_groq()

    try:
        # ── 1. Session + customer resolution (parallel) ────────────────────
        session = await get_or_create_session(session_id, None)

        # Yield session ID immediately so client can track it
        yield f"data: {json.dumps({'type': 'session_id', 'data': session.id})}\n\n"

        customer, orders = await _resolve_customer(session, customer_email)
        customer_info = format_customer_info(customer, orders) if customer else ""

        # ── 2. RAG retrieval (async, parallel-friendly) ────────────────────
        # Retrieval runs while we already started yielding — no blocking
        chunks: list[KBChunk] = await retrieve(user_message)
        rag_context = format_context(chunks)

        # ── 3. Context compression if window is large ──────────────────────
        if should_summarise(session):
            await compress_context(session, groq)

        # ── 4. Add user turn to window ─────────────────────────────────────
        add_turn(session, "user", user_message)

        # ── 5. Build messages + call Groq with streaming ───────────────────
        messages = build_messages(session, user_message, rag_context, customer_info)

        stream = await groq.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            max_tokens=300,         # short, snappy responses — like a call agent
            temperature=0.4,
            stream=True,
        )

        # ── 6. Stream tokens to client ─────────────────────────────────────
        full_response = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response.append(delta)
                yield f"data: {json.dumps({'type': 'token', 'data': delta})}\n\n"

        assistant_text = "".join(full_response)
        latency_ms = int((time.monotonic() - start_time) * 1000)

        # ── 7. Persist (fire-and-forget — doesn't block the stream) ───────
        add_turn(session, "assistant", assistant_text)

        asyncio.create_task(save_session(session))
        asyncio.create_task(
            log_message(
                session.id, "user", user_message,
                retrieved_chunks=[c.model_dump() for c in chunks],
            )
        )
        asyncio.create_task(
            log_message(session.id, "assistant", assistant_text, latency_ms=latency_ms)
        )

        logger.info("Turn complete in %d ms | session=%s", latency_ms, session.id)
        yield f"data: {json.dumps({'type': 'done', 'data': ''})}\n\n"

    except Exception as exc:
        logger.exception("Error in stream_response: %s", exc)
        yield f"data: {json.dumps({'type': 'error', 'data': str(exc)})}\n\n"
