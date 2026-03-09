"""
Context & State Manager
=======================
Handles the AI context window for each session.

Strategy (addresses assignment requirement directly):
  - Sliding window: keep the last MAX_CONTEXT_TURNS turns in sessions.context_window
  - When window fills (SUMMARY_TRIGGER_TURNS), compress older turns into a
    short paragraph stored in sessions.context_summary
  - Each LLM call receives: system prompt + summary (if any) + recent window
  - This keeps token count flat over long conversations → latency stays constant
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

from groq import AsyncGroq

from backend.core.config import get_settings
from backend.db.client import get_supabase
from backend.models.schemas import ChatMessage, Session

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Session CRUD ──────────────────────────────────────────────────────────────

async def get_or_create_session(session_id: Optional[str], customer_id: Optional[str]) -> Session:
    db = get_supabase()

    if session_id:
        row = db.table("sessions").select("*").eq("id", session_id).maybe_single().execute()
        if row.data:
            d = row.data
            return Session(
                id=d["id"],
                customer_id=d.get("customer_id"),
                context_window=[ChatMessage(**m) for m in (d.get("context_window") or [])],
                context_summary=d.get("context_summary"),
            )

    # Create new session
    payload: dict = {"status": "active", "context_window": [], "channel": "text"}
    if customer_id:
        payload["customer_id"] = customer_id

    new = db.table("sessions").insert(payload).execute()
    d = new.data[0]
    return Session(id=d["id"], customer_id=d.get("customer_id"))


async def save_session(session: Session) -> None:
    db = get_supabase()
    db.table("sessions").update({
        "context_window": [m.model_dump(mode="json") for m in session.context_window],
        "context_summary": session.context_summary,
        "last_active_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", session.id).execute()


async def log_message(
    session_id: str,
    role: str,
    content: str,
    retrieved_chunks: list | None = None,
    latency_ms: int | None = None,
) -> None:
    db = get_supabase()
    db.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "retrieved_chunks": retrieved_chunks or [],
        "latency_ms": latency_ms,
    }).execute()


# ── Context window management ─────────────────────────────────────────────────

def add_turn(session: Session, role: str, content: str) -> None:
    """Append a message to the context window."""
    session.context_window.append(
        ChatMessage(role=role, content=content, ts=datetime.now(timezone.utc))
    )


def should_summarise(session: Session) -> bool:
    return len(session.context_window) >= settings.summary_trigger_turns


async def compress_context(session: Session, groq_client: AsyncGroq) -> None:
    """
    Summarise the oldest half of the context window into context_summary.
    This keeps the window lean while preserving semantic continuity.
    
    Trade-off: one extra Groq call (~300ms) every SUMMARY_TRIGGER_TURNS turns.
    Amortised cost is low; prevents the window from growing unbounded.
    """
    half = len(session.context_window) // 2
    old_turns = session.context_window[:half]
    session.context_window = session.context_window[half:]  # keep recent half

    # Build a minimal transcript for the summariser
    transcript = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in old_turns
    )
    existing_summary = session.context_summary or ""

    prompt = (
        "You are compressing a support conversation into a concise memory summary. "
        "Preserve customer name, issues raised, resolutions given, and order details mentioned. "
        "Be brief — 3-5 sentences max.\n\n"
        f"EXISTING SUMMARY:\n{existing_summary}\n\n"
        f"NEW TURNS TO INCORPORATE:\n{transcript}"
    )

    resp = await groq_client.chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )
    session.context_summary = resp.choices[0].message.content.strip()
    logger.debug("Context compressed. New summary length: %d chars", len(session.context_summary))


# ── Build LLM messages list ───────────────────────────────────────────────────

def build_messages(
    session: Session,
    user_message: str,
    rag_context: str,
    customer_info: str,
) -> List[dict]:
    """
    Assemble the messages array sent to Groq on each turn.

    Structure:
      [system] → identity + customer info + RAG context
      [assistant] → context_summary (if present, injected as prior assistant memory)
      [...recent context_window turns]
      [user] → current message
    """
    system_content = _build_system_prompt(customer_info, rag_context)
    messages = [{"role": "system", "content": system_content}]

    # Inject compressed memory as a pseudo-prior assistant turn
    if session.context_summary:
        messages.append({
            "role": "assistant",
            "content": f"[Earlier in this conversation: {session.context_summary}]",
        })

    # Sliding window — most recent turns
    for turn in session.context_window[-settings.max_context_turns:]:
        messages.append({"role": turn.role, "content": turn.content})

    # Current user message
    messages.append({"role": "user", "content": user_message})
    return messages


def _build_system_prompt(customer_info: str, rag_context: str) -> str:
    base = (
        "You are Kira, a fast and friendly AI support agent for HelpKart — an Indian e-commerce platform. "
        "You speak naturally, like a real call-centre agent on a live call. "
        "Keep responses concise (2-4 sentences). Never repeat yourself. "
        "If you don't know something, say so clearly — never hallucinate order details or policies. "
        "Always address the customer by name when you know it.\n\n"
    )

    if customer_info:
        base += f"CUSTOMER ON THE LINE:\n{customer_info}\n\n"

    if rag_context:
        base += f"RELEVANT KNOWLEDGE BASE (use this to ground your answer):\n{rag_context}\n\n"

    base += (
        "RULES:\n"
        "- Never invent order IDs, tracking numbers, or policy details\n"
        "- If context is missing, say 'Let me check that for you'\n"
        "- Keep the conversation flowing naturally\n"
    )
    return base
