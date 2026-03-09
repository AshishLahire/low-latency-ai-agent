from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID


# ── Conversation ──────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str                       # "user" | "assistant" | "system"
    content: str
    ts: Optional[datetime] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None   # None → new session
    customer_email: Optional[str] = None
    message: str


class StreamChunk(BaseModel):
    type: str           # "token" | "done" | "error" | "context_loaded"
    data: str = ""


# ── Customer ──────────────────────────────────────────────────────────────────

class CustomerProfile(BaseModel):
    id: UUID
    name: str
    email: str
    tier: str


# ── Orders ────────────────────────────────────────────────────────────────────

class Order(BaseModel):
    id: UUID
    status: str
    total_amount: float
    currency: str
    items: List[Any]
    tracking_number: Optional[str]
    created_at: datetime


# ── RAG ───────────────────────────────────────────────────────────────────────

class KBChunk(BaseModel):
    id: str
    title: str
    content: str
    category: str
    similarity: float


# ── Session ───────────────────────────────────────────────────────────────────

class Session(BaseModel):
    id: str
    customer_id: Optional[str]
    context_window: List[ChatMessage] = []
    context_summary: Optional[str] = None
