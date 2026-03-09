# Low-Latency AI Support Agent with RAG

## Architecture Overview
- Backend: FastAPI with WebSockets.
- DB: Supabase PostgreSQL with pgvector.
- RAG: Sentence-Transformers + Groq LLM.
- Frontend: HTML/JS.

## Low-Latency Decisions
- WebSockets for bi-dir streaming.
- Async retrieval/generation.
- Groq for fast inference.

## RAG Details
- Embed KB locally.
- Vector search via pgvector.
- Minimal context injection.

## Context/State Management
- DB-stored history with sliding window + summarization.

See main.py for code.

## Sample Transcripts
User: Hi, what's my order status? (Assume logged in as john@example.com)
Agent: Your order ORD001 is shipped. Details: Product A, B, total $100.

User: (Interrupts) And return policy?
Agent: Returns within 30 days. No repeat of prior info.