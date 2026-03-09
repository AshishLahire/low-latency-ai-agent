# HelpKart AI Support Agent

> Low-latency conversational AI support agent with RAG, Groq (llama3-8b), Supabase (pgvector), and FastAPI SSE streaming.

---

## Architecture Overview

```
Browser (EventSource)
      │  SSE stream (token-by-token)
      ▼
FastAPI  /chat/stream
      │
      ├── Session & context manager  ← Supabase (sessions table)
      ├── Customer/order resolver    ← Supabase (customers, orders)
      ├── RAG retrieval              ← Supabase pgvector (knowledge_base)
      │      └── local MiniLM embed → cosine search (IVFFlat)
      └── Groq llama3-8b stream     ← token-by-token → SSE → browser
```

---

## Low-Latency Design Decisions

### Why SSE over WebSockets?
SSE is a unidirectional, HTTP/1.1-compatible stream. For LLM token delivery it is strictly better than WebSockets because:
- No handshake round-trip — connection opens immediately
- Fewer protocol layers (no WS framing, masking, heartbeats)
- Natively reconnectable by the browser's EventSource API
- WebSockets would only add value for **barge-in** (voice interruption mid-stream) — noted as a v2 feature

### Latency budget per turn
| Step | Target | How |
|---|---|---|
| Session fetch | ~30 ms | Single indexed PK lookup |
| Customer fetch | ~30 ms | Indexed by email |
| Embedding | ~5 ms | Local all-MiniLM-L6-v2 (384-dim, CPU) |
| Vector search | ~20 ms | pgvector IVFFlat approximate NN |
| Groq TTFT | ~250 ms | llama3-8b-8192 (fastest OSS model on Groq) |
| **Total TTFT** | **< 400 ms** | |

### Context window strategy
- **Sliding window**: only the last `MAX_CONTEXT_TURNS` (default 10) turns are sent to Groq
- **Memory compression**: when the window hits `SUMMARY_TRIGGER_TURNS` (8), the oldest half is summarised into `sessions.context_summary` via one cheap Groq call
- **Result**: token count per call stays flat regardless of conversation length → latency stays constant, not O(n)

### RAG optimisations
- `IVFFlat` index (lists=100): approximate NN, ~5× faster than exact cosine at the cost of <1% recall drop
- `top_k=3` + `similarity_threshold=0.35`: only high-signal chunks injected — reduces prompt tokens and hallucination risk
- Retrieval runs async while session/customer data loads — parallel, not sequential
- Fire-and-forget persistence (asyncio.create_task) — DB writes never block the stream

### Backpressure
FastAPI's `StreamingResponse` uses Python async generators. If the client reads slowly, the generator naturally pauses at the `yield` — no explicit backpressure needed.

### Dropped connections
When a client disconnects mid-stream, the `async for chunk in stream` loop raises `GeneratorExit`. The Groq stream is abandoned and no further DB writes occur. No zombie coroutines.

---

## RAG Implementation

```
Knowledge base (text) 
  → embed_batch() with all-MiniLM-L6-v2
  → VECTOR(384) stored in Supabase knowledge_base.embedding
  → IVFFlat index for fast ANN search

Per user turn:
  query → embed() → pgvector cosine search → top-3 chunks
  → format_context() → injected into system prompt
```

The `match_knowledge_base` Supabase RPC handles the vector search server-side — no raw embedding data round-trips to the application layer.

---

## Database Schema

See `helpkart_schema.sql` for the full schema with comments.

Key tables:
- `customers` — profiles, tier
- `orders` — state machine (pending → shipped → delivered)
- `knowledge_base` — RAG articles with `vector(384)` embeddings
- `sessions` — context window (JSONB) + compressed summary
- `messages` — append-only turn log with latency tracking

---

## Setup

### 1. Supabase
1. Create a new Supabase project
2. Run `helpkart_schema.sql` in the SQL editor
3. Run `backend/db/rpc_functions.sql` in the SQL editor
4. Copy your project URL and service key

### 2. Environment
```bash
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY
```

### 3. Install & run
```bash
pip install -r requirements.txt

# From the helpkart/ root:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Ingest knowledge base
```bash
curl -X POST http://localhost:8000/admin/ingest
```
This embeds all KB articles into pgvector. Run once (idempotent).

### 5. Open the UI
Visit `http://localhost:8000` — the frontend is served as static files.

---

## Project Structure

```
helpkart/
├── requirements.txt
├── .env.example
├── frontend/
│   └── index.html          # Streaming chat UI (pure HTML/JS)
└── backend/
    ├── main.py             # FastAPI app + lifespan
    ├── api/
    │   ├── chat.py         # POST /chat/stream  (SSE)
    │   └── admin.py        # POST /admin/ingest, GET /admin/health
    ├── core/
    │   ├── config.py       # Settings (pydantic-settings)
    │   ├── chat_service.py # Main orchestrator — RAG + context + Groq stream
    │   └── context.py      # Sliding window, summarisation, message builder
    ├── db/
    │   ├── client.py       # Supabase singleton
    │   ├── customers.py    # Customer + order helpers
    │   └── rpc_functions.sql  # pgvector RPC
    ├── rag/
    │   └── pipeline.py     # Embed, ingest, retrieve, format_context
    └── models/
        └── schemas.py      # Pydantic models
```

---

## Trade-offs & Assumptions

| Decision | Trade-off |
|---|---|
| SSE over WebSockets | No barge-in support (acceptable for text; add WS for voice) |
| llama3-8b over 70b | ~3× faster, slightly less nuanced — right call for support |
| all-MiniLM-L6-v2 (local) | No API cost, ~5ms, but slightly lower quality than OpenAI embeddings |
| top_k=3 RAG chunks | Minimal context = lower tokens = lower latency; may miss edge cases |
| JSONB context window | Single-row read per turn; no JOIN; serialisation cost is negligible |
| Memory compression | Adds one Groq call every ~8 turns; prevents unbounded context growth |
