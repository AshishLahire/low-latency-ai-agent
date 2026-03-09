from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.models.schemas import ChatRequest
from backend.core.chat_service import stream_response

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """
    POST /chat/stream
    
    Streams the agent response as Server-Sent Events (SSE).

    Why SSE over WebSockets?
    - SSE is unidirectional (server → client) which matches the LLM token stream perfectly
    - No handshake overhead — lower TTFT than WS for single-turn responses
    - Trivially handled by EventSource API in any browser
    - WebSockets would add value only if we needed client interrupts mid-stream
      (e.g. barge-in for voice). That's noted as a future enhancement.

    Backpressure: FastAPI's StreamingResponse handles slow clients automatically.
    Dropped connections: generator will raise GeneratorExit on disconnect,
    caught by the async for loop — no zombie tasks.
    """
    generator = stream_response(
        session_id=req.session_id,
        customer_email=req.customer_email,
        user_message=req.message,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
        },
    )
