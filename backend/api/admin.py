from fastapi import APIRouter, BackgroundTasks
from backend.rag.pipeline import ingest_knowledge_base

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/ingest")
async def trigger_ingestion(background_tasks: BackgroundTasks):
    """
    POST /admin/ingest
    Kick off KB embedding ingestion in the background.
    Safe to call repeatedly — only embeds rows with null embedding.
    """
    background_tasks.add_task(ingest_knowledge_base)
    return {"status": "ingestion started in background"}


@router.get("/health")
async def health():
    return {"status": "ok"}
