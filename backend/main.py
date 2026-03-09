"""
HelpKart AI Support Agent — FastAPI Application
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.chat import router as chat_router
from backend.api.admin import router as admin_router
from backend.rag.pipeline import _get_model   # warm up embedding model at startup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up the embedding model so the first user request
    doesn't pay the ~2s model-load penalty.
    """
    logger.info("Warming up embedding model...")
    _get_model()   # loads and caches
    logger.info("HelpKart agent ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="HelpKart AI Support Agent",
    description="Low-latency conversational AI with RAG, Groq, and Supabase",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(admin_router)

# Serve frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
