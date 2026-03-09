from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Supabase
    supabase_url: str
    supabase_service_key: str

    # Groq
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 3
    rag_similarity_threshold: float = 0.35

    # Context management
    max_context_turns: int = 10
    summary_trigger_turns: int = 8

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
