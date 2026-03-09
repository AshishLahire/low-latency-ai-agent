from functools import lru_cache
from supabase import create_client, Client
from backend.core.config import get_settings


@lru_cache()
def get_supabase() -> Client:
    """
    Single shared Supabase client (service role).
    Service role bypasses RLS — only used server-side, never exposed to clients.
    """
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_key)
