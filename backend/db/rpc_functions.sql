-- ============================================================
-- Supabase RPC: match_knowledge_base
-- Run this in the Supabase SQL editor AFTER the main schema.
-- Called by the RAG pipeline for every user turn.
-- ============================================================

CREATE OR REPLACE FUNCTION match_knowledge_base(
    query_embedding     VECTOR(384),
    match_count         INT     DEFAULT 3,
    similarity_threshold FLOAT  DEFAULT 0.35,
    filter_category     TEXT    DEFAULT NULL   -- optional category pre-filter
)
RETURNS TABLE (
    id          UUID,
    title       TEXT,
    content     TEXT,
    category    TEXT,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.title,
        kb.content,
        kb.category,
        1 - (kb.embedding <=> query_embedding) AS similarity
    FROM knowledge_base kb
    WHERE
        kb.is_active = TRUE
        AND (filter_category IS NULL OR kb.category = filter_category)
        AND 1 - (kb.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY kb.embedding <=> query_embedding   -- ascending distance = best match first
    LIMIT match_count;
END;
$$;
