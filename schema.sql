-- Semantic DNA Sequencing — Database Schema
-- Safe to run on both fresh and existing databases (all statements are idempotent)

CREATE TABLE IF NOT EXISTS corpus_stats (
    token      TEXT PRIMARY KEY,
    doc_count  INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS corpus_meta (
    id         INTEGER PRIMARY KEY DEFAULT 1,
    total_docs INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT single_row CHECK (id = 1)
);
INSERT INTO corpus_meta (id, total_docs) VALUES (1, 0) ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS documents (
    id               SERIAL PRIMARY KEY,
    content          TEXT NOT NULL,
    title            TEXT,
    semantic_genome  JSONB NOT NULL DEFAULT '[]'::JSONB,
    source_doc_id    INTEGER,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- GIN index on genome JSONB for fast trigram hash @> containment queries
CREATE INDEX IF NOT EXISTS idx_genome_gin
    ON documents
    USING GIN (semantic_genome jsonb_path_ops);

-- Functional GIN index for BM25 full-text scoring via ts_rank
CREATE INDEX IF NOT EXISTS idx_content_tsv
    ON documents
    USING GIN (to_tsvector('english', coalesce(title, '') || ' ' || content));

-- Query analytics log
CREATE TABLE IF NOT EXISTS search_logs (
    id                   SERIAL PRIMARY KEY,
    query                TEXT NOT NULL,
    results_count        INTEGER NOT NULL DEFAULT 0,
    candidates_evaluated INTEGER NOT NULL DEFAULT 0,
    top_score            FLOAT,
    latency_ms           INTEGER NOT NULL DEFAULT 0,
    searched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Migrations: safely add new columns to existing databases
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_doc_id INTEGER;
