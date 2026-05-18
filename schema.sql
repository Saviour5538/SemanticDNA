-- Semantic DNA Sequencing — Database Schema

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
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- GIN index on the genome JSONB column using jsonb_path_ops.
-- Supports @> containment queries for fast trigram hash pre-filtering.
CREATE INDEX IF NOT EXISTS idx_genome_gin
    ON documents
    USING GIN (semantic_genome jsonb_path_ops);
