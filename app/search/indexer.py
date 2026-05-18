import psycopg2.extras

from app.genome.extractor import extract_genome
from app.genome.entropy import update_corpus_stats
from app.genome.pos_tagger import compute_all_pos_tags
from app.config import settings


def ingest_document(conn, content: str, title: str | None) -> int:
    """Extract genome and persist document. Returns the new document id."""
    # Tokenize to get unique terms for corpus_stats update
    tokens, _ = compute_all_pos_tags(content)
    unique_tokens = list({t.lower() for t in tokens if t.isalpha()})

    # Update IDF stats before extracting genome so entropy reflects this doc
    update_corpus_stats(conn, unique_tokens)

    genome = extract_genome(content, conn, smoothing=settings.entropy_smoothing)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (content, title, semantic_genome)
            VALUES (%s, %s, %s::jsonb)
            RETURNING id
            """,
            (content, title, psycopg2.extras.Json(genome)),
        )
        row = cur.fetchone()

    return row[0]
