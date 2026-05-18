import psycopg2.extras

from app.genome.extractor import extract_genome, extract_genome_batch
from app.genome.entropy import update_corpus_stats
from app.genome.pos_tagger import compute_all_pos_tags
from app.config import settings


def ingest_document(
    conn,
    content: str,
    title: str | None,
    update_stats: bool = True,
) -> int:
    """Extract genome and persist document. Returns the new document id.

    Set update_stats=False when ingesting chunks from one source document
    and corpus_stats has already been updated for the whole source document.
    """
    if update_stats:
        tokens, _ = compute_all_pos_tags(content)
        unique_tokens = list({t.lower() for t in tokens if t.isalpha()})
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


def ingest_documents_batch(
    conn,
    items: list[tuple[str, str | None]],
) -> list[int]:
    """Ingest multiple chunks from one source document efficiently.

    Improvements over calling ingest_document() in a loop:
    - Runs spaCy once via nlp.pipe() instead of once per chunk (~3x faster).
    - Fetches corpus_stats in a single DB query for all chunks combined.
    - corpus_stats must already be updated before calling this (Fix 1).

    Returns the inserted document IDs in the same order as items.
    """
    if not items:
        return []

    texts = [content for content, _ in items]
    titles = [title for _, title in items]

    # One spaCy pass + one DB stats fetch for all chunks
    genomes = extract_genome_batch(texts, conn, smoothing=settings.entropy_smoothing)

    doc_ids: list[int] = []
    with conn.cursor() as cur:
        for content, title, genome in zip(texts, titles, genomes):
            cur.execute(
                """
                INSERT INTO documents (content, title, semantic_genome)
                VALUES (%s, %s, %s::jsonb)
                RETURNING id
                """,
                (content, title, psycopg2.extras.Json(genome)),
            )
            doc_ids.append(cur.fetchone()[0])

    return doc_ids
