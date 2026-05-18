import math

import psycopg2.extras


def compute_entropy_for_window(
    words: list[str],
    corpus_stats: dict[str, int],
    total_docs: int,
    smoothing: float = 0.5,
) -> float:
    """Average IDF across the words in the window."""
    scores = []
    for w in words:
        df = corpus_stats.get(w.lower(), 0)
        idf = math.log((total_docs + 1) / (df + smoothing))
        scores.append(idf)
    return sum(scores) / len(scores) if scores else 0.0


def update_corpus_stats(conn, unique_tokens: list[str]) -> None:
    """Upsert doc_count for each token and increment total_docs."""
    if not unique_tokens:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """
            INSERT INTO corpus_stats (token, doc_count)
            VALUES (%s, 1)
            ON CONFLICT (token)
            DO UPDATE SET doc_count = corpus_stats.doc_count + 1,
                          updated_at = NOW()
            """,
            [(t,) for t in unique_tokens],
        )
        cur.execute(
            "UPDATE corpus_meta SET total_docs = total_docs + 1 WHERE id = 1"
        )


def fetch_corpus_stats(
    conn, tokens: list[str]
) -> tuple[dict[str, int], int]:
    """Return (token→doc_count mapping, total_docs) for the given tokens."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        lower_tokens = list({t.lower() for t in tokens})
        cur.execute(
            "SELECT token, doc_count FROM corpus_stats WHERE token = ANY(%s)",
            (lower_tokens,),
        )
        stats = {row["token"]: row["doc_count"] for row in cur.fetchall()}

        cur.execute("SELECT total_docs FROM corpus_meta WHERE id = 1")
        row = cur.fetchone()
        total_docs = row["total_docs"] if row else 0

    return stats, total_docs
