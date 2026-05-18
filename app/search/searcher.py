import json
import re
from collections import Counter

import psycopg2.extras

from app.genome.extractor import extract_genome
from app.genome.pos_tagger import pos_sequence_similarity
from app.models import SearchResponse, SearchResult
from app.config import settings

_STRIP_WORDS = frozenset({
    "what", "who", "where", "when", "why", "how", "which", "whose", "whom",
    "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "can", "could", "would", "should", "shall", "will",
    "have", "has", "had", "may", "might", "must",
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "but", "with", "from", "by", "about", "into", "through", "it", "its",
    "this", "that", "these", "those", "i", "me", "my", "we", "you", "he",
    "she", "they", "them", "their", "our", "your",
})


def preprocess_query(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", query.lower())
    content = [t for t in tokens if t not in _STRIP_WORDS]
    return " ".join(content) if len(content) >= 2 else query

DEFAULT_WEIGHTS = {
    "trigram": 0.40,
    "phonetic": 0.20,
    "pos_sequence": 0.20,
    "context_depth": 0.10,
    "entropy": 0.10,
}


def _get_candidates(conn, query_hashes: list[str], limit: int) -> list[int]:
    """GIN pre-filter: one @> seek per trigram hash, UNIONed together."""
    if not query_hashes:
        # Fall back to returning all documents (small corpora / empty query)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents LIMIT %s", (limit,))
            return [row[0] for row in cur.fetchall()]

    parts = []
    params = []
    for h in query_hashes:
        parts.append(
            "SELECT id FROM documents WHERE semantic_genome @> %s::jsonb"
        )
        params.append(json.dumps([{"trigram_hash": h}]))

    sql = " UNION ".join(parts) + f" LIMIT {limit}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        ids = [row[0] for row in cur.fetchall()]

    # When GIN finds no trigram matches, fall back to scoring all documents so
    # phonetic / POS dimensions can still surface relevant results.
    if not ids:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents LIMIT %s", (limit,))
            ids = [row[0] for row in cur.fetchall()]

    return ids


def _score_candidate(
    query_genome: list[dict],
    doc_genome: list[dict],
    weights: dict[str, float],
) -> dict[str, float]:
    # --- Dimension 1: Trigram Jaccard ---
    q_hashes = {g["trigram_hash"] for g in query_genome}
    d_hashes = {g["trigram_hash"] for g in doc_genome}
    intersection = len(q_hashes & d_hashes)
    union = len(q_hashes | d_hashes)
    trigram_score = intersection / union if union else 0.0

    # --- Dimension 2: Phonetic word-level Jaccard ---
    # Split each gene's combined "W1-W2-W3" code into individual word phonetics
    # so that a matching word in any window position is counted.
    q_ph = Counter(code for g in query_genome for code in g["phonetic_code"].split("-"))
    d_ph = Counter(code for g in doc_genome for code in g["phonetic_code"].split("-"))
    shared = sum((q_ph & d_ph).values())
    total = sum((q_ph | d_ph).values())
    phonetic_score = shared / total if total else 0.0

    # --- Dimension 3: POS sequence — best-match average ---
    q_pos = [g["pos_sequence"] for g in query_genome]
    d_pos = [g["pos_sequence"] for g in doc_genome]
    if q_pos and d_pos:
        pos_total = 0.0
        for qp in q_pos:
            best = max((pos_sequence_similarity(qp, dp) for dp in d_pos), default=0.0)
            pos_total += best
        pos_score = pos_total / len(q_pos)
    else:
        pos_score = 0.0

    # --- Dimension 4: Context depth proximity ---
    q_depth = sum(g["context_depth"] for g in query_genome) / len(query_genome)
    d_depth = sum(g["context_depth"] for g in doc_genome) / len(doc_genome)
    depth_score = 1.0 - abs(q_depth - d_depth) / 3.0

    # --- Dimension 5: Entropy alignment ---
    q_ent = sum(g["entropy_score"] for g in query_genome) / len(query_genome)
    d_ent = sum(g["entropy_score"] for g in doc_genome) / len(doc_genome)
    max_ent = max(q_ent, d_ent, 1e-9)
    entropy_score = 1.0 - abs(q_ent - d_ent) / max_ent

    total = (
        weights["trigram"] * trigram_score
        + weights["phonetic"] * phonetic_score
        + weights["pos_sequence"] * pos_score
        + weights["context_depth"] * depth_score
        + weights["entropy"] * entropy_score
    )

    return {
        "trigram": round(trigram_score, 4),
        "phonetic": round(phonetic_score, 4),
        "pos_sequence": round(pos_score, 4),
        "context_depth": round(depth_score, 4),
        "entropy": round(entropy_score, 4),
        "total": round(total, 4),
        "matched_trigrams": intersection,
    }


def search(
    conn,
    query: str,
    limit: int = 10,
    min_score: float = 0.0,
    weights: dict[str, float] | None = None,
) -> SearchResponse:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    effective_query = preprocess_query(query)
    query_genome = extract_genome(effective_query, conn, smoothing=settings.entropy_smoothing)
    query_hashes = [g["trigram_hash"] for g in query_genome]

    candidate_ids = _get_candidates(conn, query_hashes, settings.gin_candidate_limit)

    if not candidate_ids:
        return SearchResponse(
            query=query,
            results=[],
            candidates_evaluated=0,
            query_genome_length=len(query_genome),
        )

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, title, content, semantic_genome FROM documents WHERE id = ANY(%s)",
            (candidate_ids,),
        )
        rows = cur.fetchall()

    results: list[SearchResult] = []
    for row in rows:
        doc_genome = row["semantic_genome"]
        if not doc_genome:
            continue
        scores = _score_candidate(query_genome, doc_genome, w)
        if scores["total"] < min_score:
            continue
        results.append(
            SearchResult(
                document_id=row["id"],
                title=row["title"],
                content_preview=row["content"][:200],
                total_score=scores["total"],
                dimension_scores={k: v for k, v in scores.items() if k not in ("total", "matched_trigrams")},
                matched_trigrams=scores["matched_trigrams"],
            )
        )

    results.sort(key=lambda r: r.total_score, reverse=True)

    return SearchResponse(
        query=query,
        results=results[:limit],
        candidates_evaluated=len(rows),
        query_genome_length=len(query_genome),
    )
