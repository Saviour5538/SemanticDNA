import json
import re
import time
from collections import Counter, defaultdict

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

# Genome cache: keyed on (effective_query, total_docs, smoothing).
# total_docs in the key ensures automatic invalidation on every ingest/delete.
_genome_cache: dict[tuple, list[dict]] = {}
_CACHE_MAX = 256

DEFAULT_WEIGHTS = {
    "trigram":       0.35,
    "phonetic":      0.17,
    "pos_sequence":  0.17,
    "context_depth": 0.09,
    "entropy":       0.09,
    "bm25":          0.13,
}


def preprocess_query(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", query.lower())
    content = [t for t in tokens if t not in _STRIP_WORDS]
    return " ".join(content) if len(content) >= 2 else query


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _get_total_docs(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT total_docs FROM corpus_meta WHERE id = 1")
        row = cur.fetchone()
    return row[0] if row else 0


def _get_query_genome(query: str, total_docs: int, conn) -> list[dict]:
    """Return query genome, using cache to avoid re-running spaCy on repeat queries."""
    key = (query, total_docs, settings.entropy_smoothing)
    if key not in _genome_cache:
        if len(_genome_cache) >= _CACHE_MAX:
            _genome_cache.pop(next(iter(_genome_cache)))
        _genome_cache[key] = extract_genome(query, conn, smoothing=settings.entropy_smoothing)
    return _genome_cache[key]


# ── GIN pre-filter ─────────────────────────────────────────────────────────────

def _get_candidates(conn, query_hashes: list[str], limit: int) -> list[int]:
    if not query_hashes:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents LIMIT %s", (limit,))
            return [row[0] for row in cur.fetchall()]

    parts = []
    params = []
    for h in query_hashes:
        parts.append("SELECT id FROM documents WHERE semantic_genome @> %s::jsonb")
        params.append(json.dumps([{"trigram_hash": h}]))

    sql = " UNION ".join(parts) + f" LIMIT {limit}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        ids = [row[0] for row in cur.fetchall()]

    if not ids:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM documents LIMIT %s", (limit,))
            ids = [row[0] for row in cur.fetchall()]

    return ids


# ── BM25 via PostgreSQL ts_rank ────────────────────────────────────────────────

def _get_bm25_scores(conn, candidate_ids: list[int], query: str) -> dict[int, float]:
    """Fetch ts_rank (BM25 proxy) for each candidate in one SQL query."""
    if not candidate_ids or not query.strip():
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id,
                   ts_rank(
                       to_tsvector('english', coalesce(title, '') || ' ' || content),
                       plainto_tsquery('english', %s)
                   ) AS bm25
            FROM documents
            WHERE id = ANY(%s)
            """,
            (query, candidate_ids),
        )
        return {row[0]: float(row[1]) for row in cur.fetchall()}


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_candidate(
    query_genome: list[dict],
    doc_genome: list[dict],
    weights: dict[str, float],
    bm25_score: float = 0.0,
    title_boost: float = 0.0,
) -> dict[str, float]:
    # Dimension 1: Trigram Jaccard
    q_hashes = {g["trigram_hash"] for g in query_genome}
    d_hashes = {g["trigram_hash"] for g in doc_genome}
    intersection = len(q_hashes & d_hashes)
    union = len(q_hashes | d_hashes)
    trigram_score = intersection / union if union else 0.0

    # Dimension 2: Phonetic word-level Jaccard
    q_ph = Counter(code for g in query_genome for code in g["phonetic_code"].split("-"))
    d_ph = Counter(code for g in doc_genome for code in g["phonetic_code"].split("-"))
    shared = sum((q_ph & d_ph).values())
    total_ph = sum((q_ph | d_ph).values())
    phonetic_score = shared / total_ph if total_ph else 0.0

    # Dimension 3: POS sequence best-match average
    q_pos = [g["pos_sequence"] for g in query_genome]
    d_pos = [g["pos_sequence"] for g in doc_genome]
    if q_pos and d_pos:
        pos_total = sum(
            max((pos_sequence_similarity(qp, dp) for dp in d_pos), default=0.0)
            for qp in q_pos
        )
        pos_score = pos_total / len(q_pos)
    else:
        pos_score = 0.0

    # Dimension 4: Context depth proximity
    q_depth = sum(g["context_depth"] for g in query_genome) / len(query_genome)
    d_depth = sum(g["context_depth"] for g in doc_genome) / len(doc_genome)
    depth_score = max(0.0, 1.0 - abs(q_depth - d_depth) / 3.0)

    # Dimension 5: Entropy alignment
    q_ent = sum(g["entropy_score"] for g in query_genome) / len(query_genome)
    d_ent = sum(g["entropy_score"] for g in doc_genome) / len(doc_genome)
    max_ent = max(q_ent, d_ent, 1e-9)
    entropy_score = 1.0 - abs(q_ent - d_ent) / max_ent

    # Dimension 6: BM25 (ts_rank normalised to ~[0,1])
    bm25_norm = min(bm25_score * 10.0, 1.0)

    weighted = (
        weights.get("trigram",       0.35) * trigram_score
        + weights.get("phonetic",    0.17) * phonetic_score
        + weights.get("pos_sequence",0.17) * pos_score
        + weights.get("context_depth",0.09) * depth_score
        + weights.get("entropy",     0.09) * entropy_score
        + weights.get("bm25",        0.13) * bm25_norm
    )

    total = round(min(weighted + title_boost, 1.0), 4)

    return {
        "trigram":       round(trigram_score, 4),
        "phonetic":      round(phonetic_score, 4),
        "pos_sequence":  round(pos_score, 4),
        "context_depth": round(depth_score, 4),
        "entropy":       round(entropy_score, 4),
        "bm25":          round(bm25_norm, 4),
        "title_boost":   round(title_boost, 4),
        "total":         total,
        "matched_trigrams": intersection,
    }


# ── Title boost ────────────────────────────────────────────────────────────────

_PART_SUFFIX = re.compile(r"\s+[—\-]+\s+Part\s+\d+$", re.IGNORECASE)


def _base_title(title: str | None) -> str:
    if not title:
        return ""
    return _PART_SUFFIX.sub("", title).strip()


def _compute_title_boost(query_words: set[str], title: str | None) -> float:
    """Additive boost (max 0.10) when query words appear in the document title."""
    if not title or not query_words:
        return 0.0
    title_words = set(re.findall(r"[a-z]+", _base_title(title).lower()))
    overlap = len(query_words & title_words)
    return round(0.10 * overlap / len(query_words), 4)


# ── Deduplication ──────────────────────────────────────────────────────────────

def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    groups: dict[str, list[SearchResult]] = defaultdict(list)
    for r in results:
        key = _base_title(r.title) or str(r.document_id)
        groups[key].append(r)

    deduped: list[SearchResult] = []
    for group in groups.values():
        group_sorted = sorted(group, key=lambda r: r.total_score, reverse=True)
        best = group_sorted[0]
        sub_chunks = [
            {
                "document_id":   r.document_id,
                "title":         r.title,
                "content_preview": r.content_preview,
                "total_score":   r.total_score,
            }
            for r in group_sorted[1:]
        ]
        best = best.model_copy(update={
            "title":          _base_title(best.title) if len(group) > 1 else best.title,
            "chunks_matched": len(group),
            "sub_chunks":     sub_chunks if sub_chunks else None,
        })
        deduped.append(best)
    return deduped


# ── Query log ──────────────────────────────────────────────────────────────────

def _log_search(conn, query: str, results_count: int,
                candidates_evaluated: int, top_score: float | None,
                latency_ms: int) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_logs
                    (query, results_count, candidates_evaluated, top_score, latency_ms)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (query, results_count, candidates_evaluated, top_score, latency_ms),
            )
    except Exception:
        pass  # never let logging break search


# ── Public API ─────────────────────────────────────────────────────────────────

def search(
    conn,
    query: str,
    limit: int = 10,
    min_score: float = 0.05,
    weights: dict[str, float] | None = None,
) -> SearchResponse:
    t0 = time.monotonic()
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    effective_query = preprocess_query(query)
    query_words = set(effective_query.lower().split())

    total_docs = _get_total_docs(conn)
    query_genome = _get_query_genome(effective_query, total_docs, conn)
    query_hashes = [g["trigram_hash"] for g in query_genome]

    candidate_ids = _get_candidates(conn, query_hashes, settings.gin_candidate_limit)

    if not candidate_ids:
        _log_search(conn, query, 0, 0, None, int((time.monotonic() - t0) * 1000))
        return SearchResponse(
            query=query, results=[],
            candidates_evaluated=0,
            query_genome_length=len(query_genome),
        )

    # Fetch genomes + BM25 scores in parallel queries
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id, title, content, semantic_genome FROM documents WHERE id = ANY(%s)",
            (candidate_ids,),
        )
        rows = cur.fetchall()

    bm25_map = _get_bm25_scores(conn, candidate_ids, effective_query)

    results: list[SearchResult] = []
    for row in rows:
        doc_genome = row["semantic_genome"]
        if not doc_genome:
            continue
        title_boost = _compute_title_boost(query_words, row["title"])
        scores = _score_candidate(
            query_genome, doc_genome, w,
            bm25_score=bm25_map.get(row["id"], 0.0),
            title_boost=title_boost,
        )
        # Vocabulary signal gate: require at least one of trigram / phonetic / BM25
        # to have a non-zero score. Without this, documents with similar grammatical
        # structure but completely different vocabulary surface through POS alone.
        has_vocab_signal = (
            scores["trigram"] > 0
            or scores["phonetic"] > 0.05
            or scores["bm25"] > 0.01
            or title_boost > 0
        )
        if not has_vocab_signal:
            continue
        if scores["total"] < min_score:
            continue
        results.append(SearchResult(
            document_id=row["id"],
            title=row["title"],
            content_preview=row["content"],
            total_score=scores["total"],
            dimension_scores={
                k: v for k, v in scores.items()
                if k not in ("total", "matched_trigrams")
            },
            matched_trigrams=scores["matched_trigrams"],
        ))

    results.sort(key=lambda r: r.total_score, reverse=True)
    results = _deduplicate(results)
    results.sort(key=lambda r: r.total_score, reverse=True)
    final = results[:limit]

    latency_ms = int((time.monotonic() - t0) * 1000)
    top_score = final[0].total_score if final else None
    _log_search(conn, query, len(final), len(rows), top_score, latency_ms)

    return SearchResponse(
        query=query,
        results=final,
        candidates_evaluated=len(rows),
        query_genome_length=len(query_genome),
    )
