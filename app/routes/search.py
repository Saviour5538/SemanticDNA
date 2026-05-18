from fastapi import APIRouter, Query

from app.database import get_connection, get_cursor
from app.models import AnalyticsResponse, HealthResponse, SearchRequest, SearchResponse
from app.search.searcher import search as run_search

router = APIRouter()


@router.post("", response_model=SearchResponse)
def search_post(body: SearchRequest):
    with get_connection() as conn:
        return run_search(
            conn,
            query=body.query,
            limit=body.limit,
            min_score=body.min_score,
            weights=body.weights,
        )


@router.get("", response_model=SearchResponse)
def search_get(
    q: str = Query(min_length=1),
    limit: int = Query(default=10, ge=1, le=100),
    min_score: float = Query(default=0.05, ge=0.0, le=1.0),
):
    with get_connection() as conn:
        return run_search(conn, query=q, limit=limit, min_score=min_score)


@router.get("/health", response_model=HealthResponse)
def health():
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT total_docs FROM corpus_meta WHERE id = 1")
            meta = cur.fetchone()
            cur.execute("SELECT COUNT(*) AS cnt FROM corpus_stats")
            vocab = cur.fetchone()
    return HealthResponse(
        status="ok",
        total_documents=meta["total_docs"] if meta else 0,
        vocab_size=vocab["cnt"] if vocab else 0,
    )


@router.get("/analytics", response_model=AnalyticsResponse)
def analytics():
    """Return search usage stats from the query log."""
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM search_logs")
            total = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT COUNT(*) AS cnt FROM search_logs WHERE results_count = 0"
            )
            zero_results = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT ROUND(AVG(latency_ms)) AS avg_ms FROM search_logs"
            )
            avg_latency = cur.fetchone()["avg_ms"] or 0

            cur.execute(
                """
                SELECT query, COUNT(*) AS freq
                FROM search_logs
                GROUP BY query
                ORDER BY freq DESC
                LIMIT 10
                """
            )
            top_queries = [
                {"query": r["query"], "count": r["freq"]}
                for r in cur.fetchall()
            ]

            cur.execute(
                """
                SELECT query, results_count, top_score, latency_ms, searched_at
                FROM search_logs
                ORDER BY searched_at DESC
                LIMIT 20
                """
            )
            recent = [dict(r) for r in cur.fetchall()]

    return AnalyticsResponse(
        total_searches=total,
        zero_result_searches=zero_results,
        avg_latency_ms=int(avg_latency),
        top_queries=top_queries,
        recent_searches=recent,
    )
