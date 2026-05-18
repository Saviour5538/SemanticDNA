from fastapi import APIRouter, Query

from app.database import get_connection, get_cursor
from app.models import HealthResponse, SearchRequest, SearchResponse
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
    min_score: float = Query(default=0.0, ge=0.0, le=1.0),
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
