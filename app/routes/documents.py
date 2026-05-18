import io
import re
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.database import get_connection, get_cursor
from app.genome.entropy import decrement_corpus_stats, update_corpus_stats
from app.genome.pos_tagger import compute_all_pos_tags
from app.models import DocumentDetail, DocumentIngest, DocumentResponse, JobResponse, SemanticGene
from app.search.indexer import ingest_document, ingest_documents_batch
from app.search.searcher import _base_title, _score_candidate, _get_bm25_scores, DEFAULT_WEIGHTS
from app.config import settings

router = APIRouter()

# In-memory job store for async upload tracking
_jobs: dict[str, dict[str, Any]] = {}


def _row_to_response(row: dict) -> DocumentResponse:
    return DocumentResponse(
        id=row["id"],
        title=row["title"],
        content_preview=row["content"],
        genome_length=len(row["semantic_genome"] or []),
        ingested_at=row["ingested_at"],
    )


_CHUNK_SIZE = 150
_CHUNK_MIN  = 20
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


async def _extract_text(file: UploadFile) -> str:
    data = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def _chunk_text(text: str) -> list[str]:
    """Sentence-boundary chunking. Keeps sentences intact and carries the last
    sentence of each chunk into the next for context continuity."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    for sent in sentences:
        w = len(sent.split())
        if buf_words + w > _CHUNK_SIZE and buf_words >= _CHUNK_MIN:
            chunks.append(" ".join(buf))
            buf = [buf[-1]] if buf else []
            buf_words = len(buf[0].split()) if buf else 0
        buf.append(sent)
        buf_words += w

    if buf_words >= _CHUNK_MIN:
        chunks.append(" ".join(buf))

    return chunks or [text]


def _run_ingest(job_id: str, content: str, effective_title: str) -> None:
    """Background task: chunk, ingest, update job status."""
    try:
        chunks = _chunk_text(content)
        chunk_titles = [
            f"{effective_title} — Part {i + 1}" if len(chunks) > 1 else effective_title
            for i in range(len(chunks))
        ]
        total_genes = 0
        with get_connection() as conn:
            all_unique_tokens: set[str] = set()
            for chunk in chunks:
                toks, _ = compute_all_pos_tags(chunk)
                all_unique_tokens.update(t.lower() for t in toks if t.isalpha())
            update_corpus_stats(conn, list(all_unique_tokens))

            doc_ids = ingest_documents_batch(conn, list(zip(chunks, chunk_titles)))

            with get_cursor(conn) as cur:
                cur.execute(
                    "SELECT semantic_genome FROM documents WHERE id = ANY(%s)", (doc_ids,)
                )
                for row in cur.fetchall():
                    total_genes += len(row["semantic_genome"] or [])

        _jobs[job_id] = {
            "status": "done",
            "title": effective_title,
            "chunks_created": len(chunks),
            "document_ids": doc_ids,
            "total_genes": total_genes,
        }
    except Exception as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc)}


# ── Upload endpoints ───────────────────────────────────────────────────────────

@router.post("/upload", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
):
    """Upload a file and ingest it asynchronously. Returns a job_id to poll."""
    content = await _extract_text(file)
    if not content.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the file.")

    effective_title = title or file.filename
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "title": effective_title}

    background_tasks.add_task(_run_ingest, job_id, content, effective_title)

    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "processing",
                                                   "message": f"Ingesting '{effective_title}' in background."})


@router.get("/jobs/{job_id}", response_model=None)
def get_job(job_id: str):
    """Poll the status of an async upload job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ── CRUD endpoints ─────────────────────────────────────────────────────────────

@router.post("", status_code=201, response_model=DocumentResponse)
def create_document(body: DocumentIngest):
    with get_connection() as conn:
        doc_id = ingest_document(conn, body.content, body.title)
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT id, title, content, semantic_genome, ingested_at FROM documents WHERE id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
    return _row_to_response(row)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    offset = (page - 1) * page_size
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                """
                SELECT id, title, content, semantic_genome, ingested_at
                FROM documents
                ORDER BY ingested_at DESC
                LIMIT %s OFFSET %s
                """,
                (page_size, offset),
            )
            rows = cur.fetchall()
    return [_row_to_response(r) for r in rows]


@router.delete("", status_code=200)
def delete_all_documents():
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("DELETE FROM documents")
            cur.execute("DELETE FROM corpus_stats")
            cur.execute("UPDATE corpus_meta SET total_docs = 0 WHERE id = 1")
    return {"deleted": True}


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: int):
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT id, title, content, semantic_genome, ingested_at FROM documents WHERE id = %s",
                (doc_id,),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    genome = [SemanticGene(**g) for g in (row["semantic_genome"] or [])]
    return DocumentDetail(
        id=row["id"],
        title=row["title"],
        content=row["content"],
        content_preview=row["content"],
        genome_length=len(genome),
        ingested_at=row["ingested_at"],
        semantic_genome=genome,
    )


@router.get("/{doc_id}/similar")
def similar_documents(doc_id: int, limit: int = Query(default=5, ge=1, le=20)):
    """Find documents similar to doc_id using its stored genome as the query."""
    import json
    import psycopg2.extras
    from app.genome.pos_tagger import pos_sequence_similarity

    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT id, title, semantic_genome FROM documents WHERE id = %s",
                (doc_id,),
            )
            src = cur.fetchone()
        if not src:
            raise HTTPException(status_code=404, detail="Document not found")

        query_genome = src["semantic_genome"] or []
        if not query_genome:
            return {"results": []}

        query_hashes = [g["trigram_hash"] for g in query_genome]
        parts = [
            "SELECT id FROM documents WHERE semantic_genome @> %s::jsonb AND id != %s"
            for _ in query_hashes
        ]
        params: list = []
        for h in query_hashes:
            params.extend([json.dumps([{"trigram_hash": h}]), doc_id])

        sql = " UNION ".join(parts) + f" LIMIT {settings.gin_candidate_limit}"
        with conn.cursor() as cur:
            cur.execute(sql, params)
            candidate_ids = [r[0] for r in cur.fetchall()]

        if not candidate_ids:
            return {"results": []}

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, title, content, semantic_genome FROM documents WHERE id = ANY(%s)",
                (candidate_ids,),
            )
            rows = cur.fetchall()

        bm25_map = _get_bm25_scores(conn, candidate_ids, _base_title(src["title"]) or "")

        results = []
        for row in rows:
            doc_genome = row["semantic_genome"] or []
            if not doc_genome:
                continue
            scores = _score_candidate(
                query_genome, doc_genome, DEFAULT_WEIGHTS,
                bm25_score=bm25_map.get(row["id"], 0.0),
            )
            if scores["total"] < 0.05:
                continue
            results.append({
                "document_id": row["id"],
                "title": row["title"],
                "content_preview": row["content"][:400],
                "similarity_score": scores["total"],
                "dimension_scores": {
                    k: v for k, v in scores.items()
                    if k not in ("total", "matched_trigrams")
                },
            })

        results.sort(key=lambda r: r["similarity_score"], reverse=True)
        return {"source_id": doc_id, "results": results[:limit]}


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("SELECT content FROM documents WHERE id = %s", (doc_id,))
            row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        toks, _ = compute_all_pos_tags(row["content"])
        unique_tokens = list({t.lower() for t in toks if t.isalpha()})
        decrement_corpus_stats(conn, unique_tokens)
        with get_cursor(conn) as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    return JSONResponse(status_code=204, content=None)
