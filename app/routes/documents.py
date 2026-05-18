import io

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.database import get_connection, get_cursor
from app.genome.entropy import update_corpus_stats
from app.genome.pos_tagger import compute_all_pos_tags
from app.models import DocumentDetail, DocumentIngest, DocumentResponse, SemanticGene
from app.search.indexer import ingest_document, ingest_documents_batch

router = APIRouter()


def _row_to_response(row: dict) -> DocumentResponse:
    return DocumentResponse(
        id=row["id"],
        title=row["title"],
        content_preview=row["content"][:200],
        genome_length=len(row["semantic_genome"] or []),
        ingested_at=row["ingested_at"],
    )


_CHUNK_SIZE = 150   # words per chunk
_CHUNK_OVERLAP = 30 # words of overlap between consecutive chunks


async def _extract_text(file: UploadFile) -> str:
    data = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= _CHUNK_SIZE:
        return [text]
    chunks = []
    step = _CHUNK_SIZE - _CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i : i + _CHUNK_SIZE]
        if len(chunk_words) < 20:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
):
    content = await _extract_text(file)
    if not content.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the file.")
    effective_title = title or file.filename
    chunks = _chunk_text(content)
    chunk_titles = [
        f"{effective_title} — Part {i + 1}" if len(chunks) > 1 else effective_title
        for i, _ in enumerate(chunks)
    ]
    doc_ids = []
    total_genes = 0
    with get_connection() as conn:
        # Fix 1: collect unique tokens across ALL chunks and update corpus_stats
        # exactly once so IDF doc_count reflects the source document, not chunk count.
        all_unique_tokens: set[str] = set()
        for chunk in chunks:
            toks, _ = compute_all_pos_tags(chunk)
            all_unique_tokens.update(t.lower() for t in toks if t.isalpha())
        update_corpus_stats(conn, list(all_unique_tokens))

        # Fix 2: ingest all chunks in one batch (single nlp.pipe pass + one DB stats fetch)
        doc_ids = ingest_documents_batch(conn, list(zip(chunks, chunk_titles)))

        with get_cursor(conn) as cur:
            cur.execute(
                "SELECT semantic_genome FROM documents WHERE id = ANY(%s)",
                (doc_ids,),
            )
            for row in cur.fetchall():
                total_genes += len(row["semantic_genome"] or [])
    return JSONResponse(
        status_code=201,
        content={
            "title": effective_title,
            "chunks_created": len(chunks),
            "document_ids": doc_ids,
            "total_genes": total_genes,
        },
    )


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
        content_preview=row["content"][:200],
        genome_length=len(genome),
        ingested_at=row["ingested_at"],
        semantic_genome=genome,
    )


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: int):
    with get_connection() as conn:
        with get_cursor(conn) as cur:
            cur.execute("DELETE FROM documents WHERE id = %s RETURNING id", (doc_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail="Document not found")
    return JSONResponse(status_code=204, content=None)
