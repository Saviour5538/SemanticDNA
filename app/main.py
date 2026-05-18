import os
from contextlib import asynccontextmanager

import psycopg2
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import close_pool, init_pool
from app.genome.pos_tagger import get_nlp
from app.routes import documents, search

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema.sql")


def _apply_schema():
    """Run schema.sql on every startup. All statements are idempotent (IF NOT EXISTS / ADD COLUMN IF NOT EXISTS)."""
    conn = psycopg2.connect(settings.database_url)
    conn.autocommit = True
    with open(_SCHEMA_PATH) as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _apply_schema()          # always safe — only creates what's missing
    init_pool(settings.database_url)
    get_nlp()                # warm up spaCy model once at startup
    yield
    close_pool()


app = FastAPI(
    title="Semantic DNA API",
    version="0.1.0",
    description="Bio-inspired semantic search — no vector embeddings, pure PostgreSQL.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(search.router, prefix="/search", tags=["search"])


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/static/index.html")
