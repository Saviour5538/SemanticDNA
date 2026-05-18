from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import close_pool, init_pool
from app.genome.pos_tagger import get_nlp
from app.routes import documents, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool(settings.database_url)
    get_nlp()  # warm up spaCy model once at startup
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
