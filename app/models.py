from datetime import datetime

from pydantic import BaseModel, Field


class SemanticGene(BaseModel):
    trigram_hash: str
    phonetic_code: str
    pos_sequence: str
    context_depth: int = Field(ge=1, le=4)
    entropy_score: float


class DocumentIngest(BaseModel):
    content: str = Field(min_length=1)
    title: str | None = None


class DocumentResponse(BaseModel):
    id: int
    title: str | None
    content_preview: str
    genome_length: int
    ingested_at: datetime


class DocumentDetail(DocumentResponse):
    content: str
    semantic_genome: list[SemanticGene]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)
    weights: dict[str, float] | None = None


class SearchResult(BaseModel):
    document_id: int
    title: str | None
    content_preview: str
    total_score: float
    dimension_scores: dict[str, float]
    matched_trigrams: int
    chunks_matched: int = 1
    sub_chunks: list[dict] | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    candidates_evaluated: int
    query_genome_length: int


class HealthResponse(BaseModel):
    status: str
    total_documents: int
    vocab_size: int


class AnalyticsResponse(BaseModel):
    total_searches: int
    zero_result_searches: int
    avg_latency_ms: int
    top_queries: list[dict]
    recent_searches: list[dict]


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str
