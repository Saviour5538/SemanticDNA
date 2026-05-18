# 🧬 Semantic DNA Search

> **Bio-inspired semantic search without vector embeddings — built entirely on PostgreSQL.**

A novel approach to semantic search that encodes every document as a **semantic genome** — an array of 6-dimensional gene objects, one per 3-word sliding window. Search is a two-phase pipeline: fast GIN index pre-filtering followed by transparent, weighted multi-dimensional scoring. No ML APIs. No vector databases. Pure SQL.

---

## Table of Contents

- [Overview](#overview)
- [The Core Concept](#the-core-concept)
- [The 6 Genome Dimensions](#the-6-genome-dimensions)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Using the UI](#using-the-ui)
- [API Reference](#api-reference)
- [Search Tips](#search-tips)
- [Configuration](#configuration)
- [How Scoring Works](#how-scoring-works)
- [Tech Stack](#tech-stack)

---

## Overview

Traditional semantic search collapses text into opaque fixed-length vectors via external ML models, losing word order, grammatical structure, and explainability. **Semantic DNA** takes a different approach:

| Traditional (Vector) | Semantic DNA |
|---|---|
| Requires ML model / API | Pure Python + SQL |
| Black-box similarity score | Fully explainable per-dimension breakdown |
| Fixed-size dense vector | Variable-length genome array |
| Word order lost in compression | Word order preserved via trigrams |
| Requires vector database | Standard PostgreSQL |
| GPU resources for embedding | Runs on any machine |
| Re-embed entire corpus on model upgrade | Incremental corpus stats update |

---

## The Core Concept

Just as biological DNA encodes life using 4 nucleotides in sequences, Semantic DNA encodes text meaning using 6 dimensions in sliding 3-word windows.

```
"neural networks need computing power"
         ↓
┌─────────────────────────────────────────────────────┐
│  Window 1: [neural, networks, need]                 │
│  → trigram_hash:  "a3f8c21d0b9e"                   │
│  → phonetic_code: "NRL-NTWRKS-ND"                  │
│  → pos_sequence:  "ADJ-NOUN-VERB"                   │
│  → context_depth: 3                                 │
│  → entropy_score: 2.847                             │
├─────────────────────────────────────────────────────┤
│  Window 2: [networks, need, computing]              │
│  → ...                                              │
└─────────────────────────────────────────────────────┘
         +
  BM25 (ts_rank) scored separately per document
```

Each document's genome is stored as a JSONB array in PostgreSQL with a GIN index for fast pre-filtering.

---

## The 6 Genome Dimensions

### 1. Trigram Hash — `35% weight`
MD5 of each 3-word window (first 12 hex chars). Captures exact phrase matches and preserves word order. Uses lemmatized tokens so "running" and "ran" share the same hash.

- `"the quick brown"` → MD5 → `"4f2a1c9b8e3d"`
- Unlike vectors: `"quick brown fox"` will **never** accidentally match `"brown quick fox"`

### 2. Phonetic Code — `17% weight`
Vowel-stripped, consonant-normalized sound skeleton. Enables typo and misspelling tolerance.

**Transformations applied (in order):**
```
ck→k, ph→f, gh→f, wh→w, qu→kw
x→ks, soft-c→s, c→k, z→s
collapse doubles (nn→n, ll→l, ...)
remove vowels (a, e, i, o, u)
UPPERCASE result
```

**Example:** `"Pharmacokinetics"` and `"Farmakokenetics"` both → `"FRMKKNTKS"`

### 3. POS Sequence — `17% weight`
spaCy Universal POS tags per window, stored as `"DET-ADJ-NOUN"` strings. Finds documents with the same grammatical structure even across completely different vocabularies.

**Example:** `"The huge lion"` and `"A big bear"` both → `"DET-ADJ-NOUN"` → 100% match

### 4. Context Depth — `9% weight`
Hierarchical specificity score (1–4) per word, averaged across the window.

| Level | Category | Examples |
|---|---|---|
| L1 | Function words | the, a, is, and, in, on |
| L2 | General content | database, method, system |
| L3 | Domain-specific | pharmacokinetics, concurrency |
| L4 | Technical jargon | IPv6, XLA, HTTP2, CRISPR |

Ensures expert queries match expert documents and introductory queries match introductory content.

### 5. Entropy Score — `9% weight`
IDF-based rarity scoring: `log((N + 1) / (df(t) + smoothing))`

- Common words (`"the"`, `"use"`) → low entropy → low weight
- Rare technical terms (`"CRISPR"`, `"XLA"`) → high entropy → high weight
- Corpus stats updated at every document ingest — no retraining needed
- Corpus stats decremented when a document is deleted

### 6. BM25 — `13% weight`
PostgreSQL `ts_rank` on a functional GIN index over `to_tsvector('english', title || content)`. Provides strong keyword relevance signal and ensures documents with no vocabulary overlap with the query are filtered out via the **vocabulary signal gate**.

**Vocabulary signal gate:** A result is dropped entirely if all of the following are zero:
- Trigram overlap
- Phonetic overlap (≤ 0.05)
- BM25 score (≤ 0.01)
- Title boost

This prevents grammatically-similar but semantically-unrelated documents from surfacing.

---

## Architecture

```
                         ┌─────────────────────────────┐
    Query Text           │      FastAPI Application      │
         │               └─────────────────────────────┘
         ▼                           │
  ┌─────────────┐          ┌─────────┴──────────┐
  │ Stop-word   │          │   Genome Extractor  │
  │  Stripping  │──────────│  (6-dim per window) │
  └─────────────┘          │  + genome cache      │
                           └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  GIN Pre-filter      │  ← PostgreSQL JSONB index
                           │  @> trigram_hash     │    narrows to ~1000 docs
                           └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  BM25 via ts_rank    │  ← Functional GIN index
                           │  (parallel SQL)      │    on tsvector column
                           └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  Weighted Scoring    │
                           │  6 dimensions        │
                           │  vocabulary gate     │
                           │  title boost         │
                           └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  Deduplication       │
                           │  (group by title,    │
                           │   keep best chunk)   │
                           └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  Ranked Results      │
                           │  + dimension scores  │
                           │  + sub-chunks        │
                           └─────────────────────┘
```

**Document Ingest Pipeline:**
```
File Upload / Text Paste
         │
         ▼
  Text Extraction (PDF/TXT/MD)
         │
         ▼
  Sentence-boundary Chunking (150 words, last sentence carried over)
         │
         ▼
  Corpus Stats Update (IDF) — once per source document
         │
         ▼
  Batch Genome Extraction (spaCy nlp.pipe across all chunks)
         │
         ▼
  Batch INSERT INTO documents (JSONB genome)
         │
         ▼
  Job status → "done" (async, polled by UI)
```

---

## Project Structure

```
SemanticDNA/
├── schema.sql                  # Database tables + GIN indexes (auto-applied on startup)
├── requirements.txt
├── .env.example
│
├── app/
│   ├── main.py                 # FastAPI app + lifespan (schema, pool init, spaCy warmup)
│   ├── config.py               # Settings via pydantic-settings (.env)
│   ├── database.py             # psycopg2 ThreadedConnectionPool
│   ├── models.py               # Pydantic request/response models
│   │
│   ├── genome/                 # Core encoding pipeline
│   │   ├── trigram.py          # MD5 hash of 3-word windows
│   │   ├── phonetic.py         # Vowel-strip + consonant normalization
│   │   ├── pos_tagger.py       # spaCy POS tags + lemmatization (singleton NLP model)
│   │   ├── context_depth.py    # L1-L4 word specificity classifier
│   │   ├── entropy.py          # IDF scoring + corpus_stats DB ops
│   │   └── extractor.py        # Orchestrates all dimensions; batch mode via nlp.pipe
│   │
│   ├── search/
│   │   ├── indexer.py          # Document ingest pipeline (single + batch)
│   │   └── searcher.py         # GIN pre-filter + BM25 + weighted scoring + dedup
│   │
│   ├── routes/
│   │   ├── documents.py        # CRUD + async upload + job polling + similarity endpoint
│   │   └── search.py           # POST+GET /search, /health, /analytics
│   │
│   └── static/
│       └── index.html          # Single-page UI (vanilla HTML/CSS/JS)
│
└── scripts/
    ├── setup_db.py             # Manual schema runner (optional — auto-runs on startup)
    └── seed_data.py            # Ingest 6 sample documents via the API
```

---

## Prerequisites

- **Python 3.10+**
- **PostgreSQL 14+** (running locally or remote)
- No GPU required. No external ML APIs.

---

## Setup & Installation

### 1. Clone / navigate to the project

```bash
cd SemanticDNA
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure environment

Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:

```bash
cp .env.example .env
```

```env
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/semanticdna
SPACY_MODEL=en_core_web_sm
GIN_CANDIDATE_LIMIT=1000
ENTROPY_SMOOTHING=0.5
```

### 4. Create the database

```bash
# Windows (PowerShell)
$env:PGPASSWORD = "your_password"
& "C:\Program Files\PostgreSQL\16\bin\psql.exe" -U postgres -c "CREATE DATABASE semanticdna;"

# Linux / macOS
createdb -U postgres semanticdna
```

### 5. (Optional) Load sample documents

```bash
python scripts/seed_data.py
```

Ingests 6 pre-built documents designed to exercise all dimensions:
- TensorFlow XLA vulnerability (L4 depth, high entropy)
- Quick Brown Fox (L1/L2 depth, low entropy)
- Pharmacokinetics Overview (Latin roots, L3/L4 depth)
- Misspelled Pharmakokenetics (tests phonetic dimension)
- Neural Networks / GPU Clusters (tests POS + trigram)
- Commercial Aviation Maintenance (cross-domain POS match)

> **Note:** The database schema is applied automatically on every server startup. You do not need to run `setup_db.py` manually.

---

## Running the Application

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

On startup the server:
- Applies the database schema automatically (idempotent — safe to run repeatedly)
- Initializes the PostgreSQL connection pool
- Warms up the spaCy NLP model (avoids cold-start latency on first request)
- Serves the browser UI via `/static/index.html`

---

## Using the UI

The UI is a SaaS-style single-page application with two areas:

### Hero Search Bar
Type a query and press Enter or click **Search →**. Results appear below the hero section with per-document dimension score breakdowns.

### App Workspace (scroll down)

**Left panel — Ingest Document:**
- **Paste Text tab:** Paste any text content directly
- **Upload File tab:** Upload `.pdf`, `.txt`, or `.md` files
  - PDF text is extracted automatically via `pypdf`
  - Files are automatically **chunked** at sentence boundaries (150-word target) for better retrieval
  - Ingestion happens asynchronously — a progress message polls until complete
- Optionally provide a title; defaults to the filename for uploads
- **Clear All** button removes all indexed documents and resets corpus stats

**Right panel — Search:**
- Results show total score + **Show breakdown** (expandable dimension bars for all 6 dimensions)
- **View full content** expands to show the complete chunk text
- Multi-chunk documents show the best-matching chunk with a **Show other chunk(s)** toggle
- **Doc similarity** is available via the API — see `/documents/{id}/similar`

**Status pill (top nav):** Shows live document count and vocabulary size.

---

## API Reference

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents` | Ingest plain text document (synchronous) |
| `POST` | `/documents/upload` | Upload file (PDF/TXT/MD) — async, returns job_id |
| `GET` | `/documents/jobs/{job_id}` | Poll async upload job status |
| `GET` | `/documents` | List all documents (paginated) |
| `GET` | `/documents/{id}` | Get document with full genome |
| `GET` | `/documents/{id}/similar` | Find similar documents by genome |
| `DELETE` | `/documents/{id}` | Delete document + decrement corpus stats |
| `DELETE` | `/documents` | Delete all documents + reset corpus |

**Ingest text:**
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "Your document text here", "title": "My Doc"}'
```

**Upload file (async):**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf" \
  -F "title=My Report"
# Returns: {"job_id": "uuid", "status": "processing"}

curl http://localhost:8000/documents/jobs/{job_id}
# Returns: {"status": "done", "chunks_created": 4, "total_genes": 312}
```

**Find similar documents:**
```bash
curl http://localhost:8000/documents/5/similar?limit=5
```

---

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Full search with body |
| `GET` | `/search?q=...&limit=10` | Quick GET search |
| `GET` | `/search/health` | Returns doc count + vocab size |
| `GET` | `/search/analytics` | Query logs, top queries, avg latency |

**POST search with custom weights:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "trigram hash phrase matching",
    "limit": 10,
    "min_score": 0.05,
    "weights": {
      "trigram": 0.5,
      "phonetic": 0.15,
      "pos_sequence": 0.15,
      "context_depth": 0.08,
      "entropy": 0.07,
      "bm25": 0.05
    }
  }'
```

**Response:**
```json
{
  "query": "trigram hash phrase matching",
  "results": [
    {
      "document_id": 3,
      "title": "Semantic DNA — Part 2",
      "content_preview": "A hashing mechanism that converts a sliding window...",
      "total_score": 0.714,
      "dimension_scores": {
        "trigram": 0.82,
        "phonetic": 0.61,
        "pos_sequence": 0.67,
        "context_depth": 0.85,
        "entropy": 0.93,
        "bm25": 0.74,
        "title_boost": 0.10
      },
      "matched_trigrams": 3,
      "chunks_matched": 2,
      "sub_chunks": [...]
    }
  ],
  "candidates_evaluated": 18,
  "query_genome_length": 4
}
```

**Analytics:**
```bash
curl http://localhost:8000/search/analytics
# Returns top queries, zero-result rate, avg latency, recent searches
```

---

## Search Tips

The system is a **document retrieval engine**, not a Q&A chatbot. It finds relevant documents — not extracted answers.

| Do this | Not this |
|---|---|
| `trigram hash phrase matching` | `what is a trigram hash?` |
| `phonetic typo normalization` | `how does phonetic code work?` |
| `entropy IDF rare terms` | `what is entropy score?` |
| `neural networks GPU computing` | `tell me about neural networks` |

**Why?** Question words (`what`, `is`, `how`, `why`) are automatically stripped before genome extraction. Keywords directly produce more gene windows and higher trigram overlap scores.

**Typo tolerance:** The phonetic dimension handles misspellings automatically.
```
"farmakokenetics" → finds "Pharmacokinetics" documents
"nural netwerks"  → finds "Neural Networks" documents
```

**Cross-domain matching:** The POS dimension finds structural parallels.
```
"aircraft require maintenance schedules"
→ also matches "neural networks require computational resources"
(same NOUN-VERB-NOUN-NOUN pattern)
```

---

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model name |
| `GIN_CANDIDATE_LIMIT` | `1000` | Max candidates from GIN pre-filter |
| `ENTROPY_SMOOTHING` | `0.5` | Laplace smoothing for IDF scores |

**Default dimension weights** (overridable per-query via the `weights` field):

| Dimension | Weight | Rationale |
|---|---|---|
| Trigram Hash | 35% | Strongest signal — exact phrase overlap |
| BM25 | 13% | Keyword relevance via PostgreSQL ts_rank |
| Phonetic Code | 17% | Typo and misspelling tolerance |
| POS Sequence | 17% | Grammatical structure match |
| Context Depth | 9% | Expertise level alignment |
| Entropy Score | 9% | Rare-term boosting |

---

## How Scoring Works

### Phase 1 — GIN Pre-filter

For each trigram hash in the query genome, one `@>` containment query hits the JSONB GIN index:

```sql
SELECT id FROM documents WHERE semantic_genome @> '[{"trigram_hash":"a3f8c21d"}]'::jsonb
UNION
SELECT id FROM documents WHERE semantic_genome @> '[{"trigram_hash":"9b8e3df2"}]'::jsonb
LIMIT 1000
```

If GIN returns zero results (e.g. a misspelled query), the system falls back to scoring all documents — ensuring the phonetic and POS dimensions can still surface relevant results.

### Phase 2 — BM25 (parallel)

Simultaneously, `ts_rank` is computed against a functional GIN index:

```sql
SELECT id, ts_rank(to_tsvector('english', coalesce(title,'') || ' ' || content),
                   plainto_tsquery('english', $query)) AS bm25
FROM documents WHERE id = ANY($candidate_ids)
```

### Phase 3 — Weighted Scoring + Gate

Each candidate is scored across all 6 dimensions:

```
Trigram Jaccard    = |query_hashes ∩ doc_hashes| / |query_hashes ∪ doc_hashes|
Phonetic Jaccard   = shared phonetic codes / total phonetic codes
POS Score          = avg(max(pos_similarity(qp, dp) for dp in doc) for qp in query)
Context Depth      = 1.0 - |avg_query_depth - avg_doc_depth| / 3.0
Entropy Alignment  = 1.0 - |avg_query_entropy - avg_doc_entropy| / max(both)
BM25               = min(ts_rank × 10, 1.0)
Title Boost        = +0.10 additive when query words appear in doc title

Total = 0.35×trigram + 0.17×phonetic + 0.17×pos + 0.09×depth + 0.09×entropy + 0.13×bm25 + title_boost
      (capped at 1.0)
```

**Vocabulary signal gate:** result dropped if trigram = 0 AND phonetic ≤ 0.05 AND bm25 ≤ 0.01 AND title_boost = 0.

### Phase 4 — Deduplication

Chunks from the same source document (matched by base title, stripping `— Part N` suffixes) are collapsed: the highest-scoring chunk is the primary result; remaining chunks appear as expandable `sub_chunks`.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | [FastAPI](https://fastapi.tiangolo.com/) 0.115 |
| ASGI server | [uvicorn](https://www.uvicorn.org/) |
| Database | [PostgreSQL](https://www.postgresql.org/) 14+ |
| DB driver | [psycopg2](https://www.psycopg.org/) with ThreadedConnectionPool |
| NLP / POS tagging | [spaCy](https://spacy.io/) `en_core_web_sm` |
| PDF extraction | [pypdf](https://github.com/py-pdf/pypdf) |
| File uploads | python-multipart |
| Settings | pydantic-settings |
| UI | Vanilla HTML / CSS / JavaScript (no framework) |

---

## Inspired By

> *"The future of search is not a single model, but an ecosystem of complementary methods."*

This project implements the **Semantic DNA Sequencing** concept introduced by Rahuul Siingh — a bio-inspired framework for explainable semantic search that operates entirely within standard database infrastructure.
