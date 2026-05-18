# 🧬 Semantic DNA Search

> **Bio-inspired semantic search without vector embeddings — built entirely on PostgreSQL.**

A novel approach to semantic search that encodes every document as a **semantic genome** — an array of 5-dimensional gene objects, one per 3-word sliding window. Search is a two-phase pipeline: fast GIN index pre-filtering followed by transparent, weighted multi-dimensional scoring. No ML APIs. No vector databases. Pure SQL.

---

## Table of Contents

- [Overview](#overview)
- [The Core Concept](#the-core-concept)
- [The 5 Genome Dimensions](#the-5-genome-dimensions)
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

---

## The Core Concept

Just as biological DNA encodes life using 4 nucleotides in sequences, Semantic DNA encodes text meaning using 5 dimensions in sliding 3-word windows.

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
```

Each document's genome is stored as a JSONB array in PostgreSQL with a GIN index for sub-millisecond pre-filtering.

---

## The 5 Genome Dimensions

### 1. Trigram Hash — `40% weight`
MD5 of each 3-word window (first 12 hex chars). Captures exact phrase matches and preserves word order.

- `"the quick brown"` → MD5 → `"4f2a1c9b8e3d"`
- Unlike vectors: `"quick brown fox"` will **never** accidentally match `"brown quick fox"`

### 2. Phonetic Code — `20% weight`
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

### 3. POS Sequence — `20% weight`
spaCy Universal POS tags per window, stored as `"DET-ADJ-NOUN"` strings. Finds documents with the same grammatical structure even across completely different vocabularies.

**Example:** `"The huge lion"` and `"A big bear"` both → `"DET-ADJ-NOUN"` → 100% match

### 4. Context Depth — `10% weight`
Hierarchical specificity score (1–4) per word, averaged across the window.

| Level | Category | Examples |
|---|---|---|
| L1 | Function words | the, a, is, and, in, on |
| L2 | General content | database, method, system |
| L3 | Domain-specific | pharmacokinetics, concurrency |
| L4 | Technical jargon | IPv6, XLA, HTTP2, CRISPR |

Ensures expert queries match expert documents and introductory queries match introductory content.

### 5. Entropy Score — `10% weight`
IDF-based rarity scoring: `log((N + 1) / (df(t) + smoothing))`

- Common words (`"the"`, `"use"`) → low entropy → low weight
- Rare technical terms (`"CRISPR"`, `"XLA"`) → high entropy → high weight
- Corpus stats updated at every document ingest — no retraining needed

---

## Architecture

```
                         ┌─────────────────────────────┐
    Query Text           │      FastAPI Application      │
         │               └─────────────────────────────┘
         ▼                           │
  ┌─────────────┐          ┌─────────┴──────────┐
  │ Stop-word   │          │   Genome Extractor  │
  │  Stripping  │──────────│  (5-dim per window) │
  └─────────────┘          └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  GIN Pre-filter      │  ← PostgreSQL index
                           │  @> trigram_hash     │    narrows to ~500 docs
                           └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  Weighted Scoring    │
                           │  5 dimensions        │
                           │  configurable weights│
                           └─────────────────────┘
                                     │
                           ┌─────────▼──────────┐
                           │  Ranked Results      │
                           │  + dimension scores  │
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
  Chunking (150 words, 30-word overlap)
         │
         ▼
  Corpus Stats Update (IDF)
         │
         ▼
  Genome Extraction (5 dims × N windows)
         │
         ▼
  INSERT INTO documents (JSONB genome)
```

---

## Project Structure

```
SemanticDNA/
├── schema.sql                  # Database tables + GIN index
├── requirements.txt
├── .env.example
│
├── app/
│   ├── main.py                 # FastAPI app + lifespan (pool init, spaCy warmup)
│   ├── config.py               # Settings via pydantic-settings (.env)
│   ├── database.py             # psycopg2 ThreadedConnectionPool
│   ├── models.py               # Pydantic request/response models
│   │
│   ├── genome/                 # Core encoding pipeline
│   │   ├── trigram.py          # MD5 hash of 3-word windows
│   │   ├── phonetic.py         # Vowel-strip + consonant normalization
│   │   ├── pos_tagger.py       # spaCy POS tags (singleton NLP model)
│   │   ├── context_depth.py    # L1–L4 word specificity classifier
│   │   ├── entropy.py          # IDF scoring + corpus_stats DB ops
│   │   └── extractor.py        # Orchestrates all 5 dimensions → genome list
│   │
│   ├── search/
│   │   ├── indexer.py          # Document ingest pipeline
│   │   └── searcher.py         # GIN pre-filter + weighted multi-dim scoring
│   │
│   ├── routes/
│   │   ├── documents.py        # POST/GET/DELETE /documents, POST /documents/upload
│   │   └── search.py           # POST+GET /search, GET /search/health
│   │
│   └── static/
│       └── index.html          # Single-page UI (vanilla HTML/CSS/JS)
│
└── scripts/
    ├── setup_db.py             # Run schema.sql against the database
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
GIN_CANDIDATE_LIMIT=500
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

### 5. Run the schema setup

```bash
python scripts/setup_db.py
```

This creates the `documents`, `corpus_stats`, and `corpus_meta` tables, and the GIN index.

### 6. (Optional) Load sample documents

```bash
python scripts/seed_data.py
```

Ingests 6 pre-built documents designed to exercise all 5 dimensions:
- TensorFlow XLA vulnerability (L4 depth, high entropy)
- Quick Brown Fox (L1/L2 depth, low entropy)
- Pharmacokinetics Overview (Latin roots, L3/L4 depth)
- Misspelled Pharmakokenetics (tests phonetic dimension)
- Neural Networks / GPU Clusters (tests POS + trigram)
- Commercial Aviation Maintenance (cross-domain POS match)

---

## Running the Application

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

The server:
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
  - Files > 150 words are automatically **chunked** into overlapping segments for better retrieval
- Optionally provide a title; defaults to the filename for uploads

**Right panel — Search:**
- Same search bar synced with the hero
- Results show total score + expandable dimension breakdown bars

**Status pill (top nav):** Shows live document count and vocabulary size.

---

## API Reference

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents` | Ingest plain text document |
| `POST` | `/documents/upload` | Upload file (PDF/TXT/MD) — auto-chunked |
| `GET` | `/documents` | List all documents (paginated) |
| `GET` | `/documents/{id}` | Get document with full genome |
| `DELETE` | `/documents/{id}` | Delete a document |

**Ingest text:**
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "Your document text here", "title": "My Doc"}'
```

**Upload file:**
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@report.pdf" \
  -F "title=My Report"
```

**Response:**
```json
{
  "id": 1,
  "title": "My Doc",
  "content_preview": "Your document text here...",
  "genome_length": 42,
  "ingested_at": "2025-05-18T10:30:00Z"
}
```

---

### Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/search` | Full search with body |
| `GET` | `/search?q=...&limit=10` | Quick GET search |
| `GET` | `/search/health` | Returns doc count + vocab size |

**POST search with custom weights:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "trigram hash phrase matching",
    "limit": 10,
    "min_score": 0.1,
    "weights": {
      "trigram": 0.5,
      "phonetic": 0.2,
      "pos_sequence": 0.15,
      "context_depth": 0.1,
      "entropy": 0.05
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
        "entropy": 0.93
      },
      "matched_trigrams": 3
    }
  ],
  "candidates_evaluated": 8,
  "query_genome_length": 2
}
```

**GET search (quick):**
```bash
curl "http://localhost:8000/search?q=neural+networks+computing&limit=5"
```

**Health check:**
```bash
curl http://localhost:8000/search/health
# {"status":"ok","total_documents":14,"vocab_size":312}
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

**Why?** Question words (`what`, `is`, `how`, `why`) are automatically stripped before genome extraction. But using keywords directly produces more gene windows and higher trigram overlap scores.

**Typo tolerance:** The phonetic dimension handles misspellings automatically.
```
"farmakokenetics" → finds "Pharmacokinetics" documents
"nural netwerks"  → finds "Neural Networks" documents
```

**Cross-domain matching:** The POS dimension finds structural parallels.
```
"aircraft require maintenance schedules"
→ also matches "neural networks require computational resources"
(same DET-NOUN-VERB-NOUN pattern)
```

---

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy model name |
| `GIN_CANDIDATE_LIMIT` | `500` | Max candidates from GIN pre-filter |
| `ENTROPY_SMOOTHING` | `0.5` | Laplace smoothing for IDF scores |

**Default dimension weights** (overridable per-query):

| Dimension | Weight | Rationale |
|---|---|---|
| Trigram Hash | 40% | Strongest signal for semantic match |
| Phonetic Code | 20% | Noise tolerance |
| POS Sequence | 20% | Structure match |
| Context Depth | 10% | Expertise level alignment |
| Entropy Score | 10% | Rare-term boosting |

---

## How Scoring Works

### Phase 1 — GIN Pre-filter

For each trigram hash in the query genome, one `@>` containment query is fired against the GIN index:

```sql
SELECT id FROM documents WHERE semantic_genome @> '[{"trigram_hash":"a3f8c21d"}]'::jsonb
UNION
SELECT id FROM documents WHERE semantic_genome @> '[{"trigram_hash":"9b8e3df2"}]'::jsonb
LIMIT 500
```

If GIN returns zero results (e.g., a misspelled query), the system falls back to scoring all documents — ensuring the phonetic and POS dimensions can still surface relevant results.

### Phase 2 — Weighted Scoring

Each candidate document is scored across 5 dimensions:

```
Trigram Jaccard    = |query_hashes ∩ doc_hashes| / |query_hashes ∪ doc_hashes|

Phonetic Jaccard   = shared phonetic codes / total phonetic codes
                     (word-level, split on "-" for per-word comparison)

POS Score          = avg(max(pos_similarity(qp, dp) for dp in doc) for qp in query)
                     where pos_similarity = position-wise tag match ratio

Context Depth      = 1.0 - |avg_query_depth - avg_doc_depth| / 3.0

Entropy Alignment  = 1.0 - |avg_query_entropy - avg_doc_entropy| / max(both)

Total = 0.40×trigram + 0.20×phonetic + 0.20×pos + 0.10×depth + 0.10×entropy
```

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
