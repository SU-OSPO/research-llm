# research-llm — Ingestion & Summarization Pipeline

A local pipeline that builds a research-paper knowledge base for Syracuse
University publications and provides a summarization/chatbot workflow on top of
it. The codebase is organized into **five files** across two independent
projects:

| File | Project | Role |
| --- | --- | --- |
| `config.py` | RAG | Central, environment-overridable configuration |
| `fetch.py` | RAG | Acquire data: OpenAlex fetch/download + Docling processing |
| `ingest.py` | RAG | Transform & index: normalize → chunk → ChromaDB / Neo4j |
| `main.py` | RAG | Orchestrator that runs the full RAG pipeline |
| `summarizer.py` | Summarizer | T5/LLaMA paper summarizer + chatbot (standalone) |

The **RAG pipeline** (`config` + `fetch` + `ingest` + `main`) turns the OpenAlex
API into two vector collections and a knowledge graph. The **summarizer**
(`summarizer.py`) is a separate tool that extracts text from local PDFs into
SQLite, summarizes it with T5, and answers questions with LLaMA.

---

## Overview

The RAG pipeline does four things, in order:

1. **Fetch** all Syracuse works + authors from OpenAlex and download fulltexts
2. **Process** downloaded PDFs/TEI through Docling into structured sections
3. **Normalize** the raw records into clean, deduplicated JSONL
4. **Index** the results into ChromaDB (full + abstract-only collections) and Neo4j

The summarizer is independent and does three things: extract PDF text into a
SQLite `works` table, generate T5 summaries, and optionally fine-tune — plus an
interactive LLaMA chatbot grounded in the stored summaries.

---

## Architecture

### Data files (RAG)

All paths are relative to `DATA_DIR` (default `data/`) and configurable.

| File | Produced by | Contents |
| --- | --- | --- |
| `raw/works.jsonl`, `raw/authors.jsonl` | `fetch` | Raw OpenAlex records |
| `raw/works_with_fulltext.jsonl` | `fetch` | Works + `fulltext_status` / `fulltext_path` |
| `fulltext/{id}.tei.xml` \| `{id}.pdf` | `fetch` | Downloaded fulltexts |
| `raw/works_with_docling.jsonl` | `fetch` | Works + `docling_status` / `docling_path` |
| `docling/{id}.json` | `fetch` | Structured sections per work |
| `raw/normalized_works.jsonl`, `raw/normalized_authors.jsonl` | `ingest` | Clean records |
| `chroma_db/` (collection `syracuse_papers`) | `ingest` | Full chunked vectors |
| `chroma_abstracts/` (collection `syracuse_abstracts`) | `ingest` | Abstract-only vectors |
| `sync_state.json` | `fetch` | Incremental-fetch watermark |

### Status values

* `fulltext_status`: `tei_xml` · `pdf` · `none`
* `docling_status`: `docling_ok` · `fallback_pdf` · `none`

### Chunk types (ChromaDB)

`chunk_work()` emits: `title_abstract`, `keywords`, `section` (or
`fallback_text` when Docling fell back to pdfminer), `table`, `figure_caption`.
The abstract-only collection emits just `title_abstract` and `keywords`.

### Neo4j graph model

* **Nodes:** `Author` · `Work` · `Topic`
* **Edges:** `AUTHORED` (position, is_corresponding) · `HAS_TOPIC` (score) ·
  `CITES` · `COLLABORATES_WITH` (shared_works, derived)

### Summarizer database

A single SQLite `works` table: `id`, `file_name` (unique), `full_text`,
`summary`, `summary_status` (default `unsummarized`), `progress` (default `0`).

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Dependencies are grouped by stage — install only what you need:

```bash
# Fetch/download
pip install requests

# Docling processing (torch optional but strongly recommended for speed)
pip install docling pdfminer.six pypdf
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Indexing
pip install chromadb langchain-huggingface sentence-transformers

# Neo4j (also requires a running Neo4j server)
pip install neo4j

# Summarizer
pip install transformers datasets torch pandas pdfminer.six pypdf
```

---

## Running

### Full RAG pipeline

```bash
python main.py                       # fetch → docling → normalize → chroma → neo4j
python main.py --module fetch        # fetch + download only
python main.py --module docling      # Docling only
python main.py --module ingest       # normalize + chroma + neo4j
python main.py --skip-download       # fetch metadata only, no PDFs
python main.py --skip-docling        # skip Docling processing
python main.py --skip-neo4j          # skip the graph build
python main.py --abstracts           # also build the abstract-only collection
python main.py --incremental         # only new/updated records
```

### Individual stages

Each RAG file also runs standalone with its own subcommands:

```bash
python fetch.py fetch [--incremental] [--skip-download]
python fetch.py docling [--incremental]
python fetch.py all [--incremental] [--skip-download]

python ingest.py normalize
python ingest.py chroma     [--incremental]
python ingest.py abstracts  [--incremental] [--chroma-dir DIR] [--collection NAME]
python ingest.py neo4j      [--incremental]
python ingest.py all        [--incremental] [--skip-neo4j] [--abstracts]
```

### Summarizer

```bash
python summarizer.py build-db     # extract PDF text → SQLite
python summarizer.py summarize    # T5-summarize unsummarized works
python summarizer.py fine-tune    # fine-tune T5 on (text, summary) pairs
python summarizer.py pipeline     # build-db → summarize → fine-tune
python summarizer.py chat         # interactive LLaMA chatbot over summaries
```

> `--incremental` upserts into existing collections and only processes new
> records; without it, each stage rebuilds from scratch.

---

## Module Reference

### `config.py`

Central configuration. Every value is overridable by an environment variable of
the same name. Groups: OpenAlex (institution ID, API key, email, paging), paths,
ChromaDB (full + abstracts), Neo4j, embedding model/device, chunking, and
download tuning. No hardcoded paths live anywhere else in the RAG pipeline.

### `fetch.py`

Data acquisition. Two entry points:

* **`run_fetch(incremental=False, skip_download=False)`** — pulls works +
  authors from OpenAlex (cursor-paginated, rate-limited), reconstructs abstracts
  from the inverted index, and downloads fulltext per work in priority order:
  OpenAlex TEI → open-access PDF → Unpaywall. Resumable; supports an incremental
  watermark via `sync_state.json`.
* **`run_docling(incremental=False)`** — runs downloaded PDFs through Docling
  (GPU-aware) into structured sections, with a pdfminer/pypdf fallback. Safe to
  interrupt and resume; already-processed files on disk are auto-recovered.

### `ingest.py`

Transform & index. Entry points:

* **`run_normalize()`** — raw OpenAlex JSONL → normalized works/authors. Strips
  HTML, resolves Syracuse authorship, extracts topics/keywords, and
  auto-selects the richest available input (docling > fulltext > raw).
* **`run_chroma(rebuild=True)`** — chunks each work (`chunk_work`) and indexes the
  full collection into ChromaDB using normalized HuggingFace embeddings.
* **`run_abstracts(rebuild=True, ...)`** — builds a lightweight title+abstract
  collection (skips works whose abstract is missing or under 20 characters).
* **`run_neo4j(rebuild=True)`** — builds the knowledge graph and derives
  `COLLABORATES_WITH` edges in batches.

Also exposes the reusable helpers `normalize_work`, `normalize_author`, and
`chunk_work`.

### `main.py`

Orchestrator. Wires the stages together with `--module` selection and
`--skip-*` / `--incremental` / `--abstracts` flags, logging timing and a summary
per stage. Imports each stage lazily so `--help` and partial runs work without
the heavy dependencies installed.

### `summarizer.py`

Standalone T5/LLaMA project. Extracts PDF text into SQLite, summarizes with T5,
optionally fine-tunes, and serves a LLaMA chatbot over the stored summaries.
Models and the database connection load **lazily** (on first use), so importing
the module — or running a single subcommand — never loads both models at once.

---

## Data Flow

```mermaid
flowchart TD
    A[OpenAlex API] --> B[fetch.run_fetch]
    B --> C[(raw/works.jsonl<br/>raw/authors.jsonl)]
    B --> D[(fulltext/*.pdf | *.tei.xml)]
    B --> E[(works_with_fulltext.jsonl)]

    E --> F[fetch.run_docling]
    D --> F
    F --> G[(docling/*.json)]
    F --> H[(works_with_docling.jsonl)]

    H --> I[ingest.run_normalize]
    I --> J[(normalized_works.jsonl<br/>normalized_authors.jsonl)]

    J --> K[ingest.run_chroma]      --> K1[[ChromaDB: syracuse_papers]]
    J --> L[ingest.run_abstracts]   --> L1[[ChromaDB: syracuse_abstracts]]
    J --> M[ingest.run_neo4j]       --> M1[[Neo4j graph]]
```

The summarizer runs independently:

```text
local PDFs
  -> summarizer build-db   -> SQLite works.full_text
  -> summarizer summarize  -> works.summary (T5)
  -> summarizer fine-tune  -> fine-tuned T5
  -> summarizer chat       -> LLaMA answers grounded in works.summary
```

---

## Configuration Reference

Set any of these as environment variables to override the defaults.

### RAG (`config.py`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENALEX_INSTITUTION_ID` | `I70983195` | Syracuse University |
| `OPENALEX_API_KEY` / `OPENALEX_EMAIL` | *(empty)* | OpenAlex auth / polite pool |
| `DATA_DIR` | `data` | Root for all pipeline outputs |
| `CHROMA_DIR` / `CHROMA_COLLECTION` | `data/chroma_db` / `syracuse_papers` | Full collection |
| `CHROMA_ABSTRACTS_DIR` / `CHROMA_ABSTRACTS_COLLECTION` | `data/chroma_abstracts` / `syracuse_abstracts` | Abstract collection |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | `bolt://localhost:7687` / `neo4j` / … / `syr-rag-abstracts` | Graph connection |
| `EMBED_MODEL` / `EMBED_DEVICE` | `sentence-transformers/all-MiniLM-L6-v2` / `cpu` | Embeddings |
| `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `512` / `128` | Chunking |
| `DOWNLOAD_WORKERS` / `DOWNLOAD_TIMEOUT` / `DOWNLOAD_MAX_RETRIES` | `8` / `30` / `3` | Download tuning |
| `UNPAYWALL_EMAIL` | `OPENALEX_EMAIL` | Unpaywall lookups |

### Summarizer (`summarizer.py`)

| Variable | Default |
| --- | --- |
| `T5_DB_PATH` | `C:\codes\t5-db\researchers.db` |
| `T5_PDF_FOLDER` | `C:\codes\t5-db\download_pdfs` |
| `T5_MODEL` | `t5-small` |
| `T5_FINETUNE_DIR` | `C:\codes\t5-db\fine_tuned_t5` |
| `LLAMA_MODEL_PATH` | `C:\codes\llama32\Llama-3.2-1B-Instruct` |
| `LLAMA_FINETUNE_DIR` | `C:\codes\llama32\fine_tuned_llama` |

---

## Notes

* **Provenance.** These five files were refactored from an earlier 20-file
  layout; overlapping/duplicate helpers were merged and colliding names
  resolved. A verbatim archive of the original 20 files is preserved separately
  in `consolidated_pipeline.py`.
* **No `config_full` / `pdf_pre` needed.** The old code referenced a
  `config_full` module and a `pdf_pre` module that were never part of the repo.
  The RAG pipeline now uses `config.py` for everything, and the summarizer ships
  its own `extract_text_from_pdf` (pdfminer → pypdf).
* **Graceful degradation.** Every stage checks for its input and reports clearly
  if a prerequisite step hasn't run; Neo4j is skipped automatically when the
  driver isn't installed.