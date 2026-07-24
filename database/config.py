"""
config.py — Central configuration for the Syracuse RAG pipeline.

Every value is overridable via an environment variable of the same name.
Imported by fetch.py, ingest.py and main.py.
"""
import os


def _env(name, default):
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default

def _env_int(name, default):
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default

def _env_float(name, default):
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default

def _env_bool(name, default):
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes")


# ── OpenAlex ──────────────────────────────────────────────────────────────────
OPENALEX_INSTITUTION_ID = _env("OPENALEX_INSTITUTION_ID", "I70983195")  # Syracuse University
OPENALEX_API_KEY        = _env("OPENALEX_API_KEY", "")
OPENALEX_EMAIL          = _env("OPENALEX_EMAIL", "")
OPENALEX_PER_PAGE       = _env_int("OPENALEX_PER_PAGE", 100)
OPENALEX_RATE_DELAY     = _env_float("OPENALEX_RATE_DELAY", 0.12)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = _env("DATA_DIR", "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
FULLTEXT_DIR    = os.path.join(DATA_DIR, "fulltext")
DOCLING_DIR     = os.path.join(DATA_DIR, "docling")
SYNC_STATE_FILE = os.path.join(DATA_DIR, "sync_state.json")

# ── ChromaDB — full chunked collection ────────────────────────────────────────
CHROMA_DIR        = _env("CHROMA_DIR", os.path.join(DATA_DIR, "chroma_db"))
CHROMA_COLLECTION = _env("CHROMA_COLLECTION", "syracuse_papers")

# ── ChromaDB — abstract-only collection ───────────────────────────────────────
CHROMA_ABSTRACTS_DIR        = _env("CHROMA_ABSTRACTS_DIR", os.path.join(DATA_DIR, "chroma_abstracts"))
CHROMA_ABSTRACTS_COLLECTION = _env("CHROMA_ABSTRACTS_COLLECTION", "syracuse_abstracts")

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI      = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD", "")
NEO4J_DATABASE = _env("NEO4J_DATABASE", "syr-rag-abstracts")

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL  = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = _env("EMBED_DEVICE", "cpu")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_MAX_TOKENS     = _env_int("CHUNK_MAX_TOKENS", 512)
CHUNK_OVERLAP_TOKENS = _env_int("CHUNK_OVERLAP_TOKENS", 128)

# ── Download ──────────────────────────────────────────────────────────────────
DOWNLOAD_WORKERS     = _env_int("DOWNLOAD_WORKERS", 8)
DOWNLOAD_TIMEOUT     = _env_int("DOWNLOAD_TIMEOUT", 30)
DOWNLOAD_MAX_RETRIES = _env_int("DOWNLOAD_MAX_RETRIES", 3)
DOWNLOAD_RATE_DELAY  = _env_float("DOWNLOAD_RATE_DELAY", 0.15)
UNPAYWALL_EMAIL      = _env("UNPAYWALL_EMAIL", OPENALEX_EMAIL or "email")
