# config_full.py — Cluster edition (OrangeGrid / Linux paths)
import os

_MODELS_ROOT = os.path.expanduser("~/models")
_DATA_ROOT   = os.path.expanduser("~/research-llm-data")

# ── Embedding model ──────────────────────────────────────────────────────
EMBED_MODEL = os.environ.get("EMBED_MODEL",
    os.path.join(_MODELS_ROOT, "all-MiniLM-L6-v2"))

# ── LLM model paths ─────────────────────────────────────────────────────
LLAMA_1B     = os.environ.get("LLAMA_1B",
    os.path.join(_MODELS_ROOT, "Llama-3.2-1B-Instruct"))
LLAMA_3B     = os.environ.get("LLAMA_3B",
    os.path.join(_MODELS_ROOT, "Llama-3.2-3B-Instruct"))
LLAMA_8B     = os.environ.get("LLAMA_8B",
    os.path.join(_MODELS_ROOT, "Llama-3.1-8B-Instruct"))
LLAMA_70B    = os.environ.get("LLAMA_70B",
    os.path.join(_MODELS_ROOT, "Llama-3.3-70B-Instruct"))

GEMMA_4_E2B  = os.environ.get("GEMMA_4_E2B",
    os.path.join(_MODELS_ROOT, "gemma-4-E2B-it"))
GEMMA_4_E4B  = os.environ.get("GEMMA_4_E4B",
    os.path.join(_MODELS_ROOT, "gemma-4-E4B-it"))
GEMMA_4_26B  = os.environ.get("GEMMA_4_26B",
    os.path.join(_MODELS_ROOT, "gemma-4-26B-A4B-it"))
GEMMA_4_31B  = os.environ.get("GEMMA_4_31B",
    os.path.join(_MODELS_ROOT, "gemma-4-31B-it"))

QWEN_14B     = os.environ.get("QWEN_14B",
    os.path.join(_MODELS_ROOT, "Qwen2.5-14B-Instruct"))
GPT_OSS_20B  = os.environ.get("GPT_OSS_20B",
    os.path.join(_MODELS_ROOT, "gpt-oss-20b"))

# ── ChromaDB directories ────────────────────────────────────────────────
CHROMA_DIR_FULL      = os.environ.get("CHROMA_DIR_FULL",
    os.path.join(_DATA_ROOT, "chroma_store_full"))
CHROMA_DIR_OPENALEX  = os.environ.get("CHROMA_DIR_OPENALEX",
    os.path.join(_DATA_ROOT, "chroma_db"))
CHROMA_DIR_ABSTRACTS = os.environ.get("CHROMA_DIR_ABSTRACTS",
    os.path.join(_DATA_ROOT, "chroma_abstracts"))

CHROMA_COLLECTION_FULL      = os.environ.get("CHROMA_COLLECTION_FULL", "papers_all")
CHROMA_COLLECTION_OPENALEX  = os.environ.get("CHROMA_COLLECTION_OPENALEX", "syracuse_papers")
CHROMA_COLLECTION_ABSTRACTS = os.environ.get("CHROMA_COLLECTION_ABSTRACTS", "syracuse_abstracts")
CHROMA_COLLECTION           = CHROMA_COLLECTION_FULL   # legacy alias

# ── Neo4j (disabled on cluster — no server) ──────────────────────────────
NEO4J_DB          = os.environ.get("NEO4J_DB", "syr-rag")
NEO4J_DB_OPENALEX = os.environ.get("NEO4J_DB_OPENALEX", "syr-rag-openalex")
NEO4J_DB_ABSTRACTS = os.environ.get("NEO4J_DB_ABSTRACTS", "syr-rag-abstracts")
