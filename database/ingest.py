"""
ingest.py — Transform & index: normalize → chunk → ChromaDB / Neo4j.

Merges the former normalize.py, chunker.py, ingest_chroma.py,
ingest_abstracts.py and ingest_neo4j.py into one module. Public entry points:

    run_normalize()   raw OpenAlex JSONL → normalized_works/authors.jsonl
    run_chroma()      normalized_works.jsonl → full chunked ChromaDB collection
    run_abstracts()   normalized_works.jsonl → title+abstract-only ChromaDB collection
    run_neo4j()       normalized_*.jsonl → Neo4j knowledge graph

Plus the reusable helpers  normalize_work / normalize_author  and  chunk_work.

Collisions resolved during the merge:
  * normalize's string coercer `_safe(val, default)` → renamed `_ntext`
    (distinct from the ChromaDB metadata coercer, kept as `_meta_safe`).
  * the duplicated `_context_prefix` (chunker + ingest_abstracts) → single copy.
  * HuggingFace embedder construction → single `_make_embedder` factory.

Standalone:
    python ingest.py normalize
    python ingest.py chroma       [--incremental]
    python ingest.py abstracts    [--incremental] [--chroma-dir DIR] [--collection NAME]
    python ingest.py neo4j        [--incremental]
    python ingest.py all          [--incremental] [--skip-neo4j] [--abstracts]
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)

INSTITUTION_ID   = config.OPENALEX_INSTITUTION_ID
BATCH_SIZE       = 5000   # chunks accumulated before an embed/upsert flush
CHROMA_MAX_BATCH = 166    # ChromaDB hard limit per upsert call
NEO4J_BATCH      = 500


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _meta_safe(val: Any) -> Any:
    """Coerce a value to a ChromaDB-safe metadata scalar."""
    if val is None:
        return ""
    if isinstance(val, (int, float, bool)):
        return val
    return str(val).strip()


def _make_embedder(model_name: str, device: str, batch_size: int = 128):
    """Build the langchain HuggingFaceEmbeddings the RAG engine queries with.

    normalize_embeddings=True so query-time cosine similarity is correct.
    Raises ImportError if langchain-huggingface is missing (callers handle it).
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ░░  NORMALIZE  (was normalize.py)                                             ░░
# ═══════════════════════════════════════════════════════════════════════════════

def _oa_id(url: Optional[str]) -> str:
    if not url:
        return ""
    return url.rsplit("/", 1)[-1].strip()

def _ntext(val: Any, default: str = "") -> str:
    """String coercer for normalization (was normalize.py's `_safe`)."""
    if val is None:
        return default
    return str(val).strip() or default

def _year(raw: Any) -> Optional[int]:
    if isinstance(raw, int):
        return raw
    m = re.search(r"\b(19|20)\d{2}\b", str(raw or ""))
    return int(m.group(0)) if m else None

def _clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def normalize_work(raw: Dict) -> Optional[Dict]:
    work_id = _oa_id(raw.get("id"))
    title   = _clean_html(_ntext(raw.get("title") or raw.get("display_name")))
    if not work_id or not title:
        return None

    # ── Authors ───────────────────────────────────────────────────────────────
    all_authors: List[Dict] = []
    su_authors:  List[Dict] = []

    for auth in (raw.get("authorships") or []):
        author_obj   = auth.get("author") or {}
        institutions = auth.get("institutions") or []
        name         = _ntext(author_obj.get("display_name"))
        if not name:
            continue

        is_su = any(
            _oa_id(inst.get("id")) == INSTITUTION_ID
            or INSTITUTION_ID in [_oa_id(l) for l in (inst.get("lineage") or [])]
            for inst in institutions
        )

        rec = {
            "id":               _oa_id(author_obj.get("id")),
            "name":             name,
            "orcid":            _ntext(author_obj.get("orcid")),
            "position":         _ntext(auth.get("author_position")),
            "is_corresponding": bool(auth.get("is_corresponding")),
            "is_su":            is_su,
        }
        all_authors.append(rec)
        if is_su:
            su_authors.append(rec)

    primary_researcher  = su_authors[0]["name"] if su_authors else (all_authors[0]["name"] if all_authors else "")
    su_researcher_names = [a["name"] for a in su_authors]

    authors_str = ", ".join(a["name"] for a in all_authors[:10])
    if len(all_authors) > 10:
        authors_str += f" et al. ({len(all_authors)} authors)"

    # ── Topics + Keywords ─────────────────────────────────────────────────────
    topics = []
    for t in (raw.get("topics") or [])[:5]:
        topics.append({
            "id":       _oa_id(t.get("id")),
            "name":     _ntext(t.get("display_name")),
            "subfield": _ntext((t.get("subfield") or {}).get("display_name")),
            "field":    _ntext((t.get("field") or {}).get("display_name")),
            "domain":   _ntext((t.get("domain") or {}).get("display_name")),
            "score":    float(t.get("score", 0) or 0),
        })

    keywords = []
    for k in (raw.get("keywords") or []):
        kw = _ntext(k.get("display_name") or k.get("keyword"))
        if kw and kw not in keywords:
            keywords.append(kw)

    # ── Misc ──────────────────────────────────────────────────────────────────
    oa = raw.get("open_access") or {}

    # ── Fulltext + Docling fields ─────────────────────────────────────────────
    fulltext_status = raw.get("fulltext_status", "none")
    docling_status  = raw.get("docling_status",  "none")

    return {
        "openalex_id":        work_id,
        "doi":                _ntext(raw.get("doi")),
        "title":              title,
        "abstract":           _ntext(raw.get("abstract_text")),
        "publication_year":   _year(raw.get("publication_year")),
        "publication_date":   _ntext(raw.get("publication_date")),
        "work_type":          _ntext(raw.get("type")),
        "cited_by_count":     int(raw.get("cited_by_count") or 0),
        "oa_status":          _ntext(oa.get("oa_status")),
        "oa_url":             _ntext(oa.get("oa_url")),
        "primary_researcher": primary_researcher,
        "su_researchers":     su_researcher_names,
        "authors_str":        authors_str,
        "all_authors":        all_authors,
        "topics":             topics,
        "keywords":           keywords,
        # References — kept for Neo4j citation edges, not ingested into Chroma
        "referenced_works":   [r for r in (raw.get("referenced_works") or []) if r],
        # Fulltext
        "fulltext_status":    fulltext_status,
        "fulltext_path":      raw.get("fulltext_path", ""),
        "has_fulltext":       fulltext_status in ("tei_xml", "pdf"),
        # Docling
        "docling_status":     docling_status,
        "docling_path":       raw.get("docling_path", ""),
        "has_docling":        docling_status in ("docling_ok", "fallback_pdf"),
    }


def normalize_author(raw: Dict) -> Optional[Dict]:
    author_id = _oa_id(raw.get("id"))
    name      = _ntext(raw.get("display_name"))
    if not author_id or not name:
        return None

    stats = raw.get("summary_stats") or {}
    topics = [
        {"id": _oa_id(t.get("id")), "name": _ntext(t.get("display_name")), "count": int(t.get("count", 0) or 0)}
        for t in (raw.get("topics") or [])[:10]
    ]
    affiliations = [
        {"id": _oa_id(i.get("id")), "name": _ntext(i.get("display_name")),
         "country": _ntext(i.get("country_code")), "type": _ntext(i.get("type"))}
        for i in (raw.get("last_known_institutions") or [])
    ]

    return {
        "openalex_id":    author_id,
        "name":           name,
        "orcid":          _ntext(raw.get("orcid")),
        "alt_names":      [_ntext(n) for n in (raw.get("display_name_alternatives") or []) if _ntext(n)],
        "works_count":    int(raw.get("works_count") or 0),
        "cited_by_count": int(raw.get("cited_by_count") or 0),
        "h_index":        int(stats.get("h_index") or 0),
        "i10_index":      int(stats.get("i10_index") or 0),
        "topics":         topics,
        "affiliations":   affiliations,
    }


def run_normalize() -> Dict[str, int]:
    raw_dir = Path(config.RAW_DIR)

    # Auto-select best available input
    candidates = [
        raw_dir / "works_with_docling.jsonl",
        raw_dir / "works_with_fulltext.jsonl",
        raw_dir / "works.jsonl",
    ]
    works_in = next((p for p in candidates if p.exists()), None)
    if works_in is None:
        logger.error("No works input file found — run fetch first")
        return {"works": 0, "authors": 0}

    label = {
        "works_with_docling.jsonl":  "docling (best)",
        "works_with_fulltext.jsonl": "fulltext only",
        "works.jsonl":               "raw (no fulltext)",
    }.get(works_in.name, works_in.name)
    logger.info("Input: %s [%s]", works_in.name, label)

    works_out   = raw_dir / "normalized_works.jsonl"
    authors_in  = raw_dir / "authors.jsonl"
    authors_out = raw_dir / "normalized_authors.jsonl"

    # ── Works ─────────────────────────────────────────────────────────────────
    works_count = skipped = 0
    seen_ids    = set()
    ft_counts   = {"tei_xml": 0, "pdf": 0, "none": 0}
    dl_counts   = {"docling_ok": 0, "fallback_pdf": 0, "none": 0}

    with open(works_in, encoding="utf-8") as fin, \
         open(works_out, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            norm = normalize_work(raw)
            if norm and norm["openalex_id"] not in seen_ids:
                seen_ids.add(norm["openalex_id"])
                fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                works_count += 1
                ft_counts[norm["fulltext_status"]] = ft_counts.get(norm["fulltext_status"], 0) + 1
                dl_counts[norm["docling_status"]]  = dl_counts.get(norm["docling_status"], 0) + 1
            else:
                skipped += 1

    logger.info(
        "Works: %d normalized (%d skipped) | "
        "Fulltext — pdf:%d tei:%d none:%d | "
        "Docling — ok:%d fallback:%d none:%d",
        works_count, skipped,
        ft_counts["pdf"], ft_counts["tei_xml"], ft_counts["none"],
        dl_counts["docling_ok"], dl_counts["fallback_pdf"], dl_counts["none"],
    )

    # ── Authors ───────────────────────────────────────────────────────────────
    authors_count = 0
    seen_authors  = set()

    if authors_in.exists():
        with open(authors_in, encoding="utf-8") as fin, \
             open(authors_out, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                norm = normalize_author(raw)
                if norm and norm["openalex_id"] not in seen_authors:
                    seen_authors.add(norm["openalex_id"])
                    fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    authors_count += 1
        logger.info("Authors: %d normalized", authors_count)

    return {"works": works_count, "authors": authors_count}


# ═══════════════════════════════════════════════════════════════════════════════
# ░░  CHUNKER  (was chunker.py)                                                 ░░
# ═══════════════════════════════════════════════════════════════════════════════

def _token_est(text: str) -> int:
    return int(len(text.split()) * 1.3)

def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def chunk_with_overlap(text: str) -> List[str]:
    max_tok = config.CHUNK_MAX_TOKENS
    overlap = config.CHUNK_OVERLAP_TOKENS

    sentences = _sentence_split(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current: List[str] = []
    current_tok = 0

    for sent in sentences:
        sent_tok = _token_est(sent)
        if current_tok + sent_tok > max_tok and current:
            chunks.append(" ".join(current))
            keep, keep_tok = [], 0
            for s in reversed(current):
                t = _token_est(s)
                if keep_tok + t > overlap:
                    break
                keep.insert(0, s)
                keep_tok += t
            current, current_tok = keep, keep_tok
        current.append(sent)
        current_tok += sent_tok

    if current:
        chunks.append(" ".join(current))
    return chunks


def _load_docling_sections(docling_path: str) -> List[Dict]:
    if not docling_path or not Path(docling_path).exists():
        return []
    try:
        data = json.loads(Path(docling_path).read_text(encoding="utf-8"))
        return data.get("sections") or []
    except Exception as e:
        logger.debug("Failed to load Docling JSON %s: %s", docling_path, e)
        return []


def _context_prefix(work: Dict) -> str:
    """Short prefix prepended to embed_text only (not the stored text)."""
    parts = []
    if work.get("primary_researcher"):
        parts.append(f"Research by {work['primary_researcher']} at Syracuse University")
    topics = work.get("topics") or []
    if topics and topics[0].get("name"):
        parts.append(f"in {topics[0]['name']}")
    if work.get("publication_year"):
        parts.append(f"({work['publication_year']})")
    prefix = " ".join(parts)
    return f"{prefix}. " if prefix else ""

def _chunk_id(work_id: str, ctype: str, idx: int) -> str:
    return f"{work_id}_{ctype}_{idx}"


def chunk_work(work: Dict) -> List[Dict]:
    """Produce all chunks for a single normalized work."""
    chunks: List[Dict] = []
    work_id = work["openalex_id"]
    title   = work.get("title", "")
    prefix  = _context_prefix(work)
    topics  = work.get("topics") or []

    base_meta = {
        "paper_id":        work_id,
        "researcher":      work.get("primary_researcher", ""),
        "authors":         work.get("authors_str", ""),
        "year":            str(work.get("publication_year") or ""),
        "topic":           topics[0]["name"] if topics else "",
        "primary_topic":   topics[0]["name"] if topics else "",
        "doi":             work.get("doi", ""),
        "title":           title,
        "work_type":       work.get("work_type", ""),
        "cited_by_count":  work.get("cited_by_count", 0),
        "fulltext_status": work.get("fulltext_status", "none"),
        "docling_status":  work.get("docling_status", "none"),
        "has_fulltext":    work.get("has_fulltext", False),
        # referenced_works intentionally excluded — stored in Neo4j only
    }
    su = work.get("su_researchers") or []
    if su:
        base_meta["su_researchers"] = " | ".join(su)

    def _add(ctype, idx, text, extra_meta=None):
        cid = _chunk_id(work_id, ctype, idx)
        m = {**base_meta, "chunk_type": ctype, "chunk": cid, **(extra_meta or {})}
        chunks.append({
            "chunk_id":    cid,
            "work_id":     work_id,
            "chunk_type":  ctype,
            "chunk_index": idx,
            "text":        text,
            "embed_text":  f"{prefix}{text}",
            "metadata":    m,
        })

    # ── 1. Title + Abstract (always) ──────────────────────────────────────────
    abstract = work.get("abstract", "")
    ta       = f"{title}\n\n{abstract}".strip() if abstract else title
    if title:
        _add("title_abstract", 0, ta)

    # ── 2. Keywords + Topics (always if present) ──────────────────────────────
    keywords = work.get("keywords") or []
    if keywords or topics:
        parts = [title]
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords[:15]))
        if topics:
            tnames = [t["name"] for t in topics[:5] if t.get("name")]
            if tnames:
                parts.append("Research topics: " + ", ".join(tnames))
            fields = list({t["field"] for t in topics[:5] if t.get("field")})
            if fields:
                parts.append("Fields: " + ", ".join(fields))
            domains = list({t["domain"] for t in topics[:5] if t.get("domain")})
            if domains:
                parts.append("Domains: " + ", ".join(domains))
        _add("keywords", 1, ". ".join(parts))

    # ── 3. Docling fulltext sections ──────────────────────────────────────────
    docling_status = work.get("docling_status", "none")
    docling_path   = work.get("docling_path", "")

    if docling_status in ("docling_ok", "fallback_pdf") and docling_path:
        sections = _load_docling_sections(docling_path)
        idx = 100

        for sec in sections:
            heading      = sec.get("heading", "Section")
            body         = sec.get("text", "").strip()
            element_type = sec.get("element_type", "section")
            level        = sec.get("level", 1)

            if not body or len(body) < 30:
                continue

            sec_meta = {"section_heading": heading, "section_level": level}

            if element_type in ("table", "figure_caption"):
                ctype = "table" if element_type == "table" else "figure_caption"
                _add(ctype, idx, f"{title} — {heading}: {body}", sec_meta)
                idx += 1
            else:
                ctype = "section" if docling_status == "docling_ok" else "fallback_text"
                for sub in chunk_with_overlap(body):
                    _add(ctype, idx, f"{title} — {heading}: {sub}", sec_meta)
                    idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# ░░  CHROMADB — full chunked collection  (was ingest_chroma.py)               ░░
# ═══════════════════════════════════════════════════════════════════════════════

def run_chroma(rebuild: bool = True) -> Dict[str, Any]:
    works_file = Path(config.RAW_DIR) / "normalized_works.jsonl"
    if not works_file.exists():
        logger.error("No normalized_works.jsonl — run normalize first")
        return {"works": 0, "chunks": 0}

    try:
        import chromadb
    except ImportError:
        logger.error("chromadb not installed — run: pip install chromadb")
        return {"error": "chromadb not installed"}

    try:
        embedder = _make_embedder(config.EMBED_MODEL, config.EMBED_DEVICE)
    except ImportError:
        logger.error("langchain-huggingface not installed — run: pip install langchain-huggingface")
        return {"error": "langchain-huggingface not installed"}

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    if rebuild:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
            logger.info("Deleted existing collection '%s'", config.CHROMA_COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    batch_ids:   List[str]  = []
    batch_docs:  List[str]  = []
    batch_metas: List[Dict] = []
    chunk_type_counts: Dict[str, int] = {}
    total_chunks = total_works = 0
    t0 = time.time()

    def _flush_batch():
        if not batch_ids:
            return
        vectors = embedder.embed_documents(batch_docs)
        for start in range(0, len(batch_ids), CHROMA_MAX_BATCH):
            end = start + CHROMA_MAX_BATCH
            collection.upsert(
                ids=batch_ids[start:end],
                documents=batch_docs[start:end],
                embeddings=vectors[start:end],
                metadatas=batch_metas[start:end],
            )
        elapsed = time.time() - t0
        logger.info(
            "Upserted %d chunks | total: %d from %d works | %.0f chunks/s",
            len(batch_ids), total_chunks, total_works,
            total_chunks / max(0.1, elapsed),
        )
        batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    with open(works_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            work   = json.loads(line)
            chunks = chunk_work(work)
            if not chunks:
                continue

            total_works += 1
            for chunk in chunks:
                meta = {k: _meta_safe(v) for k, v in chunk["metadata"].items()}
                meta["work_id"] = chunk["work_id"]

                batch_ids.append(chunk["chunk_id"])
                batch_docs.append(chunk["embed_text"])
                batch_metas.append(meta)
                total_chunks += 1

                ct = chunk["chunk_type"]
                chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

            if len(batch_ids) >= BATCH_SIZE:
                _flush_batch()

    _flush_batch()

    elapsed = time.time() - t0
    logger.info(
        "Chroma done: %d chunks from %d works in %.1fs (%.0f chunks/s)",
        total_chunks, total_works, elapsed, total_chunks / max(0.1, elapsed),
    )
    logger.info("Chunk types: %s", chunk_type_counts)

    return {
        "works": total_works,
        "chunks": total_chunks,
        "elapsed_s": round(elapsed, 1),
        **chunk_type_counts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ░░  CHROMADB — abstract-only collection  (was ingest_abstracts.py)           ░░
# ═══════════════════════════════════════════════════════════════════════════════

def _abstract_chunks(work: Dict) -> List[Dict]:
    """1-2 lightweight chunks per work: title_abstract (+ keywords)."""
    work_id = work.get("openalex_id", "")
    if not work_id:
        return []

    title    = str(work.get("title", "") or "").strip()
    abstract = str(work.get("abstract", "") or "").strip()
    if not title:
        return []

    topics   = work.get("topics") or []
    keywords = work.get("keywords") or []
    prefix   = _context_prefix(work)
    su       = work.get("su_researchers") or []

    base_meta: Dict[str, Any] = {
        "paper_id":        work_id,
        "researcher":      _meta_safe(work.get("primary_researcher", "")),
        "authors":         _meta_safe(work.get("authors_str", "")),
        "year":            str(work.get("publication_year") or ""),
        "topic":           topics[0]["name"] if topics else "",
        "primary_topic":   topics[0]["name"] if topics else "",
        "doi":             _meta_safe(work.get("doi", "")),
        "title":           title,
        "work_type":       _meta_safe(work.get("work_type", "")),
        "cited_by_count":  work.get("cited_by_count", 0),
        "has_fulltext":    False,
        "fulltext_status": "none",
        "docling_status":  "none",
    }
    if su:
        base_meta["su_researchers"] = " | ".join(su)

    chunks: List[Dict] = []

    ta_text = f"{title}\n\n{abstract}".strip() if abstract else title
    chunks.append({
        "chunk_id":   f"{work_id}_title_abstract_0",
        "embed_text": f"{prefix}{ta_text}",
        "metadata":   {**base_meta, "chunk_type": "title_abstract",
                       "chunk": f"{work_id}_title_abstract_0"},
    })

    if keywords or topics:
        kw_parts = [title]
        if keywords:
            kw_parts.append("Keywords: " + ", ".join(keywords[:15]))
        if topics:
            tnames = [t["name"] for t in topics[:5] if t.get("name")]
            if tnames:
                kw_parts.append("Research topics: " + ", ".join(tnames))
            fields = list({t["field"] for t in topics[:5] if t.get("field")})
            if fields:
                kw_parts.append("Fields: " + ", ".join(fields))
            domains = list({t["domain"] for t in topics[:5] if t.get("domain")})
            if domains:
                kw_parts.append("Domains: " + ", ".join(domains))
        kw_text = ". ".join(kw_parts)
        chunks.append({
            "chunk_id":   f"{work_id}_keywords_1",
            "embed_text": f"{prefix}{kw_text}",
            "metadata":   {**base_meta, "chunk_type": "keywords",
                           "chunk": f"{work_id}_keywords_1"},
        })

    return chunks


def run_abstracts(rebuild: bool = True,
                  chroma_dir: Optional[str] = None,
                  collection_name: Optional[str] = None) -> Dict[str, Any]:
    """Read normalized_works.jsonl → title+abstract-only ChromaDB collection."""
    works_file = Path(config.RAW_DIR) / "normalized_works.jsonl"
    if not works_file.exists():
        logger.error("No normalized_works.jsonl — run normalize first")
        return {"works": 0, "chunks": 0, "error": "input file not found"}

    try:
        import chromadb
    except ImportError:
        logger.error("chromadb not installed — run: pip install chromadb")
        return {"error": "chromadb not installed"}

    try:
        embedder = _make_embedder(config.EMBED_MODEL, config.EMBED_DEVICE)
    except ImportError:
        logger.error("langchain-huggingface not installed — run: pip install langchain-huggingface")
        return {"error": "langchain-huggingface not installed"}

    out_dir  = chroma_dir or config.CHROMA_ABSTRACTS_DIR
    col_name = collection_name or config.CHROMA_ABSTRACTS_COLLECTION
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Input:       %s", works_file)
    logger.info("Chroma dir:  %s", out_dir)
    logger.info("Collection:  %s", col_name)
    logger.info("Embed model: %s (device=%s)", config.EMBED_MODEL, config.EMBED_DEVICE)

    client = chromadb.PersistentClient(path=out_dir)

    if rebuild:
        try:
            client.delete_collection(col_name)
            logger.info("Deleted existing collection '%s'", col_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_ids:   List[str]  = []
    batch_docs:  List[str]  = []
    batch_metas: List[Dict] = []
    chunk_type_counts: Dict[str, int] = {}
    total_chunks = total_works = skipped_no_abstract = 0
    t0 = time.time()

    def _flush_batch():
        if not batch_ids:
            return
        vectors = embedder.embed_documents(batch_docs)
        collection.upsert(
            ids=batch_ids, documents=batch_docs,
            embeddings=vectors, metadatas=batch_metas,
        )
        elapsed = time.time() - t0
        logger.info(
            "Upserted %d chunks | total: %d from %d works | %.0f chunks/s",
            len(batch_ids), total_chunks, total_works,
            total_chunks / max(0.1, elapsed),
        )
        batch_ids.clear(); batch_docs.clear(); batch_metas.clear()

    with open(works_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip works with no abstract — title-only entries are very low signal
            abstract = str(work.get("abstract", "") or "").strip()
            if not abstract or len(abstract) < 20:
                skipped_no_abstract += 1
                continue

            chunks = _abstract_chunks(work)
            if not chunks:
                continue

            total_works += 1
            for chunk in chunks:
                meta = {k: _meta_safe(v) for k, v in chunk["metadata"].items()}

                batch_ids.append(chunk["chunk_id"])
                batch_docs.append(chunk["embed_text"])
                batch_metas.append(meta)
                total_chunks += 1

                ct = meta.get("chunk_type", "unknown")
                chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

            if len(batch_ids) >= BATCH_SIZE:
                _flush_batch()

    _flush_batch()

    elapsed = time.time() - t0
    logger.info(
        "Abstract-only ingest done: %d chunks from %d works in %.1fs (%.0f chunks/s)",
        total_chunks, total_works, elapsed, total_chunks / max(0.1, elapsed),
    )
    logger.info("Skipped (no abstract): %d", skipped_no_abstract)
    logger.info("Chunk types: %s", chunk_type_counts)

    result = {
        "works":               total_works,
        "chunks":              total_chunks,
        "skipped_no_abstract": skipped_no_abstract,
        "elapsed_s":           round(elapsed, 1),
        "chroma_dir":          out_dir,
        "collection":          col_name,
        **chunk_type_counts,
    }
    logger.info("Result: %s", result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ░░  NEO4J — knowledge graph  (was ingest_neo4j.py)                           ░░
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class Neo4jGraph:
    def __init__(self):
        if not HAS_NEO4J:
            raise RuntimeError("neo4j not installed — run: pip install neo4j")
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self.db = config.NEO4J_DATABASE
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s (db=%s)", config.NEO4J_URI, self.db)

    def close(self):
        self.driver.close()

    def _run(self, q: str, **p):
        with self.driver.session(database=self.db) as s:
            s.run(q, **p)

    def _batch(self, q: str, key: str, data: list):
        if data:
            with self.driver.session(database=self.db) as s:
                s.run(q, **{key: data})

    def create_constraints(self):
        for q in [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.oid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work)   REQUIRE w.oid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic)  REQUIRE t.oid IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (w:Work)   ON (w.title)",
            "CREATE INDEX IF NOT EXISTS FOR (w:Work)   ON (w.year)",
        ]:
            try:
                self._run(q)
            except Exception:
                pass

    def clear(self):
        """Delete all nodes/edges in batches to avoid transaction memory limits."""
        total_deleted = 0
        while True:
            with self.driver.session(database=self.db) as s:
                result = s.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS deleted"
                )
                deleted = result.single()["deleted"]
                total_deleted += deleted
                if deleted == 0:
                    break
                logger.info("  Cleared %d nodes (total so far: %d)", deleted, total_deleted)
        logger.info("Cleared all Neo4j nodes/edges (%d total)", total_deleted)

    def upsert_works(self, b):
        self._batch("""
            UNWIND $b AS w
            MERGE (x:Work {oid: w.oid})
            SET x.title = w.title, x.doi = w.doi, x.year = w.year,
                x.cited = w.cited, x.type = w.type,
                x.researcher = w.researcher,
                x.docling_status = w.docling_status
        """, "b", b)

    def upsert_authors(self, b):
        self._batch("""
            UNWIND $b AS a
            MERGE (x:Author {oid: a.oid})
            SET x.name = a.name, x.orcid = a.orcid,
                x.works_count = a.works_count,
                x.cited = a.cited, x.h_index = a.h_index
        """, "b", b)

    def upsert_topics(self, b):
        self._batch("""
            UNWIND $b AS t
            MERGE (x:Topic {oid: t.oid})
            SET x.name = t.name, x.subfield = t.subfield,
                x.field = t.field, x.domain = t.domain
        """, "b", b)

    def create_authored(self, b):
        self._batch("""
            UNWIND $b AS e
            MATCH (a:Author {oid: e.aid})
            MATCH (w:Work   {oid: e.wid})
            MERGE (a)-[r:AUTHORED]->(w)
            SET r.position = e.pos, r.is_corresponding = e.corr
        """, "b", b)

    def create_has_topic(self, b):
        self._batch("""
            UNWIND $b AS e
            MATCH (w:Work  {oid: e.wid})
            MATCH (t:Topic {oid: e.tid})
            MERGE (w)-[r:HAS_TOPIC]->(t)
            SET r.score = e.score
        """, "b", b)

    def create_cites(self, b):
        self._batch("""
            UNWIND $b AS e
            MATCH (a:Work {oid: e.src})
            MATCH (b:Work {oid: e.dst})
            MERGE (a)-[:CITES]->(b)
        """, "b", b)

    def derive_collaborations(self):
        # Run in batches to avoid transaction memory limit
        with self.driver.session(database=self.db) as s:
            total = s.run("MATCH (a:Author) RETURN count(a) AS c").single()["c"]

        batch_size  = 2000
        offset      = 0
        total_edges = 0
        while offset < total:
            with self.driver.session(database=self.db) as s:
                result = s.run("""
                    MATCH (a1:Author)
                    WITH a1 SKIP $skip LIMIT $limit
                    MATCH (a1)-[:AUTHORED]->(w:Work)<-[:AUTHORED]-(a2:Author)
                    WHERE a1.oid < a2.oid
                    WITH a1, a2, COUNT(w) AS shared
                    MERGE (a1)-[r:COLLABORATES_WITH]-(a2)
                    SET r.shared_works = shared
                    RETURN count(r) AS edges
                """, skip=offset, limit=batch_size)
                edges = result.single()["edges"]
                total_edges += edges
            offset += batch_size
            logger.info("  Collaborations batch offset=%d edges_so_far=%d", offset, total_edges)

        logger.info("COLLABORATES_WITH edges derived: %d total", total_edges)


def run_neo4j(rebuild: bool = True) -> Dict[str, int]:
    if not HAS_NEO4J:
        logger.error("neo4j driver not installed — skipping")
        return {"status": "skipped"}

    raw_dir      = Path(config.RAW_DIR)
    works_file   = raw_dir / "normalized_works.jsonl"
    authors_file = raw_dir / "normalized_authors.jsonl"

    graph = Neo4jGraph()
    valid_cites: List[Dict] = []

    try:
        if rebuild:
            graph.clear()
        graph.create_constraints()

        # ── Author nodes ──────────────────────────────────────────────────────
        author_count = 0
        if authors_file.exists():
            batch = []
            with open(authors_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    a = json.loads(line)
                    batch.append({
                        "oid": a["openalex_id"], "name": a["name"],
                        "orcid": a.get("orcid", ""),
                        "works_count": a.get("works_count", 0),
                        "cited": a.get("cited_by_count", 0),
                        "h_index": a.get("h_index", 0),
                    })
                    if len(batch) >= NEO4J_BATCH:
                        graph.upsert_authors(batch)
                        author_count += len(batch)
                        batch.clear()
            if batch:
                graph.upsert_authors(batch)
                author_count += len(batch)
            logger.info("Author nodes: %d", author_count)

        # ── Work nodes + edges ────────────────────────────────────────────────
        work_count       = 0
        seen_topics:     Set[str]   = set()
        known_works:     Set[str]   = set()
        cite_edges:      List[Dict] = []

        work_batch:       List[Dict] = []
        auth_node_batch:  List[Dict] = []
        topic_batch:      List[Dict] = []
        authored_batch:   List[Dict] = []
        topic_edge_batch: List[Dict] = []

        if works_file.exists():
            with open(works_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    w   = json.loads(line)
                    wid = w["openalex_id"]
                    known_works.add(wid)

                    work_batch.append({
                        "oid": wid, "title": w.get("title", ""),
                        "doi": w.get("doi", ""),
                        "year": w.get("publication_year"),
                        "cited": w.get("cited_by_count", 0),
                        "type": w.get("work_type", ""),
                        "researcher": w.get("primary_researcher", ""),
                        "docling_status": w.get("docling_status", "none"),
                    })

                    for a in (w.get("all_authors") or []):
                        aid = a.get("id", "")
                        if not aid:
                            continue
                        auth_node_batch.append({
                            "oid": aid, "name": a.get("name", ""),
                            "orcid": a.get("orcid", ""),
                            "works_count": 0, "cited": 0, "h_index": 0,
                        })
                        authored_batch.append({
                            "aid": aid, "wid": wid,
                            "pos": a.get("position", ""),
                            "corr": a.get("is_corresponding", False),
                        })

                    for t in (w.get("topics") or []):
                        tid = t.get("id", "")
                        if not tid:
                            continue
                        if tid not in seen_topics:
                            seen_topics.add(tid)
                            topic_batch.append({
                                "oid": tid, "name": t.get("name", ""),
                                "subfield": t.get("subfield", ""),
                                "field": t.get("field", ""),
                                "domain": t.get("domain", ""),
                            })
                        topic_edge_batch.append({
                            "wid": wid, "tid": tid,
                            "score": float(t.get("score", 0)),
                        })

                    for ref in (w.get("referenced_works") or []):
                        if ref:
                            dst = ref.rsplit("/", 1)[-1] if "/" in ref else ref
                            cite_edges.append({"src": wid, "dst": dst})

                    if len(work_batch) >= NEO4J_BATCH:
                        graph.upsert_works(work_batch);           work_count += len(work_batch); work_batch.clear()
                        graph.upsert_authors(auth_node_batch);    auth_node_batch.clear()
                        graph.upsert_topics(topic_batch);         topic_batch.clear()
                        graph.create_authored(authored_batch);    authored_batch.clear()
                        graph.create_has_topic(topic_edge_batch); topic_edge_batch.clear()
                        logger.info("  ... %d works processed", work_count)

            # Flush remainder
            if work_batch:
                graph.upsert_works(work_batch);           work_count += len(work_batch)
            if auth_node_batch:
                graph.upsert_authors(auth_node_batch)
            if topic_batch:
                graph.upsert_topics(topic_batch)
            if authored_batch:
                graph.create_authored(authored_batch)
            if topic_edge_batch:
                graph.create_has_topic(topic_edge_batch)

            # Citation edges — only between works we actually have
            valid_cites = [e for e in cite_edges if e["dst"] in known_works]
            for i in range(0, len(valid_cites), NEO4J_BATCH):
                graph.create_cites(valid_cites[i:i + NEO4J_BATCH])

            logger.info(
                "Work nodes: %d | Topics: %d | Citations: %d",
                work_count, len(seen_topics), len(valid_cites),
            )

        graph.derive_collaborations()

        result = {
            "authors":   author_count,
            "works":     work_count,
            "topics":    len(seen_topics),
            "citations": len(valid_cites),
        }
        logger.info("Neo4j complete: %s", result)
        return result

    finally:
        graph.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Normalize → chunk → ChromaDB / Neo4j")
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("normalize", help="Raw OpenAlex JSONL → normalized_*.jsonl")

    p_chroma = sub.add_parser("chroma", help="Full chunked ChromaDB collection")
    p_chroma.add_argument("--incremental", action="store_true")

    p_abs = sub.add_parser("abstracts", help="Abstract-only ChromaDB collection")
    p_abs.add_argument("--incremental", action="store_true")
    p_abs.add_argument("--chroma-dir", type=str, default=None)
    p_abs.add_argument("--collection", type=str, default=None)

    p_neo = sub.add_parser("neo4j", help="Neo4j knowledge graph")
    p_neo.add_argument("--incremental", action="store_true")

    p_all = sub.add_parser("all", help="normalize → chroma → neo4j")
    p_all.add_argument("--incremental", action="store_true")
    p_all.add_argument("--skip-neo4j",  action="store_true")
    p_all.add_argument("--abstracts",   action="store_true", help="also build abstracts collection")

    args = parser.parse_args()

    if args.stage == "normalize":
        print(run_normalize())
    elif args.stage == "chroma":
        print(run_chroma(rebuild=not args.incremental))
    elif args.stage == "abstracts":
        print(run_abstracts(rebuild=not args.incremental,
                            chroma_dir=args.chroma_dir,
                            collection_name=args.collection))
    elif args.stage == "neo4j":
        print(run_neo4j(rebuild=not args.incremental))
    elif args.stage == "all":
        rebuild = not args.incremental
        print(run_normalize())
        print(run_chroma(rebuild=rebuild))
        if args.abstracts:
            print(run_abstracts(rebuild=rebuild))
        if not args.skip_neo4j:
            print(run_neo4j(rebuild=rebuild))


if __name__ == "__main__":
    _cli()
