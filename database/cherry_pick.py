#!/usr/bin/env python3
"""
cherry_pick.py — Download and ingest a SINGLE paper (by arXiv link) into an
existing ChromaDB collection, WITHOUT rebuilding or altering anything already in
it. The new paper's chunks are upserted; every existing entry is left untouched.

TWO MODES
─────────
1. DEFAULT (refactored pipeline → syracuse_papers, openalex schema):
   Reuses config / ingest / fetch. Metadata from OpenAlex (by arXiv DOI) with an
   arXiv-API fallback; normalized + chunked + embedded like the rest of that
   collection; chunk ids keyed by work_id.

       python cherry_pick.py https://arxiv.org/abs/2105.05238
       python cherry_pick.py 2105.05238 --abstracts --neo4j --dry-run

2. --paper-id  (legacy schema → papers_all, attach to an EXISTING record):
   For the app's main collection (papers_all), which keys on paper_id and whose
   answer path pins by paper_id. This inherits an existing record's metadata,
   extracts the PDF body itself (docling → pdfminer → pypdf), and writes chunks
   '{paper_id}_section_N' with paper_id / chunk_type / has_body=True so the app's
   title-anchor + load_full_paper_docs surface the body with NO app change.

       python cherry_pick.py https://arxiv.org/abs/2205.13487 ^
           --paper-id 42047 ^
           --chroma-dir "C:\\codes\\new_pipeline\\Syr_research_all\\chroma_store_full" ^
           --collection papers_all ^
           --embed-model "C:\\codes\\models\\all-MiniLM-L6-v2" ^
           --replace

Common options:
    --collection NAME    Target collection (default: config.CHROMA_COLLECTION)
    --chroma-dir DIR     ChromaDB directory (default: config.CHROMA_DIR)
    --embed-model PATH   Embedding model (default: config.EMBED_MODEL) — MUST match
                         the model the target collection was built with.
    --embed-device DEV   cpu | cuda (default: config.EMBED_DEVICE)
    --no-docling         Skip Docling; title+abstract (+pdfminer) only
    --force              Re-embed even if the paper is already present (default mode)
    --dry-run            Fetch/normalize/chunk only; write nothing
Legacy-mode only:
    --paper-id ID        Existing paper_id to attach the body to (enables legacy mode)
    --replace            Delete that paper_id's existing chunks first (clears the
                         stale title-only / has_body=False chunk)
    --max-chunks N       Cap body section chunks written (default 60)
Refactored-mode only:
    --abstracts          Also add to the abstract-only collection
    --neo4j              Also add to the Neo4j graph (adds nodes/edges only; no wipe)
"""

import argparse
import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config          # FIX: was 'import new_pipeline.RAG.db.config as config' (unimportable)
import ingest
import fetch  # reuse Docling converter / section extraction / pdfminer fallback

logger = logging.getLogger(__name__)

ARXIV_API    = "http://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org"
_UA = {"User-Agent": f"research-llm/1.0 (mailto:{config.OPENALEX_EMAIL or 'noreply@example.com'})"}
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

LEGACY_CHUNK_MAX_CHARS = 2000
LEGACY_CHROMA_MAX_BATCH = 512


# ═══════════════════════════════════════════════════════════════════════════════
# arXiv identifier parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arxiv_id(text: str) -> Tuple[str, str, str]:
    """Accept a full arXiv URL or a bare id and return
    (arxiv_id_with_version, base_id_no_version, default_work_id)."""
    s = text.strip()
    m = re.search(r"arxiv\.org/(?:pdf|abs)/(.+)$", s, re.IGNORECASE)
    core = m.group(1) if m else s
    core = core.split("?")[0].split("#")[0].strip("/")
    if core.lower().endswith(".pdf"):
        core = core[:-4]
    arxiv_id = core.strip()
    base_id  = re.sub(r"v\d+$", "", arxiv_id)
    work_id  = "arxiv_" + arxiv_id.replace("/", "_")
    return arxiv_id, base_id, work_id


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata: OpenAlex (preferred) → arXiv API (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_openalex_by_arxiv(base_id: str) -> Optional[Dict]:
    params = {}
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY
    url = f"{OPENALEX_API}/works/doi:10.48550/arXiv.{base_id}"
    try:
        r = requests.get(url, params=params, headers=_UA, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug("OpenAlex lookup failed: %s", e)
    return None


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def arxiv_atom_to_raw(xml_text: str, arxiv_id: str, base_id: str, work_id: str) -> Dict:
    root  = ET.fromstring(xml_text)
    entry = root.find("a:entry", _ATOM_NS)
    if entry is None:
        raise ValueError(f"No arXiv entry found for {arxiv_id}")

    title     = _collapse_ws(entry.findtext("a:title", default="", namespaces=_ATOM_NS))
    abstract  = _collapse_ws(entry.findtext("a:summary", default="", namespaces=_ATOM_NS))
    published = (entry.findtext("a:published", default="", namespaces=_ATOM_NS) or "").strip()
    year      = int(published[:4]) if published[:4].isdigit() else None

    authors = [
        _collapse_ws(a.findtext("a:name", default="", namespaces=_ATOM_NS))
        for a in entry.findall("a:author", _ATOM_NS)
    ]
    authors = [a for a in authors if a]

    cats = []
    primary = entry.find("arxiv:primary_category", _ATOM_NS)
    if primary is not None and primary.get("term"):
        cats.append(primary.get("term"))
    for c in entry.findall("a:category", _ATOM_NS):
        term = c.get("term")
        if term and term not in cats:
            cats.append(term)
    topics = [{"id": "", "display_name": t, "score": 0.0} for t in cats]

    doi_raw = _collapse_ws(entry.findtext("arxiv:doi", default="", namespaces=_ATOM_NS))
    doi = f"https://doi.org/{doi_raw}" if doi_raw else f"https://doi.org/10.48550/arXiv.{base_id}"

    authorships = []
    for i, name in enumerate(authors):
        authorships.append({
            "author_position":  "first" if i == 0 else "middle",
            "is_corresponding": False,
            "author":           {"id": "", "display_name": name, "orcid": None},
            "institutions":     [],
        })

    return {
        "id":               f"https://openalex.org/{work_id}",
        "title":            title,
        "display_name":     title,
        "type":             "article",
        "publication_year": year,
        "publication_date": published[:10],
        "cited_by_count":   0,
        "doi":              doi,
        "abstract_text":    abstract,
        "open_access":      {"oa_status": "green", "oa_url": f"https://arxiv.org/pdf/{arxiv_id}"},
        "authorships":      authorships,
        "topics":           topics,
        "keywords":         [],
        "referenced_works": [],
    }


def fetch_arxiv_metadata(arxiv_id: str, base_id: str, work_id: str) -> Dict:
    r = requests.get(ARXIV_API, params={"id_list": arxiv_id, "max_results": 1},
                     headers=_UA, timeout=30)
    r.raise_for_status()
    return arxiv_atom_to_raw(r.text, arxiv_id, base_id, work_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Download + Docling  (shared)
# ═══════════════════════════════════════════════════════════════════════════════

def download_pdf(arxiv_id: str, work_id: str, dest_dir: Optional[str] = None) -> Path:
    ft_dir = Path(dest_dir or config.FULLTEXT_DIR)
    ft_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = ft_dir / f"{work_id}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        logger.info("PDF already downloaded: %s", pdf_path)
        return pdf_path

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    logger.info("Downloading %s", url)
    timeout = getattr(config, "DOWNLOAD_TIMEOUT", 60)
    r = requests.get(url, headers=_UA, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    content = r.content
    if not content or content[:4] != b"%PDF":
        raise RuntimeError(f"Downloaded content is not a PDF (arxiv_id={arxiv_id})")
    pdf_path.write_bytes(content)
    logger.info("Saved %d KB → %s", len(content) // 1024, pdf_path)
    return pdf_path


def process_pdf(work_id: str, pdf_path: Path, use_docling: bool) -> Tuple[str, str]:
    if use_docling:
        converter = fetch._get_converter()
        if converter is not None:
            try:
                conv_res = converter.convert(str(pdf_path))
                sections = fetch._docling_to_sections(conv_res)
                if sections:
                    saved = fetch._save_sections(work_id, str(pdf_path), sections, fetch.STATUS_OK)
                    return fetch.STATUS_OK, saved
                logger.warning("Docling produced 0 sections — falling back to pdfminer")
            except Exception as e:
                logger.warning("Docling failed (%s) — falling back to pdfminer", e)
    else:
        logger.info("Docling disabled — using pdfminer text extraction")
    return fetch._pdfminer_fallback(work_id, str(pdf_path))


# ═══════════════════════════════════════════════════════════════════════════════
# Refactored-mode record builder  (unchanged behavior)
# ═══════════════════════════════════════════════════════════════════════════════

def build_record(arxiv_url_or_id: str, use_docling: bool = True) -> Tuple[Dict, str]:
    arxiv_id, base_id, work_id = parse_arxiv_id(arxiv_url_or_id)
    logger.info("arXiv id: %s (base %s)", arxiv_id, base_id)

    raw = fetch_openalex_by_arxiv(base_id)
    if raw is not None:
        source = "openalex"
        raw["abstract_text"] = (
            fetch._reconstruct_abstract(raw.pop("abstract_inverted_index", None))
            or raw.get("abstract_text", "")
        )
        work_id = ingest._oa_id(raw.get("id")) or work_id
        logger.info("Metadata: OpenAlex (%s)", work_id)
    else:
        source = "arxiv"
        raw = fetch_arxiv_metadata(arxiv_id, base_id, work_id)
        logger.info("Metadata: arXiv API (not in OpenAlex)")

    pdf_path = download_pdf(arxiv_id, work_id)
    raw["fulltext_status"] = fetch.FT_PDF
    raw["fulltext_path"]   = str(pdf_path)

    ds, dp = process_pdf(work_id, pdf_path, use_docling)
    raw["docling_status"] = ds
    raw["docling_path"]   = dp
    logger.info("Docling status: %s", ds)

    norm = ingest.normalize_work(raw)
    if norm is None:
        raise RuntimeError("normalize_work returned None (missing id or title)")
    return norm, source


def add_chunks_to_collection(chunks: List[Dict], chroma_dir: str, collection_name: str,
                             embed_model: str, embed_device: str, force: bool = False) -> Dict:
    if not chunks:
        return {"added": 0, "reason": "no chunks"}

    import chromadb
    embedder = ingest._make_embedder(embed_model, embed_device)

    os.makedirs(chroma_dir, exist_ok=True)
    client     = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"},
    )

    ids = [c["chunk_id"] for c in chunks]
    try:
        existing = collection.get(ids=ids)
        present  = set(existing.get("ids") or [])
    except Exception:
        present = set()
    if present and not force:
        logger.warning("Paper already present in '%s' (%d chunks). Use --force to re-embed.",
                       collection_name, len(present))
        return {"added": 0, "already_present": len(present), "collection": collection_name}

    docs  = [c["embed_text"] for c in chunks]
    metas = []
    for c in chunks:
        m = {k: ingest._meta_safe(v) for k, v in c["metadata"].items()}
        m["work_id"] = c["work_id"]
        metas.append(m)

    vectors = embedder.embed_documents(docs)
    for start in range(0, len(ids), ingest.CHROMA_MAX_BATCH):
        end = start + ingest.CHROMA_MAX_BATCH
        collection.upsert(
            ids=ids[start:end], documents=docs[start:end],
            embeddings=vectors[start:end], metadatas=metas[start:end],
        )

    total = collection.count()
    logger.info("Upserted %d chunks into '%s' (collection now holds %d)",
                len(ids), collection_name, total)
    return {"added": len(ids), "collection": collection_name, "collection_total": total}


def add_to_neo4j(work: Dict) -> Dict:
    if not ingest.HAS_NEO4J:
        logger.warning("neo4j driver not installed — skipping graph update")
        return {"status": "skipped"}

    graph = ingest.Neo4jGraph()
    try:
        graph.create_constraints()
        wid = work["openalex_id"]
        graph.upsert_works([{
            "oid": wid, "title": work.get("title", ""), "doi": work.get("doi", ""),
            "year": work.get("publication_year"), "cited": work.get("cited_by_count", 0),
            "type": work.get("work_type", ""), "researcher": work.get("primary_researcher", ""),
            "docling_status": work.get("docling_status", "none"),
        }])
        author_nodes, authored = [], []
        for a in (work.get("all_authors") or []):
            aid = a.get("id", "")
            if not aid:
                continue
            author_nodes.append({"oid": aid, "name": a.get("name", ""), "orcid": a.get("orcid", ""),
                                 "works_count": 0, "cited": 0, "h_index": 0})
            authored.append({"aid": aid, "wid": wid, "pos": a.get("position", ""),
                             "corr": a.get("is_corresponding", False)})
        graph.upsert_authors(author_nodes)
        graph.create_authored(authored)
        topic_nodes, topic_edges = [], []
        for t in (work.get("topics") or []):
            tid = t.get("id", "")
            if not tid:
                continue
            topic_nodes.append({"oid": tid, "name": t.get("name", ""), "subfield": t.get("subfield", ""),
                                "field": t.get("field", ""), "domain": t.get("domain", "")})
            topic_edges.append({"wid": wid, "tid": tid, "score": float(t.get("score", 0))})
        graph.upsert_topics(topic_nodes)
        graph.create_has_topic(topic_edges)
        logger.info("Added work %s to Neo4j (%d authors, %d topics)",
                    wid, len(author_nodes), len(topic_nodes))
        return {"work": wid, "authors": len(author_nodes), "topics": len(topic_nodes)}
    finally:
        graph.close()


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY MODE  (--paper-id): attach body to an existing paper_id in papers_all
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text(pdf_path: Path, use_docling: bool = True) -> str:
    """Self-contained PDF → text: docling markdown → pdfminer → pypdf."""
    if use_docling:
        try:
            from docling.document_converter import DocumentConverter
            conv = DocumentConverter().convert(str(pdf_path))
            md = conv.document.export_to_markdown()
            if md and len(md) > 500:
                logger.info("Docling extracted %d chars", len(md)); return md
            logger.warning("Docling output thin — falling back")
        except Exception as e:
            logger.warning("Docling unavailable/failed (%s) — falling back", e)
    try:
        from pdfminer.high_level import extract_text as _pm
        txt = _pm(str(pdf_path)) or ""
        if len(txt) > 500:
            logger.info("pdfminer extracted %d chars", len(txt)); return txt
    except Exception as e:
        logger.warning("pdfminer failed (%s)", e)
    try:
        from pypdf import PdfReader
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages)
        logger.info("pypdf extracted %d chars", len(txt)); return txt
    except Exception as e:
        logger.error("pypdf failed (%s)", e)
    return ""


def _chunk_body(text: str, max_chars: int = LEGACY_CHUNK_MAX_CHARS) -> List[str]:
    text = re.sub(r"[ \t]+", " ", text or "")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 > max_chars and buf:
            chunks.append(buf.strip()); buf = p
        else:
            buf = (buf + "\n" + p) if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if len(c) > 80]


def run_legacy_attach(args) -> None:
    """Attach a paper's body to an existing paper_id, matching papers_all schema."""
    import chromadb
    chroma_dir = args.chroma_dir or config.CHROMA_DIR
    collection = args.collection or config.CHROMA_COLLECTION
    embed_model = args.embed_model or config.EMBED_MODEL
    embed_device = args.embed_device or getattr(config, "EMBED_DEVICE", "cpu")
    pid = str(args.paper_id).strip()

    col = chromadb.PersistentClient(path=chroma_dir).get_collection(collection)
    existing = col.get(where={"paper_id": pid}, include=["documents", "metadatas"])
    if not existing["ids"]:
        sys.exit(f"paper_id {pid} not found in '{collection}' @ {chroma_dir}")
    base_meta = dict(existing["metadatas"][0])
    title = str(base_meta.get("title", "")).strip()
    logger.info("Attaching to paper_id=%s title=%r (currently %d chunk(s))",
                pid, title[:60], len(existing["ids"]))

    arxiv_id, _, work_id = parse_arxiv_id(args.arxiv)
    logger.info("arXiv id: %s", arxiv_id)
    pdf = download_pdf(arxiv_id, work_id, dest_dir=args.chroma_dir and "data/fulltext")
    body = _extract_text(pdf, use_docling=not args.no_docling)
    if len(body) < 500:
        sys.exit(f"body extraction too thin ({len(body)} chars) — "
                 f"pip install docling pdfminer.six pypdf, or check the PDF")
    body_chunks = _chunk_body(body)[: args.max_chunks]
    logger.info("body → %d section chunks", len(body_chunks))

    def _meta(idx: int, ctype: str) -> Dict:
        m = dict(base_meta)                 # inherit title/researcher/doi/dates/etc.
        m["chunk"] = idx
        m["chunk_type"] = ctype
        m["chunk_id"] = f"{pid}_{ctype}_{idx}"
        m["has_body"] = True
        m["arxiv_id"] = arxiv_id
        return m

    ta_text = existing["documents"][0] if existing["documents"] else title
    ids  = [f"{pid}_title_abstract_1"]
    docs = [ta_text or title]
    metas = [_meta(1, "title_abstract")]
    for i, ch in enumerate(body_chunks, start=2):
        ids.append(f"{pid}_section_{i}")
        docs.append(ch)
        metas.append(_meta(i, "section"))

    print("\n" + "═" * 60)
    print(f"  paper_id:   {pid}")
    print(f"  title:      {title[:66]}")
    print(f"  arxiv:      {arxiv_id}")
    print(f"  new chunks: {len(ids)}  (1 title_abstract + {len(body_chunks)} section)")
    print(f"  has_body:   False → True")
    print("═" * 60 + "\n")

    if args.dry_run:
        print("--dry-run: nothing written.")
        for m, d in list(zip(metas, docs))[:3]:
            print(f"  [{m['chunk_type']} {m['chunk']}] {d[:110]!r}")
        return

    embedder = ingest._make_embedder(embed_model, embed_device)
    vectors = embedder.embed_documents(docs)

    if args.replace and existing["ids"]:
        col.delete(ids=existing["ids"])
        logger.info("deleted %d existing chunk(s) for paper_id=%s", len(existing["ids"]), pid)

    for s in range(0, len(ids), LEGACY_CHROMA_MAX_BATCH):
        e = s + LEGACY_CHROMA_MAX_BATCH
        col.upsert(ids=ids[s:e], documents=docs[s:e],
                   embeddings=vectors[s:e], metadatas=metas[s:e])

    print(f"Legacy attach: upserted {len(ids)} chunks for paper_id={pid}. "
          f"Collection now holds {col.count()}.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Add a single arXiv paper to an existing DB (no rebuild).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("arxiv", help="arXiv URL or id, e.g. https://arxiv.org/abs/2205.13487")
    parser.add_argument("--collection", default=None, help="Target collection (default: config.CHROMA_COLLECTION)")
    parser.add_argument("--chroma-dir", default=None, help="ChromaDB dir (default: config.CHROMA_DIR)")
    parser.add_argument("--embed-model", default=None, help="Embed model path (default: config.EMBED_MODEL)")
    parser.add_argument("--embed-device", default=None, help="cpu|cuda (default: config.EMBED_DEVICE)")
    parser.add_argument("--paper-id", default=None,
                        help="LEGACY MODE: attach body to this existing paper_id in papers_all")
    parser.add_argument("--replace", action="store_true",
                        help="(legacy) delete the paper_id's existing chunks first")
    parser.add_argument("--max-chunks", type=int, default=60, help="(legacy) cap body section chunks")
    parser.add_argument("--abstracts",  action="store_true", help="(default mode) also add to abstracts collection")
    parser.add_argument("--neo4j",      action="store_true", help="(default mode) also add to Neo4j graph")
    parser.add_argument("--no-docling", action="store_true", help="Skip Docling; title+abstract (+pdfminer) only")
    parser.add_argument("--force",      action="store_true", help="(default mode) re-embed even if already present")
    parser.add_argument("--dry-run",    action="store_true", help="Fetch/normalize/chunk only; write nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    # ── LEGACY MODE ───────────────────────────────────────────────
    if args.paper_id:
        run_legacy_attach(args)
        return

    # ── DEFAULT (refactored pipeline) MODE ───────────────────────
    collection   = args.collection or config.CHROMA_COLLECTION
    chroma_dir   = args.chroma_dir or config.CHROMA_DIR
    embed_model  = args.embed_model or config.EMBED_MODEL
    embed_device = args.embed_device or getattr(config, "EMBED_DEVICE", "cpu")

    work, source = build_record(args.arxiv, use_docling=not args.no_docling)

    chunks     = ingest.chunk_work(work)
    abs_chunks = ingest._abstract_chunks(work)
    types: Dict[str, int] = {}
    for c in chunks:
        types[c["chunk_type"]] = types.get(c["chunk_type"], 0) + 1

    print("\n" + "═" * 60)
    print(f"  Paper:      {work.get('title', '')[:70]}")
    print(f"  Work id:    {work['openalex_id']}  (metadata: {source})")
    print(f"  Authors:    {work.get('authors_str', '')[:70]}")
    print(f"  Year:       {work.get('publication_year')}   Fulltext: {work.get('docling_status')}")
    print(f"  Full chunks:     {len(chunks)}  {types}")
    print(f"  Abstract chunks: {len(abs_chunks)}")
    print("═" * 60 + "\n")

    if args.dry_run:
        print("--dry-run: nothing written. Remove --dry-run to add to the database.")
        return

    result = add_chunks_to_collection(chunks, chroma_dir, collection,
                                      embed_model, embed_device, force=args.force)
    print(f"Full collection:      {result}")

    if args.abstracts:
        r = add_chunks_to_collection(abs_chunks, config.CHROMA_ABSTRACTS_DIR,
                                     config.CHROMA_ABSTRACTS_COLLECTION,
                                     embed_model, embed_device, force=args.force)
        print(f"Abstracts collection: {r}")

    if args.neo4j:
        r = add_to_neo4j(work)
        print(f"Neo4j graph:          {r}")


if __name__ == "__main__":
    main()
