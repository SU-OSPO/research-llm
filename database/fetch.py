"""
fetch.py — Data acquisition: pull from OpenAlex, download fulltexts, run Docling.

Merges the former fetch_and_download.py and docling_process.py into one module
with two public entry points:

    run_fetch(incremental=False, skip_download=False)
        1. Fetch Syracuse University works + authors from the OpenAlex API
           → data/raw/works.jsonl, data/raw/authors.jsonl
        2. Download fulltext per work (OpenAlex TEI → OA PDF → Unpaywall)
           → data/fulltext/{work_id}.tei.xml | .pdf
        3. Annotate with fulltext_status / fulltext_path
           → data/raw/works_with_fulltext.jsonl

    run_docling(incremental=False)
        Process downloaded fulltexts through Docling (pdfminer fallback)
           → data/docling/{work_id}.json
           → data/raw/works_with_docling.jsonl  (docling_status / docling_path)

Standalone:
    python fetch.py fetch [--incremental] [--skip-download]
    python fetch.py docling [--incremental]
    python fetch.py all [--incremental] [--skip-download]

Install (download + docling):
    pip install requests docling pdfminer.six
    pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Suppress Docling's per-document initialization noise
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling_core").setLevel(logging.WARNING)
logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
logging.getLogger("docling.pipeline").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

API_BASE     = "https://api.openalex.org"
FULLTEXT_DIR = Path(config.FULLTEXT_DIR)
DOCLING_DIR  = Path(config.DOCLING_DIR)

# fulltext_status constants
FT_TEI  = "tei_xml"
FT_PDF  = "pdf"
FT_NONE = "none"

# docling_status constants
STATUS_OK       = "docling_ok"
STATUS_FALLBACK = "fallback_pdf"
STATUS_NONE     = "none"

DOCLING_BATCH_SIZE = 8   # number of PDFs per Docling batch — tune if OOM


# ═══════════════════════════════════════════════════════════════════════════════
# ░░  PART 1 — FETCH + DOWNLOAD  (was fetch_and_download.py)                    ░░
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sync state (watermark for incremental fetches) ──────────────────────────────
class SyncState:
    def __init__(self):
        self.path = Path(config.SYNC_STATE_FILE)

    def load(self) -> Dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self, state: Dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def get_watermark(self) -> Optional[str]:
        return self.load().get("last_updated_date")

    def update(self, counts: Dict):
        state = self.load()
        state["last_updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state["last_sync_at"]      = datetime.now(timezone.utc).isoformat()
        for k, v in counts.items():
            state[k] = state.get(k, 0) + v
        self.save(state)


# ── OpenAlex helpers ────────────────────────────────────────────────────────────
def _api_params(**extra) -> Dict:
    params: Dict[str, Any] = {}
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL
    params.update(extra)
    return params


def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
    if not inverted_index:
        return ""
    positions: List[Tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for pos in idxs:
            positions.append((pos, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _cursor_paginate(endpoint: str, params: Dict, desc: str = "records") -> Iterator[Dict]:
    params = dict(params)
    params["cursor"] = "*"
    total    = 0
    page_num = 0

    while True:
        page_num += 1
        try:
            url  = f"{API_BASE}/{endpoint}"
            resp = requests.get(url, params=params, timeout=30)
            if page_num == 1:
                logger.info("API → %s", resp.url)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("API error on %s: %s", endpoint, e)
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", 5))
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
                continue
            raise

        data    = resp.json()
        meta    = data.get("meta", {})
        results = data.get("results", [])

        if page_num == 1:
            logger.info("Total available: %s", meta.get("count", "?"))

        for item in results:
            total += 1
            yield item

        next_cursor = meta.get("next_cursor")
        if not next_cursor or not results:
            break

        params["cursor"] = next_cursor
        time.sleep(config.OPENALEX_RATE_DELAY)

        if total % 1000 == 0:
            logger.info("  ... %d %s fetched", total, desc)

    logger.info("Fetched %d %s total", total, desc)


def _verify_institution(inst_id: str) -> bool:
    try:
        resp = requests.get(f"{API_BASE}/institutions/{inst_id}", params=_api_params(), timeout=15)
        if resp.status_code == 200:
            d = resp.json()
            logger.info("Institution: %s (%s) — %d works", inst_id, d.get("display_name"), d.get("works_count", 0))
            return True
        logger.error(
            "Institution %s not found (HTTP %d). "
            "Find ID at: https://api.openalex.org/autocomplete/institutions?q=syracuse+university",
            inst_id, resp.status_code,
        )
        return False
    except Exception:
        logger.error("Could not verify institution %s", inst_id, exc_info=True)
        return False


# ── Fetch ───────────────────────────────────────────────────────────────────────
def fetch_works(from_updated_date: Optional[str] = None) -> Iterator[Dict]:
    inst_id = config.OPENALEX_INSTITUTION_ID
    if not _verify_institution(inst_id):
        raise RuntimeError(f"Invalid institution ID: {inst_id}")

    filters = [f"authorships.institutions.lineage:{inst_id}"]
    if from_updated_date:
        filters.append(f"from_updated_date:{from_updated_date}")
        logger.info("Incremental fetch — works updated since %s", from_updated_date)
    else:
        logger.info("Full fetch — all works for %s", inst_id)

    params = _api_params(filter=",".join(filters), per_page=config.OPENALEX_PER_PAGE)

    for work in _cursor_paginate("works", params, desc="works"):
        work["abstract_text"] = _reconstruct_abstract(work.pop("abstract_inverted_index", None))
        yield work


def fetch_authors() -> Iterator[Dict]:
    inst_id = config.OPENALEX_INSTITUTION_ID
    logger.info("Fetching authors for %s", inst_id)
    params = _api_params(
        filter=f"affiliations.institution.lineage:{inst_id}",
        per_page=config.OPENALEX_PER_PAGE,
    )
    yield from _cursor_paginate("authors", params, desc="authors")


# ── Download helpers ──────────────────────────────────────────────────────────
def _http_get(url: str, params: Optional[Dict] = None, stream: bool = False) -> Optional[requests.Response]:
    for attempt in range(config.DOWNLOAD_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params or {}, timeout=config.DOWNLOAD_TIMEOUT, stream=stream)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", 10))
                time.sleep(wait)
                continue
            if resp.status_code in (403, 404, 410):
                return None
        except requests.exceptions.RequestException:
            if attempt < config.DOWNLOAD_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def _save_text(path: Path, content: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning("Save failed %s: %s", path, e)
        return False


def _save_bytes(path: Path, content: bytes) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return True
    except Exception as e:
        logger.warning("Save failed %s: %s", path, e)
        return False


# ── Download strategies ───────────────────────────────────────────────────────
def _try_openalex_tei(work_id: str, tei_path: Path) -> bool:
    url    = f"{API_BASE}/works/{work_id}/fulltext"
    params = _api_params()
    resp   = _http_get(url, params=params, stream=False)
    if resp is None:
        return False
    try:
        data        = resp.json()
        tei_content = data.get("tei_xml") or data.get("fulltext") or ""
        if tei_content and len(tei_content) > 200:
            return _save_text(tei_path, tei_content)
    except Exception:
        pass
    return False


def _try_pdf(url: str, pdf_path: Path) -> bool:
    if not url:
        return False
    resp = _http_get(url, stream=True)
    if resp is None:
        return False
    content = resp.content
    if content and content[:4] == b"%PDF":
        return _save_bytes(pdf_path, content)
    return False


def _try_unpaywall(doi: str, pdf_path: Path) -> bool:
    if not doi or not config.UNPAYWALL_EMAIL:
        return False
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi_clean:
        return False
    resp = _http_get(f"https://api.unpaywall.org/v2/{doi_clean}", params={"email": config.UNPAYWALL_EMAIL})
    if resp is None:
        return False
    try:
        data    = resp.json()
        best    = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or ""
        if not pdf_url:
            for loc in (data.get("oa_locations") or []):
                pdf_url = loc.get("url_for_pdf") or ""
                if pdf_url:
                    break
        if pdf_url:
            return _try_pdf(pdf_url, pdf_path)
    except Exception:
        pass
    return False


def _download_work(work: Dict) -> Tuple[str, str]:
    """Download fulltext for a single work. Returns (fulltext_status, local_path)."""
    raw_id  = work.get("id") or ""
    work_id = raw_id.rsplit("/", 1)[-1]
    if not work_id:
        return FT_NONE, ""

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    tei_path = FULLTEXT_DIR / f"{work_id}.tei.xml"
    pdf_path = FULLTEXT_DIR / f"{work_id}.pdf"

    # Already downloaded?
    if tei_path.exists() and tei_path.stat().st_size > 200:
        return FT_TEI, str(tei_path)
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return FT_PDF, str(pdf_path)

    # Strategy 1: OpenAlex TEI XML
    if _try_openalex_tei(work_id, tei_path):
        return FT_TEI, str(tei_path)

    # Strategy 2: OA PDF from oa_url
    oa     = work.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    if oa_url and _try_pdf(oa_url, pdf_path):
        return FT_PDF, str(pdf_path)

    # Strategy 3: PDF from primary_location
    primary = work.get("primary_location") or {}
    pdf_url = primary.get("pdf_url") or ""
    if not pdf_url:
        landing = primary.get("landing_page_url") or ""
        if landing.endswith(".pdf"):
            pdf_url = landing
    if pdf_url and pdf_url != oa_url and _try_pdf(pdf_url, pdf_path):
        return FT_PDF, str(pdf_path)

    # Strategy 4: Unpaywall
    doi = work.get("doi") or ""
    if doi and _try_unpaywall(doi, pdf_path):
        return FT_PDF, str(pdf_path)

    return FT_NONE, ""


def run_fetch(incremental: bool = False, skip_download: bool = False) -> Dict[str, Any]:
    """Fetch works + authors from OpenAlex, download fulltexts, write annotated JSONL."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    sync    = SyncState()
    from_dt = sync.get_watermark() if incremental else None
    mode    = "a" if incremental else "w"

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    logger.info("━" * 50)
    logger.info("FETCH: Pulling works from OpenAlex")
    logger.info("━" * 50)

    works_raw_path   = Path(config.RAW_DIR) / "works.jsonl"
    authors_raw_path = Path(config.RAW_DIR) / "authors.jsonl"

    works_list: List[Dict] = []
    with open(works_raw_path, mode, encoding="utf-8") as f:
        for work in fetch_works(from_updated_date=from_dt):
            f.write(json.dumps(work, ensure_ascii=False) + "\n")
            works_list.append(work)
            if len(works_list) % 500 == 0:
                logger.info("  ... %d works written", len(works_list))

    logger.info("Works fetched: %d → %s", len(works_list), works_raw_path)

    authors_count = 0
    with open(authors_raw_path, mode, encoding="utf-8") as f:
        for author in fetch_authors():
            f.write(json.dumps(author, ensure_ascii=False) + "\n")
            authors_count += 1

    logger.info("Authors fetched: %d → %s", authors_count, authors_raw_path)

    if skip_download:
        out_path = Path(config.RAW_DIR) / "works_with_fulltext.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for work in works_list:
                work["fulltext_status"] = FT_NONE
                work["fulltext_path"]   = ""
                f.write(json.dumps(work, ensure_ascii=False) + "\n")
        logger.info("Download skipped — all works marked as title+abstract only")
        sync.update({"works_fetched": len(works_list), "authors_fetched": authors_count})
        return {"works": len(works_list), "authors": authors_count, FT_TEI: 0, FT_PDF: 0, FT_NONE: len(works_list)}

    # ── Step 2: Download ──────────────────────────────────────────────────────
    logger.info("━" * 50)
    logger.info("DOWNLOAD: Fetching fulltexts (%d workers)", config.DOWNLOAD_WORKERS)
    logger.info("━" * 50)

    already_done: Dict[str, Dict] = {}
    out_path = Path(config.RAW_DIR) / "works_with_fulltext.jsonl"
    if incremental and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    w   = json.loads(line)
                    wid = (w.get("id") or "").rsplit("/", 1)[-1]
                    if wid and w.get("fulltext_status"):
                        already_done[wid] = w
                except json.JSONDecodeError:
                    continue

    to_download = [
        w for w in works_list
        if (w.get("id") or "").rsplit("/", 1)[-1] not in already_done
    ]

    logger.info("To download: %d (already done: %d)", len(to_download), len(already_done))

    counts  = {FT_TEI: 0, FT_PDF: 0, FT_NONE: 0}
    results: Dict[str, Tuple[str, str]] = {}

    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(_download_work, w): w for w in to_download}
        done = 0
        for future in as_completed(future_map):
            work = future_map[future]
            wid  = (work.get("id") or "").rsplit("/", 1)[-1]
            try:
                status, path = future.result()
            except Exception as e:
                logger.warning("Download error %s: %s", wid, e)
                status, path = FT_NONE, ""

            results[wid]   = (status, path)
            counts[status] = counts.get(status, 0) + 1
            done          += 1

            if done % 100 == 0 or done == len(to_download):
                logger.info(
                    "Progress: %d/%d | TEI: %d | PDF: %d | None: %d",
                    done, len(to_download), counts[FT_TEI], counts[FT_PDF], counts[FT_NONE],
                )

    # ── Step 3: Write output JSONL ─────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        for w in already_done.values():
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
        for work in to_download:
            wid                     = (work.get("id") or "").rsplit("/", 1)[-1]
            status, path            = results.get(wid, (FT_NONE, ""))
            work["fulltext_status"] = status
            work["fulltext_path"]   = path
            f.write(json.dumps(work, ensure_ascii=False) + "\n")

    summary = {
        "works":   len(works_list),
        "authors": authors_count,
        FT_TEI:    counts[FT_TEI],
        FT_PDF:    counts[FT_PDF],
        FT_NONE:   counts[FT_NONE],
    }
    logger.info("━" * 50)
    logger.info("DONE: %s → %s", summary, out_path)
    logger.info("━" * 50)

    sync.update({"works_fetched": len(works_list), "authors_fetched": authors_count})
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# ░░  PART 2 — DOCLING PROCESSING  (was docling_process.py)                     ░░
# ═══════════════════════════════════════════════════════════════════════════════

_CONVERTER = None


def _build_converter():
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError:
        logger.error("Docling not installed — run: pip install docling")
        return None

    # Detect GPU
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info("CUDA available: %s (%.1f GB VRAM)",
                        torch.cuda.get_device_name(0),
                        torch.cuda.get_device_properties(0).total_memory / 1e9)
    except ImportError:
        pass

    # Resolve accelerator device
    accelerator_device = "cpu"
    if use_gpu:
        try:
            from docling.datamodel.pipeline_options import AcceleratorDevice
            accelerator_device = AcceleratorDevice.CUDA
        except (ImportError, AttributeError):
            os.environ["DOCLING_DEVICE"] = "cuda"
            accelerator_device = "cuda"

    # Build pipeline options — explicit and consistent so the hash never changes
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr             = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=accelerator_device if accelerator_device != "cpu" else AcceleratorDevice.CPU,
        )
        logger.info("Docling accelerator: %s via AcceleratorOptions", "GPU (CUDA)" if use_gpu else "CPU")
    except (ImportError, AttributeError):
        if use_gpu:
            try:
                pipeline_options.accelerator_device = AcceleratorDevice.CUDA
                logger.info("Docling accelerator: GPU via accelerator_device")
            except Exception:
                logger.info("Docling accelerator: GPU via env var")
        else:
            logger.warning("CUDA not available — Docling running on CPU (slow)")

    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        logger.info("Docling converter ready")
        return converter
    except Exception as e:
        logger.error("Failed to build Docling converter: %s", e)
        return None


def _get_converter():
    global _CONVERTER
    if _CONVERTER is None:
        _CONVERTER = _build_converter()
    return _CONVERTER


def _docling_to_sections(conv_res) -> List[Dict]:
    """Convert a Docling ConversionResult into a flat list of section dicts."""
    sections: List[Dict] = []
    current_heading = "Body"
    current_level   = 1
    current_paras:  List[str] = []

    def _flush():
        nonlocal current_paras
        text = " ".join(current_paras).strip()
        if text and len(text) > 30:
            sections.append({
                "heading":      current_heading,
                "level":        current_level,
                "text":         text,
                "element_type": "section",
            })
        current_paras = []

    try:
        # Docling >= 2.x
        for item, _ in conv_res.document.iterate_items():
            itype = type(item).__name__

            if itype in ("SectionHeaderItem", "HeadingItem"):
                _flush()
                current_heading = (getattr(item, "text", "") or "").strip() or current_heading
                current_level   = getattr(item, "level", 1) or 1

            elif itype == "TextItem":
                text = (getattr(item, "text", "") or "").strip()
                if text and len(text) > 10:
                    current_paras.append(text)

            elif itype == "TableItem":
                _flush()
                try:
                    try:
                        table_md = item.export_to_markdown(doc=conv_res.document)
                    except TypeError:
                        table_md = item.export_to_markdown()
                    if table_md and len(table_md.strip()) > 10:
                        sections.append({
                            "heading":      f"{current_heading} [Table]",
                            "level":        current_level,
                            "text":         table_md.strip(),
                            "element_type": "table",
                        })
                except Exception:
                    pass

            elif itype == "FigureItem":
                caption = ""
                try:
                    caption = " ".join(
                        c.text for c in (getattr(item, "captions", []) or [])
                        if hasattr(c, "text")
                    ).strip()
                except Exception:
                    pass
                if caption:
                    _flush()
                    sections.append({
                        "heading":      f"{current_heading} [Figure]",
                        "level":        current_level,
                        "text":         caption,
                        "element_type": "figure_caption",
                    })

        _flush()

    except AttributeError:
        # Docling 1.x fallback
        try:
            d = conv_res.document.export_to_dict()
            for item in d.get("body", []):
                label = item.get("label", "")
                text  = (item.get("text") or "").strip()
                if label in ("section-header", "title"):
                    _flush()
                    current_heading = text or current_heading
                elif label == "text" and text and len(text) > 10:
                    current_paras.append(text)
                elif label == "table" and text:
                    _flush()
                    sections.append({
                        "heading":      f"{current_heading} [Table]",
                        "level":        1,
                        "text":         text,
                        "element_type": "table",
                    })
            _flush()
        except Exception as e:
            logger.debug("Docling 1.x export failed: %s", e)

    return sections


def _pdfminer_text(pdf_path: str) -> str:
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(pdf_path)
        if text and len(text.strip()) > 100:
            return text.strip()
    except ImportError:
        logger.warning("pdfminer.six not installed — run: pip install pdfminer.six")
    except Exception as e:
        logger.debug("pdfminer failed for %s: %s", pdf_path, e)

    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        pages  = [p.extract_text() or "" for p in reader.pages]
        text   = "\n\n".join(p.strip() for p in pages if p.strip())
        if text and len(text.strip()) > 100:
            return text.strip()
    except Exception as e:
        logger.debug("pypdf failed for %s: %s", pdf_path, e)

    return ""


def _save_sections(work_id: str, ft_path: str, sections: List[Dict], status: str) -> str:
    DOCLING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCLING_DIR / f"{work_id}.json"
    out_path.write_text(
        json.dumps({
            "work_id":  work_id,
            "source":   ft_path,
            "status":   status,
            "sections": sections,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(out_path)


def _pdfminer_fallback(wid: str, ft_path: str) -> Tuple[str, str]:
    """Try pdfminer on a PDF. Returns (status, saved_path)."""
    if not ft_path.endswith(".pdf"):
        return STATUS_NONE, ""
    text = _pdfminer_text(ft_path)
    if text:
        saved = _save_sections(wid, ft_path, [{
            "heading": "Full Text", "level": 1,
            "text": text, "element_type": "section",
        }], STATUS_FALLBACK)
        return STATUS_FALLBACK, saved
    return STATUS_NONE, ""


def run_docling(incremental: bool = False) -> Dict[str, int]:
    """Process all works with downloaded fulltexts through Docling (resumable)."""
    works_in  = Path(config.RAW_DIR) / "works_with_fulltext.jsonl"
    works_out = Path(config.RAW_DIR) / "works_with_docling.jsonl"

    if not works_in.exists():
        logger.error("No works_with_fulltext.jsonl — run the fetch step first")
        return {STATUS_OK: 0, STATUS_FALLBACK: 0, STATUS_NONE: 0, "total": 0}

    all_works: List[Dict] = []
    with open(works_in, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_works.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    already_done: Dict[str, Dict] = {}
    if incremental and works_out.exists():
        with open(works_out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    w   = json.loads(line)
                    wid = (w.get("id") or "").rsplit("/", 1)[-1]
                    if wid and "docling_status" in w:
                        already_done[wid] = w
                except json.JSONDecodeError:
                    continue

    # Detect works already processed on disk (safe resume after crash)
    pre_recovered = 0
    to_process    = []
    for w in all_works:
        wid      = (w.get("id") or "").rsplit("/", 1)[-1]
        out_path = DOCLING_DIR / f"{wid}.json"
        if wid in already_done:
            continue
        if out_path.exists() and out_path.stat().st_size > 100:
            w["docling_status"] = STATUS_OK
            w["docling_path"]   = str(out_path)
            already_done[wid]   = w
            pre_recovered      += 1
            continue
        to_process.append(w)

    if pre_recovered:
        logger.info("Auto-recovered %d already-processed works from data/docling/", pre_recovered)

    has_ft = [w for w in to_process if w.get("fulltext_status", STATUS_NONE) != STATUS_NONE]
    no_ft  = [w for w in to_process if w.get("fulltext_status", STATUS_NONE) == STATUS_NONE]

    logger.info(
        "Total: %d | Already done: %d | To process: %d (with fulltext: %d, no fulltext: %d)",
        len(all_works), len(already_done), len(to_process), len(has_ft), len(no_ft),
    )

    counts:  Dict[str, int]             = {STATUS_OK: 0, STATUS_FALLBACK: 0, STATUS_NONE: 0}
    results: Dict[str, Tuple[str, str]] = {}

    path_to_wid:  Dict[str, str]  = {}
    path_to_work: Dict[str, Dict] = {}
    pdf_paths:    List[str]       = []

    for work in has_ft:
        wid     = (work.get("id") or "").rsplit("/", 1)[-1]
        ft_path = work.get("fulltext_path", "")
        if ft_path and Path(ft_path).exists():
            path_to_wid[ft_path]  = wid
            path_to_work[ft_path] = work
            pdf_paths.append(ft_path)
        else:
            results[wid]         = (STATUS_NONE, "")
            counts[STATUS_NONE] += 1

    converter = _get_converter()
    t0        = time.time()
    done      = 0

    if converter is not None and pdf_paths:
        logger.info("Processing %d files through Docling (batch_size=%d)...", len(pdf_paths), DOCLING_BATCH_SIZE)
        try:
            for conv_res in converter.convert_all(pdf_paths, raises_on_error=False):
                try:
                    ft_path = str(conv_res.input.file)
                except Exception:
                    ft_path = ""

                wid = path_to_wid.get(ft_path, "")
                if not wid:
                    for p, w in path_to_wid.items():
                        if Path(p).name == Path(ft_path).name:
                            wid     = w
                            ft_path = p
                            break

                if not wid or wid in results:
                    continue

                done += 1
                try:
                    sections = _docling_to_sections(conv_res)
                    if not sections:
                        raise ValueError("0 sections extracted")
                    saved             = _save_sections(wid, ft_path, sections, STATUS_OK)
                    results[wid]      = (STATUS_OK, saved)
                    counts[STATUS_OK] += 1
                except Exception as e:
                    logger.debug("Extraction failed %s: %s — pdfminer fallback", wid, e)
                    status, saved   = _pdfminer_fallback(wid, ft_path)
                    results[wid]    = (status, saved)
                    counts[status] += 1

                if done % 50 == 0 or done == len(pdf_paths):
                    elapsed = time.time() - t0
                    rate    = done / max(1, elapsed)
                    eta     = (len(pdf_paths) - done) / max(0.01, rate)
                    logger.info(
                        "Progress: %d/%d (%.1f/min) | OK: %d | Fallback: %d | None: %d | ETA: %.0fs",
                        done, len(pdf_paths), rate * 60,
                        counts[STATUS_OK], counts[STATUS_FALLBACK], counts[STATUS_NONE], eta,
                    )

        except Exception as err:
            logger.warning("convert_all failed: %s — falling back per-file", err)
            for ft_path in pdf_paths:
                wid = path_to_wid.get(ft_path, "")
                if not wid or wid in results:
                    continue
                try:
                    conv_res = converter.convert(ft_path)
                    sections = _docling_to_sections(conv_res)
                    if not sections:
                        raise ValueError("0 sections")
                    saved             = _save_sections(wid, ft_path, sections, STATUS_OK)
                    results[wid]      = (STATUS_OK, saved)
                    counts[STATUS_OK] += 1
                except Exception:
                    status, saved   = _pdfminer_fallback(wid, ft_path)
                    results[wid]    = (status, saved)
                    counts[status] += 1
                done += 1

    elif pdf_paths:
        logger.warning("Docling unavailable — using pdfminer fallback for all %d files", len(pdf_paths))
        for ft_path in pdf_paths:
            wid             = path_to_wid.get(ft_path, "")
            status, saved   = _pdfminer_fallback(wid, ft_path)
            results[wid]    = (status, saved)
            counts[status] += 1

    for work in no_ft:
        wid = (work.get("id") or "").rsplit("/", 1)[-1]
        results[wid]         = (STATUS_NONE, "")
        counts[STATUS_NONE] += 1

    with open(works_out, "w", encoding="utf-8") as f:
        for w in already_done.values():
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
        for work in to_process:
            wid                    = (work.get("id") or "").rsplit("/", 1)[-1]
            status, path           = results.get(wid, (STATUS_NONE, ""))
            work["docling_status"] = status
            work["docling_path"]   = path
            f.write(json.dumps(work, ensure_ascii=False) + "\n")

    summary = {**counts, "total": len(all_works)}
    logger.info("━" * 50)
    logger.info(
        "DONE: OK=%d | Fallback=%d | None=%d → %s",
        counts[STATUS_OK], counts[STATUS_FALLBACK], counts[STATUS_NONE], works_out,
    )
    logger.info("━" * 50)
    return summary


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
    parser = argparse.ArgumentParser(description="OpenAlex fetch/download + Docling processing")
    sub = parser.add_subparsers(dest="stage", required=True)

    p_fetch = sub.add_parser("fetch", help="Fetch OpenAlex data and download fulltexts")
    p_fetch.add_argument("--incremental",   action="store_true")
    p_fetch.add_argument("--skip-download", action="store_true")

    p_doc = sub.add_parser("docling", help="Process downloaded fulltexts with Docling")
    p_doc.add_argument("--incremental", action="store_true")

    p_all = sub.add_parser("all", help="fetch → docling")
    p_all.add_argument("--incremental",   action="store_true")
    p_all.add_argument("--skip-download", action="store_true")

    args = parser.parse_args()

    if args.stage == "fetch":
        print(run_fetch(incremental=args.incremental, skip_download=args.skip_download))
    elif args.stage == "docling":
        print(run_docling(incremental=args.incremental))
    elif args.stage == "all":
        print(run_fetch(incremental=args.incremental, skip_download=args.skip_download))
        if not args.skip_download:
            print(run_docling(incremental=args.incremental))


if __name__ == "__main__":
    _cli()
