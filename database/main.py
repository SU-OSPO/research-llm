"""
main.py — Syracuse RAG ingestion pipeline orchestrator.

Stages:
  fetch    fetch.run_fetch       (OpenAlex API + fulltext download)
  docling  fetch.run_docling     (Docling PDF/TEI processing)
  ingest   ingest.run_normalize → run_chroma [→ run_abstracts] [→ run_neo4j]

Usage:
  python main.py                          # full pipeline
  python main.py --module fetch           # fetch + download only
  python main.py --module docling         # docling only
  python main.py --module ingest          # normalize + chroma + neo4j
  python main.py --skip-download          # fetch only, no PDFs
  python main.py --skip-docling           # skip docling step
  python main.py --skip-neo4j             # skip neo4j
  python main.py --abstracts              # also build the abstracts collection
  python main.py --incremental            # only new records
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _banner(title: str):
    logger.info("")
    logger.info("┌" + "─" * 58 + "┐")
    logger.info("│  %-56s│" % title)
    logger.info("└" + "─" * 58 + "┘")


def main():
    parser = argparse.ArgumentParser(
        description="Syracuse RAG — Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modules:
  fetch    — Pull works/authors from OpenAlex + download PDFs/TEI
  docling  — Process downloaded files through Docling
  ingest   — Normalize → ChromaDB [→ abstracts] → Neo4j
  all      — Run everything (default)

Examples:
  python main.py                        Full pipeline
  python main.py --module fetch         Fetch + download only
  python main.py --module docling       Docling only
  python main.py --module ingest        Normalize + Chroma + Neo4j
  python main.py --skip-download        No PDF download
  python main.py --skip-docling         Skip Docling
  python main.py --skip-neo4j           No Neo4j
  python main.py --abstracts            Also build abstracts collection
  python main.py --incremental          New records only
        """,
    )
    parser.add_argument("--module", choices=["fetch", "docling", "ingest", "all"], default="all")
    parser.add_argument("--incremental",   action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-docling",  action="store_true")
    parser.add_argument("--skip-neo4j",    action="store_true")
    parser.add_argument("--abstracts",     action="store_true",
                        help="Also build the title+abstract-only ChromaDB collection")
    args = parser.parse_args()

    t_start = time.time()
    results = {}

    # ── Module 1: Fetch + Download ─────────────────────────────────────────────
    if args.module in ("fetch", "all"):
        _banner("MODULE 1: Fetch from OpenAlex + Download Fulltexts")
        t0 = time.time()
        from fetch import run_fetch
        result = run_fetch(incremental=args.incremental, skip_download=args.skip_download)
        logger.info("Module 1 done in %.1fs: %s", time.time() - t0, result)
        results["fetch"] = result

    # ── Module 2: Docling ──────────────────────────────────────────────────────
    run_docling_step = (
        args.module in ("docling", "all")
        and not args.skip_docling
        and not args.skip_download
    )
    if run_docling_step:
        _banner("MODULE 2: Docling Fulltext Processing")
        t0 = time.time()
        from fetch import run_docling
        result = run_docling(incremental=args.incremental)
        logger.info("Module 2 done in %.1fs: %s", time.time() - t0, result)
        results["docling"] = result
    elif args.module in ("docling", "all"):
        logger.info("Module 2 (Docling): SKIPPED")

    # ── Module 3: Ingest ───────────────────────────────────────────────────────
    if args.module in ("ingest", "all"):
        rebuild = not args.incremental

        # 3a. Normalize
        _banner("MODULE 3a: Normalize")
        t0 = time.time()
        from ingest import run_normalize
        result = run_normalize()
        logger.info("Normalize done in %.1fs: %s", time.time() - t0, result)
        results["normalize"] = result

        # 3b. ChromaDB (full chunked)
        _banner("MODULE 3b: Ingest into ChromaDB")
        t0 = time.time()
        from ingest import run_chroma
        result = run_chroma(rebuild=rebuild)
        logger.info("Chroma done in %.1fs: %s", time.time() - t0, result)
        results["chroma"] = result

        # 3c. ChromaDB (abstracts) — optional
        if args.abstracts:
            _banner("MODULE 3c: Ingest Abstracts Collection")
            t0 = time.time()
            from ingest import run_abstracts
            result = run_abstracts(rebuild=rebuild)
            logger.info("Abstracts done in %.1fs: %s", time.time() - t0, result)
            results["abstracts"] = result

        # 3d. Neo4j
        if not args.skip_neo4j:
            _banner("MODULE 3d: Build Neo4j Knowledge Graph")
            t0 = time.time()
            try:
                from ingest import run_neo4j
                result = run_neo4j(rebuild=rebuild)
                logger.info("Neo4j done in %.1fs: %s", time.time() - t0, result)
                results["neo4j"] = result
            except Exception as e:
                logger.error("Neo4j failed: %s", e)
                logger.info("Use --skip-neo4j to skip")
        else:
            logger.info("Module 3d (Neo4j): SKIPPED")

    # ── Summary ────────────────────────────────────────────────────────────────
    _banner(f"Pipeline complete in {time.time() - t_start:.1f}s")
    for k, v in results.items():
        logger.info("  %-12s %s", k + ":", v)


if __name__ == "__main__":
    main()
