# cache_manager.py
import gc
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional

from rag_utils import norm_text, short_hash

CACHE_DIR = os.getenv("RAG_CACHE_DIR", "cache")
CACHE_KEY_VERSION = os.getenv("RAG_CACHE_VERSION", "v3")


def state_signature_from_state(state: Dict[str, Any]) -> str:
    if not isinstance(state, dict):
        return "t0|a|s"
    turns = state.get("turns", [])
    extra = state.get("extra_state", {}) if isinstance(state.get("extra_state"), dict) else {}
    anchor_value = ""
    for src in (state, extra):
        if isinstance(src.get("anchor"), dict):
            anchor_value = str(src["anchor"].get("value", "") or "")
            break
    rolling = str(state.get("rolling_summary", "") or "") or str(extra.get("rolling_summary", extra.get("last_focus", "")) or "")
    return f"t{max(0, len(turns or []))}|a{short_hash(anchor_value, length=10)}|s{short_hash(rolling, length=10)}"


def build_cache_key(*, user_key: str, resolved_text: str,
                    effective_mode: str, state_signature: str) -> str:
    return (f"{CACHE_KEY_VERSION}|{norm_text(user_key)}|{norm_text(effective_mode) or 'default'}"
            f"|{norm_text(state_signature)}|{short_hash(norm_text(resolved_text), length=16)}")


def should_cache_turn(*, retrieval_text: str, rewrite_blocked: bool) -> bool:
    return not rewrite_blocked and bool(norm_text(retrieval_text))


def retrieval_cache_summary(docs: Iterable[Any], *, retrieval_text: str, limit_ids: int = 12) -> Dict[str, Any]:
    docs_list = list(docs or [])
    ids: List[str] = []
    for d in docs_list:
        meta = getattr(d, "metadata", {}) or {}
        pid = norm_text(meta.get("paper_id", ""))
        chunk = norm_text(meta.get("chunk", meta.get("chunk_id", meta.get("id", ""))))
        doc_id = f"{pid}::{chunk}".strip(":") if (pid or chunk) else norm_text(meta.get("title", ""))
        if doc_id and doc_id not in ids:
            ids.append(doc_id)
            if len(ids) >= max(1, int(limit_ids)):
                break
    return {
        "retrieval_text_hash": short_hash(retrieval_text, length=12),
        "retrieval_text_chars": len(norm_text(retrieval_text)),
        "doc_count": len(docs_list),
        "doc_ids": ids,
        "doc_ids_hash": short_hash("|".join(ids), length=12),
    }


def _gpu_release() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def clear_cache(*, user_key: Optional[str] = None, mode: Optional[str] = None,
                older_than_seconds: Optional[float] = None, version: Optional[str] = None) -> Dict[str, int]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    _gpu_release()

    filters = []
    if user_key is not None:
        h = short_hash(user_key, length=12)
        filters.append(lambda name, _p, h=h: h in name)
    if mode is not None:
        m = norm_text(mode)
        filters.append(lambda name, _p, m=m: m in name)
    if older_than_seconds is not None:
        limit = float(older_than_seconds)
        filters.append(lambda _n, p, limit=limit: (time.time() - os.path.getmtime(p)) > limit)
    if version is not None:
        filters.append(lambda name, _p, v=str(version): name.startswith(v))
    elif not filters:
        cur = CACHE_KEY_VERSION
        filters.append(lambda name, _p, cur=cur: not name.startswith(cur))

    deleted = skipped = 0
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        if all(f(name, p) for f in filters):
            try:
                (shutil.rmtree if os.path.isdir(p) else os.remove)(p)
                deleted += 1
            except Exception:
                skipped += 1
        else:
            skipped += 1
    return {"deleted": deleted, "skipped": skipped}


def clear_cache_all() -> Dict[str, int]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    _gpu_release()
    deleted = skipped = 0
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        try:
            (shutil.rmtree if os.path.isdir(p) else os.remove)(p)
            deleted += 1
        except Exception:
            skipped += 1
    return {"deleted": deleted, "skipped": skipped}