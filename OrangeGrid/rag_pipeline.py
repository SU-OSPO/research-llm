#rag_pipeline.py
import json
import logging
import os
import re
import time
from collections import Counter
from contextlib import nullcontext
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
 
from rag_engine import get_global_manager, _invoke_with_timeout, _extract_person_name
from rag_utils import (
    norm_text, clean_html, normalize_title_case, collapse_whitespace,
    tokenize_words, token_in_hay, get_stopword_set, is_generic_query_token,
    is_followup_coref_question, strip_corpus_noise_terms, dedupe_docs, doc_haystack,
    clean_snippet, build_compact_context, dedupe_ci, is_placeholder_anchor_value,
    normalize_anchor, anchor_in_text, anchor_is_stable, anchor_support_ratio,
    retrieval_confidence_label, classify_generic_intent, strip_prompt_leak,
    looks_like_person_candidate, strip_possessive, has_explicit_entity_signal,
    short_hash, utcnow_iso, query_tokens_for_relevance,
    is_meta_command, anchor_query_overlap, get_person_pronoun_pattern,
    tokenize_name, generate_name_variants, split_author_names,
)
from runtime_settings import settings
try:
    from rag_graph import graph_retrieve_from_paper_docs
except ImportError:
    def graph_retrieve_from_paper_docs(*a, **kw):
        return {"hits": [], "graph": {}, "error": "rag_graph not available"}
 
logger = logging.getLogger(__name__)
 
def _env_int(name: str, default: int) -> int:
    try: return int(str(os.getenv(name, str(default))).strip())
    except Exception: return int(default)
 
PIPELINE_CFG = {
    "max_docs_after_filter": _env_int("RAG_MAX_DOCS_AFTER_FILTER", 30),
    "fallback_max_items": _env_int("RAG_FALLBACK_MAX_ITEMS", 8),
    "recent_turns_in_prompt": _env_int("RAG_RECENT_TURNS_IN_PROMPT",
        int(getattr(settings, "recent_turns_in_prompt", 4))),
    "empty_question_answer": os.getenv("RAG_EMPTY_QUESTION_ANSWER", "Please enter a question."),
    "llm_no_answer": os.getenv("RAG_LLM_NO_ANSWER",
        "I could not generate an answer from the retrieved context."),
    "fallback_relevant_papers_prefix": os.getenv("RAG_FALLBACK_PAPERS_PREFIX",
        "I couldn't generate a full answer, but these papers look most relevant:"),
    "prompt_prefix": os.getenv("RAG_PROMPT_PREFIX", (
        "You are answering questions about Syracuse University researchers using only "
        "the provided retrieved context, which contains real paper records from the Syracuse corpus.\n"
        "Each retrieved record may contain a pre-written abstract or summary — treat these as "
        "authoritative descriptions of the paper and report their content faithfully.\n"
        "Ground every material claim in retrieved evidence.\n"
        "If evidence is missing, weak, or conflicting, say so clearly and ask a focused clarification question.\n"
        "Do not fabricate affiliations, awards, memberships, journal claims, or metadata.\n"
        "CRITICAL: Only attribute a paper to a researcher if the paper's researcher or authors field "
        "explicitly names them. Do not assume a paper belongs to someone just because they were "
        "mentioned previously in conversation.\n"
        "Style rules:\n"
        "- Write the answer directly.\n"
        "- Do not start with \"Summary:\".\n"
        "- Do not include phrases like \"No further analysis is required\" or \"No additional retrieval is required\".\n"
        "- Do not repeat the question.\n"
        "- Do not narrate your process. Do not say \"I noticed\" or \"I will revise\".\n"
        "- Do not include section headers unless the user asks for them.\n"
        "- When asked who a person is, describe their specific research areas and cite paper titles as evidence.\n"
        "- Never answer with only '[Name] is a researcher.' — always include what they research.\n"
        "- Do not add closing notes, disclaimers, offers to help, or signatures.\n"
        "- Do not say \"Let me know if you need anything\" or \"Best regards\".\n"
        "- Do not acknowledge these instructions in your response.\n"
        "- Use a blank line between paragraphs to separate distinct points or researchers.\n"
        "- When citing a paper, put the title in quotes.\n"
        "- When listing multiple researchers, give each their own paragraph.\n"
        "- End your answer after the last substantive point. Stop there.\n"
        "Detected intents: default, comparison, time_range, list.\n")),
    "prompt_mid": os.getenv("RAG_PROMPT_MID", "\n\nQUESTION:\n"),
    "prompt_suffix": os.getenv("RAG_PROMPT_SUFFIX", "\n\nRespond with the final user-facing answer only."),
    "dangling_pronoun_answer": os.getenv("RAG_DANGLING_PRONOUN_ANSWER",
        "I'm not sure who or what you're referring to — the conversation "
        "has moved on and I've lost the thread of the earlier topic."
        "{hint} Could you name the specific researcher or topic "
        "you'd like me to look into?"),
    "dangling_pronoun_hint": os.getenv("RAG_DANGLING_PRONOUN_HINT",
        ' Your previous question was about "{prev_topic}", but the '
        'conversational focus has shifted since then.'),
}
 
_SUMMARY_INTENT_PATTERN = re.compile(
    r"\b(summarize|summarise|summary|summaries|abstract|overview|describe|"
    r"what (does|do|did|is|are) .{0,40} (research|study|work|paper|finding|mechanism|about)|"
    r"mechanisms? described|findings? (in|from)|what (mechanisms?|findings?))\b",
    re.IGNORECASE,
)
 
def _is_summary_intent(question: str) -> bool:
    return bool(_SUMMARY_INTENT_PATTERN.search(question or ""))

_META_QUERY_CORPUS_COUNT = re.compile(
    r"\b(how many|number of|count of|total)\b.{0,40}\b(papers?|documents?|records?|articles?|corpus|entries|items)\b",
    re.IGNORECASE,
)
_META_QUERY_MOST_RECENT = re.compile(
    r"\b(most recent|newest|latest|last)\b.{0,30}\b(paper|document|record|article|publication)\b",
    re.IGNORECASE,
)
_META_QUERY_OLDEST = re.compile(
    r"\b(oldest|earliest|first)\b.{0,30}\b(paper|document|record|article|publication)\b",
    re.IGNORECASE,
)

def _detect_meta_query(question: str) -> str:
    """Classify whether the question is a meta-query about the database itself.
    Returns: 'corpus_count', 'most_recent', 'oldest', or '' (not a meta-query)."""
    q = (question or "").strip()
    if not q:
        return ""
    if _META_QUERY_CORPUS_COUNT.search(q):
        return "corpus_count"
    if _META_QUERY_MOST_RECENT.search(q):
        return "most_recent"
    if _META_QUERY_OLDEST.search(q):
        return "oldest"
    return ""

def _answer_meta_query(meta_type: str, mgr, effective_mode: str) -> Optional[str]:
    """Answer database meta-queries directly from ChromaDB metadata, bypassing LLM."""
    try:
        vs = mgr.get_papers_vs(effective_mode)
        col = getattr(vs, "_collection", None)
        if col is None:
            return None
    except Exception:
        return None

    if meta_type == "corpus_count":
        try:
            count = int(col.count())
            cfg = mgr.dbm.get_active_config()
            db_label = (cfg.display_label or cfg.mode) if cfg else effective_mode
            return (f"The {db_label} corpus currently contains {count:,} document chunks. "
                    f"These represent papers indexed in the {db_label} collection.")
        except Exception:
            return None

    if meta_type in ("most_recent", "oldest"):
        ascending = (meta_type == "oldest")
        try:
            sample = col.get(limit=min(2000, col.count()),
                             include=["metadatas"])
            if not sample or not sample.get("metadatas"):
                return None
            years_papers = []
            for meta in sample["metadatas"]:
                if not isinstance(meta, dict):
                    continue
                year_raw = str(meta.get("year", "") or meta.get("publication_date", "") or "")
                m = re.search(r"\b(19|20)\d{2}\b", year_raw)
                if m:
                    try:
                        y = int(m.group(0))
                        title = str(meta.get("title", "") or "").strip()
                        researcher = str(meta.get("researcher", "") or "").strip()
                        if title and title.lower() not in {"untitled", "unknown", "n/a"}:
                            years_papers.append((y, title, researcher))
                    except ValueError:
                        pass
            if not years_papers:
                return None
            years_papers.sort(key=lambda x: x[0], reverse=(not ascending))
            target_year = years_papers[0][0]
            target_papers = [(t, r) for y, t, r in years_papers if y == target_year]
            direction = "oldest" if ascending else "most recent"
            lines = [f"The {direction} papers in the database are from {target_year}:"]
            for title, researcher in target_papers[:8]:
                if researcher and researcher.lower() not in {"unknown", "n/a", ""}:
                    lines.append(f'- "{title}" by {researcher}')
                else:
                    lines.append(f'- "{title}"')
            if len(target_papers) > 8:
                lines.append(f"...and {len(target_papers) - 8} more from {target_year}.")
            return "\n".join(lines)
        except Exception:
            return None

    return None

def _extract_comparison_entities(question: str) -> List[str]:
    """Extract multiple person names from a comparison query like
    'Compare the research areas of Duncan Brown and Alexander Nitz'."""
    q = (question or "").strip()
    if not q:
        return []
    splitters = re.compile(
        r"\b(?:and|vs\.?|versus|compared\s+(?:to|with)|with)\b", re.IGNORECASE)
    from rag_engine import _extract_person_name
    people = []
    parts = splitters.split(q)
    for part in parts:
        name = strip_possessive(_extract_person_name(part.strip()) or "")
        if name and len(name) > 3 and looks_like_person_candidate(name):
            if not any(norm_text(name) == norm_text(p) for p in people):
                people.append(name)
    return people

def _doc_to_source_md(d) -> str:
    meta = d.metadata or {}
    authors = re.sub(r"</?[a-zA-Z][^>]*>", "", str(meta.get("authors", "") or "")).strip()
    title = normalize_title_case(str(meta.get("title", "") or ""))
    year = str(meta.get("year", meta.get("publication_date", "")) or "").strip()
    year_match = re.search(r"\b(19|20)\d{2}\b", year)
    year_short = year_match.group(0) if year_match else ""
    doi = str(meta.get("doi", "") or "").strip()
 
    if not title or title.lower() == "untitled":
        title = "[Untitled]"
 
    parts = [f"{authors}." if authors else "[Unknown author].",
             f"({year_short})." if year_short else "(n.d.).",
             f"*{title}*."]
    if doi:
        if doi.startswith("10."): doi = f"https://doi.org/{doi}"
        if doi.startswith("http"): parts.append(doi)
    return " ".join(parts)
 
def _doc_to_ref(d) -> Dict[str, str]:
    meta = getattr(d, "metadata", {}) or {}
    return {
        "paper_id": str(meta.get("paper_id", "")),
        "chunk": str(meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))),
        "title": str(meta.get("title", "")),
    }
 
def _filter_noisy_docs(docs, question: str, *, person_name="",
                       anchor_value="", anchor_confidence=0.0,
                       summary_intent=False) -> list:
    deduped = dedupe_docs(docs)
    if not deduped:
        return []
    tokens = query_tokens_for_relevance(question)
    limit = int(PIPELINE_CFG["max_docs_after_filter"])
    if not tokens:
        return deduped[:limit]
    n_tok = max(1, len(tokens))
    p_sig = float(1.0 if person_name and looks_like_person_candidate(person_name)
                  else 0.7 if has_explicit_entity_signal(question) else 0.0)
    a_sig = max(0.0, min(1.0, float(anchor_confidence)))
    w_t = float(getattr(settings, "rerank_w_token", 1.0))
    w_p = float(getattr(settings, "rerank_w_person", 3.0))
    if summary_intent and person_name:
        w_p = max(w_p, 5.0)
    w_a = float(getattr(settings, "rerank_w_anchor", 2.0))
    w_c = float(getattr(settings, "rerank_w_chunk", 0.5))
    sp  = float(getattr(settings, "rerank_surname_penalty", 0.6))
    scored = []
    for d in deduped:
        hay = doc_haystack(d)
        meta = getattr(d, "metadata", {}) or {}
        tf = sum(1 for t in tokens if token_in_hay(t, hay)) / n_tok
        ps = 0.0
        if person_name:
            rp = _doc_person_match_score(d, person_name)
            ps = (rp / 4.0) * (sp if 0 < rp < 1.0 else 1.0)
        am = 1.0 if anchor_value and a_sig > 0 and anchor_in_text(anchor_value, hay) else 0.0
        ct = str(meta.get("chunk_type", "")).lower()
        cb = 1.0 if ct == "title_abstract" else (0.3 if ct == "keywords" else 0.0)
        s = tf * w_t + ps * w_p * p_sig + am * w_a * a_sig + cb * w_c
        scored.append((d, s))
    scored.sort(key=lambda x: -x[1])
    kept = [d for d, _ in scored[:limit]]
    return kept if kept else deduped[:limit]
 
def _normalize_meta_value(value) -> str:
    return norm_text(str(value or ""))
 
def _metadata_key_allowed(key: str) -> bool:
    k = (key or "").strip().lower()
    if not k or k in {"chunk", "chunk_id", "id", "doc_id", "source", "path", "url"}:
        return False
    return not (k.endswith("_id") and k != "paper_id")
 
def _metadata_value_allowed(value: str) -> bool:
    v = _normalize_meta_value(value)
    if not v or len(v) < 3 or is_placeholder_anchor_value(v):
        return False
    return not (re.fullmatch(r"[0-9\W]+", v) or re.fullmatch(r"(19|20)\d{2}", v))
 
def _iter_doc_metadata_key_values(d) -> List[Tuple[str, str, str]]:
    meta = getattr(d, "metadata", {}) or {}
    out = []
    for key, raw in meta.items():
        k = str(key or "").strip()
        if not k or not _metadata_key_allowed(k) or isinstance(raw, (list, dict, tuple, set)):
            continue
        raw_text = re.sub(r"\s+", " ", str(raw or "").strip())
        v = _normalize_meta_value(raw_text)
        if raw_text and _metadata_value_allowed(raw_text):
            out.append((k, v, raw_text))
    return out
 
def _person_name_signatures(name: str) -> Dict[str, str]:
    toks = tokenize_name(name)
    if len(toks) < 2:
        return {}
    first, last = toks[0], toks[-1]
    sig: Dict[str, str] = {
        "first": first.lower(),
        "last": last.lower(),
        "last_name": last,
        "variants": generate_name_variants(toks),
    }
    middles = toks[1:-1]
    if middles:
        sig["middle_tokens"] = [m.lower() for m in middles]
    return sig

def _name_match_strength(text: str, sig: Dict[str, str]) -> int:
    """Score how well *text* matches the person described by *sig*.

    Returns 3 (full/first-name match), 2 (initial match), 1 (last-name
    only), or 0 (no match).  Middle names are handled dynamically via a
    regex gap that allows any number of middle tokens / initials.
    """
    if not text or not sig:
        return 0
    first = str(sig.get("first", "") or "").strip()
    last = str(sig.get("last", "") or "").strip()
    if not first or not last:
        return 0
    hay = re.sub(r"[,\(\)\[\]]+", " ", _normalize_meta_value(text))
    hay = re.sub(r"\s+", " ", hay).strip()
    if not hay:
        return 0

    _MID_GAP = r"(?:\s+[a-z]\.?\s*)*\s+"

    if (re.search(rf"\b{re.escape(first)}{_MID_GAP}{re.escape(last)}\b", hay)
        or re.search(rf"\b{re.escape(last)}{_MID_GAP}{re.escape(first)}\b", hay)):
        return 3
    fi = first[0]
    for v in (sig.get("variants") or []):
        vl = v.lower()
        if vl.startswith(first) and vl in hay:
            return 3

    if (re.search(rf"\b{re.escape(fi)}\.?{_MID_GAP}{re.escape(last)}\b", hay)
        or re.search(rf"\b{re.escape(last)}{_MID_GAP}{re.escape(fi)}\.?\b", hay)):
        return 2
    for v in (sig.get("variants") or []):
        vl = v.lower()
        if vl.startswith(fi) and not vl.startswith(first) and vl in hay:
            return 2

    if re.search(rf"\b{re.escape(last)}\b", hay):
        return 1
    return 0

def _doc_person_match_score(d, person_name: str) -> float:
    sig = _person_name_signatures(person_name)
    if not sig:
        return 0.0
    meta = getattr(d, "metadata", {}) or {}
    researcher_text = str(meta.get("researcher", "") or "")
    authors_text = str(meta.get("authors", "") or "")
    researcher_strength = _name_match_strength(researcher_text, sig)
    author_strength = 0
    for author in split_author_names(authors_text):
        author_strength = max(author_strength, _name_match_strength(author, sig))
    if author_strength == 0 and authors_text:
        author_strength = _name_match_strength(authors_text, sig)

    content_strength = 0
    page_content = str(getattr(d, "page_content", "") or "")
    if page_content:
        content_strength = _name_match_strength(page_content[:2000], sig)

    if researcher_strength >= 3:
        return 4.0
    if author_strength >= 3:
        return 3.0
    if researcher_strength == 2:
        return 2.0
    if author_strength == 2:
        return 1.5
    if content_strength >= 3:
        return 1.2
    if content_strength == 2:
        return 1.0
    if researcher_strength == 1:
        return 0.3
    if author_strength == 1:
        return 0.2
    if content_strength == 1:
        return 0.15
    return 0.0
 
def _rank_docs_for_person(docs, person_name: str) -> List[Tuple[Any, float]]:
    ranked: List[Tuple[Any, float]] = []
    for d in docs or []:
        ranked.append((d, _doc_person_match_score(d, person_name)))
 
    def _sort_key(item: Tuple[Any, float]) -> Tuple[float, str, str]:
        doc, score = item
        meta = getattr(doc, "metadata", {}) or {}
        paper_id = str(meta.get("paper_id", "") or "")
        chunk = str(meta.get("chunk", meta.get("chunk_id", meta.get("id", ""))) or "")
        return (-float(score), paper_id, chunk)
 
    ranked.sort(key=_sort_key)
    return ranked
 
def _select_docs_for_person(ranked_docs: List[Tuple[Any, float]], *,
                            max_docs: int) -> Tuple[List[Any], Dict[str, int]]:
    strong_docs = [d for d, score in ranked_docs if score >= 2.0]
    matched_docs = [d for d, score in ranked_docs if score >= 1.0]
    weak_docs = [d for d, score in ranked_docs if score < 1.0]
    limit = max(4, int(max_docs or 0))
 
    if strong_docs:
        keep = strong_docs[:limit]
        if len(keep) < limit:
            keep.extend(matched_docs[:max(0, limit - len(keep))])
        if len(keep) < limit:
            keep.extend(weak_docs[:max(0, limit - len(keep))])
    elif matched_docs:
        keep = matched_docs[:limit]
        if len(keep) < limit:
            keep.extend(weak_docs[:max(0, limit - len(keep))])
    else:
        keep = [d for d, _ in ranked_docs[:limit]]
 
    return dedupe_docs(keep), {
        "strong": len(strong_docs),
        "matched": len(matched_docs),
        "total": len(ranked_docs),
    }
 
_CONFIDENCE_ORDER = ("weak", "low", "medium", "high")
 
def _downgrade_confidence(label: str, *, steps: int = 1) -> str:
    key = str(label or "").strip().lower()
    if key not in _CONFIDENCE_ORDER:
        return key or "weak"
    idx = _CONFIDENCE_ORDER.index(key)
    return _CONFIDENCE_ORDER[max(0, idx - max(1, int(steps)))]
 
def _downshift_confidence_for_person_support(label: str, *,
                                             strong_count: int, matched_count: int,
                                             total_count: int) -> str:
    if total_count <= 0:
        return label
    if matched_count <= 0:
        return "weak"
 
    out = str(label or "weak").strip().lower()
    strong_ratio = float(strong_count) / max(1, int(total_count))
    matched_ratio = float(matched_count) / max(1, int(total_count))

    has_strong_absolute = matched_count >= 4
    has_some_strong = strong_count >= 2

    if out == "high" and not has_strong_absolute and (strong_ratio < 0.3 or matched_ratio < 0.5):
        out = "medium"
    if out in {"high", "medium"} and not has_some_strong and strong_ratio < 0.1 and matched_ratio < 0.3:
        out = _downgrade_confidence(out, steps=1)
    if out in {"medium", "low"} and matched_count < 2 and matched_ratio < 0.15:
        out = _downgrade_confidence(out, steps=1)
    return out
 
def _dominant_metadata_filter_from_docs(docs, question: str, *,
                                        majority_ratio: float = 0.6, min_count: int = 3) -> Dict[str, Any]:
    result = {
        "dominant": False, "key": "", "value": "", "count": 0, "ratio": 0.0,
        "confidence": 0.0, "confidence_floor": float(getattr(settings, "dominant_min_confidence", 0.72)),
        "n_docs": len(docs or []), "runner_up_count": 0, "filter": {},
    }
    if not docs:
        return result
    n_docs = len(docs)
    counts: Dict[Tuple[str, str], int] = {}
    exemplars: Dict[Tuple[str, str], str] = {}
    for d in docs:
        seen_pairs: set = set()
        for key, norm_val, raw_val in _iter_doc_metadata_key_values(d):
            if key.lower() == "authors":
                for name in split_author_names(raw_val):
                    nv = _normalize_meta_value(name)
                    if not _metadata_value_allowed(name):
                        continue
                    pair = (key, nv)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        counts[pair] = counts.get(pair, 0) + 1
                        exemplars.setdefault(pair, name)
            else:
                pair = (key, norm_val)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    counts[pair] = counts.get(pair, 0) + 1
                    exemplars.setdefault(pair, raw_val)
    if not counts:
        return result
 
    def _sort_key(item):
        (k, nv), cnt = item
        ALLOWED_FILTER_KEYS = {"researcher"}
        return (-cnt,
                0 if k.lower() in ALLOWED_FILTER_KEYS else 2,
                0 if k.lower() == "researcher" else 1,
                k, nv)
 
    ranked = sorted(counts.items(), key=_sort_key)
    (best_key, best_nv), best_count = ranked[0]
    runner_up = next((cnt for (_, nv), cnt in ranked[1:] if nv != best_nv), 0)
    ratio = float(best_count) / max(1, n_docs)
    margin = float(best_count - runner_up) / max(1, n_docs)
    confidence = max(0.0, min(1.0, 0.85 * ratio + 0.15 * max(0.0, margin)))
    if ratio >= 0.8 and best_count >= max(4, min_count):
        confidence = max(confidence, min(0.98, ratio))
 
    result.update({
        "key": best_key, "value": exemplars.get((best_key, best_nv), best_nv),
        "count": int(best_count), "ratio": ratio, "confidence": confidence,
        "runner_up_count": int(runner_up),
    })
    conf_floor = float(getattr(settings, "dominant_min_confidence", 0.72))
    result["confidence_floor"] = conf_floor
 
    _DOMINANT_ALLOWED_KEYS = {"researcher"}
    if (best_count >= max(1, min_count) and ratio >= float(majority_ratio)
        and confidence >= conf_floor
        and best_key.lower() in _DOMINANT_ALLOWED_KEYS
        and not is_placeholder_anchor_value(str(result.get("value", "") or ""))):
        result["dominant"] = True
        result["filter"] = {"key": result["key"], "value": result["value"]}
    return result
 
def _insufficient_context_answer(question: str, intent: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    if q:
        return (f'I couldn\'t find enough matching evidence for "{q}" in the retrieved papers. '
                "Please clarify the entity, topic, or time range you want.")
    return "I couldn't find enough matching evidence. Please clarify the entity or topic you want."
 
def _uncertain_retrieval_answer(question: str, *, anchor_value: str = "", reason: str = "") -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    anchor = re.sub(r"\s+", " ", (anchor_value or "").strip())
    if anchor and reason.strip().lower() == "inconsistent":
        return (f'I\'m not confident the retrieved evidence matches the current focus on "{anchor}". '
                f'Could you confirm whether you want to continue with "{anchor}" or switch topics for "{q}"?')
    if q:
        return (f'I\'m not confident there is enough consistent evidence yet for "{q}". '
                "Please add a specific entity, paper title, or year so I can answer accurately.")
    return "I'm not confident there is enough consistent evidence yet. Please add a specific entity, paper title, or year."
 
_BLOCKED_MARKERS = (
    "detected intent:", "retrieval count:", "pipeline_json", "chroma_retrieval",
    "metadata filter:", "session_id:", "turn_count:",
    "please provide the answer in the requested format",
    "note: the user-facing answer", "the final answer is",
    "prompt_max_docs", "prompt_doc_text_limit", "retrieval_confidence",
)
 
_BOILERPLATE_PHRASES = (
    "no further analysis is required", "no additional retrieval is required",
    "no further synthesis is required", "no additional analysis is required",
    "no further retrieval is required",
    "please let me know if you would like me to revise",
    "i have revised the answer", "i've revised the response",
    "i made some minor changes", "let me know if you have any further requests",
    "please let me know if you need any further assistance",
    "please let me know if you need anything else",
    "if you need any further assistance", "i can help with anything else",
    "let me know if you need anything else", "please provide the next question",
    "i am ready to help", "i'm ready to help", "i am ready when you are",
    "the original question was not provided", "i will follow the provided style rules",
    "session diagnostics", "llm calls this turn",
    "note: the answer has been reformatted", "has been reformatted to",
    "to better match the required response format",
    "to match the required response format", "reformatted to better match",
    "however, i noticed that the response contains",
    "i will revise the response to only include",
    "to only include information present in the retrieved",
    "let me know if you'd like me to proceed",
    "let me know if you would like me to proceed",
    "i can help with the next question",
    "i'm here to assist you", "i'm here to help you", "best regards",
    "remove any extraneous text",
    "leave any intermediate reasoning out",
    "leave out any extraneous text",
    "leave out citations, links, and extraneous details",
    "provide a clear and concise description",
    "provide a single clear statement",
    "provide a clear, concise description",
    "provide a clear description",
    "cite relevant papers as evidence",
    "include a least one paragraph describing",
    "include a brief description of the methodology",
    "focus on describing the main ideas",
    "answer must be written in third-person voice",
    "leave the formatting intact",
    "retrieved sources",
)
 
_SELF_REF_TERMS = ("reformatted", "required response format", "response format",
                    "answer format", "the answer has been", "this answer has been")
 
def _normalize_for_similarity(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", re.sub(r"\s+", " ", str(text or "").strip().lower()))
 
def _strip_leading_answer_labels(text: str) -> str:
    cleaned = str(text or "").strip()
    for _ in range(3):
        updated = re.sub(r"^\s*(summary|final summary|answer|response|final response|the final answer is)\s*:\s*",
                          "", cleaned, flags=re.IGNORECASE).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned
 
def _is_closure_or_process(text: str) -> bool:
    lower = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if lower.startswith("note:") and any(t in lower for t in (
        "pipeline_json", "retrieval_count", "chroma_retrieval",
        "prompt_max_docs", "prompt_doc_text_limit",
        "metadata_filter", "cache_version",
        "reformatted", "revised", "reformat",
        "response format", "required format", "requested format")):
        return True
    if re.match(r"^note:\s.*(format|revised|reformatted)", lower):
        return True
    if re.match(r"^(also,?\s*)?(i can help|let me know|best regards|please let me know)", lower):
        return True
    if re.match(r"^(however,?\s*)?i noticed that the response", lower):
        return True
    if re.match(r"^i will revise the response", lower):
        return True
    return any(m in lower for m in (
        "please let me know if you need", "if you need any further assistance",
        "i can help with anything else", "let me know if you need anything else",
        "let me know if you have any further requests", "i have revised the answer",
        "i've revised the response", "i made some minor changes",
        "please provide the next question", "i am ready to help",
        "i'm ready to help", "i am ready when you are",
        "let me know if you would like me to proceed",
        "let me know if you'd like me to proceed",
        "i'm here to assist you", "i'm here to help",
        "best regards", "[assistant]",
        "i can help with the next question",
        "i will revise the response to only include",
        "i noticed that the response contains some extraneous",
        "to only include information present in the retrieved",
    ))
 
def _sanitize_user_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
 
    kept = [line for line in raw.splitlines()
            if not any(m in line.strip().lower() for m in _BLOCKED_MARKERS)
            and not any(p in line.strip().lower() for p in _BOILERPLATE_PHRASES)]
    cleaned = _strip_leading_answer_labels("\n".join(kept).strip()) or _strip_leading_answer_labels(raw)
    if not cleaned:
        return raw
 
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", cleaned) if c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in cleaned.splitlines() if c.strip()]
 
    result, seen_norms = [], []
    for chunk in chunks:
        para = _strip_leading_answer_labels(re.sub(r"\s+", " ", chunk).strip())
        if not para:
            continue
        lower = para.lower()
        if lower.startswith("confidence level"):
            continue
        if _is_closure_or_process(para):
            continue
        norm = _normalize_for_similarity(para)
        if not norm or any(norm == p or SequenceMatcher(None, norm, p).ratio() >= 0.80 for p in seen_norms):
            continue
        seen_norms.append(norm)
        result.append(para)
 
    text_out = "\n\n".join(result).strip()
    if text_out:
        kept_sents, last_norms = [], []
        for p in re.split(r"(?<=[.!?])\s+", text_out):
            sent = p.strip()
            if not sent:
                continue
            n = _normalize_for_similarity(sent)
            if n and n in last_norms:
                continue
            kept_sents.append(sent)
            last_norms = (last_norms + [n])[-6:]
        text_out = " ".join(kept_sents).strip()
    if text_out:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text_out) if p.strip()]
        text_out = " ".join(p for p in parts if not _is_closure_or_process(p)).strip()
    if text_out and not re.search(r'[.!?"\')\\]]\s*$', text_out):
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text_out) if p.strip()]
        if len(parts) > 1:
            tail = parts[-1]
            if not re.match(r"^[-*]|\d+\.", tail) and len(re.findall(r"[A-Za-z0-9]+", tail)) <= 14:
                text_out = " ".join(parts[:-1]).strip() or text_out
    if text_out:
        text_out = "\n".join(ln for ln in text_out.splitlines()
                             if not (ln.strip().lower().startswith("note:")
                                     and any(t in ln.strip().lower() for t in _SELF_REF_TERMS))).strip()
    return text_out or _strip_leading_answer_labels(cleaned) or cleaned
 
def sanitize_answer_for_display(text: str) -> str:
    cleaned = _sanitize_user_answer(text)
    cleaned = re.sub(r"</?[a-zA-Z][^>]*>", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
 
    if "\n\n" in cleaned:
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()
 
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if len(sentences) <= 3:
        return cleaned.strip()
 
    _NEW_BLOCK = re.compile(
        r"^([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:'s)?\s+"
        r"|Additionally[,\s]|Furthermore[,\s]|Moreover[,\s]|Another\s+researcher"
        r"|In\s+addition[,\s]|Separately[,\s]|Meanwhile[,\s]|In\s+contrast[,\s]"
        r"|His\s+research\s+also|Her\s+research\s+also|Their\s+research\s+also"
        r"|While\s+(?:his|her|their|none|not)|Although\s)", re.IGNORECASE)
 
    paragraphs, current = [], []
    for sent in sentences:
        if current and len(current) >= 2 and _NEW_BLOCK.match(sent):
            paragraphs.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
            if len(current) >= 4:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(paragraphs)).strip()
 
def _build_anchor_from_dominance(dominance) -> Dict[str, Any]:
    if not dominance.get("dominant"):
        return {}
    key = re.sub(r"\s+", " ", str(dominance.get("key", "") or "").strip().lower())
    value = re.sub(r"\s+", " ", str(dominance.get("value", "") or "").strip())
    if not key or not value or is_placeholder_anchor_value(value):
        return {}
    confidence = float(dominance.get("confidence", 0.0) or 0.0)
    if confidence < float(getattr(settings, "dominant_min_confidence", 0.72)):
        return {}
    return {"type": key, "value": value, "source": f"dominant_metadata:{key}",
            "confidence": max(0.0, min(1.0, confidence))}
 
def _choose_anchor_update(*, current_anchor, candidate_anchor, dominance, question,
                          resolved_question=""):
    anchor_now = normalize_anchor(current_anchor)
    candidate = normalize_anchor(candidate_anchor)
 
    if anchor_now and not candidate:
        anchor_value = str(anchor_now.get("value", "") or "").strip()
        if anchor_value and not anchor_query_overlap(anchor_value, question):
            if is_followup_coref_question(question):
                return anchor_now, "kept_followup_coref"
            return {}, "cleared_no_query_overlap"
        return anchor_now, "kept_no_dominance"
 
    if not candidate:
        return {}, "none"
    if not anchor_now:
        candidate_value = str(candidate.get("value", "") or "").strip()
        if candidate_value:
            has_raw_overlap = anchor_query_overlap(candidate_value, question)
            has_resolved_overlap = (bool(resolved_question)
                                    and anchor_query_overlap(candidate_value, resolved_question))
            if not has_raw_overlap and not has_resolved_overlap:
                return {}, "blocked_no_query_overlap"
        return candidate, "set_from_dominance"
 
    same_type = _normalize_meta_value(anchor_now.get("type", "")) == _normalize_meta_value(candidate.get("type", ""))
    same_value = _normalize_meta_value(anchor_now.get("value", "")) == _normalize_meta_value(candidate.get("value", ""))
    if same_type and same_value:
        anchor_now["confidence"] = max(float(anchor_now.get("confidence", 0) or 0),
                                       float(candidate.get("confidence", 0) or 0))
        anchor_now["source"] = candidate.get("source", anchor_now.get("source", "retrieval"))
        return anchor_now, "kept_reinforced"
 
    candidate_value = str(candidate.get("value", "") or "").strip()
    if candidate_value:
        has_raw_overlap = anchor_query_overlap(candidate_value, question)
        has_resolved_overlap = (bool(resolved_question)
                                and anchor_query_overlap(candidate_value, resolved_question))
        if not has_raw_overlap and not has_resolved_overlap:
            if is_followup_coref_question(question):
                return anchor_now, "kept_candidate_no_query_overlap"
            return anchor_now, "kept_candidate_no_query_overlap"
 
    replace_conf = float(getattr(settings, "dominant_replace_confidence", 0.82))
    strong_ratio = float(dominance.get("ratio", 0) or 0) >= max(
        replace_conf, float(getattr(settings, "dominant_majority_ratio", 0.6)) + 0.15)
    if float(candidate.get("confidence", 0) or 0) >= replace_conf and (
        has_explicit_entity_signal(question) or has_explicit_entity_signal(resolved_question or "")
        or strong_ratio):
        return candidate, "replaced_with_strong_evidence"
    return anchor_now, "kept_ambiguous_switch_requires_confirmation"
 
def _extract_answer_text(raw_answer) -> str:
    raw_text = str(raw_answer or "")
    answer_text = strip_prompt_leak(raw_text).strip()
    if answer_text:
        return answer_text
    for marker in ("ANSWER:", "FINAL RESPONSE:", "RESPONSE:"):
        if marker in raw_text:
            cand = raw_text.split(marker)[-1].strip()
            if cand:
                return cand
    return raw_text.strip()
 
def _runtime_prompt_token_budget(runtime, reserved_new_tokens: int) -> int:
    if runtime is None:
        return 0
    max_ctx = 0
    for source in [getattr(getattr(runtime, "model", None), "config", None),
                   getattr(runtime, "tokenizer", None)]:
        if source and max_ctx <= 0:
            try: max_ctx = int(getattr(source, "max_position_embeddings", 0) or
                               getattr(source, "model_max_length", 0) or 0)
            except Exception: pass
    if max_ctx <= 0 or max_ctx > 131072:
        max_ctx = 8192
    return max(256, max_ctx - max(64, int(reserved_new_tokens) + 24))
 
def _compose_answer_prompt(*, base_sections, style_hint, question_for_answer, context_blob):
    system_parts = list(base_sections)
    system_parts.append("ANSWER POLICY:\n" + style_hint)
    system_content = "\n\n".join(system_parts)

    user_parts = []
    ctx = (context_blob or "").strip()
    if ctx:
        user_parts.append("RETRIEVED CONTEXT:\n" + ctx)
    user_parts.append(PIPELINE_CFG["prompt_mid"].lstrip("\n")
                      + (question_for_answer or "") + PIPELINE_CFG["prompt_suffix"])
    user_content = "\n\n".join(user_parts)

    try:
        from rag_engine import _DirectGenerationLLM
        sep = _DirectGenerationLLM.SYSTEM_USER_SEP
    except ImportError:
        sep = "\n\n"
    return system_content + sep + user_content
 
def _fit_prompt_to_budget(*, runtime, docs, base_sections, style_hint,
                          question_for_answer, max_docs, text_limit,
                          min_docs=1, min_text_limit=120):
    docs_cap = max(1, int(max_docs))
    text_cap = max(96, int(text_limit))
    min_docs = max(1, int(min_docs))
    min_text_limit = max(96, int(min_text_limit))
    budget = _runtime_prompt_token_budget(
        runtime, int(getattr(settings, "answer_max_new_tokens", 384)))
 
    if not docs:
        return _compose_answer_prompt(base_sections=base_sections, style_hint=style_hint,
                                      question_for_answer=question_for_answer, context_blob=""), docs_cap, text_cap
 
    prompt = _compose_answer_prompt(
        base_sections=base_sections, style_hint=style_hint,
        question_for_answer=question_for_answer,
        context_blob=build_compact_context(docs, max_docs=docs_cap, text_limit=text_cap))
 
    phase = 1
    for _ in range(36):
        if budget <= 0 or runtime is None:
            return prompt, docs_cap, text_cap
        try:
            if int(runtime.count_tokens(prompt)) <= budget:
                return prompt, docs_cap, text_cap
        except Exception:
            return prompt, docs_cap, text_cap
 
        prev = (docs_cap, text_cap)
        if phase == 1:
            if text_cap > min_text_limit:
                text_cap = max(min_text_limit, text_cap - max(20, text_cap // 6))
            else:
                phase = 2
        if phase == 2:
            if docs_cap > min_docs and text_cap <= min_text_limit:
                docs_cap = max(min_docs, docs_cap - max(1, docs_cap // 5))
            elif text_cap > min_text_limit:
                text_cap = max(min_text_limit, text_cap - max(16, text_cap // 6))
            elif docs_cap > min_docs:
                docs_cap = max(min_docs, docs_cap - 1)
            elif docs_cap > 1:
                docs_cap -= 1
            elif text_cap > 96:
                text_cap = max(96, text_cap - 16)
            else:
                return prompt, docs_cap, text_cap
 
        if (docs_cap, text_cap) == prev:
            return prompt, docs_cap, text_cap
        prompt = _compose_answer_prompt(
            base_sections=base_sections, style_hint=style_hint,
            question_for_answer=question_for_answer,
            context_blob=build_compact_context(docs, max_docs=docs_cap, text_limit=text_cap))
    return prompt, docs_cap, text_cap
 
def _clip_sentences(text: str, *, max_sentences: int = 2, max_chars: int = 320) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if not compact:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", compact) if p.strip()]
    if parts:
        compact = " ".join(parts[:max(1, max_sentences)]).strip()
    return compact[:max_chars].rstrip() + "..." if len(compact) > max_chars else compact
 
def _clean_assistant_turn_for_prompt(text: str) -> str:
    blocked = ("summary:", "no further analysis is required", "no additional retrieval is required",
               "no further synthesis is required", "the retrieved context", "confidence level",
               "please let me know if you need", "if you need any further assistance",
               "i can help with anything else", "let me know if you need anything else",
               "i have revised the answer", "i've revised the response")
    kept = [line for line in str(text or "").splitlines()
            if not any(b in re.sub(r"\s+", " ", line or "").strip().lower() for b in blocked)]
    return _clip_sentences(_strip_leading_answer_labels(" ".join(kept).strip()),
                           max_sentences=2, max_chars=360)
 
def _rolling_summary_for_prompt(summary_text: str) -> str:
    raw = str(summary_text or "").strip()
    if not raw:
        return ""
    aliases = {"focus": "Current focus", "current focus": "Current focus",
               "researcher mentions": "Researcher mentions", "researchers": "Researcher mentions",
               "entities": "Core entities", "entities discussed": "Core entities",
               "core entities": "Core entities", "findings": "Key themes",
               "key findings so far": "Key themes", "key themes": "Key themes",
               "constraints": "Constraints", "open questions": "Open questions"}
    sections: Dict[str, List[str]] = {k: [] for k in set(aliases.values())}
    current_key = None
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m:
            current_key = aliases.get((m.group(1) or "").strip().lower())
            if current_key:
                rest = (m.group(2) or "").strip().lstrip("- ").strip()
                if rest:
                    sections[current_key].append(rest)
            continue
        if current_key:
            sections[current_key].append(line.lstrip("- ").strip())
 
    has_boilerplate = any(m in raw.lower() for m in (
        "summary:", "no further analysis is required", "no additional retrieval is required",
        "no further synthesis is required", "the retrieved context", "confidence level"))
    if has_boilerplate:
        keys = ["Current focus", "Core entities"]
    else:
        return raw
    lines = []
    for key in keys:
        vals = [re.sub(r"\s+", " ", str(v or "").strip()) for v in (sections.get(key) or [])
                if str(v or "").strip() and str(v or "").strip() != "(none)"]
        if vals:
            lines.append(f"{key}: {' | '.join(vals)}")
    return "\n".join(lines).strip()
 
def _build_recent_turns_context(state, max_turns: int) -> str:
    turns = list(state.get("recent_turns") or state.get("turns", []) or [])
    if max_turns <= 0 or not turns:
        return ""
    rows = []
    for t in turns[-max_turns:]:
        role = str(t.get("role", "") or "").strip().lower()
        text = str(t.get("text", "") or "").strip()
        if not role or not text:
            continue
        text = (_clean_assistant_turn_for_prompt(text) if role == "assistant"
                else _clip_sentences(text, max_sentences=2, max_chars=420))
        if text:
            rows.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
    return "\n".join(rows).strip()
 
def _fallback_answer_from_docs(question: str, docs, intent: str = "default") -> Optional[str]:
    if not docs:
        return None
    top = list(docs[:12])
    years = []
    for d in top:
        m = re.search(r"\b(19|20)\d{2}\b", str((getattr(d, "metadata", {}) or {}).get(
            "year", getattr(d, "metadata", {}).get("publication_date", "")) or ""))
        if m:
            try: years.append(int(m.group(0)))
            except Exception: pass
 
    works = []
    for d in top[:4]:
        title = str((getattr(d, "metadata", {}) or {}).get("title", "") or "").strip()
        if not title or title.lower() == "untitled":
            continue
        year = str((getattr(d, "metadata", {}) or {}).get("year", "") or "").strip()
        works.append(f"{title} ({year})" if year else title)
 
    leads = {"comparison": "I couldn't generate a full comparison, but these retrieved papers are most relevant to compare.",
             "list": "I couldn't generate a full list answer, but these retrieved papers are most relevant."}
    lead = leads.get(intent, "I couldn't generate a complete narrative answer, but these retrieved papers are the strongest evidence.")
 
    lines = [lead]
    if years:
        lo, hi = min(years), max(years)
        lines.append(f"Publication years represented: {lo}" + (f" to {hi}" if lo != hi else "") + ".")
    if works:
        lines.append("Most relevant papers: " + "; ".join(works) + ".")
    return " ".join(lines).strip() if len(lines) > 1 else None
 
def _clean_title_for_answer(title: str) -> str:
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", str(title or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text or text.lower() == "untitled":
        return ""
    return normalize_title_case(text)
 
def _supported_researcher_evidence(docs, *, max_researchers: int = 6,
                                   max_titles_per_researcher: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        researcher = re.sub(r"\s+", " ", str(meta.get("researcher", "") or "").strip())
        if not researcher or not looks_like_person_candidate(researcher):
            continue
        row = evidence.setdefault(researcher, {"count": 0, "titles": []})
        row["count"] += 1
        title = _clean_title_for_answer(meta.get("title", ""))
        if title and title not in row["titles"]:
            row["titles"].append(title)
    ranked = sorted(evidence.items(), key=lambda item: (-int(item[1].get("count", 0)), item[0]))
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, payload in ranked[:max(1, max_researchers)]:
        out.append((name, {
            "count": int(payload.get("count", 0) or 0),
            "titles": list(payload.get("titles", [])[:max(1, max_titles_per_researcher)]),
        }))
    return out
 
def _extract_person_like_spans(text: str, *, max_items: int = 24) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    pattern = re.compile(r"\b(?:[A-Z][A-Za-z'\-]*\.?\s+){1,3}[A-Z][A-Za-z'\-]+\b")
    out: List[str] = []
    seen = set()
    for span in pattern.findall(raw):
        candidate = re.sub(r"\s+", " ", span.strip())
        if not candidate or not looks_like_person_candidate(candidate):
            continue
        key = norm_text(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max(1, max_items):
            break
    return out
 
def _answer_mentions_unsupported_researcher(answer: str, docs) -> bool:
    answer_people = _extract_person_like_spans(answer)
    if not answer_people:
        return False
 
    doc_sigs: List[Dict[str, str]] = []
    doc_names: List[str] = []
    seen_names = set()
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        candidates = []
        researcher = re.sub(r"\s+", " ", str(meta.get("researcher", "") or "").strip())
        if researcher:
            candidates.append(researcher)
        authors = str(meta.get("authors", "") or "")
        candidates.extend(split_author_names(authors))
        for candidate in candidates:
            person = re.sub(r"\s+", " ", str(candidate or "").strip())
            if not person or not looks_like_person_candidate(person):
                continue
            key = norm_text(person)
            if key in seen_names:
                continue
            seen_names.add(key)
            sig = _person_name_signatures(person)
            if sig:
                doc_sigs.append(sig)
                doc_names.append(person)
 
    if not doc_sigs:
        return False
 
    for person in answer_people:
        if any(_name_match_strength(person, sig) >= 2 for sig in doc_sigs):
            continue
        answer_sig = _person_name_signatures(person)
        if answer_sig and any(_name_match_strength(doc_name, answer_sig) >= 2 for doc_name in doc_names):
            continue
        person_last = (answer_sig.get("last", "") or "").lower() if answer_sig else ""
        if person_last and any((sig.get("last", "") or "").lower() == person_last for sig in doc_sigs):
            continue
        return True
    return False
 
def _build_researcher_extract_answer(docs, *, max_researchers: int = 5) -> Optional[str]:
    evidence = _supported_researcher_evidence(docs, max_researchers=max_researchers,
                                              max_titles_per_researcher=5)
    if not evidence:
        return None
    intro = "Based on the retrieved papers, the following researchers are directly supported by the evidence:"
    paragraphs = [intro]
    for name, payload in evidence:
        titles = list(payload.get("titles", []) or [])
        count = int(payload.get("count", 0) or 0)
        if titles:
            title_text = "; ".join(f'"{t}"' for t in titles[:5])
            paragraphs.append(f"{name}: supported by {title_text}.")
        else:
            paragraphs.append(f"{name}: supported by {count} retrieved paper(s).")
    return "\n\n".join(paragraphs).strip()
 
def _collect_doc_titles(docs) -> set:
    """Collect normalized titles from retrieved documents for validation."""
    titles = set()
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        raw_title = re.sub(r"</?[a-zA-Z][^>]*>", "", str(meta.get("title", "") or "")).strip()
        raw_title = re.sub(r"\s+", " ", raw_title).strip()
        if raw_title and raw_title.lower() not in {"untitled", "unknown", "n/a"}:
            titles.add(raw_title.lower())
    return titles
 
def _strip_hallucinated_citations(answer: str, docs) -> str:
    """Remove lines containing fabricated paper titles not found in retrieved
    documents.  Only targets long quoted strings (30+ chars) that look like
    full paper titles — short quoted phrases are left alone since they may be
    legitimate emphasis or terminology.  Uses fuzzy matching to account for
    minor formatting differences between the LLM's citation and the doc title."""
    if not answer or not docs:
        return answer
 
    doc_titles = _collect_doc_titles(docs)
    if not doc_titles:
        return answer
 
    quoted_pattern = re.compile(r'"([^"]{30,200})"')
    citation_pattern = re.compile(
        r'[A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*["\u201c]([^"\u201d]{30,200})["\u201d]')
 
    lines = answer.split("\n")
    cleaned_lines = []
    for line in lines:
        has_fabricated = False
        for pattern in [quoted_pattern, citation_pattern]:
            for match in pattern.finditer(line):
                cited_title = re.sub(r"\s+", " ", match.group(1)).strip().lower()
                if not cited_title:
                    continue
                matched = False
                for doc_title in doc_titles:
                    if cited_title in doc_title or doc_title in cited_title:
                        matched = True
                        break
                    ratio = SequenceMatcher(None, cited_title, doc_title).ratio()
                    if ratio >= 0.5:
                        matched = True
                        break
                if not matched:
                    has_fabricated = True
                    break
            if has_fabricated:
                break
 
        if not has_fabricated:
            cleaned_lines.append(line)
 
    result = "\n".join(cleaned_lines).strip()
    return result if result else answer
 
def _retrieve_for_person(
    eng, person_name: str, retrieval_q: str, budget_papers: int,
    query_embedding, raw_question: str,
    *, pa_val: str = "", pa_conf: float = 0.0, summary_intent: bool = False,
    existing_docs=None,
) -> tuple:
    """Multi-stage retrieval for a named person.

    Stage 1  — exact metadata filter on the ``researcher`` field.
    Stage 1b — co-author / variant search: ``$or`` across researcher name
               variants **and** ``$contains`` on the ``authors`` field so
               that collaborators (not just PIs) are discoverable.
    Stage 2  — hybrid vector + keyword fallback.
               2a) Name-targeted vector search: uses the **person's name**
                   as the query (not the full question) so the embedding
                   matches chunks that mention the person rather than
                   biographical boilerplate.
               2b) Keyword / document-content search: uses ChromaDB's
                   ``where_document $contains`` to find any chunk whose
                   raw text includes the person's last name — a BM25-like
                   substring match that catches co-author mentions the
                   vector model might miss.
               Results from 2a and 2b are merged and post-filtered by
               ``_doc_person_match_score``.

    Returns (docs, stage, person_support)
        stage: 1=metadata hit, 2=fulltext/hybrid fallback, 0=not found
    """
    existing = list(existing_docs or [])
    name_toks = [t for t in re.findall(r"[A-Za-z]+", str(person_name or "")) if t]
    name_parts = [p.lower() for p in name_toks if len(p) > 2]
    last_name = name_toks[-1] if name_toks else ""

    def _noisy_filter(docs):
        return _filter_noisy_docs(
            docs, retrieval_q,
            person_name=person_name,
            anchor_value=pa_val,
            anchor_confidence=pa_conf,
            summary_intent=summary_intent,
        )

    def _rank_and_select(merged):
        ranked = _rank_docs_for_person(merged, person_name)
        return _select_docs_for_person(
            ranked,
            max_docs=min(max(6, budget_papers or 12), 12),
        )

    stage1_docs = eng.retrieve_papers(
        retrieval_q, budget_papers,
        query_embedding=None,
        where_filter={"researcher": person_name},
        raw_question=raw_question,
    )
    if stage1_docs:
        filtered = _noisy_filter(stage1_docs)
        merged = dedupe_docs(existing + filtered)
        selected, support = _rank_and_select(merged)
        if support["matched"] > 0:
            matched_count = support["matched"]
            print(f"[RETRIEVE] Stage1 metadata hit for '{person_name}': {matched_count} matched")
            return selected, 1, support

    if hasattr(eng, "retrieve_papers_by_author") and len(name_toks) >= 2:
        print(f"[RETRIEVE] Stage1 miss — trying co-author search for '{person_name}'")
        stage1b_docs = eng.retrieve_papers_by_author(
            retrieval_q, person_name, budget_papers,
            query_embedding=query_embedding,
        )
        if stage1b_docs:
            filtered = _noisy_filter(stage1b_docs)
            merged = dedupe_docs(existing + filtered)
            selected, support = _rank_and_select(merged)
            if support["matched"] > 0:
                matched_count = support["matched"]
                print(f"[RETRIEVE] Stage1b co-author hit for '{person_name}': "
                      f"{matched_count} matched from {len(stage1b_docs)} docs")
                return selected, 1, support

    if not bool(getattr(settings, "fulltext_fallback_enable", True)):
        return list(existing), 0, {"strong": 0, "matched": 0, "total": 0}

    print(f"[RETRIEVE] Stage1/1b miss for '{person_name}' — falling back to hybrid search")

    name_query = person_name
    vector_docs = eng.retrieve_papers(
        name_query, budget_papers,
        query_embedding=query_embedding,
        where_filter=None,
        raw_question=raw_question,
    )

    if retrieval_q.lower().strip() != name_query.lower().strip():
        topic_docs = eng.retrieve_papers(
            retrieval_q, budget_papers,
            query_embedding=query_embedding,
            where_filter=None,
            raw_question=raw_question,
        )
        vector_docs = dedupe_docs(vector_docs + topic_docs)

    keyword_docs = []
    if hasattr(eng, "keyword_search_papers") and last_name and len(last_name) > 2:
        kw_variants = generate_name_variants(name_toks)
        if last_name.lower() not in {v.lower() for v in kw_variants}:
            kw_variants.append(last_name)
        for kw in kw_variants:
            keyword_docs = eng.keyword_search_papers(
                [kw], budget_papers,
                query_embedding=query_embedding,
            )
            if keyword_docs:
                break

    all_docs = dedupe_docs(vector_docs + keyword_docs)

    name_matched = [
        d for d in all_docs
        if _doc_person_match_score(d, person_name) >= 1.0
    ]
    if not name_matched:
        name_matched = [
            d for d in all_docs
            if any(part in doc_haystack(d) for part in name_parts)
        ]
    if name_matched:
        filtered = _noisy_filter(name_matched)
        merged = dedupe_docs(existing + filtered)
        selected, support = _rank_and_select(merged)
        if support["matched"] > 0:
            matched_count = support["matched"]
            print(f"[RETRIEVE] Stage2 hybrid hit for '{person_name}': "
                  f"{matched_count} matched from {len(name_matched)} name-filtered docs "
                  f"(vector={len(vector_docs)}, keyword={len(keyword_docs)})")
            return selected, 2, support

    if all_docs:
        merged = dedupe_docs(existing + _noisy_filter(all_docs))
        selected, support = _rank_and_select(merged)
        print(f"[RETRIEVE] Stage2 no name match for '{person_name}' — "
              f"returning {len(selected)} unfiltered vector docs (conf=weak)")
        return selected, 2, {"strong": 0, "matched": 0, "total": len(selected)}

    return list(existing), 0, {"strong": 0, "matched": 0, "total": 0}

def answer_question(question: str, user_key: str,
                    use_graph: Optional[bool] = None, stateless: Optional[bool] = None) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {"answer": PIPELINE_CFG["empty_question_answer"],
                "sources": [], "graph_hits": [], "graph_graph": {}, "graph_error": ""}
 
    if is_meta_command(q):
        mgr = get_global_manager()
        stateless_flag = bool(stateless) if stateless is not None else bool(getattr(settings, "stateless_default", False))
        if not stateless_flag:
            try:
                store = mgr.store
                state = store.load(user_key)
                turns = state.get("turns", []) or []
                turns.append({"role": "user", "text": q})
                turns.append({"role": "assistant", "text": "Topic cleared."})
                store.save(user_key, "", turns, extra_state={
                    "last_focus": "", "last_topic": "",
                    "anchor": {}, "anchor_last_action": "meta_command_reset",
                    "summary_updated": False, "retrieval_confidence": "",
                })
            except Exception:
                pass
        return {"answer": "Topic cleared. What would you like to explore next?",
                "sources": [], "graph_hits": [], "graph_graph": {}, "graph_error": "",
                "timing_ms": {"total_ms": 0.0}, "llm_calls": {"answer_llm_calls": 0, "utility_llm_calls": 0}}
 
    t0_total = time.perf_counter()
    generation_time_ms = 0.0
    answer_llm_calls = 0
 
    stateless = bool(stateless) if stateless is not None else bool(getattr(settings, "stateless_default", False))
    use_graph_flag = settings.use_graph if use_graph is None else bool(use_graph)
    mgr = get_global_manager()
    requested_mode = str(getattr(settings, "active_mode", "") or "").strip()
    effective_mode = mgr.dbm.resolve_mode(requested_mode)
    answer_model_key = str(getattr(settings, "answer_model_key", "") or settings.llm_model)

    meta_query_type = _detect_meta_query(q)
    if meta_query_type:
        mgr.switch_mode(effective_mode)
        meta_answer = _answer_meta_query(meta_query_type, mgr, effective_mode)
        if meta_answer:
            if not stateless:
                try:
                    state = mgr.store.load(user_key)
                    turns = state.get("turns", []) or []
                    turns.append({"role": "user", "text": q})
                    turns.append({"role": "assistant", "text": meta_answer})
                    mgr.store.save(user_key, state.get("rolling_summary", "") or "", turns)
                except Exception:
                    pass
            total_ms = (time.perf_counter() - t0_total) * 1000.0
            return {"answer": meta_answer,
                    "sources": [], "graph_hits": [], "graph_graph": {}, "graph_error": "",
                    "timing_ms": {"total_ms": round(total_ms, 2), "rewrite_ms": 0.0,
                                  "retrieval_total_ms": 0.0, "generation_ms": 0.0},
                    "llm_calls": {"answer_llm_calls": 0, "utility_llm_calls": 0}}
 
    pre_state = {"rolling_summary": "", "turns": []} if stateless else mgr.store.load(user_key)
    mgr.switch_mode(effective_mode)
    mgr.switch_answer_model(answer_model_key)
 
    engine_lock_ctx = nullcontext()
    if hasattr(mgr, "answer_generation_lock") and not bool(getattr(settings, "allow_utility_concurrency", False)):
        engine_lock_ctx = mgr.answer_generation_lock
 
    with engine_lock_ctx:
        eng = mgr.get_engine(user_key, mode=effective_mode, stateless=stateless)
        context = eng.prepare_context(q, stateless=stateless)
        context.raw_question = q
 
        paper_docs_raw = list(context.paper_docs or [])
        resolved_q = (context.rewritten_question or "").strip()
        prev_user_text = ""
        for t in reversed(list(pre_state.get("turns", []) or [])):
            if str((t or {}).get("role", "") or "").strip().lower() == "user":
                prev_user_text = str((t or {}).get("text", "") or "").strip()
                break
        resolved_q = re.sub(r"\s+", " ", (resolved_q or q)).strip()
 
        summary_intent = _is_summary_intent(q)
        detected_intent = str(getattr(context, "detected_intent", "") or classify_generic_intent(q))
        retrieval_q = resolved_q or q
        raw_retrieval_count = len(paper_docs_raw)

        query_embedding = eng._embed_query_vector(
            strip_corpus_noise_terms(retrieval_q) or retrieval_q
        ) if retrieval_q.strip() else None
 
        _early_person = strip_possessive(_extract_person_name(q) or "")
        if not (_early_person and len(_early_person) > 3 and looks_like_person_candidate(_early_person)):
            if resolved_q and resolved_q != q:
                _early_person = strip_possessive(_extract_person_name(resolved_q) or "")
                if not (_early_person and len(_early_person) > 3 and looks_like_person_candidate(_early_person)):
                    _early_person = ""
            else:
                _early_person = ""

        if summary_intent and _early_person:
            _person_toks = set(tokenize_words(_early_person))
            _q_toks = tokenize_words(retrieval_q)
            _kept = [t for t in _q_toks
                     if t in _person_toks or (len(t) >= 4 and not is_generic_query_token(t))]
            _focused = " ".join(_kept).strip()
            if _focused and len(_focused) >= len(_early_person):
                retrieval_q = _focused

        _pre_extra = pre_state.get("extra_state", {}) if isinstance(pre_state.get("extra_state"), dict) else {}
        _pre_anc = _pre_extra.get("anchor") or pre_state.get("anchor") or {}
        _pa_val = str((_pre_anc or {}).get("value", "") or "").strip()
        _pa_conf = float((_pre_anc or {}).get("confidence", 0) or 0)
 
        first_pass_docs = _filter_noisy_docs(paper_docs_raw, retrieval_q,
            person_name=_early_person, anchor_value=_pa_val, anchor_confidence=_pa_conf,
            summary_intent=summary_intent)
        dominance = _dominant_metadata_filter_from_docs(
            first_pass_docs, resolved_q or q,
            majority_ratio=float(getattr(settings, "dominant_majority_ratio", 0.6)),
            min_count=int(getattr(settings, "dominant_min_count", 3)))
        dominant_filter = dict(dominance.get("filter", {}) or {})
        dom_conf = float(dominance.get("confidence", 0) or 0)
        dom_ratio = float(dominance.get("ratio", 0) or 0)
        replace_conf = float(getattr(settings, "dominant_replace_confidence", 0.82))
        maj_ratio = float(getattr(settings, "dominant_majority_ratio", 0.6))
        dom_usable = (bool(dominance.get("dominant")) and bool(dominant_filter)
                      and not is_placeholder_anchor_value(str(dominant_filter.get("value", "") or "")))
 
        paper_docs = list(first_pass_docs)
        dom_second_count = 0
        budget_papers = int((context.budgets or {}).get("BUDGET_PAPERS", getattr(settings, "budget_papers", 0)))
        explicit_person_name = ""
        person_support = {"strong": 0, "matched": 0, "total": 0}

        _is_coref_followup = is_followup_coref_question(q)
        _anchor_is_researcher = (
            _pa_val
            and _pa_conf >= float(getattr(settings, "anchor_stable_confidence", 0.72))
            and str((_pre_anc or {}).get("type", "")).strip().lower() == "researcher"
        )

        # --- Fix: detect unresolvable person-pronoun references ---
        # If the question uses person pronouns (he/she/him/her/they/them) but
        # we have no active researcher anchor, no person detected in the raw or
        # rewritten query, and the rewrite didn't inject any substantive entity,
        # the pronoun can't be resolved.  Flag it so we return a clarification
        # message rather than a thin answer from generic retrieval.
        #
        # Only person pronouns trigger this — impersonal pronouns (it/this/that)
        # referring to topics often work fine with broad retrieval.  Questions
        # like "tell me more about that" retrieve topically even without explicit
        # resolution, but "what else has he published" is truly unresolvable.
        _person_pat = get_person_pronoun_pattern()
        _dangling_pronoun = False
        if (_is_coref_followup and not _anchor_is_researcher and not _early_person
                and _person_pat and _person_pat.search(q)):
            # Check whether the rewrite actually resolved the pronoun by
            # comparing substantive tokens between raw and resolved queries.
            _min_injected = max(1, int(getattr(settings, "dangling_pronoun_min_injected", 2)))
            _min_raw = max(1, int(getattr(settings, "dangling_pronoun_min_raw_substantive", 2)))
            _raw_subst = {t for t in tokenize_words(q)
                          if len(t) >= 4 and not is_generic_query_token(t)}
            _res_subst = {t for t in tokenize_words(resolved_q or q)
                          if len(t) >= 4 and not is_generic_query_token(t)}
            _injected = _res_subst - _raw_subst
            if len(_injected) < _min_injected and len(_raw_subst) < _min_raw:
                # Neither the raw query nor the rewrite has enough substance
                # to anchor the retrieval — this is a dangling person pronoun.
                _dangling_pronoun = True
        # --- end fix ---

        if _is_coref_followup and _anchor_is_researcher and not _early_person:
            _anchor_docs, _, _ = _retrieve_for_person(
                eng, _pa_val, retrieval_q, budget_papers,
                query_embedding, q,
                pa_val=_pa_val, pa_conf=_pa_conf,
                summary_intent=summary_intent,
                existing_docs=paper_docs,
            )
            if _anchor_docs:
                paper_docs = dedupe_docs(_anchor_docs)
                dom_second_count = len(paper_docs)
                _early_person = _pa_val
                explicit_person_name = _pa_val
 
        if dom_usable:
            where = {str(dominant_filter.get("key", "")): str(dominant_filter.get("value", ""))}
            second = _filter_noisy_docs(
                eng.retrieve_papers(retrieval_q, budget_papers, query_embedding=None,
                                    where_filter=where, raw_question=q), retrieval_q,
                person_name=_early_person, anchor_value=_pa_val, anchor_confidence=_pa_conf,
                summary_intent=summary_intent)
            dom_second_count = len(second)
            paper_docs = dedupe_docs(paper_docs + second)
 
        _person_from_q = strip_possessive(_extract_person_name(q) or "")
        if not (_person_from_q and len(_person_from_q) > 3 and looks_like_person_candidate(_person_from_q)):
            _person_from_q = ""
        _person_from_resolved = ""
        if resolved_q and resolved_q != q:
            _person_from_resolved = strip_possessive(_extract_person_name(resolved_q) or "")
            if not (_person_from_resolved and len(_person_from_resolved) > 3
                    and looks_like_person_candidate(_person_from_resolved)):
                _person_from_resolved = ""
        _detected_person = _person_from_q or _person_from_resolved

        if not dom_usable and _detected_person:
            person_name = _detected_person
            explicit_person_name = person_name

            selected_docs, _retrieval_stage, person_support = _retrieve_for_person(
                eng, person_name, retrieval_q, budget_papers,
                query_embedding, q,
                pa_val=_pa_val, pa_conf=_pa_conf,
                summary_intent=summary_intent,
                existing_docs=paper_docs,
            )

            if person_support["matched"] > 0:
                paper_docs = dedupe_docs(selected_docs)
            else:
                explicit_person_name = ""
                person_support = {"strong": 0, "matched": 0, "total": 0}

            dominance = _dominant_metadata_filter_from_docs(
                paper_docs, resolved_q or q,
                majority_ratio=float(getattr(settings, "dominant_majority_ratio", 0.6)),
                min_count=int(getattr(settings, "dominant_min_count", 3)))
            dominant_filter = dict(dominance.get("filter", {}) or {})
 
        post_filter_count = len(paper_docs)
        mem_docs = context.mem_docs or []

        _comparison_ran = False
        if detected_intent == "comparison":
            comparison_people = _extract_comparison_entities(q)
            if len(comparison_people) >= 2:
                all_comparison_docs = list(paper_docs)
                _found_any = False
                for comp_person in comparison_people:
                    if not comp_person:
                        continue
                    comp_docs, _, comp_support = _retrieve_for_person(
                        eng, comp_person, retrieval_q, budget_papers,
                        query_embedding, q,
                        pa_val=_pa_val, pa_conf=_pa_conf,
                    )
                    if comp_docs and comp_support["matched"] > 0:
                        all_comparison_docs.extend(comp_docs)
                        _found_any = True
                if _found_any:
                    paper_docs = dedupe_docs(all_comparison_docs)
                    post_filter_count = len(paper_docs)
                    _comparison_ran = True
                    dominance = _dominant_metadata_filter_from_docs(
                        paper_docs, resolved_q or q,
                        majority_ratio=float(getattr(settings, "dominant_majority_ratio", 0.6)),
                        min_count=int(getattr(settings, "dominant_min_count", 3)))
                    dominant_filter = dict(dominance.get("filter", {}) or {})
 
        anchor_before = normalize_anchor(getattr(context, "anchor", {}) or getattr(eng, "anchor", {}))
        candidate_anchor = _build_anchor_from_dominance(dominance)
 
        if explicit_person_name and person_support["matched"] > 0:
            explicit_conf = min(
                0.9,
                max(
                    0.58,
                    float(person_support["matched"]) / max(1, int(person_support["total"])),
                ),
            )
            candidate_anchor = {
                "type": "researcher",
                "value": explicit_person_name,
                "source": "explicit_entity_signal",
                "confidence": explicit_conf,
            }
 
        if not candidate_anchor and not dom_usable:
            person_name = strip_possessive(_extract_person_name(q) or "")
            if person_name and len(person_name) > 3 and looks_like_person_candidate(person_name):
                ratio = anchor_support_ratio(person_name, paper_docs)
                if ratio >= 0.1:
                    candidate_anchor = {
                        "type": "researcher", "value": person_name,
                        "source": "explicit_entity_signal",
                        "confidence": min(0.85, max(0.6, ratio)),
                    }
 
        anchor_after, anchor_action = _choose_anchor_update(
            current_anchor=anchor_before, candidate_anchor=candidate_anchor,
            dominance=dominance, question=q, resolved_question=resolved_q)
 
        if anchor_after and anchor_after.get("value"):
            anchor_val = str(anchor_after.get("value", "") or "").strip()
            if anchor_val and not anchor_in_text(anchor_val, q):
                a_toks = {t for t in tokenize_words(anchor_val) if len(t) >= 3 and not is_generic_query_token(t)}
                q_toks = {t for t in tokenize_words(q) if len(t) >= 3}
                if a_toks and not (a_toks & q_toks) and not is_followup_coref_question(q):
                    anchor_after = {}
                    anchor_action = "cleared_no_query_overlap"
 
        eng.anchor = dict(anchor_after or {})
        eng.anchor_last_action = anchor_action
        context.anchor = dict(anchor_after or {})
        context.paper_docs = list(paper_docs)
 
        anchor_value = str(anchor_after.get("value", "") or "").strip()
        anc_ratio = anchor_support_ratio(anchor_value, paper_docs) if anchor_value else 1.0
 
        anchor_confirmed_by_dominance = (
            dom_second_count > 0
            and anchor_value
            and _normalize_meta_value(dominant_filter.get("value", "")) == _normalize_meta_value(anchor_value)
        )
        if (anchor_value and anc_ratio < 0.15
                and not is_followup_coref_question(q)
                and not anchor_confirmed_by_dominance):
            anchor_after = {}
            anchor_action = "cleared_low_support_ratio"
            anchor_value = ""
            anc_ratio = 1.0
            eng.anchor = {}
            eng.anchor_last_action = anchor_action
            context.anchor = {}
 
        anchor_consistent = (not anchor_value) or (anc_ratio >= float(
            getattr(settings, "anchor_consistency_min_ratio", 0.45)))
        retrieval_confidence = retrieval_confidence_label(
            docs_count=post_filter_count, anchor_consistent=anchor_consistent)
        if explicit_person_name and person_support["total"] > 0:
            retrieval_confidence = _downshift_confidence_for_person_support(
                retrieval_confidence,
                strong_count=int(person_support["strong"]),
                matched_count=int(person_support["matched"]),
                total_count=int(person_support["total"]),
            )
        eng.last_retrieval_confidence = retrieval_confidence
        eng.last_anchor_support_ratio = float(anc_ratio)
 
        if _early_person and post_filter_count > 0:
            _rs = {norm_text(str((getattr(d, "metadata", {}) or {}).get("researcher", "")))
                   for d in paper_docs} - {"", "unknown", "n/a"}
            if len(_rs) > 3 and len(_rs) / max(1, post_filter_count) > 0.5:
                retrieval_confidence = _downgrade_confidence(retrieval_confidence, steps=1)
                eng.last_retrieval_confidence = retrieval_confidence

        _dom_found_person = (
            dom_usable
            and str(dominant_filter.get("key", "")).lower() == "researcher"
            and bool(dominant_filter.get("value", ""))
        )
        _anchor_followup_found = (
            _is_coref_followup and _anchor_is_researcher
            and dom_second_count > 0
        )

        if (_detected_person and post_filter_count > 0
                and explicit_person_name == ""
                and person_support["matched"] == 0
                and not _dom_found_person
                and not _anchor_followup_found
                and not _comparison_ran):
            retrieval_confidence = "weak"
            eng.last_retrieval_confidence = retrieval_confidence
 
        _cf = {"high": 1.0, "medium": 0.85, "low": 0.65, "weak": 0.45, "inconsistent": 0.4
               }.get(retrieval_confidence, 0.7)

        _runtime = getattr(mgr, "answer_runtime", None)
        _token_budget = _runtime_prompt_token_budget(
            _runtime, int(getattr(settings, "answer_max_new_tokens", 384)))
        _REFERENCE_BUDGET = 16000
        _budget_ratio = min(1.0, max(0.25, _token_budget / _REFERENCE_BUDGET)) if _token_budget > 0 else 0.5
        _base_max_docs = int(getattr(settings, "prompt_max_docs", 24))
        prompt_max_docs = max(2, int(_base_max_docs * _cf * _budget_ratio))
        prompt_text_limit = max(160, int(int(getattr(settings, "prompt_doc_text_limit", 800)) * _cf))
 
        rolling_summary_text = pre_state.get("rolling_summary", "") or ""
        pre_turns_count = len(pre_state.get("turns", []) or [])
        topic_shifted = anchor_action in {
            "cleared_no_query_overlap", "cleared_low_support_ratio",
        } or (anchor_action == "none" and not anchor_before and not anchor_after)
        recent_turns_ctx = ""
        if not topic_shifted:
            recent_turns_ctx = _build_recent_turns_context(
                pre_state, max_turns=min(6, max(2, int(PIPELINE_CFG["recent_turns_in_prompt"]))))
 
        prompt_base = [PIPELINE_CFG["prompt_prefix"].rstrip()]
        if recent_turns_ctx:
            prompt_base.append("RECENT TURNS (for coreference only, not evidence):\n" + recent_turns_ctx)
        if mem_docs:
            mem_lines = [f"- {str(getattr(md, 'page_content', '') or '').strip()[:200]}"
                         for md in mem_docs[:6]
                         if str(getattr(md, "page_content", "") or "").strip()]
            if mem_lines:
                prompt_base.append("CONVERSATION MEMORY:\n" + "\n".join(mem_lines))
 
        if _cf < 0.5:
            prompt_base.append(
                f"EVIDENCE STRENGTH: Low ({retrieval_confidence}). "
                "Clearly state uncertainty. Do not present weak evidence as definitive.")
 
        question_for_answer = (resolved_q or q).lower()
 
        style_hints = {
            "summary": ("The retrieved context contains paper records with a 'summary' field. "
                        "Use ONLY the content from these summary fields to answer. "
                        "Extract and report the key mechanisms, findings, and methods described "
                        "in those summaries directly — do not invent or paraphrase beyond what "
                        "is stated. Attribute each point to its paper title and researcher. "
                        "If a paper has no summary field, describe it based on its title only. "
                        "Write at least one detailed paragraph per relevant paper."),
            "comparison": "Compare the relevant items directly and keep claims grounded in retrieved evidence.",
            "time_range": ("If asked about the studied time period, report it only when explicitly stated in retrieved context. "
                           "If not stated, say that directly and optionally provide publication years as a separate fallback. "
                           "Do not infer studied period from publication years."),
            "list": "Provide a concise list with short evidence-backed descriptors.",
        }
        if summary_intent:
            style_hint = style_hints["summary"]
        elif detected_intent in style_hints:
            style_hint = style_hints[detected_intent]
        else:
            style_hint = (
                "Write a direct, evidence-grounded answer using ONLY the retrieved paper records below. "
                "For each researcher in the records: name them, describe their specific research topics "
                "from the summary fields, and cite at least one paper title in quotes. "
                "Give each researcher their own paragraph. "
                "Write at least 3 sentences per researcher. "
                "Do not use outside knowledge. Do not fabricate facts."
            )

        if _detected_person and explicit_person_name and paper_docs:
            _topics_set = set()
            for _d in paper_docs:
                _m = getattr(_d, "metadata", {}) or {}
                for _tk in ("primary_topic", "topic", "field", "department"):
                    _tv = str(_m.get(_tk, "") or "").strip().lower()
                    if _tv and len(_tv) > 3 and _tv not in {"unknown", "n/a", "other"}:
                        _topics_set.add(_tv)
            _title_words = {}
            for _d in paper_docs[:12]:
                _m = getattr(_d, "metadata", {}) or {}
                _t = str(_m.get("title", "") or "").lower()
                for _w in re.findall(r"[a-z]{5,}", _t):
                    if not is_generic_query_token(_w):
                        _title_words[_w] = _title_words.get(_w, 0) + 1
            _max_freq = max(_title_words.values()) if _title_words else 0
            _n_unique = len([w for w, c in _title_words.items() if c >= 2])
            if len(_topics_set) > 2 or (_max_freq <= 2 and _n_unique <= 1 and len(paper_docs) > 4):
                style_hint += (
                    "\nIMPORTANT: The retrieved papers for this person may span multiple distinct "
                    "research areas. This could indicate multiple researchers with the same name "
                    "in the database. If you notice the papers cover unrelated fields, note this "
                    "explicitly and describe each research area separately. "
                    "Do not blend unrelated research areas into a single description."
                )
 
        _by_r = {}
        for _d in paper_docs:
            _by_r.setdefault(norm_text(str((getattr(_d, "metadata", {}) or {}).get("researcher", ""))) or "_", []).append(_d)
        if len(_by_r) >= 2:
            _cap = max(2, prompt_max_docs // len(_by_r))
            _bal = [d for rds in _by_r.values() for d in rds[:_cap]][:prompt_max_docs]
        else:
            _bal = paper_docs
 
        prompt, used_docs, used_text_limit = _fit_prompt_to_budget(
            runtime=getattr(mgr, "answer_runtime", None), docs=_bal,
            base_sections=prompt_base, style_hint=style_hint,
            question_for_answer=question_for_answer,
            max_docs=prompt_max_docs, text_limit=prompt_text_limit)
 
        _person_not_found = (
            _detected_person
            and not explicit_person_name
            and not _dom_found_person
            and not _anchor_followup_found
            and not _comparison_ran
            and person_support.get("matched", 0) == 0
            and retrieval_confidence == "weak"
        )

        # --- Fix: produce a clarification message for dangling pronouns ---
        # If we flagged this question as having an unresolvable pronoun
        # reference, return a helpful clarification prompt instead of
        # running the LLM on generic / irrelevant retrieval results.
        if _dangling_pronoun and not explicit_person_name and not _dom_found_person:
            _prev_user = prev_user_text or ""
            _hint = ""
            if _prev_user:
                _prev_topic = _extract_person_name(_prev_user) or ""
                if not _prev_topic:
                    _hint_max = max(10, int(getattr(settings, "dangling_pronoun_hint_max_chars", 60)))
                    _prev_topic = collapse_whitespace(_prev_user)[:_hint_max].rstrip()
                if _prev_topic:
                    try:
                        _hint = PIPELINE_CFG["dangling_pronoun_hint"].format(prev_topic=_prev_topic)
                    except (KeyError, IndexError):
                        _hint = ""
            answer_text = PIPELINE_CFG["dangling_pronoun_answer"].format(hint=_hint)
        elif not paper_docs or _person_not_found:
            if _person_not_found and _detected_person:
                answer_text = (
                    f'I could not find any papers by "{_detected_person}" in the '
                    f'retrieved corpus. The name may be stored differently in the '
                    f'database, or this researcher may not be in the current collection. '
                    f'Please try a different spelling or provide additional context '
                    f'(e.g., department or research area).'
                )
            else:
                answer_text = _insufficient_context_answer(q, detected_intent)
        else:
            prompt_for_call = prompt
 
            low_evidence_guard = (
                retrieval_confidence in {"weak", "low"}
                and anchor_action in {"cleared_no_query_overlap", "cleared_weak_retrieval", "none"}
                and post_filter_count <= 4
            )
 
            if retrieval_confidence == "inconsistent" or low_evidence_guard:
                base_no_summary = [s for s in prompt_base if not s.startswith("ROLLING SUMMARY")]
                if retrieval_confidence == "inconsistent":
                    inconsistent_guard_style = (
                        "CRITICAL: The retrieved evidence is inconsistent — the papers do not "
                        "all match the current question. "
                        "You MUST ONLY report what is EXPLICITLY stated in the retrieved paper records. "
                        "Do NOT draw on outside knowledge, training data, or prior conversation topics. "
                        "List the researchers found in the retrieved records along with their paper titles. "
                        "For each researcher, briefly describe what their retrieved papers are about. "
                        "If a researcher's paper is off-topic, still list it — do not invent relevance. "
                        "Do NOT fabricate facts, affiliations, or locations not in the records."
                    )
                    prompt_for_call, used_docs, used_text_limit = _fit_prompt_to_budget(
                        runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                        base_sections=base_no_summary, style_hint=inconsistent_guard_style,
                        question_for_answer=question_for_answer,
                        max_docs=prompt_max_docs, text_limit=prompt_text_limit)
                elif low_evidence_guard:
                    guard_style = (
                        "CRITICAL: Very few papers were retrieved and evidence is weak. "
                        "Only describe what is EXPLICITLY stated in the retrieved papers. "
                        "Do NOT speculate, infer connections, or relate this topic to "
                        "previous conversation topics. If the papers don't clearly address "
                        "the question, say so directly and list the papers that were found. "
                        "Do NOT fabricate relevance between unrelated research areas."
                    )
                    prompt_for_call, used_docs, used_text_limit = _fit_prompt_to_budget(
                        runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                        base_sections=base_no_summary, style_hint=guard_style,
                        question_for_answer=question_for_answer,
                        max_docs=prompt_max_docs, text_limit=prompt_text_limit)
                else:
                    prompt_for_call, used_docs, used_text_limit = _fit_prompt_to_budget(
                        runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                        base_sections=base_no_summary, style_hint=style_hint,
                        question_for_answer=question_for_answer,
                        max_docs=prompt_max_docs, text_limit=prompt_text_limit)
 
            t0_gen = time.perf_counter()
            answer_llm_calls += 1
            raw_answer = ""
            llm_error = None
            try:
                raw_answer = _invoke_with_timeout(mgr.answer_runtime.llm, prompt_for_call,
                                                  int(getattr(settings, "llm_timeout_s", 40)))
            except Exception as e:
                llm_error = f"{type(e).__name__}: {e}"
                logger.exception("Answer LLM call failed (attempt 1): %s", llm_error)
            generation_time_ms = (time.perf_counter() - t0_gen) * 1000.0
            answer_text = _extract_answer_text(raw_answer)
 
            if getattr(settings, "debug_rag", False):
                _raw_len = len(raw_answer or "")
                _ext_len = len(answer_text or "")
                print(f"\n[ANSWER_DEBUG] raw_len={_raw_len} extracted_len={_ext_len} "
                      f"gen_ms={generation_time_ms:.0f} err={llm_error or 'none'}")
                if _raw_len > 0 and _ext_len == 0:
                    print(f"[ANSWER_DEBUG] RAW_ANSWER_FIRST_500: {(raw_answer or '')[:500]}")
                elif _raw_len > 0 and _ext_len < 24:
                    print(f"[ANSWER_DEBUG] SHORT_ANSWER: {answer_text!r}")
 
            _timeout_s = int(getattr(settings, "llm_timeout_s", 40))
            _timed_out = generation_time_ms >= (_timeout_s * 1000 * 0.95)
            if not answer_text and not _timed_out:
                retry_base = base_no_summary if (retrieval_confidence == "inconsistent" or low_evidence_guard) else prompt_base
                retry_prompt, _, _ = _fit_prompt_to_budget(
                    runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                    base_sections=retry_base, style_hint=style_hint,
                    question_for_answer=question_for_answer,
                    max_docs=max(1, min(6, used_docs // 2)),
                    text_limit=max(140, min(320, int(max(120, used_text_limit) * 0.7))))
                t0_r = time.perf_counter()
                answer_llm_calls += 1
                raw_retry = ""
                try:
                    raw_retry = _invoke_with_timeout(mgr.answer_runtime.llm, retry_prompt,
                                                     int(getattr(settings, "llm_timeout_s", 40)))
                except Exception as e:
                    retry_err = f"{type(e).__name__}: {e}"
                    logger.exception("Answer LLM call failed (retry): %s", retry_err)
                    if llm_error is None:
                        llm_error = retry_err
                generation_time_ms += (time.perf_counter() - t0_r) * 1000.0
                answer_text = _extract_answer_text(raw_retry)
 
            if not answer_text:
                answer_text = PIPELINE_CFG["llm_no_answer"]
 
        if paper_docs:
            cleaned = (answer_text or "").strip()
            weak_or_inconsistent = retrieval_confidence in {"weak", "inconsistent"}
            if (not cleaned
                or cleaned == PIPELINE_CFG["llm_no_answer"]
                or len(cleaned) < 24
                or (weak_or_inconsistent and len(cleaned) < 120)):
                answer_text = _fallback_answer_from_docs(q, paper_docs, detected_intent) or answer_text
 
            researcher_evidence = _supported_researcher_evidence(paper_docs)
            explicit_person_query = bool(strip_possessive(_extract_person_name(q) or ""))
 
            answer_text = _strip_hallucinated_citations(answer_text, paper_docs)
 
            has_unsupported = _answer_mentions_unsupported_researcher(answer_text, paper_docs)
 
            if has_unsupported and retrieval_confidence in {"weak", "low"}:
                safe_answer = _build_researcher_extract_answer(paper_docs)
                if safe_answer:
                    answer_text = safe_answer
                else:
                    answer_text = _fallback_answer_from_docs(q, paper_docs, detected_intent) or answer_text
            elif (not explicit_person_query and len(researcher_evidence) >= 2
                  and has_unsupported and retrieval_confidence == "inconsistent"):
                answer_text = _build_researcher_extract_answer(paper_docs) or answer_text
 
        pre_sanitize = (answer_text or "").strip()
        sanitized = _sanitize_user_answer(answer_text).strip()
        if sanitized and len(sanitized) >= len(pre_sanitize) * 0.2:
            answer_text = sanitized
        elif sanitized:
            answer_text = _strip_leading_answer_labels(pre_sanitize).strip() or pre_sanitize
        else:
            answer_text = _strip_leading_answer_labels(pre_sanitize).strip() or pre_sanitize or PIPELINE_CFG["llm_no_answer"]
        eng.last_answer_llm_calls = int(answer_llm_calls)
        eng.finalize_turn(context, answer_text, no_results=not paper_docs)
 
    return _build_output(
        answer_text=answer_text, paper_docs=paper_docs, mem_docs=mem_docs,
        eng=eng, mgr=mgr, q=q, resolved_q=resolved_q, retrieval_q=retrieval_q,
        detected_intent=detected_intent, effective_mode=effective_mode,
        requested_mode=requested_mode, retrieval_confidence=retrieval_confidence,
        dominance=dominance, dominant_filter=dominant_filter, dom_usable=dom_usable,
        anchor_before=anchor_before, anchor_after=anchor_after, anchor_action=anchor_action,
        anc_ratio=anc_ratio, anchor_consistent=anchor_consistent,
        raw_retrieval_count=raw_retrieval_count, first_pass_docs=first_pass_docs,
        dom_second_count=dom_second_count, post_filter_count=post_filter_count,
        rolling_summary_text=rolling_summary_text, pre_turns_count=pre_turns_count,
        stateless=stateless, t0_total=t0_total, generation_time_ms=generation_time_ms,
        answer_llm_calls=answer_llm_calls, user_key=user_key, use_graph_flag=use_graph_flag,
        pre_state=pre_state,
        prompt_max_docs=prompt_max_docs,
        prompt_text_limit=prompt_text_limit)
 
def _build_output(*, answer_text, paper_docs, mem_docs, eng, mgr, q, resolved_q,
                  retrieval_q, detected_intent, effective_mode, requested_mode,
                  retrieval_confidence, dominance, dominant_filter, dom_usable,
                  anchor_before, anchor_after, anchor_action, anc_ratio, anchor_consistent,
                  raw_retrieval_count, first_pass_docs, dom_second_count, post_filter_count,
                  rolling_summary_text, pre_turns_count, stateless, t0_total,
                  generation_time_ms, answer_llm_calls, user_key, use_graph_flag,
                  pre_state,
                  prompt_max_docs, prompt_text_limit) -> Dict[str, Any]:
 
    post_state = ({"rolling_summary": rolling_summary_text, "turns": pre_state.get("turns", []) or []}
                  if stateless else mgr.store.load(user_key))
    post_turns = post_state.get("turns", []) or []
    post_summary = post_state.get("rolling_summary", "") or ""
    post_extra = post_state.get("extra_state", {}) if isinstance(post_state.get("extra_state"), dict) else {}
    post_anchor = normalize_anchor(post_state.get("anchor") or post_extra.get("anchor") or anchor_after)
    post_anchor_action = str(post_state.get("anchor_last_action") or post_extra.get("anchor_last_action") or anchor_action)
    post_summary_updated = bool(post_state.get("summary_updated") if "summary_updated" in post_state
                                else post_extra.get("summary_updated", getattr(eng, "last_summary_updated", False)))
    post_conf = str(post_state.get("retrieval_confidence") or post_extra.get("retrieval_confidence") or retrieval_confidence)
    post_anc_ratio = float(post_extra.get("anchor_support_ratio",
                           getattr(eng, "last_anchor_support_ratio", anc_ratio)) or 0.0)
 
    sources, seen_titles = [], set()
    for d in paper_docs:
        try:
            meta = getattr(d, "metadata", {}) or {}
            tk = re.sub(r"\s+", " ", str(meta.get("title", "") or "").strip().lower())
            if tk and tk != "untitled" and tk in seen_titles:
                continue
            if tk and tk != "untitled":
                seen_titles.add(tk)
            sources.append(_doc_to_source_md(d))
        except Exception:
            pass
 
    timing = {
        "rewrite_ms": round(float(getattr(eng, "last_rewrite_time_ms", 0) or 0), 2),
        "retrieval_total_ms": round(float(getattr(eng, "last_retrieval_time_ms", 0) or 0), 2),
        "generation_ms": round(float(generation_time_ms or 0), 2),
        "total_ms": round((time.perf_counter() - t0_total) * 1000.0, 2),
    }
    llm_calls = {
        "answer_llm_calls": int(getattr(eng, "last_answer_llm_calls", answer_llm_calls) or 0),
        "utility_llm_calls": int(getattr(eng, "last_utility_llm_calls", 0) or 0),
    }
    timing["llm_calls"] = llm_calls
 
    session_state = {
        "session_id": user_key, "pre_turn_count": pre_turns_count,
        "turn_count": len(post_turns), "turn_count_delta": len(post_turns) - pre_turns_count,
        "pre_summary_len": len(rolling_summary_text), "summary_len": len(post_summary),
        "summary_len_delta": len(post_summary) - len(rolling_summary_text),
        "summary_non_empty": bool(post_summary.strip()),
        "turns_json_chars": len(json.dumps(post_turns, ensure_ascii=False)),
        "detected_intent": detected_intent, "requested_mode": requested_mode,
        "effective_mode": effective_mode, "anchor": post_anchor,
        "anchor_action": post_anchor_action, "summary_updated": post_summary_updated,
        "retrieval_confidence": post_conf, "anchor_support_ratio": round(post_anc_ratio, 4),
        "timing_ms": timing, "llm_calls": llm_calls,
    }
 
    chroma_refs = [_doc_to_ref(d) for d in paper_docs[:12]]
    mem_refs = [_doc_to_ref(d) for d in mem_docs[:8]]
 
    chroma_json = {
        "count": len(paper_docs), "retrieval_count_raw": raw_retrieval_count,
        "first_pass_count": len(first_pass_docs), "dominant_second_pass_count": dom_second_count,
        "fallback_unfiltered_count": 0, "post_filter_count": post_filter_count,
        "dominance": dominance, "dominant_metadata_filter": dominant_filter,
        "dominant_filter_usable": dom_usable,
        "anchor_support_ratio": round(float(anc_ratio), 4),
        "anchor_consistent": bool(anchor_consistent), "retrieval_confidence": retrieval_confidence,
        "doc_refs": chroma_refs,
    }
 
    user_query_json = {
        "text": q, "resolved_text": resolved_q, "standalone_question": resolved_q,
        "detected_intent": detected_intent, "retrieval_text": retrieval_q,
        "dominance": dominance, "dominant_metadata_filter": dominant_filter,
        "dominant_filter_usable": dom_usable,
        "anchor_before": anchor_before, "anchor_after": anchor_after, "anchor_action": anchor_action,
        "rewrite_blocked": bool(getattr(eng, "last_rewrite_blocked", False)),
        "rewrite_anchor_valid": bool(getattr(eng, "last_rewrite_anchor_valid", False)),
        "requested_mode": requested_mode, "effective_mode": effective_mode,
        "stateless": stateless, "timestamp": utcnow_iso(),
    }
 
    rolling_json = {"summary": post_summary, "turns": len(post_turns), "updated": post_summary_updated}
    combined = {
        "user_query": user_query_json, "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_json,
        "conversation_memory": {"count": len(mem_docs), "doc_refs": mem_refs},
        "session_state": session_state,
        "context_size_chars": len(build_compact_context(paper_docs, max_docs=prompt_max_docs,
                                                        text_limit=prompt_text_limit)),
        "timing_ms": timing, "llm_calls": llm_calls,
    }
 
    if getattr(settings, "debug_rag", False):
        try:
            print("\n[PIPELINE_JSON]")
            print(json.dumps(combined, ensure_ascii=False, indent=2))
        except Exception:
            pass
 
    out = {
        "answer": answer_text, "sources": sources,
        "graph_hits": [], "graph_graph": {}, "graph_error": "",
        "user_query": user_query_json, "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_json,
        "conversation_memory": {"count": len(mem_docs), "doc_refs": mem_refs},
        "session_state": session_state,
        "timing_ms": timing, "llm_calls": llm_calls, "pipeline_json": combined,
    }
 
    if use_graph_flag:
        g = graph_retrieve_from_paper_docs(paper_docs, height=650)
        out["graph_hits"] = g.get("hits", []) or []
        out["graph_graph"] = g.get("graph", {}) or {}
        out["graph_error"] = g.get("error", "") or ""

    return out
 
assert callable(answer_question), (
    "rag_pipeline.answer_question must be defined in this module — "
    "do not move it without updating all callers."
)
assert callable(sanitize_answer_for_display), (
    "rag_pipeline.sanitize_answer_for_display must be defined in this module — "
    "do not move it without updating all callers."
)