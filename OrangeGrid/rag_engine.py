#rag_engine.py
import os
import re
import uuid
import json
import gc
import logging
import time
import threading
import queue
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Set

import psutil
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
try:
    import nltk
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree
except Exception:
    nltk = ne_chunk = pos_tag = word_tokenize = Tree = None

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

import config_full as config
from session_store import SessionStore
from database_manager import DatabaseManager
from runtime_settings import settings
from rag_utils import (
    norm_text as _norm_text, clean_html, normalize_title_case as _normalize_title_case,
    collapse_whitespace, tokenize_words as _tokenize_words, token_in_hay,
    get_stopword_set as _get_stopword_set, get_generic_query_terms as _get_generic_query_terms,
    get_followup_phrases as _get_followup_phrases, get_followup_pronoun_pattern as _get_followup_pronoun_pattern,
    get_person_pronoun_pattern as _get_person_pronoun_pattern,
    is_generic_query_token as _is_generic_query_token, is_followup_coref_question as _is_followup_coref_question,
    bootstrap_nltk_data as _bootstrap_nltk_data, strip_corpus_noise_terms as _strip_corpus_noise_terms,
    dedupe_docs, doc_haystack, truncate_text as _truncate_text, clean_snippet as _clean_snippet,
    build_compact_context, dedupe_ci as _dedupe_ci,
    is_placeholder_anchor_value as _is_placeholder_anchor_value, normalize_anchor as _normalize_anchor,
    anchor_in_text as _anchor_in_text, anchor_is_stable as _anchor_is_stable,
    anchor_support_ratio as _anchor_support_ratio, retrieval_confidence_label as _retrieval_confidence_label,
    classify_generic_intent as _classify_generic_intent, strip_prompt_leak as _strip_prompt_leak,
    looks_like_person_candidate as _looks_like_person_candidate, strip_possessive as _strip_possessive,
    has_explicit_entity_signal as _has_explicit_entity_signal, query_tokens_for_relevance,
    is_meta_command as _is_meta_command,
    get_name_token_set as _get_name_token_set,
    get_english_word_set as _get_english_word_set,
    tokenize_name, generate_name_variants, split_author_names,
)

logger = logging.getLogger(__name__)

MEMORY_DIR = os.getenv("RAG_MEMORY_DIR", "chroma_memory")
CACHE_DIR = os.getenv("RAG_CACHE_DIR", "cache")
STATE_DB = os.getenv("RAG_STATE_DB", "chat_state.sqlite")
_EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "").strip().lower()
if _EMBED_DEVICE not in {"cuda", "cpu"}:
    _EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _make_local_chroma_client(persist_dir: str) -> chromadb.Client:
    _ensure_dir(persist_dir)
    return chromadb.PersistentClient(path=persist_dir)

def _dbg(title: str, obj: Any = None, limit: int = 2000) -> None:
    if not getattr(settings, "debug_rag", False):
        return
    print(f"\n{title}")
    if obj is not None:
        s = obj if isinstance(obj, str) else repr(obj)
        print(s[:limit] + "\n...truncated..." if limit and len(s) > limit else s)

@dataclass
class Turn:
    role: str
    text: str

@dataclass
class EngineContext:
    raw_question: str
    rewritten_question: str
    detected_intent: str
    paper_docs: List[Document]
    mem_docs: List[Document]
    stateless: bool
    user_turns: int
    allow_prev_context: bool
    allow_summary: bool
    budgets: Dict[str, int]
    anchor: Dict[str, Any]

def available_ram_mb() -> int:
    return int(psutil.virtual_memory().available / (1024 * 1024))

def available_vram_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free_b, _ = torch.cuda.mem_get_info()
        return int(free_b / (1024 * 1024))
    except Exception:
        return 0

def dynamic_budgets() -> Dict[str, int]:
    ram, vram = available_ram_mb(), available_vram_mb()
    base_memory = int(getattr(settings, "budget_memory", 700))
    base_papers = int(getattr(settings, "budget_papers", 3200))
    base_trigger = int(getattr(settings, "trigger_tokens", 7200))

    pressure = 0
    if ram < 2000: pressure += 2
    elif ram < 4000: pressure += 1
    if torch.cuda.is_available():
        if vram < 1500: pressure += 2
        elif vram < 3000: pressure += 1

    scales = {
        3: (0.65, 0.7, 0.88, 300, 1200, 3000),
        2: (0.78, 0.84, 0.92, 350, 1600, 3500),
        1: (0.9, 0.92, 0.97, 450, 2000, 4200),
    }
    if pressure >= 3:
        sm, sp, st_, mm, mp, mt = scales[3]
    elif pressure == 2:
        sm, sp, st_, mm, mp, mt = scales[2]
    elif pressure == 1:
        sm, sp, st_, mm, mp, mt = scales[1]
    else:
        return {"BUDGET_MEMORY": base_memory, "BUDGET_PAPERS": base_papers, "TRIGGER": base_trigger}

    return {
        "BUDGET_MEMORY": max(mm, int(base_memory * sm)),
        "BUDGET_PAPERS": max(mp, int(base_papers * sp)),
        "TRIGGER": max(mt, int(base_trigger * st_)),
    }

def _no_results_summary_line(question: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    return f"No results for: {q}" if q else "No results for the last query."

_SUMMARY_SECTIONS: Tuple[str, ...] = (
    "Current focus", "Researcher mentions", "Core entities",
    "Key themes", "Constraints", "Open questions",
)

_SUMMARY_ALIASES: Dict[str, str] = {
    "focus": "Current focus", "current focus": "Current focus",
    "researcher mentions": "Researcher mentions", "researchers": "Researcher mentions",
    "entities": "Core entities", "entities discussed": "Core entities",
    "core entities": "Core entities", "findings": "Key themes",
    "key findings so far": "Key themes", "key themes": "Key themes",
    "constraints": "Constraints", "open questions": "Open questions",
}

def _summary_template_empty() -> str:
    blocks = [("Constraints: Use only retrieved Syracuse corpus context." if sec == "Constraints"
                else f"{sec}: (none)") for sec in _SUMMARY_SECTIONS]
    return "\n".join(blocks)

def _extract_summary_sections(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {sec: [] for sec in _SUMMARY_SECTIONS}
    current: Optional[str] = None
    for raw_line in (text or "").splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m:
            canonical = _SUMMARY_ALIASES.get((m.group(1) or "").strip().lower())
            if canonical:
                current = canonical
                rest = (m.group(2) or "").strip().lstrip("- ").strip()
                if rest:
                    out[canonical].append(rest)
                continue
            current = None
            continue
        if current is not None:
            out[current].append(line.lstrip("- ").strip())
    return out

def _format_summary_sections(sections: Dict[str, List[str]]) -> str:
    blocks = []
    for sec in _SUMMARY_SECTIONS:
        vals = [re.sub(r"\s+", " ", v.strip()) for v in (sections.get(sec) or []) if v and v.strip()]
        blocks.append(f"{sec}: {' | '.join(vals or ['(none)'])}")
    return "\n".join(blocks).strip()

def _clean_answer_for_summary_signal(text: str) -> str:
    blocked = ("no further analysis is required", "no additional retrieval is required",
               "no further synthesis is required", "confidence level", "the retrieved context")
    lines = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line or "").strip()
        if not line or any(f in line.lower() for f in blocked):
            continue
        line = re.sub(r"^(summary|answer|response)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if line:
            lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()

def _extract_answer_theme_keywords(answer_text: str, *, max_items: int = 6) -> List[str]:
    cleaned = _clean_answer_for_summary_signal(answer_text)
    if not cleaned:
        return []
    stopset = _get_stopword_set()
    _ACADEMIC_NOISE = {
        "research", "researchers", "study", "studies", "paper", "papers",
        "work", "works", "findings", "finding", "results", "result",
        "analysis", "approach", "approaches", "method", "methods",
        "model", "models", "systems", "system", "based", "using",
        "including", "various", "specific", "particular", "several",
        "different", "important", "significant", "potential", "novel",
        "their", "these", "those", "other", "such", "also", "well",
        "however", "therefore", "provides", "suggests", "describes",
        "explores", "examines", "investigates", "focuses", "primarily",
        "complex", "computational", "mathematical", "theoretical",
    }
    counts: Dict[str, int] = {}
    for tok in _tokenize_words(cleaned):
        t = tok.lower().strip()
        if not t or len(t) < 4 or t.isdigit() or (stopset and t in stopset) or _is_generic_query_token(t):
            continue
        if t in _ACADEMIC_NOISE:
            continue
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
        return [first_sentence[:180].rstrip()] if first_sentence else []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in ranked][:max(1, max_items)]

def _sanitize_entity_values(values: List[str], *, max_items: int = 8) -> List[str]:
    stopset = _get_stopword_set()
    out, seen = [], set()
    for raw in values or []:
        v = re.sub(r"\s+", " ", str(raw or "").strip())
        if not v or len(v) < 3 or re.fullmatch(r"[A-Z][a-z]?$", v):
            continue
        toks = _tokenize_words(v)
        if not toks or len(toks) > 8:
            continue
        if len(toks) == 1 and ((stopset and toks[0] in stopset) or _is_generic_query_token(toks[0])):
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= max(1, max_items):
            break
    return out

def build_rolling_summary(previous_summary: str, user_question: str,
                          retrieval_metadata: str, assistant_answer: str) -> str:
    base = previous_summary if (previous_summary or "").strip() else _summary_template_empty()
    sections = _extract_summary_sections(base)

    q = re.sub(r"\s+", " ", (user_question or "").strip())
    a = re.sub(r"\s+", " ", (assistant_answer or "").strip())
    raw_meta = re.sub(r"\s+", " ", (retrieval_metadata or "").strip())

    compress_threshold = max(600, int(getattr(settings, "summary_compress_threshold_chars", 1400)))
    base_max_items = max(1, int(getattr(settings, "summary_max_items_per_field", 6)))
    max_items_per_field = max(3, base_max_items - 1) if len(base) > compress_threshold else base_max_items

    if q:
        sections["Current focus"] = [q[:220]]

    meta_people_counts: Dict[str, int] = {}
    meta_people_display: Dict[str, str] = {}
    meta_topic_counts: Dict[str, int] = {}
    meta_topic_display: Dict[str, str] = {}

    def _bump(counter: Dict[str, int], display_map: Dict[str, str], value: str) -> None:
        val = re.sub(r"\s+", " ", str(value or "").strip())
        if not val:
            return
        key = _norm_text(val)
        if not key:
            return
        counter[key] = counter.get(key, 0) + 1
        display_map.setdefault(key, val)

    for part in re.split(r"\s*\|\s*", raw_meta):
        item = re.sub(r"\s+", " ", (part or "").strip())
        if not item or len(item) < 3:
            continue
        m = re.match(r"^(researcher|author|topic|entity)\s*:\s*(.+)$", item, re.IGNORECASE)
        label = (m.group(1).strip().lower() if m else "")
        payload = re.sub(r"\s+", " ", (m.group(2) if m else item).strip())
        if not payload:
            continue

        if label in {"researcher", "author"}:
            people = ([payload] if label == "researcher"
                      else [n for n in re.split(r"\s*[;,]\s*|\s+and\s+", payload) if n.strip()])
            for person in people:
                candidate = _strip_possessive(re.sub(r"\s+", " ", person).strip())
                if _looks_like_person_candidate(candidate):
                    _bump(meta_people_counts, meta_people_display, candidate)
            continue

        topic = payload
        if topic.lower().startswith(("untitled", "unknown", "n/a")):
            continue
        if len(topic) > 100:
            topic = topic[:97].rstrip() + "..."
        if _looks_like_person_candidate(topic):
            _bump(meta_people_counts, meta_people_display, topic)
            continue
        topic_tokens = [t for t in _tokenize_words(topic)
                        if len(t) >= 3 and not _is_generic_query_token(t)]
        if len(topic_tokens) < 2:
            continue
        _bump(meta_topic_counts, meta_topic_display, topic)

    researcher_names: List[str] = []
    for key, count in sorted(meta_people_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= 2:
            researcher_names.append(meta_people_display.get(key, key))

    meta_entities: List[str] = []
    for key, count in sorted(meta_topic_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count >= 2:
            meta_entities.append(meta_topic_display.get(key, key))

    person_from_q = _extract_person_name(q)
    if person_from_q:
        person_key = _norm_text(person_from_q)
        answer_has_person = person_key and person_key in _norm_text(a)
        if meta_people_counts.get(person_key, 0) > 0 or answer_has_person:
            researcher_names.insert(0, person_from_q)

    if researcher_names:
        current_people = _dedupe_ci(researcher_names)[-8:]
        existing = [v for v in sections.get("Researcher mentions", [])
                    if v and v.strip() and v.strip() != "(none)"]
        existing_keys = {_norm_text(v) for v in existing if v}
        current_keys = {_norm_text(v) for v in current_people if v}
        if person_from_q or (current_keys and existing_keys and not (current_keys & existing_keys)):
            sections["Researcher mentions"] = current_people
        else:
            sections["Researcher mentions"] = _dedupe_ci(existing + current_people)[-8:]
    elif _has_explicit_entity_signal(q):
        sections["Researcher mentions"] = ["(none)"]

    if meta_entities:
        existing = [v for v in sections.get("Core entities", [])
                    if v and v.strip() and v.strip() != "(none)"]
        sections["Core entities"] = _dedupe_ci(existing + meta_entities)[-max_items_per_field:]

    if a:
        existing_themes = [v for v in sections.get("Key themes", [])
                          if v and v.strip() and v.strip() != "(none)"]
        theme_keywords = _extract_answer_theme_keywords(a, max_items=max_items_per_field)
        if theme_keywords:
            sections["Key themes"] = _dedupe_ci(existing_themes + theme_keywords)[-max_items_per_field:]

    if not sections.get("Constraints"):
        sections["Constraints"] = ["Use only retrieved Syracuse corpus context."]
    if not sections.get("Open questions"):
        sections["Open questions"] = ["(none)"]

    for sec in _SUMMARY_SECTIONS:
        vals = [v for v in sections.get(sec, []) if v and v.strip() and v.strip() != "(none)"]
        cap = 8 if sec == "Researcher mentions" else max_items_per_field
        sections[sec] = vals[-cap:]

    limit = max(1, int(getattr(settings, "summary_max_chars", 1800)))
    summary = _format_summary_sections(sections)
    for _ in range(256):
        if len(summary) <= limit:
            break
        changed = False
        for key in ("Core entities", "Key themes", "Open questions", "Current focus"):
            vals = sections.get(key, [])
            if len(vals) > 1:
                sections[key] = vals[1:]
                changed = True
                break
        if not changed:
            break
        summary = _format_summary_sections(sections)

    if len(summary) > limit:
        summary = summary[:limit].rstrip()
    if summary.strip():
        return summary
    return (previous_summary or "").strip() or _summary_template_empty()

_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-invoke")
_vram_release_lock = threading.Lock()
_vram_release_counter = 0
_VRAM_RELEASE_EVERY_N = 5

def _release_vram_cache() -> None:
    global _vram_release_counter
    with _vram_release_lock:
        _vram_release_counter += 1
        should_release = _vram_release_counter >= _VRAM_RELEASE_EVERY_N
        if should_release:
            _vram_release_counter = 0
    if should_release:
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def _invoke_with_timeout(llm: Any, prompt: str, timeout_s: int) -> str:
    if timeout_s <= 0:
        result = str(llm.invoke(prompt) or "")
        _release_vram_cache()
        return result
    fut = _LLM_EXECUTOR.submit(llm.invoke, prompt)
    try:
        result = str(fut.result(timeout=timeout_s) or "")
    except FuturesTimeout:
        logger.warning("LLM invoke timed out after %ds", timeout_s)
        result = ""
    _release_vram_cache()
    return result

def _regenerate_rolling_summary(*, llm, old_summary, new_turns_text, question,
                                ner_line, source_context, no_results, timeout_s) -> str:
    prompt_turns = (new_turns_text or "").strip()
    if no_results:
        line = _no_results_summary_line(question)
        prompt_turns = (prompt_turns + "\n" + line).strip() if prompt_turns else line

    m = re.search(r"ASSISTANT:\s*(.+)", new_turns_text or "", re.IGNORECASE | re.DOTALL)
    answer_snippet = re.sub(r"\s+", " ", m.group(1)).strip() if m else re.sub(r"\s+", " ", new_turns_text or "").strip()

    if prompt_turns:
        try:
            msgs = SUMMARY_PROMPT.format_messages(old_summary=old_summary, new_turns=prompt_turns)
            candidate = _invoke_with_timeout(llm, msgs[0].content + "\n" + msgs[1].content, timeout_s).strip()
            if candidate:
                old_summary = candidate
        except Exception:
            logger.warning("LLM summary regeneration failed", exc_info=True)

    return build_rolling_summary(old_summary, question,
                                 " ".join([source_context or "", ner_line or ""]).strip(),
                                 answer_snippet)

_ANCHOR_ESCAPE_PATTERN = re.compile(
    r"\b(who else|what else|other (researchers?|faculty|people|authors?|scientists?)"
    r"|others|besides|apart from|different (from|researcher|faculty)|not .{0,30} but"
    r"|also (stud(y|ies)|work|research)|anyone else|somebody else|someone else"
    r"|studies this (topic|area|field|subject)|works on this)\b",
    re.IGNORECASE,
)

def _is_anchor_escape_question(question: str) -> bool:
    return bool(_ANCHOR_ESCAPE_PATTERN.search(question or ""))

def _looks_like_person_token(token: str) -> bool:
    return bool(token and (re.match(r"^[A-Z][A-Za-z\-']+$", token) or re.match(r"^[A-Z]\.$", token)))

def _extract_entities_regex(raw: str, *, max_items: int = 6) -> Dict[str, List[str]]:
    people, entities = [], []
    for quoted in re.findall(r"\"([^\"]{3,120})\"", raw):
        entities.append(quoted.strip())
    for span in re.compile(r"\b(?:[A-Z][A-Za-z'\-\.]*\s+){1,5}[A-Z][A-Za-z'\-\.]*\b").findall(raw):
        toks = [t for t in re.split(r"\s+", span.strip()) if t]
        if 2 <= len(toks) <= 4 and all(_looks_like_person_token(t) for t in toks):
            people.append(span.strip())
        else:
            entities.append(span.strip())
    for acr in re.findall(r"\b[A-Z]{2,10}\b", raw):
        entities.append(acr)

    people = _dedupe_ci(people)[:max_items]
    entities = _dedupe_ci(entities)[:max_items]
    blocked_tokens = set(_tokenize_words(" ".join(people + entities)))
    tokens = [t for t in _tokenize_words(raw) if len(t) >= 4 and t not in blocked_tokens and not t.isdigit()]
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    topics = [k for k, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))][:max_items]
    orgs = _dedupe_ci([e for e in entities if re.fullmatch(r"[A-Z]{2,10}", e)])[:max_items]
    return {"people": people, "orgs": orgs, "entities": entities, "topics": topics}

def _extract_entities_nltk(raw: str, *, max_items: int = 6) -> Optional[Dict[str, List[str]]]:
    if not all((nltk, word_tokenize, pos_tag, ne_chunk)):
        return None

    def _run():
        tokens = word_tokenize(raw)
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged, binary=False)
        people, orgs, entities = [], [], []
        for node in tree:
            if Tree is not None and isinstance(node, Tree):
                label = str(node.label() or "").upper()
                phrase = " ".join(tok for tok, _ in node.leaves()).strip()
                if not phrase:
                    continue
                if label == "PERSON":
                    people.append(phrase)
                elif label in {"ORGANIZATION", "GPE", "FACILITY"}:
                    orgs.append(phrase)
                    entities.append(phrase)
                else:
                    entities.append(phrase)
        for tok, pos_ in tagged:
            if re.fullmatch(r"[A-Z]{2,10}", tok):
                orgs.append(tok); entities.append(tok)
            elif pos_ == "NNP" and re.fullmatch(r"[A-Z][A-Za-z\-']+", tok):
                entities.append(tok)
        for quoted in re.findall(r"\"([^\"]{3,120})\"", raw):
            entities.append(quoted.strip())
        stopset = _get_stopword_set()
        blocked = set(_tokenize_words(" ".join(people + orgs + entities)))
        counts: Dict[str, int] = {}
        for tok in tokens:
            if not re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok):
                continue
            t = tok.lower()
            if len(t) >= 4 and t not in stopset and t not in blocked:
                counts[t] = counts.get(t, 0) + 1
        topics = [k for k, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))][:max_items]
        return {
            "people": _dedupe_ci(people)[:max_items],
            "orgs": _dedupe_ci(orgs)[:max_items],
            "entities": _dedupe_ci(entities)[:max_items],
            "topics": topics,
        }

    try:
        return _run()
    except LookupError:
        _bootstrap_nltk_data()
        try:
            return _run()
        except Exception:
            return None
    except Exception:
        return None

def _extract_entities_basic(text: str, *, max_items: int = 6) -> Dict[str, List[str]]:
    raw = (text or "").strip()
    if not raw:
        return {"people": [], "orgs": [], "entities": [], "topics": []}
    out = _extract_entities_nltk(raw, max_items=max_items) or _extract_entities_regex(raw, max_items=max_items)
    return {
        "people": _sanitize_entity_values(out.get("people") or [], max_items=max_items),
        "orgs": _sanitize_entity_values(out.get("orgs") or [], max_items=max_items),
        "entities": _sanitize_entity_values(out.get("entities") or [], max_items=max_items),
        "topics": _dedupe_ci([t for t in (out.get("topics") or [])
                              if t and not _is_generic_query_token(t)])[:max_items],
    }

def _build_ner_context_text(docs: List[Document], max_docs: int = 12) -> str:
    parts: List[str] = []
    for d in docs[:max_docs]:
        meta = d.metadata or {}
        researcher = re.sub(r"\s+", " ", str(meta.get("researcher", "") or "").strip())
        if researcher and _looks_like_person_candidate(researcher):
            parts.append(f"researcher:{researcher}")

        raw_authors = re.sub(r"\s+", " ", str(meta.get("authors", "") or "").strip())
        if raw_authors:
            for author in split_author_names(raw_authors)[:6]:
                candidate = _strip_possessive(author)
                if candidate and _looks_like_person_candidate(candidate):
                    parts.append(f"author:{candidate}")

        topic = re.sub(r"\s+", " ", str(meta.get("primary_topic", "") or "").strip())
        if topic and not _is_placeholder_anchor_value(topic):
            toks = [t for t in _tokenize_words(topic)
                    if len(t) >= 3 and not _is_generic_query_token(t)]
            if len(toks) >= 2:
                parts.append(f"topic:{topic}")
    return " | ".join(parts)

def _extract_summary_topic_keywords(summary: str, *, max_chars: int = 180) -> str:
    if not summary:
        return ""
    themes = _extract_summary_sections(summary).get("Key themes", [])
    individual = []
    for t in themes:
        if not t or t.strip() == "(none)":
            continue
        for sub in re.split(r"\s*\|\s*", t):
            sub = sub.strip()
            if sub and not _is_generic_query_token(sub) and len(sub) >= 4:
                individual.append(sub)
    parts = _dedupe_ci(individual)[:4]
    result = " ".join(parts).strip()
    return result[:max_chars].rstrip() if len(result) > max_chars else result

def _summary_query_from_text(summary: str, *, max_chars: int = 320) -> str:
    if not summary:
        return ""
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = " ".join(lines[-3:])
    return tail[:max_chars - 1].rstrip() + "…" if len(tail) > max_chars else tail

def _summary_keywords_overlap_anchor(topic_keywords: str, anchor_value: str) -> bool:
    """Return True if the summary topic keywords share at least one meaningful
    token with the anchor value.  This prevents injecting stale summary keywords
    from prior conversation topics into the retrieval query."""
    if not topic_keywords or not anchor_value:
        return False
    a_toks = set(_tokenize_words(anchor_value))
    kw_toks = set(_tokenize_words(topic_keywords))
    a_toks = {t for t in a_toks if len(t) >= 4 and not _is_generic_query_token(t)}
    kw_toks = {t for t in kw_toks if len(t) >= 4 and not _is_generic_query_token(t)}
    if not a_toks or not kw_toks:
        return False
    return bool(a_toks & kw_toks)

def _extract_person_name(question: str) -> str:
    raw = (question or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"(\w)['\\u2019]s\b", r"\1", raw)
    stopset = _get_stopword_set()

    _epn_name_tokens = _get_name_token_set()
    _epn_english_words = _get_english_word_set()

    def _titlecase_token(token: str) -> str:
        tok = _strip_possessive(token or "")
        if re.fullmatch(r"[A-Za-z]\.", tok):
            return tok.upper()
        return tok.capitalize() if tok.islower() else tok

    def _normalize_candidate(candidate: str) -> str:
        parts = [_titlecase_token(tok) for tok in re.findall(r"[A-Za-z][A-Za-z\.\-']*", candidate or "")]
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def _accept(candidate: str) -> str:
        c = _strip_possessive(_normalize_candidate(candidate))
        if not c or not _looks_like_person_candidate(c):
            return ""
        return c

    raw_tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z\.\-']*", cleaned) if t]
    if 2 <= len(raw_tokens) <= 4:
        candidate = _accept(" ".join(raw_tokens))
        if candidate:
            return candidate

    def _neighbor_score(token: str) -> int:
        if not token:
            return 2
        bare = _strip_possessive(token).rstrip(".").lower()
        if not bare:
            return 1
        if (stopset and bare in stopset) or _is_generic_query_token(bare):
            return 2
        if len(bare) <= 2:
            return 1
        return 0

    ranked: List[Tuple[float, int, int, str]] = []
    max_width = min(4, len(raw_tokens))
    for width in range(2, max_width + 1):
        for start in range(0, len(raw_tokens) - width + 1):
            candidate = _accept(" ".join(raw_tokens[start:start + width]))
            if not candidate:
                continue
            prev_tok = raw_tokens[start - 1] if start > 0 else ""
            next_tok = raw_tokens[start + width] if start + width < len(raw_tokens) else ""
            initial_bonus = 1 if any(re.fullmatch(r"[A-Za-z]\.", _titlecase_token(tok))
                                     for tok in raw_tokens[start:start + width]) else 0
            width_score = {2: 6, 3: 4, 4: 2}.get(width, 0)
            score = float(width_score + _neighbor_score(prev_tok) + _neighbor_score(next_tok) + initial_bonus)
            ranked.append((score, width, start, candidate))

    if ranked:
        ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3].lower()))
        return ranked[0][3]
    return ""

_DEGENERATE_ANSWER_PATTERNS = re.compile(
    r"^\s*(?:[\w\s\.\,\-]+ is a researcher[\.\s]*"
    r"|[\w\s\.\,\-]+ is an? \w+[\.\s]*"
    r"|I (could not|couldn't|cannot|can't) (find|generate|provide|answer))\s*$",
    re.IGNORECASE,
)

def _answer_is_bad(answer: str) -> bool:
    a = (answer or "").strip()
    return not a or len(_tokenize_words(a)) < 10 or bool(_DEGENERATE_ANSWER_PATTERNS.match(a))

def _extract_focus_from_question(question: str) -> str:
    raw = (question or "").strip()
    if not raw:
        return ""
    ents = _extract_entities_basic(raw, max_items=6)
    for key, limit in [("topics", 4), ("entities", 3), ("orgs", 2)]:
        vals = [v for v in ents.get(key, []) if not _is_generic_query_token(v)]
        if vals:
            return " ".join(vals[:limit])
    return ""

def _is_invalid_focus_value(text: str) -> bool:
    toks = [t for t in _tokenize_words(text) if t]
    if not toks or max((len(t) for t in toks), default=0) < 3:
        return True
    return not [t for t in toks if not _is_generic_query_token(t)]

def _query_is_short_or_pronoun(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _is_followup_coref_question(q):
        return True
    return len(_tokenize_words(q)) < max(1, int(getattr(settings, "followup_query_max_words", 8)))

def _inject_anchor_into_query(question: str, anchor_value: str) -> str:
    q, anchor = collapse_whitespace(question), collapse_whitespace(anchor_value)
    if not q or not anchor or _is_placeholder_anchor_value(anchor) or _anchor_in_text(anchor, q):
        return q
    pronoun_re = re.compile(
        r"\b(him|her|them|they|it|this|that|those|these|he|she|his|hers|their|there)\b",
        re.IGNORECASE,
    )
    replaced = collapse_whitespace(pronoun_re.sub(anchor, q, count=1))
    if _anchor_in_text(anchor, replaced):
        return replaced
    stem = q.rstrip(" ?")
    return f"{stem} for {anchor}?" if stem else anchor

class _DirectGenerationLLM:
    def __init__(self, *, model, tokenizer, generation_kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = dict(generation_kwargs or {})
        try:
            self._device = next(model.parameters()).device
        except StopIteration:
            self._device = torch.device("cpu")

    SYSTEM_USER_SEP = "\n<<SYS_USER_BOUNDARY>>\n"

    def invoke(self, prompt: str) -> str:
        p = str(prompt or "")
        if not p:
            return ""
        max_new = int(self.generation_kwargs.get("max_new_tokens", 256))
        # Gemma 3/4 (and some other models) ship with model_max_length and/or
        # max_position_embeddings set to a sentinel (~2^63) meaning "unlimited".
        # Passing such a value to the Rust-backed fast tokenizer's truncation
        # overflows native usize and raises OverflowError. Clamp to a safe upper
        # bound (1M tokens is safe on all backends and larger than any real prompt).
        _TOK_MAX_SAFE = 1_000_000
        _raw_ctx = (getattr(self.model.config, "max_position_embeddings", 0) or
                    getattr(self.tokenizer, "model_max_length", 4096) or 4096)
        try:
            max_ctx = int(_raw_ctx)
        except (TypeError, ValueError, OverflowError):
            max_ctx = 4096
        if max_ctx <= 0 or max_ctx > _TOK_MAX_SAFE:
            max_ctx = 8192

        formatted_prompt = p
        _has_chat_template = (
            hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None) is not None
        )
        if _has_chat_template:
            try:
                if self.SYSTEM_USER_SEP in p:
                    sys_part, user_part = p.split(self.SYSTEM_USER_SEP, 1)
                    messages = [
                        {"role": "system", "content": sys_part.strip()},
                        {"role": "user", "content": user_part.strip()},
                    ]
                else:
                    messages = [{"role": "user", "content": p}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = p

        enc = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True,
                             max_length=max(64, max_ctx - max_new - 16))
        enc = {k: v.to(self._device) for k, v in enc.items()}

        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": bool(self.generation_kwargs.get("do_sample", False)),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.15, "use_cache": True, "return_dict_in_generate": False,
        }
        if gen_kwargs["do_sample"]:
            for key in ("temperature", "top_p"):
                val = self.generation_kwargs.get(key)
                if val is not None:
                    gen_kwargs[key] = float(val)

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)
            prompt_len = int(enc["input_ids"].shape[1])
            result = self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

        del enc, out
        return result

def _needs_trust_remote_code(model_id_or_path: str, local_only: bool) -> bool:
    """Return True only for local models that bundle custom tokenizer/model classes.

    ``trust_remote_code=True`` tells the transformers library to execute the
    custom Python files that shipped with the model when it was downloaded.
    Because all models in this deployment are loaded from local disk, this
    carries no network-fetch risk — it only runs code that was already
    reviewed and placed on disk.

    We still restrict it to an explicit allowlist so that adding a new model
    path can never silently enable custom-code execution without a deliberate
    decision to add it here.

    Override via ``RAG_TRUST_REMOTE_CODE=1`` as an audited escape hatch.
    """
    if os.getenv("RAG_TRUST_REMOTE_CODE", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if not local_only:
        return False
    path = str(model_id_or_path or "").strip()
    path_is_local = (
        os.path.sep in path
        or path.startswith("/")
        or (len(path) > 2 and path[1] == ":")
        or os.path.isdir(path)
    )
    if not path_is_local:
        return False
    _TRUST_REMOTE_CODE_ALLOWLIST = {
        "qwen-2.5-14b", "qwen_14b", "14b",
        "gpt-oss-20b", "gpt_oss_20b", "20b",
        "gemma-4",
    }
    basename = os.path.basename(path.rstrip("/\\")).lower()
    return any(allowed in basename for allowed in _TRUST_REMOTE_CODE_ALLOWLIST)

class ModelRuntime:
    def __init__(self, model_id_or_path: str, *, max_new_tokens: int,
                 do_sample: bool = False, temperature: float = 0.0, top_p: Optional[float] = None,
                 load_in_8bit: bool = False, quantize_bits: int = 0, local_only: bool = True):
        if getattr(settings, "force_gpu", True) and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is False")

        self.model_id_or_path = model_id_or_path
        self.quantize_bits = quantize_bits if quantize_bits else (8 if load_in_8bit else 0)
        trust_rc = _needs_trust_remote_code(model_id_or_path, local_only)

        with torch.no_grad():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id_or_path, local_files_only=local_only, use_fast=True,
                    trust_remote_code=trust_rc)
            except Exception as _tok_err:
                if "ModelWrapper" in str(_tok_err) or "untagged enum" in str(_tok_err):
                    logger.warning(
                        "Fast tokenizer failed for %s (likely version mismatch), "
                        "falling back to slow tokenizer: %s", model_id_or_path, _tok_err)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_id_or_path, local_files_only=local_only, use_fast=False,
                        trust_remote_code=trust_rc)
                else:
                    raise
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Clamp tokenizer.model_max_length to prevent OverflowError from
            # the Rust tokenizer when truncation=True is used. Gemma 3/4 ship
            # with sentinel values (~2^63) that overflow native usize.
            try:
                _cur_max_len = int(getattr(self.tokenizer, "model_max_length", 0))
            except (TypeError, ValueError, OverflowError):
                _cur_max_len = 0
            if _cur_max_len <= 0 or _cur_max_len > 1_000_000:
                try:
                    self.tokenizer.model_max_length = 1_000_000
                except Exception:
                    pass

            _attn_impl = "eager"
            _gpu_sm = 0
            if torch.cuda.is_available():
                try:
                    _gpu_sm = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
                except Exception:
                    pass

            if _gpu_sm >= 80:
                try:
                    import flash_attn
                    _attn_impl = "flash_attention_2"
                except ImportError:
                    pass

            if _attn_impl == "eager":
                if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                    _attn_impl = "sdpa"

            logger.info("Loading %s | attn=%s | gpu_sm=%d | quant=%dbit",
                        model_id_or_path, _attn_impl, _gpu_sm, self.quantize_bits)
            print(f"[MODEL_LOAD] {model_id_or_path} | attn={_attn_impl} | "
                  f"gpu_sm={_gpu_sm} | quant={self.quantize_bits}bit")

            _common_kwargs = dict(
                local_files_only=local_only,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_rc,
            )
            if _attn_impl != "eager":
                _common_kwargs["attn_implementation"] = _attn_impl

            if self.quantize_bits in (4, 8) and torch.cuda.is_available():
                # --- Guard: detect poisoned CUDA context from a prior model ---
                try:
                    torch.cuda.synchronize()
                    _probe = torch.tensor([1.0], device="cuda")
                    _ = _probe + _probe
                    torch.cuda.synchronize()
                    del _probe
                except RuntimeError as _cuda_err:
                    raise RuntimeError(
                        f"CUDA is in a bad state before loading {model_id_or_path}. "
                        f"A previous model likely triggered a device-side assert. "
                        f"Restart the process to reset the GPU. Error: {_cuda_err}"
                    ) from _cuda_err

                # --- Guard: detect models shipped with a native quantization
                # config (e.g. MXFP4, GPTQ, AWQ).  These MUST be loaded with
                # their own config — passing a BitsAndBytesConfig will fail
                # with a "you are passing a BitsAndBytesConfig" error.
                # We inspect config.json before loading the full model.
                _native_quant_config = None
                try:
                    _cfg_path = model_id_or_path
                    if os.path.isdir(_cfg_path):
                        import json as _json
                        _config_file = os.path.join(_cfg_path, "config.json")
                        if os.path.isfile(_config_file):
                            with open(_config_file, "r", encoding="utf-8") as _f:
                                _model_cfg = _json.load(_f)
                            _qc = _model_cfg.get("quantization_config")
                            if isinstance(_qc, dict) and _qc.get("quant_type"):
                                _native_quant_config = _qc
                                logger.info(
                                    "Detected native quantization in %s: %s",
                                    model_id_or_path, _qc.get("quant_type"))
                                print(f"[MODEL_LOAD] Native quant detected: "
                                      f"{_qc.get('quant_type')} — skipping BnB")
                except Exception as _cfg_err:
                    logger.debug("Could not inspect config.json: %s", _cfg_err)

                if _native_quant_config is not None:
                    # Model is already quantized (MXFP4, GPTQ, AWQ, etc.)
                    # Load it directly — transformers will use its native config.
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_id_or_path,
                            device_map="auto",
                            **_common_kwargs,
                        )
                        _native_type = _native_quant_config.get("quant_type", "unknown")
                        logger.info("Loaded %s with native %s quantization",
                                    model_id_or_path, _native_type)
                        print(f"[MODEL_LOAD] Loaded {model_id_or_path} with "
                              f"native {_native_type} quantization")
                    except TypeError:
                        _fallback = {k: v for k, v in _common_kwargs.items()
                                     if k != "attn_implementation"}
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_id_or_path,
                            device_map="auto",
                            **_fallback,
                        )

                elif _native_quant_config is None:
                    # --- Gemma safety: NF4 4-bit causes CUDA device-side assert
                    # on SM < 8.0 (Turing).  Disabling double-quant alone is not
                    # sufficient — the NF4 kernel itself triggers the assert.
                    # Use FP4 (still 4-bit, but Turing-compatible) instead of NF4.
                    _basename_lower = os.path.basename(
                        model_id_or_path.rstrip("/\\")).lower()
                    _is_gemma = "gemma" in _basename_lower
                    _force_fp4 = (_is_gemma and _gpu_sm < 80 and self.quantize_bits == 4)
                    if _force_fp4:
                        logger.warning(
                            "Gemma on SM %d (< 80): NF4 4-bit causes CUDA "
                            "device-side assert on Turing. Using FP4 instead.",
                            _gpu_sm)
                        print(f"[MODEL_LOAD] Gemma on SM {_gpu_sm}: "
                              f"using FP4 quant type (NF4 incompatible with Turing)")

                    try:
                        from transformers import BitsAndBytesConfig
                        import bitsandbytes
                        if self.quantize_bits == 4:
                            # SM 7.5 (Turing) can hit CUDA assertions with NF4 +
                            # double-quant on some architectures (notably Gemma).
                            # Use double-quant only on SM >= 8.0 (Ampere+).
                            # Use FP4 quant type for Gemma on pre-Ampere GPUs.
                            _use_dq = (_gpu_sm >= 80) and not _force_fp4
                            _quant_type = "fp4" if _force_fp4 else "nf4"
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type=_quant_type,
                                bnb_4bit_use_double_quant=_use_dq,
                            )
                        else:
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_id_or_path,
                                quantization_config=quantization_config,
                                device_map="auto",
                                **_common_kwargs,
                            )
                        except TypeError:
                            _fallback = {k: v for k, v in _common_kwargs.items()
                                         if k != "attn_implementation"}
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_id_or_path,
                                quantization_config=quantization_config,
                                device_map="auto",
                                **_fallback,
                            )
                        logger.info("Loaded %s in %d-bit mode", model_id_or_path, self.quantize_bits)
                    except (ImportError, Exception) as e:
                        _err_str = str(e)
                        logger.warning("%d-bit loading failed (%s) — attempting fallback",
                                       self.quantize_bits, e)
                        print(f"[MODEL_LOAD] {self.quantize_bits}-bit failed: {_err_str[:120]}")

                        _model_loaded = False

                        # --- Try to recover CUDA state before retrying ---
                        if "assert" in _err_str.lower() or "cuda" in _err_str.lower():
                            try:
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            except Exception:
                                pass

                        # --- Retry 4-bit with FP4 quant type (more compatible on SM 7.x) ---
                        # Skip if FP4 was already the primary strategy (_force_fp4).
                        if self.quantize_bits == 4 and not _model_loaded and not _force_fp4:
                            try:
                                print("[MODEL_LOAD] Trying 4-bit with fp4 quant type...")
                                _fp4_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_quant_type="fp4",
                                    bnb_4bit_use_double_quant=False,
                                )
                                _fb_kwargs = {k: v for k, v in _common_kwargs.items()
                                              if k != "attn_implementation"}
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_id_or_path,
                                    quantization_config=_fp4_config,
                                    device_map="auto",
                                    **_fb_kwargs,
                                )
                                _model_loaded = True
                                print(f"[MODEL_LOAD] Loaded {model_id_or_path} in 4-bit fp4 (fallback from nf4)")
                            except Exception as e_fp4:
                                print(f"[MODEL_LOAD] 4-bit fp4 also failed: {str(e_fp4)[:120]}")

                        if self.quantize_bits == 4 and not _model_loaded:
                            try:
                                print("[MODEL_LOAD] Trying 8-bit fallback...")
                                _8bit_config = BitsAndBytesConfig(load_in_8bit=True)
                                _fb_kwargs = {k: v for k, v in _common_kwargs.items()
                                              if k != "attn_implementation"}
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_id_or_path,
                                    quantization_config=_8bit_config,
                                    device_map="auto",
                                    **_fb_kwargs,
                                )
                                self.quantize_bits = 8
                                _model_loaded = True
                                print(f"[MODEL_LOAD] Loaded {model_id_or_path} in 8-bit (fallback from 4-bit)")
                            except Exception as e2:
                                logger.warning("8-bit fallback also failed: %s", e2)
                                print(f"[MODEL_LOAD] 8-bit fallback also failed: {str(e2)[:120]}")

                        if not _model_loaded:
                            try:
                                print("[MODEL_LOAD] Trying fp16 with device_map=auto...")
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_id_or_path, torch_dtype=torch.float16,
                                    device_map="auto", **_common_kwargs)
                                _model_loaded = True
                            except TypeError:
                                _fallback = {k: v for k, v in _common_kwargs.items()
                                             if k != "attn_implementation"}
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    model_id_or_path, torch_dtype=torch.float16,
                                    device_map="auto", **_fallback)
                                _model_loaded = True
                            except Exception as e3:
                                _torch_ver = getattr(torch, "__version__", "unknown")
                                _bnb_ver = "unknown"
                                try:
                                    _bnb_ver = bitsandbytes.__version__
                                except Exception:
                                    pass
                                _tf_ver = "unknown"
                                try:
                                    import transformers as _tf
                                    _tf_ver = _tf.__version__
                                except Exception:
                                    pass
                                print(f"[MODEL_LOAD] All loading strategies failed for {model_id_or_path}")
                                print(f"[MODEL_LOAD] Versions: torch={_torch_ver}, "
                                      f"bitsandbytes={_bnb_ver}, transformers={_tf_ver}")
                                print(f"[MODEL_LOAD] For 4-bit: requires torch>=2.1, "
                                      f"bitsandbytes>=0.41, transformers>=4.36")
                                raise RuntimeError(
                                    f"Failed to load {model_id_or_path} with any quantization "
                                    f"strategy (4-bit, 8-bit, fp16). Original error: {_err_str}"
                                ) from e3
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                _model_loaded = False
                _actual_precision = "native"

                if torch.cuda.is_available():
                    # --- STEP 1: Try native (fp16/bf16) ---
                    try:
                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_id_or_path, torch_dtype=dtype,
                                device_map="auto", **_common_kwargs)
                        except TypeError:
                            _fallback = {k: v for k, v in _common_kwargs.items()
                                         if k != "attn_implementation"}
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_id_or_path, torch_dtype=dtype,
                                device_map="auto", **_fallback)
                        _model_loaded = True
                        _actual_precision = "fp16"
                        print(f"[MODEL_LOAD] Loaded {model_id_or_path} at native fp16 precision")
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as _oom:
                        _oom_msg = str(_oom)
                        if "out of memory" in _oom_msg.lower() or "CUDA" in _oom_msg:
                            print(f"[MODEL_LOAD] Native fp16 OOM for {model_id_or_path} — trying 8-bit...")
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            raise

                    # --- STEP 2: Try 8-bit ---
                    if not _model_loaded:
                        try:
                            from transformers import BitsAndBytesConfig
                            _8bit_config = BitsAndBytesConfig(load_in_8bit=True)
                            _fb_kwargs = {k: v for k, v in _common_kwargs.items()
                                          if k != "attn_implementation"}
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_id_or_path,
                                quantization_config=_8bit_config,
                                device_map="auto", **_fb_kwargs)
                            self.quantize_bits = 8
                            _model_loaded = True
                            _actual_precision = "8bit"
                            print(f"[MODEL_LOAD] Loaded {model_id_or_path} in 8-bit (fallback from native)")
                        except (RuntimeError, torch.cuda.OutOfMemoryError, Exception) as _e8:
                            print(f"[MODEL_LOAD] 8-bit also failed: {str(_e8)[:120]} — trying 4-bit...")
                            gc.collect()
                            torch.cuda.empty_cache()

                    # --- STEP 3: Try 4-bit ---
                    if not _model_loaded:
                        try:
                            from transformers import BitsAndBytesConfig
                            _4bit_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                            )
                            _fb_kwargs = {k: v for k, v in _common_kwargs.items()
                                          if k != "attn_implementation"}
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_id_or_path,
                                quantization_config=_4bit_config,
                                device_map="auto", **_fb_kwargs)
                            self.quantize_bits = 4
                            _model_loaded = True
                            _actual_precision = "4bit"
                            print(f"[MODEL_LOAD] Loaded {model_id_or_path} in 4-bit (fallback from 8-bit)")
                        except Exception as _e4:
                            print(f"[MODEL_LOAD] All precision levels failed for {model_id_or_path}")
                            raise RuntimeError(
                                f"Failed to load {model_id_or_path} at any precision "
                                f"(native fp16, 8-bit, 4-bit). Last error: {_e4}"
                            ) from _e4

                    # Store the actual precision used for reporting
                    self._actual_precision = _actual_precision

                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id_or_path, torch_dtype=dtype,
                        device_map=None, low_cpu_mem_usage=False,
                        local_files_only=local_only,
                        trust_remote_code=trust_rc)
                    self._actual_precision = "cpu_fp32"

        self.model.eval()

        self.generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample), "return_full_text": False,
        }
        if do_sample:
            self.generation_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                self.generation_kwargs["top_p"] = float(top_p)
        if not do_sample:
            try:
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
            except Exception:
                pass

        self.llm = _DirectGenerationLLM(model=self.model, tokenizer=self.tokenizer,
                                        generation_kwargs=self.generation_kwargs)

        self._warmup()

    def _warmup(self) -> None:
        """Run a minimal forward pass to pre-compile CUDA kernels
        and validate the model can actually produce tokens."""
        try:
            dummy = self.tokenizer("The answer is", return_tensors="pt",
                                   truncation=True, max_length=8)
            target_device = next(self.model.parameters()).device
            dummy = {k: v.to(target_device) for k, v in dummy.items()}
            with torch.inference_mode():
                out = self.model.generate(
                    **dummy, max_new_tokens=4,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # --- Validate: did the model produce any new tokens? ---
            prompt_len = int(dummy["input_ids"].shape[1])
            new_tokens = out[0][prompt_len:]
            del dummy

            if torch.cuda.is_available():
                torch.cuda.synchronize()  # surface async CUDA errors

            if len(new_tokens) == 0:
                raise RuntimeError(
                    f"Model {self.model_id_or_path} produced 0 new tokens during "
                    f"warmup — the GPU is likely in a bad state (CUDA device-side "
                    f"assert). Restart the process to reset the GPU context."
                )
            del out
            logger.info("Warmup OK — model generated %d tokens", len(new_tokens))

        except RuntimeError:
            raise  # propagate validation failures
        except Exception:
            logger.debug("Model warmup failed (non-fatal)", exc_info=True)

    def close(self) -> None:
        model = getattr(self, "model", None)
        for attr in ("llm", "model", "tokenizer"):
            try:
                delattr(self, attr)
            except Exception:
                pass
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

def _fast_count_tokens(text: str) -> int:
    """Character-based token estimate: ~4 chars per token.
    Used inside pack_docs during retrieval so we never call the LLM tokenizer
    on every doc — the real tokenizer is slow on large models (8B+) and was
    the sole reason retrieval appeared slower with bigger models.
    The estimate is intentionally conservative (rounds up) so budget enforcement
    stays safe.  Accuracy within ±15% is sufficient here; _fit_prompt_to_budget
    uses the real tokenizer later when building the final prompt.
    """
    return max(1, (len(text) + 3) // 4)

def pack_docs(docs: List[Document], budget: int, count_tokens_fn) -> List[Document]:
    MIN_DOCS = 8
    out, total = [], 0
    for d in docs:
        meta = getattr(d, "metadata", None) or {}
        researcher = str(meta.get("researcher", "") or "").strip()
        summary_raw = ""
        for skey in ("summary", "abstract", "description"):
            candidate = str(meta.get(skey, "") or "").strip()
            if candidate and len(candidate) > 20:
                summary_raw = candidate[:600]
                break
        formatted = "\n".join(filter(None, [
            f"title: {str(meta.get('title', '') or '')}",
            f"researcher: {researcher}" if researcher else None,
            f"authors: {str(meta.get('authors', '') or '')}",
            f"year: {str(meta.get('year', meta.get('publication_date', '')) or '')}",
            f"summary: {summary_raw}" if summary_raw else None,
            str(d.page_content or "")[:500],
        ]))
        t = _fast_count_tokens(formatted)
        if total + t > budget and len(out) >= MIN_DOCS:
            break
        out.append(d)
        total += t
    return out

format_docs_compact = build_compact_context

class _TransformerMeanEmbeddings:
    def __init__(self, model_name: str, device: str) -> None:
        import transformers as _tf
        _tf_version = tuple(int(x) for x in _tf.__version__.split(".")[:2])
        _dtype_kwarg = "dtype" if _tf_version >= (4, 49) else "torch_dtype"

        last_err: Optional[Exception] = None
        for local_only in (True, False):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=local_only, use_fast=True,
                    trust_remote_code=False)
                self.model = AutoModel.from_pretrained(
                    model_name, local_files_only=local_only,
                    trust_remote_code=False,
                    **{_dtype_kwarg: torch.float32})
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if not local_only:
                    raise
                logger.info("Embedding model not available locally (%s), retrying with download …", exc)
        if last_err is not None:
            raise last_err
        self.model.to(device).eval()
        self.device = device

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out = []
        for i in range(0, len(texts), 32):
            batch = [str(t or "") for t in texts[i:i + 32]]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = torch.nn.functional.normalize(
                    (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9), p=2, dim=1)
                out.extend(pooled.detach().tolist())
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        vecs = self._encode_texts([text])
        return vecs[0] if vecs else []

class _BertFallbackEmbeddings:
    """Last-resort embedder that imports BertModel / BertTokenizer directly.

    Newer transformers versions sometimes fail to resolve 'BertModel' via
    AutoModel when loading from a local cache.  Importing the concrete class
    from ``transformers.models.bert`` bypasses the auto-resolution machinery.
    """

    def __init__(self, model_name: str, device: str) -> None:
        from transformers.models.bert.modeling_bert import BertModel as _BertModel
        from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast as _Tok

        for local_only in (True, False):
            try:
                self.tokenizer = _Tok.from_pretrained(
                    model_name, local_files_only=local_only)
                self.model = _BertModel.from_pretrained(
                    model_name, local_files_only=local_only)
                break
            except Exception:
                if not local_only:
                    raise
        self.model.to(device).eval()
        self.device = device

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out = []
        for i in range(0, len(texts), 32):
            batch = [str(t or "") for t in texts[i:i + 32]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = torch.nn.functional.normalize(
                    (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9),
                    p=2, dim=1)
                out.extend(pooled.detach().tolist())
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        vecs = self._encode_texts([text])
        return vecs[0] if vecs else []

def build_embeddings() -> Any:
    model_name = config.EMBED_MODEL
    errors: List[str] = []

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "trust_remote_code": False},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
        )
    except Exception as e:
        errors.append(f"HuggingFaceEmbeddings: {e}")
        logger.warning("Embedding init attempt failed (HuggingFaceEmbeddings): %s", e)

    try:
        return _TransformerMeanEmbeddings(model_name, "cpu")
    except Exception as e:
        errors.append(f"AutoModel: {e}")
        logger.warning("Embedding init attempt failed (AutoModel): %s", e)

    try:
        return _BertFallbackEmbeddings(model_name, "cpu")
    except Exception as e:
        errors.append(f"BertModel direct: {e}")
        logger.warning("Embedding init attempt failed (BertModel direct): %s", e)

    raise RuntimeError(
        f"All embedding initialization strategies failed for '{model_name}':\n"
        + "\n".join(f"  - {err}" for err in errors)
    )

def clear_runtime_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _resolve_llm_path(llm_model_key: str) -> str:
    key = (llm_model_key or "").strip().lower()
    if key in {"llama-3.2-1b", "llama_1b", "1b"}:
        return str(getattr(settings, "llama_1b_path", "") or "").strip() or config.LLAMA_1B
    if key in {"llama-3.1-8b", "llama_8b", "8b"}:
        return str(getattr(settings, "llama_8b_path", "") or "").strip() or getattr(config, "LLAMA_8B", "")
    if key in {"gemma-4-e2b", "gemma_4_e2b"}:
        return str(getattr(settings, "gemma_4_e2b_path", "") or "").strip() or getattr(config, "GEMMA_4_E2B", "")
    if key in {"gemma-4-e4b", "gemma_4_e4b"}:
        return str(getattr(settings, "gemma_4_e4b_path", "") or "").strip() or getattr(config, "GEMMA_4_E4B", "")
    if key in {"gemma-4-26b", "gemma_4_26b"}:
        return str(getattr(settings, "gemma_4_26b_path", "") or "").strip() or getattr(config, "GEMMA_4_26B", "")
    if key in {"gemma-4-31b", "gemma_4_31b"}:
        return str(getattr(settings, "gemma_4_31b_path", "") or "").strip() or getattr(config, "GEMMA_4_31B", "")
    if key in {"qwen-2.5-14b", "qwen_14b", "14b"}:
        return str(getattr(settings, "qwen_14b_path", "") or "").strip() or getattr(config, "QWEN_14B", "")
    if key in {"gpt-oss-20b", "gpt_oss_20b", "20b"}:
        return str(getattr(settings, "gpt_oss_20b_path", "") or "").strip() or getattr(config, "GPT_OSS_20B", "openai/gpt-oss-20b")
    if key in {"llama-3.3-70b", "llama-3.1-70b", "llama_70b", "70b"}:
        return getattr(config, "LLAMA_70B", os.path.expanduser("~/models/Llama-3.3-70B-Instruct"))
    return config.LLAMA_3B

_4BIT_MODEL_KEYS: set = set()  # empty = all models load at native precision

def _quantize_bits(llm_model_key: str) -> int:
    """Return quantization bits.
    Native-first strategy: always return 0 (full precision).
    ModelRuntime handles OOM fallback to 8-bit then 4-bit automatically.
    """
    return 0

def _is_remote_model(llm_model_key: str) -> bool:
    """Return True if the resolved model path is a HuggingFace Hub ID (not a local directory)."""
    import os as _os
    path = _resolve_llm_path(llm_model_key)
    if not path:
        return True
    if _os.path.sep in path or path.startswith("/") or (len(path) > 2 and path[1] == ":"):
        return False
    if _os.path.isdir(path):
        return False
    return True

MEMORY_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Return JSON only with keys facts, decisions, preferences, tasks. "
               'Each value is a list of {{"text": str, "salience": 1-5}}. No extra keys. No prose.'),
    ("human", "User:\n{user}\n\nAssistant:\n{assistant}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Update the running conversation summary.\n"
     "Return ONLY this template and keep it concise:\n"
     "Current focus:\n- ...\nCore entities:\n- ...\nKey themes:\n- ...\n"
     "Constraints:\n- ...\nOpen questions:\n- ...\n"
     "Use factual statements only from user question, retrieved metadata, and final answer. "
     "Do not include raw copied blocks or partial NER fragments."),
    ("human", "Summary so far:\n{old_summary}\n\nNew turns:\n{new_turns}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user question into a standalone question for retrieval.\n"
     "Use rolling summary and recent turns to resolve pronouns and omitted context.\n"
     "If anchor_value is provided, use it to resolve referential language.\n"
     "Keep concrete entities, dates, venues, paper ids, and constraints.\n"
     "Do not answer the question.\n"
     "Return JSON only with key: standalone_question.\n"
     "If no rewrite is needed, standalone_question should equal the user question.\n"
     "For pronoun or referential follow-ups, standalone_question must include anchor_value when available.\n"
     'Example: "what field does he study" -> '
     '{"standalone_question":"What field does William Gearty study based on the retrieved papers"}'),
    ("human", "Anchor value:\n{anchor_value}\n\nRolling summary:\n{rolling_summary}\n\n"
              "Recent turns:\n{recent_turns}\n\nUser question:\n{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer only using the provided Syracuse corpus papers text.\n"
     "Each paper record may contain a pre-written abstract or summary — treat these as "
     "authoritative and report their content faithfully rather than paraphrasing loosely.\n"
     "Do not suggest external websites, databases, or sources.\n"
     "Every major claim must be anchored to at least one retrieved paper title.\n"
     "Do not answer with only a list of titles. Synthesize first.\n"
     "IMPORTANT: You must produce a substantive answer of at least 3-5 sentences.\n"
     "Never respond with only a single sentence such as '[Name] is a researcher.' — "
     "always describe the specific research topics, methods, or findings shown in the papers.\n"
     "If papers were retrieved, produce the best-supported answer. "
     "Use insufficient-information refusal only when zero relevant papers are retrieved.\n"
     "If field or role is not explicitly stated, infer it from repeated terms in titles/summaries and mark it as inferred.\n"
     "When identifying a person, list their key research themes with paper evidence.\n"
     "Choose output structure by detected intent category (default, comparison, list, time_range).\n\n"
     "FORMATTING RULES:\n"
     "- Use a blank line between paragraphs to separate distinct points or researchers.\n"
     "- When citing a paper, put the title in quotes.\n"
     "- When listing multiple researchers, give each their own paragraph.\n"
     "- Do NOT add closing remarks, offers to help, or signatures at the end.\n"
     "- Do NOT say 'I noticed' or 'I will revise' or narrate your own process.\n"
     "- End your answer after the last substantive point. Stop there."),
    ("human", "Papers:\n{papers}\n\nQuestion:\n{question}"),
])

@dataclass
class UtilityJob:
    session_id: str
    user_text: str
    assistant_text: str
    new_turns_text: str
    run_summary: bool
    run_memory_extract: bool
    last_focus: Optional[str] = None
    last_topic: Optional[str] = None
    no_results: bool = False
    ner_line: str = ""
    retrieval_meta_text: str = ""
    turns_already_persisted: bool = False
    rolling_summary_snapshot: Optional[str] = None

class UtilityWorker:
    def __init__(self, *, store: SessionStore, memory_vs: Chroma, runtime: ModelRuntime):
        self.store, self.memory_vs, self.runtime = store, memory_vs, runtime
        self._utility_lock = threading.Lock()
        self.q: queue.Queue[UtilityJob] = queue.Queue(
            maxsize=int(getattr(settings, "utility_queue_max", 2000)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._io_lock = threading.Lock()
        self._memory_add_counter = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="utility-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def drain_session(self, session_id: str) -> int:
        drained, kept = 0, []
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if job.session_id == session_id:
                drained += 1
            else:
                kept.append(job)
            self.q.task_done()
        for job in kept:
            try:
                self.q.put_nowait(job)
            except queue.Full:
                logger.warning("Queue full when re-enqueuing job for session %s", job.session_id)
        return drained

    def submit(self, job: UtilityJob) -> None:
        try:
            self.q.put_nowait(job)
        except queue.Full:
            logger.warning("Utility queue full, dropping job for session %s", job.session_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._process(job)
            except Exception:
                logger.error("Utility worker failed for session %s", job.session_id, exc_info=True)
            finally:
                try:
                    self.q.task_done()
                except ValueError:
                    pass

    def _prune_memory_if_needed(self, session_id: str) -> None:
        col = getattr(self.memory_vs, "_collection", None)
        if col is None:
            return
        try:
            ids = (col.get(where={"session_id": session_id}) or {}).get("ids") or []
            max_n = int(getattr(settings, "memory_max_per_session", 500))
            target = int(getattr(settings, "memory_prune_target", 420))
            if len(ids) > max_n:
                col.delete(ids=ids[:max(0, len(ids) - target)])
        except Exception:
            logger.warning("Memory prune failed for session %s", session_id, exc_info=True)

    def _extract_memory(self, session_id: str, user: str, assistant: str) -> None:
        try:
            msgs = MEMORY_EXTRACT_PROMPT.format_messages(user=user, assistant=assistant)
            data = json.loads(self.runtime.llm.invoke(msgs[0].content + "\n" + msgs[1].content))
        except json.JSONDecodeError:
            return
        except Exception:
            logger.warning("Memory extraction failed for session %s", session_id, exc_info=True)
            return

        texts, metas, seen = [], [], set()
        if not isinstance(data, dict):
            data = {}
        for key in ("facts", "decisions", "preferences", "tasks"):
            # Defensive coercion: the utility LLM occasionally returns a dict
            # (single object) or other non-list value where a list is expected.
            # Slicing a dict raises "TypeError: unhashable type: 'slice'", which
            # crashes the utility worker. Coerce to a list of dicts.
            raw = data.get(key, [])
            if isinstance(raw, dict):
                items = [raw]
            elif isinstance(raw, (list, tuple)):
                items = list(raw)
            elif raw is None:
                items = []
            else:
                try:
                    items = list(raw)
                except TypeError:
                    items = []
            for obj in items[:20]:
                if not isinstance(obj, dict):
                    continue
                text = re.sub(r"\s+", " ", str(obj.get("text") or "").strip())[:220]
                if not text or text.lower() in seen:
                    continue
                seen.add(text.lower())
                try:
                    sal = max(1, min(5, int(obj.get("salience") or 3)))
                except (TypeError, ValueError):
                    sal = 3
                texts.append(text)
                metas.append({"type": key, "salience": sal, "session_id": session_id})

        if not texts:
            return
        try:
            self.memory_vs.add_texts(texts=texts, metadatas=metas,
                                     ids=[str(uuid.uuid4()) for _ in texts])
            self._memory_add_counter += len(texts)
        except Exception:
            logger.warning("memory_vs.add_texts failed for session %s", session_id, exc_info=True)
            return

        self._prune_memory_if_needed(session_id)
        if self._memory_add_counter >= int(getattr(settings, "memory_persist_every_n_adds", 25)):
            self._memory_add_counter = 0
            try:
                self.memory_vs.persist()
            except Exception:
                pass

    def _update_summary(self, session_id: str, new_turns_text: str, *,
                        no_results=False, question="", ner_line="",
                        retrieval_meta_text="", rolling_summary_snapshot=None) -> None:
        state = self.store.load(session_id)
        old_summary = rolling_summary_snapshot or state.get("rolling_summary", "") or ""
        turns = state.get("turns", []) or []
        if bool(getattr(settings, "enable_llm_summary_regen", False)):
            rolling = _regenerate_rolling_summary(
                llm=self.runtime.llm, old_summary=old_summary, new_turns_text=new_turns_text,
                question=question, ner_line=ner_line, source_context=retrieval_meta_text,
                no_results=no_results, timeout_s=int(getattr(settings, "llm_timeout_s", 40)))
        else:
            m = re.search(r"ASSISTANT:\s*(.+)", new_turns_text or "", re.IGNORECASE | re.DOTALL)
            snippet = re.sub(r"\s+", " ", m.group(1)).strip() if m else re.sub(r"\s+", " ", new_turns_text or "").strip()
            rolling = build_rolling_summary(old_summary, question,
                                            " ".join([retrieval_meta_text or "", ner_line or ""]).strip(),
                                            assistant_answer=snippet)
        self.store.save(session_id, rolling, turns)

    def _append_turns(self, session_id, user_text, assistant_text, last_focus=None, last_topic=None):
        state = self.store.load(session_id)
        turns = state.get("turns", []) or []
        turns.append({"role": "user", "text": user_text})
        turns.append({"role": "assistant", "text": assistant_text})
        extra = {}
        if last_focus is not None: extra["last_focus"] = last_focus
        if last_topic is not None: extra["last_topic"] = last_topic
        self.store.save(session_id, state.get("rolling_summary", "") or "", turns, extra_state=extra)

    def _process(self, job: UtilityJob) -> None:
        with self._utility_lock, self._io_lock:
            if not job.turns_already_persisted:
                self._append_turns(job.session_id, job.user_text, job.assistant_text,
                                   job.last_focus, job.last_topic)
            if job.run_memory_extract:
                self._extract_memory(job.session_id, job.user_text, job.assistant_text)
            if job.run_summary:
                self._update_summary(
                    job.session_id, job.new_turns_text, no_results=job.no_results,
                    question=job.user_text, ner_line=job.ner_line,
                    retrieval_meta_text=job.retrieval_meta_text,
                    rolling_summary_snapshot=job.rolling_summary_snapshot)

class Engine:
    def __init__(self, *, answer_runtime, utility_runtime, papers_vs, memory_vs,
                 store, session_id, utility_worker, stateless=False,
                 manager=None):
        self.answer_runtime = answer_runtime
        self.utility_runtime = utility_runtime
        self.papers_vs = papers_vs
        self.memory_vs = memory_vs
        self.store = store
        self.session_id = session_id
        self.utility_worker = utility_worker
        self.stateless = bool(stateless)
        self._manager = manager

        self.last_focus = self.last_topic = self.rolling_summary = ""
        self.anchor: Dict[str, Any] = {}
        self.anchor_last_action = "none"
        self.last_rewrite_referential = self.last_rewrite_anchor_valid = False
        self.last_rewrite_blocked = self.last_summary_updated = False
        self.last_retrieval_confidence = "weak"
        self.last_anchor_support_ratio = 0.0
        self.last_rewrite_time_ms = self.last_retrieval_time_ms = 0.0
        self.last_answer_llm_calls = self.last_utility_llm_calls = 0
        self.turns: List[Turn] = []

        if not self.stateless:
            state = self.store.load(session_id)
            self.last_focus = (state.get("last_focus", "") or "").strip()
            self.last_topic = (state.get("last_topic", "") or "").strip()
            self.rolling_summary = (state.get("rolling_summary", "") or "").strip()
            extra = state.get("extra_state", {}) if isinstance(state.get("extra_state"), dict) else {}
            self.anchor = _normalize_anchor(extra.get("anchor", {}))
            if self.anchor:
                self.anchor_last_action = str(extra.get("anchor_last_action", "loaded") or "loaded")
            for obj in (state.get("turns", []) or []):
                role = (obj.get("role") or "").strip()
                text = (obj.get("text") or "").strip()
                if role and text:
                    self.turns.append(Turn(role=role, text=text))

    def _user_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def _recent_turns_text(self, max_turns: int) -> str:
        if max_turns <= 0:
            return ""
        return "\n".join(
            f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}"
            for t in self.turns[-max_turns:]).strip()

    def _last_user_turn_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "user" and t.text.strip():
                return t.text.strip()
        return ""

    def _get_utility_runtime(self):
        """Lazily fetch the utility runtime from the EngineManager.

        The utility model loads on a background thread.  When Engine is
        constructed during the first question, utility_runtime may still be
        None.  By re-checking the manager we pick it up once it finishes
        loading, enabling query rewriting from Q1 onward.
        """
        if self.utility_runtime is not None:
            return self.utility_runtime
        mgr = self._manager
        if mgr is not None:
            rt = getattr(mgr, "utility_runtime", None)
            if rt is not None:
                self.utility_runtime = rt
                return rt
        return None

    def _rewrite_query_structured(self, question: str, *, anchor_value: str = "") -> Dict[str, Any]:
        q = (question or "").strip()
        fallback = {"standalone_question": q}
        utility_rt = self._get_utility_runtime()
        if not bool(getattr(settings, "rewrite_enable", True)) or utility_rt is None:
            _dbg("[REWRITE] skipped: enable=%s, utility_rt=%s" % (
                getattr(settings, "rewrite_enable", True), utility_rt is not None))
            return fallback
        try:
            msgs = REWRITE_PROMPT.format_messages(
                rolling_summary=self.rolling_summary or "",
                recent_turns=self._recent_turns_text(
                    max(1, min(3, int(getattr(settings, "rewrite_max_recent_turns", 3))))) or "",
                question=question, anchor_value=anchor_value or "(none)")
            prompt_text = msgs[0].content + "\n" + msgs[1].content
            timeout_s = int(getattr(settings, "rewrite_timeout_s", 10))
            raw = str(utility_rt.llm.invoke(prompt_text) or "")
            self.last_utility_llm_calls += 1
            _dbg("[REWRITE] raw response", raw[:300])
            m = re.search(r"\{.*\}", str(raw or "").strip(), re.DOTALL)
            sq = re.sub(r"\s+", " ", str(json.loads(m.group(0) if m else raw).get("standalone_question", "") or "")).strip()
            max_chars = int(getattr(settings, "rewrite_max_chars", 220))
            return {"standalone_question": (sq or q)[:max_chars]}
        except Exception as exc:
            _dbg("[REWRITE] exception: %s" % exc)
            return fallback

    def maybe_rewrite_query(self, raw_q: str) -> str:
        q = (raw_q or "").strip()
        self.last_rewrite_referential = self.last_rewrite_anchor_valid = self.last_rewrite_blocked = False
        if not q or _is_anchor_escape_question(q):
            return q

        word_count = len(_tokenize_words(q))
        max_words = max(1, int(getattr(settings, "followup_query_max_words", 8)))
        is_short = word_count < max_words
        pronoun_pat = _get_followup_pronoun_pattern()
        has_pronoun = pronoun_pat is not None and pronoun_pat.search(q) is not None
        has_followup_phrase = any(p and p in q.lower() for p in _get_followup_phrases())
        is_referential = bool(has_pronoun or has_followup_phrase)

        if word_count >= max_words and (
            re.search(r"\b[A-Z]{2,}\b", q) or
            re.search(r"\babout\s+\w+", q, re.IGNORECASE) or
            re.search(r'"[^"]{3,}"', q)
        ):
            return q

        if not (is_short or has_pronoun or has_followup_phrase):
            return q

        self.last_rewrite_referential = is_referential
        anchor_data = _normalize_anchor(self.anchor)
        anchor_value = str(anchor_data.get("value", "") or "").strip() if _anchor_is_stable(anchor_data) else ""

        standalone = (self._rewrite_query_structured(q, anchor_value=anchor_value)
                      .get("standalone_question") or q).strip()
        if is_referential and anchor_value and not _anchor_in_text(anchor_value, standalone):
            standalone = _inject_anchor_into_query(standalone or q, anchor_value).strip()
        if is_referential and anchor_value:
            self.last_rewrite_anchor_valid = _anchor_in_text(anchor_value, standalone)
        return standalone

    def _log_retrieval_state(self, *, where_filter):
        col = getattr(self.papers_vs, "_collection", None)
        name = str(getattr(col, "name", "") or getattr(col, "_name", "") or "").strip()
        count = -1
        if col:
            try: count = int(col.count())
            except Exception: pass
        fj = json.dumps(where_filter, ensure_ascii=False) if where_filter else "{}"
        print(f"[RETRIEVE] collection={name or '(unknown)'} count={count} filter={fj}")
        if count == 0:
            print("[RETRIEVE] WARNING: collection count is zero.")

    def _embed_query_vector(self, query: str) -> Optional[List[float]]:
        q = (query or "").strip()
        if not q:
            return None
        ef = getattr(self.papers_vs, "_embedding_function", None) or getattr(self.papers_vs, "embedding_function", None)
        try:
            if hasattr(ef, "embed_query"):
                vec = ef.embed_query(q)
                if isinstance(vec, list) and vec:
                    return vec
            elif callable(ef):
                vecs = ef([q])
                if isinstance(vecs, list) and vecs and isinstance(vecs[0], list):
                    return vecs[0]
        except Exception:
            pass
        return None

    def _vector_retrieve_docs(self, vs, *, query_embedding, k, fetch_k,
                              lambda_mult, where_filter, prefer_mmr) -> Optional[List[Document]]:
        if not query_embedding:
            return None
        for method_name, use_mmr in [("max_marginal_relevance_search_by_vector", True),
                                      ("similarity_search_by_vector", False)]:
            if use_mmr and not prefer_mmr:
                continue
            method = getattr(vs, method_name, None)
            if method is None:
                continue
            base_kwargs = ({"embedding": query_embedding, "k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
                           if use_mmr else {"embedding": query_embedding, "k": k})
            alt_kwargs = dict(base_kwargs)
            alt_kwargs["query_embedding"] = alt_kwargs.pop("embedding")
            variants = [base_kwargs, alt_kwargs]
            if where_filter:
                for b in list(variants):
                    variants.insert(0, {**b, "filter": where_filter})
            for kwargs in variants:
                try:
                    got = method(**kwargs)
                    if isinstance(got, list):
                        return got
                except Exception:
                    continue
        return None

    def _retrieve_once(self, query, *, query_embedding=None, search_k=None,
                       fetch_k=None, lambda_mult=None, where_filter=None) -> List[Document]:
        k = int(search_k or getattr(settings, "search_k", 30))
        fk = int(fetch_k or getattr(settings, "search_fetch_k", 140))
        if k <= 0 or fk <= 0:
            col = getattr(self.papers_vs, "_collection", None)
            total = 0
            if col:
                try: total = int(col.count())
                except Exception: pass
            total = total or max(k, fk, 10000)
            if k <= 0: k = total
            if fk <= 0: fk = total
        fk = max(fk, 2 * k)
        lm = max(0.3, min(0.95, float(lambda_mult if lambda_mult is not None else getattr(settings, "mmr_lambda", 0.6))))

        if query_embedding:
            vect_docs = self._vector_retrieve_docs(
                self.papers_vs, query_embedding=query_embedding, k=k, fetch_k=fk,
                lambda_mult=lm, where_filter=where_filter, prefer_mmr=True)
            if isinstance(vect_docs, list):
                return vect_docs

        _retrieval_timeout = int(getattr(settings, "retrieval_timeout_s", 60))

        def _invoke_with_retrieval_timeout(retriever, q, timeout_s):
            """Run retriever.invoke in a thread with a timeout."""
            if timeout_s <= 0:
                return retriever.invoke(q)
            _result_box = [None]
            _error_box = [None]
            def _run():
                try:
                    _result_box[0] = retriever.invoke(q)
                except Exception as exc:
                    _error_box[0] = exc
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=timeout_s)
            if t.is_alive():
                logger.warning("Retrieval timed out after %ds (filter=%s)", timeout_s, where_filter)
                print(f"[RETRIEVE] TIMEOUT after {timeout_s}s")
                return []
            if _error_box[0] is not None:
                raise _error_box[0]
            return _result_box[0] if _result_box[0] is not None else []

        search_kwargs = {"k": k, "fetch_k": fk, "lambda_mult": lm}
        if where_filter:
            search_kwargs["filter"] = where_filter
            try:
                sim_kwargs = {"k": min(k, 30)}
                sim_kwargs["filter"] = where_filter
                retriever = self.papers_vs.as_retriever(
                    search_type="similarity", search_kwargs=sim_kwargs)
                result = _invoke_with_retrieval_timeout(retriever, query, _retrieval_timeout)
                if result:
                    return result
            except RuntimeError as e:
                if "contiguous" not in str(e).lower() and "ef" not in str(e).lower():
                    raise
                logger.warning("Similarity+filter failed (%s), trying MMR", e)
        try:
            retriever = self.papers_vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
            return _invoke_with_retrieval_timeout(retriever, query, _retrieval_timeout)
        except RuntimeError as e:
            if "contiguous" not in str(e).lower() and "ef" not in str(e).lower():
                raise
            logger.warning("HNSW error with filter=%s, falling back: %s", where_filter, e)

        if where_filter:
            try:
                fallback_kwargs = {"k": min(k, 20)}
                fallback_kwargs["filter"] = where_filter
                retriever = self.papers_vs.as_retriever(
                    search_type="similarity", search_kwargs=fallback_kwargs)
                return _invoke_with_retrieval_timeout(retriever, query, _retrieval_timeout)
            except RuntimeError:
                logger.warning("HNSW fallback similarity+filter also failed")

        try:
            plain_kwargs = {"k": k, "fetch_k": fk, "lambda_mult": lm}
            retriever = self.papers_vs.as_retriever(search_type="mmr", search_kwargs=plain_kwargs)
            return _invoke_with_retrieval_timeout(retriever, query, _retrieval_timeout)
        except RuntimeError:
            logger.warning("HNSW fallback MMR without filter also failed")

        return []

    def _merge_unique_docs(self, a, b):
        return dedupe_docs((a or []) + (b or []))

    def _rerank_papers_with_llm(self, query: str, docs: List[Document]) -> List[Document]:
        if not bool(getattr(settings, "rerank_enable", True)) or self.utility_runtime is None or not docs:
            return docs
        cand_k = max(1, int(getattr(settings, "rerank_candidate_k", 30)))
        final_k = max(1, int(getattr(settings, "rerank_final_k", 12)))
        candidates = list(docs[:cand_k])
        if len(candidates) <= final_k:
            return candidates

        rows = []
        for i, d in enumerate(candidates):
            meta = d.metadata or {}
            snippet = re.sub(r"\s+", " ", str(d.page_content or ""))[:260]
            rows.append(f"{i}. title={meta.get('title','')} | researcher={meta.get('researcher','')}"
                        f" | year={meta.get('year','')} | snippet={snippet}")
        prompt = (f"Select the most relevant chunks for answering the query.\n"
                  f'Return JSON only as {{"keep":[idx,...]}} with at most {final_k} indices.\n\n'
                  f"Query:\n{query}\n\nCandidates:\n" + "\n".join(rows))
        raw = _invoke_with_timeout(self.utility_runtime.llm, prompt,
                                   max(1, int(getattr(settings, "rerank_timeout_s", 12))))
        self.last_utility_llm_calls += 1

        keep_idx = []
        try:
            m = re.search(r"\{.*\}", str(raw or "").strip(), re.DOTALL)
            for v in (json.loads(m.group(0) if m else raw).get("keep", []) or []):
                try:
                    i = int(v)
                except Exception:
                    continue
                if 0 <= i < len(candidates) and i not in keep_idx:
                    keep_idx.append(i)
        except Exception:
            pass

        selected = [candidates[i] for i in keep_idx[:final_k]]
        have = {id(d) for d in selected}
        for d in candidates:
            if len(selected) >= final_k:
                break
            if id(d) not in have:
                selected.append(d)
                have.add(id(d))
        return selected[:final_k]

    def _explicit_topic_shift(self, question: str) -> bool:
        q = (question or "").strip()
        if not q:
            return False
        if _is_meta_command(q):
            return True
        if any(c in q.lower() for c in ("switch topic", "different topic", "new topic",
               "unrelated", "change subject", "another area", "instead let's discuss")):
            return True
        anchor_data = _normalize_anchor(self.anchor)
        anchor_value = str(anchor_data.get("value", "") or "").strip()
        if anchor_value and not _is_placeholder_anchor_value(anchor_value):
            person_name = _extract_person_name(q)
            if person_name and len(person_name) > 3:
                if not _anchor_in_text(anchor_value, person_name) and not _anchor_in_text(person_name, anchor_value):
                    return True
        return False

    def _post_filter_retrieved_docs(self, docs, *, query):
        if not docs:
            return docs
        unique = self._merge_unique_docs(docs, [])
        q_tokens = set(t for t in _tokenize_words(_strip_corpus_noise_terms(query) or query)
                       if not _is_generic_query_token(t))
        if not q_tokens:
            return unique
        if len(q_tokens) > 4:
            return unique
        pruned = [d for d in unique if q_tokens & set(_tokenize_words(doc_haystack(d)))]
        return pruned if pruned else unique

    def retrieve_papers(self, query, budget_papers, *, query_embedding=None,
                        search_k=None, fetch_k=None, lambda_mult=None,
                        where_filter=None, raw_question="") -> List[Document]:
        q = _strip_corpus_noise_terms((query or "").strip()) or (query or "").strip()
        if not q:
            return []
        self._log_retrieval_state(where_filter=where_filter)
        docs = self._retrieve_once(q, query_embedding=query_embedding, search_k=search_k,
                                   fetch_k=fetch_k, lambda_mult=lambda_mult, where_filter=where_filter)

        if not where_filter:
            _person_in_query = _extract_person_name(q)
            if _person_in_query and len(_person_in_query) > 3 and _looks_like_person_candidate(_person_in_query):
                _ptoks = [t for t in re.findall(r"[A-Za-z]+", str(_person_in_query).strip()) if t]
                if len(_ptoks) >= 2:
                    for _var in generate_name_variants(_ptoks):
                        _var = _var.strip()
                        if not _var:
                            continue
                        _person_docs = self._retrieve_once(
                            q, query_embedding=None, search_k=search_k,
                            fetch_k=fetch_k, lambda_mult=lambda_mult,
                            where_filter={"researcher": _var})
                        if _person_docs:
                            docs = self._merge_unique_docs(docs, _person_docs)
                            break

        is_followup = _query_is_short_or_pronoun(raw_question or q)
        anchor_value = str((self.anchor or {}).get("value", "") or "").strip()
        anchor_ratio = _anchor_support_ratio(anchor_value, docs) if anchor_value else 0.0
        min_ratio = float(getattr(settings, "anchor_consistency_min_ratio", 0.45))
        explicit_entity = _has_explicit_entity_signal(raw_question or q)

        if (bool(getattr(settings, "retrieval_dual_query", True)) and is_followup
            and _anchor_is_stable(self.anchor) and anchor_ratio >= min_ratio
            and not where_filter
            and ((not explicit_entity) or _anchor_in_text(anchor_value, raw_question or q)
                 or _is_followup_coref_question(raw_question or q))):
            summary_q = _summary_query_from_text(self.rolling_summary)
            if summary_q:
                q2 = f"{q} {summary_q}".strip()
                if q2 != q:
                    docs = self._merge_unique_docs(docs, self._retrieve_once(
                        q2, query_embedding=query_embedding, search_k=search_k,
                        fetch_k=fetch_k, lambda_mult=lambda_mult, where_filter=where_filter))

        docs = self._post_filter_retrieved_docs(docs, query=q)
        return pack_docs(docs, budget_papers, self.answer_runtime.count_tokens) if budget_papers > 0 else docs

    def retrieve_papers_by_author(self, query: str, person_name: str,
                                  budget_papers: int, *,
                                  query_embedding=None) -> List[Document]:
        """Retrieve papers where *person_name* appears in the ``authors``
        metadata field (co-author search) or in common ``researcher`` name
        variants, using ChromaDB ``$or`` / ``$contains`` filters.

        This fills the gap where Stage 1 exact-match on ``researcher`` fails
        because the person is a co-author rather than the primary researcher.
        """
        toks = [t for t in re.findall(r"[A-Za-z]+", str(person_name or "").strip()) if t]
        if len(toks) < 2:
            return []
        last = toks[-1]

        or_clauses = [{"researcher": v} for v in generate_name_variants(toks)]
        or_clauses.append({"authors": {"$contains": last}})
        where_filter = {"$or": or_clauses}

        q = _strip_corpus_noise_terms((query or "").strip()) or (query or "").strip()
        if not q:
            return []
        self._log_retrieval_state(where_filter={"$or": f"[researcher variants + authors $contains {last}]"})
        docs = self._retrieve_once(q, query_embedding=query_embedding,
                                   where_filter=where_filter)
        docs = self._post_filter_retrieved_docs(docs, query=q)
        return pack_docs(docs, budget_papers, self.answer_runtime.count_tokens) if budget_papers > 0 else docs

    def keyword_search_papers(self, keywords: List[str], budget: int,
                              *, query_embedding=None) -> List[Document]:
        """Search papers by keyword presence in the raw document text.

        Uses ChromaDB's ``where_document`` ``$contains`` operator — a true
        substring match on the stored chunk content.  When *query_embedding*
        is provided the results are also ranked by vector similarity;
        otherwise they are returned in ChromaDB's internal order.

        Returns at most *budget* ``Document`` objects.
        """
        col = getattr(self.papers_vs, "_collection", None)
        if col is None or not keywords:
            return []

        kws = list(dict.fromkeys(kw.strip() for kw in keywords if kw.strip()))
        if not kws:
            return []

        budget = max(1, min(budget, 60))

        try:
            if query_embedding:
                if len(kws) == 1:
                    where_doc = {"$contains": kws[0]}
                else:
                    where_doc = {"$or": [{"$contains": kw} for kw in kws]}
                raw = col.query(
                    query_embeddings=[query_embedding],
                    where_document=where_doc,
                    n_results=budget,
                    include=["documents", "metadatas"],
                )
                ids = (raw.get("ids") or [[]])[0]
                documents = (raw.get("documents") or [[]])[0]
                metadatas = (raw.get("metadatas") or [[]])[0]
            else:
                if len(kws) == 1:
                    where_doc = {"$contains": kws[0]}
                else:
                    where_doc = {"$or": [{"$contains": kw} for kw in kws]}
                raw = col.get(
                    where_document=where_doc,
                    limit=budget,
                    include=["documents", "metadatas"],
                )
                ids = raw.get("ids") or []
                documents = raw.get("documents") or []
                metadatas = raw.get("metadatas") or []

            docs: List[Document] = []
            for i in range(len(ids)):
                text = documents[i] if i < len(documents) else ""
                meta = metadatas[i] if i < len(metadatas) else {}
                docs.append(Document(page_content=text or "", metadata=meta or {}))
            print(f"[RETRIEVE] keyword_search found {len(docs)} docs for keywords={kws}")
            return docs

        except Exception as e:
            logger.warning("keyword_search_papers failed: %s", e, exc_info=True)
            return []

    def retrieve_memory(self, query, budget_memory, *, query_embedding=None) -> List[Document]:
        docs = []
        if query_embedding:
            vect_docs = self._vector_retrieve_docs(
                self.memory_vs, query_embedding=query_embedding, k=12, fetch_k=24,
                lambda_mult=0.4, where_filter={"session_id": self.session_id}, prefer_mmr=False)
            if isinstance(vect_docs, list):
                docs = vect_docs
        if not docs:
            docs = self.memory_vs.as_retriever(
                search_kwargs={"k": 12, "filter": {"session_id": self.session_id}}).invoke(query)
        return pack_docs(docs, budget_memory, self.answer_runtime.count_tokens)

    def _rewrite_query_if_needed(self, raw_q: str) -> Tuple[str, List[Document], str]:
        q = (raw_q or "").strip()
        if not q:
            return q, [], "default"

        t0 = time.perf_counter()
        query_for_retrieval = self.maybe_rewrite_query(q)
        self.last_rewrite_time_ms = (time.perf_counter() - t0) * 1000.0

        anchor_data = _normalize_anchor(self.anchor)
        anchor_value = str(anchor_data.get("value", "") or "").strip() if _anchor_is_stable(anchor_data) else ""
        referential = _is_followup_coref_question(q)
        anchor_escape = _is_anchor_escape_question(q)

        # --- Fix: topic-pivot anchor handoff ---
        # When an anchor-escape question ("who else studies it") fires with no
        # active person anchor, inject the last known topic so the pronoun
        # resolves to the subject of the preceding topic question.  This covers
        # sequences like Q3("what is LIGO and Virgo?") → Q4("who else studies it")
        # where the person anchor was cleared but the topic is still relevant.
        #
        # Guard: only inject topic when the question uses impersonal pronouns
        # (it/this/that/those/these).  If the question uses person pronouns
        # (he/she/him/her/they/them), the user is referring to a *person* and
        # injecting a topic would produce a nonsensical query like
        # "what else has LIGO published".  Those cases are handled downstream
        # by the dangling-pronoun detector in the pipeline.
        _person_pat = _get_person_pronoun_pattern()
        _has_person_pronoun = bool(_person_pat and _person_pat.search(q))
        _min_topic_chars = max(1, int(getattr(settings, "topic_inject_min_chars", 4)))
        if (anchor_escape and not anchor_value
                and self._user_turn_count() > 0
                and not _has_person_pronoun):
            _topic_for_inject = (self.last_topic or "").strip()
            if not _topic_for_inject:
                # Fall back: extract topic focus from the previous user turn
                prev_user = self._last_user_turn_text()
                if prev_user and _norm_text(prev_user) != _norm_text(q):
                    _topic_for_inject = (_extract_focus_from_question(prev_user) or "").strip()
            if not _topic_for_inject:
                # Last resort: pull topic keywords from the rolling summary
                _topic_for_inject = _extract_summary_topic_keywords(
                    self.rolling_summary, max_chars=140).strip()
            if _topic_for_inject and len(_topic_for_inject) > _min_topic_chars:
                if not _anchor_in_text(_topic_for_inject, query_for_retrieval):
                    query_for_retrieval = _inject_anchor_into_query(
                        query_for_retrieval or q, _topic_for_inject).strip()
                    _dbg("[REWRITE] anchor-escape topic inject",
                         f"topic='{_topic_for_inject}' → query='{query_for_retrieval}'")
        # --- end fix ---

        if referential and anchor_value and not anchor_escape and not _anchor_in_text(anchor_value, query_for_retrieval):
            query_for_retrieval = _inject_anchor_into_query(query_for_retrieval or q, anchor_value).strip()

        if referential and (not anchor_value or not _anchor_in_text(anchor_value, query_for_retrieval)):
            _q_substantive_toks = [t for t in _tokenize_words(q)
                                   if len(t) >= 3 and not _is_generic_query_token(t)]
            _skip_prev_inject = len(_q_substantive_toks) >= 4

            if _skip_prev_inject:
                query_for_retrieval = (query_for_retrieval or q).strip()
            else:
                prev_user = self._last_user_turn_text()
                if prev_user and _norm_text(prev_user) != _norm_text(q):
                    prev_person = _extract_person_name(prev_user)
                    if prev_person and len(prev_person) > 3:
                        query_for_retrieval = _inject_anchor_into_query(q, prev_person).strip()
                    else:
                        prev_focus = _extract_focus_from_question(prev_user)
                        if prev_focus and len(prev_focus) > 3:
                            query_for_retrieval = f"{q} {prev_focus}".strip()
                        else:
                            query_for_retrieval = (query_for_retrieval or q).strip()
                else:
                    query_for_retrieval = (query_for_retrieval or q).strip()
            if anchor_value:
                topic_kw = _extract_summary_topic_keywords(self.rolling_summary, max_chars=140)
                max_chars = max(64, int(getattr(settings, "rewrite_max_chars", 220)))
                if topic_kw and _summary_keywords_overlap_anchor(topic_kw, anchor_value):
                    if _norm_text(topic_kw) not in _norm_text(query_for_retrieval):
                        trial = f"{query_for_retrieval} {topic_kw}".strip()
                        query_for_retrieval = trial[:max_chars].rstrip() if len(trial) > max_chars else trial
            self.last_rewrite_blocked = True

        if not query_for_retrieval:
            query_for_retrieval = q

        if (not self.last_rewrite_blocked and self._user_turn_count() > 0
            and not _has_explicit_entity_signal(q) and _query_is_short_or_pronoun(q)
            and self.rolling_summary and anchor_value):
            topic_kw = _extract_summary_topic_keywords(self.rolling_summary, max_chars=140)
            if (topic_kw and not _anchor_in_text(topic_kw[:40], query_for_retrieval)
                and _summary_keywords_overlap_anchor(topic_kw, anchor_value)):
                query_for_retrieval = f"{query_for_retrieval} {topic_kw}".strip()

        max_chars = max(64, int(getattr(settings, "rewrite_max_chars", 220)))
        if len(query_for_retrieval) > max_chars:
            query_for_retrieval = query_for_retrieval[:max_chars].rstrip()

        return query_for_retrieval, [], _classify_generic_intent(query_for_retrieval or q)

    def _persist_light_state(self, *, last_focus, last_topic, extra_state=None):
        state = self.store.load(self.session_id)
        extra = {"last_focus": last_focus, "last_topic": last_topic}
        if isinstance(extra_state, dict):
            extra.update({k: v for k, v in extra_state.items() if k})
        self.store.save(self.session_id, state.get("rolling_summary", "") or "",
                        state.get("turns", []) or [], extra_state=extra)

    def prepare_context(self, question: str, *, stateless: bool = False) -> EngineContext:
        self.last_answer_llm_calls = self.last_utility_llm_calls = 0
        budgets = dynamic_budgets()
        raw_q = (question or "").strip()

        if (not stateless) and self._explicit_topic_shift(raw_q):
            self.last_focus = self.last_topic = ""
            self.rolling_summary = ""
            self.anchor = {}
            self.anchor_last_action = "topic_shift"
            try:
                state = self.store.load(self.session_id)
                turns = state.get("turns", []) or []
                self.store.save(self.session_id, "", turns, extra_state={
                    "last_focus": "", "last_topic": "",
                    "anchor": {}, "anchor_last_action": "topic_shift",
                    "summary_updated": False,
                })
            except Exception:
                logger.warning("Failed to persist topic shift", exc_info=True)

        user_turns = self._user_turn_count()
        allow_prev = (user_turns > 0) and (not stateless)
        rewritten_q, focus_mem_docs, detected_intent = self._rewrite_query_if_needed(raw_q)

        search_k = fetch_k = None
        if _query_is_short_or_pronoun(raw_q):
            base_k = int(getattr(settings, "search_k", 30))
            base_fk = int(getattr(settings, "search_fetch_k", 140))
            search_k = max(base_k, int(base_k * float(getattr(settings, "followup_k_mult", 2.0))))
            fetch_k = max(base_fk, int(base_fk * float(getattr(settings, "followup_fetch_k_mult", 2.0))))

        query_embedding = None
        if rewritten_q.strip() and (
            hasattr(self.papers_vs, "max_marginal_relevance_search_by_vector")
            or hasattr(self.papers_vs, "similarity_search_by_vector")
            or (allow_prev and hasattr(self.memory_vs, "similarity_search_by_vector"))
        ):
            query_embedding = self._embed_query_vector(_strip_corpus_noise_terms(rewritten_q) or rewritten_q)

        if rewritten_q.strip():
            t0 = time.perf_counter()
            paper_docs = self.retrieve_papers(rewritten_q, budgets["BUDGET_PAPERS"],
                                              query_embedding=query_embedding, search_k=search_k,
                                              fetch_k=fetch_k, raw_question=raw_q)
            self.last_retrieval_time_ms = (time.perf_counter() - t0) * 1000.0
        else:
            paper_docs = []
            self.last_retrieval_time_ms = 0.0

        mem_docs = (focus_mem_docs or (
            self.retrieve_memory(rewritten_q, budgets["BUDGET_MEMORY"], query_embedding=query_embedding)
            if rewritten_q.strip() else [])) if allow_prev else []

        return EngineContext(
            raw_question=raw_q, rewritten_question=rewritten_q, detected_intent=detected_intent,
            paper_docs=paper_docs, mem_docs=mem_docs, stateless=bool(stateless),
            user_turns=user_turns, allow_prev_context=allow_prev,
            allow_summary=not stateless, budgets=budgets, anchor=_normalize_anchor(self.anchor))

    def _choose_last_focus_from_answer(self, question, paper_docs):
        q = (question or "").strip()
        name = _extract_person_name(q)
        if name and not _is_invalid_focus_value(name):
            name_support = _anchor_support_ratio(name, paper_docs) if paper_docs else 0.0
            short_query = len(_tokenize_words(q)) <= max(6, len(_tokenize_words(name)) + 2)
            if short_query or name_support >= 0.15:
                return name
        focus = _extract_focus_from_question(q)
        if focus and not _is_invalid_focus_value(focus):
            return focus
        counts: Dict[str, int] = {}
        for d in paper_docs:
            r = (d.metadata or {}).get("researcher", "").strip()
            if r and not _is_invalid_focus_value(r):
                counts[r] = counts.get(r, 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0] if counts else ""

    def _choose_last_topic_from_question(self, question):
        q = (question or "").strip()
        if not q:
            return ""
        person_name = _extract_person_name(q)
        if person_name:
            q_tokens = len(_tokenize_words(q))
            name_tokens = max(1, len(_tokenize_words(person_name)))
            if q_tokens <= max(6, name_tokens + 3):
                return ""
        topic = _extract_focus_from_question(q)
        return topic if topic and not _is_invalid_focus_value(topic) else ""

    def finalize_turn(self, context: EngineContext, answer: str, *, no_results: bool = False) -> None:
        if context.stateless:
            return

        prior = self.store.load(self.session_id)
        prior_focus = str(prior.get("last_focus", "") or "")
        prior_topic = str(prior.get("last_topic", "") or "")
        prior_anchor = prior.get("anchor", {}) if isinstance(prior.get("anchor", {}), dict) else {}
        prior_anchor_action = str(prior.get("anchor_last_action", "") or "")

        new_focus = self._choose_last_focus_from_answer(context.rewritten_question, context.paper_docs)
        new_topic = self._choose_last_topic_from_question(context.rewritten_question)
        new_turns_text = f"USER: {context.raw_question}\nASSISTANT: {answer}\n"

        max_docs = int(getattr(settings, "ner_context_max_docs", 12))
        if self.last_retrieval_confidence in {"weak", "inconsistent"}:
            max_docs = min(max_docs, max(2, int(getattr(settings, "low_conf_ner_context_max_docs", 6))))
        retrieval_meta = _build_ner_context_text(context.paper_docs, max_docs=max_docs)

        anchor_value = str((self.anchor or {}).get("value", "") or "").strip()
        min_anchor_ratio = float(getattr(settings, "anchor_consistency_min_ratio", 0.45))
        anchor_ratio = _anchor_support_ratio(anchor_value, context.paper_docs) if anchor_value else 1.0
        anchor_consistent = (not anchor_value) or (anchor_ratio >= min_anchor_ratio)
        anchor_hq = (not anchor_value) or _anchor_is_stable(self.anchor)
        retrieval_weak = len(context.paper_docs) < max(1, int(getattr(settings, "retrieval_weak_min_docs", 3)))
        anchor_absent = bool(anchor_value) and anchor_ratio == 0.0

        derived_confidence = (_retrieval_confidence_label(
            docs_count=len(context.paper_docs), anchor_consistent=anchor_consistent)
            if context.paper_docs else "weak")
        current_confidence = str(self.last_retrieval_confidence or "").strip().lower()
        confidence_rank = {"weak": 0, "inconsistent": 1, "low": 2, "medium": 3, "high": 4}
        if current_confidence in confidence_rank and derived_confidence in confidence_rank:
            self.last_retrieval_confidence = (
                current_confidence
                if confidence_rank[current_confidence] < confidence_rank[derived_confidence]
                else derived_confidence
            )
        else:
            self.last_retrieval_confidence = current_confidence or derived_confidence

        if no_results or retrieval_weak or anchor_absent:
            self.last_focus = re.sub(r"\s+", " ", (context.raw_question or "").strip())[:220]
            self.last_topic = ""
            self.anchor = {}
            self.anchor_last_action = "cleared_weak_retrieval"

        summary_should_update = bool(context.allow_summary and not no_results and not retrieval_weak
                                     and anchor_consistent and anchor_hq
                                     and self.last_retrieval_confidence in {"medium", "high"})
        self.last_summary_updated = summary_should_update
        self.last_anchor_support_ratio = float(anchor_ratio)

        safe = (not no_results and not retrieval_weak
                and self.last_retrieval_confidence in {"medium", "high"}
                and anchor_consistent and anchor_hq)

        anchor_source = str((self.anchor or {}).get("source", "") or "").strip().lower()
        person_anchor_set_this_turn = (
            anchor_source in {"explicit_entity_signal", "dominant_metadata:researcher"}
            and anchor_value
            and self.anchor_last_action in {"set_from_dominance", "kept_reinforced"}
            and anchor_ratio > 0
        )

        if safe:
            if new_focus: self.last_focus = new_focus
            if new_topic: self.last_topic = new_topic
            anchor_to_save = _normalize_anchor(self.anchor)
            action_to_save = self.anchor_last_action
        elif no_results or retrieval_weak or anchor_absent:
            anchor_to_save = _normalize_anchor(self.anchor)
            action_to_save = self.anchor_last_action
        elif person_anchor_set_this_turn:
            if new_focus: self.last_focus = new_focus
            anchor_to_save = _normalize_anchor(self.anchor)
            action_to_save = self.anchor_last_action
        else:
            self.last_focus, self.last_topic = prior_focus, prior_topic
            self.anchor = _normalize_anchor(prior_anchor)
            self.anchor_last_action = prior_anchor_action
            anchor_to_save = _normalize_anchor(prior_anchor)
            action_to_save = prior_anchor_action

        retrieval_meta_for_summary = retrieval_meta if summary_should_update else ""
        llm_regen = bool(getattr(settings, "enable_llm_summary_regen", False))
        memory_ok = (not no_results and anchor_consistent
                     and self.last_retrieval_confidence in {"medium", "high"})
        run_memory = bool(memory_ok and (
            bool(getattr(settings, "memory_extract_first_turn", True)) or context.allow_prev_context))

        extra_state = {
            "last_focus": self.last_focus, "last_topic": self.last_topic,
            "anchor": anchor_to_save, "anchor_last_action": action_to_save,
            "summary_updated": self.last_summary_updated,
            "retrieval_confidence": self.last_retrieval_confidence,
            "anchor_support_ratio": self.last_anchor_support_ratio,
            "rewrite_anchor_valid": self.last_rewrite_anchor_valid,
            "rewrite_blocked": self.last_rewrite_blocked,
        }

        state = self.store.load(self.session_id)
        rolling = state.get("rolling_summary", "") or ""
        if summary_should_update:
            rolling = build_rolling_summary(rolling, context.raw_question, retrieval_meta_for_summary, answer)
        turns = state.get("turns", []) or []
        turns.append({"role": "user", "text": context.raw_question})
        turns.append({"role": "assistant", "text": answer})
        self.store.save(self.session_id, rolling, turns, extra_state=extra_state)

        if self.utility_worker is not None and int(getattr(settings, "enable_utility_background", 1)) == 1:
            self.utility_worker.submit(UtilityJob(
                session_id=self.session_id, user_text=context.raw_question,
                assistant_text=answer, new_turns_text=new_turns_text,
                run_summary=bool(summary_should_update and llm_regen),
                run_memory_extract=run_memory,
                last_focus=self.last_focus if new_focus else None,
                last_topic=self.last_topic if new_topic else None,
                no_results=no_results, ner_line="",
                retrieval_meta_text=retrieval_meta_for_summary,
                turns_already_persisted=True, rolling_summary_snapshot=rolling))
            return

        if summary_should_update and llm_regen and self.utility_runtime is not None:
            rolling = _regenerate_rolling_summary(
                llm=self.utility_runtime.llm, old_summary=rolling,
                new_turns_text=new_turns_text, question=context.raw_question,
                ner_line="", source_context=retrieval_meta_for_summary,
                no_results=no_results, timeout_s=int(getattr(settings, "llm_timeout_s", 40)))
            self.last_utility_llm_calls += 1
            self.store.save(self.session_id, rolling, turns, extra_state=extra_state)

    def ask(self, question: str) -> Tuple[str, List[Document], List[Document]]:
        context = self.prepare_context(question, stateless=False)
        raw_q = context.raw_question
        pap_docs = list(context.paper_docs)

        _dbg("[RAG] raw_question", raw_q)
        _dbg("[RAG] rewritten_question", context.rewritten_question)

        if not pap_docs:
            answer = f"I could not find anything for {raw_q} in the retrieved corpus."
            self.finalize_turn(context, answer, no_results=True)
            return answer, [], []

        max_docs = int(getattr(settings, "prompt_max_docs", 24))
        text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
        papers_ctx = format_docs_compact(pap_docs, max_docs=max_docs, text_limit=text_limit)
        msgs = ANSWER_PROMPT.format_messages(papers=papers_ctx,
                                             question=context.rewritten_question.lower())
        _sep = _DirectGenerationLLM.SYSTEM_USER_SEP
        full_prompt = _sep.join(m.content for m in msgs)

        tok = self.answer_runtime.count_tokens(full_prompt)
        trigger = context.budgets["TRIGGER"]
        if tok > trigger:
            pap_docs = pack_docs(pap_docs, max(600, int(context.budgets["BUDGET_PAPERS"] * 0.7)),
                                 self.answer_runtime.count_tokens)
            papers_ctx = format_docs_compact(pap_docs, max_docs=max_docs, text_limit=text_limit)
            msgs = ANSWER_PROMPT.format_messages(papers=papers_ctx,
                                                 question=context.rewritten_question.lower())
            full_prompt = _sep.join(m.content for m in msgs)

        try:
            self.last_answer_llm_calls += 1
            raw_answer = _invoke_with_timeout(self.answer_runtime.llm, full_prompt,
                                              int(getattr(settings, "llm_timeout_s", 300)))
        except Exception:
            logger.error("Answer LLM failed for session %s", self.session_id, exc_info=True)
            raw_answer = ""

        answer = _strip_prompt_leak(str(raw_answer)).strip()
        if not answer:
            answer = "I found related papers, but the answer could not be cleanly extracted."

        self.finalize_turn(context, answer)
        return answer, context.paper_docs, []

class EngineManager:
    def __init__(self) -> None:
        _ensure_dir(CACHE_DIR)
        _ensure_dir(MEMORY_DIR)
        self.dbm = DatabaseManager()
        self.dbm.ensure_dirs_exist()
        self.store = SessionStore(STATE_DB)
        self.embeddings = build_embeddings()

        self.active_mode = self.dbm.resolve_mode(getattr(settings, "active_mode", ""))
        if not self.active_mode:
            raise RuntimeError("No retrieval mode configured")
        self.papers_vs_cache: Dict[str, Chroma] = {}
        self._memory_client = _make_local_chroma_client(MEMORY_DIR)
        self.memory_vs = Chroma(collection_name="memory", persist_directory=MEMORY_DIR,
                                embedding_function=self.embeddings, client=self._memory_client)

        self.answer_runtime: Optional[ModelRuntime] = None
        self.utility_runtime: Optional[ModelRuntime] = None
        self.active_answer_model_key = self.active_utility_model_key = ""
        self.active_answer_max_new_tokens = 0
        self.answer_generation_lock = threading.Lock()
        self.utility_worker: Optional[UtilityWorker] = None
        self._utility_suppressed = False

    def get_papers_vs(self, mode: str) -> Chroma:
        m = self.dbm.resolve_mode(mode)
        if not m:
            raise RuntimeError("No database config available")
        if m not in self.papers_vs_cache:
            cfg = self.dbm.get_config(m)
            if cfg is None:
                raise RuntimeError("No database config available")
            _ensure_dir(cfg.chroma_dir)
            self.papers_vs_cache[m] = Chroma(
                collection_name=cfg.collection, persist_directory=cfg.chroma_dir,
                embedding_function=self.embeddings, client=_make_local_chroma_client(cfg.chroma_dir))
        return self.papers_vs_cache[m]

    def switch_mode(self, mode: str) -> None:
        resolved = self.dbm.resolve_mode(mode)
        if not resolved:
            raise RuntimeError("No retrieval mode configured")
        self.active_mode = resolved
        self.dbm.switch_config(resolved)

    _VRAM_HEADROOM_MB = 2500

    def _vram_is_tight(self) -> bool:
        """Return True if free VRAM is below the headroom threshold."""
        vram_free = available_vram_mb()
        if vram_free <= 0:
            return False
        return vram_free < self._VRAM_HEADROOM_MB

    def _evict_utility_if_needed(self) -> None:
        """Unload the utility model and stop its worker to free VRAM."""
        if self.utility_worker is not None:
            try:
                self.utility_worker.stop()
            except Exception:
                pass
            self.utility_worker = None
        if self.utility_runtime is not None:
            try:
                self.utility_runtime.close()
            except Exception:
                pass
            self.utility_runtime = None
            self.active_utility_model_key = ""
            self._utility_suppressed = True
            print("[VRAM] Evicted utility model to free VRAM for answer generation")

    def _switch_runtime(self, attr_name, key_attr, key, max_tokens_attr=None, desired_max=None):
        key = (key or "").strip().lower()
        current_key = getattr(self, key_attr)
        current_rt = getattr(self, attr_name)
        if desired_max is not None:
            if key == current_key and current_rt is not None and getattr(self, max_tokens_attr, 0) == desired_max:
                return
        else:
            if key == current_key and current_rt is not None:
                return
        if current_rt is not None:
            try: current_rt.close()
            except Exception: pass
            setattr(self, attr_name, None)
        max_new = desired_max or int(getattr(settings, "utility_max_new_tokens", 256))
        q_bits = _quantize_bits(key)
        local_only = not _is_remote_model(key)

        if attr_name == "answer_runtime":
            if q_bits > 0:
                self._evict_utility_if_needed()
            else:
                self._utility_suppressed = False

        rt = ModelRuntime(_resolve_llm_path(key), max_new_tokens=max_new,
                          do_sample=False, temperature=0.0, quantize_bits=q_bits,
                          local_only=local_only)
        setattr(self, attr_name, rt)
        setattr(self, key_attr, key)
        if max_tokens_attr and desired_max is not None:
            setattr(self, max_tokens_attr, desired_max)

        if attr_name == "answer_runtime" and self._vram_is_tight():
            self._evict_utility_if_needed()

    def switch_answer_model(self, llm_model_key: str) -> None:
        desired = int(getattr(settings, "answer_max_new_tokens", 768))
        self._switch_runtime("answer_runtime", "active_answer_model_key", llm_model_key,
                             "active_answer_max_new_tokens", desired)

    def switch_model(self, llm_model_key: str) -> None:
        self.switch_answer_model(llm_model_key)

    def switch_utility_model(self, llm_model_key: str) -> None:
        self._switch_runtime("utility_runtime", "active_utility_model_key", llm_model_key)

    def _ensure_utility_worker(self) -> None:
        if getattr(self, "_utility_suppressed", False):
            return
        if self._vram_is_tight():
            return
        if self.utility_worker is not None:
            if self.utility_worker._thread is not None and self.utility_worker._thread.is_alive():
                return
            self.utility_worker = None
        self.switch_utility_model(getattr(settings, "utility_model_key", "llama-3.2-1b"))
        if self.utility_runtime is None:
            return
        self.utility_worker = UtilityWorker(store=self.store, memory_vs=self.memory_vs,
                                            runtime=self.utility_runtime)
        self.utility_worker.start()

    def get_engine(self, session_id, mode, *, stateless=False) -> Engine:
        if self.answer_runtime is None:
            self.switch_answer_model(getattr(settings, "answer_model_key", "llama-3.2-3b"))

        enable_bg = int(getattr(settings, "enable_utility_background", 1)) == 1
        suppressed = getattr(self, "_utility_suppressed", False)
        if enable_bg and not suppressed and not self._vram_is_tight():
            if self.utility_runtime is None:
                self.switch_utility_model(getattr(settings, "utility_model_key", "llama-3.2-1b"))
            self._ensure_utility_worker()

        return Engine(answer_runtime=self.answer_runtime,
                      utility_runtime=self.utility_runtime,
                      papers_vs=self.get_papers_vs(mode), memory_vs=self.memory_vs,
                      store=self.store, session_id=session_id,
                      utility_worker=self.utility_worker, stateless=stateless,
                      manager=self)

    def reset_session(self, session_id: str) -> None:
        if self.utility_worker is not None:
            drained = self.utility_worker.drain_session(session_id)
            if drained:
                _dbg(f"[RESET] drained {drained} pending utility jobs for session {session_id}")
        try:
            self.store.reset(session_id)
        except Exception:
            logger.error("Failed to reset session store for %s", session_id, exc_info=True)
        try:
            col = getattr(self.memory_vs, "_collection", None)
            if col is not None:
                ids = (col.get(where={"session_id": session_id}) or {}).get("ids") or []
                if ids:
                    col.delete(ids=ids)
            try: self.memory_vs.persist()
            except Exception: pass
        except Exception:
            logger.error("Failed to clear memory vectors for session %s", session_id, exc_info=True)

_GLOBAL_MANAGER: Optional[EngineManager] = None
_GLOBAL_MANAGER_LOCK = threading.Lock()

def get_global_manager() -> EngineManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        with _GLOBAL_MANAGER_LOCK:
            if _GLOBAL_MANAGER is None:
                _GLOBAL_MANAGER = EngineManager()
    return _GLOBAL_MANAGER