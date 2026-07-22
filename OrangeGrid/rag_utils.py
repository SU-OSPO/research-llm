#rag_utils.py
import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
 
from runtime_settings import settings
 
def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())
 
def clean_html(s: str) -> str:
    return re.sub(r"</?[a-zA-Z][^>]*>", "", str(s or "")).lower().strip()
 
def normalize_title_case(s: str) -> str:
    raw = re.sub(r"</?[a-zA-Z][^>]*>", "", str(s or "")).strip()
    if not raw:
        return raw
    if any(c.isupper() for c in raw[1:]) and any(c.islower() for c in raw):
        return raw
    return raw.title()
 
def collapse_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())
 
def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:['\-\.][a-z0-9]+)?", norm_text(s))
 
def token_in_hay(token: str, hay: str) -> bool:
    if not token or not hay:
        return False
    return re.search(rf"\b{re.escape(token)}\b", hay) is not None
 
_NLTK_BOOTSTRAP_ATTEMPTED = False
_NLTK_STOPWORDS_CACHE: Optional[Set[str]] = None
_NLTK_WORDS_CACHE: Optional[Set[str]] = None
_NLTK_NAMES_CACHE: Optional[Set[str]] = None
 
def bootstrap_nltk_data() -> None:
    global _NLTK_BOOTSTRAP_ATTEMPTED
    if _NLTK_BOOTSTRAP_ATTEMPTED:
        return
    _NLTK_BOOTSTRAP_ATTEMPTED = True
    try:
        import nltk as _nltk
        for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                     "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
                     "maxent_ne_chunker_tab", "words", "stopwords", "names",
                     "wordnet", "omw-1.4"):
            try:
                _nltk.download(pkg, quiet=True)
            except Exception:
                pass
    except Exception:
        pass
 
_WORDNET_AVAILABLE: Optional[bool] = None
_WORDNET_TOKEN_CACHE: Dict[str, bool] = {}
 
def _wordnet_is_common_word(token: str) -> bool:
    """Return True if *token* has WordNet synsets as a noun, verb, adjective,
    or adverb — i.e. it is a recognized common English word, not a proper noun.
 
    WordNet does not index proper nouns (person/place names), so a word that
    has synsets is very likely a common word rather than a name component.
    Some names like 'brown', 'grant', 'mark' DO appear in WordNet (as
    adjectives/verbs), but those also appear in the NLTK names corpus, so the
    caller can combine both signals.
 
    Results are cached per-token for the lifetime of the process.
    """
    global _WORDNET_AVAILABLE
    if token in _WORDNET_TOKEN_CACHE:
        return _WORDNET_TOKEN_CACHE[token]
    if _WORDNET_AVAILABLE is False:
        return False
    try:
        from nltk.corpus import wordnet as _wn
        _WORDNET_AVAILABLE = True
        result = bool(_wn.synsets(token))
    except LookupError:
        bootstrap_nltk_data()
        try:
            from nltk.corpus import wordnet as _wn
            _WORDNET_AVAILABLE = True
            result = bool(_wn.synsets(token))
        except Exception:
            _WORDNET_AVAILABLE = False
            result = False
    except Exception:
        _WORDNET_AVAILABLE = False
        result = False
    _WORDNET_TOKEN_CACHE[token] = result
    return result
 
def get_stopword_set() -> Set[str]:
    global _NLTK_STOPWORDS_CACHE
    if _NLTK_STOPWORDS_CACHE is not None:
        return _NLTK_STOPWORDS_CACHE
    stopset: Set[str] = set()
    for attempt in range(2):
        try:
            from nltk.corpus import stopwords as _sw
            stopset = set(w.lower() for w in _sw.words("english"))
            break
        except LookupError:
            if attempt == 0:
                bootstrap_nltk_data()
        except Exception:
            break
    _NLTK_STOPWORDS_CACHE = stopset
    return _NLTK_STOPWORDS_CACHE
 
def get_english_word_set() -> Set[str]:
    global _NLTK_WORDS_CACHE
    if _NLTK_WORDS_CACHE is not None:
        return _NLTK_WORDS_CACHE
    wordset: Set[str] = set()
    for attempt in range(2):
        try:
            from nltk.corpus import words as _words
            wordset = {w.lower() for w in _words.words() if isinstance(w, str) and len(w) >= 3}
            break
        except LookupError:
            if attempt == 0:
                bootstrap_nltk_data()
        except Exception:
            break
    _NLTK_WORDS_CACHE = wordset
    return _NLTK_WORDS_CACHE
 
def get_name_token_set() -> Set[str]:
    global _NLTK_NAMES_CACHE
    if _NLTK_NAMES_CACHE is not None:
        return _NLTK_NAMES_CACHE
    nameset: Set[str] = set()
    for attempt in range(2):
        try:
            from nltk.corpus import names as _names
            nameset = {w.lower() for w in _names.words() if isinstance(w, str) and len(w) >= 2}
            break
        except LookupError:
            if attempt == 0:
                bootstrap_nltk_data()
        except Exception:
            break
    _NLTK_NAMES_CACHE = nameset
    return _NLTK_NAMES_CACHE
 
_GENERIC_QUERY_TERMS_CACHE: Optional[Set[str]] = None
_FOLLOWUP_PHRASES_CACHE: Optional[List[str]] = None
_FOLLOWUP_PRONOUN_PATTERN_CACHE: Optional[re.Pattern] = None
_FOLLOWUP_PRONOUN_PATTERN_READY: bool = False
_PERSON_PRONOUN_PATTERN_CACHE: Optional[re.Pattern] = None
_PERSON_PRONOUN_PATTERN_READY: bool = False
 
def _split_config_terms(raw: str) -> List[str]:
    return [s for item in re.split(r"[,;\n|]+", raw or "") if (s := item.strip().lower())]
 
def get_generic_query_terms() -> Set[str]:
    global _GENERIC_QUERY_TERMS_CACHE
    if _GENERIC_QUERY_TERMS_CACHE is None:
        _GENERIC_QUERY_TERMS_CACHE = set(_split_config_terms(
            str(getattr(settings, "generic_query_terms", "") or "")))
    return _GENERIC_QUERY_TERMS_CACHE
 
def get_followup_phrases() -> List[str]:
    global _FOLLOWUP_PHRASES_CACHE
    if _FOLLOWUP_PHRASES_CACHE is None:
        _FOLLOWUP_PHRASES_CACHE = _split_config_terms(
            str(getattr(settings, "followup_phrases", "") or ""))
    return _FOLLOWUP_PHRASES_CACHE
 
def get_followup_pronoun_pattern() -> Optional[re.Pattern]:
    global _FOLLOWUP_PRONOUN_PATTERN_READY, _FOLLOWUP_PRONOUN_PATTERN_CACHE
    if _FOLLOWUP_PRONOUN_PATTERN_READY:
        return _FOLLOWUP_PRONOUN_PATTERN_CACHE
    _FOLLOWUP_PRONOUN_PATTERN_READY = True
    raw = str(getattr(settings, "followup_pronoun_regex", "") or "").strip()
    if raw:
        try:
            _FOLLOWUP_PRONOUN_PATTERN_CACHE = re.compile(raw, re.IGNORECASE)
        except re.error:
            _FOLLOWUP_PRONOUN_PATTERN_CACHE = None
    return _FOLLOWUP_PRONOUN_PATTERN_CACHE
 
def get_person_pronoun_pattern() -> Optional[re.Pattern]:
    """Compiled pattern for person pronouns (he/she/him/her/they/them/…).

    Used by the topic-pivot and dangling-pronoun fixes to distinguish
    person references from impersonal topic references (it/this/that).
    Reads from ``settings.person_pronoun_regex`` so the set of pronouns
    is fully configurable at runtime via environment variables.
    """
    global _PERSON_PRONOUN_PATTERN_READY, _PERSON_PRONOUN_PATTERN_CACHE
    if _PERSON_PRONOUN_PATTERN_READY:
        return _PERSON_PRONOUN_PATTERN_CACHE
    _PERSON_PRONOUN_PATTERN_READY = True
    raw = str(getattr(settings, "person_pronoun_regex", "") or "").strip()
    if raw:
        try:
            _PERSON_PRONOUN_PATTERN_CACHE = re.compile(raw, re.IGNORECASE)
        except re.error:
            _PERSON_PRONOUN_PATTERN_CACHE = None
    return _PERSON_PRONOUN_PATTERN_CACHE
 
def is_generic_query_token(token: str) -> bool:
    t = (token or "").strip().lower()
    if not t or len(t) < max(1, int(getattr(settings, "generic_token_min_len", 3))):
        return True
    if t in get_generic_query_terms():
        return True
    stopset = get_stopword_set()
    return bool(stopset and t in stopset)
 
def is_followup_coref_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    pat = get_followup_pronoun_pattern()
    if pat is not None and pat.search(q):
        return True
    lowered = q.lower()
    return any(key and key in lowered for key in get_followup_phrases())
 
def bust_caches(changed_field: str) -> None:
    global _GENERIC_QUERY_TERMS_CACHE, _FOLLOWUP_PHRASES_CACHE
    global _FOLLOWUP_PRONOUN_PATTERN_CACHE, _FOLLOWUP_PRONOUN_PATTERN_READY
    global _PERSON_PRONOUN_PATTERN_CACHE, _PERSON_PRONOUN_PATTERN_READY
    if changed_field in {"generic_query_terms", "generic_token_min_len"}:
        _GENERIC_QUERY_TERMS_CACHE = None
    if changed_field == "followup_phrases":
        _FOLLOWUP_PHRASES_CACHE = None
    if changed_field == "followup_pronoun_regex":
        _FOLLOWUP_PRONOUN_PATTERN_CACHE = None
        _FOLLOWUP_PRONOUN_PATTERN_READY = False
    if changed_field == "person_pronoun_regex":
        _PERSON_PRONOUN_PATTERN_CACHE = None
        _PERSON_PRONOUN_PATTERN_READY = False
 
_CORPUS_NOISE_TERMS = re.compile(
    r"\b(syracuse|university|faculty|professor|researcher|department|"
    r"college|school|campus|institute|lab|laboratory|su\b|at su\b|at syracuse)",
    re.IGNORECASE,
)
 
def strip_corpus_noise_terms(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", _CORPUS_NOISE_TERMS.sub(" ", query)).strip()
    tokens = [t for t in tokenize_words(cleaned) if not is_generic_query_token(t)]
    return cleaned if tokens else (query or "").strip()
 
def dedupe_docs(docs: List[Any]) -> List[Any]:
    seen: set = set()
    out: List[Any] = []
    for d in docs or []:
        meta = getattr(d, "metadata", None) or {}
        key = (str(meta.get("paper_id", "")) + "::" +
               str(meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out
 
def doc_haystack(d: Any) -> str:
    meta = getattr(d, "metadata", None) or {}
    meta_parts = [str(v or "") for v in meta.values()
                  if not isinstance(v, (list, dict, tuple, set))]
    return (" ".join(meta_parts) + " " + str(getattr(d, "page_content", "") or "")).lower()
 
def truncate_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(".", 1)[0].strip()
    return (cut + "...") if cut else (text[:limit].rstrip() + "...")
 
def clean_snippet(meta: Dict[str, Any], text: str, *, limit: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = re.sub(
        r"^Paper\s+ID:\s*\S+\s+Researcher:\s*[^\n]*?"
        r"(?:Title:\s*[^\n]*?)?(?:Authors:\s*[^\n]*?)?"
        r"(?:Primary\s+Topic:\s*[^\n]*?)?(?:Info:\s*[^\n]*?)?"
        r"(?:DOI:\s*\S+\s*)?",
        "", raw, flags=re.IGNORECASE,
    ).strip()
    if raw.lower().startswith("paper id:"):
        raw = re.sub(r"^Paper\s+ID:\s*\d+.*?(?:Summary:\s*)?", "", raw,
                      flags=re.IGNORECASE | re.DOTALL).strip()
 
    lowered = raw.lower()
    for value in (str((meta or {}).get("title", "") or ""),
                  str((meta or {}).get("authors", "") or "")):
        v = value.strip()
        if v and lowered.startswith(v.lower()):
            raw = raw[len(v):].lstrip(" \n\t:-|")
            lowered = raw.lower()
 
    first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
    if first_line.count(",") >= 6 and len(first_line) <= 240:
        raw = "\n".join(raw.splitlines()[1:]).strip()
 
    return truncate_text(collapse_whitespace(raw), limit)
 
def _extract_summary_from_page_content(page_content: str) -> str:
    """Extract the Summary section from the stored document text.
 
    chroma_ingest.py embeds documents as:
        Paper ID: ...  Researcher: ...  Title: ...
        Summary:\n<summary text>
        Fulltext (...):\n<fulltext>
 
    Summary is NOT in metadata — it lives inside page_content.
    This extracts it by finding the Summary: marker and reading
    until the next section header or end of text.
    """
    text = str(page_content or "")
    if not text:
        return ""
    m = re.search(r"(?im)^Summary:\s*", text)
    if not m:
        return ""
    after = text[m.end():]
    stop = re.search(r"(?im)^[A-Za-z][A-Za-z _]*(?:\([^)]*\))?\s*:", after)
    if stop:
        after = after[:stop.start()]
    return collapse_whitespace(clean_html(after.strip()))
 
def build_compact_context(docs: List[Any], max_docs: Optional[int] = None,
                          text_limit: Optional[int] = None) -> str:
    if max_docs is None:
        max_docs = int(getattr(settings, "prompt_max_docs", 24))
    if text_limit is None:
        text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
 
    def _researcher_sort_key(d):
        meta = getattr(d, "metadata", None) or {}
        r = collapse_whitespace(str(meta.get("researcher", "") or "")).lower()
        if not r or r in ("unknown", "n/a", "none", ""):
            return (1, "", str(meta.get("paper_id", "")))
        return (0, r, str(meta.get("paper_id", "")))
 
    sorted_docs = sorted(docs[:max_docs * 2] if docs else [],
                         key=_researcher_sort_key)[:max_docs]
 
    blocks: List[str] = []
    last_researcher = None
    for d in sorted_docs:
        meta = getattr(d, "metadata", None) or {}
        page_content = getattr(d, "page_content", "") or ""
        researcher = collapse_whitespace(str(meta.get("researcher", "") or ""))
 
        _PH = {"n/a", "na", "unknown", "none", "null", "not available", "not provided", "untitled"}
        _ma = str(meta.get("abstract_text", "") or "").strip()
        _raw_summary = _ma if len(_ma) > 10 and _ma.lower() not in _PH else _extract_summary_from_page_content(page_content)
        _summary_is_real = bool(_raw_summary) and len(_raw_summary) > 10 and _raw_summary.lower() not in _PH
        summary_text = truncate_text(_raw_summary, text_limit) if _summary_is_real else ""
 
        snippet = ""
        if not summary_text:
            snippet = clean_snippet(meta, page_content, limit=text_limit)
 
        researcher_key = researcher.lower().strip() if researcher else ""
        if researcher_key and researcher_key != (last_researcher or ""):
            blocks.append(f"--- Papers by {researcher} ---")
            last_researcher = researcher_key
        elif not researcher_key and last_researcher:
            last_researcher = None
 
        lines = [
            f"title: {clean_html(str(meta.get('title', '') or ''))}",
            f"researcher: {researcher}" if researcher else None,
            f"authors: {clean_html(str(meta.get('authors', '') or ''))}",
            f"year: {meta.get('year', meta.get('publication_date', ''))}",
            f"primary_topic: {clean_html(str(meta.get('primary_topic', '') or ''))}"
                if meta.get("primary_topic") else None,
            f"summary: {summary_text}" if summary_text else None,
            f"snippet: {clean_html(snippet)}" if snippet else None,
        ]
        blocks.append("\n".join(ln for ln in lines if ln is not None))
    return "\n\n".join(blocks)
 
def dedupe_ci(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for item in items:
        key = collapse_whitespace(item).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out
 
def is_placeholder_anchor_value(value: str) -> bool:
    raw = collapse_whitespace(str(value or "")).lower()
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    if not compact:
        return True
    return compact in {
        "na", "nslasha", "unknown", "none", "null", "nil", "empty",
        "unspecified", "notavailable", "notapplicable", "notprovided", "tbd",
        "untitled", "notitle", "notitled", "unnamed",
    }
 
def normalize_anchor(anchor: Any) -> Dict[str, Any]:
    if not isinstance(anchor, dict):
        return {}
    value = collapse_whitespace(str(anchor.get("value", "") or ""))
    if not value or is_placeholder_anchor_value(value):
        return {}
    try:
        confidence = max(0.0, min(1.0, float(anchor.get("confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "type": collapse_whitespace(str(anchor.get("type", "") or "")).lower() or "metadata",
        "value": value,
        "source": collapse_whitespace(str(anchor.get("source", "") or "")) or "retrieval",
        "confidence": confidence,
    }
 
def anchor_in_text(anchor_value: str, text: str) -> bool:
    if is_placeholder_anchor_value(anchor_value):
        return False
    a, t = norm_text(anchor_value), norm_text(text)
    if not a or not t:
        return False
    if a in t:
        return True
    a_toks = [tok for tok in tokenize_words(a) if len(tok) >= 3]
    return len(a_toks) >= 2 and all(tok in set(tokenize_words(t)) for tok in a_toks)
 
def anchor_is_stable(anchor: Dict[str, Any]) -> bool:
    data = normalize_anchor(anchor)
    if not data:
        return False
    return float(data.get("confidence", 0.0) or 0.0) >= float(
        getattr(settings, "anchor_stable_confidence", 0.72))
 
def anchor_support_ratio(anchor_value: str, docs: List[Any]) -> float:
    anchor = collapse_whitespace(str(anchor_value or ""))
    if not anchor or not docs or is_placeholder_anchor_value(anchor):
        return 0.0
 
    exact_count = sum(1 for d in docs if anchor_in_text(anchor, doc_haystack(d)))
    if exact_count > 0:
        return float(exact_count) / max(1, len(docs))
 
    toks = [t for t in re.findall(r"[A-Za-z]+", anchor) if t]
    if len(toks) >= 2:
        first, last = toks[0].lower(), toks[-1].lower()
        def _name_matches_doc(d):
            meta = getattr(d, "metadata", None) or {}
            for field in ("researcher", "authors"):
                raw = collapse_whitespace(str(meta.get(field, "") or "")).lower()
                if not raw:
                    continue
                if first in raw and last in raw:
                    return True
                if last in raw and re.search(rf"\b{re.escape(first[0])}\.?\s", raw):
                    return True
            return False
        fuzzy_count = sum(1 for d in docs if _name_matches_doc(d))
        if fuzzy_count > 0:
            return float(fuzzy_count) / max(1, len(docs))
 
    return 0.0
 
def retrieval_confidence_label(*, docs_count: int, anchor_consistent: bool) -> str:
    min_docs = max(1, int(getattr(settings, "retrieval_weak_min_docs", 3)))
    if docs_count <= min_docs:
        return "weak"
    if not anchor_consistent:
        return "inconsistent"
    if docs_count >= max(8, min_docs * 2):
        return "high"
    if docs_count >= max(4, min_docs + 1):
        return "medium"
    return "low"
 
def classify_generic_intent(question: str) -> str:
    """Classify question intent using grammatical structure only.
    No hardcoded domain terms — detection is based entirely on
    sentence structure (interrogatives, verb forms, word order)."""
    q = (question or "").strip().lower()
    if not q:
        return "default"
 
    if re.search(r"\b(?:compar\w*|differ\w*|versus|vs\.?|similarit\w*)\b", q):
        return "comparison"
 
    if re.search(r"\b(?:when\s+(?:did|was|were|is)|(?:what|which)\s+(?:year|time\s+period|date\s+range|time\s+range|time\s+frame))\b", q):
        return "time_range"
 
    if re.search(r"\b(?:who|what|which)\s+else\b", q):
        return "list"
    if re.match(r"^\s*(?:identify|list|name|find|suggest|show)\b", q):
        return "list"
    if re.search(r"\b(?:who|what)\s+(?:are|were|study|studies|work\s+on|works?\s+in)\b", q):
        return "list"
    if re.search(r"\bshow\s+me\b", q):
        return "list"
    if re.match(r"^\s*which\b", q):
        return "list"
 
    return "default"
 
_PROMPT_LEAK_MARKERS = (
    "papers:", "paper context:", "question:",
    "system:", "user:", "assistant:",
    "[paper", "[mem", "[recent", "[summary",
)
 
_PROMPT_LEAK_PREFIX_RE = re.compile(
    r"^\s*(?:papers\s*:|paper context\s*:|question\s*:|system\s*:|user\s*:|assistant\s*:"
    r"|\[paper|\[mem|\[recent|\[summary)",
    re.IGNORECASE,
)
 
def strip_prompt_leak(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return a
    if any(m in a.lower() for m in _PROMPT_LEAK_MARKERS):
        a = "\n".join(ln.strip() for ln in a.splitlines()
                      if not _PROMPT_LEAK_PREFIX_RE.match(ln)).strip()
    return re.sub(r"\n{3,}", "\n\n", a).strip()
 
def looks_like_person_candidate(name: str) -> bool:
    if not name:
        return False
    cleaned_name = re.sub(r"\s+", " ", str(name).strip())
    if not cleaned_name:
        return False
    if ":" in cleaned_name or "?" in cleaned_name:
        return False
    toks = cleaned_name.split()
    if len(toks) < 2 or len(toks) > 4:
        return False
    if any(re.search(r"\d", t) for t in toks):
        return False
    if all(len(t.replace(".", "")) <= 1 for t in toks):
        return False
    if not all(re.match(r"^[A-Za-z][A-Za-z\.\-']*$", t) for t in toks):
        return False
    lower_toks = [t.rstrip(".").lower() for t in toks]
    _NON_NAME_WORDS = {
        "and", "but", "for", "nor", "yet", "not", "with", "without",
        "the", "from", "into", "onto", "upon", "over", "under", "about",
        "between", "through", "during", "before", "after", "above", "below",
        "studies", "study", "research", "using", "based", "toward",
        "towards", "within", "across", "among", "along",
        "new", "old", "how", "why", "what", "when", "where", "which",
        "does", "did", "has", "had", "was", "were", "are", "been",
    }
    if any(tok in _NON_NAME_WORDS for tok in lower_toks):
        return False
    stopset = get_stopword_set()
    if stopset and (
        lower_toks[0] in stopset or lower_toks[-1] in stopset
        or any(tok in stopset for tok in lower_toks[1:-1])
    ):
        return False
    if any(is_generic_query_token(tok) for tok in lower_toks if len(tok) > 1):
        return False
 
    has_initial = any(re.fullmatch(r"[A-Z]\.", t) for t in toks)
    lexical_parts = []
    for t in toks:
        bare = t.rstrip(".")
        if len(bare) <= 1:
            continue
        if t.isupper() and len(bare) > 2:
            return False
        if len(bare) > 3 and bare[0].islower():
            return False
        lexical_parts.append(bare.lower())
 
    real_parts = [t for t in toks if len(t.rstrip(".")) >= 2 and t[0].isupper()]
    if len(real_parts) < 2 and not (len(real_parts) == 1 and has_initial):
        return False
 
    english_words = get_english_word_set()
    name_tokens = get_name_token_set()
    if lexical_parts and english_words:
        non_last_parts = lexical_parts[:-1] or lexical_parts
        name_hits = sum(1 for t in non_last_parts if t in name_tokens)
        name_hits_all = sum(1 for t in lexical_parts if t in name_tokens)
        uncommon_hits = sum(1 for t in non_last_parts if t not in english_words)
        english_hits = sum(1 for t in lexical_parts if t in english_words)
        if not has_initial and name_hits_all == 0 and name_hits == 0 and uncommon_hits == 0 and english_hits == len(lexical_parts):
            return False
 
    suffix_like_parts = [t for t in lexical_parts if re.search(
        r"(?:tion|sion|ment|ness|ity|ship|isms?|ence|ance|ing|ed|ers?"
        r"|ial|ive|ory|ics|ent|ular|ior|ary|ical|ise|ize|ous|ated|ised|ized)$", t)]
    if lexical_parts and len(suffix_like_parts) >= max(2, len(lexical_parts) - 1) and not has_initial:
        return False
    if (not has_initial and len(lexical_parts) >= 2 and lexical_parts[-1].endswith("s")
        and suffix_like_parts):
        return False
 
    if not has_initial and lexical_parts:
        n_name = 0
        n_common = 0
        n_uncommon = 0
 
        for t in lexical_parts:
            if name_tokens and t in name_tokens:
                n_name += 1
            elif _wordnet_is_common_word(t):
                n_common += 1
            elif english_words and t in english_words:
                n_common += 1
            else:
                n_uncommon += 1
 
        n_total = len(lexical_parts)
 
        if n_common > 0 and n_name == 0:
            return False
 
        if n_name == 0 and n_uncommon == 0:
            return False
 
        if n_name == 1 and n_uncommon == 0 and n_total >= 2:
            if lexical_parts[0] not in (name_tokens or set()):
                return False
 
    return True
 
def strip_possessive(name: str) -> str:
    s = (name or "").strip()
    for suffix in ("\u2019s", "'s"):
        if s.endswith(suffix):
            return s[:-len(suffix)].strip()
    for suffix in ("\u2019", "'"):
        if s.endswith(suffix):
            return s[:-1].strip()
    return s
 
def tokenize_name(name: str) -> List[str]:
    """Split a person name into alphabetic tokens.

    >>> tokenize_name("Duncan A. Brown")
    ['Duncan', 'A', 'Brown']
    """
    return [t for t in re.findall(r"[A-Za-z]+", str(name or "").strip()) if t]

def generate_name_variants(toks: List[str]) -> List[str]:
    """Generate common academic name variants from a token list.

    Given ``["Duncan", "Andrew", "Brown"]`` produces (most specific first)::

        Duncan Andrew Brown      -- all tokens full
        Duncan A. Brown          -- middles as initials
        D. A. Brown              -- all pre-surname tokens as initials
        Duncan Brown             -- middles omitted
        D. Brown                 -- first initial, middles omitted

    Works for any number of middle names (0, 1, 2, ...).
    """
    if len(toks) < 2:
        return list(toks)
    first, last = toks[0], toks[-1]
    middles = toks[1:-1]

    variants: List[str] = []
    seen: set = set()

    def _add(v: str) -> None:
        v = v.strip()
        key = re.sub(r"\s+", " ", v).lower()
        if key and key not in seen:
            seen.add(key)
            variants.append(v)

    if middles:
        mid_full = " ".join(middles)
        mid_init = " ".join(f"{m[0]}." for m in middles)
        _add(f"{first} {mid_full} {last}")
        _add(f"{first} {mid_init} {last}")
        _add(f"{first[0]}. {mid_init} {last}")

    _add(f"{first} {last}")
    _add(f"{first[0]}. {last}")
    _add(f"{first[0]} {last}")

    return variants

def split_author_names(raw_authors: str) -> List[str]:
    """Split a raw authors string on ``;``, ``,``, or ``and``.

    Returns a list of cleaned author name strings (>= 2 chars each).
    """
    if not raw_authors:
        return []
    return [n for p in re.split(r"\s*[;,]\s*|\s+and\s+", raw_authors)
            if (n := re.sub(r"\s+", " ", p.strip())) and len(n) >= 2]

def has_explicit_entity_signal(question: str, ents: Optional[Dict[str, List[str]]] = None) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if re.search(r"\"[^\"]{3,120}\"", q):
        return True
    from rag_engine import _extract_entities_basic
    data = ents if ents is not None else _extract_entities_basic(q, max_items=4)
    if data.get("people") or data.get("orgs") or data.get("entities"):
        return True
    topic_tokens = [t for t in data.get("topics", []) if not is_generic_query_token(t)]
    return len(topic_tokens) >= max(1, int(getattr(settings, "retrieval_topic_min_terms", 2)))
 
def short_hash(value: Any, *, length: int = 12) -> str:
    data = norm_text(str(value or "")).encode("utf-8", errors="ignore")
    return hashlib.sha256(data).hexdigest()[:max(6, int(length))]
 
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
 
_META_COMMAND_PATTERNS = re.compile(
    r"^\s*(switch\s+topic|change\s+(?:the\s+)?(?:subject|topic)|new\s+topic"
    r"|different\s+topic|start\s+over|let'?s?\s+(?:talk|discuss)\s+(?:about\s+)?something\s+else"
    r"|never\s*mind|forget\s+(?:it|that)|reset|clear\s+(?:context|history|conversation)"
    r"|move\s+on|next\s+topic)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
 
def is_meta_command(question: str) -> bool:
    """Return True if the question is a conversational meta-command, not a research query."""
    q = (question or "").strip()
    return bool(q and _META_COMMAND_PATTERNS.match(q))
 
def anchor_query_overlap(anchor_value: str, question: str) -> bool:
    """Return True if the question has meaningful token overlap with the anchor value."""
    a_val = norm_text(anchor_value)
    q_val = norm_text(question)
    if not a_val or not q_val:
        return False
    if a_val in q_val or q_val in a_val:
        return True
    a_toks = set(t for t in tokenize_words(a_val) if len(t) >= 3)
    q_toks = set(t for t in tokenize_words(q_val) if len(t) >= 3 and not is_generic_query_token(t))
    if not a_toks:
        return False
    return bool(a_toks & q_toks)
 
_INSTITUTIONAL_TERMS_RE = re.compile(
    r"\b(syracuse\s+university|syracuse|university|faculty|professor|"
    r"researcher|department|college|school|campus|institute)\b",
    re.IGNORECASE,
)
 
def query_tokens_for_relevance(question: str) -> List[str]:
    q = _INSTITUTIONAL_TERMS_RE.sub("", (question or "").strip().lower())
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        return []
    stopset = get_stopword_set()
    min_len = max(1, int(getattr(settings, "retrieval_keyword_min_term_len", 3)))
    out, seen = [], set()
    for t in re.findall(r"[a-z0-9\-]{2,}", q):
        if len(t) >= min_len and (not stopset or t not in stopset) and t not in seen:
            seen.add(t)
            out.append(t)
    return out