# runtime_settings.py
import os
from dataclasses import dataclass
 
_SENTINEL = object()
 
 
def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default
 
def _env_int(name: str, default: int) -> int:
    try: return int(_env(name, str(default)))
    except Exception: return int(default)
 
def _env_float(name: str, default: float) -> float:
    try: return float(_env(name, str(default)))
    except Exception: return float(default)
 
def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "y", "on"}
 
 
@dataclass
class RuntimeSettings:
    # Core
    active_mode: str = _env("RAG_ACTIVE_MODE", "full")
    llm_model: str = _env("RAG_LLM_MODEL", "llama-3.2-3b")
    use_graph: bool = _env_bool("RAG_USE_GRAPH", False)
    stateless_default: bool = _env_bool("RAG_STATELESS_DEFAULT", False)
    debug_rag: bool = _env_bool("RAG_DEBUG", True)
    force_gpu: bool = _env_bool("RAG_FORCE_GPU", True)
 
    # Generation
    answer_max_new_tokens: int = _env_int("RAG_ANSWER_MAX_NEW_TOKENS", _env_int("RAG_MAX_NEW_TOKENS", 1024))
    llm_timeout_s: int = _env_int("RAG_LLM_TIMEOUT_S", 300)
    prompt_doc_text_limit: int = _env_int("RAG_PROMPT_DOC_TEXT_LIMIT", 600)
    prompt_max_docs: int = _env_int("RAG_PROMPT_MAX_DOCS", 16)
 
    # Search
    search_k: int = _env_int("RAG_SEARCH_K", 60)
    search_fetch_k: int = _env_int("RAG_SEARCH_FETCH_K", 150)
    mmr_lambda: float = _env_float("RAG_MMR_LAMBDA", 0.6)
 
    # Budgets
    budget_memory: int = _env_int("RAG_BUDGET_MEMORY", 900)
    budget_papers: int = _env_int("RAG_BUDGET_PAPERS", 20000)
    trigger_tokens: int = _env_int("RAG_TRIGGER_TOKENS", 6000)
 
    # Memory
    memory_max_per_session: int = _env_int("RAG_MEMORY_MAX_PER_SESSION", 500)
    memory_prune_target: int = _env_int("RAG_MEMORY_PRUNE_TARGET", 420)
    memory_persist_every_n_adds: int = _env_int("RAG_MEMORY_PERSIST_EVERY_N_ADDS", 25)
    memory_extract_first_turn: bool = _env_bool("RAG_MEMORY_EXTRACT_FIRST_TURN", True)
    qa_cache_enable: bool = _env_bool("RAG_QA_CACHE_ENABLE", False)
 
    # Retrieval tuning
    retrieval_dual_query: bool = _env_bool("RAG_RETRIEVAL_DUAL_QUERY", True)
    retrieval_keyword_min_term_len: int = _env_int("RAG_RETRIEVAL_KEYWORD_MIN_TERM_LEN", 3)
    retrieval_topic_min_terms: int = _env_int("RAG_RETRIEVAL_TOPIC_MIN_TERMS", 2)
    dominant_majority_ratio: float = _env_float("RAG_DOMINANT_MAJORITY_RATIO", 0.6)
    dominant_min_count: int = _env_int("RAG_DOMINANT_MIN_COUNT", 3)
    dominant_min_confidence: float = _env_float("RAG_DOMINANT_MIN_CONFIDENCE", 0.72)
    dominant_replace_confidence: float = _env_float("RAG_DOMINANT_REPLACE_CONFIDENCE", 0.82)
    metadata_filter_min_results: int = _env_int("RAG_METADATA_FILTER_MIN_RESULTS", 4)
    retrieval_weak_min_docs: int = _env_int("RAG_RETRIEVAL_WEAK_MIN_DOCS", 3)
    anchor_stable_confidence: float = _env_float("RAG_ANCHOR_STABLE_CONFIDENCE", 0.72)
    anchor_consistency_min_ratio: float = _env_float("RAG_ANCHOR_CONSISTENCY_MIN_RATIO", 0.45)
    low_conf_prompt_max_docs: int = _env_int("RAG_LOW_CONF_PROMPT_MAX_DOCS", 10)
    low_conf_prompt_doc_text_limit: int = _env_int("RAG_LOW_CONF_PROMPT_DOC_TEXT_LIMIT", 450)
    low_conf_ner_context_max_docs: int = _env_int("RAG_LOW_CONF_NER_CONTEXT_MAX_DOCS", 6)
 
    # Follow-up detection
    followup_pronoun_regex: str = _env("RAG_FOLLOWUP_PRONOUN_REGEX",
        r"\b(him|her|them|they|it|this|that|those|these|he|she|his|hers|their|there)\b")
    followup_phrases: str = _env("RAG_FOLLOWUP_PHRASES",
        "who else,what else,tell me more,more about,what field,which field")
    followup_query_max_words: int = _env_int("RAG_FOLLOWUP_QUERY_MAX_WORDS", 8)
    followup_k_mult: float = _env_float("RAG_FOLLOWUP_K_MULT", 2.0)
    followup_fetch_k_mult: float = _env_float("RAG_FOLLOWUP_FETCH_K_MULT", 2.0)
    generic_query_terms: str = _env("RAG_GENERIC_QUERY_TERMS",
        "about,again,also,anything,anyone,can,could,does,else,field,give,help,him,her,it,its,"
        "know,me,more,need,other,others,please,tell,that,them,they,those,this,what,which,who,whom,why,work")
    generic_token_min_len: int = _env_int("RAG_GENERIC_TOKEN_MIN_LEN", 3)
 
    # NER / Summary
    ner_context_max_docs: int = _env_int("RAG_NER_CONTEXT_MAX_DOCS", 18)
    summary_max_chars: int = _env_int("RAG_SUMMARY_MAX_CHARS", 1400)
    summary_max_items_per_field: int = _env_int("RAG_SUMMARY_MAX_ITEMS_PER_FIELD", 6)
    summary_recent_turns_keep: int = _env_int("RAG_SUMMARY_RECENT_TURNS_KEEP", 8)
    recent_turns_in_prompt: int = _env_int("RAG_RECENT_TURNS_IN_PROMPT", 2)
 
    # Rewrite
    rewrite_enable: bool = _env_bool("RAG_REWRITE_ENABLE", True)
    rewrite_timeout_s: int = _env_int("RAG_REWRITE_TIMEOUT_S", 10)
    rewrite_max_recent_turns: int = _env_int("RAG_REWRITE_MAX_RECENT_TURNS", 3)
    rewrite_max_chars: int = _env_int("RAG_REWRITE_MAX_CHARS", 220)
 
    # Rerank
    rerank_enable: bool = _env_bool("RAG_RERANK_ENABLE", False)
    rerank_candidate_k: int = _env_int("RAG_RERANK_CANDIDATE_K", 30)
    rerank_final_k: int = _env_int("RAG_RERANK_FINAL_K", 12)
    rerank_timeout_s: int = _env_int("RAG_RERANK_TIMEOUT_S", 12)
    rerank_w_token: float = _env_float("RAG_RERANK_W_TOKEN", 1.0)
    rerank_w_person: float = _env_float("RAG_RERANK_W_PERSON", 3.0)
    rerank_w_anchor: float = _env_float("RAG_RERANK_W_ANCHOR", 2.0)
    rerank_w_chunk: float = _env_float("RAG_RERANK_W_CHUNK", 0.5)
    rerank_surname_penalty: float = _env_float("RAG_RERANK_SURNAME_PENALTY", 0.6)
    fulltext_fallback_enable: bool = _env_bool("RAG_FULLTEXT_FALLBACK_ENABLE", True)
 
    # Models
    answer_model_key: str = _env("RAG_ANSWER_MODEL_KEY", "llama-3.2-3b")
    utility_model_key: str = _env("RAG_UTILITY_MODEL_KEY", "llama-3.2-1b")
    llama_1b_path: str = _env("LLAMA_1B", _env("RAG_LLAMA_1B_PATH", ""))
    llama_8b_path: str = _env("LLAMA_8B", _env("RAG_LLAMA_8B_PATH", ""))
    gemma_4_e2b_path: str = _env("GEMMA_4_E2B", _env("RAG_GEMMA_4_E2B_PATH", "/home/arapte/models/gemma-4-E2B-it"))
    gemma_4_e4b_path: str = _env("GEMMA_4_E4B", _env("RAG_GEMMA_4_E4B_PATH", "/home/arapte/models/gemma-4-E4B-it"))
    gemma_4_26b_path: str = _env("GEMMA_4_26B", _env("RAG_GEMMA_4_26B_PATH", "/home/arapte/models/gemma-4-26B-A4B-it"))
    gemma_4_31b_path: str = _env("GEMMA_4_31B", _env("RAG_GEMMA_4_31B_PATH", "/home/arapte/models/gemma-4-31B-it"))
    qwen_14b_path: str = _env("QWEN_14B", _env("RAG_QWEN_14B_PATH", "/home/arapte/models/Qwen2.5-14B-Instruct"))
    gpt_oss_20b_path: str = _env("GPT_OSS_20B", _env("RAG_GPT_OSS_20B_PATH", "/home/arapte/models/gpt-oss-20b"))
    quantize_8bit: bool = _env_bool("RAG_QUANTIZE_8BIT", False)
    utility_max_new_tokens: int = _env_int("RAG_UTILITY_MAX_NEW_TOKENS", 256)
    utility_queue_max: int = _env_int("RAG_UTILITY_QUEUE_MAX", 2000)
    enable_utility_background: int = _env_int("RAG_ENABLE_UTILITY_BACKGROUND", 1)
    enable_llm_summary_regen: bool = _env_bool("RAG_ENABLE_LLM_SUMMARY_REGEN", False)
    allow_utility_concurrency: bool = _env_bool("RAG_ALLOW_UTILITY_CONCURRENCY", False)
 
    # Session
    session_turns_keep: int = _env_int("RAG_SESSION_TURNS_KEEP", 24)
    session_turns_max_chars: int = _env_int("RAG_SESSION_TURNS_MAX_CHARS", 32000)
    session_turn_trim_target_chars: int = _env_int("RAG_SESSION_TURN_TRIM_TARGET_CHARS", 24000)
    summary_compress_threshold_chars: int = _env_int("RAG_SUMMARY_COMPRESS_THRESHOLD_CHARS", 1400)
 
    # Topic-pivot / dangling-pronoun handling
    person_pronoun_regex: str = _env("RAG_PERSON_PRONOUN_REGEX",
        r"\b(he|she|him|her|his|hers|they|them|their)\b")
    topic_inject_min_chars: int = _env_int("RAG_TOPIC_INJECT_MIN_CHARS", 4)
    dangling_pronoun_min_injected: int = _env_int("RAG_DANGLING_PRONOUN_MIN_INJECTED", 2)
    dangling_pronoun_min_raw_substantive: int = _env_int("RAG_DANGLING_PRONOUN_MIN_RAW_SUBSTANTIVE", 2)
    dangling_pronoun_hint_max_chars: int = _env_int("RAG_DANGLING_PRONOUN_HINT_MAX_CHARS", 60)

    _CACHE_BUSTING_FIELDS: frozenset = frozenset({
        "generic_query_terms", "generic_token_min_len",
        "followup_phrases", "followup_pronoun_regex",
        "person_pronoun_regex",
    })
 
    def __setattr__(self, name: str, value: object) -> None:
        old = getattr(self, name, _SENTINEL)
        super().__setattr__(name, value)
        if old is not _SENTINEL and old != value and name in self._CACHE_BUSTING_FIELDS:
            try:
                from rag_utils import bust_caches
                bust_caches(name)
            except Exception:
                pass
 
 
settings = RuntimeSettings()