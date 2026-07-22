# Syracuse Research Assistant

A local retrieval augmented generation application for Syracuse research discovery. The repository ingests paper records from SQLite into Chroma, retrieves and filters evidence, tracks short term conversational state across turns, and serves answers through a Streamlit chat interface, an interactive terminal chat, or a benchmark harness. It also includes an optional graph view that turns retrieved papers into a lightweight entity relationship network.

This README is intentionally code aware. It explains how the repository works, what each major module does, how the main functions contribute to the pipeline, and how data moves from ingestion to answer generation.

## Table of contents

1. Project summary
2. Main capabilities
3. Repository structure
4. End to end flow
5. File by file technical reference
6. Retrieval behavior
7. Data contracts
8. Setup and local execution
9. Running the three entry points
10. Runtime operations
11. Strengths and limitations
12. Recommended next improvements

## Project summary

At a practical level, the application solves four problems:

1. It converts paper metadata, summaries, and full text stored in SQLite into Chroma documents with normalized metadata, using sentence aware overlapping chunking so that context is preserved at chunk boundaries.
2. It retrieves relevant paper chunks for a user question, with additional logic for follow ups, person centered questions, anchor stability, and retrieval confidence.
3. It maintains a rolling summary and recent turns so follow up questions can be resolved without losing topic continuity.
4. It renders the interaction in three surfaces — Streamlit, an interactive terminal chat, and a benchmark harness — and supports switching between multiple datasets and multiple answer models at runtime.

## Main capabilities

### Multiple datasets under one UI

Three corpus modes are available at runtime:

1. `full` — the legacy SQLite backed corpus with full paper text, named "Legacy DB" in the UI.
2. `openalex` — papers ingested from OpenAlex with Docling extracted full text, named "OpenAlex DB".
3. `abstracts` — OpenAlex records using abstracts only, named "Abstracts Only".

Each mode has its own Chroma directory, Chroma collection, and Neo4j database. Switching datasets in the Streamlit sidebar or via the `/db` command in the terminal chat flips all three together, and the graph config follows the active dataset automatically through `config_graph.get_neo4j_db()`.

### Multiple answer models with 4 bit quantization

Five answer models are available and can be selected at runtime:

1. LLaMA 3.2 3B — runs unquantized on a single consumer GPU.
2. LLaMA 3.1 8B — loaded in 4 bit.
3. Gemma 3 12B — loaded in 4 bit.
4. Qwen 2.5 14B — loaded in 4 bit.
5. GPT-OSS 20B — loaded in 4 bit.

Model switching is handled by `EngineManager.switch_answer_model()`. When a quantized answer model is loaded and free VRAM falls below a headroom threshold, the engine evicts the utility model to protect answer generation capacity.

### Retrieval first, generation second

The system is built so retrieval drives the answer path. The prompt instructs the answer model to stay grounded in the Syracuse research corpus, and several downstream checks downgrade or replace unsafe answers when retrieval is weak.

### Conversation continuity through anchors

The system tracks a current anchor, which is the dominant subject inferred from recent retrieval and conversation state. Short or pronoun based follow ups can be interpreted relative to that anchor when evidence is strong enough. Anchor validation checks both the raw user question and the resolved or rewritten question, preventing spurious anchor drift when follow up queries use pronouns.

### Weak evidence handling

The code distinguishes between confident retrieval, weak retrieval, and inconsistent retrieval. When evidence quality drops, the prompt is narrowed, the guardrails become stricter, and the pipeline can fall back to safer extractive answers. Confidence downshifting for person queries is calibrated so that initial format name matches such as D. Brown matching Duncan Brown are treated as strong evidence rather than triggering unnecessary downgrades.

### Citation grounding and hallucination prevention

After the answer model generates a response, the pipeline validates quoted paper titles against the actual retrieved document set using fuzzy matching. Lines containing fabricated citations that do not match any retrieved paper are stripped before the answer reaches the user. This complements the researcher grounding check, which uses bidirectional name matching to avoid false positives when metadata stores names in initial format.

### Meta query handling

Questions like "how many papers does the corpus contain", "what is the most recent paper", and short commands such as "switch topic" are intercepted before retrieval and answered directly from the active Chroma collection or by clearing state. This avoids wasting a retrieval round on structural questions about the corpus itself.

### Local friendly runtime behavior

The repository is designed for local model paths, persistent Chroma storage, offline friendly execution, and explicit session reset controls.

## Repository structure

```
.
|-- benchmark_rag.py
|-- chroma_ingest.py
|-- config_full.py
|-- config_graph.py
|-- conversation_memory.py
|-- database_manager.py
|-- rag_chat.py
|-- rag_engine.py
|-- rag_graph.py
|-- rag_pipeline.py
|-- rag_utils.py
|-- runtime_settings.py
|-- session_store.py
`-- streamlit_app.py
```

## End to end flow

```
flowchart TD
    U[User submits question] --> UI1[Entry point appends user message]
    UI1 --> P0[rag_pipeline.answer_question]

    P0 --> V1[Validate question and check for meta commands like switch topic]
    V1 --> META{Meta query like how many papers or most recent paper}
    META -->|yes| METAANS[Answer directly from Chroma collection and return]
    META -->|no| M1[get_global_manager and runtime settings]

    M1 --> SW1[Switch active dataset and answer model if changed]
    SW1 --> S1[Load persistent session state from SessionStore]
    S1 --> S2[Read rolling summary recent turns extra state and current anchor]

    S2 --> I1[Classify broad intent and summary intent]
    I1 --> F1[Detect follow up or coreference query]
    F1 --> F2{Short query pronoun query or follow up phrase}
    F2 -->|yes| A1[Inspect current anchor and rolling summary]
    F2 -->|no| Q1[Use original question as retrieval basis]

    A1 --> A2{Anchor stable and supported enough}
    A2 -->|yes| Q2[Inject anchor or rewrite retrieval text]
    A2 -->|no| Q1

    Q1 --> R0[Build retrieval query text]
    Q2 --> R0

    R0 --> R1[Open active Chroma collection for current mode]
    R1 --> R3[Run vector retrieval with search_k and fetch_k budgets]
    R3 --> R4{Dual query or expanded follow up retrieval enabled}
    R4 -->|yes| R5[Run secondary retrieval path and merge candidates]
    R4 -->|no| R6[Continue with initial candidates]
    R5 --> D0
    R6 --> D0

    D0[Deduplicate candidate docs by paper and chunk] --> D1[Build haystacks and normalize metadata]
    D1 --> D2[Filter noisy docs using relevance tokens from the question]
    D2 --> P1{Person centered query detected}

    P1 -->|yes| P2[Extract person name from question]
    P2 --> P3[Build name signatures and score docs for person support]
    P3 --> P5[Select person focused subset]
    P5 --> G0

    P1 -->|no| G0[Proceed with filtered docs]

    G0 --> G1[Analyze metadata dominance across retrieved docs]
    G1 --> G2[Estimate anchor support ratio with fuzzy person name matching]
    G2 --> G3[Assign retrieval confidence label]
    G3 --> G4{Weak or inconsistent retrieval}

    G4 -->|yes| G5[Downshift confidence and reduce prompt doc limits]
    G4 -->|no| G6[Keep normal prompt budget]

    G5 --> AN1[Build candidate anchor from dominance analysis]
    G6 --> AN1
    AN1 --> AN2[Choose keep replace or ignore anchor using raw and resolved question overlap]

    AN2 --> CTX1[Build rolling summary block]
    CTX1 --> CTX2[Build recent turns block]
    CTX2 --> CTX3[Build compact context from docs grouped by researcher]
    CTX3 --> CTX4[Fit context to token budget with text first shrink strategy]

    CTX4 --> PR1[Compose grounded answer prompt]
    PR1 --> LLM1[Invoke answer model with timeout guard]
    LLM1 --> LLM2{Model returned usable answer}

    LLM2 -->|no| FB1[Fallback answer from retrieved docs]
    LLM2 -->|yes| HC1[Strip hallucinated citations not found in retrieved docs]

    HC1 --> SAN1[Sanitize raw answer text and remove prompt leakage]
    SAN1 --> SAN3{Answer mentions unsupported researchers using bidirectional matching}
    SAN3 -->|yes| FB2[Replace with supported researcher extract answer]
    SAN3 -->|no| SAN4[Keep sanitized answer]

    FB1 --> OUT1[Final answer selected]
    FB2 --> OUT1
    SAN4 --> OUT1

    OUT1 --> SUM1[Update rolling summary from question retrieval and answer]
    SUM1 --> ST1[Trim turns sanitize extra state and persist session]

    ST1 --> GC1{Graph mode enabled}
    GC1 -->|yes| GC2[rag_graph.graph_retrieve_from_paper_docs builds in memory graph]
    GC1 -->|no| PKG1[Build output payload without graph]
    GC2 --> PKG2[Build output payload with graph]

    PKG1 --> UI2[Return payload]
    PKG2 --> UI2
    UI2 --> UI4[Streamlit or terminal renders answer timing sources and graph]
```

### Memory and anchor lifecycle

```
flowchart LR
    Q[New user turn] --> S[Load rolling summary recent turns current anchor]
    S --> F{Follow up or coreference question}
    F -->|no| N1[No anchor injection needed]
    F -->|yes| A1[Check anchor stability]
    A1 --> A2{Anchor confidence and support ratio sufficient}
    A2 -->|yes| A3[Inject or rewrite query with anchor]
    A2 -->|no| A4[Do not trust prior anchor]

    N1 --> R[Retrieve documents]
    A3 --> R
    A4 --> R

    R --> D[Compute dominance and retrieval confidence]
    D --> U[Choose keep replace or clear anchor against raw and resolved question]
    U --> T[Trim recent turns]
    T --> RS[Update rolling summary]
    RS --> P[Persist summary turns anchor and extra state in SQLite]
    P --> NX[Next user turn uses updated state]
```

### Retrieval and fallback decision graph

```
flowchart TD
    Q0[Resolved retrieval query] --> R1[Retrieve Chroma candidates]
    R1 --> R2[Deduplicate and relevance filter]
    R2 --> P{Person query}
    P -->|yes| P1[Rank by person support with initial name matching]
    P -->|no| D1[Use filtered docs]
    P1 --> D1
    D1 --> C1[Compute dominance anchor support and confidence]
    C1 --> C2{Confidence high or medium}
    C2 -->|yes| G1[Normal prompt limits]
    C2 -->|no| G2[Low confidence prompt limits]
    G1 --> L1[Invoke answer model]
    G2 --> L1
    L1 --> H1[Strip hallucinated citations]
    H1 --> A1{Answer usable and grounded}
    A1 -->|yes| A2[Sanitize and return answer]
    A1 -->|no| F1[Fallback answer from docs]
    A2 --> A3{Unsupported researcher mentioned using bidirectional matching}
    A3 -->|yes| F2[Replace with supported researcher extract]
    A3 -->|no| DONE[Return final answer]
    F1 --> DONE
    F2 --> DONE
```

## File by file technical reference

## 1. `chroma_ingest.py`

This script builds the paper retrieval corpus from SQLite and stores it in Chroma. It is specific to a schema containing `research_info` and `works` tables.

### Main responsibilities

1. Read configuration for SQLite, Chroma, embedding model, and chunk sizing.
2. Detect the actual full text column name in the `works` table, tolerating typos such as `fultext`.
3. Join research metadata with summary and full text content.
4. Build one canonical paper document per paper id.
5. Chunk long paper text using a sentence aware overlapping splitter.
6. Upsert all chunks into a persistent Chroma collection in Chroma safe sub batches of 166.

### Chunking strategy

Chunking is controlled by `CHUNK_MAX_TOKENS` (default 512) and `CHUNK_OVERLAP_TOKENS` (default 128). The splitter first breaks the document into sentences, then greedily packs sentences into a chunk until the token estimate would exceed `CHUNK_MAX_TOKENS`. When a new chunk starts, tail sentences from the previous chunk are carried over up to `CHUNK_OVERLAP_TOKENS` so that context spanning a boundary is not lost. This is mirrored by the pipeline side chunker so retrieval and ingestion stay consistent.

### Functions

#### `safe_meta(val, default="N/A")`

Normalizes metadata values into Chroma safe scalar values. `None` becomes a default placeholder. Blank strings are replaced with the default. Numeric and boolean values are preserved.

#### `_pick_year(pub_date)`

Extracts a four digit year from a publication date string. If extraction fails, it falls back to a normalized publication date string.

#### `_token_est(text)`

Produces a rough word based token estimate used by the chunker.

#### `_sentence_split(text)`

Splits on sentence terminators to support the sentence aware chunker.

#### `chunk_with_overlap(text)`

Splits text into token bounded chunks with sentence aware overlap as described above.

#### `_join_nonempty(*parts, sep="newline")`

Concatenates only nonempty text fragments. Used to construct the final stored document body cleanly.

#### `detect_works_fulltext_column(conn)`

Checks the SQLite schema for possible full text column names and returns whichever exists. This is a compatibility safeguard for schema drift.

#### `fetch_rows(conn, works_fulltext_col)`

Runs the left join that pulls research metadata together with summary and full text into one result set.

#### `build_paper_document(row, works_fulltext_colname)`

Transforms one database row into a Chroma ready payload consisting of paper id, full document text, and metadata. The document text includes labeled sections for paper id, researcher, title, authors, primary topic, info, DOI, publication date, summary, and full text.

#### `main()`

Orchestrates the full ingestion run. It creates the Chroma client, initializes the embedding function, rebuilds the collection, groups rows by `paper_id`, selects the richest candidate row, chunks it with overlap, and batch upserts the chunks.

### Important implementation note

Both summary and full text are stored inside `page_content`. This is why later retrieval utilities parse summaries from document text rather than metadata.

## 2. `config_full.py`

This file provides environment driven defaults for every registered dataset, not just the primary one.

### Functions

#### `_env(name, default)`

Reads an environment variable and returns a default when it is missing or blank.

### Important values

1. `SQLITE_DB_FULL` for the SQLite source path of the legacy corpus
2. `CHROMA_DIR_FULL`, `CHROMA_COLLECTION_FULL` for the legacy Chroma store
3. `CHROMA_DIR_OPENALEX`, `CHROMA_COLLECTION_OPENALEX`, `NEO4J_DB_OPENALEX` for the OpenAlex full text dataset
4. `CHROMA_DIR_ABSTRACTS`, `CHROMA_COLLECTION_ABSTRACTS`, `NEO4J_DB_ABSTRACTS` for the OpenAlex abstracts only dataset
5. `LLAMA_1B`, `LLAMA_3B`, `LLAMA_8B`, `GEMMA_12B`, `QWEN_14B`, `GPT_OSS_20B` for local model paths
6. `EMBED_MODEL` for the embedding model path or name
7. `CHUNK_MAX_CHARS`, `PAPERS_PER_BATCH`, `CHROMA_MAX_BATCH` for ingestion throughput
8. `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`, `NEO4J_DB` for graph configuration

## 3. `config_graph.py`

This file isolates graph specific configuration and makes the Neo4j database selection follow the active dataset.

### Functions

#### `_env(name, default)`

Same environment lookup pattern as the main config.

#### `get_neo4j_db()`

Returns the Neo4j database name for the currently active dataset. It first tries to read the active `DatabaseManager` through the global engine manager, so switching datasets in the UI also switches the graph DB. It falls back to the static `NEO4J_DB` environment value when the manager has not yet been constructed.

### Important values

1. `NEO4J_URI`
2. `NEO4J_USER`
3. `NEO4J_PASS`
4. `NEO4J_DB` as a static fallback
5. `GRAPH_TOP_K`

In the provided repository snapshot, `rag_graph.py` builds the graph in memory from retrieved documents rather than querying Neo4j directly, so these settings are reserved for future or external graph workflows.

## 4. `runtime_settings.py`

This file is the runtime tuning surface of the application. It centralizes search size, prompt size, memory retention, follow up detection, reranking, model choice, rewrite behavior, utility worker behavior, and session persistence.

### Helper functions

#### `_env(name, default)`, `_env_int(...)`, `_env_float(...)`, `_env_bool(...)`

Typed environment variable readers with safe fallbacks.

### `RuntimeSettings`

This dataclass holds the active runtime configuration. Field groups and notable fields include:

#### Core runtime

`active_mode`, `llm_model`, `use_graph`, `stateless_default`, `debug_rag`, `force_gpu`.

#### Generation and prompt sizing

`answer_max_new_tokens`, `llm_timeout_s`, `prompt_doc_text_limit`, `prompt_max_docs`.

#### Retrieval sizing

`search_k`, `search_fetch_k`, `mmr_lambda`.

#### Budgets

`budget_memory`, `budget_papers`, `trigger_tokens`.

#### Memory

`memory_max_per_session`, `memory_prune_target`, `memory_persist_every_n_adds`, `memory_extract_first_turn`, `qa_cache_enable`.

#### Retrieval tuning

`retrieval_dual_query`, `retrieval_keyword_min_term_len`, `retrieval_topic_min_terms`, `dominant_majority_ratio`, `dominant_min_count`, `dominant_min_confidence`, `dominant_replace_confidence`, `metadata_filter_min_results`, `retrieval_weak_min_docs`, `anchor_stable_confidence`, `anchor_consistency_min_ratio`, `low_conf_prompt_max_docs`, `low_conf_prompt_doc_text_limit`, `low_conf_ner_context_max_docs`.

#### Follow up detection

`followup_pronoun_regex`, `followup_phrases`, `followup_query_max_words`, `followup_k_mult`, `followup_fetch_k_mult`, `generic_query_terms`, `generic_token_min_len`.

#### NER and summary

`ner_context_max_docs`, `summary_max_chars`, `summary_max_items_per_field`, `summary_recent_turns_keep`, `recent_turns_in_prompt`.

#### Rewrite

`rewrite_enable`, `rewrite_timeout_s`, `rewrite_max_recent_turns`, `rewrite_max_chars`.

#### Rerank

`rerank_enable`, `rerank_candidate_k`, `rerank_final_k`, `rerank_timeout_s`, `rerank_w_token`, `rerank_w_person`, `rerank_w_anchor`, `rerank_w_chunk`, `rerank_surname_penalty`, `fulltext_fallback_enable`.

#### Models

`answer_model_key`, `utility_model_key`, `llama_1b_path`, `llama_8b_path`, `gemma_12b_path`, `qwen_14b_path`, `gpt_oss_20b_path`, `quantize_8bit` (controls whether large models use 4 bit quantization in the current engine), `utility_max_new_tokens`, `utility_queue_max`, `enable_utility_background`, `enable_llm_summary_regen`, `allow_utility_concurrency`.

#### Session

`session_turns_keep`, `session_turns_max_chars`, `session_turn_trim_target_chars`, `summary_compress_threshold_chars`.

#### Topic pivot and dangling pronoun handling

`person_pronoun_regex`, `topic_inject_min_chars`, `dangling_pronoun_min_injected`, `dangling_pronoun_min_raw_substantive`, `dangling_pronoun_hint_max_chars`.

### `RuntimeSettings.__setattr__(...)`

Intercepts changes to cache busting fields (`generic_query_terms`, `generic_token_min_len`, `followup_phrases`, `followup_pronoun_regex`, `person_pronoun_regex`) and calls `rag_utils.bust_caches(...)` so cached regexes and token sets are rebuilt on next use. This prevents stale query parsing behavior after live runtime changes.

### `settings = RuntimeSettings()`

Creates the shared runtime settings instance imported across the codebase.

## 5. `database_manager.py`

This file abstracts corpus mode selection. Three modes are registered by default: `full`, `openalex`, and `abstracts`. Each mode carries its own Chroma directory, collection name, Neo4j database name, and human readable display label.

### Types and functions

#### `DatabaseConfig`

Dataclass describing one searchable corpus mode with `mode`, `chroma_dir`, `collection`, `description`, `neo4j_db`, and `display_label`.

#### `DatabaseManager.__init__()`

Creates the registry and registers the three built in modes.

#### `register_config(name, cfg)`

Adds a named configuration mode.

#### `resolve_mode(requested_mode)`

Resolves a requested mode case insensitively and falls back to the first available mode when needed.

#### `switch_config(name)`

Makes the resolved mode active.

#### `get_active_config()`, `get_config(name)`, `list_configs()`

Accessors for active and registered configs.

#### `ensure_dirs_exist()`

Creates configured Chroma directories if they do not already exist.

#### `display_labels()`

Returns the mode to display label mapping used by the sidebar and terminal chat.

#### `get_active_neo4j_db()`

Returns the Neo4j database name for the active config, with fallback to the static `NEO4J_DB`.

#### `validate_active_config()`

Post switch health check. Opens the active Chroma collection and returns a dict with `healthy` and `doc_count`, or a failure reason. The benchmark and terminal chat both use this to warn when a selected dataset is missing or empty.

## 6. `conversation_memory.py`

This file is now a thin utility layer that contains only the hard reset entry point.

### Functions

#### `hard_reset_memory(user_key)`

Performs the strongest memory reset. It first tries to reset state through the global engine manager. If that fails, it directly resets the `SessionStore` backed SQLite state and then removes any persisted entries for the session from the Chroma memory collection at `RAG_MEMORY_DIR`. All failures are logged but swallowed so a partial reset never throws to the UI.

Note that earlier in project history this file hosted process level QA and pipeline caches. Those have been removed in favor of always fresh retrieval, which simplifies reasoning about correctness and makes the Streamlit sidebar operate with just Reset Memory and Restart Conversation controls.

## 7. `session_store.py`

This file is the durable session state store backed by SQLite.

### Persistent schema

The `chat_state` table stores:

1. `session_id`
2. `rolling_summary`
3. `turns_json`
4. `extra_state_json`

The store automatically performs an ADD COLUMN migration if it finds an older schema without `extra_state_json`.

### Connection strategy

`SessionStore` uses a per thread SQLite connection via `threading.local()`, enables WAL journal mode and `synchronous=NORMAL` pragmas, and wraps writes in `BEGIN IMMEDIATE` transactions so concurrent Streamlit and benchmark workers do not corrupt state.

### Functions

#### `_safe_json_loads(raw, default)`

Defensively parses JSON and guarantees the parsed type matches the expected default structure.

#### `_trim_turns(turns)`

Keeps recent user and assistant turns and trims older turns when character limits (`session_turns_max_chars`, `session_turn_trim_target_chars`) are exceeded.

#### `_sanitize_anchor(value)`

Normalizes anchor dictionaries before persistence. It ensures the anchor value exists and clamps confidence into the zero to one range.

#### `_sanitize_extra_state(extra_state)`

Sanitizes auxiliary state such as anchor details, retrieval confidence, and control flags before storage.

#### `SessionStore.load(session_id)` / `save(...)` / `reset(session_id)` / `close()`

Load, save, reset, and per thread connection close. `save` preserves an existing summary and extra state when the caller passes blanks, which matters for meta query paths that should not overwrite continuity state.

## 8. `rag_utils.py`

This file provides the low level helper functions shared across retrieval, prompt construction, answer cleanup, and anchor logic.

### Text normalization helpers

`norm_text`, `clean_html`, `normalize_title_case`, `collapse_whitespace`, `tokenize_words`, `token_in_hay`.

### Lexical caches and NLTK bootstrap

`bootstrap_nltk_data`, `get_stopword_set`, `get_english_word_set`, `get_name_token_set`, `_wordnet_is_common_word`.

### Runtime configurable query parsing

`_split_config_terms`, `get_generic_query_terms`, `get_followup_phrases`, `get_followup_pronoun_pattern`, `get_person_pronoun_pattern`, `is_generic_query_token`, `is_followup_coref_question`, `bust_caches`.

### Corpus specific cleanup

#### `strip_corpus_noise_terms(query)`

Removes broad Syracuse terms such as `university`, `faculty`, or `campus` when they would hurt retrieval specificity.

### Document shaping helpers

`dedupe_docs`, `doc_haystack`, `truncate_text`, `clean_snippet`, `_extract_summary_from_page_content`, `dedupe_ci`.

#### `build_compact_context(docs, max_docs=None, text_limit=None)`

Builds the compact document context block used in prompts from title, researcher, authors, year, primary topic, extracted summary, and fallback snippet. Documents are sorted by researcher name before context assembly so the LLM sees coherent per researcher clusters, and researcher group separators are inserted when the researcher changes. This prevents the pattern where the model invents placeholder labels like "Unknown researcher 1" and "Unknown researcher 2" when docs from different people are interleaved.

### Anchor and confidence helpers

`is_placeholder_anchor_value`, `normalize_anchor`, `anchor_in_text`, `anchor_is_stable`.

#### `anchor_support_ratio(anchor_value, docs)`

Measures how strongly the retrieved set supports the anchor. Includes fuzzy person name matching so that a full name anchor like "Duncan Brown" correctly matches documents whose metadata stores the initial format "D. Brown". The function first tries exact text matching and falls back to initial plus last name matching across researcher and authors fields.

#### `retrieval_confidence_label(docs_count, anchor_consistent)`

Maps retrieval support into confidence labels such as high, medium, weak, or inconsistent.

### Intent and answer cleanup helpers

`classify_generic_intent`, `strip_prompt_leak`, `looks_like_person_candidate`, `strip_possessive`, `tokenize_name`, `generate_name_variants`, `split_author_names`, `has_explicit_entity_signal`, `short_hash`, `utcnow_iso`, `is_meta_command`, `anchor_query_overlap`, `query_tokens_for_relevance`.

The name related helpers (`tokenize_name`, `generate_name_variants`, `split_author_names`) produce name signatures that the pipeline and anchor support code use to match across "Duncan Brown", "D. Brown", "Brown, D.", and similar variants.

## 9. `rag_engine.py`

This file is the runtime core. It manages dynamic resource budgets, model runtimes with quantization and attention backends, rolling summary construction, entity extraction, query shaping, the utility worker for background utility calls, and the global engine manager.

### Runtime and debug helpers

`_ensure_dir`, `_make_local_chroma_client`, `_dbg`.

### Resource awareness

#### `available_ram_mb()` / `available_vram_mb()`

Report available system and GPU memory.

#### `dynamic_budgets()`

Adjusts memory, paper, and token budgets based on current RAM and VRAM conditions. This is a key operational safeguard because it reduces load when resources are tight.

### Rolling summary helpers

`_summary_template_empty`, `_extract_summary_sections`, `_format_summary_sections`, `_clean_answer_for_summary_signal`, `_extract_answer_theme_keywords`, `_sanitize_entity_values`.

#### `build_rolling_summary(previous_summary, user_question, retrieval_metadata, assistant_answer)`

Updates the rolling summary using the current question, retrieval metadata, and generated answer. This is central to conversation continuity because later turns can rely on the summary even when older turns are trimmed.

### LLM timeout and summary regeneration

#### `_release_vram_cache()`

Periodically clears cached VRAM.

#### `_invoke_with_timeout(llm, prompt, timeout_s)`

Runs model invocation behind a timeout wrapper to prevent the UI from hanging indefinitely.

#### `_regenerate_rolling_summary(...)`

Optionally asks a utility model to regenerate the rolling summary, then post processes that output into the deterministic summary structure.

### Entity extraction and anchor support

`_is_anchor_escape_question`, `_looks_like_person_token`, `_extract_entities_regex`, `_extract_entities_nltk`, `_extract_entities_basic`, `_build_ner_context_text`, `_extract_summary_topic_keywords`, `_summary_query_from_text`, `_summary_keywords_overlap_anchor`.

#### `_extract_person_name(question)`

Extracts a person name from the user question. This is important because the pipeline uses person specific ranking behavior when it detects a person centered query.

### Query shaping and document packing

`_answer_is_bad`, `_extract_focus_from_question`, `_is_invalid_focus_value`, `_query_is_short_or_pronoun`, `_inject_anchor_into_query`, `pack_docs`.

### Model runtime

#### `ModelRuntime`

Wraps a Hugging Face model, tokenizer, and generation configuration. It selects between FlashAttention 2, PyTorch SDPA, and eager attention based on what the environment supports and falls back gracefully. Quantization uses 4 bit via `bitsandbytes` for the larger answer models (8B, 12B, 14B, 20B) and unquantized weights for the 3B and 1B models. It provides `close()` so the manager can deterministically free VRAM when switching models.

#### `_resolve_llm_path(llm_model_key)` / `_quantize_bits(...)` / `_is_remote_model(...)`

Map an internal model key (for example `gemma-3-12b`) to a concrete local path, to its quantization width, and to whether the path refers to a local directory or a Hugging Face Hub identifier.

#### `build_embeddings()` / `clear_runtime_cache()`

Embedding model construction and a helper to clear cached runtimes.

### Utility worker

#### `UtilityJob` / `UtilityWorker`

Background queue based execution of utility model calls such as query rewriting, rerank scoring, and optional summary regeneration. The worker can be disabled or suppressed when VRAM is tight. It is bounded by `utility_queue_max` to prevent unbounded growth.

### `EngineManager`

The singleton that owns the `DatabaseManager`, `SessionStore`, embeddings, per mode `Chroma` clients, answer and utility model runtimes, the answer generation lock, and the utility worker.

Key methods:

1. `get_papers_vs(mode)` — lazily opens and caches the Chroma wrapper for a given mode.
2. `switch_mode(mode)` — resolves and switches the active dataset in both the manager and the `DatabaseManager`.
3. `switch_answer_model(llm_model_key)` / `switch_model(...)` — switch answer model, evicting the utility runtime first if the new answer model needs quantization.
4. `_vram_is_tight()` / `_evict_utility_if_needed()` — VRAM protection. A headroom constant (around 2.5 GB) governs when the utility runtime is unloaded to keep answer generation from thrashing.
5. `_switch_runtime(...)` — generic runtime swap used by both answer and utility paths.
6. `reset_session(session_id)` — clears persistent session state and, when appropriate, memory collection entries for one user.
7. `get_engine(user_key, mode, stateless)` — lazily constructs or returns the per user `Engine` that owns turn level context and retrieval.

#### `get_global_manager()`

Returns the process wide `EngineManager` instance used by the pipeline, the Streamlit UI, the terminal chat, and the benchmark harness.

## 10. `rag_pipeline.py`

This file is the main orchestration layer. It connects intent detection, meta query handling, retrieval, dominance analysis, anchor updates, prompt construction, answer generation, hallucination detection, fallback logic, graph generation, and final payload assembly.

### Pipeline config

#### `PIPELINE_CFG`

Dictionary holding pipeline specific runtime settings, including prompt framing, style rules that enforce grounded answers, fallback controls, maximum document counts, and dangling pronoun answer templates.

### Intent and meta query helpers

#### `_is_summary_intent(question)`

Detects whether the user wants a summary style answer.

#### `_detect_meta_query(question)`

Recognizes structural questions about the corpus itself, such as "how many papers" and "what is the most recent paper".

#### `_answer_meta_query(meta_type, mgr, effective_mode)`

Answers recognized meta queries directly from the active Chroma collection without invoking the answer model. This keeps meta questions cheap and consistent.

### Document and metadata helpers

`_doc_to_source_md`, `_doc_to_ref`, `_filter_noisy_docs`, `_normalize_meta_value`, `_metadata_key_allowed`, `_metadata_value_allowed`, `_iter_doc_metadata_key_values`.

### Person specific retrieval helpers

These functions are among the most important specialized behaviors in the pipeline.

`_person_name_signatures`, `_name_match_strength`, `_doc_person_match_score`, `_rank_docs_for_person`, `_select_docs_for_person`.

`_select_docs_for_person` uses a strong match threshold calibrated so that initial format name matches such as "D. Brown" matching "Duncan Brown" count as strong evidence. This prevents the confidence downshift cascade that previously caused sterile extractive answers on first turn person queries.

### Confidence and dominance helpers

#### `_downgrade_confidence(label, steps=1)`

Lowers a retrieval confidence label.

#### `_downshift_confidence_for_person_support(label, ...)`

Further reduces confidence only when person specific evidence is weaker than expected. Thresholds are calibrated for academic metadata where initial format names are common.

#### `_dominant_metadata_filter_from_docs(docs, question, ...)`

Finds a dominant metadata value across retrieved docs that can serve as an anchor candidate or a signal that retrieval coheres around one subject.

### User facing fallback answers

`_insufficient_context_answer`, `_uncertain_retrieval_answer`.

### Answer sanitation helpers

`_normalize_for_similarity`, `_strip_leading_answer_labels`, `_is_closure_or_process`, `_sanitize_user_answer`.

#### `sanitize_answer_for_display(text)`

Applies final display safe cleanup to the answer. The Streamlit app calls this before rendering.

### Anchor update helpers

#### `_build_anchor_from_dominance(dominance)`

Creates a candidate anchor from dominance analysis.

#### `_choose_anchor_update(current_anchor, candidate_anchor, dominance, question, resolved_question="")`

Decides whether to keep, replace, or ignore the anchor. It validates anchor candidates against both the raw user question and the resolved or rewritten question. This prevents anchor drift where an unrelated researcher from noisy retrieval results would silently become the anchor when the user asked a follow up using pronouns. A new candidate anchor is blocked with action `blocked_no_query_overlap` when it has zero token overlap with both the raw and resolved questions.

### Prompt composition

`_extract_answer_text`, `_runtime_prompt_token_budget`, `_compose_answer_prompt`.

#### `_fit_prompt_to_budget(...)`

Shrinks document count and context length iteratively so the prompt fits within model limits. The fitting strategy uses a two phase approach. Phase one shrinks only the per document text limit while preserving document count, since having more documents in the context is more important for answer quality than per document verbosity. Phase two alternates between shrinking documents and text when the text floor has been reached.

### Prompt context helpers

`_clip_sentences`, `_clean_assistant_turn_for_prompt`, `_rolling_summary_for_prompt`, `_build_recent_turns_context`.

### Fallback answer synthesis

#### `_fallback_answer_from_docs(question, docs, intent="default")`

Produces a non model fallback answer directly from retrieved docs.

#### `_supported_researcher_evidence(docs, ...)`

Builds structured researcher evidence from retrieved docs, used by the replacement extractive answer when hallucination risk is high.

#### `_answer_mentions_unsupported_researcher(answer, docs)`

Checks whether the answer names researchers not supported by the retrieved evidence. Uses bidirectional name matching: it checks answer names against document metadata signatures and also checks document names against answer name signatures. This prevents false positives when the answer uses a full name like "Duncan Brown" but the metadata stores "D. Brown". A last name only fallback is applied so that shared surnames are not flagged as unsupported.

#### `_build_researcher_extract_answer(docs, max_researchers=5)`

Builds a safer extractive answer that lists supported researchers and papers when hallucination risk is high.

### Citation hallucination detection

#### `_collect_doc_titles(docs)`

Collects normalized titles from retrieved documents for validation against the generated answer.

#### `_strip_hallucinated_citations(answer, docs)`

Removes lines from the generated answer that contain quoted paper titles not found in the retrieved document set. Uses fuzzy matching with a `SequenceMatcher` ratio threshold to account for minor formatting differences. This catches fabricated citations that small models sometimes invent when given insufficient context.

### Main entry point

#### `answer_question(question, user_key, use_graph=None, stateless=None)`

This is the main application entry point. At a high level it:

1. validates the question
2. short circuits meta commands (topic reset) and meta queries (corpus stats)
3. loads engine and session state
4. switches to the requested dataset and answer model when they have changed
5. resolves graph and stateless behavior
6. detects follow up status and intent
7. performs retrieval and optional anchor aware rewrite
8. filters and analyzes documents
9. computes confidence and dominance
10. updates or preserves the anchor using both raw and resolved question overlap
11. builds a prompt from summary turns and compact context grouped by researcher
12. invokes the answer model
13. strips hallucinated citations from the generated answer
14. sanitizes or replaces weak answers using bidirectional name matching
15. persists updated state
16. returns a UI ready result payload

### Output assembly

#### `_build_output(...)`

Constructs the final structured response object returned to Streamlit and the other entry points, including answer text, sources, timing, graph, anchor, and retrieval diagnostics.

## 11. `rag_graph.py`

This file builds an in memory relationship graph from retrieved paper documents.

### Functions

#### `_safe_str(x)`

Normalizes values into non null strings.

#### `_split_authors(s, limit=25)`

Splits author strings on commas, semicolons, the word "and", or pipes, deduplicates them, and limits output size.

#### `paper_docs_to_graph_hits(paper_docs, max_papers=40)`

Converts retrieved documents into simplified graph ready paper hits.

#### `build_graph_from_hits(hits, height=650, include_topics=True, include_authors=True, max_authors_per_paper=12)`

Creates nodes and edges for papers, researchers, authors, and topics.

#### `graph_retrieve_from_paper_docs(paper_docs, height=650)`

Convenience wrapper that converts retrieved paper docs into a graph payload.

## 12. `streamlit_app.py`

This file defines the Streamlit UI and the operational controls.

### Utility functions

`_safe_call`, `_esc`, `_esc_answer`, `_render_graph`.

### App bootstrap behavior

On startup the file sets environment variables for local execution, imports the pipeline entry point, initializes the page, creates a `user_key` if needed, acquires the global manager, and enables graph and debug behavior.

### Sidebar controls

1. **System Memory** — live RAM and VRAM usage bars.
2. **Reset Memory** — clears stored conversational memory for the current session id.
3. **Restart Conversation** — clears transcript and memory and issues a fresh session id.
4. **Dataset** — selectbox driven by `DatabaseManager.display_labels()`. Switching flips Chroma and Neo4j together and clears the per mode vector store cache.
5. **Model** — selectbox listing the five answer models. Switching invokes `EngineManager.switch_answer_model(...)` and shows a spinner while the new model is loaded or re quantized.
6. **Session Diagnostics** — a JSON block showing `session_id`, `active_dataset`, `chroma_collection`, `neo4j_db`, requested and loaded `answer_model`, attention implementation, GPU name, and session counters such as `turn_count`, `summary_len`, and `retrieval_confidence`.

### `_refresh_sidebar(user_key)`

Refreshes RAM and VRAM diagnostics and rewrites the session diagnostics JSON.

### Chat loop behavior

For each submitted prompt the UI:

1. appends the user message to session state
2. calls `answer_question(prompt, user_key=USER_KEY, use_graph=True, stateless=False)`
3. renders the sanitized answer
4. shows timing and LLM call counts
5. shows retrieved sources in an expander
6. shows graph output inside a collapsible toggle
7. appends the assistant response to chat history
8. refreshes diagnostics

## 13. `rag_chat.py`

This file is the interactive terminal chat entry point. It is a pure CLI mirror of the Streamlit loop and is useful for headless or remote sessions.

### Startup

Selects a model and database at launch, either via `--model` and `--db` flags or an interactive prompt, then enters a REPL. Every question and answer is saved incrementally to a timestamped JSON log using the same field structure as the benchmark.

### Slash commands

1. `/model <key>` — switch to a different answer model.
2. `/db <key>` — switch to a different dataset.
3. `/status` — show current model, database, and session info.
4. `/reset` — clear conversation memory.
5. `/stateless` — toggle stateless mode (no conversation history).
6. `/verbose` — toggle verbose output (timing, retrieval details).
7. `/history` — show conversation history.
8. `/log` — show the path to the current session's JSON log.
9. `/help` — show command reference.
10. `/quit` or `/exit` — exit the chat.

Color output uses ANSI escapes; input is provided through `readline` so arrow key history works.

## 14. `benchmark_rag.py`

This file runs a full product matrix of answer models against all registered datasets using a fixed question set and writes structured results.

### Design

Every `(model, database)` permutation is executed in its own subprocess. This isolation is intentional: a CUDA device side assert inside one model cannot corrupt the next, and VRAM is fully released between runs. The parent process collects per permutation JSON files and assembles them into one report.

### Output

1. `benchmark_results_YYYYMMDD_HHMMSS.json` — full structured results per permutation and per question.
2. `benchmark_summary_YYYYMMDD_HHMMSS.txt` — human readable summary table.

### Usage

```
python benchmark_rag.py
python benchmark_rag.py --models 3b 8b
python benchmark_rag.py --databases full
python benchmark_rag.py --dry-run
```

The dry run path does not import the pipeline, so it is safe to use for validating the question list and argument parsing without loading models.

## Retrieval behavior in plain language

### Normal question path

1. Classify the question intent.
2. Strip overly generic corpus noise terms if helpful.
3. Run Chroma retrieval against the active dataset.
4. Deduplicate and filter documents.
5. Check whether results cohere around one topic or person.
6. Build prompt context from summary, recent turns, and compact doc context grouped by researcher.
7. Invoke the answer model with grounding instructions.
8. Strip any hallucinated citations from the answer.

### Follow up question path

1. Detect short or pronoun based follow up patterns.
2. Consult anchor and rolling summary.
3. Optionally rewrite or expand the retrieval query with anchor context.
4. Retrieve with larger search multipliers when configured.
5. Recompute anchor stability and support ratio using fuzzy person name matching before answer generation.
6. Validate anchor updates against both raw and resolved questions.

### Weak or inconsistent retrieval path

1. Downgrade retrieval confidence using person support thresholds.
2. Shrink prompt size to reduce noise using low confidence caps.
3. Apply stricter anti hallucination guidance.
4. Fall back to an extractive answer when synthesis is unsafe.
5. Replace answers that mention unsupported researchers using bidirectional name matching.

### Meta query path

1. Detect a meta query such as "how many papers" or "most recent paper".
2. Answer directly from the active Chroma collection.
3. Skip retrieval and the answer model entirely.
4. Persist the turn so conversation continuity is preserved.

## Data contracts

### Chroma document metadata

Each ingested chunk typically includes:

1. `paper_id`
2. `research_info_id`
3. `researcher`
4. `title`
5. `authors`
6. `doi`
7. `year`
8. `publication_date`
9. `primary_topic`
10. `chunk`
11. `chunks_total`

### Session state shape

Persistent session state in SQLite contains:

1. `rolling_summary`
2. `turns_json` — an array of `{role, text}` objects
3. `extra_state_json` — anchor, anchor last action, retrieval confidence, last focus, last topic, summary updated flag, rewrite flags, and anchor support ratio

## Setup and local execution

### Requirements

1. Python 3.10 or newer
2. SQLite database with `research_info` and `works`
3. ChromaDB
4. Streamlit (for the web UI)
5. Transformers and Torch
6. `bitsandbytes` for 4 bit quantization of large models
7. Sentence transformer compatible embedding dependencies
8. LangChain core and Chroma integrations
9. Optional `streamlit_agraph` for graph rendering
10. Optional `readline` support (standard on Linux and macOS) for the terminal chat

### Clone the repository

```
git clone <your-repo-url>
cd <your-repo-folder>/rag
```

### Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with the standard `Scripts\activate` command for your shell.

### Install dependencies

If you add a requirements file:

```
pip install -r requirements.txt
```

A likely starting point from the visible imports is:

```
pip install streamlit chromadb langchain-core langchain-chroma langchain-huggingface \
            transformers torch bitsandbytes accelerate psutil tqdm nltk \
            streamlit-agraph sentence-transformers
```

### Configure the environment

Review and update:

1. `config_full.py`
2. `config_graph.py`
3. `runtime_settings.py`

The most important values are the SQLite database path, Chroma persistence paths for all three datasets, collection names, Neo4j DB names, local model paths for every answer model you plan to load, the embedding model path, and runtime budget settings.

### Build the Chroma index

```
python chroma_ingest.py
```

The ingestion script reads `SQLITE_DB_FULL`, builds one canonical document per paper id, chunks it with sentence aware overlap, and upserts to `CHROMA_COLLECTION_FULL` inside `CHROMA_DIR_FULL`. For the OpenAlex and abstracts datasets, use their respective ingestion pipelines (not included in this snapshot) to populate `CHROMA_DIR_OPENALEX` and `CHROMA_DIR_ABSTRACTS`.

## Running the three entry points

### Streamlit web UI

```
streamlit run streamlit_app.py
```

The sidebar exposes dataset selection, model selection, and session reset controls.

### Terminal chat

```
python rag_chat.py
python rag_chat.py --model llama-3.1-8b --db openalex
python rag_chat.py --list
```

Use the slash commands inside the REPL to change models, switch datasets, toggle verbose output, or reset memory without exiting.

### Benchmark harness

```
python benchmark_rag.py
```

Runs every `(model, database)` permutation on the built in question set and writes both JSON and text reports. Each permutation executes in an isolated subprocess.

## Runtime operations

### Why summaries are extracted from page content

The ingestion path stores summaries in `page_content`, not metadata. That is why `rag_utils._extract_summary_from_page_content()` exists and why prompt context building parses document text.

### Why follow ups work without keeping the entire transcript

The system uses a rolling summary and a trimmed set of recent turns. This preserves continuity without letting the prompt grow without bound.

### Why some answers become more extractive

That usually means retrieval confidence was weak or inconsistent, so the guardrails chose a safer output form.

### Why reset behavior can feel stronger than chat clearing

The UI reset paths clear not just visible chat history but also persistent SQLite state and entries in the Chroma memory collection. Restart Conversation additionally rotates the session id so nothing from the previous conversation can leak in through anchor state.

### Why the prompt budget loop now prefers shrinking text over dropping documents

The budget fitting strategy shrinks per document text limits first before reducing the number of documents. Having more documents in the context, even with shorter summaries, produces better grounded answers than having fewer documents with longer summaries. This is especially important for multi researcher queries where the model needs to see evidence from several different people.

### Why anchor updates check the resolved question

When a user asks a follow up like "summarize the mechanisms described in his papers", the raw question contains only pronouns. The resolved question after coreference rewriting contains the actual entity name. Anchor validation checks both, preventing a dominant but unrelated researcher from noisy retrieval results from silently hijacking the conversation anchor.

### Why citation validation runs after generation

Small language models sometimes fabricate plausible looking paper titles and journal references when given insufficient context. The post generation citation check compares all quoted titles in the answer against the actual retrieved documents and removes lines containing fabricated references. This is a lightweight safeguard that runs without additional model calls.

### Why the utility model can be unloaded while you chat

When a 4 bit quantized answer model is loaded and free VRAM drops below a headroom threshold, the engine evicts the utility model and stops its worker. Query rewriting, reranking, and summary regeneration degrade gracefully. Answer generation is protected. The utility model is reloaded automatically when a lighter answer model is selected or when VRAM frees up.

### Why datasets and models can be switched mid conversation

Both the Streamlit sidebar and the terminal `/db` and `/model` commands go through `EngineManager.switch_mode(...)` and `EngineManager.switch_answer_model(...)`. Session state is preserved across switches because it is owned by `SessionStore`, not by the model runtime. The anchor and rolling summary travel with the user.

