"""
RAG Live Chat
=============
Interactive terminal chat interface for the RAG pipeline.
Select a model and database at launch, then chat freely.
Every question/answer is saved incrementally to a timestamped JSON log
using the same field structure as benchmark_rag.py.

Usage:
    python rag_chat.py
    python rag_chat.py --model llama-3.2-3b --db full
    python rag_chat.py --model gemma-4-e4b --db openalex
    python rag_chat.py --list

Commands inside the chat:
    /model <key>       — switch to a different model
    /db <key>          — switch to a different database
    /status            — show current model, database, and session info
    /reset             — clear conversation memory
    /stateless         — toggle stateless mode (no conversation history)
    /verbose           — toggle verbose output (timing, retrieval details)
    /history           — show conversation history
    /log               — show the path to the current session's JSON log
    /help              — show this help
    /quit or /exit     — exit the chat

Examples:
    python rag_chat.py -m llama-3.1-8b -d openalex
    python rag_chat.py --model qwen-2.5-14b --db abstracts
    python rag_chat.py -o ./logs
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
import uuid
import readline
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAG_DEBUG", "0")
os.environ.setdefault("RAG_EMBED_DEVICE", "cpu")
os.environ.setdefault("RAG_LLM_TIMEOUT_S", "300")

MODELS: Dict[str, str] = {
    "llama-3.2-3b":   "LLaMA 3.2 3B (native)",
    "llama-3.1-8b":   "LLaMA 3.1 8B (native)",
    "gemma-4-e2b":    "Gemma 4 E2B (native)",
    "gemma-4-e4b":    "Gemma 4 E4B (native)",
    "gemma-4-26b":    "Gemma 4 26B-A4B (native)",
    "gemma-4-31b":    "Gemma 4 31B (native)",
    "qwen-2.5-14b":   "Qwen 2.5 14B (native)",
    "gpt-oss-20b":    "GPT-OSS 20B (native)",
    "llama-3.3-70b":   "LLaMA 3.3 70B (native)",
}

DATABASES: Dict[str, str] = {
    "full":      "Legacy DB (full text)",
    "openalex":  "OpenAlex DB",
    "abstracts": "Abstracts Only",
}

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"
    CYAN    = "\033[36m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE    = "\033[34m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BG_DARK = "\033[48;5;236m"

def styled(text: str, *styles: str) -> str:
    return "".join(styles) + text + C.RESET

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _answer_word_count(text: str) -> int:
    return len((text or "").split())

def _source_count(out: Dict[str, Any]) -> int:
    return len(out.get("sources", []) or [])

def _gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch
        info["available"] = torch.cuda.is_available()
        if info["available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{cap[0]}.{cap[1]}"
            free, total = torch.cuda.mem_get_info(0)
            info["vram_total_gb"] = round(total / 1e9, 2)
            info["vram_free_gb"] = round(free / 1e9, 2)
    except Exception:
        pass
    return info

BANNER = (
    f"\n{C.CYAN}{C.BOLD}"
    "\u256d\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256e\n"
    "\u2502                                                               \u2502\n"
    "\u2502  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2557   \u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2557        \u2588\u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2557   \u2502\n"
    "\u2502  \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u255a\u2588\u2588\u2557 \u2588\u2588\u2554\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557       \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d   \u2502\n"
    "\u2502  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557  \u255a\u2588\u2588\u2588\u2588\u2554\u255d \u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d \u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2588\u2557  \u2502\n"
    "\u2502  \u255a\u2550\u2550\u2550\u2550\u2588\u2588\u2551   \u255a\u2588\u2588\u2554\u255d  \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557 \u255a\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2551   \u2588\u2588\u2551  \u2502\n"
    "\u2502  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551    \u2588\u2588\u2551   \u2588\u2588\u2551  \u2588\u2588\u2551        \u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d  \u2502\n"
    "\u2502  \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d    \u255a\u2550\u255d   \u255a\u2550\u255d  \u255a\u2550\u255d        \u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u255d   \u2502\n"
    "\u2502                                                               \u2502\n"
    f"\u2570\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u256f{C.RESET}\n"
)

class SessionLogger:
    """Incrementally saves a JSON log matching the benchmark_rag.py schema."""

    def __init__(self, output_dir: str = "."):
        self.tag: str = _now_tag()
        self.output_dir: str = output_dir
        self.t0: float = time.perf_counter()
        self.gpu: Dict[str, Any] = _gpu_info()

        os.makedirs(output_dir, exist_ok=True)
        self.json_path: str = os.path.join(output_dir, f"chat_session_{self.tag}.json")

        self.runs: List[Dict[str, Any]] = []
        self.current_run: Optional[Dict[str, Any]] = None
        self.events: List[Dict[str, Any]] = []
        self.total_questions: int = 0
        self.total_errors: int = 0

    def start_run(self, model_key: str, db_key: str,
                  model_load_s: float = 0.0,
                  db_health: Optional[Dict[str, Any]] = None) -> None:
        self.current_run = {
            "model_key": model_key,
            "model_label": MODELS.get(model_key, model_key),
            "db_key": db_key,
            "db_label": DATABASES.get(db_key, db_key),
            "model_load_s": round(model_load_s, 2),
            "started_at": _now_iso(),
            "error": "",
            "questions": [],
            "db_health": db_health or {},
        }
        self.runs.append(self.current_run)

    def ensure_run(self, model_key: str, db_key: str) -> None:
        if (self.current_run is not None
                and self.current_run["model_key"] == model_key
                and self.current_run["db_key"] == db_key):
            return
        self.start_run(model_key, db_key)

    def log_question(self, question: str, raw_out: Dict[str, Any],
                     wall_time_s: float, error: str = "") -> Dict[str, Any]:
        answer = str(raw_out.get("answer", "") or "").strip()
        timing = raw_out.get("timing_ms", {}) if isinstance(raw_out.get("timing_ms"), dict) else {}
        llm_calls = raw_out.get("llm_calls", {}) if isinstance(raw_out.get("llm_calls"), dict) else {}
        chroma = raw_out.get("chroma_retrieval", {}) if isinstance(raw_out.get("chroma_retrieval"), dict) else {}
        user_query = raw_out.get("user_query", {}) if isinstance(raw_out.get("user_query"), dict) else {}

        record: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "answer_word_count": _answer_word_count(answer),
            "source_count": _source_count(raw_out),
            "error": error,
            "wall_time_s": round(wall_time_s, 2),
            "timestamp": _now_iso(),
            "timing": {
                "total_ms": round(wall_time_s * 1000, 2),
                "rewrite_ms": round(float(timing.get("rewrite_ms", 0) or 0), 2),
                "retrieval_ms": round(float(timing.get("retrieval_total_ms", 0) or 0), 2),
                "generation_ms": round(float(timing.get("generation_ms", 0) or 0), 2),
            },
            "llm_calls": {
                "answer": int(llm_calls.get("answer_llm_calls", 0) or 0),
                "utility": int(llm_calls.get("utility_llm_calls", 0) or 0),
            },
            "retrieval": {
                "doc_count": int(chroma.get("count", 0) or 0),
                "raw_count": int(chroma.get("retrieval_count_raw", 0) or 0),
                "confidence": str(chroma.get("retrieval_confidence", "") or ""),
            },
            "query": {
                "resolved": str(user_query.get("resolved_text", "") or ""),
                "detected_intent": str(user_query.get("detected_intent", "") or ""),
                "anchor_action": str(user_query.get("anchor_action", "") or ""),
            },
        }

        self.total_questions += 1
        if error:
            self.total_errors += 1

        if self.current_run is not None:
            self.current_run["questions"].append(record)

        self.save()
        return record

    def log_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "type": event_type,
            "timestamp": _now_iso(),
            "details": details or {},
        })
        self.save()

    def _build_snapshot(self) -> Dict[str, Any]:
        return {
            "session_tag": self.tag,
            "mode": "live_chat",
            "timestamp": _now_iso(),
            "total_time_s": round(time.perf_counter() - self.t0, 2),
            "total_questions": self.total_questions,
            "total_errors": self.total_errors,
            "gpu": self.gpu,
            "events": self.events,
            "runs": self.runs,
        }

    def save(self) -> None:
        try:
            snapshot = self._build_snapshot()
            tmp = self.json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.json_path)
        except Exception as e:
            print(styled(f"  [WARN] Failed to save session log: {e}", C.RED))

class RAGChat:
    """Manages the RAG pipeline session."""

    def __init__(self, logger: SessionLogger):
        self.mgr = None
        self.settings = None
        self.answer_fn = None
        self.hard_reset_memory = None

        self.logger: SessionLogger = logger
        self.current_model: Optional[str] = None
        self.current_db: Optional[str] = None
        self.user_key: str = f"chat_{uuid.uuid4().hex[:8]}"
        self.stateless: bool = False
        self.verbose: bool = False
        self.history: List[Dict[str, str]] = []
        self.question_count: int = 0

    def _ensure_imports(self):
        if self.mgr is not None:
            return
        print(styled("  Loading RAG pipeline...", C.DIM))
        from rag_pipeline import answer_question
        from rag_engine import get_global_manager
        from runtime_settings import settings
        from conversation_memory import hard_reset_memory

        self.mgr = get_global_manager()
        self.settings = settings
        self.answer_fn = answer_question
        self.hard_reset_memory = hard_reset_memory
        print(styled("  Pipeline loaded.", C.GREEN))

    def switch_model(self, model_key: str) -> float:
        self._ensure_imports()
        print(styled(f"  Loading model {MODELS[model_key]}...", C.DIM))
        t0 = time.perf_counter()
        self.settings.answer_model_key = model_key
        self.settings.llm_model = model_key
        self.mgr.switch_answer_model(model_key)
        elapsed = time.perf_counter() - t0
        self.current_model = model_key
        print(styled(f"  Model loaded in {elapsed:.1f}s", C.GREEN))

        self.logger.log_event("model_switch", {
            "model_key": model_key,
            "model_label": MODELS[model_key],
            "load_time_s": round(elapsed, 2),
        })

        if self.current_db:
            db_health = self._get_db_health(self.current_db)
            self.logger.start_run(model_key, self.current_db,
                                  model_load_s=elapsed, db_health=db_health)
        return elapsed

    def switch_db(self, db_key: str) -> None:
        self._ensure_imports()
        print(styled(f"  Switching to database {DATABASES[db_key]}...", C.DIM))
        self.mgr.switch_mode(db_key)
        self.settings.active_mode = db_key
        if hasattr(self.mgr, "papers_vs_cache"):
            self.mgr.papers_vs_cache.pop(db_key, None)
        self.current_db = db_key

        doc_count = self._get_doc_count(db_key)
        extra = f" ({doc_count} documents)" if doc_count >= 0 else ""
        print(styled(f"  Database active: {DATABASES[db_key]}{extra}", C.GREEN))

        db_health = self._get_db_health(db_key)
        self.logger.log_event("db_switch", {
            "db_key": db_key,
            "db_label": DATABASES[db_key],
            "doc_count": doc_count,
        })

        if self.current_model:
            self.logger.start_run(self.current_model, db_key, db_health=db_health)

    def _get_doc_count(self, db_key: str) -> int:
        try:
            vs = self.mgr.get_papers_vs(db_key)
            col = getattr(vs, "_collection", None)
            if col is not None:
                return int(col.count())
        except Exception:
            pass
        return -1

    def _get_db_health(self, db_key: str) -> Dict[str, Any]:
        health: Dict[str, Any] = {"healthy": True, "doc_count": -1}
        try:
            if hasattr(self.mgr, "dbm") and hasattr(self.mgr.dbm, "validate_active_config"):
                health = self.mgr.dbm.validate_active_config()
            if health.get("doc_count", -1) < 0:
                health["doc_count"] = self._get_doc_count(db_key)
        except Exception:
            pass
        return health

    def reset_session(self):
        try: self.hard_reset_memory(self.user_key)
        except Exception: pass
        self.user_key = f"chat_{uuid.uuid4().hex[:8]}"
        self.history.clear()
        self.question_count = 0
        self.logger.log_event("session_reset", {"new_user_key": self.user_key})

    def free_vram(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def ask(self, question: str) -> Dict[str, Any]:
        self._ensure_imports()
        self.question_count += 1

        if self.current_model and self.current_db:
            self.logger.ensure_run(self.current_model, self.current_db)

        t0 = time.perf_counter()
        error = ""
        out: Dict[str, Any] = {}
        try:
            out = self.answer_fn(
                question,
                user_key=self.user_key,
                use_graph=False,
                stateless=self.stateless,
            )
        except KeyboardInterrupt:
            error = "KeyboardInterrupt (skipped)"
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        wall_time = time.perf_counter() - t0

        record = self.logger.log_question(question, out, wall_time, error=error)

        answer = str(out.get("answer", "") or "").strip()
        sources = out.get("sources", []) or []

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": sources,
            "error": error,
            "wall_time_s": record["wall_time_s"],
            "timing": record["timing"],
            "retrieval": record["retrieval"],
            "query": record["query"],
            "llm_calls": record["llm_calls"],
        }

def pick_option(title: str, options: Dict[str, str], prompt_text: str = "Enter choice") -> str:
    keys = list(options.keys())
    print(f"\n{styled(title, C.BOLD, C.CYAN)}")
    print(styled("\u2500" * 40, C.DIM))
    for i, (key, label) in enumerate(options.items(), 1):
        print(f"  {styled(str(i), C.BOLD, C.YELLOW)}  {label}  {styled(f'({key})', C.DIM)}")
    print(styled("\u2500" * 40, C.DIM))

    while True:
        try:
            raw = input(f"  {prompt_text} [1-{len(keys)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if raw in options:
            return raw
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
        except ValueError:
            pass
        print(styled(f"  Invalid choice. Enter 1-{len(keys)} or a key name.", C.RED))

def handle_command(cmd: str, args_str: str, chat: RAGChat) -> bool:
    """Handle a /command. Returns True if the chat loop should continue."""

    if cmd in ("quit", "exit", "q"):
        return False

    elif cmd == "help":
        print(f"""
{styled("Available commands:", C.BOLD, C.CYAN)}
  {styled("/model <key>", C.YELLOW)}    Switch model     (keys: {', '.join(MODELS.keys())})
  {styled("/db <key>", C.YELLOW)}       Switch database  (keys: {', '.join(DATABASES.keys())})
  {styled("/status", C.YELLOW)}         Show current config & session info
  {styled("/reset", C.YELLOW)}          Clear conversation memory & start fresh
  {styled("/stateless", C.YELLOW)}      Toggle stateless mode (currently: {'ON' if chat.stateless else 'OFF'})
  {styled("/verbose", C.YELLOW)}        Toggle verbose output (currently: {'ON' if chat.verbose else 'OFF'})
  {styled("/history", C.YELLOW)}        Show conversation history
  {styled("/log", C.YELLOW)}            Show session JSON log file path & stats
  {styled("/help", C.YELLOW)}           Show this help
  {styled("/quit", C.YELLOW)}           Exit the chat
""")

    elif cmd == "model":
        key = args_str.strip()
        if not key:
            key = pick_option("Select Model", MODELS, "Model")
        if key not in MODELS:
            print(styled(f"  Unknown model '{key}'. Available: {', '.join(MODELS.keys())}", C.RED))
            return True
        if key == chat.current_model:
            print(styled(f"  Already using {MODELS[key]}.", C.YELLOW))
            return True
        try:
            chat.free_vram()
            chat.switch_model(key)
        except Exception as e:
            print(styled(f"  Failed to switch model: {e}", C.RED))

    elif cmd == "db":
        key = args_str.strip()
        if not key:
            key = pick_option("Select Database", DATABASES, "Database")
        if key not in DATABASES:
            print(styled(f"  Unknown database '{key}'. Available: {', '.join(DATABASES.keys())}", C.RED))
            return True
        if key == chat.current_db:
            print(styled(f"  Already using {DATABASES[key]}.", C.YELLOW))
            return True
        try:
            chat.switch_db(key)
        except Exception as e:
            print(styled(f"  Failed to switch database: {e}", C.RED))

    elif cmd == "status":
        model_label = MODELS.get(chat.current_model, "None") if chat.current_model else "None"
        db_label = DATABASES.get(chat.current_db, "None") if chat.current_db else "None"
        print(f"""
{styled("Session Status", C.BOLD, C.CYAN)}
  Model:      {styled(model_label, C.GREEN)} {styled(f'({chat.current_model})', C.DIM) if chat.current_model else ''}
  Database:   {styled(db_label, C.GREEN)} {styled(f'({chat.current_db})', C.DIM) if chat.current_db else ''}
  Session ID: {styled(chat.user_key, C.DIM)}
  Questions:  {chat.question_count}
  Stateless:  {'ON' if chat.stateless else 'OFF'}
  Verbose:    {'ON' if chat.verbose else 'OFF'}
  Log file:   {styled(chat.logger.json_path, C.BLUE)}
  Runs:       {len(chat.logger.runs)}
""")

    elif cmd == "reset":
        chat.reset_session()
        print(styled("  Session reset. Conversation memory cleared.", C.GREEN))

    elif cmd == "stateless":
        chat.stateless = not chat.stateless
        state = "ON" if chat.stateless else "OFF"
        print(styled(f"  Stateless mode: {state}", C.YELLOW))
        chat.logger.log_event("toggle_stateless", {"stateless": chat.stateless})

    elif cmd == "verbose":
        chat.verbose = not chat.verbose
        state = "ON" if chat.verbose else "OFF"
        print(styled(f"  Verbose mode: {state}", C.YELLOW))

    elif cmd == "history":
        if not chat.history:
            print(styled("  No conversation history yet.", C.DIM))
        else:
            print(styled("\nConversation History", C.BOLD, C.CYAN))
            print(styled("\u2500" * 50, C.DIM))
            for entry in chat.history:
                if entry["role"] == "user":
                    print(f"  {styled('You:', C.BOLD, C.YELLOW)} {entry['content']}")
                else:
                    snippet = entry["content"][:120]
                    if len(entry["content"]) > 120:
                        snippet += "..."
                    print(f"  {styled('RAG:', C.BOLD, C.GREEN)} {snippet}")
            print()

    elif cmd == "log":
        lg = chat.logger
        print(f"""
{styled("Session Log", C.BOLD, C.CYAN)}
  File:       {styled(lg.json_path, C.BLUE)}
  Questions:  {lg.total_questions}
  Errors:     {lg.total_errors}
  Runs:       {len(lg.runs)}
  Events:     {len(lg.events)}
  Duration:   {time.perf_counter() - lg.t0:.0f}s
""")

    else:
        print(styled(f"  Unknown command '/{cmd}'. Type /help for options.", C.RED))

    return True

def print_answer(result: Dict[str, Any], verbose: bool = False):
    answer = result["answer"]
    error = result.get("error", "")
    sources = result.get("sources", [])

    if error:
        print(f"\n{styled('Error', C.BOLD, C.RED)}: {error}")
        if not answer:
            return

    print(f"\n{styled('Answer', C.BOLD, C.GREEN)}")
    print(styled("\u2500" * 60, C.DIM))
    print(answer)

    if sources:
        print(f"\n{styled(f'Sources ({len(sources)})', C.BOLD, C.BLUE)}")
        print(styled("\u2500" * 60, C.DIM))
        for i, src in enumerate(sources, 1):
            if isinstance(src, dict):
                title = src.get("title", src.get("name", "Untitled"))
                authors = src.get("authors", "")
                year = src.get("year", "")
                meta_parts = []
                if authors:
                    meta_parts.append(str(authors)[:60])
                if year:
                    meta_parts.append(str(year))
                meta = ", ".join(meta_parts)
                print(f"  {styled(str(i) + '.', C.CYAN)} {title}")
                if meta:
                    print(f"     {styled(meta, C.DIM)}")
            else:
                print(f"  {styled(str(i) + '.', C.CYAN)} {src}")

    if verbose:
        t = result.get("timing", {})
        r = result.get("retrieval", {})
        q = result.get("query", {})
        lc = result.get("llm_calls", {})
        print(f"\n{styled('Details', C.BOLD, C.MAGENTA)}")
        print(styled("\u2500" * 60, C.DIM))
        print(f"  Wall time:    {result.get('wall_time_s', 0):.2f}s")
        print(f"  Rewrite:      {t.get('rewrite_ms', 0):.0f}ms")
        print(f"  Retrieval:    {t.get('retrieval_ms', 0):.0f}ms")
        print(f"  Generation:   {t.get('generation_ms', 0):.0f}ms")
        print(f"  Confidence:   {r.get('confidence', '?')}")
        print(f"  Docs matched: {r.get('doc_count', '?')}")
        print(f"  LLM calls:    answer={lc.get('answer', 0)}, utility={lc.get('utility', 0)}")
        if q.get("resolved"):
            print(f"  Resolved:     {q['resolved']}")
        if q.get("detected_intent"):
            print(f"  Intent:       {q['detected_intent']}")
        if q.get("anchor_action"):
            print(f"  Anchor:       {q['anchor_action']}")

    if not verbose:
        wall = result.get("wall_time_s", 0)
        conf = result.get("retrieval", {}).get("confidence", "")
        gen = result.get("timing", {}).get("generation_ms", 0)
        parts = [f"{wall:.1f}s"]
        if gen:
            parts.append(f"gen {gen:.0f}ms")
        if conf:
            parts.append(f"conf={conf}")
        if sources:
            parts.append(f"{len(sources)} sources")
        sep = " \u00b7 "
        print(f"\n{styled(sep.join(parts), C.DIM)}")

    print()

def get_prompt(chat: RAGChat) -> str:
    model_short = chat.current_model or "?"
    db_short = chat.current_db or "?"
    cross = styled("\u00d7", C.DIM)
    arrow = styled("\u203a", C.BOLD, C.WHITE)
    return f"{styled(f'[{model_short}', C.CYAN)}{cross}{styled(f'{db_short}]', C.CYAN)} {arrow} "

def main():
    parser = argparse.ArgumentParser(
        description="SYR-RAG Live Chat \u2014 interactive terminal interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --model llama-3.2-3b --db full
  %(prog)s -m gemma-4-e4b -d openalex
  %(prog)s --list
  %(prog)s -o ./logs
        """,
    )
    parser.add_argument("-m", "--model", default=None,
                        help=f"Model key. Options: {', '.join(MODELS.keys())}")
    parser.add_argument("-d", "--db", default=None,
                        help=f"Database key. Options: {', '.join(DATABASES.keys())}")
    parser.add_argument("-o", "--output-dir", default=".",
                        help="Directory for the session JSON log (default: current dir)")
    parser.add_argument("--list", action="store_true",
                        help="List available models and databases, then exit")
    parser.add_argument("--verbose", action="store_true",
                        help="Start with verbose output enabled")
    parser.add_argument("--stateless", action="store_true",
                        help="Start in stateless mode (no conversation memory)")
    args = parser.parse_args()

    if args.list:
        print(f"\n{styled('Available Models:', C.BOLD, C.CYAN)}")
        for key, label in MODELS.items():
            print(f"  {styled(key, C.YELLOW):<22} {label}")
        print(f"\n{styled('Available Databases:', C.BOLD, C.CYAN)}")
        for key, label in DATABASES.items():
            print(f"  {styled(key, C.YELLOW):<22} {label}")
        print()
        return

    print(BANNER)

    if args.model and args.model not in MODELS:
        print(styled(f"Unknown model '{args.model}'. Available: {', '.join(MODELS.keys())}", C.RED))
        sys.exit(1)
    if args.db and args.db not in DATABASES:
        print(styled(f"Unknown database '{args.db}'. Available: {', '.join(DATABASES.keys())}", C.RED))
        sys.exit(1)

    model_key = args.model or pick_option("Select Model", MODELS, "Model")
    db_key = args.db or pick_option("Select Database", DATABASES, "Database")

    logger = SessionLogger(output_dir=args.output_dir)
    print(f"\n  {styled('Session log:', C.DIM)} {styled(logger.json_path, C.BLUE)}")

    chat = RAGChat(logger=logger)
    chat.verbose = args.verbose
    chat.stateless = args.stateless

    print()
    try:
        chat.switch_model(model_key)
        chat.switch_db(db_key)
    except Exception as e:
        print(styled(f"\nFailed to initialize: {e}", C.RED))
        print(styled("Make sure the RAG pipeline modules are in the same directory.", C.DIM))
        sys.exit(1)

    print(f"\n{styled('Ready! Type your question, or /help for commands.', C.BOLD, C.GREEN)}")
    print(styled("\u2500" * 60, C.DIM))

    while True:
        try:
            user_input = input(get_prompt(chat)).strip()
        except (EOFError, KeyboardInterrupt):
            print(styled("\n\nGoodbye!", C.CYAN))
            logger.log_event("session_end", {"reason": "interrupt"})
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input[1:].split(None, 1)
            cmd = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""
            if not handle_command(cmd, cmd_args, chat):
                logger.log_event("session_end", {"reason": "quit"})
                print(styled("\nGoodbye!", C.CYAN))
                break
            continue

        try:
            result = chat.ask(user_input)
            print_answer(result, verbose=chat.verbose)
        except KeyboardInterrupt:
            print(styled("\n  Generation interrupted.", C.YELLOW))
        except Exception as e:
            print(styled(f"\n  Error: {e}", C.RED))
            traceback.print_exc()

    total = logger.total_questions
    errs = logger.total_errors
    elapsed = time.perf_counter() - logger.t0
    print(f"\n{styled('Session Summary', C.BOLD, C.CYAN)}")
    print(styled("\u2500" * 40, C.DIM))
    print(f"  Questions:  {total}")
    print(f"  Errors:     {errs}")
    print(f"  Duration:   {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Runs:       {len(logger.runs)}")
    print(f"  Log saved:  {styled(logger.json_path, C.BLUE)}")
    print()

if __name__ == "__main__":
    main()