
"""
RAG Benchmark Script
====================
Tests every permutation of (model × database) with a fixed set of questions.
Results are saved to a timestamped JSON file and a human-readable summary.

Usage:
    python benchmark_rag.py
    python benchmark_rag.py --models 3b 8b
    python benchmark_rag.py --databases full
    python benchmark_rag.py --dry-run

Output:
    benchmark_results_YYYYMMDD_HHMMSS.json   — full structured results
    benchmark_summary_YYYYMMDD_HHMMSS.txt    — human-readable summary table
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAG_DEBUG", "0")
os.environ.setdefault("RAG_EMBED_DEVICE", "cpu")
os.environ.setdefault("RAG_LLM_TIMEOUT_S", "300")

# ---------------------------------------------------------------------------
# Subprocess isolation: each model×db permutation runs in its own process
# so a CUDA device-side assert in one model cannot corrupt the next.
# ---------------------------------------------------------------------------
_SINGLE_RUN_FLAG = "--single-run"

def _run_single_in_subprocess(model_key: str, db_key: str,
                               questions: List[str], output_path: str,
                               timeout: int = 7200) -> Dict[str, Any]:
    """Spawn a child process for one model×db permutation."""
    import subprocess, tempfile
    config = {"model_key": model_key, "db_key": db_key,
              "questions": questions, "output_path": output_path}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                      delete=False, dir=".") as f:
        json.dump(config, f, ensure_ascii=False)
        config_path = f.name

    env = {**os.environ, "RAG_LLM_TIMEOUT_S": os.environ.get("RAG_LLM_TIMEOUT_S", "300")}
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), _SINGLE_RUN_FLAG, config_path],
            timeout=timeout, env=env,
        )
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"model_key": model_key, "db_key": db_key,
                "error": f"Subprocess exited {proc.returncode} but no output",
                "model_label": MODELS.get(model_key, model_key),
                "db_label": DATABASES.get(db_key, db_key),
                "model_load_s": 0, "questions": [], "db_health": {}}
    except subprocess.TimeoutExpired:
        return {"model_key": model_key, "db_key": db_key,
                "error": f"Subprocess timed out after {timeout}s",
                "model_label": MODELS.get(model_key, model_key),
                "db_label": DATABASES.get(db_key, db_key),
                "model_load_s": 0, "questions": [], "db_health": {}}
    except Exception as e:
        return {"model_key": model_key, "db_key": db_key,
                "error": f"Subprocess failed: {e}",
                "model_label": MODELS.get(model_key, model_key),
                "db_label": DATABASES.get(db_key, db_key),
                "model_load_s": 0, "questions": [], "db_health": {}}
    finally:
        try: os.unlink(config_path)
        except OSError: pass


def _single_run_entrypoint(config_path: str) -> None:
    """Child-process entry: load manager, run one conversation, write result."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model_key = config["model_key"]
    db_key = config["db_key"]
    questions = config["questions"]
    output_path = config["output_path"]

    runner = BenchmarkRunner()
    runner._ensure_imports()
    result = runner.run_conversation(questions, model_key, db_key)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


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

QUESTIONS: List[str] = [
    "who is duncan brown",
    "Summarize the mechanisms described in his papers",
    "what is LIGO and Virgo?",
    "who else studies it",

    "Which faculty have demonstrated expertise in intrinsically disordered proteins, especially in relation to environmental sensing, emergent cellular behavior, or plant systems?",
    "who is william gearty",
    "what does he study",
    "who else studies paleontology",

    "Identify researchers whose work could contribute to precision, personalized recovery pathways for neurological injury using continuous behavioral or physiological monitoring.",
    "Tell me about Alexander Nitz's research at Syracuse University",
    "computer science research at syracuse university",

    "who is Melissa Green",
    "what topics does she publish on",

    "Compare the research areas of Duncan Brown and Alexander Nitz",

    "What papers were published on machine learning between 2020 and 2024?",

    "List faculty working on climate change or environmental sustainability",
    "who works on artificial intelligence",

    "who is collin capano, what does he study, and who does he collaborate with",

    "who is David Smith",
    "tell me about his most cited work",

    "what kind of research does the physics department do",
    "biology research at syracuse",

    "Is there any research on quantum computing at Syracuse?",
    "who studies dark matter",

    "tell me more about that",
    "what else has he published",

    "Which researchers have collaborated on gravitational wave detection?",

    "what is the most recent paper in the database",
    "how many papers does the corpus contain",
    "switch topic",
    "who works on both machine learning and healthcare",
]

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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

def _answer_word_count(text: str) -> int:
    return len((text or "").split())

def _source_count(out: Dict[str, Any]) -> int:
    return len(out.get("sources", []) or [])

class BenchmarkRunner:
    def __init__(self):
        self.mgr = None
        self.settings = None

    def _ensure_imports(self):
        """Lazy import so the script can parse args / --dry-run without loading models."""
        if self.mgr is not None:
            return
        from rag_pipeline import answer_question
        from rag_engine import get_global_manager
        from runtime_settings import settings
        from conversation_memory import hard_reset_memory

        self.mgr = get_global_manager()
        self.settings = settings
        self.answer_question = answer_question
        self.hard_reset_memory = hard_reset_memory

    def _switch_model(self, model_key: str) -> float:
        """Switch answer model, return time taken in seconds."""
        t0 = time.perf_counter()
        self.settings.answer_model_key = model_key
        self.settings.llm_model = model_key
        self.mgr.switch_answer_model(model_key)
        elapsed = time.perf_counter() - t0
        return elapsed

    def _switch_database(self, db_key: str) -> None:
        """Switch active database/dataset."""
        self.mgr.switch_mode(db_key)
        self.settings.active_mode = db_key
        if hasattr(self.mgr, "papers_vs_cache"):
            self.mgr.papers_vs_cache.pop(db_key, None)

    def _validate_database(self, db_key: str) -> Dict[str, Any]:
        """Check DB health before running questions against it."""
        health: Dict[str, Any] = {"healthy": True, "doc_count": -1}
        try:
            if hasattr(self.mgr, "dbm") and hasattr(self.mgr.dbm, "validate_active_config"):
                health = self.mgr.dbm.validate_active_config()
            if health.get("doc_count", -1) < 0:
                try:
                    vs = self.mgr.get_papers_vs(db_key)
                    col = getattr(vs, "_collection", None)
                    if col is not None:
                        health["doc_count"] = int(col.count())
                except Exception:
                    pass
        except Exception:
            pass
        return health

    def _reset_session(self, user_key: str) -> None:
        """Full session reset between question sequences."""
        try:
            self.hard_reset_memory(user_key)
        except Exception:
            pass

    def _free_vram(self) -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run_single_question(self, question: str, user_key: str,
                            stateless: bool = False) -> Dict[str, Any]:
        """Run a single question and return structured result."""
        t0 = time.perf_counter()
        error = ""
        out: Dict[str, Any] = {}
        try:
            out = self.answer_question(
                question, user_key=user_key,
                use_graph=False, stateless=stateless,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        answer = str(out.get("answer", "") or "").strip()
        timing = out.get("timing_ms", {}) if isinstance(out.get("timing_ms"), dict) else {}
        llm_calls = out.get("llm_calls", {}) if isinstance(out.get("llm_calls"), dict) else {}
        chroma = out.get("chroma_retrieval", {}) if isinstance(out.get("chroma_retrieval"), dict) else {}
        user_query = out.get("user_query", {}) if isinstance(out.get("user_query"), dict) else {}

        return {
            "question": question,
            "answer": answer,
            "answer_word_count": _answer_word_count(answer),
            "source_count": _source_count(out),
            "error": error,
            "timing": {
                "total_ms": round(elapsed_ms, 2),
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

    def run_conversation(self, questions: List[str], model_key: str,
                         db_key: str, *, on_progress=None,
                         _results_ref: Optional[List] = None) -> Dict[str, Any]:
        """Run a full conversation (sequence of questions) for one model×db pair."""
        self._ensure_imports()

        user_key = f"bench_{model_key}_{db_key}_{uuid.uuid4().hex[:8]}"

        print(f"\n  [DB] Switching to {db_key}...")
        self._switch_database(db_key)

        health = self._validate_database(db_key)
        if not health.get("healthy", True):
            reason = health.get("reason", "unknown")
            doc_count = health.get("doc_count", 0)
            print(f"  [WARN] Database '{db_key}' unhealthy: {reason} (docs: {doc_count})")

        print(f"  [MODEL] Loading {model_key}...")
        try:
            model_load_s = self._switch_model(model_key)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  [ERROR] Failed to load {model_key}: {e}")
            return {
                "model_key": model_key,
                "model_label": MODELS.get(model_key, model_key),
                "db_key": db_key,
                "db_label": DATABASES.get(db_key, db_key),
                "model_load_s": 0,
                "error": f"Model load failed: {e}",
                "questions": [],
                "db_health": health,
            }
        print(f"  [MODEL] {model_key} loaded in {model_load_s:.1f}s")

        self._reset_session(user_key)

        results = _results_ref if _results_ref is not None else []

        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q[:60]}{'...' if len(q) > 60 else ''}")
            t0 = time.perf_counter()
            result = self.run_single_question(q, user_key=user_key, stateless=False)
            wall_s = time.perf_counter() - t0
            result["wall_time_s"] = round(wall_s, 2)
            results.append(result)

            wc = result["answer_word_count"]
            src = result["source_count"]
            gen = result["timing"]["generation_ms"]
            conf = result["retrieval"]["confidence"]
            anchor = result["query"]["anchor_action"]
            err = result["error"]
            status = f"  \u2192 {wc} words, {src} sources, {gen:.0f}ms gen, conf={conf}"
            if anchor:
                status += f", anchor={anchor}"
            if err:
                status += f" ERROR: {err}"
            print(status)

            if on_progress is not None:
                on_progress()

        self._reset_session(user_key)

        return {
            "model_key": model_key,
            "model_label": MODELS.get(model_key, model_key),
            "db_key": db_key,
            "db_label": DATABASES.get(db_key, db_key),
            "model_load_s": round(model_load_s, 2),
            "session_id": user_key,
            "error": "",
            "questions": results,
            "db_health": health,
        }

    def run_all(self, model_keys: List[str], db_keys: List[str],
                questions: List[str], *, output_dir: str = ".") -> Dict[str, Any]:
        """Run all permutations, saving results incrementally after each one."""
        self._ensure_imports()

        tag = _now_tag()
        gpu = _gpu_info()
        combos = list(product(model_keys, db_keys))
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, f"benchmark_results_{tag}.json")
        summary_path = os.path.join(output_dir, f"benchmark_summary_{tag}.txt")

        print(f"\n{'='*70}")
        print(f"RAG BENCHMARK \u2014 {len(combos)} permutations "
              f"({len(model_keys)} models \u00d7 {len(db_keys)} databases)")
        print(f"Questions per conversation: {len(questions)}")
        print(f"Total question runs: {len(combos) * len(questions)}")
        if gpu.get("available"):
            print(f"GPU: {gpu.get('device_name', '?')} "
                  f"(SM {gpu.get('compute_capability', '?')}, "
                  f"{gpu.get('vram_total_gb', '?')} GB)")
        print(f"Output: {json_path}")
        print(f"{'='*70}\n")

        all_runs: List[Dict[str, Any]] = []
        current_run_ref: List[Optional[Dict[str, Any]]] = [None]
        benchmark_t0 = time.perf_counter()

        def _build_results() -> Dict[str, Any]:
            runs_snapshot = list(all_runs)
            if current_run_ref[0] is not None:
                runs_snapshot.append(current_run_ref[0])
            return {
                "benchmark_tag": tag,
                "timestamp": _now_iso(),
                "total_time_s": round(time.perf_counter() - benchmark_t0, 2),
                "gpu": gpu,
                "status": "in_progress" if len(all_runs) < len(combos) else "complete",
                "progress": f"{len(all_runs)}/{len(combos)}",
                "config": {
                    "models": model_keys,
                    "databases": db_keys,
                    "question_count": len(questions),
                    "questions": questions,
                },
                "runs": runs_snapshot,
            }

        def _save_incremental() -> None:
            try:
                snapshot = _build_results()
                tmp = json_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
                os.replace(tmp, json_path)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"  [WARN] Failed to save incremental results: {e}")

        for idx, (model_key, db_key) in enumerate(combos, 1):
            _LINE = "\u2500" * 60
            header = f"[{idx}/{len(combos)}] {MODELS.get(model_key, model_key)} \u00d7 {DATABASES.get(db_key, db_key)}"
            print(f"\n{_LINE}")
            print(header)
            print(f"{_LINE}")

            # --- Each permutation runs in a subprocess with a fresh CUDA context ---
            run_output_path = os.path.join(
                output_dir, f"_bench_run_{model_key}_{db_key}_{tag}.json")

            try:
                run = _run_single_in_subprocess(
                    model_key, db_key, questions, run_output_path,
                    timeout=int(os.environ.get("RAG_LLM_TIMEOUT_S", "300")) * len(questions) + 600,
                )
            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Ctrl+C received — saving partial results...")
                all_runs.append({
                    "model_key": model_key, "model_label": MODELS.get(model_key, model_key),
                    "db_key": db_key, "db_label": DATABASES.get(db_key, db_key),
                    "error": "Interrupted by user", "model_load_s": 0, "questions": [],
                })
                current_run_ref[0] = None
                _save_incremental()
                print(f"[INTERRUPTED] Partial results saved to: {json_path}")
                sys.exit(0)

            # Print per-question summaries from the completed run
            for qi, qr in enumerate(run.get("questions", []), 1):
                wc = qr.get("answer_word_count", 0)
                src = qr.get("source_count", 0)
                gen = qr.get("timing", {}).get("generation_ms", 0)
                conf = qr.get("retrieval", {}).get("confidence", "")
                err = qr.get("error", "")
                line = f"  [{qi}/{len(questions)}] {qr.get('question', '')[:50]}..."
                line += f"  \u2192 {wc}w, {src}src, {gen:.0f}ms"
                if conf: line += f", {conf}"
                if err: line += f" ERROR: {err}"
                print(line)

            if run.get("error"):
                print(f"  [ERROR] {run['error']}")

            current_run_ref[0] = None
            all_runs.append(run)

            # Clean up temp file
            try: os.unlink(run_output_path)
            except OSError: pass

            _save_incremental()
            print(f"  [SAVED] Results saved ({len(all_runs)}/{len(combos)} permutations)")

        results = _build_results()
        results["status"] = "complete"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results

def generate_summary(results: Dict[str, Any], filepath: str) -> None:
    _DASH = "\u2014"
    lines: List[str] = []
    lines.append(f"RAG Benchmark Summary \u2014 {results['timestamp']}")
    lines.append(f"Total time: {results['total_time_s']:.0f}s")
    gpu = results.get("gpu", {})
    if gpu.get("available"):
        lines.append(f"GPU: {gpu.get('device_name', '?')} "
                      f"(SM {gpu.get('compute_capability', '?')})")
    lines.append("")

    col = (f"{'Model':<22} {'Database':<16} {'Load(s)':<8} {'Q#':<4} "
           f"{'Words':<7} {'Sources':<8} {'Gen(ms)':<9} {'Total(ms)':<10} "
           f"{'Conf':<12} {'Anchor':<28} {'Error'}")
    lines.append(col)
    lines.append("\u2500" * len(col))

    for run in results.get("runs", []):
        model = run.get("model_label", "?")[:21]
        db = run.get("db_label", "?")[:15]
        load_s = run.get("model_load_s", 0)
        run_err = run.get("error", "")

        if run_err:
            lines.append(f"{model:<22} {db:<16} {_DASH:<8} {_DASH:<4} "
                         f"{_DASH:<7} {_DASH:<8} {_DASH:<9} {_DASH:<10} "
                         f"{_DASH:<12} {_DASH:<28} {run_err}")
            continue

        for i, q in enumerate(run.get("questions", []), 1):
            words = q.get("answer_word_count", 0)
            srcs = q.get("source_count", 0)
            gen = q.get("timing", {}).get("generation_ms", 0)
            total = q.get("timing", {}).get("total_ms", 0)
            conf = q.get("retrieval", {}).get("confidence", "")
            anchor = q.get("query", {}).get("anchor_action", "")
            err = q.get("error", "")
            load_col = f"{load_s:.1f}" if i == 1 else ""
            lines.append(
                f"{model if i == 1 else '':<22} "
                f"{db if i == 1 else '':<16} "
                f"{load_col:<8} "
                f"{i:<4} "
                f"{words:<7} "
                f"{srcs:<8} "
                f"{gen:<9.0f} "
                f"{total:<10.0f} "
                f"{conf:<12} "
                f"{anchor:<28} "
                f"{err}"
            )
        lines.append("")

    lines.append("\n" + "=" * 60)
    lines.append("AGGREGATE STATS PER MODEL (across all databases)")
    lines.append("=" * 60)
    model_stats: Dict[str, Dict[str, Any]] = {}
    for run in results.get("runs", []):
        mk = run.get("model_key", "?")
        if mk not in model_stats:
            model_stats[mk] = {
                "label": run.get("model_label", mk),
                "total_words": 0, "total_questions": 0,
                "total_gen_ms": 0, "total_sources": 0,
                "errors": 0, "load_times": [],
                "confidence_counts": {},
            }
        ms = model_stats[mk]
        lt = run.get("model_load_s", 0)
        if lt > 0:
            ms["load_times"].append(lt)
        for q in run.get("questions", []):
            ms["total_questions"] += 1
            ms["total_words"] += q.get("answer_word_count", 0)
            ms["total_sources"] += q.get("source_count", 0)
            ms["total_gen_ms"] += q.get("timing", {}).get("generation_ms", 0)
            if q.get("error"):
                ms["errors"] += 1
            conf = q.get("retrieval", {}).get("confidence", "unknown")
            ms["confidence_counts"][conf] = ms["confidence_counts"].get(conf, 0) + 1

    lines.append(f"{'Model':<22} {'Avg Words':<11} {'Avg Gen(ms)':<13} "
                 f"{'Avg Srcs':<10} {'Errors':<8} {'Avg Load(s)':<12} {'Confidence'}")
    lines.append("\u2500" * 100)
    for mk, ms in model_stats.items():
        n = max(1, ms["total_questions"])
        avg_load = sum(ms["load_times"]) / max(1, len(ms["load_times"])) if ms["load_times"] else 0
        conf_str = ", ".join(f"{k}:{v}" for k, v in sorted(ms["confidence_counts"].items()))
        lines.append(
            f"{ms['label']:<22} "
            f"{ms['total_words']/n:<11.1f} "
            f"{ms['total_gen_ms']/n:<13.0f} "
            f"{ms['total_sources']/n:<10.1f} "
            f"{ms['errors']:<8} "
            f"{avg_load:<12.1f} "
            f"{conf_str}"
        )

    lines.append("\n" + "=" * 60)
    lines.append("AGGREGATE STATS PER DATABASE (across all models)")
    lines.append("=" * 60)
    db_stats: Dict[str, Dict[str, Any]] = {}
    for run in results.get("runs", []):
        dk = run.get("db_key", "?")
        if dk not in db_stats:
            db_stats[dk] = {
                "label": run.get("db_label", dk),
                "total_words": 0, "total_questions": 0,
                "total_retrieval_ms": 0, "total_sources": 0,
                "confidence_counts": {},
                "doc_count": run.get("db_health", {}).get("doc_count", -1),
            }
        ds = db_stats[dk]
        for q in run.get("questions", []):
            ds["total_questions"] += 1
            ds["total_words"] += q.get("answer_word_count", 0)
            ds["total_sources"] += q.get("source_count", 0)
            ds["total_retrieval_ms"] += q.get("timing", {}).get("retrieval_ms", 0)
            conf = q.get("retrieval", {}).get("confidence", "unknown")
            ds["confidence_counts"][conf] = ds["confidence_counts"].get(conf, 0) + 1

    lines.append(f"{'Database':<16} {'Docs':<10} {'Avg Words':<11} {'Avg Ret(ms)':<13} "
                 f"{'Avg Srcs':<10} {'Confidence distribution'}")
    lines.append("\u2500" * 90)
    for dk, ds in db_stats.items():
        n = max(1, ds["total_questions"])
        conf_str = ", ".join(f"{k}:{v}" for k, v in sorted(ds["confidence_counts"].items()))
        doc_str = str(ds["doc_count"]) if ds["doc_count"] >= 0 else "?"
        lines.append(
            f"{ds['label']:<16} "
            f"{doc_str:<10} "
            f"{ds['total_words']/n:<11.1f} "
            f"{ds['total_retrieval_ms']/n:<13.0f} "
            f"{ds['total_sources']/n:<10.1f} "
            f"{conf_str}"
        )

    lines.append("\n" + "=" * 60)
    lines.append("PER-QUESTION DIAGNOSTICS (across all runs)")
    lines.append("=" * 60)
    q_diag: Dict[int, Dict[str, Any]] = {}
    for run in results.get("runs", []):
        for i, q in enumerate(run.get("questions", []), 1):
            if i not in q_diag:
                q_diag[i] = {
                    "question": q.get("question", "")[:60],
                    "runs": 0, "weak": 0, "inconsistent": 0,
                    "high": 0, "medium": 0,
                    "total_words": 0, "total_gen_ms": 0,
                    "errors": 0,
                }
            d = q_diag[i]
            d["runs"] += 1
            conf = q.get("retrieval", {}).get("confidence", "")
            if conf in d:
                d[conf] += 1
            d["total_words"] += q.get("answer_word_count", 0)
            d["total_gen_ms"] += q.get("timing", {}).get("generation_ms", 0)
            if q.get("error"):
                d["errors"] += 1

    lines.append(f"{'Q#':<4} {'Avg Words':<10} {'Avg Gen':<9} "
                 f"{'Weak':<6} {'Inc':<6} {'Med':<6} {'High':<6} {'Question'}")
    lines.append("\u2500" * 90)
    for qi in sorted(q_diag.keys()):
        d = q_diag[qi]
        n = max(1, d["runs"])
        lines.append(
            f"{qi:<4} "
            f"{d['total_words']/n:<10.0f} "
            f"{d['total_gen_ms']/n:<9.0f} "
            f"{d['weak']:<6} "
            f"{d['inconsistent']:<6} "
            f"{d.get('medium', 0):<6} "
            f"{d['high']:<6} "
            f"{d['question']}"
        )

    report = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSummary written to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark \u2014 test all model\u00d7database permutations")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Model keys to test (default: all). Options: {', '.join(MODELS.keys())}")
    parser.add_argument("--databases", nargs="+", default=None,
                        help=f"Database keys to test (default: all). Options: {', '.join(DATABASES.keys())}")
    parser.add_argument("--questions-file", default=None,
                        help="Path to a text file with one question per line (overrides built-in QUESTIONS list)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files (default: current dir)")
    parser.add_argument("--stateless", action="store_true",
                        help="Run each question independently (no conversation state)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    model_keys = args.models if args.models else list(MODELS.keys())
    db_keys = args.databases if args.databases else list(DATABASES.keys())

    def _resolve_key(short, lookup):
        if short in lookup:
            return short
        matches = [k for k in lookup if short.lower() in k.lower()]
        return matches[0] if len(matches) == 1 else short

    model_keys = [_resolve_key(m, MODELS) for m in model_keys]
    db_keys = [_resolve_key(d, DATABASES) for d in db_keys]

    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        print(f"[INFO] Loaded {len(questions)} questions from {args.questions_file}")
    else:
        questions = QUESTIONS

    for mk in model_keys:
        if mk not in MODELS:
            print(f"ERROR: Unknown model '{mk}'. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)
    for dk in db_keys:
        if dk not in DATABASES:
            print(f"ERROR: Unknown database '{dk}'. Available: {', '.join(DATABASES.keys())}")
            sys.exit(1)

    combos = list(product(model_keys, db_keys))

    if args.dry_run:
        print(f"\nDRY RUN \u2014 {len(combos)} permutations would be tested:")
        for i, (mk, dk) in enumerate(combos, 1):
            print(f"  {i}. {MODELS[mk]} \u00d7 {DATABASES[dk]}")
        print(f"\nQuestions ({len(questions)}):")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
        print(f"\nTotal question runs: {len(combos) * len(questions)}")
        return

    runner = BenchmarkRunner()
    results = runner.run_all(model_keys, db_keys, questions, output_dir=args.output_dir)

    tag = results["benchmark_tag"]
    json_path = os.path.join(args.output_dir, f"benchmark_results_{tag}.json")
    summary_path = os.path.join(args.output_dir, f"benchmark_summary_{tag}.txt")
    generate_summary(results, summary_path)

    total = results["total_time_s"]
    n_runs = sum(len(r.get("questions", [])) for r in results.get("runs", []))
    n_question_errors = sum(1 for r in results.get("runs", [])
                            for q in r.get("questions", []) if q.get("error"))
    n_permutation_errors = sum(1 for r in results.get("runs", []) if r.get("error"))
    n_errors = n_question_errors + n_permutation_errors
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"  Permutations: {len(combos)}")
    print(f"  Total questions: {n_runs}")
    print(f"  Question errors: {n_question_errors}")
    print(f"  Permutation errors: {n_permutation_errors} (model load/crash)")
    print(f"  Total errors: {n_errors}")
    print(f"  Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"  Results: {json_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # --- Subprocess entry point: run a single model×db permutation ---
    if len(sys.argv) >= 3 and sys.argv[1] == _SINGLE_RUN_FLAG:
        _single_run_entrypoint(sys.argv[2])
        sys.exit(0)
    main()