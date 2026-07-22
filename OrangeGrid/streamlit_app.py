# streamlit_app.py
import gc
import os
import html
import uuid

import streamlit as st

_LOCAL_EMBED_PATH = "/home/arapte/models/all-MiniLM-L6-v2"
if os.path.isdir(_LOCAL_EMBED_PATH):
    os.environ.setdefault("EMBED_MODEL", _LOCAL_EMBED_PATH)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/home/arapte/.cache/huggingface")

from rag_pipeline import answer_question, sanitize_answer_for_display
from runtime_settings import settings
from rag_engine import get_global_manager
from conversation_memory import hard_reset_memory

import psutil
import torch

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    HAS_AGRAPH = True
except ImportError:
    HAS_AGRAPH = False

ENGINE_MANAGER = get_global_manager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _esc(value: object) -> str:
    return html.escape(str(value or ""))


def _esc_answer(value: object) -> str:
    return html.escape(sanitize_answer_for_display(str(value or "")))


def _ram_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "used_gb": vm.used / (1024 ** 3),
        "total_gb": vm.total / (1024 ** 3),
        "pct": vm.percent,
    }


def _vram_stats():
    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        used_b = total_b - free_b
        return {
            "used_gb": used_b / (1024 ** 3),
            "total_gb": total_b / (1024 ** 3),
            "pct": (used_b / total_b * 100) if total_b else 0.0,
        }
    except Exception:
        return None


def _session_metrics(user_key: str) -> dict:
    state = ENGINE_MANAGER.store.load(user_key)
    return {
        "session_id": user_key,
        "turn_count": len(state.get("turns", []) or []),
        "summary_len": len(state.get("rolling_summary", "") or ""),
        "retrieval_confidence": str(state.get("retrieval_confidence", "") or ""),
    }


def _render_graph(g: dict, graph_key: str = "g") -> None:
    if not HAS_AGRAPH:
        return
    nodes_raw = g.get("nodes", []) or []
    edges_raw = g.get("edges", []) or []
    if not nodes_raw or not edges_raw:
        return

    toggle_key = f"graph_open_{graph_key}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False

    if st.button(
        "Hide Graph" if st.session_state[toggle_key] else "Show Graph",
        key=f"graph_btn_{graph_key}",
    ):
        st.session_state[toggle_key] = not st.session_state[toggle_key]

    if not st.session_state[toggle_key]:
        return

    TYPE_CONFIG = {
        "paper":      {"color": "#4A90D9", "size": 22},
        "researcher": {"color": "#E87B3A", "size": 26},
        "topic":      {"color": "#5BAD72", "size": 20},
        "author":     {"color": "#C0C0C0", "size": 16},
    }

    nodes = []
    for n in nodes_raw:
        nid = str(n.get("id", "")).strip()
        label = str(n.get("label", "")).strip() or nid
        ntype = str(n.get("type", "")).strip().lower()
        cfg = TYPE_CONFIG.get(ntype, {"color": "#888", "size": 18})
        nodes.append(Node(id=nid, label=label[:48], size=cfg["size"], color=cfg["color"]))

    edges = []
    for e in edges_raw:
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        if src and tgt:
            edges.append(Edge(source=src, target=tgt, label=str(e.get("type", ""))))

    with st.expander("Graph", expanded=True):
        agraph(
            nodes=nodes, edges=edges,
            config=Config(
                width="100%", height=int(g.get("height", 650)),
                directed=True, physics=True, hierarchical=False,
            ),
        )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Syracuse Research Assistant", layout="wide")
st.title("Syracuse Research Assistant")

if "user_key" not in st.session_state:
    st.session_state["user_key"] = str(uuid.uuid4())
USER_KEY = st.session_state["user_key"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # ── Session ───────────────────────────────────────────────────────────
    st.subheader("Session")

    if st.button("Reset Memory"):
        _safe_call(hard_reset_memory, USER_KEY)
        st.success("Memory cleared.")

    if st.button("Restart Conversation"):
        st.session_state["messages"] = []
        _safe_call(hard_reset_memory, USER_KEY)
        st.session_state["user_key"] = str(uuid.uuid4())
        st.success("Conversation restarted — models will reload on next question.")
        st.rerun()

    st.divider()

    # ── Dataset ───────────────────────────────────────────────────────────
    st.subheader("Dataset")

    _db_labels = ENGINE_MANAGER.dbm.display_labels()
    _label_to_key = {v: k for k, v in _db_labels.items()}
    _options = list(_db_labels.values())

    if "active_dataset" not in st.session_state:
        st.session_state["active_dataset"] = _db_labels.get(
            ENGINE_MANAGER.active_mode, _options[0])

    selected_label = st.selectbox(
        "Choose dataset",
        options=_options,
        index=(_options.index(st.session_state["active_dataset"])
               if st.session_state["active_dataset"] in _options else 0),
        key="dataset_selector",
    )

    if selected_label != st.session_state["active_dataset"]:
        new_mode = _label_to_key[selected_label]
        ENGINE_MANAGER.switch_mode(new_mode)
        settings.active_mode = new_mode
        ENGINE_MANAGER.papers_vs_cache.pop(new_mode, None)
        st.session_state["active_dataset"] = selected_label
        st.success(f"Switched to **{selected_label}**")
        st.rerun()
    else:
        current_key = _label_to_key.get(selected_label, "full")
        if ENGINE_MANAGER.active_mode != current_key:
            ENGINE_MANAGER.switch_mode(current_key)
            settings.active_mode = current_key

    st.divider()

    # ── Answer Model ──────────────────────────────────────────────────────
    st.subheader("Answer Model")

    _MODEL_OPTIONS = {
        "LLaMA 3.2 3B":              "llama-3.2-3b",
        "LLaMA 3.1 8B (native)":      "llama-3.1-8b",
        "Gemma 4 E2B (native)":       "gemma-4-e2b",
        "Gemma 4 E4B (native)":       "gemma-4-e4b",
        "Gemma 4 26B-A4B (native)":   "gemma-4-26b",
        "Gemma 4 31B (native)":       "gemma-4-31b",
        "Qwen 2.5 14B (native)":      "qwen-2.5-14b",
        "GPT-OSS 20B (native)":       "gpt-oss-20b",
        "LLaMA 3.3 70B (native)":     "llama-3.3-70b",
    }
    _model_labels = list(_MODEL_OPTIONS.keys())

    if "active_model" not in st.session_state:
        current_key = getattr(settings, "answer_model_key", "llama-3.2-3b")
        st.session_state["active_model"] = next(
            (lbl for lbl, k in _MODEL_OPTIONS.items() if k == current_key),
            _model_labels[0],
        )

    selected_model_label = st.selectbox(
        "Answer model",
        options=_model_labels,
        index=(_model_labels.index(st.session_state["active_model"])
               if st.session_state["active_model"] in _model_labels else 0),
        key="model_selector",
    )

    if selected_model_label != st.session_state["active_model"]:
        new_model_key = _MODEL_OPTIONS[selected_model_label]
        # Guard: skip reload if already loaded under a different alias
        already_loaded = (
            ENGINE_MANAGER.active_answer_model_key == new_model_key
            and ENGINE_MANAGER.answer_runtime is not None
        )
        if not already_loaded:
            settings.answer_model_key = new_model_key
            settings.llm_model = new_model_key
            try:
                mgr = ENGINE_MANAGER
                if mgr.answer_runtime is not None:
                    mgr.answer_runtime.close()
                    mgr.answer_runtime = None
                    mgr.active_answer_model_key = ""
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            st.session_state["active_model"] = selected_model_label
            with st.spinner(f"Loading **{selected_model_label}**..."):
                try:
                    ENGINE_MANAGER.switch_answer_model(new_model_key)
                    st.success(f"**{selected_model_label}** loaded and ready.")
                except Exception as e:
                    st.error(f"Failed to load {selected_model_label}: {e}")
            st.rerun()
        else:
            st.session_state["active_model"] = selected_model_label

    st.divider()

    # ── System Memory ─────────────────────────────────────────────────────
    st.subheader("System Memory")
    ram_placeholder = st.empty()
    vram_placeholder = st.empty()

    st.divider()
    st.subheader("Session Diagnostics")
    diagnostics_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Sidebar refresh
# ---------------------------------------------------------------------------

def _refresh_sidebar(user_key: str) -> None:
    ram = _ram_stats()
    ram_placeholder.caption(
        f"**RAM** {ram['used_gb']:.1f} / {ram['total_gb']:.1f} GB  ({ram['pct']:.0f}%)"
    )
    vram = _vram_stats()
    if vram:
        vram_placeholder.caption(
            f"**VRAM** {vram['used_gb']:.1f} / {vram['total_gb']:.1f} GB  ({vram['pct']:.0f}%)"
        )
    else:
        vram_placeholder.caption("**VRAM** — GPU not available")

    metrics = _session_metrics(user_key)
    diagnostics_placeholder.json({
        "session_id":           metrics["session_id"],
        "turn_count":           metrics["turn_count"],
        "summary_len":          metrics["summary_len"],
        "retrieval_confidence": metrics["retrieval_confidence"],
    })


_refresh_sidebar(USER_KEY)

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for idx, msg in enumerate(st.session_state["messages"]):
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant":
            content = msg.get("content", "")
            st.write(_esc_answer(content))

            timing_caption = msg.get("timing_caption", "")
            if timing_caption:
                st.caption(timing_caption)

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"Retrieved Sources ({len(sources)})", expanded=False):
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"[{i}] {s}", unsafe_allow_html=False)

            graph_error = msg.get("graph_error", "")
            if graph_error:
                st.warning(graph_error)
            _render_graph(msg.get("graph", {}) or {}, graph_key=f"history_{idx}")
        else:
            st.write(_esc(msg.get("content", "")))

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask about Syracuse research...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(_esc(prompt))

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            out = answer_question(
                prompt,
                user_key=USER_KEY,
                use_graph=True,
                stateless=False,
            )

        answer_text = (out.get("answer") or "").strip()
        if answer_text:
            st.write(_esc_answer(answer_text))
        else:
            answer_text = "I could not generate an answer, but the retrieved sources are shown below."
            st.write(answer_text)

        # Timing
        llm_calls = out.get("llm_calls", {}) if isinstance(out.get("llm_calls"), dict) else {}
        timing = out.get("timing_ms", {}) if isinstance(out.get("timing_ms"), dict) else {}
        timing_caption = (
            f"LLM calls — answer: {llm_calls.get('answer_llm_calls', 0)} | "
            f"utility: {llm_calls.get('utility_llm_calls', 0)} | "
            f"total: {timing.get('total_ms', 0):.0f} ms"
        )
        st.caption(timing_caption)

        # Sources
        sources = list(out.get("sources", []) or [])[:10]
        if sources:
            with st.expander(f"Retrieved Sources ({len(sources)})", expanded=False):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"[{i}] {s}", unsafe_allow_html=False)

        # Graph
        g = out.get("graph_graph", {}) or {}
        if out.get("graph_error"):
            st.warning(out["graph_error"])
        _render_graph(g, graph_key=f"turn_{len(st.session_state['messages'])}")

    # Save to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer_text,
        "timing_caption": timing_caption,
        "sources": sources,
        "graph": g,
        "graph_error": out.get("graph_error", ""),
    })

    # Refresh sidebar after answer
    st.rerun()