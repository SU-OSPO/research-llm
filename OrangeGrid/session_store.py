# session_store.py
import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from runtime_settings import settings

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_state (
  session_id TEXT PRIMARY KEY,
  rolling_summary TEXT NOT NULL,
  turns_json TEXT NOT NULL,
  extra_state_json TEXT NOT NULL DEFAULT '{}'
);
"""

_DEFAULT_ROLLING_SUMMARY = (
    "Current focus:\n- (none)\n"
    "Core entities:\n- (none)\n"
    "Key themes:\n- (none)\n"
    "Constraints:\n- Use only retrieved Syracuse corpus context.\n"
    "Open questions:\n- (none)"
)


def _safe_json_loads(raw: Optional[str], default: Any) -> Any:
    try:
        data = json.loads(raw) if raw else default
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


def _trim_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    keep_n = max(2, int(getattr(settings, "session_turns_keep", 24)))
    max_chars = max(2000, int(getattr(settings, "session_turns_max_chars", 32000)))
    target_chars = max(1000, int(getattr(settings, "session_turn_trim_target_chars", 24000)))

    out: List[Dict[str, str]] = []
    for obj in list(turns or [])[-keep_n:]:
        role = str((obj or {}).get("role", "") or "").strip().lower()
        text = str((obj or {}).get("text", "") or "").strip()
        if role in {"user", "assistant"} and text:
            out.append({"role": role, "text": text})

    total_chars = sum(len(t["text"]) for t in out)
    if total_chars > max_chars:
        while len(out) > 2 and total_chars > target_chars:
            total_chars -= len(out.pop(0)["text"])

    return out


def _sanitize_anchor(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    anchor_value = str(value.get("value", "") or "").strip()
    if not anchor_value:
        return {}
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0) or 0.0)))
    except Exception:
        confidence = 0.0
    return {
        "type": str(value.get("type", "metadata") or "metadata").strip().lower(),
        "value": anchor_value,
        "source": str(value.get("source", "retrieval") or "retrieval").strip(),
        "confidence": confidence,
    }


_EXTRA_STATE_KEYS = {
    "last_focus": ("", str), "last_topic": ("", str),
    "anchor_last_action": ("", str), "summary_updated": (False, bool),
    "retrieval_confidence": ("", str), "rewrite_anchor_valid": (False, bool),
    "rewrite_blocked": (False, bool),
}

def _sanitize_extra_state(extra_state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(extra_state, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, (default, typ) in _EXTRA_STATE_KEYS.items():
        if key in extra_state:
            out[key] = typ(extra_state.get(key, default) or default)
    if "anchor_support_ratio" in extra_state:
        try:
            out["anchor_support_ratio"] = float(extra_state.get("anchor_support_ratio", 0.0) or 0.0)
        except Exception:
            out["anchor_support_ratio"] = 0.0
    out["anchor"] = _sanitize_anchor(extra_state.get("anchor"))
    return out


class SessionStore:
    """SQLite-backed session store with per-thread connections."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._schema_done = False
        self._ensure_connection()

    def _ensure_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=True)
            for pragma in ("journal_mode=WAL", "synchronous=NORMAL", "temp_store=MEMORY"):
                conn.execute(f"PRAGMA {pragma};")
            self._local.conn = conn
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._ensure_connection()

    def _init_db(self) -> None:
        if self._schema_done:
            return
        with self._init_lock:
            if self._schema_done:
                return
            conn = self._conn
            conn.execute(SCHEMA_SQL)
            conn.commit()
            try:
                cols = [row[1] for row in conn.execute("PRAGMA table_info(chat_state);").fetchall()]
                if "extra_state_json" not in cols:
                    conn.execute("ALTER TABLE chat_state ADD COLUMN extra_state_json TEXT NOT NULL DEFAULT '{}';")
                    conn.commit()
            except Exception:
                pass
            self._schema_done = True

    def load(self, session_id: str) -> Dict[str, Any]:
        self._init_db()
        try:
            row = self._conn.execute(
                "SELECT rolling_summary, turns_json, extra_state_json FROM chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            row = self._conn.execute(
                "SELECT rolling_summary, turns_json FROM chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if not row:
            return {"rolling_summary": _DEFAULT_ROLLING_SUMMARY, "turns": []}

        rolling_summary, turns_json = row[0], row[1]
        extra_state_json = row[2] if len(row) > 2 else "{}"
        turns = _trim_turns(_safe_json_loads(turns_json, []))
        extra_state = _sanitize_extra_state(_safe_json_loads(extra_state_json, {}))
        summary_value = str(rolling_summary or "") or _DEFAULT_ROLLING_SUMMARY

        state: Dict[str, Any] = {"rolling_summary": summary_value, "turns": turns}
        if extra_state:
            state["extra_state"] = extra_state
            for key in ("last_focus", "last_topic", "anchor_last_action",
                        "summary_updated", "retrieval_confidence"):
                if key in extra_state:
                    state[key] = extra_state[key]
            if "anchor" in extra_state:
                anchor_val = extra_state["anchor"]
                state["anchor"] = anchor_val if isinstance(anchor_val, dict) and anchor_val else {}
        return state

    def save(self, session_id: str, rolling_summary: str,
             turns: List[Dict[str, str]], extra_state: Optional[Dict[str, Any]] = None) -> None:
        self._init_db()
        payload = json.dumps(_trim_turns(turns or []), ensure_ascii=False)
        conn = self._conn

        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = conn.execute(
                "SELECT rolling_summary, extra_state_json FROM chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            base_summary = str(existing[0] or "") if existing else ""
            base_extra = _sanitize_extra_state(
                _safe_json_loads(existing[1], {}) if existing and len(existing) > 1 else {})

            if isinstance(extra_state, dict):
                for k, v in _sanitize_extra_state(extra_state).items():
                    base_extra[k] = v

            summary_payload = str(rolling_summary or "")
            if not summary_payload.strip():
                summary_payload = base_summary if base_summary.strip() else _DEFAULT_ROLLING_SUMMARY

            conn.execute("""
                INSERT INTO chat_state(session_id, rolling_summary, turns_json, extra_state_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  rolling_summary=excluded.rolling_summary,
                  turns_json=excluded.turns_json,
                  extra_state_json=excluded.extra_state_json
            """, (session_id, summary_payload, payload,
                  json.dumps(base_extra or {}, ensure_ascii=False)))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def reset(self, session_id: str) -> None:
        self._init_db()
        self._conn.execute("DELETE FROM chat_state WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass