#conversation_memory.py
import logging
import os

logger = logging.getLogger(__name__)

def hard_reset_memory(user_key: str) -> None:
    import sys
    mod = sys.modules.get("rag_engine")
    manager = getattr(mod, "_GLOBAL_MANAGER", None) if mod else None
    if manager is not None:
        try:
            manager.reset_session(user_key)
        except Exception:
            logger.warning("Failed to reset session via EngineManager", exc_info=True)
    else:
        try:
            from session_store import SessionStore
            SessionStore(os.getenv("RAG_STATE_DB", "chat_state.sqlite")).reset(user_key)
        except Exception:
            logger.warning("Failed to reset SessionStore directly", exc_info=True)
        try:
            import chromadb
            client = chromadb.PersistentClient(path=os.getenv("RAG_MEMORY_DIR", "chroma_memory"))
            col = client.get_collection("memory")
            ids = ((col.get(where={"session_id": user_key}) or {}).get("ids") or [])
            if ids:
                col.delete(ids=ids)
        except Exception:
            logger.warning("Failed to reset chroma memory directly", exc_info=True)