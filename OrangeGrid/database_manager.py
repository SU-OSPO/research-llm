# database_manager.py
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config_full as config


@dataclass
class DatabaseConfig:
    mode: str
    chroma_dir: str
    collection: str
    description: str
    neo4j_db: str = ""
    display_label: str = ""


class DatabaseManager:
    def __init__(self) -> None:
        self.configs: Dict[str, DatabaseConfig] = {}
        self.active_config_name: str = "full"
        self.register_config("full", DatabaseConfig(
            mode="full",
            chroma_dir=config.CHROMA_DIR_FULL,
            collection=getattr(config, "CHROMA_COLLECTION_FULL",
                               getattr(config, "CHROMA_COLLECTION", "papers_all")),
            description="Full text papers with metadata",
            neo4j_db=config.NEO4J_DB,
            display_label="Legacy DB",
        ))
        self.register_config("openalex", DatabaseConfig(
            mode="openalex",
            chroma_dir=getattr(config, "CHROMA_DIR_OPENALEX", ""),
            collection=getattr(config, "CHROMA_COLLECTION_OPENALEX", "syracuse_papers"),
            description="OpenAlex papers with Docling fulltext",
            neo4j_db=getattr(config, "NEO4J_DB_OPENALEX", "syr-rag-openalex"),
            display_label="OpenAlex DB",
        ))
        self.register_config("abstracts", DatabaseConfig(
            mode="abstracts",
            chroma_dir=getattr(config, "CHROMA_DIR_ABSTRACTS", ""),
            collection=getattr(config, "CHROMA_COLLECTION_ABSTRACTS", "syracuse_abstracts"),
            description="OpenAlex abstracts only (no fulltext)",
            neo4j_db=getattr(config, "NEO4J_DB_ABSTRACTS", "syr-rag-abstracts"),
            display_label="Abstracts Only",
        ))

    def register_config(self, name: str, cfg: DatabaseConfig) -> None:
        self.configs[name] = cfg

    def resolve_mode(self, requested_mode: str) -> str:
        available = list(self.configs.keys())
        if not available:
            return ""
        req = (requested_mode or "").strip()
        if req in self.configs:
            return req
        by_lower = {k.lower(): k for k in self.configs}
        return by_lower.get(req.lower(), available[0])

    def switch_config(self, name: str) -> bool:
        target = self.resolve_mode(name)
        if target:
            self.active_config_name = target
            return True
        return False

    def get_active_config(self) -> Optional[DatabaseConfig]:
        return self.configs.get(self.active_config_name)

    def get_config(self, name: str) -> Optional[DatabaseConfig]:
        return self.configs.get(name)

    def list_configs(self) -> List[str]:
        return list(self.configs.keys())

    def ensure_dirs_exist(self) -> None:
        for cfg in self.configs.values():
            if cfg and cfg.chroma_dir:
                os.makedirs(cfg.chroma_dir, exist_ok=True)

    def display_labels(self) -> Dict[str, str]:
        return {name: (cfg.display_label or name) for name, cfg in self.configs.items()}

    def get_active_neo4j_db(self) -> str:
        cfg = self.get_active_config()
        return (cfg.neo4j_db if cfg and cfg.neo4j_db else config.NEO4J_DB)

    def validate_active_config(self) -> Dict[str, Any]:
        """Post-switch health check — call after switch_config()."""
        cfg = self.get_active_config()
        if not cfg or not cfg.chroma_dir or not os.path.isdir(cfg.chroma_dir):
            return {"healthy": False, "reason": "missing"}
        try:
            import chromadb
            n = chromadb.PersistentClient(path=cfg.chroma_dir).get_collection(cfg.collection).count()
            return {"healthy": n > 0, "doc_count": n}
        except Exception as e:
            return {"healthy": False, "reason": str(e)}