"""
Configuration loader module (YAML + env overrides)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


def _read_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


class Settings(BaseSettings):
    app_name: str = "RAG Chatbot"
    app_version: str = "1.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    # ---------- LLM ----------
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    openai_api_key: str = ""

    # ---------- EMBEDDINGS ----------
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------- VECTOR STORE ----------
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vectorstore"
    collection_name: str = "rag_documents"

    # ---------- DOCUMENT PROCESSING ----------
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ---------- RETRIEVAL ----------
    top_k: int = 5
    similarity_threshold: float = 0.7

    # ---------- PATHS ----------
    documents_path: str = "./data/documents"
    processed_path: str = "./data/processed"
    uploads_path: str = "./data/uploads"

    # ---------- SESSIONS ----------
    use_database_session: bool = True
    session_database_url: str = ""          # e.g. postgresql://...
    session_database_path: str = "./data/session.db"
    session_timeout: int = 24
    max_history_per_session: int = 50

    # ---------- EMBEDDING FINETUNE ----------
    embedding_finetune_enabled: bool = False
    embedding_finetune_base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_finetune_output_dir: str = "./models/embedding_finetuned"
    embedding_finetune_epochs: int = 1
    embedding_finetune_batch_size: int = 16
    embedding_finetune_max_sessions: int = 200
    embedding_finetune_max_pairs_per_session: int = 50

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",  # optional but very useful
        )


def get_setting(config_path: str = "configs/config.yaml") -> Settings:
    cfg = _read_yaml(config_path)

    merged: Dict[str, Any] = {}

    app = cfg.get("app", {})
    merged.update({
        "app_name": app.get("name"),
        "app_version": app.get("version"),
        "app_host": app.get("host"),
        "app_port": app.get("port"),
        "debug": app.get("debug"),
    })

    llm = cfg.get("llm", {})
    merged.update({
        "llm_provider": llm.get("provider"),
        "llm_model": llm.get("model"),
        "llm_temperature": llm.get("temperature"),
        "llm_max_tokens": llm.get("max_tokens"),
        # YAML uses api-key, we map it safely:
        "openai_api_key": llm.get("api-key") or llm.get("api_key"),
    })

    emb = cfg.get("embeddings", {})
    merged.update({
        "embedding_provider": emb.get("provider"),
        "embedding_model": emb.get("model"),
    })

    vs = cfg.get("vector_store", {})
    merged.update({
        "vector_store_type": vs.get("type"),
        "vector_store_path": vs.get("persist_directory"),
        "collection_name": vs.get("collection_name"),
    })

    dp = cfg.get("document_processing", {})
    merged.update({
        "chunk_size": dp.get("chunk_size"),
        "chunk_overlap": dp.get("chunk_overlap"),
    })

    ret = cfg.get("retrieval", {})
    merged.update({
        "top_k": ret.get("top_k"),
        "similarity_threshold": ret.get("similarity_threshold"),
    })

    paths = cfg.get("path", {})
    merged.update({
        "documents_path": paths.get("documents") or "./data/documents",
        "processed_path": paths.get("processed") or "./data/processed",
        "uploads_path": paths.get("uploads") or "./data/uploads",
    })

    sess = cfg.get("sessions", {})
    merged.update({
        "use_database_session": sess.get("use_database"),
        "session_database_url": sess.get("database_url"),
        "session_database_path": sess.get("database_path"),
        "session_timeout": sess.get("session_timeout_hour"),
        "max_history_per_session": sess.get("max_history_per_session"),
    })

    ft = cfg.get("embedding_finetuning", {})
    merged.update({
        "embedding_finetune_enabled": ft.get("enabled"),
        "embedding_finetune_base_model": ft.get("base_model"),
        "embedding_finetune_output_dir": ft.get("output_dir"),
        "embedding_finetune_epochs": ft.get("epoch"),
        "embedding_finetune_batch_size": ft.get("batch_size"),
        "embedding_finetune_max_sessions": ft.get("max_session"),
        "embedding_finetune_max_pairs_per_session": ft.get("max_pairs_per_session"),
    })

    # Remove None values so defaults remain
    merged = {k: v for k, v in merged.items() if v is not None}

    # Create Settings (env vars will override automatically)
    settings = Settings(**merged)

    # If env has OPENAI_API_KEY, prefer it
    settings.openai_api_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)

    return settings
