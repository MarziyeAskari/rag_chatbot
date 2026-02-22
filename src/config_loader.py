from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

def _read_yaml(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- APP ----------
    app_name: str = "RAG Chatbot"
    app_version: str = "1.0.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    app_mode: str = Field(default="local", alias="APP_MODE")  # local | aws
    upload_async: bool = Field(default=False, alias="UPLOAD_ASYNC")  # true for AWS async
    sqs_queue_url: str = Field(default="", alias="SQS_QUEUE_URL")  # only needed in aws/async

    # ---------- LLM ----------
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # ---------- EMBEDDINGS ----------
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---------- VECTOR STORE ----------
    vector_store_type: str = Field(default="chroma", alias="VECTOR_STORE_TYPE")   # chroma|pgvector
    vector_store_path: str = "./data/vectorstore"
    vector_store_db_url: str = Field(default="", alias="VECTOR_STORE_DB_URL")
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

    # ---------- UPLOADS (AWS) ----------
    upload_storage: str = Field(default="local", alias="UPLOAD_STORAGE")  # local|s3
    s3_bucket_name: str = Field(default="", alias="S3_BUCKET_NAME")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_prefix: str = "uploads/"

    # ---------- SESSIONS ----------
    use_database_session: bool = True
    session_database_url: str = Field(default="", alias="SESSION_DATABASE_URL")
    session_database_path: str = "./data/sessions.db"
    session_timeout: int = 24
    max_history_per_session: int = 50


def get_setting(config_path: str = "configs/config.yaml") -> Settings:
    cfg = _read_yaml(config_path)
    merged: Dict[str, Any] = {}

    # APP
    app = cfg.get("app", {})
    merged.update({
        "app_name": app.get("name", "RAG Chatbot"),
        "app_version": app.get("version", "1.0.0"),
        "app_host": app.get("host", "0.0.0.0"),
        "app_port": app.get("port", 8000),
        "debug": app.get("debug", False),
    })

    runtime = cfg.get("runtime", {})
    merged.update({
        "app_mode": runtime.get("app_mode"),
        "upload_async": runtime.get("upload_async"),
        "sqs_queue_url": runtime.get("sqs_queue_url"),
    })

    # LLM
    llm = cfg.get("llm", {})
    merged.update({
        "llm_provider": llm.get("provider", "openai"),
        "llm_model": llm.get("model", "gpt-4o-mini"),
        "llm_temperature": llm.get("temperature", 0.7),
        "llm_max_tokens": llm.get("max_tokens", 1000),
    })

    # EMBEDDINGS (normalize provider)
    emb = cfg.get("embeddings", {})
    provider = (emb.get("provider") or "sentence-transformers").strip().lower().replace(" ", "-")
    merged.update({
        "embedding_provider": provider,
        "embedding_model": emb.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
    })

    # VECTOR DB (IMPORTANT: correct key name)
    vs = cfg.get("vector_database", {})
    merged.update({
        "vector_store_type": vs.get("type", "chroma"),
        "vector_store_path": vs.get("persist_directory", "./data/vectorstore"),
        "vector_store_db_url": vs.get("vector_store_db_url", ""),
        "collection_name": vs.get("collection_name", "rag_documents"),
    })

    # DOCUMENT PROCESSING
    dp = cfg.get("document_processing", {})
    merged.update({
        "chunk_size": dp.get("chunk_size", 1000),
        "chunk_overlap": dp.get("chunk_overlap", 200),
    })

    # RETRIEVAL
    ret = cfg.get("retrieval", {})
    merged.update({
        "top_k": ret.get("top_k", 2),
        "similarity_threshold": ret.get("similarity_threshold", 0.7),
    })

    # PATHS
    paths = cfg.get("path", {})
    merged.update({
        "documents_path": paths.get("documents", "./data/documents"),
        "processed_path": paths.get("processed", "./data/processed"),
        "uploads_path": paths.get("uploads", "./data/uploads"),
    })

    # UPLOADS
    uploads = cfg.get("uploads", {})
    merged.update({
        "upload_storage": uploads.get("storage"),
        "s3_bucket_name": uploads.get("s3_bucket"),
        "aws_region": uploads.get("aws_region"),
        "s3_prefix": uploads.get("s3_prefix"),
    })

    # SESSION (IMPORTANT: correct key name)
    sess = cfg.get("session", {})
    merged.update({
        "use_database_session": sess.get("use_database", True),
        "session_database_path": sess.get("database_path", "./data/sessions.db"),
        "session_timeout": sess.get("session_timeout_hour", 24),
        "session_database_url": sess.get("database_url"),
        "max_history_per_session": sess.get("max_history_per_session", 50),
    })
    merged = {k: v for k, v in merged.items() if v is not None}
    return Settings(**merged)
