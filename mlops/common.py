# mlops/common.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import nullcontext

import logging

from src.config_loader import get_setting, Settings
from src.documents_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_chain import RagChain

logger = logging.getLogger(__name__)


@dataclass
class BuiltComponents:
    settings: Settings
    processor: DocumentProcessor
    store: VectorStore
    chain: RagChain


def load_settings() -> Settings:
    return get_setting()


def build_processor(settings: Settings) -> DocumentProcessor:
    return DocumentProcessor(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


def build_vector_store(settings: Settings, persist_directory: Optional[str] = None) -> VectorStore:
    persist_dir = persist_directory or settings.vector_store_path
    api_key = settings.openai_api_key if settings.embedding_provider.lower() == "openai" else None
    return VectorStore(
        persist_directory=persist_dir,
        collection_name=settings.collection_name,
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        api_key=api_key,
    )


def build_rag_chain(
    settings: Settings,
    store: VectorStore,
    *,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> RagChain:
    api_key = settings.openai_api_key if settings.llm_provider.lower() == "openai" else None
    return RagChain(
        vector_store=store,
        llm_provider=settings.llm_provider.lower(),
        model=settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.llm_max_tokens,
        api_key=api_key,
        top_k=top_k if top_k is not None else settings.top_k,
        similarity_threshold=settings.similarity_threshold,
    )


def build_all(settings: Optional[Settings] = None, *, persist_directory: Optional[str] = None) -> BuiltComponents:
    settings = settings or load_settings()
    processor = build_processor(settings)
    store = build_vector_store(settings, persist_directory=persist_directory)
    chain = build_rag_chain(settings, store)
    return BuiltComponents(settings=settings, processor=processor, store=store, chain=chain)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_run_ctx(tracker, run_name: str):
    return tracker.start_run(run_name=run_name) if tracker else nullcontext()
