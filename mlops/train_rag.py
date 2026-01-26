# mlops/train_rag.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from mlops.common import load_settings, build_processor, build_vector_store, safe_run_ctx
from mlops.mlflow_utils import MlflowTracker

logger = logging.getLogger(__name__)


def train(documents_dir: str, *, clear_existing: bool = False, use_mlflow: bool = True) -> None:
    s = load_settings()

    tracker = MlflowTracker("rag_training") if use_mlflow else None
    run_name = "train_rag"

    with safe_run_ctx(tracker, run_name):
        if tracker:
            tracker.log_params({
                "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap,
                "embedding_provider": s.embedding_provider,
                "embedding_model": s.embedding_model,
                "collection_name": s.collection_name,
                "vector_store_path": s.vector_store_path,
            })

        processor = build_processor(s)
        store = build_vector_store(s)

        if clear_existing:
            store.delete_collection()
            store._load_or_create_vectorstore()

        p = Path(documents_dir)
        if not p.exists():
            raise FileNotFoundError(f"Documents directory not found: {p}")

        chunks = processor.process_directory(str(p))
        if not chunks:
            logger.warning("No chunks created; nothing to ingest.")
            return

        store.add_documents(chunks)
        size = store.get_collection_size()

        logger.info(f"Ingested chunks. Vector store size: {size}")

        if tracker:
            tracker.log_metrics({"vector_store_size": float(size)})
