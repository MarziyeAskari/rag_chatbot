# mlops/evaluate_rag.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from mlops.common import load_settings, build_vector_store, build_rag_chain, safe_run_ctx, ensure_dir
from mlops.mlflow_utils import MlflowTracker
from mlops.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def evaluate(
    questions: List[str],
    expected_answers: Optional[List[str]] = None,
    *,
    use_mlflow: bool = True,
    output_path: str = "mlops/artifacts/eval_results.json",
) -> dict:
    s = load_settings()
    tracker = MlflowTracker("rag_evaluation") if use_mlflow else None

    with safe_run_ctx(tracker, "evaluate_rag"):
        store = build_vector_store(s)
        size = store.get_collection_size()
        if size == 0:
            raise RuntimeError("Vector store is empty. Run training first.")

        chain = build_rag_chain(s, store)

        results = []
        for q in questions:
            try:
                out = chain.query(question=q, top_k=s.top_k)
                results.append({
                    "question": q,
                    "answer": out["answer"],
                    "num_sources": len(out.get("source_documents", [])),
                    "source_documents": out.get("source_documents", []),
                })
            except Exception as e:
                results.append({"question": q, "error": str(e)})

        m = calculate_metrics(results, expected_answers=expected_answers)

        payload = {"results": results, "metrics": m}
        ensure_dir(str(Path(output_path).parent))
        Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        if tracker:
            tracker.log_params({"vector_store_size": size, "top_k": s.top_k})
            tracker.log_metrics(m)
            tracker.log_artifact(output_path, artifact_path="evaluation")

        logger.info(f"Saved evaluation to {output_path}")
        return payload
