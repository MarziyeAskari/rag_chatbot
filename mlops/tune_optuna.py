
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

import optuna
from optuna.samplers import TPESampler

from mlops.common import load_settings, build_processor, build_vector_store, build_rag_chain, safe_run_ctx, ensure_dir
from mlops.mlflow_utils import MlflowTracker
from mlops.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def tune(
    questions: List[str],
    expected_answers: Optional[List[str]] = None,
    *,
    n_trials: int = 20,
    use_mlflow: bool = True,
    study_path: str = "mlops/artifacts/optuna_study.json",
) -> Dict:
    s = load_settings()
    tracker = MlflowTracker("rag_hyperparam_tuning") if use_mlflow else None

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    docs_dir = Path(s.documents_path)
    if not docs_dir.exists():
        raise FileNotFoundError(f"documents_path not found: {docs_dir}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "chunk_size": trial.suggest_int("chunk_size", 500, 2000, step=100),
            "chunk_overlap": trial.suggest_int("chunk_overlap", 50, 500, step=50),
            "top_k": trial.suggest_int("top_k", 2, 10),
            "temperature": trial.suggest_float("temperature", 0.1, 1.0, step=0.1),
            "max_tokens": trial.suggest_int("max_tokens", 500, 2000, step=100),
        }

        with tempfile.TemporaryDirectory(prefix="chroma_trial_") as tmpdir:
            processor = build_processor(s)
            processor.chunk_size = params["chunk_size"]
            processor.chunk_overlap = params["chunk_overlap"]

            store = build_vector_store(s, persist_directory=tmpdir)
            chunks = processor.process_directory(str(docs_dir))
            if chunks:
                store.add_documents(chunks)

            chain = build_rag_chain(
                s,
                store,
                top_k=params["top_k"],
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
            )

            results = []
            for q in questions:
                try:
                    out = chain.query(question=q, top_k=params["top_k"])
                    results.append({"question": q, "answer": out["answer"], "num_sources": len(out["source_documents"])})
                except Exception as e:
                    results.append({"question": q, "error": str(e)})

            m = calculate_metrics(results, expected_answers=expected_answers)
            return float(m["overall_score"])

    with safe_run_ctx(tracker, "optuna_tune"):
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        payload = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
        }
        ensure_dir(str(Path(study_path).parent))
        Path(study_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if tracker:
            tracker.log_params(payload["best_params"])
            tracker.log_metrics({"best_value": float(payload["best_value"])})
            tracker.log_artifact(study_path, artifact_path="optuna")

    return payload
