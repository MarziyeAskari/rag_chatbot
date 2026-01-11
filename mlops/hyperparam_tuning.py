import json
from pathlib import Path
from typing import Dict

import optuna
from optuna.samplers import TPESampler

from app.main import document_processor, vector_store, rag_chain
from mlops.metrics import calculate_metrics
from mlops.mlflow_utils import MlflowTracker
from src.config_loader import get_setting
import logging

from src.documents_processor import DocumentProcessor
from src.rag_chain import RagChain
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HyperparamTuning:
    def __init__(self,
                 test_question: list,
                 excepted_answers: list = None,
                 n_trials: int = 20,
                 study_name: str = "rag_hyperparameter_tuning", ):
        self.test_question = test_question
        self.excepted_answers = excepted_answers
        self.n_trials = n_trials
        self.study_name = study_name
        self.setting = get_setting()

        self.mlflow_tracker = MlflowTracker(experiment_name="rag_hyperparameter_tuning")

        self.study = optuna.create_study(
            study_name=self.study_name,
            directions="maximize",
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "chunk_size": trial.suggest_int("chunk_size", 500, 2000, step=100),
            "chunk_overlap": trial.suggest_int("chunk_overlap", 50, 500, step=50),
            "top_k": trial.suggest_int("top_k", 3, 10),
            "temperature": trial.suggest_float("temperature", 0.1, 1.0, step=0.1),
            "max_tokens": trial.suggest_int("max_tokens", 500, 2000, step=100),
        }

        logger.info(f"Trial {trial.number}: Testing parameters: {params}")

        try:
            with self.mlflow_tracker.start_run(
                    run_name=f"trial{trial.number}",
                    tags={"trial": str(trial.number), "study": self.study_name}
            ):
                self.mlflow_tracker.log_params(params)

                original_chunk_size = self.setting.chunk_size
                original_chunk_overlap = self.setting.chunk_overlap
                original_top_k = self.setting.top_k
                original_temperature = self.setting.temperature
                original_max_tokens = self.setting.max_tokens

                self.setting.chunk_size = params["chunk_size"]
                self.setting.chunk_overlap = params["chunk_overlap"]
                self.setting.top_k = params["top_k"]
                self.setting.temperature = params["temperature"]
                self.setting.max_tokens = params["max_tokens"]

                logger.info(f"Rebuilding vector store with new parameters ...")
                document_processor = DocumentProcessor(
                    chunk_size=params["chunk_size"],
                    chunk_overlap=params["chunk_overlap"],
                )

                vector_store = VectorStore(
                    persist_directory=f"{self.setting.vector_store_path}_trial_{trial.number}",
                    collection_name=self.setting.collection_name,
                    embedding_model=self.setting.embedding_model,
                    embedding_provider=self.setting.embedding_provider,
                    api_key=self.setting.api_key if self.setting.embedding_provider == "openai" else None,
                )

                document_path = Path(self.setting.document_path)
                if document_path.exists():
                    chunks = document_processor.process_directory(str(document_path))
                    if chunks:
                        vector_store.add_documents(chunks)

                rag_chain = RagChain(
                    vector_store=vector_store,
                    llm_provider=self.setting.llm_provider,
                    model=self.setting.model,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    api_key=self.setting.openai_api_key if self.setting.llm_provider == "openai" else None,
                )
                results = []
                for question in self.test_question:
                    try:
                        result = rag_chain.query(question, top_key=params["top_key"])
                        results.append({
                            "question": question,
                            "answer": result["answer"],
                            "num_source": len(result["source_documents"]), })
                    except Exception as e:
                        logger.warning(f"Error in trial {trial.number}: {str(e)}")
                        results.append({"question": question, "error": str(e)})

                metrics = calculate_metrics(results, self.excepted_answers)
                self.mlflow_tracker.log_metrics(metrics)

                if trial.number == 0 or metrics["overall_score"] > self.study.best_value:
                    self.mlflow_tracker.log_model(
                        rag_chain,
                        artifact_path="rag_model",
                        registered_model_name="rag_chatbot" if trial.number == 0 else None,
                    )

                self.setting.chunk_size = original_chunk_size
                self.setting.chunk_overlap = original_chunk_overlap
                self.setting.top_k = original_top_k
                self.setting.llm_temperature = original_temperature
                self.setting.llm_max_tokens = original_max_tokens

                return float(metrics["overall_score"])
        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {str(e)}")
            raise 0.0
    def optimize(self) -> Dict[str, float]:

        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        best_params = self.study.best_params
        best_value = self.study.best_value

        logger.info(f"Optimization completed")
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best value found: {best_value}")

        study_file = Path("mlops/optuna_study.json")
        study_date = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(self.study.n_trials),
            "study_name": self.study_name,
        }

        with open (study_file, "w") as f:
            json.dump(study_date, f, indent=2)

        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": self.study,
        }


if __name__ == "__main__":
    test_question = [
           "What is the main topic of the documents?",
        "Can you summarize the key points?",
    ]
    tuner = HyperparamTuning(test_question=test_question, n_trials=10)
    results = tuner.optimize()
    print(f"Best parameters: {results['best_params']}")
    print(f"Best score: {results['best_value']}")