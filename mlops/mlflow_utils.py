import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import mlflow.langchain

logger = logging.getLogger(__name__)


class MlflowTracker:
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "rag_chatbot",
    ):
        self.experiment_name = experiment_name

        # ---- Tracking URI ----
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow_dir = Path("./mlruns")
            mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")

        # ---- Experiment ----
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")


            self.experiment_id = experiment_id
            mlflow.set_experiment(experiment_name)

        except Exception as e:
            logger.warning(f"Error setting experiment: {e}")
            mlflow.set_experiment(experiment_name)

            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                self.experiment_id = experiment.experiment_id if experiment else None
            except Exception:
                self.experiment_id = None

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        if mlflow.active_run() is not None:
            return mlflow.active_run()

        if run_name is None:
            run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        return mlflow.start_run(run_name=run_name, tags=tags or {})

    def end_run(self):
        if mlflow.active_run() is not None:
            mlflow.end_run()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged params: {params}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged metrics: {metrics}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        mlflow.log_artifacts(local_dir, artifact_path)
        logger.info(f"Logged artifacts from: {local_dir}")

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------
    def log_model(
        self,
        model,
        artifact_path: str = "rag_model",
        registered_model_name: Optional[str] = None,
    ):
        try:
            if not hasattr(model, "retrieval_qa"):
                raise ValueError("Model must have a `retrieval_qa` attribute")

            mlflow.langchain.log_model(
                lc_model=model.retrieval_qa,
                artifact_path=artifact_path,
            )

            if registered_model_name:
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
                mlflow.register_model(model_uri, registered_model_name)
                logger.info(f"Registered model: {registered_model_name}")

            logger.info(f"logged model: {artifact_path}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")

            model_config = {
                "llm_provider": model.llm.__class__.__name__,
                "temperature": getattr(model.llm, "temperature", None),
                "max_tokens": getattr(model.llm, "max_tokens", None),
            }

            mlflow.log_dict(model_config, f"{artifact_path}/config.json")

    # ------------------------------------------------------------------
    # Search & load
    # ------------------------------------------------------------------
    def search_runs(
        self,
        experiment_ids: Optional[list] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100,
    ):
        if experiment_ids is None:
            experiment_ids = [self.experiment_id] if self.experiment_id else None
        return mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
        )

    def get_best_run(
        self,
        metric: str = "accuracy",
        ascending: bool = False,
        max_results: int = 100,
    ):
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id] if self.experiment_id else None,
            max_results=max_results)
        if runs.empty or metric not in runs.columns:
            return None

        return runs.sort_values(metric, ascending=ascending).iloc[0]

    def load_model(self, model_uri: str):
        return mlflow.langchain.load_model(model_uri)

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
    ):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
        )
        logger.info(f"Transitioned model {name} v{version} to {stage}")
