import datetime
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any,Iterable

import mlflow
import mlflow.langchain

try:
    # Only needed for version comparisons; lightweight and common in ML stacks.
    from packaging.version import Version
except Exception:  # pragma: no cover
    Version = None  # type: ignore

logger = logging.getLogger(__name__)


class MlflowTracker:
    def __init__(
            self,
            tracking_uri: Optional[str] = None,
            experiment_name: str = "rag_chatbot",
            thread_safe: bool = True,
            default_nested: bool = True,
            tracking_dir: Optional[str] = None,
    ):

        self.experiment_name = experiment_name
        self.thread_safe = thread_safe
        self.default_nested = default_nested

        self._lock = threading.Lock() if thread_safe else None

        # ---- Tracking URI (CI/CD & Docker readiness) ----
        resolved_uri = (
                tracking_uri
                or os.environ.get("MLFLOW_TRACKING_URI")
                or self._default_file_tracking_uri(tracking_dir=tracking_dir)
        )
        self._with_lock(lambda: mlflow.set_tracking_uri(resolved_uri))
        self.tracking_uri = resolved_uri

        # ---- Tracking URI
        # if tracking_uri:
        #     mlflow.set_tracking_uri(tracking_uri)
        # else:
        #     mlflow_dir = Path("./mlruns")
        #     mlflow.set_tracking_uri(f"file://{mlflow_dir.absolute()}")

        # ---- Experiment ----

        exp = self._with_lock(lambda: mlflow.set_experiment(self.experiment_name))
        self.experiment_id = getattr(exp, "experiment_id", None)
        self._log_debug(
            "Initialized mlflowTracker",
            extra={
                "tracking_uri": self.tracking_uri,
                "thread_safe": self.thread_safe,
                "experiment_id": self.experiment_id,
                "experiment_name": self.experiment_name,
                "default_nested": self.default_nested,
            }
        )


    @staticmethod
    def _default_file_tracking_uri(tracking_dir: Optional[str] = None) -> str:
        # Use absolute path so it behaves consistently in Docker/CI.
        base = Path(tracking_dir) if tracking_dir else Path(os.getenv("MLFLOW_TRACKING_DIR", "./mlruns"))
        base = base.expanduser().resolve()
        # Ensure directory exists (safe local behavior). If running on remote tracking server, env URI should be set.
        base.mkdir(parents=True, exist_ok=True)
        return f"file://{base.as_posix()}"

    def _with_lock(self, fn):
        if self._lock is None:
            return fn()
        with self._lock:
            return fn()

    def _current_run_id(self) -> Optional[str]:
        run = mlflow.active_run()
        return run.info.run_id if run is not None else None

    def _log_debug(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        run_id = self._current_run_id()
        payload = {"run_id": run_id, **(extra or {})}
        logger.debug(f"{msg} | {payload}")

    def _log_info(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        run_id = self._current_run_id()
        payload = {"run_id": run_id, **(extra or {})}
        logger.info(f"{msg} | {payload}")

    def _log_warning(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        run_id = self._current_run_id()
        payload = {"run_id": run_id, **(extra or {})}
        logger.warning(f"{msg} | {payload}")

    def _log_error(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        run_id = self._current_run_id()
        payload = {"run_id": run_id, **(extra or {})}
        logger.error(f"{msg} | {payload}")

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------
    def start_run(
            self,
            run_name: Optional[str] = None,
            tags: Optional[Dict[str, str]] = None,
            nested: Optional[bool] = None,
            reuse_if_active: bool = False,
    ):
        # if mlflow.active_run() is not None:
        #     return mlflow.active_run()

        if run_name is None:
            run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

        def _start():
            active = mlflow.active_run()
            if active is not None and reuse_if_active:
                self._log_info("Reusing active run", {"run_name": run_name})
                return active

            use_nested = self.default_nested if nested is None else nested
            run = mlflow.start_run(run_name=run_name, tags=tags or {}, nested=use_nested)
            self._log_info("Started run", {"run_name": run_name})
            return run

        return self._with_lock(_start)

    def end_run(self):
        def _end():
            if mlflow.active_run() is not None:
                run_id = self._current_run_id()
                mlflow.end_run()
                logger.info(f"Ended run | {{'run_id': {run_id}}}")

        return self._with_lock(_end)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def log_params(self, params: Dict[str, Any]):
        def _log():
            for key, value in params.items():
                mlflow.log_param(key, value)
            self._log_info(f"Logged params: {params}", {"keys": list(params.keys())})

        return self._with_lock(_log)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        def _log():
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            self._log_info("Logged metrics", {"keys": list(metrics.keys()), "step": step})

        return self._with_lock(_log)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        def _log():
            mlflow.log_artifact(local_path, artifact_path)
            self._log_info("Logged artifact", {"local_path": local_path, "artifact_path": artifact_path})

        return self._with_lock(_log)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        def _log():
            mlflow.log_artifacts(local_dir, artifact_path)
            self._log_info("Logged artifacts", {"local_dir": local_dir, "artifact_path": artifact_path})

        return self._with_lock(_log)

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------

    def _langchain_available(self) -> bool:
        try:
            import mlflow.langchain
        except Exception as e:
            self._log_warning("mlflow.langchain not available (install extras or upgrade)", {"error": str(e)})
            return False
        if Version is not None:
            try:
                v = Version(mlflow.__version__)
                if v < Version("2.7.0"):
                    self._log_warning("MLflow version may be too old for reliable mlflow.langchain",
                                      {"mlflow": mlflow.__version__})
            except Exception as e:
                pass
        return True

    def log_model(
            self,
            model: Any,
            artifact_path: str = "rag_model",
            registered_model_name: Optional[str] = None,
    ):
        """
        Expects either:
          - model.retrieval_qa (LangChain Runnable/Chain), or
          - a custom object you want to log config for (fallback path)
        """

        def _log():
            if not self._langchain_available():
                # Fallback: log config only
                self._log_warning("Skipping LangChain model logging; logging config only.",
                                  {"artifact_path": artifact_path})
                self._log_model_config_fallback(model, artifact_path)
                return

            try:
                import mlflow.langchain  # local import so missing dependency doesn't break module import

                retrieval_qa = getattr(model, "retrieval_qa", None)
                if retrieval_qa is None:
                    raise ValueError("Model must have a `retrieval_qa` attribute for mlflow.langchain.log_model")

                mlflow.langchain.log_model(
                    lc_model=retrieval_qa,
                    artifact_path=artifact_path,
                )

                if registered_model_name:
                    active = mlflow.active_run()
                    if active is None:
                        raise RuntimeError("No active run to register from. Call start_run() first.")
                    model_uri = f"runs:/{active.info.run_id}/{artifact_path}"

                    mlflow.register_model(model_uri, registered_model_name)
                    self._log_info("Registered model", {"name": registered_model_name, "model_uri": model_uri})

                self._log_info("Logged model", {"artifact_path": artifact_path})

            except Exception as e:
                self._log_error("Error logging model; falling back to config logging",
                                {"error": str(e), "artifact_path": artifact_path})
                self._log_model_config_fallback(model, artifact_path)

        return self._with_lock(_log)

    def _log_model_config_fallback(self, model: Any, artifact_path: str):
        llm = getattr(model, "llm", None)
        model_config = {
            "llm_provider": getattr(getattr(llm, "__class__", None), "__name__", None),
            "temperature": getattr(llm, "temperature", None),
            "max_tokens": getattr(llm, "max_tokens", None),
            "model_class": getattr(getattr(model, "__class__", None), "__name__", None),
        }

        try:
            mlflow.log_dict(model_config, f"{artifact_path}/config.json")
            self._log_info("Logged model config fallback", {"artifact_path": artifact_path})
        except Exception as e:
            self._log_error("Failed to log model config fallback", {"error": str(e), "artifact_path": artifact_path})

    # ------------------------------------------------------------------
        # Search & load
    # ------------------------------------------------------------------

    def search_runs(
        self,
        experiment_ids: Optional[Iterable[str]] = None,
        filter_string: Optional[str] = None,
        max_results: int = 100,
    ):
        def _search():
            exp_ids = list(experiment_ids) if experiment_ids is not None else None
            if exp_ids is None:
                if self.experiment_id is None:
                    raise ValueError("experiment_id is None. Ensure set_experiment succeeded or pass experiment_ids.")
                exp_ids = [self.experiment_id]

            runs = mlflow.search_runs(
                experiment_ids=exp_ids,
                filter_string=filter_string,
                max_results=max_results,
            )
            self._log_info("Searched runs", {"experiment_ids": exp_ids, "max_results": max_results})
            return runs

        return self._with_lock(_search)

    def get_best_run(
            self,
            metric: str = "accuracy",
            ascending: bool = False,
            max_results: int = 100,
    ):
        runs = self.search_runs(max_results=max_results)
        if runs is None or runs.empty:
            self._log_warning("No runs found", {"metric": metric})
            return None

        # MLflow search_runs returns columns like "metrics.<name>" in many setups.
        metric_col_candidates = [metric, f"metrics.{metric}"]
        metric_col = next((c for c in metric_col_candidates if c in runs.columns), None)
        if metric_col is None:
            self._log_warning("Metric not found in runs", {"metric": metric, "columns_sample": list(runs.columns)[:20]})
            return None

        best = runs.sort_values(metric_col, ascending=ascending).iloc[0]
        self._log_info("Selected best run", {"metric_col": metric_col, "ascending": ascending})
        return best

    def load_model(self, model_uri: str):
        def _load():
            if not self._langchain_available():
                raise RuntimeError("mlflow.langchain not available; cannot load LangChain model.")
            import mlflow.langchain
            model = mlflow.langchain.load_model(model_uri)
            self._log_info("Loaded model", {"model_uri": model_uri})
            return model

        return self._with_lock(_load)

        # ------------------------------------------------------------------
        # Model registry
        # ------------------------------------------------------------------

    def transition_model_stage(self, name: str, version: str, stage: str):
        def _transition():
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(name=name, version=version, stage=stage)
            self._log_info("Transitioned model stage", {"name": name, "version": version, "stage": stage})

        return self._with_lock(_transition)


