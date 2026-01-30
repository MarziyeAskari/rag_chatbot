# mlops/mlflow_utils.py
from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Optional, Dict, Any

import mlflow
import logging

logger = logging.getLogger(__name__)


class MlflowTracker:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None, tracking_dir: str = "./mlruns"):
        uri = tracking_uri or self._default_file_tracking_uri(tracking_dir)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        self.experiment_name = experiment_name
        self.tracking_uri = uri

    @staticmethod
    def _default_file_tracking_uri(tracking_dir: Optional[str] = None) -> str:
        base = Path(tracking_dir) if tracking_dir else Path(os.getenv("MLFLOW_TRACKING_DIR", "./mlruns"))
        base = base.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: as_uri() returns correct Windows form: file:///D:/...
        return base.as_uri()

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, nested: bool = False):
        run_name = run_name or f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return mlflow.start_run(run_name=run_name, tags=tags or {}, nested=nested)

    def log_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_artifacts(self, dir_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifacts(dir_path, artifact_path=artifact_path)

    def active_run_id(self) -> Optional[str]:
        run = mlflow.active_run()
        return run.info.run_id if run else None

