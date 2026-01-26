# mlops/registry.py
from __future__ import annotations

from typing import Optional, Dict, List
import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, tracking_uri: Optional[str] = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_from_run(self, model_name: str, run_id: str, artifact_path: str = "rag_model", tags: Optional[Dict[str, str]] = None) -> str:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = self.client.create_model_version(name=model_name, source=model_uri, run_id=run_id, tags=tags or {})
        logger.info(f"Registered {model_name} v{mv.version} from {model_uri}")
        return str(mv.version)

    def promote(self, model_name: str, version: str, stage: str = "Production", archive_existing: bool = True):
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Promoted {model_name} v{version} to {stage}")
