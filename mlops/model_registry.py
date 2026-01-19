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

    def register_model(
        self,
        model_name: str,
        model_uri: str,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        model_uri should usually be:
          - runs:/<run_id>/<artifact_path>
        Example:
          model_uri = f"runs:/{run_id}/rag_model"
        """
        try:
            mv = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                tags=tags or {},
            )
            logger.info(f"Registered model {model_name} version {mv.version} from {model_uri}")
            return str(mv.version)
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise

    def latest_version(self, model_name: str, stage: Optional[str] = None) -> Optional[str]:
        """
        Returns version string or None.
        """
        try:
            versions = self.client.search_model_versions(f"name = '{model_name}'")
            if not versions:
                return None

            # Filter by stage if requested
            if stage:
                versions = [v for v in versions if v.current_stage == stage]

            if not versions:
                return None

            # Pick highest version number
            versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
            return str(versions_sorted[0].version)
        except Exception as e:
            logger.error(f"Failed to get latest version for {model_name}: {e}")
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = True,
    ):
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions,
            )
            logger.info(f"Transitioned {model_name} v{version} to stage {stage}")
        except Exception as e:
            logger.error(f"Failed to transition {model_name} v{version} to {stage}: {e}")
            raise

    def get_model_versions(self, model_name: str) -> List[Dict]:
        try:
            versions = self.client.search_model_versions(f"name = '{model_name}'")
            return [
                {
                    "version": mv.version,
                    "stage": mv.current_stage,
                    "creation_timestamp": mv.creation_timestamp,
                    "run_id": mv.run_id,
                    "source": mv.source,
                }
                for mv in versions
            ]
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {e}")
            raise

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """
        Loads a LangChain model from the MLflow model registry.
        You must have mlflow.langchain installed and the model logged using mlflow.langchain.log_model.
        """
        if version and stage:
            raise ValueError("Provide either version or stage, not both.")

        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            latest = self.latest_version(model_name)
            if not latest:
                raise ValueError(f"No versions found for model {model_name}")
            model_uri = f"models:/{model_name}/{latest}"

        import mlflow.langchain
        logger.info(f"Loading model from {model_uri}")
        return mlflow.langchain.load_model(model_uri)

    def delete_model_version(self, model_name: str, version: str):
        try:
            self.client.delete_model_version(name=model_name, version=version)
            logger.info(f"Deleted model {model_name} version {version}")
        except Exception as e:
            logger.error(f"Failed to delete model {model_name} version {version}: {e}")
            raise

    def promote_to_production(self, model_name: str, version: str, archive_existing: bool = True):
        self.transition_model_stage(
            model_name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=archive_existing,
        )
        logger.info(f"Promoted model {model_name} version {version} to Production")
