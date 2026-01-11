from typing import Optional, Dict, List
import logging

import mlflow
from mlflow import MlflowClient
from torch.distributed.pipelining import stage

logger =logging.getLogger(__name__)

class ModelRegistry:

    def __init__(self
                 ,
                 tracking_url:Optional[str]=None):
        if tracking_url:
            mlflow.set_tracking_uri(tracking_url)

        self.client=MlflowClient()


    def register_model(
            self,
    model_name:str,
    model_url:str,
    tags:Optional[Dict[str, str]]=None)-> str:

        try:
            mv = self.client.create_model_version(model_name, model_url,tags=tags or {})
            logger.info(f"Registered model{model_name} version {mv.version}")
            return mv.version
        except Exception as e:
            logger.error(f"Failed to register model{model_name} : {e}")
            raise e

    def latest_version(self,model_name:str, stage: Optional[str]=None)->Optional[int]:
        try:
            if stage:
                versions = self.client.get_latest_versions(model_name, stages =[stage])
            else:
                versions = self.client.get_latest_versions(model_name)
            if versions:
                return int(versions[0].version)
            return None
        except Exception as e:
            logger.error(f"Failed to get latest version{e}")
            return None

    def transition_model_stage(self,
                               model_name:str,
                                version:str,
                               stage:str,
                               archive_existing_versions:bool=False ):
        try:
            self.client.transition_model_version_stage(name = model_name, version=version, stage=stage, archive_existing_versions=archive_existing_versions)
            logger.info(f"Transition model {model_name} v {version} to {stage} succeeded")
        except Exception as e:
            logger.error(f"Failed to transition model{model_name} to {stage} : {e}")
            raise

    def get_model_versions(self,
                           model_name:str)->List[Dict]:
        try:
            versions = self.client.search_model_versions(f"model = '{model_name}'")
            return [{
                "versions": mv.version,
                "stage": stage,
                "creation_timestamp":mv.creation_timestamp,
                "run_id":mv.run_id,
            }
            for mv in versions]
        except Exception as e:
            logger.error(f"Failed to get model versions{e}")
            raise []
    def load_model(self,
                   model_name:str,
                   version:str,
                   stage:Optional[str]=None,):
        try:
            if version:
                model_url = f"models: / = {model_name}/{version}"
            elif stage:
                model_url = f"models: / = {model_name}/{stage}"
            else:
                latest_version = self.latest_version(model_name, stage)
                if latest_version:
                    model_url = f"models: / = {model_name}/{latest_version}"
                else:
                    raise ValueError("No model version available for {model_name}")
            import mlflow.langchain
            return mlflow.langchain.load_model(model_url)
        except Exception as e:
            logger.error(f"Failed to load model{model_name} : {e}")
            raise

    def delete_model_version(self, model_name:str, version:str):
        try:
            self.client.delete_model_version(model_name, version)
            logger.info(f"Deleted model{model_name} version {version}")
        except Exception as e:
            logger.error(f"Failed to delete model{model_name} : {e}")
            raise

    def  model_to_production(self, model_name:str, version:str, archive_existing:bool=True):
        self.transition_model_stage(model_name = model_name,
                                    version = version,
                                    stage="production",
                                    archive_existing_versions=archive_existing)
        logger.info(f"Production model{model_name} version {version} succeeded")



