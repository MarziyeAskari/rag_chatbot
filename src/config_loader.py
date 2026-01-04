"""
Configration loader module
"""
import os
from distutils.command.config import config

import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from  dotenv import load_dotenv
from typing import Dict,Any
load_dotenv(verbose=True)

class Settings(BaseSettings):
    """ Application settings """
    # App Setting
    app_name: str ="RAG Chatbot"
    app_version: str ="1.0.0"
    app_host: str ="0.0.0.0"
    app_port: str ="8080"
    debug: bool =False

    #LLM Setting
    llm_provider: str ="OpenAI"
    llm_model: str ="gpt-40-mini"
    llm_temperature: float =0.7
    llm_max_tokens:int =1000
    openai_api_key: str =""

    # Embedding Setting
    embedding_provider: str ="Sentence Transformer"
    embedding_model: str ="gpt-40-mini"

    # Vector store settings
    vector_store_type: str = "chroma"
    vector_store_path: str = "./data/vectorstore"
    collection_name: str = "rag_documents"

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval settings
    top_k: int = 2
    similarity_threshold: float = 0.7

    # Paths
    documents_path: str = "./data/documents"
    processed_path: str = "./data/processed"
    uploads_path: str = "./data/uploads"

    #session management
    use_database_session: bool =True
    session_database_url: str =""
    session_database_path: str = "./data/sessions.db"
    session_timeout: int = 24
    max_history_per_session: int = 50


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any] :
    """
    Load configuration from yaml file

    """
    config_file=Path(config_path)
    if not config_file.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config:
        if "llm" in config:
            config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY",config["llm"].get("api_key",""))
        if "app" in config:
            config["app"]["port"]= int(os.getenv("API_PORT",config["app"].get("port",8000)))
            config["app"]["host"]= os.getenv("API_HOST",config["app"].get("host","0.0.0.0"))
            config["app"]["debug"]= os.getenv("DEBUG",str(config["app"].get("debug",False))).lower()=="true"

    return config

def get_setting() -> Settings:
    settings = Settings()
    configu =load_config()

    if config:
        if "app" in config:
            settings.app_name =configu["app"].get("name",settings.app_name)
            settings.app_version =configu["app"].get("version",settings.app_version)
            settings.app_host =configu["app"].get("host",settings.app_host)
            settings.app_port =configu["app"].get("port",settings.app_port)
            settings.debug =configu["app"].get("debug",settings.debug)
        if "llm" in config:
            settings.llm_provider =configu["llm"].get("provider",settings.llm_provider)
            settings.llm_model =configu["llm"].get("model",settings.llm_model)
            settings.llm_temperature=configu["llm"].get("temperature",settings.llm_temperature)
            settings.llm_max_tokens =configu["llm"].get("max_tokens",settings.llm_max_tokens)
            settings.openai_api_key=configu["llm"].get("api_key",settings.openai_api_key)
        if "embeddings" in config:
            settings.embedding_provider =configu["embeddings"].get("provider",settings.embedding_provider)
            settings.embedding_model =configu["embeddings"].get("model",settings.embedding_model)

        if "vector_database" in config:
            settings.vector_database_type = configu["vector_database"].get("type",settings.vector_database_type)
            settings.vector_database_collection_name = configu["vector_database"].get("collection_name",settings.vector_database_collection_name)
            settings.vector_database_url = configu["vector_database"].get("base_url",settings.vector_database_url)

        if "document_processing" in config:
            settings.chunk_size = configu["document_processing"].get("chunk_size",settings.chunk_size)
            settings.chunk_overlap = configu["document_processing"].get("chunk_overlap",settings.chunk_overlap)
        if "retrieval" in config:
            settings.retrieval = configu["similarity_threshold"].get("similarity_threshold",settings.similarity_threshold)
            settings.top_k = configu["retrieval"].get("top_k",settings.top_k)
        if "session" in config:
            settings.use_database_session = configu["session"].get("use_database_session",settings.use_database_session)
            settings.session_database_url = configu["session"].get("database_url",settings.session_database_url)
            settings.session_timeout = configu["session"].get("timeout",settings.session_timeout)
            settings.session_database_path = configu["session"].get("database_path",settings.session_database_path)
            settings.max_history_per_session = configu["session"].get("max_history_per_session",settings.max_history_per_session)

    return settings
