import argparse
import datetime
from pathlib import Path

from app.main import document_processor
from mlops.mlflow_utils import MlflowTracker
from src.config_loader import get_setting
from src.documents_processor import logger
from src.vector_store import VectorStore


def train(document_path: str, clear_existing: bool = False, use_mflow: bool = True):
    setting = get_setting()
    mlflow_tracker = None
    if use_mflow:
        mlflow_tracker = MlflowTracker(experiment_name="rag_training")

    run_name = f"training_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow_tracker.start_run(run_name=run_name) as run if mlflow_tracker else nullcontext():
        logger.info("Starting RAG training pipeline...")
        logger.info(f"Document patg: {document_path}")
        logger.info(f"Vector Store path: {setting.vector_store_path}")

        if mlflow_tracker:
            mlflow_tracker.log_params({
                "chunk_size": setting.chunk_size,
                "chunk_overlap": setting.chunk_overlap,
                "embedding_model": setting.embedding_model,
                "embedding_provider": setting.embedding_provider,
                "collection_name": setting.collection_name, }
            )
        logger.info(f"Initializing vector store...")
        vector_store = VectorStore(
            persist_directory=setting.vector_store_path,
            collection_name=setting.collection_name,
            embedding_model=setting.embedding_model,
            embedding_provider=setting.embedding_provider,
            api_key=setting.api_key if setting.embedding_provider == "openai" else None
        )

        if clear_existing:
            logger.info("Clearing existing vector store...")
            vector_store.delete_collection()
            vector_store._load_or_create_vectorstore()

        logger.info("processing documents...")
        document_path = Path(document_path)

        if not document_path.exists():
            logger.error(f"Document path does not exist: {document_path}")
            return

        chunks = document_processor.process_directory(str(document_path))

        if not chunks:
            logger.warning("No documents found to process")

        logger.info(f"Processing {len(chunks)} chunks from documents")

        logger.info(f"Adding chunks to vector store...")
        ids = vector_store.add_documents(chunks)

        total_chunks = vector_store.get_collection_size()

        if mlflow_tracker:
            mlflow_tracker.log_metrics({
                "total_chunks": total_chunks,
                "num_documents": len(set(chunk.metadata.get("source_file", "") for chunk in chunks))
            })

            mlflow_tracker.log_params({
                "vector_store_path": setting.vector_store_path,
            })

            logger.info(f"Training completed successfully")
            logger.info(f"Total chunks: {total_chunks}")
            logger.info("Vector store saved to: {setting.vector_store_path}")

class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAG chatbot on documents")
    parser.add_argument("--documents", type=str, default="./data/documents", help="Path to directory containing documents")
    parser.add_argument("--clear", action="store_true", help="Clear existing vector store")
    parser.add_argument("--no-mflow", action="store_true", help="Disable Mlflow tracker")
    args = parser.parse_args()
    train(args.documents, args.clear,use_mflow=not args.no_mflow)
