from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
import logging

from langchain_core.documents import Document
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./data/vectorstore",
        collection_name: str = "rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        api_key: Optional[str] = None,
        vector_store_type: str = "chroma",            # chroma | pgvector
        db_url: Optional[str] = None,                # required for pgvector
    ):
        self.collection_name = collection_name
        self.vector_store_type = (vector_store_type or "chroma").lower().strip()

        # embeddings
        if embedding_provider == "openai" and api_key:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},
            )

        self.vectorstore = None

        if self.vector_store_type == "pgvector":
            if not db_url:
                raise ValueError("db_url is required when VECTOR_STORE_TYPE=pgvector")
            self._init_pgvector(db_url)
        else:
            self._init_chroma(persist_directory)

    # -------------------------
    # init implementations
    # -------------------------
    def _init_chroma(self, persist_directory: str) -> None:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.vectorstore = Chroma(
            persist_directory=str(persist_path),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        logger.info("VectorStore=Chroma loaded at %s (collection=%s)", persist_path, self.collection_name)

    def _init_pgvector(self, db_url: str) -> None:
        try:
            from langchain_postgres.vectorstores import PGVector  # type: ignore
            self.vectorstore = PGVector(
                connection=db_url,
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                use_jsonb=True,
            )
            logger.info("VectorStore=PGVector (langchain-postgres) connected (collection=%s)", self.collection_name)
            return
        except Exception as e:
            logger.warning("langchain-postgres PGVector not usable (%s). Falling back to community PGVector.", e)

        # Fallback (deprecated but works with psycopg2 in many cases) :contentReference[oaicite:5]{index=5}
        from langchain_community.vectorstores import PGVector  # type: ignore

        self.vectorstore = PGVector(
            connection_string=db_url,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        logger.info("VectorStore=PGVector (community fallback) connected (collection=%s)", self.collection_name)

    def add_documents(self, documents: List[Document]):
        try:
            ids = self.vectorstore.add_documents(documents)
            logger.info("Added %d docs to vectorstore", len(documents))
            return ids
        except Exception as e:
            logger.error("Error adding documents: %s", str(e))
            raise

    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Document]:
        if self.vector_store_type == "chroma":
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return [doc for doc, score in results if score <= threshold]
        else:
            return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 2) -> List[Tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def delete_collection(self):
        self.vectorstore.delete_collection()
        logger.info("Deleted collection %s", self.collection_name)

    def get_collection_size(self) -> int:
        if self.vector_store_type == "chroma":
            return self.vectorstore._collection.count()
        return 0